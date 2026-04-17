//! BVH-accelerated picking with TriMesh caching.
//!
//! Provides `PickAccelerator` — a binary bounding volume hierarchy built from
//! scene objects' world-space AABBs. Ray queries traverse the BVH to quickly
//! reject non-intersecting subtrees, then test leaf objects with cached
//! `parry3d::TriMesh` instances.

use std::collections::HashMap;

use crate::scene::aabb::Aabb;
use crate::resources::mesh_store::MeshId;
use crate::scene::scene::Scene;
use crate::interaction::selection::NodeId;

use parry3d::math::Vector;
use parry3d::query::{Ray, RayCast};
use parry3d::shape::FeatureId;

/// An entry in the BVH representing a single scene object.
#[derive(Debug, Clone)]
struct BvhEntry {
    aabb: Aabb,
    node_id: NodeId,
    mesh_index: usize,
    world_transform: glam::Mat4,
}

/// BVH tree node.
enum BvhNode {
    Leaf {
        entry_indices: Vec<usize>,
        aabb: Aabb,
    },
    Interior {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { aabb, .. } => aabb,
            BvhNode::Interior { aabb, .. } => aabb,
        }
    }
}

/// BVH-accelerated picking structure with TriMesh cache.
pub struct PickAccelerator {
    entries: Vec<BvhEntry>,
    root: Option<BvhNode>,
    trimesh_cache: HashMap<usize, parry3d::shape::TriMesh>,
}

impl PickAccelerator {
    /// Build a BVH from the current scene state.
    ///
    /// `mesh_aabb_fn` provides the local-space AABB for each mesh.
    pub fn build_from_scene(scene: &Scene, mesh_aabb_fn: impl Fn(MeshId) -> Option<Aabb>) -> Self {
        let mut entries = Vec::new();
        for node in scene.nodes() {
            if !node.is_visible() {
                continue;
            }
            let Some(mesh_id) = node.mesh_id() else {
                continue;
            };
            if let Some(local_aabb) = mesh_aabb_fn(mesh_id) {
                let world_aabb = local_aabb.transformed(&node.world_transform());
                entries.push(BvhEntry {
                    aabb: world_aabb,
                    node_id: node.id(),
                    mesh_index: mesh_id.index(),
                    world_transform: node.world_transform(),
                });
            }
        }

        let indices: Vec<usize> = (0..entries.len()).collect();
        let root = if entries.is_empty() {
            None
        } else {
            Some(build_bvh_node(&entries, indices))
        };

        Self {
            entries,
            root,
            trimesh_cache: HashMap::new(),
        }
    }

    /// Pick the nearest object hit by the ray.
    ///
    /// `mesh_lookup` maps mesh_index to (positions, indices) for TriMesh construction.
    pub fn pick(
        &mut self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        mesh_lookup: &HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    ) -> Option<crate::interaction::picking::PickHit> {
        let root = self.root.as_ref()?;
        let mut best: Option<(NodeId, f32, crate::interaction::picking::PickHit)> = None;

        // Collect candidate entry indices via iterative BVH traversal (read-only).
        let mut candidates = Vec::new();
        let mut stack: Vec<&BvhNode> = vec![root];
        while let Some(node) = stack.pop() {
            if !ray_aabb_test(ray_origin, ray_dir, node.aabb()) {
                continue;
            }
            match node {
                BvhNode::Leaf { entry_indices, .. } => {
                    candidates.extend_from_slice(entry_indices);
                }
                BvhNode::Interior { left, right, .. } => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }

        // Test each candidate (may mutate trimesh_cache).
        for idx in candidates {
            let node_id = self.entries[idx].node_id;
            let mesh_index = self.entries[idx].mesh_index;
            let world_transform = self.entries[idx].world_transform;

            if let Some((toi, mut hit)) = self.test_entry_by_parts(
                mesh_index,
                &world_transform,
                ray_origin,
                ray_dir,
                mesh_lookup,
            ) {
                if best.is_none() || toi < best.as_ref().unwrap().1 {
                    hit.id = node_id;
                    best = Some((node_id, toi, hit));
                }
            }
        }

        best.map(|(_, _, hit)| hit)
    }

    fn test_entry_by_parts(
        &mut self,
        mesh_index: usize,
        world_transform: &glam::Mat4,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        mesh_lookup: &HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    ) -> Option<(f32, crate::interaction::picking::PickHit)> {
        let (positions, indices) = mesh_lookup.get(&(mesh_index as u64))?;

        // Lazily build and cache TriMesh.
        if let std::collections::hash_map::Entry::Vacant(e) = self.trimesh_cache.entry(mesh_index) {
            let verts: Vec<Vector> = positions
                .iter()
                .map(|p| Vector::new(p[0], p[1], p[2]))
                .collect();
            let tri_indices: Vec<[u32; 3]> = indices
                .chunks(3)
                .filter(|c| c.len() == 3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            if tri_indices.is_empty() {
                return None;
            }
            match parry3d::shape::TriMesh::new(verts, tri_indices) {
                Ok(tm) => {
                    e.insert(tm);
                }
                Err(_) => return None,
            }
        }

        let trimesh = self.trimesh_cache.get(&mesh_index)?;

        // Extract scale, rotation, translation from world transform.
        let (scale, rotation, translation) = world_transform.to_scale_rotation_translation();

        // Transform ray into object's local (scaled) space.
        let inv_rot = rotation.inverse();
        let local_origin = inv_rot * (ray_origin - translation);
        let local_dir = inv_rot * ray_dir;
        let inv_scale = glam::Vec3::new(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
        let scaled_origin = local_origin * inv_scale;
        let scaled_dir = (local_dir * inv_scale).normalize();

        let ray = Ray::new(
            Vector::new(scaled_origin.x, scaled_origin.y, scaled_origin.z),
            Vector::new(scaled_dir.x, scaled_dir.y, scaled_dir.z),
        );

        trimesh
            .cast_local_ray_and_get_normal(&ray, f32::MAX, true)
            .map(|intersection| {
                // Scale TOI back to world space.
                let avg_scale = (scale.x + scale.y + scale.z) / 3.0;
                let toi = intersection.time_of_impact * avg_scale;

                let triangle_index = match intersection.feature {
                    FeatureId::Face(idx) => idx,
                    _ => u32::MAX,
                };

                // Transform hit point to world space.
                // scaled_origin and scaled_dir are in inv-scaled local space, so:
                // local_hit = scaled_origin + scaled_dir * intersection.time_of_impact
                // undo inv_scale: multiply by scale to get unscaled local coords
                // then apply rotation and translation.
                let local_hit_scaled = scaled_origin + scaled_dir * intersection.time_of_impact;
                let local_hit = local_hit_scaled * scale;
                let world_pos = rotation * local_hit + translation;

                // Transform normal to world space.
                // The normal from cast_local_ray_and_get_normal is in scaled-local space.
                // Use inverse-transpose (scale the normal by inv_scale) then normalize.
                let world_normal = (rotation * (intersection.normal * inv_scale)).normalize();

                (
                    toi,
                    crate::interaction::picking::PickHit {
                        id: 0, // placeholder — caller fills in actual node_id
                        triangle_index,
                        world_pos,
                        normal: world_normal,
                        point_index: None,
                        scalar_value: None,
                    },
                )
            })
    }

    /// Invalidate the TriMesh cache for a specific mesh (e.g. after re-tessellation).
    pub fn invalidate_mesh(&mut self, mesh_index: usize) {
        self.trimesh_cache.remove(&mesh_index);
    }

    /// Clear all cached data. A full rebuild is needed.
    pub fn invalidate_all(&mut self) {
        self.trimesh_cache.clear();
        self.entries.clear();
        self.root = None;
    }

    /// Number of cached TriMesh instances.
    pub fn trimesh_cache_len(&self) -> usize {
        self.trimesh_cache.len()
    }
}

// ---------------------------------------------------------------------------
// BVH construction (SAH-based binary split)
// ---------------------------------------------------------------------------

fn build_bvh_node(entries: &[BvhEntry], indices: Vec<usize>) -> BvhNode {
    // Compute combined AABB.
    let combined = combined_aabb(entries, &indices);

    // Leaf threshold: 4 or fewer entries.
    if indices.len() <= 4 {
        return BvhNode::Leaf {
            entry_indices: indices,
            aabb: combined,
        };
    }

    // Find best split axis and position using SAH.
    let (best_axis, best_split) = find_best_split(entries, &indices, &combined);

    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();
    for &idx in &indices {
        let center = entries[idx].aabb.center();
        let val = match best_axis {
            0 => center.x,
            1 => center.y,
            _ => center.z,
        };
        if val <= best_split {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    // Fallback: if all entries ended up on one side, split in half.
    if left_indices.is_empty() || right_indices.is_empty() {
        let mid = indices.len() / 2;
        left_indices = indices[..mid].to_vec();
        right_indices = indices[mid..].to_vec();
    }

    BvhNode::Interior {
        aabb: combined,
        left: Box::new(build_bvh_node(entries, left_indices)),
        right: Box::new(build_bvh_node(entries, right_indices)),
    }
}

fn combined_aabb(entries: &[BvhEntry], indices: &[usize]) -> Aabb {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for &idx in indices {
        min = min.min(entries[idx].aabb.min);
        max = max.max(entries[idx].aabb.max);
    }
    Aabb { min, max }
}

fn find_best_split(_entries: &[BvhEntry], _indices: &[usize], parent_aabb: &Aabb) -> (usize, f32) {
    let extents = parent_aabb.max - parent_aabb.min;
    // Choose the longest axis.
    let axis = if extents.x >= extents.y && extents.x >= extents.z {
        0
    } else if extents.y >= extents.z {
        1
    } else {
        2
    };

    // Use midpoint of the longest axis as the split.
    let split = match axis {
        0 => (parent_aabb.min.x + parent_aabb.max.x) * 0.5,
        1 => (parent_aabb.min.y + parent_aabb.max.y) * 0.5,
        _ => (parent_aabb.min.z + parent_aabb.max.z) * 0.5,
    };

    (axis, split)
}

// ---------------------------------------------------------------------------
// Ray-AABB intersection test
// ---------------------------------------------------------------------------

fn ray_aabb_test(origin: glam::Vec3, dir: glam::Vec3, aabb: &Aabb) -> bool {
    let inv_dir = glam::Vec3::new(
        if dir.x.abs() > 1e-10 {
            1.0 / dir.x
        } else {
            f32::INFINITY * dir.x.signum()
        },
        if dir.y.abs() > 1e-10 {
            1.0 / dir.y
        } else {
            f32::INFINITY * dir.y.signum()
        },
        if dir.z.abs() > 1e-10 {
            1.0 / dir.z
        } else {
            f32::INFINITY * dir.z.signum()
        },
    );

    let t1 = (aabb.min - origin) * inv_dir;
    let t2 = (aabb.max - origin) * inv_dir;

    let tmin = t1.min(t2);
    let tmax = t1.max(t2);

    let tenter = tmin.x.max(tmin.y).max(tmin.z);
    let texit = tmax.x.min(tmax.y).min(tmax.z);

    texit >= tenter.max(0.0)
}

// ---------------------------------------------------------------------------
// Public API wrapper
// ---------------------------------------------------------------------------

/// Pick the nearest scene node using a BVH accelerator.
///
/// Thin wrapper around `PickAccelerator::pick`.
pub fn pick_scene_accelerated(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    accelerator: &mut PickAccelerator,
    mesh_lookup: &HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
) -> Option<crate::interaction::picking::PickHit> {
    accelerator.pick(ray_origin, ray_dir, mesh_lookup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::Material;
    use crate::resources::mesh_store::MeshId;

    fn unit_cube_mesh() -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = vec![
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ];
        let indices = vec![
            0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7,
            3, 0, 4, 5, 5, 1, 0,
        ];
        (positions, indices)
    }

    fn unit_aabb() -> Aabb {
        Aabb {
            min: glam::Vec3::splat(-0.5),
            max: glam::Vec3::splat(0.5),
        }
    }

    #[test]
    fn test_bvh_build_single() {
        let mut scene = Scene::new();
        scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.update_transforms();

        let accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));
        assert_eq!(accel.entries.len(), 1);
        assert!(accel.root.is_some());
    }

    #[test]
    fn test_bvh_pick_hit() {
        let mut scene = Scene::new();
        scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.update_transforms();

        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));

        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        let result = accel.pick(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &mesh_lookup,
        );
        assert!(result.is_some(), "should hit the cube");
    }

    #[test]
    fn test_bvh_pick_miss() {
        let mut scene = Scene::new();
        scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.update_transforms();

        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));

        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        let result = accel.pick(
            glam::Vec3::new(100.0, 100.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &mesh_lookup,
        );
        assert!(result.is_none(), "should miss");
    }

    #[test]
    fn test_bvh_pick_nearest() {
        let mut scene = Scene::new();
        scene.add(
            Some(MeshId(0)),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 2.0)),
            Material::default(),
        );
        scene.add(
            Some(MeshId(1)),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -2.0)),
            Material::default(),
        );
        scene.update_transforms();

        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));

        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions.clone(), indices.clone()));
        mesh_lookup.insert(1u64, (positions, indices));

        // Ray from z=10 toward -Z: should hit the nearer object at z=2.
        let result = accel.pick(
            glam::Vec3::new(0.0, 0.0, 10.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &mesh_lookup,
        );
        assert!(result.is_some(), "should hit something");
    }

    #[test]
    fn test_trimesh_cache_reuse() {
        let mut scene = Scene::new();
        scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.update_transforms();

        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));

        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        // First pick — builds TriMesh.
        let _ = accel.pick(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &mesh_lookup,
        );
        assert_eq!(accel.trimesh_cache_len(), 1);

        // Second pick — should reuse cached TriMesh (cache len stays 1).
        let _ = accel.pick(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &mesh_lookup,
        );
        assert_eq!(accel.trimesh_cache_len(), 1);
    }

    #[test]
    fn test_ray_aabb_hit() {
        let aabb = unit_aabb();
        assert!(ray_aabb_test(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &aabb,
        ));
    }

    #[test]
    fn test_ray_aabb_miss() {
        let aabb = unit_aabb();
        assert!(!ray_aabb_test(
            glam::Vec3::new(100.0, 100.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &aabb,
        ));
    }
}
