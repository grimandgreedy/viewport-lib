//! BVH-accelerated picking with TriMesh caching.
//!
//! Provides `PickAccelerator` : a binary bounding volume hierarchy built from
//! scene objects' world-space AABBs. Ray queries traverse the BVH to quickly
//! reject non-intersecting subtrees, then test leaf objects with cached
//! `parry3d::TriMesh` instances.
//!
//! # Skinned meshes
//!
//! CPU picking against skinned meshes is **bind-pose** by default. The GPU
//! skinning path keeps the mesh's vertex buffer untouched and applies LBS in
//! the vertex shader, so the CPU never sees the deformed positions; the BVH
//! and the cached `TriMesh` both reflect the bind pose. A click registers on
//! the bind-pose silhouette, not the rendered (deformed) silhouette.
//!
//! Two knobs handle this:
//!
//! 1. **Padded AABBs.** Use [`build_from_scene_skin_aware`](PickAccelerator::build_from_scene_skin_aware)
//!    (or pass an already-expanded AABB through the closure to
//!    [`build_from_scene`](PickAccelerator::build_from_scene)) so the BVH leaf
//!    covers the deformation envelope, not just the bind pose. Without the
//!    padding the BVH can reject rays that would actually hit the deformed
//!    mesh.
//! 2. **Optional refresh-on-pose-change.** For accurate clicks on the
//!    deformed silhouette (paying CPU cost every frame), call
//!    [`invalidate_skinned_meshes`](PickAccelerator::invalidate_skinned_meshes)
//!    after applying a [`crate::SkinnedMeshUpdate`] and pass the deformed
//!    positions in the `mesh_lookup` argument of [`pick`](PickAccelerator::pick).
//!    The next pick rebuilds the cached `TriMesh` against the current pose.
//!    On the GPU skinning path the CPU does not receive deformed positions,
//!    so this refresh path is only available when the plugin runs on
//!    [`crate::SkinningPath::Cpu`].
//!
//! GPU picking (`renderer::picking`) reads the rasterised object-ID
//! framebuffer and therefore needs no skinning awareness here: skinned meshes
//! pick the same way as static meshes.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::interaction::select::selection::NodeId;
use crate::resources::mesh::mesh_store::MeshId;
use crate::scene::aabb::Aabb;
use crate::scene::scene::Scene;

use parry3d::math::Vector;
use parry3d::query::{Ray, RayCast};
use spatial_query::{
    Aabb as SqAabb, Bvh, LeafHit, Point, QueryFilter, QueryGeometry, Ray as SqRay,
};

use crate::renderer::SubObjectRef;

/// An entry in the BVH representing a single scene object.
#[derive(Debug, Clone)]
struct BvhEntry {
    aabb: Aabb,
    node_id: NodeId,
    mesh_index: usize,
    world_transform: glam::Mat4,
}

/// BVH-accelerated picking structure with TriMesh cache.
pub struct PickAccelerator {
    entries: Vec<BvhEntry>,
    bvh: Option<Bvh<3>>,
    /// Lazily built parry `TriMesh` per mesh index. `RefCell` because the
    /// narrow-phase [`cast_leaf`] runs behind the `&self` [`QueryGeometry::test_ray`]
    /// during a query, and populates the cache on first touch.
    trimesh_cache: RefCell<HashMap<usize, parry3d::shape::TriMesh>>,
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

        let trimesh_cache = RefCell::new(HashMap::new());
        let bvh = if entries.is_empty() {
            None
        } else {
            // Build only reads leaf AABBs, so the mesh lookup is unused here.
            let empty = HashMap::new();
            let geom = PickGeom {
                entries: &entries,
                mesh_lookup: &empty,
                cache: &trimesh_cache,
            };
            Some(Bvh::build(&geom))
        };

        Self {
            entries,
            bvh,
            trimesh_cache,
        }
    }

    /// Like [`build_from_scene`](Self::build_from_scene), but pads the local
    /// AABB of meshes flagged by `is_skinned` before transforming to world
    /// space.
    ///
    /// `padding_factor` is a fraction of the bind-pose AABB's longest side
    /// added on every axis (see [`Aabb::expanded_relative`]). Pick the
    /// smallest value that still covers the worst-case pose for your content:
    /// `0.25` works for a typical character rig; rigs with extreme stretch or
    /// large limb sweeps need more.
    ///
    /// The resulting BVH is conservative: it may queue extra leaves for
    /// triangle testing, but it will not reject rays that hit the deformed
    /// mesh. Triangle tests still run against the bind-pose `TriMesh`, so
    /// picks land on the bind-pose silhouette unless you also call
    /// [`invalidate_skinned_meshes`](Self::invalidate_skinned_meshes) each
    /// frame and pass deformed positions in `mesh_lookup`.
    pub fn build_from_scene_skin_aware(
        scene: &Scene,
        mesh_aabb_fn: impl Fn(MeshId) -> Option<Aabb>,
        is_skinned: impl Fn(MeshId) -> bool,
        padding_factor: f32,
    ) -> Self {
        Self::build_from_scene(scene, |mesh_id| {
            let aabb = mesh_aabb_fn(mesh_id)?;
            if is_skinned(mesh_id) {
                Some(aabb.expanded_relative(padding_factor))
            } else {
                Some(aabb)
            }
        })
    }

    /// Update leaf bounds from the current scene, keeping the tree topology.
    ///
    /// Much cheaper than [`build_from_scene`](Self::build_from_scene) for a scene
    /// whose objects moved but did not change in number: for a moving scene, build
    /// once and refit each frame. Re-reads the world AABB of every visible,
    /// mesh-bearing node (in the same order the build walked them) and propagates
    /// the new bounds up the existing tree.
    ///
    /// Returns `false` without touching the tree when a refit is not valid: the set
    /// of visible, mesh-bearing nodes changed size (objects added, removed, or shown
    /// or hidden), or the accelerator was never built. The caller should rebuild in
    /// that case. Refit assumes the node set is otherwise stable in identity and
    /// order; toggling which objects are visible while keeping the count the same is
    /// the caller's responsibility to avoid.
    ///
    /// The tree quality drifts as objects move away from where they were built;
    /// rebuild periodically (on a timer, or when picks start feeling slow) to
    /// restore it.
    pub fn refit_from_scene(
        &mut self,
        scene: &Scene,
        mesh_aabb_fn: impl Fn(MeshId) -> Option<Aabb>,
    ) -> bool {
        let mut new_entries = Vec::with_capacity(self.entries.len());
        for node in scene.nodes() {
            if !node.is_visible() {
                continue;
            }
            let Some(mesh_id) = node.mesh_id() else {
                continue;
            };
            if let Some(local_aabb) = mesh_aabb_fn(mesh_id) {
                new_entries.push(BvhEntry {
                    aabb: local_aabb.transformed(&node.world_transform()),
                    node_id: node.id(),
                    mesh_index: mesh_id.index(),
                    world_transform: node.world_transform(),
                });
            }
        }

        if self.bvh.is_none() || new_entries.len() != self.entries.len() {
            return false;
        }
        self.entries = new_entries;

        let empty = HashMap::new();
        let geom = PickGeom {
            entries: &self.entries,
            mesh_lookup: &empty,
            cache: &self.trimesh_cache,
        };
        self.bvh.as_mut().unwrap().refit(&geom);
        true
    }

    /// Pick the nearest object hit by the ray.
    ///
    /// `mesh_lookup` maps mesh_index to (positions, indices) for TriMesh construction.
    pub fn pick(
        &mut self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        mesh_lookup: &HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    ) -> Option<crate::renderer::PickHit> {
        let bvh = self.bvh.as_ref()?;

        // Broad phase: the spatial-query BVH walks to the nearest leaf whose
        // triangle the ray hits. Its `test_ray` runs the same parry narrow-phase as
        // the final hit below, so the winning leaf is the object a brute-force scan
        // would pick.
        let geom = PickGeom {
            entries: &self.entries,
            mesh_lookup,
            cache: &self.trimesh_cache,
        };
        let ray = SqRay::new(
            Point([ray_origin.x, ray_origin.y, ray_origin.z]),
            Point([ray_dir.x, ray_dir.y, ray_dir.z]),
        );
        let hit = bvh.raycast_nearest(&geom, &ray, f32::MAX, &QueryFilter::default())?;

        // Recover the full hit for the winning leaf: exact world position, normal,
        // and sub-object, with the same math the narrow phase used.
        let entry = &self.entries[hit.leaf];
        let cast = cast_leaf(entry, ray_origin, ray_dir, mesh_lookup, &self.trimesh_cache)?;
        Some(crate::renderer::PickHit {
            id: entry.node_id,
            sub_object: cast.sub_object,
            world_pos: cast.world_pos,
            normal: cast.normal,
            scalar_value: None,
            sub_object_world_pos: None,
        })
    }

    /// Invalidate the TriMesh cache for a specific mesh (e.g. after re-tessellation).
    pub fn invalidate_mesh(&mut self, mesh_index: usize) {
        self.trimesh_cache.borrow_mut().remove(&mesh_index);
    }

    /// Drop cached `TriMesh` instances for every mesh in `skinned_mesh_ids`.
    ///
    /// Call this after applying [`crate::SkinnedMeshUpdate`]s when you want
    /// CPU picking to test the deformed silhouette rather than the bind pose.
    /// The next `pick` call will rebuild the `TriMesh` from whatever positions
    /// are passed in `mesh_lookup` (typically the just-updated, deformed
    /// positions).
    ///
    /// This is the "refresh-on-pose-change" knob mentioned in the
    /// module-level docs. Each rebuild costs roughly `O(V + T)` per skinned
    /// mesh in `parry3d::TriMesh::new`; budget accordingly when used every
    /// frame on heavy rigs.
    ///
    /// `skinned_mesh_ids` accepts any iterator of `MeshId`s; pass the ids of
    /// the meshes whose pose changed this frame. Unknown ids are ignored.
    pub fn invalidate_skinned_meshes(
        &mut self,
        skinned_mesh_ids: impl IntoIterator<Item = MeshId>,
    ) {
        let mut cache = self.trimesh_cache.borrow_mut();
        for id in skinned_mesh_ids {
            cache.remove(&id.index());
        }
    }

    /// Clear all cached data. A full rebuild is needed.
    pub fn invalidate_all(&mut self) {
        self.trimesh_cache.borrow_mut().clear();
        self.entries.clear();
        self.bvh = None;
    }

    /// Number of cached TriMesh instances.
    pub fn trimesh_cache_len(&self) -> usize {
        self.trimesh_cache.borrow().len()
    }
}

// ---------------------------------------------------------------------------
// spatial-query provider + narrow phase
// ---------------------------------------------------------------------------

/// A [`QueryGeometry`] view over the accelerator's leaves: one leaf per visible,
/// mesh-bearing scene node. `test_ray` runs the parry narrow-phase behind `&self`
/// via the `RefCell` TriMesh cache.
struct PickGeom<'a> {
    entries: &'a [BvhEntry],
    mesh_lookup: &'a HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    cache: &'a RefCell<HashMap<usize, parry3d::shape::TriMesh>>,
}

impl QueryGeometry<3> for PickGeom<'_> {
    type Id = NodeId;
    // The nearest hit is chosen by time-of-impact; the sub-object is recovered for
    // the winning leaf in `PickAccelerator::pick`, so the broad phase carries none.
    type SubObject = ();

    fn leaf_count(&self) -> usize {
        self.entries.len()
    }

    fn id(&self, leaf: usize) -> NodeId {
        self.entries[leaf].node_id
    }

    fn world_aabb(&self, leaf: usize) -> SqAabb<3> {
        let a = &self.entries[leaf].aabb;
        SqAabb::new(
            Point([a.min.x, a.min.y, a.min.z]),
            Point([a.max.x, a.max.y, a.max.z]),
        )
    }

    fn test_ray(&self, leaf: usize, ray: &SqRay<3>, max_toi: f32) -> Option<LeafHit<3>> {
        let origin = glam::Vec3::new(ray.origin[0], ray.origin[1], ray.origin[2]);
        let dir = glam::Vec3::new(ray.dir[0], ray.dir[1], ray.dir[2]);
        let cast = cast_leaf(
            &self.entries[leaf],
            origin,
            dir,
            self.mesh_lookup,
            self.cache,
        )?;
        if cast.toi < 0.0 || cast.toi > max_toi {
            return None;
        }
        Some(LeafHit::new(
            cast.toi,
            Point([cast.normal.x, cast.normal.y, cast.normal.z]),
        ))
    }
}

/// The narrow-phase result for one leaf, in world space.
struct LeafCast {
    toi: f32,
    world_pos: glam::Vec3,
    normal: glam::Vec3,
    sub_object: Option<SubObjectRef>,
}

/// Cast the ray against one leaf's cached parry `TriMesh`: transform into the mesh's
/// local (scaled) space, cast, and map the hit back to world. Builds and caches the
/// `TriMesh` on first touch.
fn cast_leaf(
    entry: &BvhEntry,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    mesh_lookup: &HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    cache: &RefCell<HashMap<usize, parry3d::shape::TriMesh>>,
) -> Option<LeafCast> {
    let mesh_index = entry.mesh_index;
    let (positions, indices) = mesh_lookup.get(&(mesh_index as u64))?;

    let mut cache = cache.borrow_mut();
    if let std::collections::hash_map::Entry::Vacant(e) = cache.entry(mesh_index) {
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
    let trimesh = cache.get(&mesh_index)?;

    // Extract scale, rotation, translation from the world transform.
    let (scale, rotation, translation) = entry.world_transform.to_scale_rotation_translation();

    // Transform the ray into the object's local (scaled) space.
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

            let sub_object = SubObjectRef::from_feature_id(intersection.feature);

            // Map the local hit back to world: undo inv_scale, then rotate and
            // translate.
            let local_hit_scaled = scaled_origin + scaled_dir * intersection.time_of_impact;
            let local_hit = local_hit_scaled * scale;
            let world_pos = rotation * local_hit + translation;

            // Normal is in scaled-local space; inverse-transpose (scale by inv_scale)
            // then rotate to world.
            let normal = (rotation * (intersection.normal * inv_scale)).normalize();

            LeafCast {
                toi,
                world_pos,
                normal,
                sub_object,
            }
        })
}

// ---------------------------------------------------------------------------
// Public API wrapper
// ---------------------------------------------------------------------------

/// Pick the nearest scene node using a BVH accelerator.
///
/// Thin wrapper around `PickAccelerator::pick`.
pub fn pick_scene_accelerated_cpu(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    accelerator: &mut PickAccelerator,
    mesh_lookup: &HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
) -> Option<crate::renderer::PickHit> {
    accelerator.pick(ray_origin, ray_dir, mesh_lookup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::mesh::mesh_store::MeshId;
    use crate::scene::material::Material;

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
        scene.add(
            Some(MeshId::from_index(0)),
            glam::Mat4::IDENTITY,
            Material::default(),
        );
        scene.update_transforms();

        let accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));
        assert_eq!(accel.entries.len(), 1);
        assert!(accel.bvh.is_some());
    }

    #[test]
    fn test_bvh_pick_hit() {
        let mut scene = Scene::new();
        scene.add(
            Some(MeshId::from_index(0)),
            glam::Mat4::IDENTITY,
            Material::default(),
        );
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
        scene.add(
            Some(MeshId::from_index(0)),
            glam::Mat4::IDENTITY,
            Material::default(),
        );
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
            Some(MeshId::from_index(0)),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 2.0)),
            Material::default(),
        );
        scene.add(
            Some(MeshId::from_index(1)),
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
        scene.add(
            Some(MeshId::from_index(0)),
            glam::Mat4::IDENTITY,
            Material::default(),
        );
        scene.update_transforms();

        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));

        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        // First pick : builds TriMesh.
        let _ = accel.pick(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &mesh_lookup,
        );
        assert_eq!(accel.trimesh_cache_len(), 1);

        // Second pick : should reuse cached TriMesh (cache len stays 1).
        let _ = accel.pick(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &mesh_lookup,
        );
        assert_eq!(accel.trimesh_cache_len(), 1);
    }

    #[test]
    fn test_build_from_scene_skin_aware_pads_only_skinned() {
        let mut scene = Scene::new();
        scene.add(
            Some(MeshId::from_index(0)),
            glam::Mat4::IDENTITY,
            Material::default(),
        );
        scene.add(
            Some(MeshId::from_index(1)),
            glam::Mat4::IDENTITY,
            Material::default(),
        );
        scene.update_transforms();

        // Mesh 1 is "skinned", mesh 0 is not. Padding factor 1.0 doubles the
        // skinned mesh's half-extents.
        let accel = PickAccelerator::build_from_scene_skin_aware(
            &scene,
            |_| Some(unit_aabb()),
            |mid| mid == MeshId::from_index(1),
            1.0,
        );

        assert_eq!(accel.entries.len(), 2);
        let static_entry = accel
            .entries
            .iter()
            .find(|e| e.mesh_index == 0)
            .expect("static mesh entry");
        let skinned_entry = accel
            .entries
            .iter()
            .find(|e| e.mesh_index == 1)
            .expect("skinned mesh entry");
        // Static mesh keeps its [-0.5, 0.5]^3 box.
        assert!((static_entry.aabb.min - glam::Vec3::splat(-0.5)).length() < 1e-5);
        assert!((static_entry.aabb.max - glam::Vec3::splat(0.5)).length() < 1e-5);
        // Skinned mesh grew by longest_side(1.0) * factor(1.0) on each axis.
        assert!((skinned_entry.aabb.min - glam::Vec3::splat(-1.5)).length() < 1e-5);
        assert!((skinned_entry.aabb.max - glam::Vec3::splat(1.5)).length() < 1e-5);
    }

    #[test]
    fn test_invalidate_skinned_meshes_clears_cached_trimesh() {
        let mut scene = Scene::new();
        scene.add(
            Some(MeshId::from_index(0)),
            glam::Mat4::IDENTITY,
            Material::default(),
        );
        scene.update_transforms();

        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        // Populate the trimesh cache via a successful pick.
        let _ = accel.pick(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &mesh_lookup,
        );
        assert_eq!(accel.trimesh_cache_len(), 1);

        // Invalidating the skinned mesh should drop the cached entry.
        accel.invalidate_skinned_meshes([MeshId::from_index(0)]);
        assert_eq!(accel.trimesh_cache_len(), 0);
    }

    /// Two picks over the same scene must agree: the BVH accelerator and a brute
    /// force ray cast against every visible node. The accelerator only changes
    /// *which* leaves get the narrow-phase test; the nearest hit is invariant, so
    /// the two must return the same object, sub-object, and hit point on every ray.
    ///
    /// Transforms are translation + rotation with unit scale, so the brute path
    /// (scale baked into vertices) and the accelerator (transform decomposed) test
    /// identical geometry. This is a broad-phase parity check, so it stays clear of
    /// the separate non-uniform-scale modelling difference between the two paths.
    #[test]
    fn pick_parity_bvh_matches_brute_force() {
        use crate::interaction::query::picking::pick_scene_nodes_cpu;

        let mut scene = Scene::new();
        let mut ids = Vec::new();
        // A dense 7x7 grid of unit cubes in the z=0 plane.
        for gx in -3..=3 {
            for gy in -3..=3 {
                let t = glam::Vec3::new(gx as f32 * 1.5, gy as f32 * 1.5, 0.0);
                ids.push(scene.add(
                    Some(MeshId::from_index(0)),
                    glam::Mat4::from_translation(t),
                    Material::default(),
                ));
            }
        }
        // A few rotated, elevated cubes (still unit scale).
        for k in 0..6 {
            let ang = k as f32 * 0.7;
            let t = glam::Vec3::new((k as f32 - 3.0) * 1.5, 0.5, 3.0 + k as f32 * 0.6);
            let m = glam::Mat4::from_rotation_translation(
                glam::Quat::from_rotation_z(ang) * glam::Quat::from_rotation_x(ang * 0.5),
                t,
            );
            ids.push(scene.add(Some(MeshId::from_index(0)), m, Material::default()));
        }
        // Hide two nodes; both pickers must skip them.
        scene.set_visible(ids[3], false);
        scene.set_visible(ids[20], false);
        scene.update_transforms();

        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));

        let rays = parity_ray_battery();
        let mut hits = 0usize;
        let mut mismatches = 0usize;
        for (o, d) in &rays {
            let brute = pick_scene_nodes_cpu(*o, *d, &scene, &mesh_lookup);
            let accel_hit = accel.pick(*o, *d, &mesh_lookup);
            if !pick_hits_agree(&brute, &accel_hit) {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "  mismatch: brute={:?} accel={:?}",
                        brute.as_ref().map(|h| (h.id, h.sub_object)),
                        accel_hit.as_ref().map(|h| (h.id, h.sub_object)),
                    );
                }
            }
            if brute.is_some() {
                hits += 1;
            }
        }

        eprintln!(
            "pick parity: {} rays, {} hits, {} mismatches",
            rays.len(),
            hits,
            mismatches,
        );
        assert_eq!(
            mismatches, 0,
            "accelerated pick disagreed with brute force on {mismatches} rays",
        );
        // Guard against a degenerate battery that barely touches the scene.
        assert!(
            hits > rays.len() / 10,
            "ray battery barely hit anything ({hits} hits) - not exercising the picker",
        );
    }

    /// Whether two pick results are the same object, sub-object, and hit point.
    fn pick_hits_agree(
        a: &Option<crate::renderer::PickHit>,
        b: &Option<crate::renderer::PickHit>,
    ) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                x.id == y.id
                    && x.sub_object == y.sub_object
                    && (x.world_pos - y.world_pos).length() < 1e-3
                    && (x.normal - y.normal).length() < 1e-3
            }
            _ => false,
        }
    }

    /// A deterministic battery of rays aimed through the scene volume, mixing hits
    /// and misses. A small LCG keeps it reproducible without a dependency.
    fn parity_ray_battery() -> Vec<(glam::Vec3, glam::Vec3)> {
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f32 / (1u64 << 31) as f32
        };
        (0..500)
            .map(|_| {
                let origin = glam::Vec3::new(
                    (rnd() - 0.5) * 18.0,
                    (rnd() - 0.5) * 18.0,
                    8.0 + rnd() * 6.0,
                );
                let target = glam::Vec3::new(
                    (rnd() - 0.5) * 9.0,
                    (rnd() - 0.5) * 9.0,
                    (rnd() - 0.5) * 5.0,
                );
                (origin, (target - origin).normalize())
            })
            .collect()
    }

    /// After shoving every object to a new position, a refit (no rebuild) must give
    /// the same picks as a brute-force scan of the moved scene. Refit re-reads all
    /// leaf AABBs, so correctness holds even though the tree topology is frozen at
    /// build time.
    #[test]
    fn refit_matches_brute_after_moving() {
        use crate::interaction::query::picking::pick_scene_nodes_cpu;

        let mut scene = Scene::new();
        let mut ids = Vec::new();
        for gx in -3..=3 {
            for gy in -3..=3 {
                let t = glam::Vec3::new(gx as f32 * 1.5, gy as f32 * 1.5, 0.0);
                ids.push(scene.add(
                    Some(MeshId::from_index(0)),
                    glam::Mat4::from_translation(t),
                    Material::default(),
                ));
            }
        }
        scene.update_transforms();

        let mut accel = PickAccelerator::build_from_scene(&scene, |_| Some(unit_aabb()));

        // Move every object to a new spot, then refit rather than rebuild.
        for (k, &id) in ids.iter().enumerate() {
            let t = glam::Vec3::new(
                (k % 7) as f32 * 1.3 - 4.0,
                (k / 7) as f32 * 1.3 - 4.0,
                (k % 3) as f32 * 1.7,
            );
            scene.set_local_transform(id, glam::Mat4::from_translation(t));
        }
        scene.update_transforms();
        assert!(accel.refit_from_scene(&scene, |_| Some(unit_aabb())));

        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        let rays = parity_ray_battery();
        let mut mismatches = 0usize;
        for (o, d) in &rays {
            let brute = pick_scene_nodes_cpu(*o, *d, &scene, &mesh_lookup);
            let refit = accel.pick(*o, *d, &mesh_lookup);
            if !pick_hits_agree(&brute, &refit) {
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "refit pick disagreed with brute force on {mismatches} rays",
        );
    }
}
