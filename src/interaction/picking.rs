/// Ray-cast picking for the 3D viewport.
///
/// Uses parry3d 0.26's glam-native API (no nalgebra required).
/// All conversions are contained here at the picking boundary.
use crate::interaction::sub_object::SubObjectRef;
use crate::resources::{AttributeData, AttributeKind, AttributeRef};
use crate::scene::traits::ViewportObject;
use parry3d::math::{Pose, Vector};
use parry3d::query::{Ray, RayCast};

// ---------------------------------------------------------------------------
// PickHit — rich hit result
// ---------------------------------------------------------------------------

/// Result of a successful ray-cast pick against a scene object.
///
/// Contains the picked object's ID plus geometric metadata about the hit point.
/// Use this for snapping, measurement, surface painting, and other hit-dependent features.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct PickHit {
    /// The object/node ID of the hit.
    pub id: u64,
    /// Typed sub-object reference — the authoritative source for sub-object identity.
    ///
    /// `Some(SubObjectRef::Face(i))` for mesh picks; `Some(SubObjectRef::Point(i))` for
    /// point cloud picks; `None` when no specific sub-object could be identified.
    pub sub_object: Option<SubObjectRef>,
    /// World-space position of the hit point (`ray_origin + ray_dir * toi`).
    pub world_pos: glam::Vec3,
    /// Surface normal at the hit point, in world space.
    pub normal: glam::Vec3,
    /// Which triangle was hit (from parry3d `FeatureId::Face`).
    /// `u32::MAX` if the feature was not a face (edge/vertex hit — rare for TriMesh).
    ///
    /// **Deprecated** — use [`sub_object`](Self::sub_object) instead.
    #[deprecated(since = "0.5.0", note = "use `sub_object` instead")]
    pub triangle_index: u32,
    /// Index of the hit point within a [`crate::renderer::PointCloudItem`].
    /// `None` for mesh picks; set when a point cloud item is hit.
    ///
    /// **Deprecated** — use [`sub_object`](Self::sub_object) instead.
    #[deprecated(since = "0.5.0", note = "use `sub_object` instead")]
    pub point_index: Option<u32>,
    /// Interpolated scalar attribute value at the hit point.
    ///
    /// Populated by the `_with_probe` picking variants when an active attribute
    /// is provided. For vertex attributes, the value is barycentric-interpolated
    /// from the three triangle corner values. For cell attributes, the value is
    /// read directly from the hit triangle index.
    pub scalar_value: Option<f32>,
}

// ---------------------------------------------------------------------------
// GpuPickHit — GPU object-ID pick result
// ---------------------------------------------------------------------------

/// Result of a GPU object-ID pick pass.
///
/// Lighter than [`PickHit`] — carries only the object identifier and the
/// clip-space depth value at the picked pixel. World position can be
/// reconstructed from `depth` + the inverse view-projection matrix if needed.
///
/// Obtained from [`crate::renderer::ViewportRenderer::pick_scene_gpu`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct GpuPickHit {
    /// The `pick_id` of the surface that was hit.
    ///
    /// Matches the `SceneRenderItem::pick_id` set by the application.
    /// Map to a domain object using whatever id-to-object registry the app
    /// maintains. A value of `0` is never returned (non-pickable surfaces are
    /// excluded from the pick pass).
    pub object_id: u64,
    /// Clip-space depth value in `[0, 1]` at the picked pixel.
    /// `0.0` = near plane, `1.0` = far plane.
    ///
    /// Reconstruct world position:
    /// ```ignore
    /// let ndc = Vec3::new(ndc_x, ndc_y, hit.depth);
    /// let world = view_proj_inv.project_point3(ndc);
    /// ```
    pub depth: f32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert screen position (in viewport-local pixels) to a world-space ray.
///
/// Returns (origin, direction) both as glam::Vec3.
///
/// # Arguments
/// * `screen_pos` — mouse position relative to viewport rect top-left
/// * `viewport_size` — viewport width and height in pixels
/// * `view_proj_inv` — inverse of (proj * view)
pub fn screen_to_ray(
    screen_pos: glam::Vec2,
    viewport_size: glam::Vec2,
    view_proj_inv: glam::Mat4,
) -> (glam::Vec3, glam::Vec3) {
    let ndc_x = (screen_pos.x / viewport_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / viewport_size.y) * 2.0; // Y flipped (screen Y down, NDC Y up)
    let near = view_proj_inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 0.0));
    let far = view_proj_inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
    let dir = (far - near).normalize();
    (near, dir)
}

/// Cast a ray against all visible viewport objects. Returns a [`PickHit`] for the
/// nearest hit, or `None` if nothing was hit.
///
/// # Arguments
/// * `ray_origin` — world-space ray origin
/// * `ray_dir` — world-space ray direction (normalized)
/// * `objects` — slice of trait objects implementing ViewportObject
/// * `mesh_lookup` — lookup table: CPU-side positions and indices by mesh_id
pub fn pick_scene(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    objects: &[&dyn ViewportObject],
    mesh_lookup: &std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
) -> Option<PickHit> {
    // parry3d 0.26 uses glam::Vec3 directly (via glamx)
    let ray = Ray::new(
        Vector::new(ray_origin.x, ray_origin.y, ray_origin.z),
        Vector::new(ray_dir.x, ray_dir.y, ray_dir.z),
    );

    let mut best_hit: Option<(u64, f32, PickHit)> = None;

    for obj in objects {
        if !obj.is_visible() {
            continue;
        }
        let Some(mesh_id) = obj.mesh_id() else {
            continue;
        };

        if let Some((positions, indices)) = mesh_lookup.get(&mesh_id) {
            // Build parry3d TriMesh for ray cast test.
            // parry3d::math::Vector == glam::Vec3 in f32 mode.
            // Pose only carries translation + rotation, so scale must be baked
            // into the vertices so the hit shape matches the visual geometry.
            let s = obj.scale();
            let verts: Vec<Vector> = positions
                .iter()
                .map(|p: &[f32; 3]| Vector::new(p[0] * s.x, p[1] * s.y, p[2] * s.z))
                .collect();

            let tri_indices: Vec<[u32; 3]> = indices
                .chunks(3)
                .filter(|c: &&[u32]| c.len() == 3)
                .map(|c: &[u32]| [c[0], c[1], c[2]])
                .collect();

            if tri_indices.is_empty() {
                continue;
            }

            match parry3d::shape::TriMesh::new(verts, tri_indices) {
                Ok(trimesh) => {
                    // Build pose from object position and rotation.
                    // cast_ray_and_get_normal with a pose automatically transforms
                    // the normal into world space.
                    let pose = Pose::from_parts(obj.position(), obj.rotation());
                    if let Some(intersection) =
                        trimesh.cast_ray_and_get_normal(&pose, &ray, f32::MAX, true)
                    {
                        let toi = intersection.time_of_impact;
                        if best_hit.is_none() || toi < best_hit.as_ref().unwrap().1 {
                            let sub_object = SubObjectRef::from_feature_id(intersection.feature);
                            let world_pos = ray_origin + ray_dir * toi;
                            // intersection.normal is already in world space (pose transforms it).
                            let normal = intersection.normal;
                            let triangle_index = if let Some(SubObjectRef::Face(i)) = sub_object {
                                i
                            } else {
                                u32::MAX
                            };
                            #[allow(deprecated)]
                            let hit = PickHit {
                                id: obj.id(),
                                sub_object,
                                triangle_index,
                                world_pos,
                                normal,
                                point_index: None,
                                scalar_value: None,
                            };
                            best_hit = Some((obj.id(), toi, hit));
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(object_id = obj.id(), error = %e, "TriMesh construction failed for picking");
                }
            }
        }
    }

    best_hit.map(|(_, _, hit)| hit)
}

/// Cast a ray against all visible scene nodes. Returns a [`PickHit`] for the nearest hit.
///
/// Same ray-cast logic as `pick_scene` but reads from `Scene::nodes()` instead
/// of `&[&dyn ViewportObject]`.
pub fn pick_scene_nodes(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    scene: &crate::scene::scene::Scene,
    mesh_lookup: &std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
) -> Option<PickHit> {
    let nodes: Vec<&dyn ViewportObject> = scene.nodes().map(|n| n as &dyn ViewportObject).collect();
    pick_scene(ray_origin, ray_dir, &nodes, mesh_lookup)
}

// ---------------------------------------------------------------------------
// Probe-aware picking — scalar value at hit point
// ---------------------------------------------------------------------------

/// Per-object attribute binding for probe-aware picking.
///
/// Maps an object ID to its active scalar attribute data so that
/// `pick_scene_with_probe` can interpolate the scalar value at the hit point.
pub struct ProbeBinding<'a> {
    /// Object/node ID this binding applies to.
    pub id: u64,
    /// Which attribute is active (name + vertex/cell kind).
    pub attribute_ref: &'a AttributeRef,
    /// The raw attribute data (vertex or cell scalars).
    pub attribute_data: &'a AttributeData,
    /// CPU-side mesh positions for barycentric computation.
    pub positions: &'a [[f32; 3]],
    /// CPU-side mesh indices (triangle list) for vertex lookup.
    pub indices: &'a [u32],
}

/// Compute barycentric coordinates of point `p` on the triangle `(a, b, c)`.
///
/// Returns `(u, v, w)` where `p ≈ u*a + v*b + w*c` and `u + v + w ≈ 1`.
/// Uses the robust area-ratio method (Cramer's rule on the edge vectors).
fn barycentric(p: glam::Vec3, a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        // Degenerate triangle — return equal weights.
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }
    let inv = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv;
    let w = (d00 * d21 - d01 * d20) * inv;
    let u = 1.0 - v - w;
    (u, v, w)
}

/// Given a `PickHit` and matching `ProbeBinding`, compute the scalar value at
/// the hit point and write it into `hit.scalar_value`.
fn probe_scalar(hit: &mut PickHit, binding: &ProbeBinding<'_>) {
    let tri_idx_raw = match hit.sub_object {
        Some(SubObjectRef::Face(i)) => i,
        _ => return,
    };

    let num_triangles = binding.indices.len() / 3;
    // parry3d may return back-face indices (idx >= num_triangles) for solid
    // meshes. Map them back to the original triangle.
    let tri_idx = if (tri_idx_raw as usize) >= num_triangles && num_triangles > 0 {
        tri_idx_raw as usize - num_triangles
    } else {
        tri_idx_raw as usize
    };

    match binding.attribute_ref.kind {
        AttributeKind::Cell => {
            // Cell attribute: one value per triangle — use directly.
            if let AttributeData::Cell(data) = binding.attribute_data {
                if let Some(&val) = data.get(tri_idx) {
                    hit.scalar_value = Some(val);
                }
            }
        }
        AttributeKind::Vertex => {
            // Vertex attribute: barycentric interpolation from triangle corners.
            if let AttributeData::Vertex(data) = binding.attribute_data {
                let base = tri_idx * 3;
                if base + 2 >= binding.indices.len() {
                    return;
                }
                let i0 = binding.indices[base] as usize;
                let i1 = binding.indices[base + 1] as usize;
                let i2 = binding.indices[base + 2] as usize;

                if i0 >= data.len() || i1 >= data.len() || i2 >= data.len() {
                    return;
                }
                if i0 >= binding.positions.len()
                    || i1 >= binding.positions.len()
                    || i2 >= binding.positions.len()
                {
                    return;
                }

                let a = glam::Vec3::from(binding.positions[i0]);
                let b = glam::Vec3::from(binding.positions[i1]);
                let c = glam::Vec3::from(binding.positions[i2]);
                let (u, v, w) = barycentric(hit.world_pos, a, b, c);
                hit.scalar_value = Some(data[i0] * u + data[i1] * v + data[i2] * w);
            }
        }
    }
}

/// Like [`pick_scene`] but also computes the scalar attribute value at the hit
/// point via barycentric interpolation (vertex attributes) or direct lookup
/// (cell attributes).
///
/// `probe_bindings` maps object IDs to their active attribute data. If the hit
/// object has no matching binding, `PickHit::scalar_value` remains `None`.
pub fn pick_scene_with_probe(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    objects: &[&dyn ViewportObject],
    mesh_lookup: &std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    probe_bindings: &[ProbeBinding<'_>],
) -> Option<PickHit> {
    let mut hit = pick_scene(ray_origin, ray_dir, objects, mesh_lookup)?;
    if let Some(binding) = probe_bindings.iter().find(|b| b.id == hit.id) {
        probe_scalar(&mut hit, binding);
    }
    Some(hit)
}

/// Like [`pick_scene_nodes`] but also computes the scalar value at the hit point.
///
/// See [`pick_scene_with_probe`] for details on probe bindings.
pub fn pick_scene_nodes_with_probe(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    scene: &crate::scene::scene::Scene,
    mesh_lookup: &std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    probe_bindings: &[ProbeBinding<'_>],
) -> Option<PickHit> {
    let mut hit = pick_scene_nodes(ray_origin, ray_dir, scene, mesh_lookup)?;
    if let Some(binding) = probe_bindings.iter().find(|b| b.id == hit.id) {
        probe_scalar(&mut hit, binding);
    }
    Some(hit)
}

/// Like [`pick_scene_accelerated`](crate::geometry::bvh::pick_scene_accelerated) but also
/// computes the scalar value at the hit point.
///
/// See [`pick_scene_with_probe`] for details on probe bindings.
pub fn pick_scene_accelerated_with_probe(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    accelerator: &mut crate::geometry::bvh::PickAccelerator,
    mesh_lookup: &std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    probe_bindings: &[ProbeBinding<'_>],
) -> Option<PickHit> {
    let mut hit = accelerator.pick(ray_origin, ray_dir, mesh_lookup)?;
    if let Some(binding) = probe_bindings.iter().find(|b| b.id == hit.id) {
        probe_scalar(&mut hit, binding);
    }
    Some(hit)
}

// ---------------------------------------------------------------------------
// RectPickResult — rubber-band / sub-object selection
// ---------------------------------------------------------------------------

/// Result of a rectangular (rubber-band) pick.
///
/// Maps each hit object's identifier to the typed sub-object references that
/// fall inside the selection rectangle:
/// - For mesh objects: [`SubObjectRef::Face`] entries whose centroid projects inside the rect.
/// - For point clouds: [`SubObjectRef::Point`] entries whose position projects inside the rect.
#[derive(Clone, Debug, Default)]
pub struct RectPickResult {
    /// Per-object typed sub-object references.
    ///
    /// Key = object identifier (`mesh_index` cast to `u64` for scene items,
    /// [`crate::renderer::PointCloudItem::id`] for point clouds).
    /// Value = [`SubObjectRef`]s inside the rect — `Face` for mesh triangles,
    /// `Point` for point cloud points.
    pub hits: std::collections::HashMap<u64, Vec<SubObjectRef>>,
}

impl RectPickResult {
    /// Returns `true` when no objects were hit.
    pub fn is_empty(&self) -> bool {
        self.hits.is_empty()
    }

    /// Total number of sub-object indices across all hit objects.
    pub fn total_count(&self) -> usize {
        self.hits.values().map(|v| v.len()).sum()
    }
}

/// Sub-object (triangle / point) selection inside a screen-space rectangle.
///
/// Projects triangle centroids (for mesh scene items) and point positions (for
/// point clouds) through `view_proj`, then tests NDC containment against the
/// rectangle defined by `rect_min`..`rect_max` (viewport-local pixels, top-left
/// origin).
///
/// This is a **pure CPU** operation — no GPU readback is required.
///
/// # Arguments
/// * `rect_min` — top-left corner of the selection rect in viewport pixels
/// * `rect_max` — bottom-right corner of the selection rect in viewport pixels
/// * `scene_items` — visible scene render items for this frame
/// * `mesh_lookup` — CPU-side mesh data keyed by `SceneRenderItem::mesh_index`
/// * `point_clouds` — point cloud items for this frame
/// * `view_proj` — combined view × projection matrix
/// * `viewport_size` — viewport width × height in pixels
pub fn pick_rect(
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
    scene_items: &[crate::renderer::SceneRenderItem],
    mesh_lookup: &std::collections::HashMap<usize, (Vec<[f32; 3]>, Vec<u32>)>,
    point_clouds: &[crate::renderer::PointCloudItem],
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> RectPickResult {
    // Convert screen rect to NDC rect.
    // Screen: x right, y down. NDC: x right, y up.
    let ndc_min = glam::Vec2::new(
        rect_min.x / viewport_size.x * 2.0 - 1.0,
        1.0 - rect_max.y / viewport_size.y * 2.0, // rect_max.y is the bottom in screen space
    );
    let ndc_max = glam::Vec2::new(
        rect_max.x / viewport_size.x * 2.0 - 1.0,
        1.0 - rect_min.y / viewport_size.y * 2.0, // rect_min.y is the top in screen space
    );

    let mut result = RectPickResult::default();

    // --- Mesh scene items ---
    for item in scene_items {
        if !item.visible {
            continue;
        }
        let Some((positions, indices)) = mesh_lookup.get(&item.mesh_index) else {
            continue;
        };

        let model = glam::Mat4::from_cols_array_2d(&item.model);
        let mvp = view_proj * model;

        let mut tri_hits: Vec<SubObjectRef> = Vec::new();

        for (tri_idx, chunk) in indices.chunks(3).enumerate() {
            if chunk.len() < 3 {
                continue;
            }
            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                continue;
            }

            let p0 = glam::Vec3::from(positions[i0]);
            let p1 = glam::Vec3::from(positions[i1]);
            let p2 = glam::Vec3::from(positions[i2]);
            let centroid = (p0 + p1 + p2) / 3.0;

            let clip = mvp * centroid.extend(1.0);
            if clip.w <= 0.0 {
                // Behind the camera — skip.
                continue;
            }
            let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);

            if ndc.x >= ndc_min.x && ndc.x <= ndc_max.x && ndc.y >= ndc_min.y && ndc.y <= ndc_max.y
            {
                tri_hits.push(SubObjectRef::Face(tri_idx as u32));
            }
        }

        if !tri_hits.is_empty() {
            result.hits.insert(item.mesh_index as u64, tri_hits);
        }
    }

    // --- Point cloud items ---
    for pc in point_clouds {
        if pc.id == 0 {
            // Not pickable.
            continue;
        }

        let model = glam::Mat4::from_cols_array_2d(&pc.model);
        let mvp = view_proj * model;

        let mut pt_hits: Vec<SubObjectRef> = Vec::new();

        for (pt_idx, pos) in pc.positions.iter().enumerate() {
            let p = glam::Vec3::from(*pos);
            let clip = mvp * p.extend(1.0);
            if clip.w <= 0.0 {
                continue;
            }
            let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);

            if ndc.x >= ndc_min.x && ndc.x <= ndc_max.x && ndc.y >= ndc_min.y && ndc.y <= ndc_max.y
            {
                pt_hits.push(SubObjectRef::Point(pt_idx as u32));
            }
        }

        if !pt_hits.is_empty() {
            result.hits.insert(pc.id, pt_hits);
        }
    }

    result
}

/// Select all visible objects whose world-space position projects inside a
/// screen-space rectangle.
///
/// Projects each object's position via `view_proj` to screen coordinates,
/// then tests containment in the rectangle defined by `rect_min`..`rect_max`
/// (in viewport-local pixels, top-left origin).
///
/// Returns the IDs of all objects inside the rectangle.
pub fn box_select(
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
    objects: &[&dyn ViewportObject],
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> Vec<u64> {
    let mut hits = Vec::new();
    for obj in objects {
        if !obj.is_visible() {
            continue;
        }
        let pos = obj.position();
        let clip = view_proj * pos.extend(1.0);
        // Behind the camera — skip.
        if clip.w <= 0.0 {
            continue;
        }
        let ndc = glam::Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        let screen = glam::Vec2::new(
            (ndc.x + 1.0) * 0.5 * viewport_size.x,
            (1.0 - ndc.y) * 0.5 * viewport_size.y,
        );
        if screen.x >= rect_min.x
            && screen.x <= rect_max.x
            && screen.y >= rect_min.y
            && screen.y <= rect_max.y
        {
            hits.push(obj.id());
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::traits::ViewportObject;
    use std::collections::HashMap;

    struct TestObject {
        id: u64,
        mesh_id: u64,
        position: glam::Vec3,
        visible: bool,
    }

    impl ViewportObject for TestObject {
        fn id(&self) -> u64 {
            self.id
        }
        fn mesh_id(&self) -> Option<u64> {
            Some(self.mesh_id)
        }
        fn model_matrix(&self) -> glam::Mat4 {
            glam::Mat4::from_translation(self.position)
        }
        fn position(&self) -> glam::Vec3 {
            self.position
        }
        fn rotation(&self) -> glam::Quat {
            glam::Quat::IDENTITY
        }
        fn is_visible(&self) -> bool {
            self.visible
        }
        fn color(&self) -> glam::Vec3 {
            glam::Vec3::ONE
        }
    }

    /// Unit cube centered at origin: 8 vertices, 12 triangles.
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
            0, 1, 2, 2, 3, 0, // front
            4, 6, 5, 6, 4, 7, // back
            0, 3, 7, 7, 4, 0, // left
            1, 5, 6, 6, 2, 1, // right
            3, 2, 6, 6, 7, 3, // top
            0, 4, 5, 5, 1, 0, // bottom
        ];
        (positions, indices)
    }

    #[test]
    fn test_screen_to_ray_center() {
        // Identity view-proj: screen center should produce a ray along -Z.
        let vp_inv = glam::Mat4::IDENTITY;
        let (origin, dir) = screen_to_ray(
            glam::Vec2::new(400.0, 300.0),
            glam::Vec2::new(800.0, 600.0),
            vp_inv,
        );
        // NDC (0,0) -> origin at (0,0,0), direction toward (0,0,1).
        assert!(origin.x.abs() < 1e-3, "origin.x={}", origin.x);
        assert!(origin.y.abs() < 1e-3, "origin.y={}", origin.y);
        assert!(dir.z.abs() > 0.9, "dir should be along Z, got {dir:?}");
    }

    #[test]
    fn test_pick_scene_hit() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(1u64, (positions, indices));

        let obj = TestObject {
            id: 42,
            mesh_id: 1,
            position: glam::Vec3::ZERO,
            visible: true,
        };
        let objects: Vec<&dyn ViewportObject> = vec![&obj];

        // Ray from +Z toward origin should hit the cube.
        let result = pick_scene(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &objects,
            &mesh_lookup,
        );
        assert!(result.is_some(), "expected a hit");
        let hit = result.unwrap();
        assert_eq!(hit.id, 42);
        // Front face of unit cube at origin is at z=0.5; ray from z=5 hits at toi=4.5.
        assert!(
            (hit.world_pos.z - 0.5).abs() < 0.01,
            "world_pos.z={}",
            hit.world_pos.z
        );
        // Normal should point roughly toward +Z (toward camera).
        assert!(hit.normal.z > 0.9, "normal={:?}", hit.normal);
    }

    #[test]
    fn test_pick_scene_miss() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(1u64, (positions, indices));

        let obj = TestObject {
            id: 42,
            mesh_id: 1,
            position: glam::Vec3::ZERO,
            visible: true,
        };
        let objects: Vec<&dyn ViewportObject> = vec![&obj];

        // Ray far from geometry should miss.
        let result = pick_scene(
            glam::Vec3::new(100.0, 100.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &objects,
            &mesh_lookup,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_pick_nearest_wins() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(1u64, (positions.clone(), indices.clone()));
        mesh_lookup.insert(2u64, (positions, indices));

        let near_obj = TestObject {
            id: 10,
            mesh_id: 1,
            position: glam::Vec3::new(0.0, 0.0, 2.0),
            visible: true,
        };
        let far_obj = TestObject {
            id: 20,
            mesh_id: 2,
            position: glam::Vec3::new(0.0, 0.0, -2.0),
            visible: true,
        };
        let objects: Vec<&dyn ViewportObject> = vec![&far_obj, &near_obj];

        // Ray from +Z toward -Z should hit the nearer object first.
        let result = pick_scene(
            glam::Vec3::new(0.0, 0.0, 10.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &objects,
            &mesh_lookup,
        );
        assert!(result.is_some(), "expected a hit");
        assert_eq!(result.unwrap().id, 10);
    }

    #[test]
    fn test_box_select_hits_inside_rect() {
        // Place object at origin, use an identity-like view_proj so it projects to screen center.
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let vp = proj * view;
        let viewport_size = glam::Vec2::new(800.0, 600.0);

        let obj = TestObject {
            id: 42,
            mesh_id: 1,
            position: glam::Vec3::ZERO,
            visible: true,
        };
        let objects: Vec<&dyn ViewportObject> = vec![&obj];

        // Rectangle around screen center should capture the object.
        let result = box_select(
            glam::Vec2::new(300.0, 200.0),
            glam::Vec2::new(500.0, 400.0),
            &objects,
            vp,
            viewport_size,
        );
        assert_eq!(result, vec![42]);
    }

    #[test]
    fn test_box_select_skips_hidden() {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let vp = proj * view;
        let viewport_size = glam::Vec2::new(800.0, 600.0);

        let obj = TestObject {
            id: 42,
            mesh_id: 1,
            position: glam::Vec3::ZERO,
            visible: false,
        };
        let objects: Vec<&dyn ViewportObject> = vec![&obj];

        let result = box_select(
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(800.0, 600.0),
            &objects,
            vp,
            viewport_size,
        );
        assert!(result.is_empty());
    }

    #[test]
    fn test_pick_scene_nodes_hit() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        let mut scene = crate::scene::scene::Scene::new();
        scene.add(
            Some(crate::resources::mesh_store::MeshId(0)),
            glam::Mat4::IDENTITY,
            crate::scene::material::Material::default(),
        );
        scene.update_transforms();

        let result = pick_scene_nodes(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &scene,
            &mesh_lookup,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_pick_scene_nodes_miss() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(0u64, (positions, indices));

        let mut scene = crate::scene::scene::Scene::new();
        scene.add(
            Some(crate::resources::mesh_store::MeshId(0)),
            glam::Mat4::IDENTITY,
            crate::scene::material::Material::default(),
        );
        scene.update_transforms();

        let result = pick_scene_nodes(
            glam::Vec3::new(100.0, 100.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &scene,
            &mesh_lookup,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_probe_vertex_attribute() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(1u64, (positions.clone(), indices.clone()));

        // Assign a per-vertex scalar: value = vertex index as f32.
        let vertex_scalars: Vec<f32> = (0..positions.len()).map(|i| i as f32).collect();

        let obj = TestObject {
            id: 42,
            mesh_id: 1,
            position: glam::Vec3::ZERO,
            visible: true,
        };
        let objects: Vec<&dyn ViewportObject> = vec![&obj];

        let attr_ref = AttributeRef {
            name: "test".to_string(),
            kind: AttributeKind::Vertex,
        };
        let attr_data = AttributeData::Vertex(vertex_scalars);
        let bindings = vec![ProbeBinding {
            id: 42,
            attribute_ref: &attr_ref,
            attribute_data: &attr_data,
            positions: &positions,
            indices: &indices,
        }];

        let result = pick_scene_with_probe(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &objects,
            &mesh_lookup,
            &bindings,
        );
        assert!(result.is_some(), "expected a hit");
        let hit = result.unwrap();
        assert_eq!(hit.id, 42);
        // scalar_value should be populated (interpolated from vertex scalars).
        assert!(
            hit.scalar_value.is_some(),
            "expected scalar_value to be set"
        );
    }

    #[test]
    fn test_probe_cell_attribute() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(1u64, (positions.clone(), indices.clone()));

        // 12 triangles in a unit cube — assign each a scalar.
        let num_triangles = indices.len() / 3;
        let cell_scalars: Vec<f32> = (0..num_triangles).map(|i| (i as f32) * 10.0).collect();

        let obj = TestObject {
            id: 42,
            mesh_id: 1,
            position: glam::Vec3::ZERO,
            visible: true,
        };
        let objects: Vec<&dyn ViewportObject> = vec![&obj];

        let attr_ref = AttributeRef {
            name: "pressure".to_string(),
            kind: AttributeKind::Cell,
        };
        let attr_data = AttributeData::Cell(cell_scalars.clone());
        let bindings = vec![ProbeBinding {
            id: 42,
            attribute_ref: &attr_ref,
            attribute_data: &attr_data,
            positions: &positions,
            indices: &indices,
        }];

        let result = pick_scene_with_probe(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &objects,
            &mesh_lookup,
            &bindings,
        );
        assert!(result.is_some());
        let hit = result.unwrap();
        // Cell attribute value should be one of the triangle scalars.
        assert!(hit.scalar_value.is_some());
        let val = hit.scalar_value.unwrap();
        assert!(
            cell_scalars.contains(&val),
            "scalar_value {val} not in cell_scalars"
        );
    }

    #[test]
    fn test_probe_no_binding_leaves_none() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup = HashMap::new();
        mesh_lookup.insert(1u64, (positions, indices));

        let obj = TestObject {
            id: 42,
            mesh_id: 1,
            position: glam::Vec3::ZERO,
            visible: true,
        };
        let objects: Vec<&dyn ViewportObject> = vec![&obj];

        // No probe bindings — scalar_value should remain None.
        let result = pick_scene_with_probe(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            &objects,
            &mesh_lookup,
            &[],
        );
        assert!(result.is_some());
        assert!(result.unwrap().scalar_value.is_none());
    }

    // ---------------------------------------------------------------------------
    // pick_rect tests
    // ---------------------------------------------------------------------------

    /// Build a simple perspective view_proj looking at the origin from +Z.
    fn make_view_proj() -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        proj * view
    }

    #[test]
    fn test_pick_rect_mesh_full_screen() {
        // A full-screen rect should capture all triangle centroids of the unit cube.
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup: std::collections::HashMap<usize, (Vec<[f32; 3]>, Vec<u32>)> =
            std::collections::HashMap::new();
        mesh_lookup.insert(0, (positions, indices.clone()));

        let item = crate::renderer::SceneRenderItem {
            mesh_index: 0,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            visible: true,
            ..Default::default()
        };

        let view_proj = make_view_proj();
        let viewport = glam::Vec2::new(800.0, 600.0);

        let result = pick_rect(
            glam::Vec2::ZERO,
            viewport,
            &[item],
            &mesh_lookup,
            &[],
            view_proj,
            viewport,
        );

        // The cube has 12 triangles; front-facing ones project inside the full-screen rect.
        assert!(!result.is_empty(), "expected at least one triangle hit");
        assert!(result.total_count() > 0);
    }

    #[test]
    fn test_pick_rect_miss() {
        // A rect far off-screen should return empty results.
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup: std::collections::HashMap<usize, (Vec<[f32; 3]>, Vec<u32>)> =
            std::collections::HashMap::new();
        mesh_lookup.insert(0, (positions, indices));

        let item = crate::renderer::SceneRenderItem {
            mesh_index: 0,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            visible: true,
            ..Default::default()
        };

        let view_proj = make_view_proj();
        let viewport = glam::Vec2::new(800.0, 600.0);

        let result = pick_rect(
            glam::Vec2::new(700.0, 500.0), // bottom-right corner, cube projects to center
            glam::Vec2::new(799.0, 599.0),
            &[item],
            &mesh_lookup,
            &[],
            view_proj,
            viewport,
        );

        assert!(result.is_empty(), "expected no hits in off-center rect");
    }

    #[test]
    fn test_pick_rect_point_cloud() {
        // Points at the origin should be captured by a full-screen rect.
        let view_proj = make_view_proj();
        let viewport = glam::Vec2::new(800.0, 600.0);

        let pc = crate::renderer::PointCloudItem {
            positions: vec![[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]],
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            id: 99,
            ..Default::default()
        };

        let result = pick_rect(
            glam::Vec2::ZERO,
            viewport,
            &[],
            &std::collections::HashMap::new(),
            &[pc],
            view_proj,
            viewport,
        );

        assert!(!result.is_empty(), "expected point cloud hits");
        let hits = result.hits.get(&99).expect("expected hits for id 99");
        assert_eq!(
            hits.len(),
            2,
            "both points should be inside the full-screen rect"
        );
        // Verify the hits are typed as Point sub-objects.
        assert!(hits.iter().all(|s| s.is_point()), "expected SubObjectRef::Point entries");
        assert_eq!(hits[0], SubObjectRef::Point(0));
        assert_eq!(hits[1], SubObjectRef::Point(1));
    }

    #[test]
    fn test_pick_rect_skips_invisible() {
        let (positions, indices) = unit_cube_mesh();
        let mut mesh_lookup: std::collections::HashMap<usize, (Vec<[f32; 3]>, Vec<u32>)> =
            std::collections::HashMap::new();
        mesh_lookup.insert(0, (positions, indices));

        let item = crate::renderer::SceneRenderItem {
            mesh_index: 0,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            visible: false, // hidden
            ..Default::default()
        };

        let view_proj = make_view_proj();
        let viewport = glam::Vec2::new(800.0, 600.0);

        let result = pick_rect(
            glam::Vec2::ZERO,
            viewport,
            &[item],
            &mesh_lookup,
            &[],
            view_proj,
            viewport,
        );

        assert!(result.is_empty(), "invisible items should be skipped");
    }

    #[test]
    fn test_pick_rect_result_type() {
        // Verify RectPickResult accessors.
        let mut r = RectPickResult::default();
        assert!(r.is_empty());
        assert_eq!(r.total_count(), 0);

        r.hits.insert(1, vec![SubObjectRef::Face(0), SubObjectRef::Face(1), SubObjectRef::Face(2)]);
        r.hits.insert(2, vec![SubObjectRef::Point(5)]);
        assert!(!r.is_empty());
        assert_eq!(r.total_count(), 4);
    }

    #[test]
    fn test_barycentric_at_vertices() {
        let a = glam::Vec3::new(0.0, 0.0, 0.0);
        let b = glam::Vec3::new(1.0, 0.0, 0.0);
        let c = glam::Vec3::new(0.0, 1.0, 0.0);

        // At vertex a: u=1, v=0, w=0.
        let (u, v, w) = super::barycentric(a, a, b, c);
        assert!((u - 1.0).abs() < 1e-5, "u={u}");
        assert!(v.abs() < 1e-5, "v={v}");
        assert!(w.abs() < 1e-5, "w={w}");

        // At vertex b: u=0, v=1, w=0.
        let (u, v, w) = super::barycentric(b, a, b, c);
        assert!(u.abs() < 1e-5, "u={u}");
        assert!((v - 1.0).abs() < 1e-5, "v={v}");
        assert!(w.abs() < 1e-5, "w={w}");

        // At centroid: u=v=w≈1/3.
        let centroid = (a + b + c) / 3.0;
        let (u, v, w) = super::barycentric(centroid, a, b, c);
        assert!((u - 1.0 / 3.0).abs() < 1e-4, "u={u}");
        assert!((v - 1.0 / 3.0).abs() < 1e-4, "v={v}");
        assert!((w - 1.0 / 3.0).abs() < 1e-4, "w={w}");
    }
}
