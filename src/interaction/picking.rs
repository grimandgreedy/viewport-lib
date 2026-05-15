/// Ray-cast picking for the 3D viewport.
///
/// Uses parry3d 0.26's glam-native API (no nalgebra required).
/// All conversions are contained here at the picking boundary.
use crate::geometry::marching_cubes::VolumeData;
use crate::interaction::sub_object::SubObjectRef;
use crate::resources::volume_mesh::{CELL_SENTINEL, VolumeMeshData};
use crate::resources::{AttributeData, AttributeKind, AttributeRef};
use crate::scene::traits::ViewportObject;
use parry3d::math::{Pose, Vector};
use parry3d::query::{Ray, RayCast};

// ---------------------------------------------------------------------------
// PickHit : rich hit result
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
    /// Typed sub-object reference : the authoritative source for sub-object identity.
    ///
    /// `Some(SubObjectRef::Face(i))` for mesh picks; `Some(SubObjectRef::Point(i))` for
    /// point cloud picks; `None` when no specific sub-object could be identified.
    pub sub_object: Option<SubObjectRef>,
    /// World-space position of the hit point (`ray_origin + ray_dir * toi`).
    pub world_pos: glam::Vec3,
    /// Surface normal at the hit point, in world space.
    pub normal: glam::Vec3,
    /// Which triangle was hit (from parry3d `FeatureId::Face`).
    /// `u32::MAX` if the feature was not a face (edge/vertex hit : rare for TriMesh).
    ///
    /// **Deprecated** : use [`sub_object`](Self::sub_object) instead.
    #[deprecated(since = "0.5.0", note = "use `sub_object` instead")]
    pub triangle_index: u32,
    /// Index of the hit point within a [`crate::renderer::PointCloudItem`].
    /// `None` for mesh picks; set when a point cloud item is hit.
    ///
    /// **Deprecated** : use [`sub_object`](Self::sub_object) instead.
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

impl PickHit {
    /// Construct a minimal `PickHit` for cases where no sub-object is identified
    /// (e.g. volume AABB hits). `normal` is an approximate inward normal.
    #[allow(deprecated)]
    pub fn object_hit(id: u64, world_pos: glam::Vec3, normal: glam::Vec3) -> Self {
        Self {
            id,
            sub_object: None,
            world_pos,
            normal,
            triangle_index: u32::MAX,
            point_index: None,
            scalar_value: None,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuPickHit : GPU object-ID pick result
// ---------------------------------------------------------------------------

/// Result of a GPU object-ID pick pass.
///
/// Lighter than [`PickHit`] : carries only the object identifier and the
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
    /// maintains. [`crate::renderer::PickId::NONE`] is never returned
    /// (non-pickable surfaces are excluded from the pick pass).
    pub object_id: crate::renderer::PickId,
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
/// * `screen_pos` : mouse position relative to viewport rect top-left
/// * `viewport_size` : viewport width and height in pixels
/// * `view_proj_inv` : inverse of (proj * view)
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
/// * `ray_origin` : world-space ray origin
/// * `ray_dir` : world-space ray direction (normalized)
/// * `objects` : slice of trait objects implementing ViewportObject
/// * `mesh_lookup` : lookup table: CPU-side positions and indices by mesh_id
pub fn pick_scene_cpu(
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
/// Same ray-cast logic as `pick_scene_cpu` but reads from `Scene::nodes()` instead
/// of `&[&dyn ViewportObject]`.
pub fn pick_scene_nodes_cpu(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    scene: &crate::scene::scene::Scene,
    mesh_lookup: &std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
) -> Option<PickHit> {
    let nodes: Vec<&dyn ViewportObject> = scene.nodes().map(|n| n as &dyn ViewportObject).collect();
    pick_scene_cpu(ray_origin, ray_dir, &nodes, mesh_lookup)
}

// ---------------------------------------------------------------------------
// Probe-aware picking : scalar value at hit point
// ---------------------------------------------------------------------------

/// Per-object attribute binding for probe-aware picking.
///
/// Maps an object ID to its active scalar attribute data so that
/// `pick_scene_with_probe_cpu` can interpolate the scalar value at the hit point.
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
        // Degenerate triangle : return equal weights.
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
            // Cell attribute: one value per triangle : use directly.
            if let AttributeData::Cell(data) = binding.attribute_data {
                if let Some(&val) = data.get(tri_idx) {
                    hit.scalar_value = Some(val);
                }
            }
        }
        AttributeKind::Face => {
            // Face attribute: one value per triangle : direct lookup (no averaging).
            if let AttributeData::Face(data) = binding.attribute_data {
                if let Some(&val) = data.get(tri_idx) {
                    hit.scalar_value = Some(val);
                }
            }
        }
        AttributeKind::FaceColor => {
            // FaceColor attribute: no scalar value to report.
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
        AttributeKind::Edge => {
            // Edge attribute: use the corner value at the closest triangle vertex
            // (edge values are already averaged to vertices at upload time).
            if let AttributeData::Edge(data) = binding.attribute_data {
                let base = tri_idx * 3;
                if base + 2 >= binding.indices.len() || data.is_empty() {
                    return;
                }
                let i0 = binding.indices[base] as usize;
                let i1 = binding.indices[base + 1] as usize;
                let i2 = binding.indices[base + 2] as usize;
                if i0 < data.len() || i1 < data.len() || i2 < data.len() {
                    // Barycentric interpolation over the per-vertex averaged values.
                    if i0 < data.len()
                        && i1 < data.len()
                        && i2 < data.len()
                        && i0 < binding.positions.len()
                        && i1 < binding.positions.len()
                        && i2 < binding.positions.len()
                    {
                        let a = glam::Vec3::from(binding.positions[i0]);
                        let b = glam::Vec3::from(binding.positions[i1]);
                        let c = glam::Vec3::from(binding.positions[i2]);
                        let (u, v, w) = barycentric(hit.world_pos, a, b, c);
                        hit.scalar_value = Some(data[i0] * u + data[i1] * v + data[i2] * w);
                    }
                }
            }
        }
        AttributeKind::Halfedge | AttributeKind::Corner => {
            // Per-corner attributes: `values[3*t + k]` is the k-th corner of the triangle.
            // Report the value at the nearest corner (flat shading).
            let extract = |data: &[f32]| -> Option<f32> {
                let base = tri_idx * 3;
                if base + 2 >= data.len() {
                    return None;
                }
                // Return the first corner value as the representative (flat per face).
                Some(data[base])
            };
            match binding.attribute_data {
                AttributeData::Halfedge(data) | AttributeData::Corner(data) => {
                    hit.scalar_value = extract(data);
                }
                _ => {}
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
pub fn pick_scene_with_probe_cpu(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    objects: &[&dyn ViewportObject],
    mesh_lookup: &std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    probe_bindings: &[ProbeBinding<'_>],
) -> Option<PickHit> {
    let mut hit = pick_scene_cpu(ray_origin, ray_dir, objects, mesh_lookup)?;
    if let Some(binding) = probe_bindings.iter().find(|b| b.id == hit.id) {
        probe_scalar(&mut hit, binding);
    }
    Some(hit)
}

/// Like [`pick_scene_nodes_cpu`] but also computes the scalar value at the hit point.
///
/// See [`pick_scene_with_probe_cpu`] for details on probe bindings.
pub fn pick_scene_nodes_with_probe_cpu(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    scene: &crate::scene::scene::Scene,
    mesh_lookup: &std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    probe_bindings: &[ProbeBinding<'_>],
) -> Option<PickHit> {
    let mut hit = pick_scene_nodes_cpu(ray_origin, ray_dir, scene, mesh_lookup)?;
    if let Some(binding) = probe_bindings.iter().find(|b| b.id == hit.id) {
        probe_scalar(&mut hit, binding);
    }
    Some(hit)
}

/// Like [`pick_scene_accelerated_cpu`](crate::geometry::bvh::pick_scene_accelerated_cpu) but also
/// computes the scalar value at the hit point.
///
/// See [`pick_scene_with_probe_cpu`] for details on probe bindings.
pub fn pick_scene_accelerated_with_probe_cpu(
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
// RectPickResult : rubber-band / sub-object selection
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
    /// Key = object identifier: [`crate::renderer::PickId`]`.0` (the scene node id)
    /// for mesh scene items, [`crate::renderer::PointCloudItem::id`] for point clouds.
    /// Value = [`SubObjectRef`]s inside the rect : `Face` for mesh triangles,
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
/// This is a **pure CPU** operation : no GPU readback is required.
///
/// # Arguments
/// * `rect_min` : top-left corner of the selection rect in viewport pixels
/// * `rect_max` : bottom-right corner of the selection rect in viewport pixels
/// * `scene_items` : visible scene render items for this frame
/// * `mesh_lookup` : CPU-side mesh data keyed by `SceneRenderItem::mesh_index`
/// * `point_clouds` : point cloud items for this frame
/// * `view_proj` : combined view × projection matrix
/// * `viewport_size` : viewport width × height in pixels
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
        let Some((positions, indices)) = mesh_lookup.get(&item.mesh_id.index()) else {
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
                // Behind the camera : skip.
                continue;
            }
            let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);

            if ndc.x >= ndc_min.x && ndc.x <= ndc_max.x && ndc.y >= ndc_min.y && ndc.y <= ndc_max.y
            {
                tri_hits.push(SubObjectRef::Face(tri_idx as u32));
            }
        }

        if !tri_hits.is_empty() {
            result.hits.insert(item.pick_id.0, tri_hits);
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
        // Behind the camera : skip.
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

// ---------------------------------------------------------------------------
// Volume ray-cast picking
// ---------------------------------------------------------------------------

/// Slab-method AABB intersection in an arbitrary coordinate space.
///
/// Returns `(t_entry, t_exit, entry_axis, entry_sign)` or `None` on miss.
/// - `entry_axis` : 0/1/2 for x/y/z
/// - `entry_sign` : ±1.0 — sign of the outward face normal on the entry face
///   (points back toward the ray origin)
fn ray_aabb_volume(
    origin: glam::Vec3,
    dir: glam::Vec3,
    bbox_min: glam::Vec3,
    bbox_max: glam::Vec3,
) -> Option<(f32, f32, usize, f32)> {
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    let mut entry_axis = 0usize;
    let mut entry_sign = -1.0f32;

    let dirs = [dir.x, dir.y, dir.z];
    let origins = [origin.x, origin.y, origin.z];
    let mins = [bbox_min.x, bbox_min.y, bbox_min.z];
    let maxs = [bbox_max.x, bbox_max.y, bbox_max.z];

    for i in 0..3 {
        let d = dirs[i];
        let o = origins[i];
        if d.abs() < 1e-12 {
            // Ray parallel to slab: origin must be inside.
            if o < mins[i] || o > maxs[i] {
                return None;
            }
        } else {
            let t1 = (mins[i] - o) / d;
            let t2 = (maxs[i] - o) / d;
            let (t_near, t_far) = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            if t_near > t_min {
                t_min = t_near;
                entry_axis = i;
                // d > 0 -> entered through min face -> outward normal = -e_i -> sign = -1.
                // d < 0 -> entered through max face -> outward normal = +e_i -> sign = +1.
                entry_sign = if d > 0.0 { -1.0 } else { 1.0 };
            }
            if t_far < t_max {
                t_max = t_far;
            }
        }
    }

    if t_min > t_max || t_max < 0.0 {
        return None;
    }
    Some((t_min, t_max, entry_axis, entry_sign))
}

/// Ray-cast a single volume using Amanatides-Woo DDA traversal.
///
/// Walks voxels in exact ray order — no steps are skipped — and returns a
/// [`PickHit`] for the first voxel whose raw scalar value falls within
/// `[item.threshold_min, item.threshold_max]`.
///
/// # Arguments
/// * `ray_origin` : world-space ray origin
/// * `ray_dir` : world-space ray direction (normalized)
/// * `id` : caller-assigned object identifier, copied into [`PickHit::id`]
/// * `item` : volume render parameters (bounding box, transform, thresholds)
/// * `volume` : CPU-side scalar field — same data passed to
///   [`upload_volume`](crate::resources::ViewportGpuResources::upload_volume)
///
/// # Returns
/// `Some(PickHit)` on a hit:
/// - `sub_object` : [`SubObjectRef::Voxel`] carrying the flat grid index
///   `ix + iy*nx + iz*nx*ny`
/// - `world_pos` : ray entry point into the hit voxel
/// - `normal` : world-space outward face normal of the voxel face entered
/// - `scalar_value` : raw scalar at the hit voxel
///
/// Returns `None` if the ray misses the bounding box or every voxel in
/// the path is outside the threshold range (or NaN).
pub fn pick_volume_cpu(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    id: u64,
    item: &crate::renderer::VolumeItem,
    volume: &VolumeData,
) -> Option<PickHit> {
    let [nx, ny, nz] = volume.dims;
    if nx == 0 || ny == 0 || nz == 0 || volume.data.is_empty() {
        return None;
    }

    // Transform ray to model-local space (handles rotation, scale, translation).
    let model = glam::Mat4::from_cols_array_2d(&item.model);
    let inv_model = model.inverse();
    let local_origin = inv_model.transform_point3(ray_origin);
    let local_dir = inv_model.transform_vector3(ray_dir);

    let bbox_min = glam::Vec3::from(item.bbox_min);
    let bbox_max = glam::Vec3::from(item.bbox_max);

    let (t_entry, t_exit, entry_axis, entry_sign) =
        ray_aabb_volume(local_origin, local_dir, bbox_min, bbox_max)?;

    // Advance to AABB surface if the ray starts outside.
    let t_start = t_entry.max(0.0);
    if t_start >= t_exit {
        return None;
    }

    // Cell dimensions in local space.
    let extent = bbox_max - bbox_min;
    let cell = extent / glam::Vec3::new(nx as f32, ny as f32, nz as f32);

    // Entry point in local space.
    let p_entry = local_origin + t_start * local_dir;

    // Starting grid cell. Nudge the fractional position slightly inside to
    // avoid landing exactly on a boundary and mis-classifying the first cell.
    let eps = 1e-4_f32;
    let frac =
        ((p_entry - bbox_min) / extent).clamp(glam::Vec3::splat(eps), glam::Vec3::splat(1.0 - eps));
    let mut ix = (frac.x * nx as f32).floor() as i32;
    let mut iy = (frac.y * ny as f32).floor() as i32;
    let mut iz = (frac.z * nz as f32).floor() as i32;
    ix = ix.clamp(0, nx as i32 - 1);
    iy = iy.clamp(0, ny as i32 - 1);
    iz = iz.clamp(0, nz as i32 - 1);

    // DDA step direction per axis (+1 or -1).
    let step_x: i32 = if local_dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if local_dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if local_dir.z >= 0.0 { 1 } else { -1 };

    // t increment to traverse one cell in each axis.
    let td_x = if local_dir.x.abs() > 1e-12 {
        cell.x / local_dir.x.abs()
    } else {
        f32::INFINITY
    };
    let td_y = if local_dir.y.abs() > 1e-12 {
        cell.y / local_dir.y.abs()
    } else {
        f32::INFINITY
    };
    let td_z = if local_dir.z.abs() > 1e-12 {
        cell.z / local_dir.z.abs()
    } else {
        f32::INFINITY
    };

    // t to the next axis-aligned boundary ahead of p_entry in each axis.
    let next_bx = bbox_min.x + (if step_x > 0 { ix + 1 } else { ix }) as f32 * cell.x;
    let next_by = bbox_min.y + (if step_y > 0 { iy + 1 } else { iy }) as f32 * cell.y;
    let next_bz = bbox_min.z + (if step_z > 0 { iz + 1 } else { iz }) as f32 * cell.z;

    let mut tmax_x = if local_dir.x.abs() > 1e-12 {
        t_start + (next_bx - p_entry.x) / local_dir.x
    } else {
        f32::INFINITY
    };
    let mut tmax_y = if local_dir.y.abs() > 1e-12 {
        t_start + (next_by - p_entry.y) / local_dir.y
    } else {
        f32::INFINITY
    };
    let mut tmax_z = if local_dir.z.abs() > 1e-12 {
        t_start + (next_bz - p_entry.z) / local_dir.z
    } else {
        f32::INFINITY
    };

    // Outward face normal in local space for the face the ray is currently entering.
    let mut entry_normal_local = glam::Vec3::ZERO;
    match entry_axis {
        0 => entry_normal_local.x = entry_sign,
        1 => entry_normal_local.y = entry_sign,
        _ => entry_normal_local.z = entry_sign,
    }

    // t at which we entered the current voxel (for computing world_pos on hit).
    let mut t_voxel_entry = t_start;

    loop {
        // Safety bounds check: exit if DDA has walked outside the grid.
        if ix < 0 || ix >= nx as i32 || iy < 0 || iy >= ny as i32 || iz < 0 || iz >= nz as i32 {
            break;
        }

        let flat = ix as u32 + iy as u32 * nx + iz as u32 * nx * ny;
        let scalar = volume.data[flat as usize];

        // Skip NaN and out-of-threshold voxels (mirrors the shader behaviour).
        if !scalar.is_nan() && scalar >= item.threshold_min && scalar <= item.threshold_max {
            let local_hit = local_origin + t_voxel_entry * local_dir;
            let world_pos = model.transform_point3(local_hit);
            // Normals transform by the inverse-transpose to handle non-uniform scale.
            let world_normal = inv_model
                .transpose()
                .transform_vector3(entry_normal_local)
                .normalize();

            #[allow(deprecated)]
            return Some(PickHit {
                id,
                sub_object: Some(SubObjectRef::Voxel(flat)),
                world_pos,
                normal: world_normal,
                triangle_index: u32::MAX,
                point_index: None,
                scalar_value: Some(scalar),
            });
        }

        // Advance to the next voxel: step along the axis with the smallest tMax.
        if tmax_x <= tmax_y && tmax_x <= tmax_z {
            if tmax_x > t_exit {
                break;
            }
            t_voxel_entry = tmax_x;
            tmax_x += td_x;
            ix += step_x;
            entry_normal_local = glam::Vec3::new(-(step_x as f32), 0.0, 0.0);
        } else if tmax_y <= tmax_z {
            if tmax_y > t_exit {
                break;
            }
            t_voxel_entry = tmax_y;
            tmax_y += td_y;
            iy += step_y;
            entry_normal_local = glam::Vec3::new(0.0, -(step_y as f32), 0.0);
        } else {
            if tmax_z > t_exit {
                break;
            }
            t_voxel_entry = tmax_z;
            tmax_z += td_z;
            iz += step_z;
            entry_normal_local = glam::Vec3::new(0.0, 0.0, -(step_z as f32));
        }
    }

    None
}

/// Compute the world-space axis-aligned bounding box of a single voxel.
///
/// Given the flat voxel index from [`SubObjectRef::Voxel`], returns
/// `(world_min, world_max)` suitable for positioning a highlight wireframe
/// around the selected voxel.
///
/// When `item.model` contains rotation or non-uniform scale the returned AABB
/// is the world-space envelope of the (non-axis-aligned) voxel — computed by
/// transforming all 8 corners.
///
/// # Panics
///
/// Panics if `flat_index` is out of bounds for `volume.dims`.
pub fn voxel_world_aabb(
    flat_index: u32,
    volume: &VolumeData,
    item: &crate::renderer::VolumeItem,
) -> (glam::Vec3, glam::Vec3) {
    let [nx, ny, nz] = volume.dims;
    let ix = flat_index % nx;
    let iy = (flat_index / nx) % ny;
    let iz = flat_index / (nx * ny);
    assert!(
        ix < nx && iy < ny && iz < nz,
        "flat_index {} out of bounds for dims {:?}",
        flat_index,
        volume.dims
    );

    let bbox_min = glam::Vec3::from(item.bbox_min);
    let bbox_max = glam::Vec3::from(item.bbox_max);
    let cell = (bbox_max - bbox_min) / glam::Vec3::new(nx as f32, ny as f32, nz as f32);

    let local_lo =
        bbox_min + glam::Vec3::new(ix as f32 * cell.x, iy as f32 * cell.y, iz as f32 * cell.z);
    let local_hi = local_lo + cell;

    let model = glam::Mat4::from_cols_array_2d(&item.model);
    let corners = [
        glam::Vec3::new(local_lo.x, local_lo.y, local_lo.z),
        glam::Vec3::new(local_hi.x, local_lo.y, local_lo.z),
        glam::Vec3::new(local_lo.x, local_hi.y, local_lo.z),
        glam::Vec3::new(local_hi.x, local_hi.y, local_lo.z),
        glam::Vec3::new(local_lo.x, local_lo.y, local_hi.z),
        glam::Vec3::new(local_hi.x, local_lo.y, local_hi.z),
        glam::Vec3::new(local_lo.x, local_hi.y, local_hi.z),
        glam::Vec3::new(local_hi.x, local_hi.y, local_hi.z),
    ];

    let world_min = corners
        .iter()
        .map(|&c| model.transform_point3(c))
        .fold(glam::Vec3::splat(f32::INFINITY), |acc, c| acc.min(c));
    let world_max = corners
        .iter()
        .map(|&c| model.transform_point3(c))
        .fold(glam::Vec3::splat(f32::NEG_INFINITY), |acc, c| acc.max(c));

    (world_min, world_max)
}

/// Pick the closest point in a [`crate::renderer::PointCloudItem`] to a screen-space click.
///
/// Projects every point through `view_proj` and returns the closest one whose
/// screen-space distance to `click_pos` is within `radius_px` pixels.  Returns
/// `None` when no point is within that radius.
///
/// # Arguments
/// * `click_pos`     – screen-space click position in viewport pixels (top-left origin)
/// * `id`            – object identifier to embed in the returned [`PickHit`]
/// * `item`          – the point cloud item to search
/// * `view_proj`     – combined view × projection matrix
/// * `viewport_size` – viewport width × height in pixels
/// * `radius_px`     – maximum screen-space distance in pixels to accept as a hit
pub fn pick_point_cloud_cpu(
    click_pos: glam::Vec2,
    id: u64,
    item: &crate::renderer::PointCloudItem,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
    radius_px: f32,
) -> Option<PickHit> {
    if id == 0 || item.positions.is_empty() {
        return None;
    }

    let model = glam::Mat4::from_cols_array_2d(&item.model);
    let mvp = view_proj * model;

    let mut best_dist_sq = radius_px * radius_px;
    let mut best_idx: Option<u32> = None;
    let mut best_world = glam::Vec3::ZERO;

    for (pt_idx, pos) in item.positions.iter().enumerate() {
        let local = glam::Vec3::from(*pos);
        let clip = mvp * local.extend(1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let sx = (ndc_x + 1.0) * 0.5 * viewport_size.x;
        let sy = (1.0 - ndc_y) * 0.5 * viewport_size.y;
        let dx = sx - click_pos.x;
        let dy = sy - click_pos.y;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = Some(pt_idx as u32);
            best_world = model.transform_point3(local);
        }
    }

    let pt_idx = best_idx?;
    #[allow(deprecated)]
    Some(PickHit {
        id,
        sub_object: Some(SubObjectRef::Point(pt_idx)),
        world_pos: best_world,
        normal: glam::Vec3::Y,
        triangle_index: u32::MAX,
        point_index: Some(pt_idx),
        scalar_value: None,
    })
}

// ---------------------------------------------------------------------------
// nearest_vertex_on_hit
// ---------------------------------------------------------------------------

/// Find the triangle corner nearest to the ray-hit world position.
///
/// Takes a hit from [`pick_scene_nodes_cpu`] (which carries a
/// [`SubObjectRef::Face`] sub-object) and returns the index of the closest
/// triangle corner as a [`SubObjectRef::Vertex`].
///
/// Returns `None` if `hit.sub_object` is not a `Face`, or if the face index is
/// out of range for the provided buffers.
///
/// # Arguments
/// * `hit`       - result from a mesh ray-cast pick
/// * `positions` - local-space vertex positions for the hit mesh
/// * `indices`   - triangle index buffer (every 3 consecutive entries form one triangle)
/// * `model`     - world transform for the hit object
pub fn nearest_vertex_on_hit(
    hit: &PickHit,
    positions: &[[f32; 3]],
    indices: &[u32],
    model: glam::Mat4,
) -> Option<SubObjectRef> {
    let face_raw = match hit.sub_object {
        Some(SubObjectRef::Face(i)) => i as usize,
        _ => return None,
    };
    let n_tri = indices.len() / 3;
    if n_tri == 0 {
        return None;
    }
    // parry3d may return a backface index offset by n_tri; normalise.
    let face = if face_raw >= n_tri {
        face_raw - n_tri
    } else {
        face_raw
    };
    if face * 3 + 2 >= indices.len() {
        return None;
    }
    let vi = [
        indices[face * 3] as usize,
        indices[face * 3 + 1] as usize,
        indices[face * 3 + 2] as usize,
    ];
    let (best_vi, _) = vi
        .iter()
        .map(|&i| {
            let p = model.transform_point3(glam::Vec3::from(positions[i]));
            (i, p.distance(hit.world_pos))
        })
        .fold(
            (vi[0], f32::MAX),
            |acc, (i, d)| {
                if d < acc.1 { (i, d) } else { acc }
            },
        );
    Some(SubObjectRef::Vertex(best_vi as u32))
}

// ---------------------------------------------------------------------------
// pick_gaussian_splat_cpu
// ---------------------------------------------------------------------------

/// Screen-space nearest-splat pick for a Gaussian splat object.
///
/// Projects every splat position through `view_proj` and returns the closest
/// one whose screen-space distance to `click_pos` is within `radius_px`
/// pixels. Returns `None` when no splat qualifies.
///
/// The returned hit carries [`SubObjectRef::Point`] with the splat index.
///
/// # Arguments
/// * `click_pos`     - screen-space click in viewport pixels (top-left origin)
/// * `id`            - object identifier embedded in the returned [`PickHit`]
/// * `positions`     - local-space splat center positions
/// * `model`         - world transform for the splat object
/// * `view_proj`     - combined view x projection matrix
/// * `viewport_size` - viewport width x height in pixels
/// * `radius_px`     - maximum screen-space distance in pixels to accept as a hit
pub fn pick_gaussian_splat_cpu(
    click_pos: glam::Vec2,
    id: u64,
    positions: &[[f32; 3]],
    model: glam::Mat4,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
    radius_px: f32,
) -> Option<PickHit> {
    if id == 0 || positions.is_empty() {
        return None;
    }
    let mvp = view_proj * model;
    let mut best_dist_sq = radius_px * radius_px;
    let mut best_idx: Option<u32> = None;
    let mut best_world = glam::Vec3::ZERO;

    for (i, pos) in positions.iter().enumerate() {
        let local = glam::Vec3::from(*pos);
        let clip = mvp * local.extend(1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let sx = (clip.x / clip.w + 1.0) * 0.5 * viewport_size.x;
        let sy = (1.0 - clip.y / clip.w) * 0.5 * viewport_size.y;
        let dx = sx - click_pos.x;
        let dy = sy - click_pos.y;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = Some(i as u32);
            best_world = model.transform_point3(local);
        }
    }

    let idx = best_idx?;
    #[allow(deprecated)]
    Some(PickHit {
        id,
        sub_object: Some(SubObjectRef::Point(idx)),
        world_pos: best_world,
        normal: glam::Vec3::Y,
        triangle_index: u32::MAX,
        point_index: Some(idx),
        scalar_value: None,
    })
}

// ---------------------------------------------------------------------------
// pick_transparent_volume_mesh_cpu
// ---------------------------------------------------------------------------

/// Double-sided Moller-Trumbore ray-triangle intersection.
///
/// Returns the ray parameter `t > 0` on hit, `None` otherwise.
fn ray_tri_mt_ds(
    orig: glam::Vec3,
    dir: glam::Vec3,
    v0: glam::Vec3,
    v1: glam::Vec3,
    v2: glam::Vec3,
) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-8 {
        return None;
    }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    if t > 1e-6 { Some(t) } else { None }
}

// Triangular face indices for each cell type used in ray picking.
// (These cover the outer boundary surface of each cell.)

// Tet: 4 triangular faces.
const VM_TET_FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

// Hex: 6 quad faces, each split into 2 triangles (12 total).
const VM_HEX_TRIS: [[usize; 3]; 12] = [
    [0, 1, 2],
    [0, 2, 3], // bottom [0,1,2,3]
    [4, 7, 6],
    [4, 6, 5], // top    [4,7,6,5]
    [0, 4, 5],
    [0, 5, 1], // front  [0,4,5,1]
    [2, 6, 7],
    [2, 7, 3], // back   [2,6,7,3]
    [0, 3, 7],
    [0, 7, 4], // left   [0,3,7,4]
    [1, 5, 6],
    [1, 6, 2], // right  [1,5,6,2]
];

// Pyramid: quad base split into 2 + 4 triangular sides = 6 triangles.
const VM_PYRAMID_TRIS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 3], // base quad [0,1,2,3]
    [0, 4, 1],
    [1, 4, 2],
    [2, 4, 3],
    [3, 4, 0], // sides
];

// Wedge: 2 tri ends + 3 quad sides (each split) = 2 + 6 = 8 triangles.
const VM_WEDGE_TRIS: [[usize; 3]; 8] = [
    [0, 2, 1],
    [3, 4, 5], // tri ends
    [0, 1, 4],
    [0, 4, 3], // side [0,1,4,3]
    [1, 2, 5],
    [1, 5, 4], // side [1,2,5,4]
    [2, 0, 3],
    [2, 3, 5], // side [2,0,3,5]
];

/// Ray-cast pick against a transparent volume mesh.
///
/// Tests the ray against each cell in `data` using the cell's outer boundary
/// triangles. Returns the frontmost hit cell as a [`SubObjectRef::Cell`].
///
/// The intersection test runs in local space (inverse-transformed ray), so
/// `model` may include translation and uniform scale without loss of accuracy.
///
/// # Arguments
/// * `ray_origin` - world-space ray origin
/// * `ray_dir`    - world-space ray direction (need not be normalized)
/// * `id`         - object identifier embedded in the returned [`PickHit`]
/// * `model`      - world transform for the volume mesh
/// * `data`       - CPU-side volume mesh
pub fn pick_transparent_volume_mesh_cpu(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    id: u64,
    model: glam::Mat4,
    data: &VolumeMeshData,
) -> Option<PickHit> {
    if id == 0 || data.cells.is_empty() {
        return None;
    }
    let model_inv = model.inverse();
    let local_origin = model_inv.transform_point3(ray_origin);
    let local_dir = model_inv.transform_vector3(ray_dir);

    let mut best_t = f32::MAX;
    let mut best_cell: Option<u32> = None;

    for (cell_idx, cell) in data.cells.iter().enumerate() {
        let p = |i: usize| glam::Vec3::from(data.positions[cell[i] as usize]);
        let tris: &[[usize; 3]] = if cell[4] == CELL_SENTINEL {
            &VM_TET_FACES
        } else if cell[5] == CELL_SENTINEL {
            &VM_PYRAMID_TRIS
        } else if cell[6] == CELL_SENTINEL {
            &VM_WEDGE_TRIS
        } else {
            &VM_HEX_TRIS
        };
        for tri in tris {
            if let Some(t) = ray_tri_mt_ds(local_origin, local_dir, p(tri[0]), p(tri[1]), p(tri[2]))
            {
                if t < best_t {
                    best_t = t;
                    best_cell = Some(cell_idx as u32);
                }
            }
        }
    }

    let cell_idx = best_cell?;
    let local_hit = local_origin + local_dir * best_t;
    let world_hit = model.transform_point3(local_hit);
    #[allow(deprecated)]
    Some(PickHit {
        id,
        sub_object: Some(SubObjectRef::Cell(cell_idx)),
        world_pos: world_hit,
        normal: -ray_dir.normalize(),
        triangle_index: u32::MAX,
        point_index: None,
        scalar_value: None,
    })
}

// ---------------------------------------------------------------------------
// pick_volume_rect
// ---------------------------------------------------------------------------

/// Rect-select above-threshold voxels from a volume object.
///
/// Projects each voxel center through `view_proj` and collects those that fall
/// inside the selection rectangle and have a scalar value within
/// `[item.threshold_min, item.threshold_max]`.
///
/// Returns a [`RectPickResult`] with [`SubObjectRef::Voxel`] entries keyed by `id`.
///
/// # Arguments
/// * `rect_min/max`  - selection rectangle corners in viewport pixels (top-left origin)
/// * `id`            - object identifier used as the key in the result
/// * `item`          - volume render item (provides model, bbox, thresholds)
/// * `volume`        - CPU-side volume data (scalar field and grid dimensions)
/// * `view_proj`     - combined view x projection matrix
/// * `viewport_size` - viewport width x height in pixels
pub fn pick_volume_rect(
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
    id: u64,
    item: &crate::renderer::VolumeItem,
    volume: &VolumeData,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> RectPickResult {
    let mut result = RectPickResult::default();
    if id == 0 {
        return result;
    }
    let model = glam::Mat4::from_cols_array_2d(&item.model);
    let bbox_min = glam::Vec3::from(item.bbox_min);
    let bbox_max = glam::Vec3::from(item.bbox_max);
    let [nx, ny, nz] = volume.dims;
    let cell = (bbox_max - bbox_min) / glam::Vec3::new(nx as f32, ny as f32, nz as f32);
    let mvp = view_proj * model;

    let mut hits: Vec<SubObjectRef> = Vec::new();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = ix + iy * nx + iz * nx * ny;
                let scalar = volume.data[flat as usize];
                if scalar.is_nan() || scalar < item.threshold_min || scalar > item.threshold_max {
                    continue;
                }
                let local_center = bbox_min
                    + cell * glam::Vec3::new(ix as f32 + 0.5, iy as f32 + 0.5, iz as f32 + 0.5);
                let clip = mvp * local_center.extend(1.0);
                if clip.w <= 0.0 {
                    continue;
                }
                let sx = (clip.x / clip.w + 1.0) * 0.5 * viewport_size.x;
                let sy = (1.0 - clip.y / clip.w) * 0.5 * viewport_size.y;
                if sx >= rect_min.x && sx <= rect_max.x && sy >= rect_min.y && sy <= rect_max.y {
                    hits.push(SubObjectRef::Voxel(flat));
                }
            }
        }
    }
    if !hits.is_empty() {
        result.hits.insert(id, hits);
    }
    result
}

// ---------------------------------------------------------------------------
// pick_transparent_volume_mesh_rect
// ---------------------------------------------------------------------------

/// Rect-select cells from a transparent volume mesh.
///
/// Projects each cell centroid through `view_proj` and collects those inside
/// the selection rectangle.
///
/// Returns a [`RectPickResult`] with [`SubObjectRef::Cell`] entries keyed by `id`.
///
/// # Arguments
/// * `rect_min/max`  - selection rectangle in viewport pixels (top-left origin)
/// * `id`            - object identifier used as the key in the result
/// * `model`         - world transform for the volume mesh
/// * `data`          - CPU-side volume mesh
/// * `view_proj`     - combined view x projection matrix
/// * `viewport_size` - viewport width x height in pixels
pub fn pick_transparent_volume_mesh_rect(
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
    id: u64,
    model: glam::Mat4,
    data: &VolumeMeshData,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> RectPickResult {
    let mut result = RectPickResult::default();
    if id == 0 || data.cells.is_empty() {
        return result;
    }
    let mvp = view_proj * model;
    let mut hits: Vec<SubObjectRef> = Vec::new();

    for (cell_idx, cell) in data.cells.iter().enumerate() {
        let nv: usize = if cell[4] == CELL_SENTINEL {
            4
        } else if cell[5] == CELL_SENTINEL {
            5
        } else if cell[6] == CELL_SENTINEL {
            6
        } else {
            8
        };
        let centroid: glam::Vec3 = cell[..nv]
            .iter()
            .map(|&vi| glam::Vec3::from(data.positions[vi as usize]))
            .sum::<glam::Vec3>()
            / nv as f32;
        let clip = mvp * centroid.extend(1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let sx = (clip.x / clip.w + 1.0) * 0.5 * viewport_size.x;
        let sy = (1.0 - clip.y / clip.w) * 0.5 * viewport_size.y;
        if sx >= rect_min.x && sx <= rect_max.x && sy >= rect_min.y && sy <= rect_max.y {
            hits.push(SubObjectRef::Cell(cell_idx as u32));
        }
    }
    if !hits.is_empty() {
        result.hits.insert(id, hits);
    }
    result
}

// ---------------------------------------------------------------------------
// pick_gaussian_splat_rect
// ---------------------------------------------------------------------------

/// Rect-select splats from a Gaussian splat object.
///
/// Projects each splat position through `view_proj` and collects those inside
/// the selection rectangle.
///
/// Returns a [`RectPickResult`] with [`SubObjectRef::Point`] entries keyed by `id`.
///
/// # Arguments
/// * `rect_min/max`  - selection rectangle in viewport pixels (top-left origin)
/// * `id`            - object identifier used as the key in the result
/// * `positions`     - local-space splat center positions
/// * `model`         - world transform for the splat object
/// * `view_proj`     - combined view x projection matrix
/// * `viewport_size` - viewport width x height in pixels
pub fn pick_gaussian_splat_rect(
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
    id: u64,
    positions: &[[f32; 3]],
    model: glam::Mat4,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> RectPickResult {
    let mut result = RectPickResult::default();
    if id == 0 || positions.is_empty() {
        return result;
    }
    let mvp = view_proj * model;
    let mut hits: Vec<SubObjectRef> = Vec::new();

    for (i, pos) in positions.iter().enumerate() {
        let local = glam::Vec3::from(*pos);
        let clip = mvp * local.extend(1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let sx = (clip.x / clip.w + 1.0) * 0.5 * viewport_size.x;
        let sy = (1.0 - clip.y / clip.w) * 0.5 * viewport_size.y;
        if sx >= rect_min.x && sx <= rect_max.x && sy >= rect_min.y && sy <= rect_max.y {
            hits.push(SubObjectRef::Point(i as u32));
        }
    }
    if !hits.is_empty() {
        result.hits.insert(id, hits);
    }
    result
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
        let result = pick_scene_cpu(
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
        let result = pick_scene_cpu(
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
        let result = pick_scene_cpu(
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

        let result = pick_scene_nodes_cpu(
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

        let result = pick_scene_nodes_cpu(
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

        let result = pick_scene_with_probe_cpu(
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

        // 12 triangles in a unit cube : assign each a scalar.
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

        let result = pick_scene_with_probe_cpu(
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

        // No probe bindings : scalar_value should remain None.
        let result = pick_scene_with_probe_cpu(
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
            mesh_id: crate::resources::mesh_store::MeshId(0),
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
            mesh_id: crate::resources::mesh_store::MeshId(0),
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
        assert!(
            hits.iter().all(|s| s.is_point()),
            "expected SubObjectRef::Point entries"
        );
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
            mesh_id: crate::resources::mesh_store::MeshId(0),
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

        r.hits.insert(
            1,
            vec![
                SubObjectRef::Face(0),
                SubObjectRef::Face(1),
                SubObjectRef::Face(2),
            ],
        );
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

    // ---------------------------------------------------------------------------
    // pick_volume_cpu / voxel_world_aabb tests
    // ---------------------------------------------------------------------------

    fn make_volume_item(
        bbox_min: [f32; 3],
        bbox_max: [f32; 3],
        threshold_min: f32,
        threshold_max: f32,
    ) -> crate::renderer::VolumeItem {
        crate::renderer::VolumeItem {
            bbox_min,
            bbox_max,
            threshold_min,
            threshold_max,
            ..crate::renderer::VolumeItem::default()
        }
    }

    fn make_volume_data(dims: [u32; 3], fill: f32) -> crate::geometry::marching_cubes::VolumeData {
        let n = (dims[0] * dims[1] * dims[2]) as usize;
        crate::geometry::marching_cubes::VolumeData {
            data: vec![fill; n],
            dims,
            origin: [0.0; 3],
            spacing: [1.0; 3],
        }
    }

    #[test]
    fn test_pick_volume_basic_hit() {
        // 3x3x3 volume, bbox [0,0,0]->[3,3,3], all scalars 0.8.
        // Ray from +y: hits the top-center voxel (ix=1, iy=2, iz=1).
        let item = make_volume_item([0.0; 3], [3.0, 3.0, 3.0], 0.5, 1.0);
        let volume = make_volume_data([3, 3, 3], 0.8);

        let hit = super::pick_volume_cpu(
            glam::Vec3::new(1.5, 10.0, 1.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            42,
            &item,
            &volume,
        );
        assert!(hit.is_some(), "expected a hit");
        let hit = hit.unwrap();

        assert_eq!(hit.id, 42);
        assert_eq!(hit.scalar_value, Some(0.8));

        // Decode the flat index.
        let flat = hit.sub_object.unwrap().index();
        let nx = 3u32;
        let ny = 3u32;
        let ix = flat % nx;
        let iy = (flat / nx) % ny;
        let iz = flat / (nx * ny);
        assert_eq!((ix, iy, iz), (1, 2, 1), "expected top-centre voxel");

        // Entry point should be on the top bbox face (y≈3).
        assert!(hit.world_pos.y > 2.9, "world_pos.y={}", hit.world_pos.y);

        // Normal should point upward (ray entered through the +y face).
        assert!(hit.normal.y > 0.9, "normal={:?}", hit.normal);
    }

    #[test]
    fn test_pick_volume_miss_aabb() {
        let item = make_volume_item([0.0; 3], [1.0; 3], 0.0, 1.0);
        let volume = make_volume_data([4, 4, 4], 0.5);

        // Ray displaced 10 units in x: should miss the unit-cube bbox entirely.
        let hit = super::pick_volume_cpu(
            glam::Vec3::new(10.0, 5.0, 0.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            1,
            &item,
            &volume,
        );
        assert!(hit.is_none(), "expected miss");
    }

    #[test]
    fn test_pick_volume_threshold_miss() {
        // All scalars (0.3) below threshold_min (0.5) -> no hit.
        let item = make_volume_item([0.0; 3], [1.0; 3], 0.5, 1.0);
        let volume = make_volume_data([4, 4, 4], 0.3);

        let hit = super::pick_volume_cpu(
            glam::Vec3::new(0.5, 5.0, 0.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            1,
            &item,
            &volume,
        );
        assert!(
            hit.is_none(),
            "expected no hit when all scalars below threshold"
        );
    }

    #[test]
    fn test_pick_volume_threshold_skip() {
        // 1x3x1 volume along y. Ray from +y enters iy=2 first.
        // iy=2: scalar 0.3 (below threshold) -> skipped.
        // iy=1: scalar 0.8 (within threshold) -> hit.
        // iy=0: not reached.
        let item = make_volume_item([0.0; 3], [1.0, 3.0, 1.0], 0.5, 1.0);
        let mut volume = make_volume_data([1, 3, 1], 0.0);
        // flat index = ix + iy*nx: nx=1, so flat = iy.
        volume.data[2] = 0.3;
        volume.data[1] = 0.8;
        volume.data[0] = 0.8;

        let hit = super::pick_volume_cpu(
            glam::Vec3::new(0.5, 10.0, 0.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            1,
            &item,
            &volume,
        );
        assert!(hit.is_some(), "expected a hit");
        let hit = hit.unwrap();
        let flat = hit.sub_object.unwrap().index();
        assert_eq!(flat, 1, "expected iy=1 (flat=1), got flat={flat}");
        assert_eq!(hit.scalar_value, Some(0.8));
    }

    #[test]
    fn test_pick_volume_nan_skip() {
        // 1x2x1 volume. iy=1 (top) is NaN; iy=0 (bottom) is 0.5.
        // Ray from +y skips NaN and hits the valid voxel.
        let item = make_volume_item([0.0; 3], [1.0, 2.0, 1.0], 0.0, 1.0);
        let mut volume = make_volume_data([1, 2, 1], 0.0);
        volume.data[1] = f32::NAN;
        volume.data[0] = 0.5;

        let hit = super::pick_volume_cpu(
            glam::Vec3::new(0.5, 10.0, 0.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            1,
            &item,
            &volume,
        );
        assert!(hit.is_some(), "expected hit after NaN skip");
        let hit = hit.unwrap();
        assert_eq!(hit.sub_object.unwrap().index(), 0, "expected iy=0 (flat=0)");
        assert_eq!(hit.scalar_value, Some(0.5));
    }

    #[test]
    fn test_pick_volume_dda_no_skip() {
        // 10x1x1 volume along x. First 9 voxels are below threshold;
        // voxel ix=9 is the only one in range. A ray with a tiny z-component
        // (nearly axis-aligned to x) must still reach voxel 9 without skipping.
        let item = make_volume_item([0.0; 3], [10.0, 1.0, 1.0], 0.5, 1.0);
        let mut volume = make_volume_data([10, 1, 1], 0.0);
        volume.data[9] = 0.8;

        let dir = glam::Vec3::new(1.0, 0.0, 0.001).normalize();
        let hit = super::pick_volume_cpu(glam::Vec3::new(-1.0, 0.5, 0.5), dir, 1, &item, &volume);
        assert!(
            hit.is_some(),
            "DDA must reach the last voxel without skipping"
        );
        let flat = hit.unwrap().sub_object.unwrap().index();
        assert_eq!(flat, 9, "expected ix=9 (flat=9), got flat={flat}");
    }

    #[test]
    fn test_voxel_world_aabb_identity() {
        // Identity model, 4x4x4 uniform bbox [0,0,0]->[4,4,4].
        let item = make_volume_item([0.0; 3], [4.0, 4.0, 4.0], 0.0, 1.0);
        let volume = make_volume_data([4, 4, 4], 0.0);

        // Voxel (0,0,0) = flat 0: occupies [0,0,0]->[1,1,1].
        let (lo, hi) = super::voxel_world_aabb(0, &volume, &item);
        assert!((lo - glam::Vec3::ZERO).length() < 1e-5, "lo={lo:?}");
        assert!((hi - glam::Vec3::ONE).length() < 1e-5, "hi={hi:?}");

        // Voxel (1,0,0) = flat 1: occupies [1,0,0]->[2,1,1].
        let (lo, hi) = super::voxel_world_aabb(1, &volume, &item);
        assert!((lo.x - 1.0).abs() < 1e-5 && (hi.x - 2.0).abs() < 1e-5);

        // Voxel (1,2,3) = flat 1 + 2*4 + 3*16 = 57: occupies [1,2,3]->[2,3,4].
        let (lo, hi) = super::voxel_world_aabb(57, &volume, &item);
        assert!(
            (lo - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5,
            "lo={lo:?}"
        );
        assert!(
            (hi - glam::Vec3::new(2.0, 3.0, 4.0)).length() < 1e-5,
            "hi={hi:?}"
        );
    }

    #[test]
    fn test_voxel_world_aabb_round_trip() {
        // Pick a voxel, then verify that world_pos from the hit lies inside
        // the AABB returned by voxel_world_aabb.
        let item = make_volume_item([0.0; 3], [3.0, 3.0, 3.0], 0.5, 1.0);
        let volume = make_volume_data([3, 3, 3], 0.8);

        let hit = super::pick_volume_cpu(
            glam::Vec3::new(1.5, 10.0, 1.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            1,
            &item,
            &volume,
        )
        .expect("expected a hit for round-trip test");

        let flat = hit.sub_object.unwrap().index();
        let (lo, hi) = super::voxel_world_aabb(flat, &volume, &item);

        let tol = 1e-3;
        assert!(
            hit.world_pos.x >= lo.x - tol && hit.world_pos.x <= hi.x + tol,
            "world_pos.x={} outside [{}, {}]",
            hit.world_pos.x,
            lo.x,
            hi.x
        );
        assert!(
            hit.world_pos.y >= lo.y - tol && hit.world_pos.y <= hi.y + tol,
            "world_pos.y={} outside [{}, {}]",
            hit.world_pos.y,
            lo.y,
            hi.y
        );
        assert!(
            hit.world_pos.z >= lo.z - tol && hit.world_pos.z <= hi.z + tol,
            "world_pos.z={} outside [{}, {}]",
            hit.world_pos.z,
            lo.z,
            hi.z
        );
    }
}
