use super::*;

// ---------------------------------------------------------------------------
// Strip index helpers (shared by polyline, tube, ribbon picking)
// ---------------------------------------------------------------------------

/// Map a global node index to its strip index by walking `strip_lengths`.
fn strip_for_node(node_idx: u32, strip_lengths: &[u32]) -> u32 {
    let mut offset = 0u32;
    for (i, &len) in strip_lengths.iter().enumerate() {
        offset += len;
        if node_idx < offset {
            return i as u32;
        }
    }
    strip_lengths.len().saturating_sub(1) as u32
}

/// Find the closest polyline segment to `click_pos` within `threshold_px` pixels.
///
/// Returns `(global_seg_idx, world_hit_pos)` on hit, `None` otherwise. Positions
/// are treated as world-space (polylines are always submitted without a model
/// transform). The hit position is the closest point on the segment in 3D,
/// interpolated at the same screen-space parameter `t` as the closest screen point.
fn pick_closest_polyline_segment(
    click_pos: glam::Vec2,
    viewport_size: glam::Vec2,
    view_proj: glam::Mat4,
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    threshold_px: f32,
) -> Option<(u32, glam::Vec3)> {
    let project = |p: [f32; 3]| -> Option<glam::Vec2> {
        let clip = view_proj * glam::Vec4::new(p[0], p[1], p[2], 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        Some(glam::Vec2::new(
            (clip.x / clip.w + 1.0) * 0.5 * viewport_size.x,
            (1.0 - clip.y / clip.w) * 0.5 * viewport_size.y,
        ))
    };

    let mut best_dist = threshold_px;
    let mut best: Option<(u32, glam::Vec3)> = None;

    macro_rules! try_seg {
        ($ai:expr, $bi:expr, $seg:expr) => {{
            if let (Some(sa), Some(sb)) = (project(positions[$ai]), project(positions[$bi])) {
                let ab = sb - sa;
                let len_sq = ab.length_squared();
                let t = if len_sq < 1e-6 {
                    0.0f32
                } else {
                    ((click_pos - sa).dot(ab) / len_sq).clamp(0.0, 1.0)
                };
                let dist = (click_pos - (sa + ab * t)).length();
                if dist < best_dist {
                    best_dist = dist;
                    let wa = glam::Vec3::from(positions[$ai]);
                    let wb = glam::Vec3::from(positions[$bi]);
                    best = Some(($seg as u32, wa.lerp(wb, t)));
                }
            }
        }};
    }

    if strip_lengths.is_empty() {
        for j in 0..positions.len().saturating_sub(1) {
            try_seg!(j, j + 1, j);
        }
    } else {
        let mut node_off = 0usize;
        let mut seg_off = 0u32;
        for &slen in strip_lengths {
            let slen = slen as usize;
            for j in 0..slen.saturating_sub(1) {
                try_seg!(node_off + j, node_off + j + 1, seg_off + j as u32);
            }
            seg_off += slen.saturating_sub(1) as u32;
            node_off += slen;
        }
    }

    best
}

/// Returns `true` if the 2D segment [a, b] touches or crosses the axis-aligned rect.
fn segment_in_rect(
    a: glam::Vec2,
    b: glam::Vec2,
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
) -> bool {
    // Quick AABB reject.
    if a.x.min(b.x) > rect_max.x
        || a.x.max(b.x) < rect_min.x
        || a.y.min(b.y) > rect_max.y
        || a.y.max(b.y) < rect_min.y
    {
        return false;
    }
    // Either endpoint inside?
    let in_r = |p: glam::Vec2| {
        p.x >= rect_min.x && p.x <= rect_max.x && p.y >= rect_min.y && p.y <= rect_max.y
    };
    if in_r(a) || in_r(b) {
        return true;
    }
    // Segment crosses one of the 4 edges (parametric intersection test).
    let crosses = |p0: glam::Vec2, p1: glam::Vec2, q0: glam::Vec2, q1: glam::Vec2| -> bool {
        let d = p1 - p0;
        let e = q1 - q0;
        let denom = d.x * e.y - d.y * e.x;
        if denom.abs() < 1e-10 {
            return false;
        }
        let diff = q0 - p0;
        let t = (diff.x * e.y - diff.y * e.x) / denom;
        let u = (diff.x * d.y - diff.y * d.x) / denom;
        t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0
    };
    let tl = rect_min;
    let tr = glam::Vec2::new(rect_max.x, rect_min.y);
    let bl = glam::Vec2::new(rect_min.x, rect_max.y);
    let br = rect_max;
    crosses(a, b, tl, tr) || crosses(a, b, tr, br) || crosses(a, b, br, bl) || crosses(a, b, bl, tl)
}

/// Map a global segment index to its strip index by walking `strip_lengths`.
fn strip_for_segment(seg_idx: u32, strip_lengths: &[u32]) -> u32 {
    let mut offset = 0u32;
    for (i, &len) in strip_lengths.iter().enumerate() {
        let segs = len.saturating_sub(1);
        offset += segs;
        if seg_idx < offset {
            return i as u32;
        }
    }
    strip_lengths.len().saturating_sub(1) as u32
}

/// Build a flat list of segment midpoints ordered by global segment index.
///
/// Shared by polyline, streamtube, tube, and ribbon click picking.
/// The index into the returned slice is the global segment index for that item.
fn build_segment_midpoints(positions: &[[f32; 3]], strip_lengths: &[u32]) -> Vec<[f32; 3]> {
    let mut midpoints = Vec::new();
    if strip_lengths.is_empty() {
        for j in 0..positions.len().saturating_sub(1) {
            let a = glam::Vec3::from(positions[j]);
            let b = glam::Vec3::from(positions[j + 1]);
            midpoints.push(((a + b) * 0.5).to_array());
        }
    } else {
        let mut node_off = 0usize;
        for &slen in strip_lengths {
            let slen = slen as usize;
            for j in 0..slen.saturating_sub(1) {
                let a = glam::Vec3::from(positions[node_off + j]);
                let b = glam::Vec3::from(positions[node_off + j + 1]);
                midpoints.push(((a + b) * 0.5).to_array());
            }
            node_off += slen;
        }
    }
    midpoints
}

// ---------------------------------------------------------------------------
// PickRectResult
// ---------------------------------------------------------------------------

/// Result of a [`ViewportRenderer::pick_rect`] call.
#[derive(Clone, Debug, Default)]
pub struct PickRectResult {
    /// IDs of whole items that have geometry inside the pick rect.
    ///
    /// Populated when [`crate::interaction::pick_mask::PickMask::OBJECT`] is set.
    pub objects: Vec<u64>,
    /// Sub-elements inside the pick rect as `(item_id, sub_object)` pairs.
    ///
    /// Populated when any sub-element bit is set in the mask. All entries
    /// belong to the same geometric dimension when the mask is
    /// dimension-homogeneous (the common case).
    pub elements: Vec<(u64, crate::interaction::sub_object::SubObjectRef)>,
}

impl PickRectResult {
    /// Returns `true` when no objects or elements were found.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.elements.is_empty()
    }
}

impl ViewportRenderer {
    // -----------------------------------------------------------------------
    // Unified CPU pick : renderer.pick()
    // -----------------------------------------------------------------------

    /// Pick the nearest item or sub-element under `click_pos`.
    ///
    /// Dispatches across all item types retained from the last `prepare()` call.
    /// The `mask` controls which item types and sub-element levels participate.
    ///
    /// Returns `None` if nothing matching the mask is under the cursor.
    ///
    /// # Arguments
    /// * `click_pos`     - cursor position in viewport pixels (top-left origin)
    /// * `viewport_size` - viewport width x height in pixels
    /// * `view_proj`     - combined view x projection matrix from the last frame
    /// * `mask`          - which item types and sub-element levels to include
    ///
    /// # Example
    /// ```rust,ignore
    /// if let Some(hit) = renderer.pick(cursor, vp_size, view_proj, PickMask::FACE) {
    ///     println!("hit face {:?} on object {}", hit.sub_object, hit.id);
    /// }
    /// ```
    pub fn pick(
        &self,
        click_pos: glam::Vec2,
        viewport_size: glam::Vec2,
        view_proj: glam::Mat4,
        mask: crate::interaction::pick_mask::PickMask,
    ) -> Option<crate::interaction::picking::PickHit> {
        use crate::interaction::pick_mask::PickMask;
        use crate::interaction::picking::{
            screen_to_ray, pick_point_cloud_cpu, pick_gaussian_splat_cpu, pick_volume_cpu,
            pick_transparent_volume_mesh_cpu, PickHit,
        };
        use crate::interaction::sub_object::SubObjectRef;
        use parry3d::math::{Pose, Vector};
        use parry3d::query::{Ray, RayCast};

        if viewport_size.x <= 0.0 || viewport_size.y <= 0.0 {
            return None;
        }

        let view_proj_inv = view_proj.inverse();
        let (ray_origin, ray_dir) = screen_to_ray(click_pos, viewport_size, view_proj_inv);

        let wants_face   = mask.intersects(PickMask::FACE);
        let wants_vertex = mask.intersects(PickMask::VERTEX);
        let wants_cell   = mask.intersects(PickMask::CELL);
        let wants_cloud  = mask.intersects(PickMask::CLOUD_POINT);
        let wants_splat  = mask.intersects(PickMask::SPLAT);
        let wants_object = mask.intersects(PickMask::OBJECT);
        let wants_mesh_sub = wants_face || wants_vertex || mask.intersects(PickMask::EDGE);

        // (toi, hit) -- nearest hit so far across all types.
        let mut best: Option<(f32, PickHit)> = None;

        let mut consider = |toi: f32, hit: PickHit| {
            if best.as_ref().map_or(true, |(bt, _)| toi < *bt) {
                best = Some((toi, hit));
            }
        };

        // Build lookup for opaque volume mesh face_to_cell maps (used in section 1
        // to convert surface Face hits to Cell hits).
        let vm_cell_map: std::collections::HashMap<u64, &[u32]> = self
            .pick_volume_mesh_items
            .iter()
            .filter(|item| item.pick_id != PickId::NONE && !item.face_to_cell.is_empty())
            .map(|item| (item.pick_id.0, item.face_to_cell.as_slice()))
            .collect();

        // 1. Surface mesh picks (FACE, VERTEX, EDGE, CELL, or OBJECT fallback).
        if wants_mesh_sub || wants_cell || wants_object {
            let ray = Ray::new(
                Vector::new(ray_origin.x, ray_origin.y, ray_origin.z),
                Vector::new(ray_dir.x, ray_dir.y, ray_dir.z),
            );
            for item in &self.pick_scene_items {
                if !item.visible || item.pick_id == PickId::NONE {
                    continue;
                }
                let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                    continue;
                };
                let (Some(positions), Some(indices)) = (&mesh.cpu_positions, &mesh.cpu_indices)
                else {
                    continue;
                };

                let model = glam::Mat4::from_cols_array_2d(&item.model);

                // Bake the full model matrix into vertex positions so that
                // non-uniform scale is handled correctly.
                let verts: Vec<Vector> = positions
                    .iter()
                    .map(|p| {
                        let wp = model.transform_point3(glam::Vec3::from(*p));
                        Vector::new(wp.x, wp.y, wp.z)
                    })
                    .collect();

                let tri_indices: Vec<[u32; 3]> = indices
                    .chunks(3)
                    .filter(|c| c.len() == 3)
                    .map(|c| [c[0], c[1], c[2]])
                    .collect();

                if tri_indices.is_empty() {
                    continue;
                }

                match parry3d::shape::TriMesh::new(verts, tri_indices) {
                    Ok(trimesh) => {
                        // Vertices are already in world space: use identity pose.
                        let identity = Pose::identity();
                        let Some(intersection) =
                            trimesh.cast_ray_and_get_normal(&identity, &ray, f32::MAX, true)
                        else {
                            continue;
                        };
                        let toi = intersection.time_of_impact;
                        let world_pos = ray_origin + ray_dir * toi;
                        let normal = intersection.normal;

                        let feature_sub = SubObjectRef::from_feature_id(intersection.feature);

                        let sub_object = if wants_face {
                            feature_sub
                        } else if wants_cell {
                            // Convert surface Face hit to originating cell index.
                            if let Some(f2c) = vm_cell_map.get(&item.pick_id.0) {
                                match feature_sub {
                                    Some(SubObjectRef::Face(face_raw)) => {
                                        let n_tri = indices.len() / 3;
                                        let face = if (face_raw as usize) >= n_tri {
                                            face_raw as usize - n_tri
                                        } else {
                                            face_raw as usize
                                        };
                                        f2c.get(face).map(|&ci| SubObjectRef::Cell(ci))
                                    }
                                    other => other,
                                }
                            } else {
                                // No cell map for this item; fall through to object-only.
                                None
                            }
                        } else if wants_vertex {
                            // Convert face hit to nearest triangle corner.
                            match feature_sub {
                                Some(SubObjectRef::Face(face_raw)) => {
                                    let n_tri = indices.len() / 3;
                                    let face = if (face_raw as usize) >= n_tri {
                                        face_raw as usize - n_tri
                                    } else {
                                        face_raw as usize
                                    };
                                    if face * 3 + 2 < indices.len() {
                                        let vis = [
                                            indices[face * 3] as usize,
                                            indices[face * 3 + 1] as usize,
                                            indices[face * 3 + 2] as usize,
                                        ];
                                        let (best_vi, _) = vis
                                            .iter()
                                            .map(|&i| {
                                                let p = model.transform_point3(
                                                    glam::Vec3::from(positions[i]),
                                                );
                                                (i, p.distance(world_pos))
                                            })
                                            .fold((vis[0], f32::MAX), |acc, (i, d)| {
                                                if d < acc.1 { (i, d) } else { acc }
                                            });
                                        Some(SubObjectRef::Vertex(best_vi as u32))
                                    } else {
                                        None
                                    }
                                }
                                other => other,
                            }
                        } else {
                            // Object-only: no sub-element.
                            None
                        };

                        #[allow(deprecated)]
                        let hit = PickHit {
                            id: item.pick_id.0,
                            sub_object,
                            world_pos,
                            normal,
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                        };
                        consider(toi, hit);
                    }
                    Err(e) => {
                        tracing::warn!(
                            pick_id = item.pick_id.0,
                            error = %e,
                            "TriMesh build failed in renderer.pick()"
                        );
                    }
                }
            }
        }

        // 2. Opaque volume mesh cell picks are handled in section 1 above via
        // vm_cell_map (face_to_cell conversion on surface Face hits).

        // 2b. Transparent volume mesh cell picks (CELL or OBJECT fallback).
        if wants_cell || wants_object {
            for item in &self.pick_tvm_items {
                if item.pick_id == 0 {
                    continue;
                }
                let Some(data) = item.volume_mesh_data.as_deref() else {
                    continue;
                };
                if let Some(mut hit) = pick_transparent_volume_mesh_cpu(
                    ray_origin,
                    ray_dir,
                    item.pick_id,
                    glam::Mat4::IDENTITY,
                    data,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if !wants_cell {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }
        }

        // 3. Point cloud picks (CLOUD_POINT or OBJECT fallback).
        if wants_cloud || wants_object {
            for item in &self.pick_point_cloud_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let radius_px = item.point_size.max(4.0);
                if let Some(mut hit) = pick_point_cloud_cpu(
                    click_pos,
                    item.id,
                    item,
                    view_proj,
                    viewport_size,
                    radius_px,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if !wants_cloud {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }
        }

        // 4. Volume voxel picks (VOXEL or OBJECT fallback).
        let wants_voxel = mask.intersects(PickMask::VOXEL);
        if wants_voxel || wants_object {
            for item in &self.pick_volume_items {
                if item.pick_id == 0 {
                    continue;
                }
                let Some(vol_data) = item.volume_data.as_deref() else {
                    continue;
                };
                if let Some(mut hit) = pick_volume_cpu(ray_origin, ray_dir, item.pick_id, item, vol_data) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if !wants_voxel {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }
        }

        // 5. Gaussian splat picks (SPLAT or OBJECT fallback).
        if wants_splat || wants_object {
            for item in &self.pick_splat_items {
                if item.pick_id == 0 {
                    continue;
                }
                let Some(gpu_set) = self.resources.gaussian_splat_store.get(item.id.0) else {
                    continue;
                };
                if gpu_set.cpu_positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                // Derive pick radius from the mean per-splat scale so that a
                // click anywhere inside the visible disc registers as a hit.
                let mean_max_scale: f32 = if gpu_set.cpu_scales.is_empty() {
                    0.05
                } else {
                    gpu_set.cpu_scales.iter()
                        .map(|s| s[0].max(s[1]).max(s[2]))
                        .sum::<f32>()
                        / gpu_set.cpu_scales.len() as f32
                };
                let world_radius = mean_max_scale * 3.0;
                let center_w = model.transform_point3(glam::Vec3::ZERO);
                let p0_clip = view_proj * center_w.extend(1.0);
                let p1_clip = view_proj * (center_w + glam::Vec3::X * world_radius).extend(1.0);
                let radius_px = if p0_clip.w.abs() > 1e-6 && p1_clip.w.abs() > 1e-6 {
                    let p0_ndc = glam::Vec2::new(p0_clip.x, p0_clip.y) / p0_clip.w;
                    let p1_ndc = glam::Vec2::new(p1_clip.x, p1_clip.y) / p1_clip.w;
                    ((p1_ndc - p0_ndc).length() * 0.5 * viewport_size.x.max(viewport_size.y))
                        .max(4.0)
                } else {
                    world_radius * 100.0
                };
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos,
                    item.pick_id,
                    &gpu_set.cpu_positions,
                    model,
                    view_proj,
                    viewport_size,
                    radius_px,
                ) {
                    // pick_gaussian_splat_cpu returns SubObjectRef::Point; remap to Splat.
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if wants_splat {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Splat(idx));
                        }
                    } else {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }
        }

        // 6. Instance picks (INSTANCE or OBJECT fallback) for glyphs, tensor glyphs, sprites.
        let wants_instance = mask.intersects(PickMask::INSTANCE);
        if wants_instance || wants_object {
            // Convert a world-space radius at the model centroid to a pixel threshold.
            let instance_radius_px = |model: glam::Mat4, world_r: f32| -> f32 {
                let center = model.transform_point3(glam::Vec3::ZERO);
                let p0 = view_proj * center.extend(1.0);
                let p1 = view_proj * (center + glam::Vec3::X * world_r).extend(1.0);
                if p0.w.abs() > 1e-6 && p1.w.abs() > 1e-6 {
                    let n0 = glam::Vec2::new(p0.x, p0.y) / p0.w;
                    let n1 = glam::Vec2::new(p1.x, p1.y) / p1.w;
                    ((n1 - n0).length() * 0.5 * viewport_size.x.max(viewport_size.y)).max(4.0)
                } else {
                    (world_r * 100.0_f32).max(4.0)
                }
            };

            // Glyphs
            for item in &self.pick_glyph_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let world_r = if item.scale_by_magnitude && !item.vectors.is_empty() {
                    let mean_mag = item.vectors.iter()
                        .map(|v| glam::Vec3::from(*v).length())
                        .sum::<f32>() / item.vectors.len() as f32;
                    (mean_mag * item.scale).max(0.01)
                } else {
                    item.scale.max(0.01)
                };
                let radius_px = instance_radius_px(model, world_r);
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos, item.id, &item.positions, model, view_proj, viewport_size, radius_px,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if wants_instance {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Instance(idx));
                        }
                    } else {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }

            // Tensor glyphs
            for item in &self.pick_tensor_glyph_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let world_r = if !item.eigenvalues.is_empty() {
                    let mean_max = item.eigenvalues.iter()
                        .map(|ev| ev[0].abs().max(ev[1].abs()).max(ev[2].abs()))
                        .sum::<f32>() / item.eigenvalues.len() as f32;
                    (mean_max * item.scale).max(0.01)
                } else {
                    item.scale.max(0.01)
                };
                let radius_px = instance_radius_px(model, world_r);
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos, item.id, &item.positions, model, view_proj, viewport_size, radius_px,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if wants_instance {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Instance(idx));
                        }
                    } else {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }

            // Sprites
            for item in &self.pick_sprite_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let radius_px = match item.size_mode {
                    SpriteSizeMode::ScreenSpace => (item.default_size * 0.5).max(4.0),
                    SpriteSizeMode::WorldSpace => {
                        instance_radius_px(model, (item.default_size * 0.5).max(0.01))
                    }
                };
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos, item.id, &item.positions, model, view_proj, viewport_size, radius_px,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if wants_instance {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Instance(idx));
                        }
                    } else {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }
        }

        // 7. Polyline node picks (POLY_NODE, STRIP, or OBJECT fallback).
        let wants_poly_node = mask.intersects(PickMask::POLY_NODE);
        let wants_strip = mask.intersects(PickMask::STRIP);
        if wants_poly_node || wants_strip || wants_object {
            for item in &self.pick_polyline_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let radius_px = (item.line_width + 4.0).max(8.0);
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos,
                    item.id,
                    &item.positions,
                    glam::Mat4::IDENTITY,
                    view_proj,
                    viewport_size,
                    radius_px,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if wants_poly_node {
                        // sub_object is already SubObjectRef::Point(node_index)
                    } else if wants_strip {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Strip(
                                strip_for_node(idx, &item.strip_lengths),
                            ));
                        }
                    } else {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }
        }

        // 8. Polyline segment picks (SEGMENT, STRIP, or OBJECT fallback).
        // Uses screen-space distance from the click to the full segment line so
        // clicking anywhere along a segment registers, not just near the midpoint.
        let wants_segment = mask.intersects(PickMask::SEGMENT);
        if wants_segment || wants_strip || wants_object {
            for item in &self.pick_polyline_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                // Half the visual line width plus a few pixels of slack.
                let threshold_px = (item.line_width / 2.0 + 4.0).max(4.0);
                let Some((seg_idx, world_pos)) = pick_closest_polyline_segment(
                    click_pos,
                    viewport_size,
                    view_proj,
                    &item.positions,
                    &item.strip_lengths,
                    threshold_px,
                ) else {
                    continue;
                };
                let toi = (world_pos - ray_origin).dot(ray_dir).max(0.0);
                let sub_object = if wants_segment {
                    Some(SubObjectRef::Segment(seg_idx))
                } else if wants_strip {
                    Some(SubObjectRef::Strip(strip_for_segment(seg_idx, &item.strip_lengths)))
                } else {
                    None
                };
                #[allow(deprecated)]
                let hit = PickHit {
                    id: item.id,
                    sub_object,
                    world_pos,
                    normal: glam::Vec3::Y,
                    triangle_index: u32::MAX,
                    point_index: None,
                    scalar_value: None,
                };
                consider(toi, hit);
            }
        }

        // 9. Streamtube / tube / ribbon segment picks (SEGMENT, STRIP, or OBJECT fallback).
        if wants_segment || wants_strip || wants_object {
            // Convert a world-space radius at a reference point to a screen-pixel threshold.
            let world_r_to_px = |ref_world: glam::Vec3, world_r: f32| -> f32 {
                let p0 = view_proj * ref_world.extend(1.0);
                let p1 = view_proj * (ref_world + glam::Vec3::X * world_r).extend(1.0);
                if p0.w.abs() > 1e-6 && p1.w.abs() > 1e-6 {
                    let n0 = glam::Vec2::new(p0.x, p0.y) / p0.w;
                    let n1 = glam::Vec2::new(p1.x, p1.y) / p1.w;
                    ((n1 - n0).length() * 0.5 * viewport_size.x.max(viewport_size.y)).max(4.0)
                } else {
                    (world_r * 100.0_f32).max(4.0)
                }
            };

            for item in &self.pick_streamtube_items {
                if item.id == 0 || item.positions.is_empty() { continue; }
                let midpoints = build_segment_midpoints(&item.positions, &item.strip_lengths);
                if midpoints.is_empty() { continue; }
                let radius_px = world_r_to_px(glam::Vec3::from(midpoints[0]), item.radius.max(0.01));
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos, item.id, &midpoints, glam::Mat4::IDENTITY, view_proj, viewport_size, radius_px,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if wants_segment {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Segment(idx));
                        }
                    } else if wants_strip {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Strip(strip_for_segment(idx, &item.strip_lengths)));
                        }
                    } else {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }

            for item in &self.pick_tube_items {
                if item.id == 0 || item.positions.is_empty() { continue; }
                let midpoints = build_segment_midpoints(&item.positions, &item.strip_lengths);
                if midpoints.is_empty() { continue; }
                let radius_px = world_r_to_px(glam::Vec3::from(midpoints[0]), item.radius.max(0.01));
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos, item.id, &midpoints, glam::Mat4::IDENTITY, view_proj, viewport_size, radius_px,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if wants_segment {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Segment(idx));
                        }
                    } else if wants_strip {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Strip(strip_for_segment(idx, &item.strip_lengths)));
                        }
                    } else {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }

            for item in &self.pick_ribbon_items {
                if item.id == 0 || item.positions.is_empty() { continue; }
                let midpoints = build_segment_midpoints(&item.positions, &item.strip_lengths);
                if midpoints.is_empty() { continue; }
                let radius_px = world_r_to_px(glam::Vec3::from(midpoints[0]), item.width.max(0.01));
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos, item.id, &midpoints, glam::Mat4::IDENTITY, view_proj, viewport_size, radius_px,
                ) {
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    if wants_segment {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Segment(idx));
                        }
                    } else if wants_strip {
                        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                            hit.sub_object = Some(SubObjectRef::Strip(strip_for_segment(idx, &item.strip_lengths)));
                        }
                    } else {
                        hit.sub_object = None;
                    }
                    consider(toi, hit);
                }
            }
        }

        best.map(|(_, hit)| hit)
    }

    // -----------------------------------------------------------------------
    // Unified CPU rect pick : renderer.pick_rect()
    // -----------------------------------------------------------------------

    /// Pick all items or sub-elements inside a screen-space rectangle.
    ///
    /// Dispatches across all item types retained from the last `prepare()` call.
    /// The `mask` controls which item types and sub-element levels participate.
    ///
    /// # Arguments
    /// * `rect_min`      - top-left corner of the selection rect in viewport pixels
    /// * `rect_max`      - bottom-right corner of the selection rect in viewport pixels
    /// * `viewport_size` - viewport width x height in pixels
    /// * `view_proj`     - combined view x projection matrix from the last frame
    /// * `mask`          - which item types and sub-element levels to include
    pub fn pick_rect(
        &self,
        rect_min: glam::Vec2,
        rect_max: glam::Vec2,
        viewport_size: glam::Vec2,
        view_proj: glam::Mat4,
        mask: crate::interaction::pick_mask::PickMask,
    ) -> PickRectResult {
        use crate::interaction::pick_mask::PickMask;
        use crate::interaction::sub_object::SubObjectRef;

        let mut result = PickRectResult::default();

        if viewport_size.x <= 0.0 || viewport_size.y <= 0.0 {
            return result;
        }

        let wants_face   = mask.intersects(PickMask::FACE);
        let wants_vertex = mask.intersects(PickMask::VERTEX);
        let wants_cell   = mask.intersects(PickMask::CELL);
        let wants_cloud  = mask.intersects(PickMask::CLOUD_POINT);
        let wants_splat  = mask.intersects(PickMask::SPLAT);
        let wants_object = mask.intersects(PickMask::OBJECT);

        // Build lookup for opaque volume mesh face_to_cell maps.
        let vm_cell_map: std::collections::HashMap<u64, &[u32]> = self
            .pick_volume_mesh_items
            .iter()
            .filter(|item| item.pick_id != PickId::NONE && !item.face_to_cell.is_empty())
            .map(|item| (item.pick_id.0, item.face_to_cell.as_slice()))
            .collect();

        // Project a local-space point through mvp and return screen coords,
        // or None if the point is behind the camera.
        let project = |mvp: glam::Mat4, local: glam::Vec3| -> Option<(f32, f32)> {
            let clip = mvp * local.extend(1.0);
            if clip.w <= 0.0 {
                return None;
            }
            let sx = (clip.x / clip.w + 1.0) * 0.5 * viewport_size.x;
            let sy = (1.0 - clip.y / clip.w) * 0.5 * viewport_size.y;
            Some((sx, sy))
        };

        let in_rect = |sx: f32, sy: f32| -> bool {
            sx >= rect_min.x && sx <= rect_max.x && sy >= rect_min.y && sy <= rect_max.y
        };

        // 1. Surface mesh picks (FACE, VERTEX, CELL, or OBJECT).
        if wants_face || wants_vertex || wants_cell || wants_object {
            for item in &self.pick_scene_items {
                if !item.visible || item.pick_id == PickId::NONE {
                    continue;
                }
                let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                    continue;
                };
                let (Some(positions), Some(indices)) =
                    (&mesh.cpu_positions, &mesh.cpu_indices)
                else {
                    continue;
                };

                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.pick_id.0;
                let mut item_hit = false;

                if wants_face {
                    for (tri_idx, chunk) in indices.chunks(3).enumerate() {
                        if chunk.len() < 3 {
                            continue;
                        }
                        let [i0, i1, i2] =
                            [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
                        if i0 >= positions.len()
                            || i1 >= positions.len()
                            || i2 >= positions.len()
                        {
                            continue;
                        }
                        let centroid = (glam::Vec3::from(positions[i0])
                            + glam::Vec3::from(positions[i1])
                            + glam::Vec3::from(positions[i2]))
                            / 3.0;
                        if let Some((sx, sy)) = project(mvp, centroid) {
                            if in_rect(sx, sy) {
                                result.elements.push((id, SubObjectRef::Face(tri_idx as u32)));
                                item_hit = true;
                            }
                        }
                    }
                } else if wants_cell {
                    // Convert boundary triangle hits to originating cell indices.
                    if let Some(f2c) = vm_cell_map.get(&id) {
                        let mut seen = std::collections::HashSet::new();
                        for (tri_idx, chunk) in indices.chunks(3).enumerate() {
                            if chunk.len() < 3 {
                                continue;
                            }
                            let [i0, i1, i2] =
                                [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
                            if i0 >= positions.len()
                                || i1 >= positions.len()
                                || i2 >= positions.len()
                            {
                                continue;
                            }
                            let centroid = (glam::Vec3::from(positions[i0])
                                + glam::Vec3::from(positions[i1])
                                + glam::Vec3::from(positions[i2]))
                                / 3.0;
                            if let Some((sx, sy)) = project(mvp, centroid) {
                                if in_rect(sx, sy) {
                                    if let Some(&ci) = f2c.get(tri_idx) {
                                        if seen.insert(ci) {
                                            result.elements.push((id, SubObjectRef::Cell(ci)));
                                        }
                                    }
                                    item_hit = true;
                                }
                            }
                        }
                    }
                } else if wants_vertex {
                    for (vi, pos) in positions.iter().enumerate() {
                        if let Some((sx, sy)) = project(mvp, glam::Vec3::from(*pos)) {
                            if in_rect(sx, sy) {
                                result.elements.push((id, SubObjectRef::Vertex(vi as u32)));
                                item_hit = true;
                            }
                        }
                    }
                } else {
                    // OBJECT only: mark as hit if any triangle centroid is in rect.
                    'tri_scan: for chunk in indices.chunks(3) {
                        if chunk.len() < 3 {
                            continue;
                        }
                        let [i0, i1, i2] =
                            [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
                        if i0 >= positions.len()
                            || i1 >= positions.len()
                            || i2 >= positions.len()
                        {
                            continue;
                        }
                        let centroid = (glam::Vec3::from(positions[i0])
                            + glam::Vec3::from(positions[i1])
                            + glam::Vec3::from(positions[i2]))
                            / 3.0;
                        if let Some((sx, sy)) = project(mvp, centroid) {
                            if in_rect(sx, sy) {
                                item_hit = true;
                                break 'tri_scan;
                            }
                        }
                    }
                }

                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }
        }

        // 2. Opaque volume mesh cell picks are handled in section 1 above via
        // vm_cell_map (face_to_cell conversion on boundary triangle hits).

        // 2b. Transparent volume mesh cell picks (CELL or OBJECT).
        if wants_cell || wants_object {
            for item in &self.pick_tvm_items {
                if item.pick_id == 0 {
                    continue;
                }
                let Some(data) = item.volume_mesh_data.as_deref() else {
                    continue;
                };
                use crate::resources::volume_mesh::CELL_SENTINEL;
                let id = item.pick_id;
                let mvp = view_proj; // TVM items are always in world space (no model transform)
                let mut item_hit = false;

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
                    if let Some((sx, sy)) = project(mvp, centroid) {
                        if in_rect(sx, sy) {
                            if wants_cell {
                                result.elements.push((id, SubObjectRef::Cell(cell_idx as u32)));
                            }
                            item_hit = true;
                        }
                    }
                }

                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }
        }

        // 3. Point cloud picks (CLOUD_POINT or OBJECT).
        if wants_cloud || wants_object {
            for item in &self.pick_point_cloud_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.id;
                let mut item_hit = false;

                for (pt_idx, pos) in item.positions.iter().enumerate() {
                    if let Some((sx, sy)) = project(mvp, glam::Vec3::from(*pos)) {
                        if in_rect(sx, sy) {
                            if wants_cloud {
                                result.elements.push((id, SubObjectRef::Point(pt_idx as u32)));
                            }
                            item_hit = true;
                        }
                    }
                }

                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }
        }

        // 4. Volume voxel picks (VOXEL or OBJECT).
        let wants_voxel = mask.intersects(PickMask::VOXEL);
        if wants_voxel || wants_object {
            for item in &self.pick_volume_items {
                if item.pick_id == 0 {
                    continue;
                }
                let Some(vol_data) = item.volume_data.as_deref() else {
                    continue;
                };
                let [nx, ny, nz] = vol_data.dims;
                if nx == 0 || ny == 0 || nz == 0 || vol_data.data.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let bbox_min = glam::Vec3::from(item.bbox_min);
                let bbox_max = glam::Vec3::from(item.bbox_max);
                let cell = (bbox_max - bbox_min)
                    / glam::Vec3::new(nx as f32, ny as f32, nz as f32);
                let id = item.pick_id;
                let mut item_hit = false;

                for iz in 0..nz {
                    for iy in 0..ny {
                        for ix in 0..nx {
                            let flat = (ix + iy * nx + iz * nx * ny) as usize;
                            let scalar = vol_data.data[flat];
                            if scalar.is_nan()
                                || scalar < item.threshold_min
                                || scalar > item.threshold_max
                            {
                                continue;
                            }
                            let center = bbox_min
                                + cell * glam::Vec3::new(
                                    ix as f32 + 0.5,
                                    iy as f32 + 0.5,
                                    iz as f32 + 0.5,
                                );
                            if let Some((sx, sy)) = project(mvp, center) {
                                if in_rect(sx, sy) {
                                    if wants_voxel {
                                        result.elements.push((id, SubObjectRef::Voxel(flat as u32)));
                                    }
                                    item_hit = true;
                                }
                            }
                        }
                    }
                }

                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }
        }

        // 5. Gaussian splat picks (SPLAT or OBJECT).
        if wants_splat || wants_object {
            for item in &self.pick_splat_items {
                if item.pick_id == 0 {
                    continue;
                }
                let Some(gpu_set) = self.resources.gaussian_splat_store.get(item.id.0) else {
                    continue;
                };
                if gpu_set.cpu_positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.pick_id;
                let mut item_hit = false;

                for (i, pos) in gpu_set.cpu_positions.iter().enumerate() {
                    if let Some((sx, sy)) = project(mvp, glam::Vec3::from(*pos)) {
                        if in_rect(sx, sy) {
                            if wants_splat {
                                result.elements.push((id, SubObjectRef::Splat(i as u32)));
                            }
                            item_hit = true;
                        }
                    }
                }

                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }
        }

        // 6. Instance picks (INSTANCE or OBJECT) for glyphs, tensor glyphs, sprites.
        let wants_instance = mask.intersects(PickMask::INSTANCE);
        if wants_instance || wants_object {
            // Glyphs
            for item in &self.pick_glyph_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.id;
                let mut item_hit = false;
                for (i, pos) in item.positions.iter().enumerate() {
                    if let Some((sx, sy)) = project(mvp, glam::Vec3::from(*pos)) {
                        if in_rect(sx, sy) {
                            if wants_instance {
                                result.elements.push((id, SubObjectRef::Instance(i as u32)));
                            }
                            item_hit = true;
                        }
                    }
                }
                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }

            // Tensor glyphs
            for item in &self.pick_tensor_glyph_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.id;
                let mut item_hit = false;
                for (i, pos) in item.positions.iter().enumerate() {
                    if let Some((sx, sy)) = project(mvp, glam::Vec3::from(*pos)) {
                        if in_rect(sx, sy) {
                            if wants_instance {
                                result.elements.push((id, SubObjectRef::Instance(i as u32)));
                            }
                            item_hit = true;
                        }
                    }
                }
                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }

            // Sprites
            for item in &self.pick_sprite_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.id;
                let mut item_hit = false;
                for (i, pos) in item.positions.iter().enumerate() {
                    if let Some((sx, sy)) = project(mvp, glam::Vec3::from(*pos)) {
                        if in_rect(sx, sy) {
                            if wants_instance {
                                result.elements.push((id, SubObjectRef::Instance(i as u32)));
                            }
                            item_hit = true;
                        }
                    }
                }
                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }
        }

        // 7. Polyline node / segment / strip / object rect picks.
        let wants_poly_node = mask.intersects(PickMask::POLY_NODE);
        let wants_segment   = mask.intersects(PickMask::SEGMENT);
        let wants_strip     = mask.intersects(PickMask::STRIP);
        if wants_poly_node || wants_segment || wants_strip || wants_object {
            for item in &self.pick_polyline_items {
                if item.id == 0 || item.positions.is_empty() {
                    continue;
                }
                let id = item.id;
                let mut item_hit = false;
                let mut strips_hit = std::collections::HashSet::<u32>::new();

                // Node pass (POLY_NODE or STRIP or OBJECT).
                if wants_poly_node || wants_strip || wants_object {
                    for (node_idx, pos) in item.positions.iter().enumerate() {
                        if let Some((sx, sy)) = project(view_proj, glam::Vec3::from(*pos)) {
                            if in_rect(sx, sy) {
                                item_hit = true;
                                if wants_poly_node {
                                    result.elements.push((id, SubObjectRef::Point(node_idx as u32)));
                                } else if wants_strip {
                                    let s = strip_for_node(node_idx as u32, &item.strip_lengths);
                                    strips_hit.insert(s);
                                }
                            }
                        }
                    }
                }

                // Segment pass (SEGMENT or STRIP or OBJECT) -- full segment/rect intersection.
                if wants_segment || (wants_strip && !wants_poly_node) || wants_object {
                    let mut node_off = 0usize;
                    let mut seg_off = 0u32;
                    macro_rules! try_seg_rect {
                        ($ai:expr, $bi:expr, $seg:expr) => {{
                            if let (Some((sax, say)), Some((sbx, sby))) = (
                                project(view_proj, glam::Vec3::from(item.positions[$ai])),
                                project(view_proj, glam::Vec3::from(item.positions[$bi])),
                            ) {
                                if segment_in_rect(
                                    glam::Vec2::new(sax, say),
                                    glam::Vec2::new(sbx, sby),
                                    rect_min, rect_max,
                                ) {
                                    item_hit = true;
                                    if wants_segment {
                                        result.elements.push((id, SubObjectRef::Segment($seg)));
                                    } else if wants_strip {
                                        let s = strip_for_segment($seg, &item.strip_lengths);
                                        strips_hit.insert(s);
                                    }
                                }
                            }
                        }};
                    }
                    if item.strip_lengths.is_empty() {
                        for j in 0..item.positions.len().saturating_sub(1) {
                            try_seg_rect!(j, j + 1, j as u32);
                        }
                    } else {
                        for &slen in &item.strip_lengths {
                            let slen = slen as usize;
                            for j in 0..slen.saturating_sub(1) {
                                try_seg_rect!(node_off + j, node_off + j + 1, seg_off + j as u32);
                            }
                            seg_off += slen.saturating_sub(1) as u32;
                            node_off += slen;
                        }
                    }
                }

                if wants_strip {
                    for s in strips_hit {
                        result.elements.push((id, SubObjectRef::Strip(s)));
                    }
                }
                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }
        }

        // 8. Streamtube / tube / ribbon segment / strip / object rect picks.
        if wants_segment || wants_strip || wants_object {
            let curve_iter = self.pick_streamtube_items.iter()
                .map(|it| (it.id, it.positions.as_slice(), it.strip_lengths.as_slice()))
                .chain(self.pick_tube_items.iter()
                    .map(|it| (it.id, it.positions.as_slice(), it.strip_lengths.as_slice())))
                .chain(self.pick_ribbon_items.iter()
                    .map(|it| (it.id, it.positions.as_slice(), it.strip_lengths.as_slice())));

            for (id, positions, strip_lengths) in curve_iter {
                if id == 0 || positions.is_empty() { continue; }
                let mut item_hit = false;
                let mut strips_hit = std::collections::HashSet::<u32>::new();
                // Build indexed midpoints (midpoint world pos, global segment index).
                let mut midpoints: Vec<([f32; 3], u32)> = Vec::new();
                if strip_lengths.is_empty() {
                    for j in 0..positions.len().saturating_sub(1) {
                        let a = glam::Vec3::from(positions[j]);
                        let b = glam::Vec3::from(positions[j + 1]);
                        midpoints.push((((a + b) * 0.5).to_array(), j as u32));
                    }
                } else {
                    let mut node_off = 0usize;
                    let mut seg_off = 0u32;
                    for &slen in strip_lengths {
                        let slen = slen as usize;
                        for j in 0..slen.saturating_sub(1) {
                            let a = glam::Vec3::from(positions[node_off + j]);
                            let b = glam::Vec3::from(positions[node_off + j + 1]);
                            midpoints.push((((a + b) * 0.5).to_array(), seg_off + j as u32));
                        }
                        seg_off += slen.saturating_sub(1) as u32;
                        node_off += slen;
                    }
                }
                for (mid, seg_idx) in &midpoints {
                    if let Some((sx, sy)) = project(view_proj, glam::Vec3::from(*mid)) {
                        if in_rect(sx, sy) {
                            item_hit = true;
                            if wants_segment {
                                result.elements.push((id, SubObjectRef::Segment(*seg_idx)));
                            } else if wants_strip {
                                let s = strip_for_segment(*seg_idx, strip_lengths);
                                strips_hit.insert(s);
                            }
                        }
                    }
                }
                if wants_strip {
                    for s in strips_hit {
                        result.elements.push((id, SubObjectRef::Strip(s)));
                    }
                }
                if wants_object && item_hit {
                    result.objects.push(id);
                }
            }
        }

        result
    }

    // -----------------------------------------------------------------------
    // Phase K : GPU object-ID picking
    // -----------------------------------------------------------------------

    /// GPU object-ID pick: renders the scene to an offscreen `R32Uint` texture
    /// and reads back the single pixel under `cursor`.
    ///
    /// This is O(1) in mesh complexity : every object is rendered with a flat
    /// `u32` ID, and only one pixel is read back. For triangle-level queries
    /// (barycentric scalar probe, exact world position), use the CPU
    /// [`crate::interaction::picking::pick_scene_cpu`] path instead.
    ///
    /// The pipeline is lazily initialized on first call : zero overhead when
    /// this method is never invoked.
    ///
    /// # Arguments
    /// * `device` : wgpu device
    /// * `queue` : wgpu queue
    /// * `cursor` : cursor position in viewport-local pixels (top-left origin)
    /// * `frame` : current grouped frame data (camera, scene surfaces, viewport size)
    ///
    /// # Returns
    /// `Some(GpuPickHit)` if an object is under the cursor, `None` if empty space.
    pub fn pick_scene_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cursor: glam::Vec2,
        frame: &FrameData,
    ) -> Option<crate::interaction::picking::GpuPickHit> {
        // In Playback mode, throttle picking to every 4th frame to reduce overhead
        // during animation. Interactive, Paused, and Capture modes always pick.
        if self.runtime_mode == crate::renderer::stats::RuntimeMode::Playback
            && self.frame_counter % 4 != 0
        {
            return None;
        }

        // Read scene items from the surface submission.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

        let ppp = frame.camera.pixels_per_point;
        let vp_w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let vp_h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        // --- bounds check (logical coordinates match the logical cursor) ---
        if cursor.x < 0.0
            || cursor.y < 0.0
            || cursor.x >= frame.camera.viewport_size[0]
            || cursor.y >= frame.camera.viewport_size[1]
            || vp_w == 0
            || vp_h == 0
        {
            return None;
        }

        // --- lazy pipeline init ---
        self.resources.ensure_pick_pipeline(device);

        // --- build PickInstance data ---
        // Only surfaces with a nonzero pick_id participate in picking.
        // Clear value 0 means "no hit" (or non-pickable surface).
        let pickable_items: Vec<&SceneRenderItem> = scene_items
            .iter()
            .filter(|item| item.visible && item.pick_id != PickId::NONE)
            .collect();

        let pick_instances: Vec<PickInstance> = pickable_items
            .iter()
            .map(|item| {
                let m = item.model;
                PickInstance {
                    model_c0: m[0],
                    model_c1: m[1],
                    model_c2: m[2],
                    model_c3: m[3],
                    object_id: item.pick_id.0 as u32,
                    _pad: [0; 3],
                }
            })
            .collect();

        if pick_instances.is_empty() {
            return None;
        }

        // --- pick instance storage buffer + bind group ---
        let pick_instance_bytes = bytemuck::cast_slice(&pick_instances);
        let pick_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_instance_buf"),
            size: pick_instance_bytes.len().max(80) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pick_instance_buf, 0, pick_instance_bytes);

        let pick_instance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_instance_bg"),
            layout: self
                .resources
                .pick_bind_group_layout_1
                .as_ref()
                .expect("ensure_pick_pipeline must be called first"),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pick_instance_buf.as_entire_binding(),
            }],
        });

        // --- pick camera uniform buffer + bind group ---
        let camera_uniform = frame.camera.render_camera.camera_uniform();
        let camera_bytes = bytemuck::bytes_of(&camera_uniform);
        let pick_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_camera_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pick_camera_buf, 0, camera_bytes);

        let pick_camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick_camera_bg"),
            layout: self
                .resources
                .pick_camera_bgl
                .as_ref()
                .expect("ensure_pick_pipeline must be called first"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pick_camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.resources.clip_volume_uniform_buf.as_entire_binding(),
                },
            ],
        });

        // --- offscreen pick textures (R32Uint + R32Float) + depth ---
        let pick_id_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_id_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_id_view = pick_id_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let pick_depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_depth_color_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_depth_view =
            pick_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_stencil_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick_ds_texture"),
            size: wgpu::Extent3d {
                width: vp_w,
                height: vp_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_stencil_view =
            depth_stencil_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- render pass ---
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pick_pass_encoder"),
        });
        {
            let mut pick_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pick_pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pick_id_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &pick_depth_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 1.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_stencil_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pick_pass.set_pipeline(
                self.resources
                    .pick_pipeline
                    .as_ref()
                    .expect("ensure_pick_pipeline must be called first"),
            );
            pick_pass.set_bind_group(0, &pick_camera_bg, &[]);
            pick_pass.set_bind_group(1, &pick_instance_bg, &[]);

            // Draw each pickable item with its instance slot.
            // Instance index in the storage buffer = position in pick_instances vec.
            for (instance_slot, item) in pickable_items.iter().enumerate() {
                let Some(mesh) = self
                    .resources
                    .mesh_store
                    .get(item.mesh_id)
                else {
                    continue;
                };
                pick_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pick_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                let slot = instance_slot as u32;
                pick_pass.draw_indexed(0..mesh.index_count, 0, slot..slot + 1);
            }
        }

        // --- copy 1×1 pixels to staging buffers ---
        // R32Uint: 4 bytes per pixel, min bytes_per_row = 256 (wgpu alignment)
        let bytes_per_row_aligned = 256u32; // wgpu requires multiples of 256

        let id_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_id_staging"),
            size: bytes_per_row_aligned as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let depth_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_depth_staging"),
            size: bytes_per_row_aligned as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Convert logical cursor to physical pixel coordinates for the pick texture readback.
        let px = (cursor.x * ppp).round() as u32;
        let py = (cursor.y * ppp).round() as u32;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &pick_id_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &id_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &pick_depth_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &depth_staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // --- map and read ---
        let (tx_id, rx_id) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        let (tx_dep, rx_dep) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        id_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_id.send(r);
            });
        depth_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx_dep.send(r);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        let _ = rx_id.recv().unwrap_or(Err(wgpu::BufferAsyncError));
        let _ = rx_dep.recv().unwrap_or(Err(wgpu::BufferAsyncError));

        let object_id = {
            let data = id_staging.slice(..).get_mapped_range();
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        id_staging.unmap();

        let depth = {
            let data = depth_staging.slice(..).get_mapped_range();
            f32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };
        depth_staging.unmap();

        // 0 = miss (clear color or non-pickable surface).
        if object_id == 0 {
            return None;
        }

        Some(crate::interaction::picking::GpuPickHit {
            object_id: PickId(object_id as u64),
            depth,
        })
    }
}
