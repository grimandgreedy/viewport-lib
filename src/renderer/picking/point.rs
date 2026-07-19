//! CPU ray-cast pick: find the nearest item or sub-element under the cursor.

use super::*;

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
        mask: PickMask,
    ) -> Option<PickHit> {
        use crate::interaction::query::picking::{
            pick_gaussian_splat_cpu, pick_point_cloud_cpu, pick_transparent_volume_mesh_cpu,
            pick_volume_cpu, screen_to_ray,
        };
        use parry3d::math::{Pose, Vector};
        use parry3d::query::{Ray, RayCast};

        if !self.cpu_pick_cache_enabled {
            warn_pick_cache_disabled();
            return None;
        }

        if viewport_size.x <= 0.0 || viewport_size.y <= 0.0 {
            return None;
        }

        let view_proj_inv = view_proj.inverse();
        let (ray_origin, ray_dir) = screen_to_ray(click_pos, viewport_size, view_proj_inv);

        let wants_face = mask.intersects(PickMask::FACE);
        let wants_vertex = mask.intersects(PickMask::VERTEX);
        let wants_cell = mask.intersects(PickMask::CELL);
        let wants_cloud = mask.intersects(PickMask::CLOUD_POINT);
        let wants_splat = mask.intersects(PickMask::SPLAT);
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
            .filter(|item| item.settings.pick_id != PickId::NONE && !item.face_to_cell.is_empty())
            .map(|item| (item.settings.pick_id.0, item.face_to_cell.as_slice()))
            .collect();

        // 1. Surface mesh picks (FACE, VERTEX, EDGE, CELL, or OBJECT fallback).
        if wants_mesh_sub || wants_cell || wants_object {
            for item in &self.pick_scene_items {
                if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                    continue;
                };
                let (Some(positions), Some(indices)) = (&mesh.cpu_positions, &mesh.cpu_indices)
                else {
                    continue;
                };
                let Some(trimesh) = mesh.cached_pick_trimesh() else {
                    continue;
                };

                let model = glam::Mat4::from_cols_array_2d(&item.model);
                // Cast the ray in the mesh's local space instead of baking the
                // model matrix into a fresh world-space vertex Vec every click:
                // `trimesh` is cached per mesh_id (see `cached_pick_trimesh`), so
                // this serves every instance of a shared mesh. `transform_vector3`
                // (not `transform_point3`) applies only the linear part, so
                // `local_dir` is not renormalized and `toi` comes out identical to
                // the world-space parametrization.
                let inv_model = model.inverse();
                let local_origin = inv_model.transform_point3(ray_origin);
                let local_dir = inv_model.transform_vector3(ray_dir);
                let ray = Ray::new(
                    Vector::new(local_origin.x, local_origin.y, local_origin.z),
                    Vector::new(local_dir.x, local_dir.y, local_dir.z),
                );

                {
                    // Vertices are in mesh-local space: use identity pose.
                    let identity = Pose::identity();
                    let Some(intersection) =
                        trimesh.cast_ray_and_get_normal(&identity, &ray, f32::MAX, true)
                    else {
                        continue;
                    };
                    let toi = intersection.time_of_impact;
                    let world_pos = ray_origin + ray_dir * toi;
                    // Transform the local-space normal back to world space with
                    // the inverse-transpose of the model's linear part, so
                    // non-uniform scale does not distort it.
                    let normal_matrix = glam::Mat3::from_mat4(model).inverse().transpose();
                    let local_normal = glam::Vec3::new(
                        intersection.normal.x,
                        intersection.normal.y,
                        intersection.normal.z,
                    );
                    let normal = normal_matrix.mul_vec3(local_normal).normalize();

                    let feature_sub = SubObjectRef::from_feature_id(intersection.feature);

                    let sub_object = if wants_face {
                        feature_sub
                    } else if wants_cell {
                        // Convert surface Face hit to originating cell index.
                        if let Some(f2c) = vm_cell_map.get(&item.settings.pick_id.0) {
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
                        } else if wants_vertex {
                            // No cell map for this item; try vertex picking instead.
                            // Fall through to the vertex branch below by
                            // re-evaluating with the vertex logic inline.
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
                                                let p = model.transform_point3(glam::Vec3::from(
                                                    positions[i],
                                                ));
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
                            // No cell map and vertex not wanted; no sub-element.
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
                                            let p = model
                                                .transform_point3(glam::Vec3::from(positions[i]));
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

                    // Only emit the hit if we produced a meaningful sub-element
                    // or the caller explicitly asked for object-level hits.
                    // Without this guard, an EDGE-only mask runs the ray-trimesh
                    // intersection (because wants_mesh_sub is true) but falls through
                    // to sub_object=None, producing a spurious object-level hit.
                    if sub_object.is_some() || wants_object {
                        #[allow(deprecated)]
                        let hit = PickHit {
                            id: item.settings.pick_id.0,
                            sub_object,
                            world_pos,
                            normal,
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        };
                        consider(toi, hit);
                    }
                }
            }
        }

        // 2. Opaque volume mesh cell picks are handled in section 1 above via
        // vm_cell_map (face_to_cell conversion on surface Face hits).

        // 2c. Scatter-volume object picks. Ray-vs-shape intersection only;
        // there is no sub-object level for participating media
        if wants_object {
            for item in &self.pick_scatter_volume_items {
                if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                    continue;
                }
                if let Some((t_enter, _)) = crate::scene::scatter_volume::ray_intersect(
                    &item.volume.shape,
                    ray_origin,
                    ray_dir,
                ) {
                    let world_pos = ray_origin + ray_dir * t_enter;
                    let normal = (world_pos
                        - match item.volume.shape {
                            crate::scene::scatter_volume::ScatterShape::Box(b) => {
                                (b.min + b.max) * 0.5
                            }
                            crate::scene::scatter_volume::ScatterShape::Sphere {
                                center, ..
                            } => glam::Vec3::from(center),
                        })
                    .try_normalize()
                    .unwrap_or(glam::Vec3::Z);
                    consider(
                        t_enter,
                        PickHit::object_hit(item.settings.pick_id.0, world_pos, normal),
                    );
                }
            }
        }

        // 2b. Interior-inclusive cell picks for volume meshes rendering
        //     transparently. Items rendering as opaque are handled in section 1
        //     above via vm_cell_map (face_to_cell on the boundary surface).
        if wants_cell || wants_object {
            for item in &self.pick_volume_mesh_items {
                if item.settings.pick_id == PickId::NONE || item.transparency.is_none() {
                    continue;
                }
                let Some(data) = item.volume_mesh_data.as_deref() else {
                    continue;
                };
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                if let Some(mut hit) = pick_transparent_volume_mesh_cpu(
                    ray_origin,
                    ray_dir,
                    item.settings.pick_id.0,
                    model,
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
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let radius_px = item.point_size.max(4.0);
                if let Some(mut hit) = pick_point_cloud_cpu(
                    click_pos,
                    item.settings.pick_id.0,
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
                if item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let Some(vol_data) = item.volume_data.as_deref() else {
                    continue;
                };
                if let Some(mut hit) =
                    pick_volume_cpu(ray_origin, ray_dir, item.settings.pick_id.0, item, vol_data)
                {
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
                if item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let Some(gpu_set) = self.resources.content.gaussian_splat_store.get(item.source)
                else {
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
                    gpu_set
                        .cpu_scales
                        .iter()
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
                    item.settings.pick_id.0,
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
            // Convert a world-space radius at a given world position to a pixel threshold.
            // Using the actual instance centroid rather than the model origin gives a correct
            // pixel size when instances are offset far from the model's local origin.
            let instance_radius_px = |world_center: glam::Vec3, world_r: f32| -> f32 {
                let p0 = view_proj * world_center.extend(1.0);
                let p1 = view_proj * (world_center + glam::Vec3::X * world_r).extend(1.0);
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
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let full_len = if item.scale_by_magnitude && !item.vectors.is_empty() {
                    let mean_mag = item
                        .vectors
                        .iter()
                        .map(|v| glam::Vec3::from(*v).length())
                        .sum::<f32>()
                        / item.vectors.len() as f32;
                    (mean_mag * item.scale).max(0.01)
                } else {
                    item.scale.max(0.01)
                };
                // Test against the midpoint of each arrow (base + half-vector) with
                // world_r = half-length. This prevents the hit circle from extending a full
                // arrow-length behind the base when the arrow points away from the camera.
                let has_vecs = item.vectors.len() == item.positions.len();
                let midpoints: Vec<[f32; 3]> = item
                    .positions
                    .iter()
                    .enumerate()
                    .map(|(i, pos)| {
                        if has_vecs {
                            let p = glam::Vec3::from(*pos);
                            let v = glam::Vec3::from(item.vectors[i]);
                            let len = if item.scale_by_magnitude {
                                v.length() * item.scale
                            } else {
                                item.scale
                            };
                            (p + v.normalize_or_zero() * len * 0.5).to_array()
                        } else {
                            *pos
                        }
                    })
                    .collect();
                let n = midpoints.len() as f32;
                let centroid = model.transform_point3(
                    midpoints
                        .iter()
                        .map(|p| glam::Vec3::from(*p))
                        .sum::<glam::Vec3>()
                        / n,
                );
                let radius_px = instance_radius_px(centroid, full_len * 0.5);
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos,
                    item.settings.pick_id.0,
                    &midpoints,
                    model,
                    view_proj,
                    viewport_size,
                    radius_px,
                ) {
                    // Report the base position, not the midpoint.
                    if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
                        if let Some(base) = item.positions.get(idx as usize) {
                            hit.world_pos = model.transform_point3(glam::Vec3::from(*base));
                        }
                        if wants_instance {
                            hit.sub_object = Some(SubObjectRef::Instance(idx));
                        } else {
                            hit.sub_object = None;
                        }
                    }
                    let toi = (hit.world_pos - ray_origin).dot(ray_dir).max(0.0);
                    consider(toi, hit);
                }
            }

            // Tensor glyphs
            for item in &self.pick_tensor_glyph_items {
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                // Use the max eigenvalue across all instances so the largest ellipsoid
                // is fully covered. Use the centroid of instance positions for an accurate
                // pixel-size estimate (instances may be far from the model origin).
                let world_r = if !item.eigenvalues.is_empty() {
                    let max_ev = item
                        .eigenvalues
                        .iter()
                        .map(|ev| ev[0].abs().max(ev[1].abs()).max(ev[2].abs()))
                        .fold(0.0_f32, f32::max);
                    (max_ev * item.scale).max(0.01)
                } else {
                    item.scale.max(0.01)
                };
                let n = item.positions.len() as f32;
                let centroid = model.transform_point3(
                    item.positions
                        .iter()
                        .map(|p| glam::Vec3::from(*p))
                        .sum::<glam::Vec3>()
                        / n,
                );
                let radius_px = instance_radius_px(centroid, world_r);
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos,
                    item.settings.pick_id.0,
                    &item.positions,
                    model,
                    view_proj,
                    viewport_size,
                    radius_px,
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
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let radius_px = match item.size_mode {
                    SpriteSizeMode::ScreenSpace => (item.default_size * 0.5).max(4.0),
                    SpriteSizeMode::WorldSpace => {
                        let n = item.positions.len() as f32;
                        let centroid = model.transform_point3(
                            item.positions
                                .iter()
                                .map(|p| glam::Vec3::from(*p))
                                .sum::<glam::Vec3>()
                                / n,
                        );
                        instance_radius_px(centroid, (item.default_size * 0.5).max(0.01))
                    }
                };
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos,
                    item.settings.pick_id.0,
                    &item.positions,
                    model,
                    view_proj,
                    viewport_size,
                    radius_px,
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
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let radius_px = (item.line_width + 4.0).max(8.0);
                if let Some(mut hit) = pick_gaussian_splat_cpu(
                    click_pos,
                    item.settings.pick_id.0,
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
                            hit.sub_object = Some(SubObjectRef::Strip(strip_for_node(
                                idx,
                                &item.strip_lengths,
                            )));
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
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
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
                    Some(SubObjectRef::Strip(strip_for_segment(
                        seg_idx,
                        &item.strip_lengths,
                    )))
                } else {
                    None
                };
                #[allow(deprecated)]
                let hit = PickHit {
                    id: item.settings.pick_id.0,
                    sub_object,
                    world_pos,
                    normal: glam::Vec3::Z,
                    triangle_index: u32::MAX,
                    point_index: None,
                    scalar_value: None,
                    sub_object_world_pos: None,
                };
                consider(toi, hit);
            }
        }

        // 9. Streamtube / tube / ribbon picks (POLY_NODE, SEGMENT, STRIP, or OBJECT).
        // Streamtube / tube: screen-space closest-segment test against each cylinder
        //     axis (both endpoints projected), not just the midpoint.
        //   Ribbon: ray-triangle intersection against the reconstructed swept quad
        //     using the parallel-transport lateral frame.
        //   POLY_NODE: control points are point-like sub-elements (pick_gaussian_splat_cpu).
        if wants_poly_node || wants_segment || wants_strip || wants_object {
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

            // POLY_NODE pass: nearest control point, promoted to Strip/Object as needed.
            if wants_poly_node || wants_strip || wants_object {
                for item in &self.pick_streamtube_items {
                    if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                        continue;
                    }
                    let ref_pos = glam::Vec3::from(item.positions[0]);
                    let radius_px = world_r_to_px(ref_pos, item.radius.max(0.01)).max(8.0);
                    if let Some(mut hit) = pick_gaussian_splat_cpu(
                        click_pos,
                        item.settings.pick_id.0,
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
                                hit.sub_object = Some(SubObjectRef::Strip(strip_for_node(
                                    idx,
                                    &item.strip_lengths,
                                )));
                            }
                        } else {
                            hit.sub_object = None;
                        }
                        consider(toi, hit);
                    }
                }
                for item in &self.pick_tube_items {
                    if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                        continue;
                    }
                    let ref_pos = glam::Vec3::from(item.positions[0]);
                    let max_r = item
                        .radius_attribute
                        .as_ref()
                        .and_then(|ra| ra.iter().copied().reduce(f32::max))
                        .unwrap_or(0.0)
                        .max(item.radius)
                        .max(0.01);
                    let radius_px = world_r_to_px(ref_pos, max_r).max(8.0);
                    if let Some(mut hit) = pick_gaussian_splat_cpu(
                        click_pos,
                        item.settings.pick_id.0,
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
                                hit.sub_object = Some(SubObjectRef::Strip(strip_for_node(
                                    idx,
                                    &item.strip_lengths,
                                )));
                            }
                        } else {
                            hit.sub_object = None;
                        }
                        consider(toi, hit);
                    }
                }
                for item in &self.pick_ribbon_items {
                    if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                        continue;
                    }
                    let ref_pos = glam::Vec3::from(item.positions[0]);
                    let radius_px = world_r_to_px(ref_pos, item.width * 0.5).max(8.0);
                    if let Some(mut hit) = pick_gaussian_splat_cpu(
                        click_pos,
                        item.settings.pick_id.0,
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
                                hit.sub_object = Some(SubObjectRef::Strip(strip_for_node(
                                    idx,
                                    &item.strip_lengths,
                                )));
                            }
                        } else {
                            hit.sub_object = None;
                        }
                        consider(toi, hit);
                    }
                }
            }

            // SEGMENT / STRIP / OBJECT pass using full geometric tests.
            if wants_segment || wants_strip || wants_object {
                // Streamtube: project each cylinder axis segment to screen and find the
                // closest point along the full segment (not just the midpoint).
                for item in &self.pick_streamtube_items {
                    if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                        continue;
                    }
                    let ref_pos = glam::Vec3::from(item.positions[0]);
                    let threshold_px = world_r_to_px(ref_pos, item.radius.max(0.01));
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
                        Some(SubObjectRef::Strip(strip_for_segment(
                            seg_idx,
                            &item.strip_lengths,
                        )))
                    } else {
                        None
                    };
                    #[allow(deprecated)]
                    consider(
                        toi,
                        PickHit {
                            id: item.settings.pick_id.0,
                            sub_object,
                            world_pos,
                            normal: glam::Vec3::Z,
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        },
                    );
                }

                // Tube: same as streamtube; uses the conservative max of uniform and
                // per-point radii for the screen-space threshold.
                for item in &self.pick_tube_items {
                    if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                        continue;
                    }
                    let ref_pos = glam::Vec3::from(item.positions[0]);
                    let max_r = item
                        .radius_attribute
                        .as_ref()
                        .and_then(|ra| ra.iter().copied().reduce(f32::max))
                        .unwrap_or(0.0)
                        .max(item.radius)
                        .max(0.01);
                    let threshold_px = world_r_to_px(ref_pos, max_r);
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
                        Some(SubObjectRef::Strip(strip_for_segment(
                            seg_idx,
                            &item.strip_lengths,
                        )))
                    } else {
                        None
                    };
                    #[allow(deprecated)]
                    consider(
                        toi,
                        PickHit {
                            id: item.settings.pick_id.0,
                            sub_object,
                            world_pos,
                            normal: glam::Vec3::Z,
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        },
                    );
                }

                // Ribbon: reconstruct the swept quad per segment (parallel-transport
                // lateral frame) and test the ray against both triangles of each quad.
                for item in &self.pick_ribbon_items {
                    if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                        continue;
                    }
                    let frames = ribbon_lateral_frames(
                        &item.positions,
                        &item.strip_lengths,
                        item.width,
                        item.width_attribute.as_deref(),
                        item.twist_attribute.as_deref(),
                    );

                    let single;
                    let strips: &[u32] = if item.strip_lengths.is_empty() {
                        single = [item.positions.len() as u32];
                        &single
                    } else {
                        &item.strip_lengths
                    };

                    let mut best_t = f32::MAX;
                    let mut best_seg: Option<(u32, glam::Vec3)> = None;
                    let mut node_off = 0usize;
                    let mut seg_off = 0u32;

                    for &slen in strips {
                        let slen = slen as usize;
                        for k in 0..slen.saturating_sub(1) {
                            let ia = node_off + k;
                            let ib = node_off + k + 1;
                            let pa = glam::Vec3::from(item.positions[ia]);
                            let pb = glam::Vec3::from(item.positions[ib]);
                            let (ua, wa) = frames[ia];
                            let (ub, wb) = frames[ib];
                            // Quad corners: c0/c1 at segment start, c2/c3 at end.
                            let c0 = pa + ua * wa; // left  at a
                            let c1 = pa - ua * wa; // right at a
                            let c2 = pb + ub * wb; // left  at b
                            let c3 = pb - ub * wb; // right at b
                            // Test 2 triangles, both front and back faces.
                            let t = ray_triangle(ray_origin, ray_dir, c0, c1, c2)
                                .or_else(|| ray_triangle(ray_origin, ray_dir, c1, c3, c2))
                                .or_else(|| ray_triangle(ray_origin, ray_dir, c2, c1, c0))
                                .or_else(|| ray_triangle(ray_origin, ray_dir, c2, c3, c1));
                            if let Some(t) = t {
                                if t < best_t {
                                    best_t = t;
                                    best_seg = Some((seg_off + k as u32, ray_origin + ray_dir * t));
                                }
                            }
                        }
                        seg_off += slen.saturating_sub(1) as u32;
                        node_off += slen;
                    }

                    if let Some((seg_idx, world_pos)) = best_seg {
                        let sub_object = if wants_segment {
                            Some(SubObjectRef::Segment(seg_idx))
                        } else if wants_strip {
                            Some(SubObjectRef::Strip(strip_for_segment(
                                seg_idx,
                                &item.strip_lengths,
                            )))
                        } else {
                            None
                        };
                        #[allow(deprecated)]
                        consider(
                            best_t,
                            PickHit {
                                id: item.settings.pick_id.0,
                                sub_object,
                                world_pos,
                                normal: glam::Vec3::Z,
                                triangle_index: u32::MAX,
                                point_index: None,
                                scalar_value: None,
                                sub_object_world_pos: None,
                            },
                        );
                    }
                }
            }
        }

        // 10. Image slice / volume surface slice / screen image object picks (OBJECT only).
        if wants_object {
            // Image slice: axis-aligned quad ray intersection.
            for item in &self.pick_image_slice_items {
                if item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let [bmin, bmax] = [item.bbox_min, item.bbox_max];
                let t = item.offset;
                // Plane normal and position along the axis.
                let (axis_idx, plane_pos) = match item.axis {
                    SliceAxis::X => (0usize, bmin[0] + t * (bmax[0] - bmin[0])),
                    SliceAxis::Y => (1usize, bmin[1] + t * (bmax[1] - bmin[1])),
                    SliceAxis::Z => (2usize, bmin[2] + t * (bmax[2] - bmin[2])),
                };
                let plane_n = {
                    let mut n = glam::Vec3::ZERO;
                    n[axis_idx] = 1.0;
                    n
                };
                let denom = plane_n.dot(ray_dir);
                if denom.abs() < 1e-6 {
                    continue;
                }
                let toi = (plane_pos - ray_origin[axis_idx]) / denom;
                if toi <= 0.0 {
                    continue;
                }
                let hit_pos = ray_origin + ray_dir * toi;
                // Check that the hit is within the slice quad's other two dimensions.
                let in_bounds = (0..3)
                    .filter(|&i| i != axis_idx)
                    .all(|i| hit_pos[i] >= bmin[i] - 1e-4 && hit_pos[i] <= bmax[i] + 1e-4);
                if in_bounds {
                    #[allow(deprecated)]
                    consider(
                        toi,
                        PickHit {
                            id: item.settings.pick_id.0,
                            sub_object: None,
                            world_pos: hit_pos,
                            normal: plane_n,
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        },
                    );
                }
            }

            // Volume surface slice: ray/mesh intersection via mesh_store CPU data.
            for item in &self.pick_volume_surface_slice_items {
                if item.settings.pick_id == PickId::NONE {
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
                let verts: Vec<parry3d::math::Vector> = positions
                    .iter()
                    .map(|p| {
                        let wp = model.transform_point3(glam::Vec3::from(*p));
                        parry3d::math::Vector::new(wp.x, wp.y, wp.z)
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
                let ray = parry3d::query::Ray::new(
                    parry3d::math::Vector::new(ray_origin.x, ray_origin.y, ray_origin.z),
                    parry3d::math::Vector::new(ray_dir.x, ray_dir.y, ray_dir.z),
                );
                if let Ok(trimesh) = parry3d::shape::TriMesh::new(verts, tri_indices) {
                    use parry3d::query::RayCast;
                    if let Some(hit) = trimesh.cast_ray_and_get_normal(
                        &parry3d::math::Pose::identity(),
                        &ray,
                        f32::MAX,
                        true,
                    ) {
                        let world_pos = ray_origin + ray_dir * hit.time_of_impact;
                        let n = hit.normal;
                        #[allow(deprecated)]
                        consider(
                            hit.time_of_impact,
                            PickHit {
                                id: item.settings.pick_id.0,
                                sub_object: None,
                                world_pos,
                                normal: glam::Vec3::new(n.x, n.y, n.z),
                                triangle_index: u32::MAX,
                                point_index: None,
                                scalar_value: None,
                                sub_object_world_pos: None,
                            },
                        );
                    }
                }
            }

            // Screen image: screen-space rect test. toi=0 so these win over any 3D hit.
            for item in &self.pick_screen_image_items {
                if item.settings.pick_id == PickId::NONE || item.width == 0 || item.height == 0 {
                    continue;
                }
                let img_w = item.width as f32 * item.scale;
                let img_h = item.height as f32 * item.scale;
                let (sx, sy) = match item.anchor {
                    ImageAnchor::TopLeft => (0.0, 0.0),
                    ImageAnchor::TopRight => (viewport_size.x - img_w, 0.0),
                    ImageAnchor::BottomLeft => (0.0, viewport_size.y - img_h),
                    ImageAnchor::BottomRight => (viewport_size.x - img_w, viewport_size.y - img_h),
                    ImageAnchor::Center => (
                        (viewport_size.x - img_w) * 0.5,
                        (viewport_size.y - img_h) * 0.5,
                    ),
                };
                if click_pos.x >= sx
                    && click_pos.x <= sx + img_w
                    && click_pos.y >= sy
                    && click_pos.y <= sy + img_h
                {
                    // No meaningful 3D position; place the hit at the near-plane.
                    let world_pos = ray_origin + ray_dir * 0.001;
                    #[allow(deprecated)]
                    consider(
                        0.0,
                        PickHit {
                            id: item.settings.pick_id.0,
                            sub_object: None,
                            world_pos,
                            normal: -ray_dir,
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        },
                    );
                }
            }
        }

        // 11. GPU implicit surface picks (OBJECT only -- no sub-element model).
        if wants_object {
            for item in &self.pick_implicit_items {
                if let Some((toi, world_pos)) = pick_implicit_sdf(ray_origin, ray_dir, item) {
                    #[allow(deprecated)]
                    consider(
                        toi,
                        PickHit {
                            id: item.id,
                            sub_object: None,
                            world_pos,
                            normal: glam::Vec3::Z,
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        },
                    );
                }
            }
        }

        // 12. GPU marching cubes surface picks (OBJECT only).
        if wants_object {
            for item in &self.pick_mc_items {
                if let Some((toi, world_pos)) = pick_mc_volume(ray_origin, ray_dir, item) {
                    #[allow(deprecated)]
                    consider(
                        toi,
                        PickHit {
                            id: item.id,
                            sub_object: None,
                            world_pos,
                            normal: glam::Vec3::Z,
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        },
                    );
                }
            }
        }

        // 13. Decal picks (OBJECT only): ray versus the decal projection box.
        // A decal is the unit box [-0.5, 0.5]^3 mapped to world by `transform`.
        // The box front face typically hugs the receiver surface, so a decal
        // that straddles a surface wins over that surface by `toi`, letting a
        // click select the decal itself. A decal whose box floats in empty
        // space is still pickable wherever the ray passes through the volume.
        if wants_object {
            for item in &self.pick_decal_items {
                if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.transform);
                if model.determinant().abs() < 1e-12 {
                    continue;
                }
                let inv = model.inverse();
                let local_origin = inv.transform_point3(ray_origin);
                let local_dir = inv.transform_vector3(ray_dir);
                if let Some(toi) = ray_unit_box_toi(local_origin, local_dir) {
                    let world_pos = ray_origin + ray_dir * toi;
                    #[allow(deprecated)]
                    consider(
                        toi,
                        PickHit {
                            id: item.settings.pick_id.0,
                            sub_object: None,
                            world_pos,
                            normal: -ray_dir.normalize_or_zero(),
                            triangle_index: u32::MAX,
                            point_index: None,
                            scalar_value: None,
                            sub_object_world_pos: None,
                        },
                    );
                }
            }
        }

        // Consult registered item-type plugins after the built-in pickers.
        // Each plugin returns its own closest hit; the router compares
        // by world-space ray t against the running best.
        if !self.item_type_plugins.is_empty() {
            let plugin_ray = crate::plugin_api::PickRay {
                origin: ray_origin,
                direction: ray_dir,
            };
            for plugin in self.item_type_plugins.values() {
                if let Some((t, hit)) = plugin.pick(&plugin_ray) {
                    consider(t, hit);
                }
            }
        }

        best.map(|(_, hit)| hit)
    }
}
