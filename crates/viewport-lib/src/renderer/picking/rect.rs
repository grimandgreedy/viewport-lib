//! Rectangle pick: collect all items and sub-elements whose projected
//! geometry falls inside a screen-space selection rectangle.

use super::*;

impl ViewportRenderer {
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
        mask: PickMask,
    ) -> PickRectResult {
        let mut result = PickRectResult::default();

        if !self.cpu_pick_cache_enabled {
            warn_pick_cache_disabled();
            return result;
        }

        if viewport_size.x <= 0.0 || viewport_size.y <= 0.0 {
            return result;
        }

        let wants_face = mask.intersects(PickMask::FACE);
        let wants_vertex = mask.intersects(PickMask::VERTEX);
        let wants_cell = mask.intersects(PickMask::CELL);
        let wants_object = mask.intersects(PickMask::OBJECT);

        // Build lookup for opaque volume mesh face_to_cell maps.
        let vm_cell_map: std::collections::HashMap<u64, &[u32]> = self
            .pick_volume_mesh_items
            .iter()
            .filter(|item| item.settings.pick_id != PickId::NONE && !item.face_to_cell.is_empty())
            .map(|item| (item.settings.pick_id.0, item.face_to_cell.as_slice()))
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
            // Broad phase: only the items whose world AABB falls in the rect's
            // frustum. The per-item body is unchanged, so the result matches a full
            // scan; items outside the frustum project outside the rect anyway.
            let candidates =
                self.rect_candidate_items(rect_min, rect_max, viewport_size, view_proj);
            for &item_index in &candidates {
                let item = &self.pick_scene_items[item_index];
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

                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.settings.pick_id.0;
                let mut item_hit = false;

                if wants_face {
                    for (tri_idx, chunk) in indices.chunks(3).enumerate() {
                        if chunk.len() < 3 {
                            continue;
                        }
                        let [i0, i1, i2] =
                            [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize];
                        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                            continue;
                        }
                        let centroid = (glam::Vec3::from(positions[i0])
                            + glam::Vec3::from(positions[i1])
                            + glam::Vec3::from(positions[i2]))
                            / 3.0;
                        if let Some((sx, sy)) = project(mvp, centroid) {
                            if in_rect(sx, sy) {
                                result
                                    .elements
                                    .push((id, SubObjectRef::Face(tri_idx as u32)));
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
                    } else if wants_vertex {
                        // No cell map; fall through to vertex picking for regular meshes.
                        for (vi, pos) in positions.iter().enumerate() {
                            if let Some((sx, sy)) = project(mvp, glam::Vec3::from(*pos)) {
                                if in_rect(sx, sy) {
                                    result.elements.push((id, SubObjectRef::Vertex(vi as u32)));
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
                        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
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
                use crate::resources::volume::volume_mesh::CELL_SENTINEL;
                let id = item.settings.pick_id.0;
                let mvp = view_proj * glam::Mat4::from_cols_array_2d(&item.model);
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
                                result
                                    .elements
                                    .push((id, SubObjectRef::Cell(cell_idx as u32)));
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
            // Sprites
            for item in &self.pick_sprite_items {
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.settings.pick_id.0;
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

        // 9. Screen image object rect picks (OBJECT only).
        if wants_object {
            // Screen image: check if the image's screen rect overlaps the pick rect.
            for item in &self.pick_screen_image_items {
                if item.settings.pick_id == PickId::NONE || item.width == 0 || item.height == 0 {
                    continue;
                }
                let img_w = item.width as f32 * item.scale;
                let img_h = item.height as f32 * item.scale;
                let [sx, sy] = crate::renderer::types::viewport_anchored_top_left(
                    item.anchor_x,
                    item.anchor_y,
                    [img_w, img_h],
                    [viewport_size.x, viewport_size.y],
                );
                // Overlap: image rect [sx, sx+img_w] x [sy, sy+img_h] vs pick rect.
                let overlap = sx <= rect_max.x
                    && sx + img_w >= rect_min.x
                    && sy <= rect_max.y
                    && sy + img_h >= rect_min.y;
                if overlap {
                    result.objects.push(item.settings.pick_id.0);
                }
            }
        }

        // 13. Decal rect picks (OBJECT only): project the decal projection box
        // (unit cube [-0.5, 0.5]^3 mapped by `transform`) and test its corners
        // and edges against the selection rect. Mirrors the ray-versus-box test
        // used by the single-item pick so box-select and click agree.
        if wants_object {
            // Unit-box corners in (x, y, z) bit order, and the 12 edges joining
            // corners that differ in exactly one axis.
            const CORNERS: [[f32; 3]; 8] = [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ];
            const EDGES: [(usize, usize); 12] = [
                (0, 1),
                (0, 2),
                (0, 4),
                (1, 3),
                (1, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
            ];
            for item in &self.pick_decal_items {
                if item.settings.hidden || item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.transform);
                if model.determinant().abs() < 1e-12 {
                    continue;
                }
                let mvp = view_proj * model;
                let sc: [Option<glam::Vec2>; 8] = std::array::from_fn(|i| {
                    project(mvp, glam::Vec3::from(CORNERS[i])).map(|(x, y)| glam::Vec2::new(x, y))
                });
                let hit = sc.iter().any(|p| p.map_or(false, |p| in_rect(p.x, p.y)))
                    || EDGES.iter().any(|&(a, b)| match (sc[a], sc[b]) {
                        (Some(a), Some(b)) => segment_in_rect(a, b, rect_min, rect_max),
                        (Some(a), None) => in_rect(a.x, a.y),
                        (None, Some(b)) => in_rect(b.x, b.y),
                        (None, None) => false,
                    });
                if hit {
                    result.objects.push(item.settings.pick_id.0);
                }
            }
        }

        // Consult registered item-type plugins after the built-in types.
        // Each plugin answers from state cached in its own prepare, the
        // same as the point pick.
        if !self.item_type_plugins.is_empty() {
            let plugin_ctx = crate::plugin_api::RectPickContext {
                rect_min,
                rect_max,
                viewport_size,
                view_proj,
                mask,
                meshes: crate::resources::MeshGeometry::new(&self.resources),
            };
            for plugin in self.item_type_plugins.values() {
                let plugin_result = plugin.pick_rect(&plugin_ctx);
                result.objects.extend(plugin_result.objects);
                result.elements.extend(plugin_result.elements);
            }
        }

        result
    }
}
