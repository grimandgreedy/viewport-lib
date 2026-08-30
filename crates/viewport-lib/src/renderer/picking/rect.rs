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
        let wants_cloud = mask.intersects(PickMask::CLOUD_POINT);
        let wants_splat = mask.intersects(PickMask::SPLAT);
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

        // 3. Point cloud picks (CLOUD_POINT or OBJECT).
        if wants_cloud || wants_object {
            for item in &self.pick_point_cloud_items {
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let mvp = view_proj * model;
                let id = item.settings.pick_id.0;
                let mut item_hit = false;

                for (pt_idx, pos) in item.positions.iter().enumerate() {
                    if let Some((sx, sy)) = project(mvp, glam::Vec3::from(*pos)) {
                        if in_rect(sx, sy) {
                            if wants_cloud {
                                result
                                    .elements
                                    .push((id, SubObjectRef::Point(pt_idx as u32)));
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
                if item.settings.pick_id == PickId::NONE {
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
                let cell = (bbox_max - bbox_min) / glam::Vec3::new(nx as f32, ny as f32, nz as f32);
                let id = item.settings.pick_id.0;
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
                                + cell
                                    * glam::Vec3::new(
                                        ix as f32 + 0.5,
                                        iy as f32 + 0.5,
                                        iz as f32 + 0.5,
                                    );
                            if let Some((sx, sy)) = project(mvp, center) {
                                if in_rect(sx, sy) {
                                    if wants_voxel {
                                        result
                                            .elements
                                            .push((id, SubObjectRef::Voxel(flat as u32)));
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
                let mvp = view_proj * model;
                let id = item.settings.pick_id.0;
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

            // Tensor glyphs
            for item in &self.pick_tensor_glyph_items {
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

        // 7. Polyline node / segment / strip / object rect picks.
        let wants_poly_node = mask.intersects(PickMask::POLY_NODE);
        let wants_segment = mask.intersects(PickMask::SEGMENT);
        let wants_strip = mask.intersects(PickMask::STRIP);
        if wants_poly_node || wants_segment || wants_strip || wants_object {
            for item in &self.pick_polyline_items {
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }
                let id = item.settings.pick_id.0;
                let mut item_hit = false;
                let mut strips_hit = std::collections::HashSet::<u32>::new();

                // Node pass (POLY_NODE or STRIP or OBJECT).
                if wants_poly_node || wants_strip || wants_object {
                    for (node_idx, pos) in item.positions.iter().enumerate() {
                        if let Some((sx, sy)) = project(view_proj, glam::Vec3::from(*pos)) {
                            if in_rect(sx, sy) {
                                item_hit = true;
                                if wants_poly_node {
                                    result
                                        .elements
                                        .push((id, SubObjectRef::Point(node_idx as u32)));
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
                                    rect_min,
                                    rect_max,
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
        if wants_poly_node || wants_segment || wants_strip || wants_object {
            // Streamtube and tube: test both projected endpoints of each segment
            // with segment_in_rect instead of the midpoint projection heuristic.
            // POLY_NODE: also check each control point individually.
            let st_tube_iter = self
                .pick_streamtube_items
                .iter()
                .map(|it| {
                    (
                        it.settings.pick_id.0,
                        it.positions.as_slice(),
                        it.strip_lengths.as_slice(),
                    )
                })
                .chain(self.pick_tube_items.iter().map(|it| {
                    (
                        it.settings.pick_id.0,
                        it.positions.as_slice(),
                        it.strip_lengths.as_slice(),
                    )
                }));

            for (id, positions, strip_lengths) in st_tube_iter {
                if id == 0 || positions.is_empty() {
                    continue;
                }
                let mut item_hit = false;
                let mut strips_hit = std::collections::HashSet::<u32>::new();

                let single_st;
                let strips_st: &[u32] = if strip_lengths.is_empty() {
                    single_st = [positions.len() as u32];
                    &single_st
                } else {
                    strip_lengths
                };

                // POLY_NODE pass: project each control point and check in_rect.
                if wants_poly_node || wants_strip || wants_object {
                    'st_nodes: for (ni, pos) in positions.iter().enumerate() {
                        if let Some((sx, sy)) = project(view_proj, glam::Vec3::from(*pos)) {
                            if in_rect(sx, sy) {
                                item_hit = true;
                                if wants_poly_node {
                                    result.elements.push((id, SubObjectRef::Point(ni as u32)));
                                } else if wants_strip {
                                    let s = strip_for_node(ni as u32, strip_lengths);
                                    strips_hit.insert(s);
                                } else {
                                    // wants_object only: no need to enumerate further nodes.
                                    break 'st_nodes;
                                }
                            }
                        }
                    }
                }

                // SEGMENT pass: test both projected endpoints of each segment.
                if wants_segment || wants_strip || wants_object {
                    let mut node_off = 0usize;
                    let mut seg_off = 0u32;
                    'st_strips: for &slen in strips_st {
                        let slen = slen as usize;
                        for j in 0..slen.saturating_sub(1) {
                            let seg_idx = seg_off + j as u32;
                            let pa = glam::Vec3::from(positions[node_off + j]);
                            let pb = glam::Vec3::from(positions[node_off + j + 1]);
                            let hit = match (project(view_proj, pa), project(view_proj, pb)) {
                                (Some((ax, ay)), Some((bx, by))) => segment_in_rect(
                                    glam::Vec2::new(ax, ay),
                                    glam::Vec2::new(bx, by),
                                    rect_min,
                                    rect_max,
                                ),
                                (Some((ax, ay)), None) => in_rect(ax, ay),
                                (None, Some((bx, by))) => in_rect(bx, by),
                                (None, None) => false,
                            };
                            if hit {
                                item_hit = true;
                                if wants_segment {
                                    result.elements.push((id, SubObjectRef::Segment(seg_idx)));
                                } else if wants_strip {
                                    let s = strip_for_segment(seg_idx, strip_lengths);
                                    strips_hit.insert(s);
                                } else {
                                    // wants_object only: no need to enumerate further segments.
                                    break 'st_strips;
                                }
                            }
                        }
                        seg_off += slen.saturating_sub(1) as u32;
                        node_off += slen;
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

            // Ribbon: reconstruct the swept quad per segment and test all four
            // quad edges with segment_in_rect (also catches quad corners inside
            // the rect via the endpoint check inside segment_in_rect).
            // POLY_NODE: also check each control point individually.
            for item in &self.pick_ribbon_items {
                if item.settings.pick_id == PickId::NONE || item.positions.is_empty() {
                    continue;
                }

                let single_r;
                let strips_r: &[u32] = if item.strip_lengths.is_empty() {
                    single_r = [item.positions.len() as u32];
                    &single_r
                } else {
                    &item.strip_lengths
                };

                let mut item_hit = false;
                let mut strips_hit = std::collections::HashSet::<u32>::new();

                // Project a world point to screen Vec2; returns None if behind camera.
                let proj2 = |p: glam::Vec3| -> Option<glam::Vec2> {
                    project(view_proj, p).map(|(x, y)| glam::Vec2::new(x, y))
                };

                // POLY_NODE pass: project each control point and check in_rect.
                if wants_poly_node || wants_strip || wants_object {
                    'rb_nodes: for (ni, pos) in item.positions.iter().enumerate() {
                        if let Some((sx, sy)) = project(view_proj, glam::Vec3::from(*pos)) {
                            if in_rect(sx, sy) {
                                item_hit = true;
                                if wants_poly_node {
                                    result.elements.push((
                                        item.settings.pick_id.0,
                                        SubObjectRef::Point(ni as u32),
                                    ));
                                } else if wants_strip {
                                    let s = strip_for_node(ni as u32, &item.strip_lengths);
                                    strips_hit.insert(s);
                                } else {
                                    break 'rb_nodes;
                                }
                            }
                        }
                    }
                }

                // SEGMENT pass: quad edge tests using ribbon_lateral_frames.
                if wants_segment || wants_strip || wants_object {
                    let frames = ribbon_lateral_frames(
                        &item.positions,
                        &item.strip_lengths,
                        item.width,
                        item.width_attribute.as_deref(),
                        item.twist_attribute.as_deref(),
                    );
                    let mut node_off = 0usize;
                    let mut seg_off = 0u32;

                    'rb_strips: for &slen in strips_r {
                        let slen = slen as usize;
                        for k in 0..slen.saturating_sub(1) {
                            let seg_idx = seg_off + k as u32;
                            let ia = node_off + k;
                            let ib = node_off + k + 1;
                            let pa = glam::Vec3::from(item.positions[ia]);
                            let pb = glam::Vec3::from(item.positions[ib]);
                            let (ua, wa) = frames[ia];
                            let (ub, wb) = frames[ib];
                            let c0 = pa + ua * wa; // left  at a
                            let c1 = pa - ua * wa; // right at a
                            let c2 = pb + ub * wb; // left  at b
                            let c3 = pb - ub * wb; // right at b
                            let sc0 = proj2(c0);
                            let sc1 = proj2(c1);
                            let sc2 = proj2(c2);
                            let sc3 = proj2(c3);
                            let edge_hit = |a: Option<glam::Vec2>, b: Option<glam::Vec2>| -> bool {
                                match (a, b) {
                                    (Some(a), Some(b)) => segment_in_rect(a, b, rect_min, rect_max),
                                    (Some(a), None) => in_rect(a.x, a.y),
                                    (None, Some(b)) => in_rect(b.x, b.y),
                                    (None, None) => false,
                                }
                            };
                            let hit = edge_hit(sc0, sc1)
                                || edge_hit(sc2, sc3)
                                || edge_hit(sc0, sc2)
                                || edge_hit(sc1, sc3);
                            if hit {
                                item_hit = true;
                                if wants_segment {
                                    result.elements.push((
                                        item.settings.pick_id.0,
                                        SubObjectRef::Segment(seg_idx),
                                    ));
                                } else if wants_strip {
                                    let s = strip_for_segment(seg_idx, &item.strip_lengths);
                                    strips_hit.insert(s);
                                } else {
                                    break 'rb_strips;
                                }
                            }
                        }
                        seg_off += slen.saturating_sub(1) as u32;
                        node_off += slen;
                    }
                }

                if wants_strip {
                    for s in strips_hit {
                        result
                            .elements
                            .push((item.settings.pick_id.0, SubObjectRef::Strip(s)));
                    }
                }
                if wants_object && item_hit {
                    result.objects.push(item.settings.pick_id.0);
                }
            }
        }

        // 9. Image slice / volume surface slice / screen image object rect picks (OBJECT only).
        if wants_object {
            // Image slice: project all 4 quad corners and check containment/edge intersection.
            for item in &self.pick_image_slice_items {
                if item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let [bmin, bmax] = [item.bbox_min, item.bbox_max];
                let t = item.offset;
                let corners: [[f32; 3]; 4] = match item.axis {
                    SliceAxis::X => {
                        let x = bmin[0] + t * (bmax[0] - bmin[0]);
                        [
                            [x, bmin[1], bmin[2]],
                            [x, bmax[1], bmin[2]],
                            [x, bmax[1], bmax[2]],
                            [x, bmin[1], bmax[2]],
                        ]
                    }
                    SliceAxis::Y => {
                        let y = bmin[1] + t * (bmax[1] - bmin[1]);
                        [
                            [bmin[0], y, bmin[2]],
                            [bmax[0], y, bmin[2]],
                            [bmax[0], y, bmax[2]],
                            [bmin[0], y, bmax[2]],
                        ]
                    }
                    SliceAxis::Z => {
                        let z = bmin[2] + t * (bmax[2] - bmin[2]);
                        [
                            [bmin[0], bmin[1], z],
                            [bmax[0], bmin[1], z],
                            [bmax[0], bmax[1], z],
                            [bmin[0], bmax[1], z],
                        ]
                    }
                };
                let sc: Vec<Option<glam::Vec2>> = corners
                    .iter()
                    .map(|&c| {
                        project(view_proj, glam::Vec3::from(c)).map(|(x, y)| glam::Vec2::new(x, y))
                    })
                    .collect();
                let hit = sc.iter().any(|p| p.map_or(false, |p| in_rect(p.x, p.y)))
                    || (0..4).any(|i| {
                        let a = sc[i];
                        let b = sc[(i + 1) % 4];
                        match (a, b) {
                            (Some(a), Some(b)) => segment_in_rect(a, b, rect_min, rect_max),
                            (Some(a), None) => in_rect(a.x, a.y),
                            (None, Some(b)) => in_rect(b.x, b.y),
                            (None, None) => false,
                        }
                    });
                if hit {
                    result.objects.push(item.settings.pick_id.0);
                }
            }

            // Volume surface slice: project each mesh vertex (with model transform) and check.
            for item in &self.pick_volume_surface_slice_items {
                if item.settings.pick_id == PickId::NONE {
                    continue;
                }
                let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                    continue;
                };
                let Some(positions) = &mesh.cpu_positions else {
                    continue;
                };
                let model = glam::Mat4::from_cols_array_2d(&item.model);
                let hit = positions.iter().any(|&p| {
                    let wp = model.transform_point3(glam::Vec3::from(p));
                    project(view_proj, wp).map_or(false, |(sx, sy)| in_rect(sx, sy))
                });
                if hit {
                    result.objects.push(item.settings.pick_id.0);
                }
            }

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

        // 11. GPU implicit surface rect picks (OBJECT only).
        //
        // For each primitive compute a conservative screen-space AABB by projecting
        // the primitive's bounding sphere. If any projected AABB corner falls inside
        // the pick rect, the item is a hit. This is approximate (the actual rendered
        // surface may be smaller) but avoids per-pixel SDF marching for rect queries.
        if wants_object {
            for item in &self.pick_implicit_items {
                let mut hit = false;
                'prim_loop: for prim in &item.primitives {
                    // Derive a bounding sphere center and radius for each primitive.
                    let (center, radius) = match prim.kind {
                        1 => {
                            // Sphere
                            let c = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
                            (c, prim.params[3].abs())
                        }
                        2 => {
                            // Box: center + max half-extent as radius
                            let c = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
                            let h = glam::Vec3::new(prim.params[4], prim.params[5], prim.params[6]);
                            (c, h.length())
                        }
                        3 => {
                            // Plane: not bounded -- skip.
                            continue;
                        }
                        4 => {
                            // Capsule: midpoint of segment + (half-length + radius)
                            let a = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
                            let b = glam::Vec3::new(prim.params[4], prim.params[5], prim.params[6]);
                            let r = prim.params[3].abs();
                            ((a + b) * 0.5, (b - a).length() * 0.5 + r)
                        }
                        _ => continue,
                    };
                    // Project 8 AABB corners of the bounding sphere box.
                    for dx in [-radius, radius] {
                        for dy in [-radius, radius] {
                            for dz in [-radius, radius] {
                                let corner = center + glam::Vec3::new(dx, dy, dz);
                                if let Some((sx, sy)) = project(view_proj, corner) {
                                    if in_rect(sx, sy) {
                                        hit = true;
                                        break 'prim_loop;
                                    }
                                }
                            }
                        }
                    }
                }
                if hit {
                    result.objects.push(item.id);
                }
            }
        }

        // 12. GPU marching cubes surface rect picks (OBJECT only).
        //
        // Iterates over all cells in the volume where the scalar field straddles
        // the isovalue (MC would generate triangles there). If any such cell's
        // center projects into the pick rect, the item is a hit.
        if wants_object {
            for item in &self.pick_mc_items {
                let vol = &item.volume_data;
                let isovalue = item.isovalue;
                let [nx, ny, nz] = vol.dims;
                let origin = glam::Vec3::from(vol.origin);
                let spacing = glam::Vec3::from(vol.spacing);

                let mut hit = false;
                'mc_rect: for iz in 0..nz.saturating_sub(1) {
                    for iy in 0..ny.saturating_sub(1) {
                        for ix in 0..nx.saturating_sub(1) {
                            // A cell straddles the isovalue when not all 8 corners
                            // are on the same side. Check for both above and below.
                            let mut has_below = false;
                            let mut has_above = false;
                            'corners: for dz in 0u32..=1 {
                                for dy in 0u32..=1 {
                                    for dx in 0u32..=1 {
                                        let s = vol.sample(ix + dx, iy + dy, iz + dz);
                                        if s < isovalue {
                                            has_below = true;
                                        } else {
                                            has_above = true;
                                        }
                                        if has_below && has_above {
                                            break 'corners;
                                        }
                                    }
                                }
                            }
                            if !(has_below && has_above) {
                                continue;
                            }
                            let cell_center = origin
                                + spacing
                                    * glam::Vec3::new(
                                        ix as f32 + 0.5,
                                        iy as f32 + 0.5,
                                        iz as f32 + 0.5,
                                    );
                            if let Some((sx, sy)) = project(view_proj, cell_center) {
                                if in_rect(sx, sy) {
                                    hit = true;
                                    break 'mc_rect;
                                }
                            }
                        }
                    }
                }
                if hit {
                    result.objects.push(item.id);
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

        result
    }
}
