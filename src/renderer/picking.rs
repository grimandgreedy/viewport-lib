use super::*;

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
            screen_to_ray, pick_point_cloud_cpu, pick_gaussian_splat_cpu, pick_volume_cpu, PickHit,
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

        // 1. Surface mesh picks (FACE, VERTEX, EDGE, or OBJECT fallback).
        if wants_mesh_sub || wants_object {
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

        // 2. Volume mesh cell picks (CELL) -- stub: face_to_cell mapping is not
        // retained in pick_scene_items. Cell-level picks will be wired in when
        // volume mesh CPU data retention lands.

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
        let wants_cloud  = mask.intersects(PickMask::CLOUD_POINT);
        let wants_splat  = mask.intersects(PickMask::SPLAT);
        let wants_object = mask.intersects(PickMask::OBJECT);

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

        // 1. Surface mesh picks (FACE, VERTEX, or OBJECT).
        if wants_face || wants_vertex || wants_object {
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

        // 2. Volume mesh cell picks (CELL) -- stub: face_to_cell not in cache.

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

        let vp_w = frame.camera.viewport_size[0] as u32;
        let vp_h = frame.camera.viewport_size[1] as u32;

        // --- bounds check ---
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

        let px = cursor.x as u32;
        let py = cursor.y as u32;

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
