//! Per-viewport finalization passes: Gaussian splat sort and wireframe,
//! sprite wireframe, debug fragment buffer, and atlas blit uniform.

use super::*;

impl ViewportRenderer {
    pub(super) fn prepare_splat_sort(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // ------------------------------------------------------------------
        // Gaussian splat: per-viewport GPU sort.
        // ------------------------------------------------------------------
        self.gaussian_splat_draw_data.clear();
        if !frame.scene.gaussian_splats.is_empty() {
            self.resources.ensure_gaussian_splat_pipelines(device);
            let vp_idx = frame.camera.viewport_index;
            let eye = frame.camera.render_camera.eye_position;
            let vp_w = frame.camera.viewport_size[0].max(1.0);
            let vp_h = frame.camera.viewport_size[1].max(1.0);
            for item in &frame.scene.gaussian_splats {
                if item.settings.hidden {
                    continue;
                }
                if self
                    .resources
                    .content
                    .gaussian_splat_store
                    .get(item.source)
                    .is_none()
                {
                    continue;
                }
                let store_index = item.source.index();
                let sh_degree = self
                    .resources
                    .content
                    .gaussian_splat_store
                    .get(item.source)
                    .unwrap()
                    .sh_degree;
                let count = self
                    .resources
                    .content
                    .gaussian_splat_store
                    .get(item.source)
                    .unwrap()
                    .count;
                self.resources.run_gaussian_splat_sort(
                    device,
                    queue,
                    store_index,
                    vp_idx,
                    eye,
                    item.model,
                    vp_w,
                    vp_h,
                    sh_degree,
                );
                self.gaussian_splat_draw_data
                    .push(crate::resources::GaussianSplatDrawData {
                        store_index,
                        viewport_index: vp_idx,
                        model: item.model,
                        count,
                        wireframe: frame.viewport.wireframe_mode || item.settings.wireframe,
                        pick_id: item.settings.pick_id,
                    });
            }
        }
    }

    pub(super) fn prepare_splat_wireframe(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // Gaussian splat wireframe overlay.
        let need_splat_wf = frame.viewport.wireframe_mode
            || frame
                .scene
                .gaussian_splats
                .iter()
                .any(|g| !g.settings.hidden && g.settings.wireframe);
        if need_splat_wf {
            self.resources.ensure_polyline_pipeline(device);
            let vp_size = frame.camera.viewport_size;
            for item in &frame.scene.gaussian_splats {
                if item.settings.hidden {
                    continue;
                }
                if !(frame.viewport.wireframe_mode || item.settings.wireframe) {
                    continue;
                }
                let Some(gpu_set) = self.resources.content.gaussian_splat_store.get(item.source)
                else {
                    continue;
                };
                let count = gpu_set.count as usize;
                let positions = gpu_set.cpu_positions.clone();
                let scales = gpu_set.cpu_scales.clone();
                let _ = gpu_set;
                let polyline = splat_wireframe_polyline(&positions, &scales, item.model, count);
                if !polyline.positions.is_empty() {
                    let gpu = self
                        .resources
                        .upload_polyline_per_frame(device, queue, &polyline, vp_size);
                    self.polyline_gpu_data.push(gpu);
                }
            }
        }
    }

    pub(super) fn prepare_sprite_wireframe(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // Sprite wireframe overlay: quad outline per sprite (<=100) or AABB box (>100).
        let need_sprite_wf = frame.viewport.wireframe_mode
            || frame
                .scene
                .sprite_items
                .iter()
                .any(|s| !s.settings.hidden && s.settings.wireframe);
        if need_sprite_wf {
            self.resources.ensure_polyline_pipeline(device);
            let vp_size = frame.camera.viewport_size;
            for item in &frame.scene.sprite_items {
                if item.settings.hidden {
                    continue;
                }
                if !(frame.viewport.wireframe_mode || item.settings.wireframe) {
                    continue;
                }
                if item.positions.is_empty() {
                    continue;
                }
                let polyline = sprite_wireframe_polyline(item, &frame.camera);
                if !polyline.positions.is_empty() {
                    let gpu = self
                        .resources
                        .upload_polyline_per_frame(device, queue, &polyline, vp_size);
                    self.polyline_gpu_data.push(gpu);
                }
            }
        }
    }

    pub(super) fn prepare_debug_buffer(&mut self, device: &crate::gpu::Device, frame: &FrameData) {
        // Debug fragment buffer: allocate or resize when debug_vis is active.
        // Must use physical pixels: clip_pos in the shader is in physical pixels,
        // and viewport_width in ClipPlanesUniform (the buffer stride) is now physical too.
        {
            let vp_idx = frame.camera.viewport_index;
            let debug_active = frame.effects.lighting.debug_vis.active;
            let ppp = frame.camera.pixels_per_point;
            let vw = (frame.camera.viewport_size[0] * ppp).max(1.0) as u32;
            let vh = (frame.camera.viewport_size[1] * ppp).max(1.0) as u32;
            let slot = &self.viewport_slots[vp_idx];

            if debug_active && slot.debug_frag_dims != (vw, vh) {
                let size = (vw as u64) * (vh as u64) * 16;
                let new_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("debug_frag_buf"),
                    size: size.max(16),
                    usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let new_bg = self.resources.create_camera_bind_group(
                    device,
                    &self.viewport_slots[vp_idx].camera_buf,
                    &self.viewport_slots[vp_idx].clip_planes_buf,
                    &self.viewport_slots[vp_idx].shadow_info_buf,
                    &self.viewport_slots[vp_idx].clip_volume_buf,
                    &new_buf,
                    "per_viewport_camera_bg_debug",
                );
                self.viewport_slots[vp_idx].debug_frag_buf = Some(new_buf);
                self.viewport_slots[vp_idx].debug_frag_dims = (vw, vh);
                self.viewport_slots[vp_idx].camera_bind_group = new_bg;
            } else if !debug_active && slot.debug_frag_buf.is_some() {
                let sentinel = &self.resources.binds.debug_frag_sentinel_buf;
                let new_bg = self.resources.create_camera_bind_group(
                    device,
                    &self.viewport_slots[vp_idx].camera_buf,
                    &self.viewport_slots[vp_idx].clip_planes_buf,
                    &self.viewport_slots[vp_idx].shadow_info_buf,
                    &self.viewport_slots[vp_idx].clip_volume_buf,
                    sentinel,
                    "per_viewport_camera_bg",
                );
                self.viewport_slots[vp_idx].debug_frag_buf = None;
                self.viewport_slots[vp_idx].debug_frag_dims = (0, 0);
                self.viewport_slots[vp_idx].camera_bind_group = new_bg;
            }
        }
    }

    pub(super) fn prepare_atlas_blit(
        &mut self,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
    ) {
        // Atlas blit uniform: compute NDC rect for the corner overlay.
        if viewport_fx.show_shadow_atlas {
            let vw = frame.camera.viewport_size[0].max(1.0);
            let vh = frame.camera.viewport_size[1].max(1.0);
            let scale = viewport_fx.atlas_viewer_scale.clamp(0.05, 1.0);
            // Atlas is square. Width in NDC = scale * 2. Height preserves pixel aspect.
            let ndc_w = scale * 2.0;
            let ndc_h = ndc_w * (vw / vh);
            let margin_x = 20.0 / vw * 2.0;
            let margin_y = 20.0 / vh * 2.0;
            #[allow(unreachable_patterns)]
            let rect = match viewport_fx.atlas_viewer_corner {
                crate::renderer::types::debug::AtlasViewerCorner::BottomRight => {
                    let xmax = 1.0 - margin_x;
                    let ymin = -1.0 + margin_y;
                    [xmax - ndc_w, ymin, xmax, ymin + ndc_h]
                }
                crate::renderer::types::debug::AtlasViewerCorner::BottomLeft => {
                    let xmin = -1.0 + margin_x;
                    let ymin = -1.0 + margin_y;
                    [xmin, ymin, xmin + ndc_w, ymin + ndc_h]
                }
                crate::renderer::types::debug::AtlasViewerCorner::TopRight => {
                    let xmax = 1.0 - margin_x;
                    let ymax = 1.0 - margin_y;
                    [xmax - ndc_w, ymax - ndc_h, xmax, ymax]
                }
                crate::renderer::types::debug::AtlasViewerCorner::TopLeft => {
                    let xmin = -1.0 + margin_x;
                    let ymax = 1.0 - margin_y;
                    [xmin, ymax - ndc_h, xmin + ndc_w, ymax]
                }
                _ => [0.4, -1.0, 1.0, -0.4], // fallback for future variants
            };
            queue.write_buffer(
                &self.resources.shadow.atlas_viewer_buf,
                0,
                bytemuck::cast_slice(&[crate::resources::AtlasBlitUniform { rect }]),
            );
        }
    }
}
