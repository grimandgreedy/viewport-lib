use super::*;
use wgpu::util::DeviceExt;

/// Emit the 2D overlay draw calls shared by every paint path: SDF shapes,
/// rects, labels, scalar bars, rulers, loading bars, and overlay images, in
/// back-to-front order. Each block is guarded by its own prepared GPU data, so
/// a path with no data for a given overlay skips it. Run after all scene
/// content so the overlays sit on top. The overlay pipelines are format-neutral
/// (no separate LDR/HDR variant), so this is shared verbatim across paths.
macro_rules! emit_overlay_2d {
    ($this:ident, $render_pass:ident) => {{
        // SDF overlay shapes (drawn before rects and labels).
        if let Some(ref sd) = $this.overlay_shape_gpu_data {
            if sd.vertex_count > 0 {
                if let Some(pipeline) = &$this.resources.overlay_shape.pipeline {
                    if let Some(vbuf) = &sd.vertex_buf {
                        $render_pass.set_pipeline(pipeline);
                        $render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        $render_pass.draw(0..sd.vertex_count, 0..1);
                    }
                }
            }
            if !sd.tex_batches.is_empty() {
                if let Some(pipeline) = &$this.resources.overlay_shape.tex_pipeline {
                    $render_pass.set_pipeline(pipeline);
                    for batch in &sd.tex_batches {
                        $render_pass.set_bind_group(0, &batch.bind_group, &[]);
                        $render_pass.set_vertex_buffer(0, batch.vertex_buf.slice(..));
                        $render_pass.draw(0..batch.vertex_count, 0..1);
                    }
                }
            }
        }
        // Overlay rects (drawn before labels so they act as backgrounds).
        if let Some(ref rr) = $this.overlay_rect_gpu_data {
            if let Some(pipeline) = &$this.resources.overlay_text.pipeline {
                $render_pass.set_pipeline(pipeline);
                $render_pass.set_bind_group(0, &rr.bind_group, &[]);
                $render_pass.set_vertex_buffer(0, rr.vertex_buf.slice(..));
                $render_pass.draw(0..rr.vertex_count, 0..1);
            }
        }
        // Overlay labels (drawn after rects).
        if let Some(ref ld) = $this.label_gpu_data {
            if let Some(pipeline) = &$this.resources.overlay_text.pipeline {
                $render_pass.set_pipeline(pipeline);
                $render_pass.set_bind_group(0, &ld.bind_group, &[]);
                $render_pass.set_vertex_buffer(0, ld.vertex_buf.slice(..));
                $render_pass.draw(0..ld.vertex_count, 0..1);
            }
        }
        // Scalar bars (drawn after labels).
        if let Some(ref sb) = $this.scalar_bar_gpu_data {
            if let Some(pipeline) = &$this.resources.overlay_text.pipeline {
                $render_pass.set_pipeline(pipeline);
                $render_pass.set_bind_group(0, &sb.bind_group, &[]);
                $render_pass.set_vertex_buffer(0, sb.vertex_buf.slice(..));
                $render_pass.draw(0..sb.vertex_count, 0..1);
            }
        }
        // Rulers (drawn after scalar bars).
        if let Some(ref rd) = $this.ruler_gpu_data {
            if let Some(pipeline) = &$this.resources.overlay_text.pipeline {
                $render_pass.set_pipeline(pipeline);
                $render_pass.set_bind_group(0, &rd.bind_group, &[]);
                $render_pass.set_vertex_buffer(0, rd.vertex_buf.slice(..));
                $render_pass.draw(0..rd.vertex_count, 0..1);
            }
        }
        // Loading bars (drawn after rulers).
        if let Some(ref lb) = $this.loading_bar_gpu_data {
            if let Some(pipeline) = &$this.resources.overlay_text.pipeline {
                $render_pass.set_pipeline(pipeline);
                $render_pass.set_bind_group(0, &lb.bind_group, &[]);
                $render_pass.set_vertex_buffer(0, lb.vertex_buf.slice(..));
                $render_pass.draw(0..lb.vertex_count, 0..1);
            }
        }
        // Overlay images (drawn last, no depth test).
        if !$this.overlay_image_gpu_data.is_empty() {
            if let Some(pipeline) = &$this.resources.screen_image.pipeline {
                $render_pass.set_pipeline(pipeline);
                for gpu in &$this.overlay_image_gpu_data {
                    $render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                    $render_pass.draw(0..6, 0..1);
                }
            }
        }
    }};
}

mod hdr_path;
mod ldr_path;
mod paint_direct;

impl ViewportRenderer {
    /// Render the scene into an intermediate dyn-res texture for the LDR callback
    /// render path (e.g. eframe's `CallbackTrait`).
    ///
    /// Call from `CallbackTrait::prepare` after [`prepare`](Self::prepare), passing the
    /// `egui_encoder`. If `current_render_scale < 1.0`, the full scene is drawn into a
    /// scaled intermediate texture and `true` is returned. Call
    /// [`paint_dyn_res_blit`](Self::paint_dyn_res_blit) from `CallbackTrait::paint`
    /// instead of [`paint`](Self::paint).
    ///
    /// If scale is 1.0 or above, nothing is encoded and `false` is returned. Call
    /// [`paint`](Self::paint) as normal.
    ///
    /// The `egui_encoder` is submitted before the surface render pass begins, so the
    /// intermediate texture is fully written before the blit reads it.
    pub(crate) fn prepare_ldr_dyn_res(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        frame: &FrameData,
    ) -> bool {
        if self.current_render_scale >= 1.0 - 0.001 {
            return false;
        }

        let vp_idx = frame.camera.viewport_index;
        let w = (frame.camera.viewport_size[0] as u32).max(1);
        let h = (frame.camera.viewport_size[1] as u32).max(1);
        let sw = ((w as f32 * self.current_render_scale) as u32).max(1);
        let sh = ((h as f32 * self.current_render_scale) as u32).max(1);

        self.ensure_dyn_res_target(device, vp_idx, [sw, sh], [w, h]);
        self.resources.ensure_dyn_res_ds_pipeline(device);

        let bg_colour = frame.viewport.background_colour.unwrap_or([
            65.0 / 255.0,
            65.0 / 255.0,
            65.0 / 255.0,
            1.0,
        ]);

        {
            let slot = &self.viewport_slots[vp_idx];
            let dr = slot.dyn_res.as_ref().unwrap();
            let colour_view = &dr.colour_view;
            let depth_view = &dr.depth_view;
            let camera_bg = &slot.camera_bind_group;
            let grid_bg = &slot.grid_bind_group;

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ldr_dyn_res_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: colour_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg_colour[0] as f64,
                            g: bg_colour[1] as f64,
                            b: bg_colour[2] as f64,
                            a: bg_colour[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            emit_draw_calls!(
                &self.resources,
                &mut render_pass,
                frame,
                self.instancing.use_instancing,
                &self.instancing.batches,
                camera_bg,
                grid_bg,
                &self.compute_filter_results,
                Some(slot),
                &self.mesh_uniforms.wireframe_bind_groups,
                &self.mesh_uniforms.bind_groups,
                &self.prepared_surfaces
            );
            emit_scivis_draw_calls!(
                &self.resources,
                &mut render_pass,
                &self.point_cloud_gpu_data,
                &self.glyph_gpu_data,
                &self.polyline_gpu_data,
                &self.volume_gpu_data,
                &self.streamtube_gpu_data,
                camera_bg,
                &self.tube_gpu_data,
                &self.image_slice_gpu_data,
                &self.tensor_glyph_gpu_data,
                &self.ribbon_gpu_data,
                &self.volume_surface_slice_gpu_data,
                &self.sprite_gpu_data,
                &self.mesh_instance_gpu_data,
                false
            );
            // TransparentVolumeMesh boundary wireframe overlay.
            if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
                if let Some(ref tvm_bg) = self.mesh_uniforms.tvm_wireframe_bg {
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for mesh_id in &self.mesh_uniforms.tvm_wireframe_draws {
                        if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                            render_pass.set_pipeline(&self.resources.wireframe_pipeline);
                            render_pass.set_bind_group(
                                2,
                                &self.resources.deform.dummy_bind_group,
                                &[],
                            );
                            render_pass.set_bind_group(1, tvm_bg, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            render_pass.set_index_buffer(
                                mesh.edge_index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                        }
                    }
                }
            }
            // Implicit surface.
            if !self.implicit_gpu_data.is_empty() {
                if let Some(ref dual) = self.resources.implicit.pipeline {
                    render_pass.set_pipeline(dual.for_format(false));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for gpu in &self.implicit_gpu_data {
                        render_pass.set_bind_group(1, &gpu.bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
            // GPU marching cubes indirect draw.
            if !self.mc_gpu_data.is_empty() {
                if let Some(ref dual) = self.resources.mc.surface_pipeline {
                    render_pass.set_pipeline(dual.for_format(false));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for mc in &self.mc_gpu_data {
                        let vol = &self.resources.mc.volumes[mc.volume_idx];
                        render_pass.set_bind_group(1, &mc.render_bg, &[]);
                        for slab in &vol.slabs {
                            render_pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                            render_pass.draw_indirect(&slab.indirect_buf, 0);
                        }
                    }
                }
            }
            // Outline composite after all scene content.
            emit_outline_composite!(&self.resources, &mut render_pass, Some(slot));
            // Sub-object highlight (LDR path).
            if let Some(sub_hl) = slot.sub_highlight.as_ref() {
                if let (Some(fill_pl), Some(edge_pl), Some(sprite_pl)) = (
                    &self.resources.sub_highlight.fill_ldr_pipeline,
                    &self.resources.sub_highlight.edge_ldr_pipeline,
                    &self.resources.sub_highlight.sprite_ldr_pipeline,
                ) {
                    if sub_hl.fill_vertex_count > 0 {
                        render_pass.set_pipeline(fill_pl);
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        render_pass.set_bind_group(1, &sub_hl.fill_bind_group, &[]);
                        render_pass.set_vertex_buffer(0, sub_hl.fill_vertex_buf.slice(..));
                        render_pass.draw(0..sub_hl.fill_vertex_count, 0..1);
                    }
                    if sub_hl.edge_segment_count > 0 {
                        render_pass.set_pipeline(edge_pl);
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        render_pass.set_bind_group(1, &sub_hl.edge_bind_group, &[]);
                        render_pass.set_vertex_buffer(0, sub_hl.edge_vertex_buf.slice(..));
                        render_pass.draw(0..6, 0..sub_hl.edge_segment_count);
                    }
                    if sub_hl.sprite_point_count > 0 {
                        render_pass.set_pipeline(sprite_pl);
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        render_pass.set_bind_group(1, &sub_hl.sprite_bind_group, &[]);
                        render_pass.set_vertex_buffer(0, sub_hl.sprite_vertex_buf.slice(..));
                        render_pass.draw(0..6, 0..sub_hl.sprite_point_count);
                    }
                }
            }
            // Screen-space image overlays.
            if !self.screen_image_gpu_data.is_empty() {
                if let Some(pipeline) = &self.resources.screen_image.pipeline {
                    render_pass.set_pipeline(pipeline);
                    for gpu in &self.screen_image_gpu_data {
                        render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
            emit_overlay_2d!(self, render_pass);
        }

        true
    }

    /// Blit the dyn-res intermediate texture into the provided render pass.
    ///
    /// Call from `CallbackTrait::paint` when
    /// [`prepare_ldr_dyn_res`](Self::prepare_ldr_dyn_res) returned `true` for the same
    /// frame. Emits a fullscreen upscale quad into `render_pass`.
    pub(crate) fn paint_dyn_res_blit<'rp>(
        &self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        let vp_idx = frame.camera.viewport_index;
        if let Some(dr) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.dyn_res.as_ref())
        {
            if let Some(pipeline) = &self.resources.post.dyn_res_upscale_ds_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &dr.upscale_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
    }

    /// Run the full HDR pipeline (OIT, EDL, tone-map) for the eframe callback model.
    ///
    /// This is the HDR counterpart of
    /// [`prepare_ldr_dyn_res`](Self::prepare_ldr_dyn_res) for use when
    /// `frame.effects.post_process.enabled` is `true`.
    ///
    /// Internally this method:
    /// 1. Calls [`prepare`](Self::prepare) to upload uniforms and run the shadow pass.
    /// 2. Ensures a per-viewport intermediate texture at the viewport's native resolution.
    /// 3. Calls the full render pipeline (including OIT and EDL) into that texture.
    ///
    /// The returned [`wgpu::CommandBuffer`] must be returned from
    /// `CallbackTrait::prepare` so eframe submits it **before** the egui render pass.
    ///
    /// Call [`paint_hdr_blit`](Self::paint_hdr_blit) from `CallbackTrait::paint` to
    /// composite the intermediate texture into the egui render pass.
    pub(crate) fn prepare_hdr_callback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        self.prepare(device, queue, frame);

        let vp_idx = frame.camera.viewport_index;
        // Intermediate texture must be at physical pixel size so it matches the
        // HDR depth buffer allocated inside render_frame_internal (which also
        // uses physical pixels). Using logical size here produces a mismatch on
        // hidpi displays between the colour attachment (this texture) and the
        // depth attachment (hdr_depth_view) in the grid/overlay passes.
        let ppp = frame.camera.pixels_per_point;
        let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        // Ensure the blit pipeline (required by create_hdr_callback_target).
        self.resources.ensure_dyn_res_pipeline(device);
        self.resources.ensure_dyn_res_ds_pipeline(device);

        // Create or resize the per-viewport intermediate texture.
        self.ensure_viewport_slot(device, vp_idx);
        let needs_create = match self.viewport_slots[vp_idx].hdr_callback.as_ref() {
            None => true,
            Some(t) => t.size != [w, h],
        };
        if needs_create {
            let target = self.resources.create_hdr_callback_target(device, [w, h]);
            self.viewport_slots[vp_idx].hdr_callback = Some(target);
        }

        // Create a fresh TextureView from the stored Texture.
        // This owned view does not borrow viewport_slots, allowing the subsequent
        // mutable call to render_frame_internal without a borrow conflict.
        let output_view = self.viewport_slots[vp_idx]
            .hdr_callback
            .as_ref()
            .unwrap()
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_frame_internal(device, queue, &output_view, vp_idx, frame)
    }

    /// HDR encode for a single viewport in the multi-viewport eframe callback model.
    ///
    /// Like [`prepare_hdr_callback`](Self::prepare_hdr_callback) but skips the internal
    /// [`prepare`](Self::prepare) call. The caller must have already called
    /// [`prepare_scene`](Self::prepare_scene) and [`prepare_viewport`](Self::prepare_viewport)
    /// for `id` before invoking this.
    ///
    /// Multi-viewport HDR sequence:
    /// 1. Call `prepare_scene` once.
    /// 2. Call `prepare_viewport` for each viewport.
    /// 3. Call this method for each viewport; collect the returned `CommandBuffer`s.
    /// 4. Return them from `CallbackTrait::prepare`.
    ///
    /// Call [`paint_hdr_blit`](Self::paint_hdr_blit) for each viewport from
    /// `CallbackTrait::paint` with the scissor/viewport rect set first.
    pub(crate) fn prepare_hdr_callback_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ViewportId,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        let vp_idx = id.0;
        let ppp = frame.camera.pixels_per_point;
        let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        self.resources.ensure_dyn_res_pipeline(device);
        self.resources.ensure_dyn_res_ds_pipeline(device);

        self.ensure_viewport_slot(device, vp_idx);
        let needs_create = match self.viewport_slots[vp_idx].hdr_callback.as_ref() {
            None => true,
            Some(t) => t.size != [w, h],
        };
        if needs_create {
            let target = self.resources.create_hdr_callback_target(device, [w, h]);
            self.viewport_slots[vp_idx].hdr_callback = Some(target);
        }

        let output_view = self.viewport_slots[vp_idx]
            .hdr_callback
            .as_ref()
            .unwrap()
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_frame_internal(device, queue, &output_view, vp_idx, frame)
    }

    /// Blit the HDR intermediate texture into the egui render pass.
    ///
    /// Call from `CallbackTrait::paint` after
    /// [`prepare_hdr_callback`](Self::prepare_hdr_callback) has been called for the
    /// same frame and viewport. Emits a fullscreen triangle into `render_pass`.
    pub(crate) fn paint_hdr_blit<'rp>(
        &self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        let vp_idx = frame.camera.viewport_index;
        if let Some(hc) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.hdr_callback.as_ref())
        {
            if let Some(pipeline) = &self.resources.post.dyn_res_upscale_ds_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &hc.blit_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
        // Shadow atlas viewer overlay.
        if frame.effects.show_shadow_atlas {
            render_pass.set_pipeline(&self.resources.shadow_atlas_viewer_pipeline);
            render_pass.set_bind_group(0, &self.resources.shadow_atlas_viewer_bg, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }

    /// Like [`paint_hdr_blit`](Self::paint_hdr_blit) but for render passes without a
    /// depth-stencil attachment. Use this when you create the blit render pass yourself
    /// (e.g. winit) and omit the depth attachment.
    pub(crate) fn paint_hdr_blit_no_ds<'rp>(
        &self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        let vp_idx = frame.camera.viewport_index;
        if let Some(hc) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.hdr_callback.as_ref())
        {
            if let Some(pipeline) = &self.resources.post.dyn_res_upscale_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &hc.blit_bind_group, &[]);
                render_pass.draw(0..3, 0..1);
            }
        }
    }

    /// Unified prepare step for the eframe `CallbackTrait::prepare` method.
    ///
    /// Replaces manual `prepare` + `prepare_ldr_dyn_res` or `prepare_hdr_callback`
    /// calls. Dispatches internally based on `frame.effects.post_process.enabled`:
    ///
    /// - HDR path (`post_process.enabled = true`): runs the full HDR pipeline (OIT,
    ///   EDL, tone-map) and returns the resulting `CommandBuffer` for eframe to
    ///   submit before the egui render pass.
    /// - LDR path: calls `prepare`, and if dynamic resolution is active, encodes the
    ///   scene into a separate `CommandBuffer` (also submitted before the render
    ///   pass). Returns an empty `Vec` when dyn-res is inactive.
    ///
    /// Call [`paint_callback`](Self::paint_callback) from `CallbackTrait::paint`.
    pub(crate) fn prepare_callback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) -> Vec<wgpu::CommandBuffer> {
        if frame.effects.post_process.enabled {
            let cb = self.prepare_hdr_callback(device, queue, frame);
            vec![cb]
        } else {
            self.prepare(device, queue, frame);
            if self.current_render_scale < 1.0 - 0.001 {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ldr_dyn_res_callback_encoder"),
                });
                self.prepare_ldr_dyn_res(&mut encoder, device, frame);
                vec![encoder.finish()]
            } else {
                Vec::new()
            }
        }
    }

    /// Unified paint step for the eframe `CallbackTrait::paint` method.
    ///
    /// Call after [`prepare_callback`](Self::prepare_callback) for the same frame.
    /// Dispatches internally to `paint_hdr_blit`, `paint_dyn_res_blit`, or `paint`
    /// based on which path `prepare_callback` activated.
    pub(crate) fn paint_callback<'rp>(
        &self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        let vp_idx = frame.camera.viewport_index;
        if frame.effects.post_process.enabled {
            if self
                .viewport_slots
                .get(vp_idx)
                .and_then(|s| s.hdr_callback.as_ref())
                .is_some()
            {
                self.paint_hdr_blit(render_pass, frame);
                return;
            }
        }
        if self.current_render_scale < 1.0 - 0.001
            && self
                .viewport_slots
                .get(vp_idx)
                .and_then(|s| s.dyn_res.as_ref())
                .is_some()
        {
            self.paint_dyn_res_blit(render_pass, frame);
        } else {
            self.paint_to(render_pass, frame);
        }
    }

    /// High-level HDR render for a single viewport identified by `id`.
    ///
    /// Unlike [`render`](Self::render), this method does **not** call
    /// [`prepare`](Self::prepare) internally.  The caller must have already called
    /// [`prepare_scene`](Self::prepare_scene) and
    /// [`prepare_viewport`](Self::prepare_viewport) for `id` before invoking this.
    ///
    /// This is the right entry point for multi-viewport frames:
    /// 1. Call `prepare_scene` once.
    /// 2. Call `prepare_viewport` for each viewport.
    /// 3. Call `render_viewport` for each viewport with its own `output_view`.
    ///
    /// Returns a [`wgpu::CommandBuffer`] ready to submit.
    pub(crate) fn render_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        id: ViewportId,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        self.render_frame_internal(device, queue, output_view, id.0, frame)
    }

    /// High-level HDR render method. Handles the full post-processing pipeline:
    /// scene -> HDR texture -> (bloom) -> (SSAO) -> tone map -> output_view.
    ///
    /// When `frame.post_process.enabled` is false, falls back to a simple LDR render
    /// pass targeting `output_view` directly.
    ///
    /// Returns a `CommandBuffer` ready to submit.
    pub(crate) fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        // Always run prepare() to upload uniforms and run the shadow pass.
        self.prepare(device, queue, frame);
        self.render_frame_internal(
            device,
            queue,
            output_view,
            frame.camera.viewport_index,
            frame,
        )
    }

    /// Deferred-submit counterpart of [`render`](Self::render): runs prepare in
    /// deferred mode, encodes the scene pass, and returns every command buffer
    /// (prepare passes first, scene pass last) without submitting. The caller
    /// submits them, in order, on the device-driving thread.
    pub(crate) fn render_deferred(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        frame: &FrameData,
    ) -> Vec<wgpu::CommandBuffer> {
        let (_stats, mut buffers) = self.prepare_deferred(device, queue, frame);
        buffers.push(self.render_frame_internal(
            device,
            queue,
            output_view,
            frame.camera.viewport_index,
            frame,
        ));
        buffers
    }

    /// Render-only path shared by `render()` and `render_viewport()`.
    ///
    /// `vp_idx` selects the per-viewport slot to use for camera/HDR state,
    /// independent of `frame.camera.viewport_index`.
    fn render_frame_internal(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        vp_idx: usize,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        let paint_start = std::time::Instant::now();
        // Take the LOD-resolved surfaces from prepare (level mesh chosen, culled
        // items hidden), then extend with the boundary draws contributed by
        // opaque volume meshes (see the matching construction in `prepare.rs`).
        // Reading the resolved list rather than the raw `frame.scene.surfaces`
        // is what carries the LOD swap and cull into the HDR scene pass.
        let scene_items_owned: Vec<SceneRenderItem> = {
            let extra = frame
                .scene
                .volume_meshes
                .iter()
                .filter(|item| item.transparency.is_none())
                .map(|item| item.to_render_item());
            self.prepared_surfaces
                .iter()
                .cloned()
                .chain(extra)
                .collect()
        };
        let scene_items: &[SceneRenderItem] = &scene_items_owned;

        let bg_colour = frame.viewport.background_colour.unwrap_or([
            65.0 / 255.0,
            65.0 / 255.0,
            65.0 / 255.0,
            1.0,
        ]);
        let ppp = frame.camera.pixels_per_point;
        let w = (frame.camera.viewport_size[0] * ppp).round() as u32;
        let h = (frame.camera.viewport_size[1] * ppp).round() as u32;

        // Ensure per-viewport HDR targets. Provides a depth buffer for both LDR and HDR paths.
        let ssaa_factor = frame.effects.post_process.ssaa_factor.max(1);
        self.ensure_viewport_hdr(
            device,
            queue,
            vp_idx,
            w.max(1),
            h.max(1),
            ssaa_factor,
            self.current_render_scale,
        );

        // Lazy-initialize GPU timestamp resources on first render call when supported.
        if self.ts_query_set.is_none()
            && device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            // One begin/end timestamp pair per measured pass.
            let ts_count = 2 * super::GPU_TS_SLOTS;
            let ts_bytes = ts_count as u64 * 8;
            self.ts_query_set = Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("ts_query_set"),
                ty: wgpu::QueryType::Timestamp,
                count: ts_count,
            }));
            self.ts_resolve_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ts_resolve_buf"),
                size: ts_bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.ts_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ts_staging_buf"),
                size: ts_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            self.ts_period = queue.get_timestamp_period();
            // Gap-filling skipped slots before resolve needs encoder-level
            // timestamp writes. Without it we fall back to resolving the whole
            // range, which is undefined for unwritten queries on some drivers.
            self.ts_can_fill_gaps = device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        }

        let cmd_buf = if !frame.effects.post_process.enabled {
            self.render_frame_ldr(device, queue, output_view, vp_idx, frame, bg_colour, w, h)
        } else {
            self.render_frame_hdr(
                device,
                queue,
                output_view,
                vp_idx,
                frame,
                scene_items,
                bg_colour,
                w,
                h,
                ssaa_factor,
            )
        };
        // CPU time spent encoding the paint pass (draw-call recording), separate
        // from prepare. Latched so last_frame_stats() reflects it after render.
        self.last_stats.cpu_paint_ms = paint_start.elapsed().as_secs_f32() * 1000.0;
        cmd_buf
    }

    /// Render a frame into `output_view` and submit it, without reading anything
    /// back. `output_view` must be a `RENDER_ATTACHMENT` view in the format the
    /// renderer was created with, sized to `frame.camera.viewport_size`.
    ///
    /// Unlike [`render_offscreen`](Self::render_offscreen), this neither copies
    /// the result to the CPU nor blocks on the GPU, so the caller can reuse one
    /// target and drive many frames back to back. Intended for headless loops
    /// (perf measurement, capture pipelines) that read results back through GPU
    /// timestamps or their own copy on a cadence rather than every frame.
    pub fn render_to_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        frame: &FrameData,
    ) {
        let cmd_buf = self.render(device, queue, output_view, frame);
        queue.submit(std::iter::once(cmd_buf));
    }

    /// Encode a full frame (prepare passes + scene pass) into command buffers
    /// without submitting any of them: the deferred-submit counterpart of
    /// [`render_to_texture`](Self::render_to_texture). Buffers are returned in
    /// submission order (prepare first, scene pass last).
    ///
    /// A render worker can call this off the device-driving thread, then hand the
    /// buffers back for the driving thread to submit via
    /// [`submit_frame`](Self::submit_frame). Submitting from a non-driving thread
    /// corrupts some drivers (see [`crate::renderer::SubmitSink`]).
    pub fn render_to_texture_deferred(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        frame: &FrameData,
    ) -> Vec<wgpu::CommandBuffer> {
        self.render_deferred(device, queue, output_view, frame)
    }

    /// Submit a batch of command buffers produced by the `*_deferred` encoders,
    /// in order, on the device-driving thread. A no-op for an empty batch.
    pub fn submit_frame(queue: &wgpu::Queue, buffers: Vec<wgpu::CommandBuffer>) {
        if !buffers.is_empty() {
            queue.submit(buffers);
        }
    }

    /// Render a frame to an offscreen texture and return raw RGBA bytes.
    ///
    /// Creates a temporary [`wgpu::Texture`] render target of the given dimensions,
    /// runs all render passes (shadow, scene, post-processing) into it via
    /// [`render()`](Self::render), then copies the result back to CPU memory.
    ///
    /// No OS window or [`wgpu::Surface`] is required. The caller is responsible for
    /// initialising the wgpu adapter with `compatible_surface: None` and for
    /// constructing a valid [`FrameData`] (including `viewport_size` matching
    /// `width`/`height`).
    ///
    /// Returns `width * height * 4` bytes in RGBA8 layout. The caller encodes to
    /// PNG/EXR independently : no image codec dependency in this crate.
    pub fn render_offscreen(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        // 1. Create offscreen texture with RENDER_ATTACHMENT | COPY_SRC usage.
        let target_format = self.resources.target_format;
        let offscreen_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen_target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // 2. Create a texture view for rendering into.
        let output_view = offscreen_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // 3. render() calls ensure_viewport_hdr which provides the depth-stencil buffer
        //    for both LDR and HDR paths, so no separate ensure_outline_target is needed.

        // 4. Render the scene into the offscreen texture.
        //    The caller must set `frame.camera.viewport_size` to `[width as f32, height as f32]`
        //    and `frame.camera.render_camera.aspect` to `width as f32 / height as f32`
        //    for correct HDR target allocation and scissor rects.
        let cmd_buf = self.render(device, queue, &output_view, frame);
        queue.submit(std::iter::once(cmd_buf));

        // 5. Copy texture -> staging buffer (wgpu requires row alignment to 256 bytes).
        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) & !(align - 1);
        let buffer_size = (padded_row * height.max(1)) as u64;

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut copy_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen_copy_encoder"),
        });
        copy_encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &offscreen_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height.max(1)),
                },
            },
            wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(copy_encoder.finish()));

        // 6. Map buffer and extract tightly-packed RGBA pixels.
        let (tx, rx) = std::sync::mpsc::channel();
        staging_buf
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        let _ = rx.recv().unwrap_or(Err(wgpu::BufferAsyncError));

        let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        {
            let mapped = staging_buf.slice(..).get_mapped_range();
            let data: &[u8] = &mapped;
            if padded_row == unpadded_row {
                // No padding : copy entire slice directly.
                pixels.extend_from_slice(data);
            } else {
                // Strip row padding.
                for row in 0..height as usize {
                    let start = row * padded_row as usize;
                    let end = start + unpadded_row as usize;
                    pixels.extend_from_slice(&data[start..end]);
                }
            }
        }
        staging_buf.unmap();

        // 7. Swizzle BGRA -> RGBA if the format stores bytes in BGRA order.
        let is_bgra = matches!(
            target_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        if is_bgra {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2); // B <-> R
            }
        }

        pixels
    }

    // ------------------------------------------------------------------
    // Backdrop blur helpers
    // ------------------------------------------------------------------

    /// Ensure the backdrop blur state textures exist at the right size.
    fn ensure_backdrop_blur_state(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let need_recreate = match &self.backdrop_blur_state {
            Some(s) => s.size != [w, h] || s.format != self.resources.target_format,
            None => true,
        };
        if !need_recreate {
            return;
        }

        let format = self.resources.target_format;
        let blur_w = (w / 2).max(1);
        let blur_h = (h / 2).max(1);

        let intermediate_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("backdrop_intermediate"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let intermediate_view = intermediate_texture.create_view(&Default::default());

        let make_blur_tex = |label: &str| {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: blur_w,
                    height: blur_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            (t, v)
        };
        let (blur_a_texture, blur_a_view) = make_blur_tex("backdrop_blur_a");
        let (blur_b_texture, blur_b_view) = make_blur_tex("backdrop_blur_b");

        self.backdrop_blur_state = Some(crate::resources::BackdropBlurState {
            intermediate_texture,
            intermediate_view,
            blur_a_texture,
            blur_a_view,
            blur_b_texture,
            blur_b_view,
            size: [w, h],
            format,
        });
    }

    /// Run the backdrop blur pipeline: blit scene to half-res, then H blur, then V blur.
    /// Returns the bind group that can be used to draw blur overlay shapes with the
    /// texture pipeline.
    fn run_backdrop_blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        source_view: &wgpu::TextureView,
        spread: f32,
    ) -> wgpu::BindGroup {
        let bs = self.backdrop_blur_state.as_ref().unwrap();
        let blur_bgl = self.resources.backdrop_blur.bgl.as_ref().unwrap();
        let blur_sampler = self.resources.backdrop_blur.sampler.as_ref().unwrap();
        let blur_pipeline = self.resources.backdrop_blur.pipeline.as_ref().unwrap();
        // Reuse dyn_res blit pipeline and BGL for the downsample pass.
        let blit_pipeline = self
            .resources
            .post
            .dyn_res_upscale_pipeline
            .as_ref()
            .unwrap();
        let blit_bgl = self.resources.post.dyn_res_upscale_bgl.as_ref().unwrap();
        let blit_sampler = self.resources.post.dyn_res_linear_sampler.as_ref().unwrap();

        // Step 1: downsample source -> blur_a (half-res) using bilinear blit.
        let downsample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop_downsample_bg"),
            layout: blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(blit_sampler),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("backdrop_downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &bs.blur_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(blit_pipeline);
            pass.set_bind_group(0, &downsample_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Spread scaled for half-res: each texel covers 2 screen pixels.
        let effective_spread = (spread / 2.0).max(1.0);

        // Step 2: horizontal blur: blur_a -> blur_b.
        let h_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur_h_uniform"),
            contents: bytemuck::cast_slice(&[1u32, effective_spread.to_bits(), 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let h_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_h_bg"),
            layout: blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bs.blur_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(blur_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: h_uniform.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("backdrop_blur_h"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &bs.blur_b_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(blur_pipeline);
            pass.set_bind_group(0, &h_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Step 3: vertical blur: blur_b -> blur_a.
        let v_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur_v_uniform"),
            contents: bytemuck::cast_slice(&[0u32, effective_spread.to_bits(), 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let v_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blur_v_bg"),
            layout: blur_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bs.blur_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(blur_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_uniform.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("backdrop_blur_v"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &bs.blur_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(blur_pipeline);
            pass.set_bind_group(0, &v_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // Build the bind group for overlay shape drawing. Uses the overlay_shape_tex
        // bind group layout (texture + sampler) so blur shapes can be drawn with the
        // existing texture pipeline.
        let tex_bgl = self.resources.overlay_shape.tex_bgl.as_ref().unwrap();
        let tex_sampler = self.resources.overlay_shape.tex_sampler.as_ref().unwrap();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop_blur_overlay_bg"),
            layout: tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bs.blur_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(tex_sampler),
                },
            ],
        })
    }

    /// Returns true if the current frame has overlay shapes that need backdrop blur.
    fn has_backdrop_blur_shapes(&self) -> bool {
        self.overlay_shape_gpu_data
            .as_ref()
            .map_or(false, |sd| sd.blur_vertex_count > 0)
    }

    /// Draw blur overlay shapes into the given render pass using the texture pipeline.
    fn draw_blur_shapes<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        blur_bind_group: &'rp wgpu::BindGroup,
    ) {
        if let Some(ref sd) = self.overlay_shape_gpu_data {
            if sd.blur_vertex_count > 0 {
                if let (Some(pipeline), Some(vbuf)) = (
                    &self.resources.overlay_shape.tex_pipeline,
                    &sd.blur_vertex_buf,
                ) {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, blur_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                    render_pass.draw(0..sd.blur_vertex_count, 0..1);
                }
            }
        }
    }
}
