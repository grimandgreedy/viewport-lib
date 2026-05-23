use super::*;
use wgpu::util::DeviceExt;

impl ViewportRenderer {
    pub(crate) fn paint_to<'rp>(&self, render_pass: &mut wgpu::RenderPass<'rp>, frame: &FrameData) {
        let vp_idx = frame.camera.viewport_index;
        let camera_bg = self.viewport_camera_bind_group(vp_idx);
        let grid_bg = self.viewport_grid_bind_group(vp_idx);
        let vp_slot = self.viewport_slots.get(vp_idx);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            grid_bg,
            &self.compute_filter_results,
            vp_slot,
            &self.wireframe_bind_groups
        );
        emit_scivis_draw_calls!(
            &self.resources,
            &mut *render_pass,
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
            false
        );
        // Gaussian splats (alpha-blended, back-to-front sorted, no depth write).
        if !self.gaussian_splat_draw_data.is_empty() {
            if let Some(ref dual) = self.resources.gaussian_splat_pipeline {
                render_pass.set_pipeline(dual.for_format(false));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for dd in &self.gaussian_splat_draw_data {
                    if dd.wireframe {
                        continue;
                    }
                    if let Some(set) = self.resources.gaussian_splat_store.get(dd.store_index) {
                        if let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) {
                            render_pass.set_bind_group(1, &vp_sort.render_bg, &[]);
                            render_pass.draw(0..6, 0..dd.count);
                        }
                    }
                }
            }
        }
        // TransparentVolumeMesh boundary wireframe overlay.
        if !self.tvm_wireframe_draws.is_empty() {
            if let Some(ref tvm_bg) = self.tvm_wireframe_bg {
                render_pass.set_bind_group(0, camera_bg, &[]);
                for mesh_id in &self.tvm_wireframe_draws {
                    if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                        render_pass.set_pipeline(&self.resources.wireframe_pipeline);
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
        // Phase 16 : GPU implicit surface (depth-writes enabled, LessEqual compare).
        if !self.implicit_gpu_data.is_empty() {
            if let Some(ref dual) = self.resources.implicit_pipeline {
                render_pass.set_pipeline(dual.for_format(false));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for gpu in &self.implicit_gpu_data {
                    render_pass.set_bind_group(1, &gpu.bind_group, &[]);
                    render_pass.draw(0..6, 0..1);
                }
            }
        }
        // Phase 17 : GPU marching cubes indirect draw.
        if !self.mc_gpu_data.is_empty() {
            render_pass.set_bind_group(0, camera_bg, &[]);
            for mc in &self.mc_gpu_data {
                let vol = &self.resources.mc_volumes[mc.volume_idx];
                if mc.wireframe || frame.viewport.wireframe_mode {
                    if let Some(ref dual) = self.resources.mc_wireframe_pipeline {
                        render_pass.set_pipeline(dual.for_format(false));
                        for (slab, wire_bg) in
                            vol.slabs.iter().zip(mc.wire_slab_bgs.iter())
                        {
                            render_pass.set_bind_group(1, wire_bg, &[]);
                            render_pass.draw_indirect(&slab.wire_indirect_buf, 0);
                        }
                    }
                } else if let Some(ref dual) = self.resources.mc_surface_pipeline {
                    render_pass.set_pipeline(dual.for_format(false));
                    render_pass.set_bind_group(1, &mc.render_bg, &[]);
                    for slab in &vol.slabs {
                        render_pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                        render_pass.draw_indirect(&slab.indirect_buf, 0);
                    }
                }
            }
        }
        // Outline composite after all scene content so translucent layers don't overdraw.
        emit_outline_composite!(&self.resources, &mut *render_pass, vp_slot);
        // Sub-object highlight (LDR path) : face fill, edge lines, vertex/point sprites.
        if let Some(sub_hl) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.sub_highlight.as_ref())
        {
            if let (Some(fill_pl), Some(edge_pl), Some(sprite_pl)) = (
                &self.resources.sub_highlight_fill_ldr_pipeline,
                &self.resources.sub_highlight_edge_ldr_pipeline,
                &self.resources.sub_highlight_sprite_ldr_pipeline,
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
        // Phase 10B : screen-space image overlays (always on top, no depth test).
        if !self.screen_image_gpu_data.is_empty() {
            if let Some(pipeline) = &self.resources.screen_image_pipeline {
                render_pass.set_pipeline(pipeline);
                for gpu in &self.screen_image_gpu_data {
                    render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                    render_pass.draw(0..6, 0..1);
                }
            }
        }
        // SDF overlay shapes (drawn before rects and labels).
        if let Some(ref sd) = self.overlay_shape_gpu_data {
            if sd.vertex_count > 0 {
                if let Some(pipeline) = &self.resources.overlay_shape_pipeline {
                    if let Some(vbuf) = &sd.vertex_buf {
                        render_pass.set_pipeline(pipeline);
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass.draw(0..sd.vertex_count, 0..1);
                    }
                }
            }
            if !sd.tex_batches.is_empty() {
                if let Some(pipeline) = &self.resources.overlay_shape_tex_pipeline {
                    render_pass.set_pipeline(pipeline);
                    for batch in &sd.tex_batches {
                        render_pass.set_bind_group(0, &batch.bind_group, &[]);
                        render_pass.set_vertex_buffer(0, batch.vertex_buf.slice(..));
                        render_pass.draw(0..batch.vertex_count, 0..1);
                    }
                }
            }
        }
        // Overlay rects (drawn before labels so they act as backgrounds).
        if let Some(ref rr) = self.overlay_rect_gpu_data {
            if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &rr.bind_group, &[]);
                render_pass.set_vertex_buffer(0, rr.vertex_buf.slice(..));
                render_pass.draw(0..rr.vertex_count, 0..1);
            }
        }
        // Overlay labels (always on top, after screen images).
        if let Some(ref ld) = self.label_gpu_data {
            if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &ld.bind_group, &[]);
                render_pass.set_vertex_buffer(0, ld.vertex_buf.slice(..));
                render_pass.draw(0..ld.vertex_count, 0..1);
            }
        }
        // Scalar bars (drawn after labels).
        if let Some(ref sb) = self.scalar_bar_gpu_data {
            if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &sb.bind_group, &[]);
                render_pass.set_vertex_buffer(0, sb.vertex_buf.slice(..));
                render_pass.draw(0..sb.vertex_count, 0..1);
            }
        }
        // Rulers (drawn after scalar bars).
        if let Some(ref rd) = self.ruler_gpu_data {
            if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &rd.bind_group, &[]);
                render_pass.set_vertex_buffer(0, rd.vertex_buf.slice(..));
                render_pass.draw(0..rd.vertex_count, 0..1);
            }
        }
        // Loading bars (drawn after rulers).
        if let Some(ref lb) = self.loading_bar_gpu_data {
            if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &lb.bind_group, &[]);
                render_pass.set_vertex_buffer(0, lb.vertex_buf.slice(..));
                render_pass.draw(0..lb.vertex_count, 0..1);
            }
        }
        // Phase 7 : overlay images (OverlayFrame, drawn last, no depth test).
        if !self.overlay_image_gpu_data.is_empty() {
            if let Some(pipeline) = &self.resources.screen_image_pipeline {
                render_pass.set_pipeline(pipeline);
                for gpu in &self.overlay_image_gpu_data {
                    render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                    render_pass.draw(0..6, 0..1);
                }
            }
        }
    }

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
                self.use_instancing,
                &self.instanced_batches,
                camera_bg,
                grid_bg,
                &self.compute_filter_results,
                Some(slot),
                &self.wireframe_bind_groups
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
                false
            );
            // TransparentVolumeMesh boundary wireframe overlay.
            if !self.tvm_wireframe_draws.is_empty() {
                if let Some(ref tvm_bg) = self.tvm_wireframe_bg {
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for mesh_id in &self.tvm_wireframe_draws {
                        if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                            render_pass.set_pipeline(&self.resources.wireframe_pipeline);
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
                if let Some(ref dual) = self.resources.implicit_pipeline {
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
                if let Some(ref dual) = self.resources.mc_surface_pipeline {
                    render_pass.set_pipeline(dual.for_format(false));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for mc in &self.mc_gpu_data {
                        let vol = &self.resources.mc_volumes[mc.volume_idx];
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
                    &self.resources.sub_highlight_fill_ldr_pipeline,
                    &self.resources.sub_highlight_edge_ldr_pipeline,
                    &self.resources.sub_highlight_sprite_ldr_pipeline,
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
                if let Some(pipeline) = &self.resources.screen_image_pipeline {
                    render_pass.set_pipeline(pipeline);
                    for gpu in &self.screen_image_gpu_data {
                        render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
            // SDF overlay shapes (drawn before rects and labels).
            if let Some(ref sd) = self.overlay_shape_gpu_data {
                if sd.vertex_count > 0 {
                    if let Some(pipeline) = &self.resources.overlay_shape_pipeline {
                        if let Some(vbuf) = &sd.vertex_buf {
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_vertex_buffer(0, vbuf.slice(..));
                            render_pass.draw(0..sd.vertex_count, 0..1);
                        }
                    }
                }
                if !sd.tex_batches.is_empty() {
                    if let Some(pipeline) = &self.resources.overlay_shape_tex_pipeline {
                        render_pass.set_pipeline(pipeline);
                        for batch in &sd.tex_batches {
                            render_pass.set_bind_group(0, &batch.bind_group, &[]);
                            render_pass.set_vertex_buffer(0, batch.vertex_buf.slice(..));
                            render_pass.draw(0..batch.vertex_count, 0..1);
                        }
                    }
                }
            }
            // Overlay rects (drawn before labels so they act as backgrounds).
            if let Some(ref rr) = self.overlay_rect_gpu_data {
                if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, &rr.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, rr.vertex_buf.slice(..));
                    render_pass.draw(0..rr.vertex_count, 0..1);
                }
            }
            // Overlay labels.
            if let Some(ref ld) = self.label_gpu_data {
                if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, &ld.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, ld.vertex_buf.slice(..));
                    render_pass.draw(0..ld.vertex_count, 0..1);
                }
            }
            // Scalar bars.
            if let Some(ref sb) = self.scalar_bar_gpu_data {
                if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, &sb.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, sb.vertex_buf.slice(..));
                    render_pass.draw(0..sb.vertex_count, 0..1);
                }
            }
            // Rulers.
            if let Some(ref rd) = self.ruler_gpu_data {
                if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, &rd.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, rd.vertex_buf.slice(..));
                    render_pass.draw(0..rd.vertex_count, 0..1);
                }
            }
            // Loading bars.
            if let Some(ref lb) = self.loading_bar_gpu_data {
                if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, &lb.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, lb.vertex_buf.slice(..));
                    render_pass.draw(0..lb.vertex_count, 0..1);
                }
            }
            // Overlay images (drawn last).
            if !self.overlay_image_gpu_data.is_empty() {
                if let Some(pipeline) = &self.resources.screen_image_pipeline {
                    render_pass.set_pipeline(pipeline);
                    for gpu in &self.overlay_image_gpu_data {
                        render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
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
            if let Some(pipeline) = &self.resources.dyn_res_upscale_ds_pipeline {
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

    /// Blit the HDR intermediate texture into the egui render pass.
    ///
    /// Call from `CallbackTrait::paint` after
    /// [`prepare_hdr_callback`](Self::prepare_hdr_callback) has been called for the
    /// same frame and viewport. Emits a fullscreen triangle into `render_pass`.
    pub(crate) fn paint_hdr_blit<'rp>(&self, render_pass: &mut wgpu::RenderPass<'rp>, frame: &FrameData) {
        let vp_idx = frame.camera.viewport_index;
        if let Some(hc) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.hdr_callback.as_ref())
        {
            if let Some(pipeline) = &self.resources.dyn_res_upscale_ds_pipeline {
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
    pub(crate) fn paint_callback<'rp>(&self, render_pass: &mut wgpu::RenderPass<'rp>, frame: &FrameData) {
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
        // Read scene items from the surface submission.
        let scene_items: &[SceneRenderItem] = match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items.as_ref(),
        };

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
        self.ensure_viewport_hdr(device, queue, vp_idx, w.max(1), h.max(1), ssaa_factor, self.current_render_scale);

        // Phase 4 : lazy-initialize GPU timestamp resources on first render call when supported.
        if self.ts_query_set.is_none()
            && device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            self.ts_query_set = Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("ts_query_set"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }));
            self.ts_resolve_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ts_resolve_buf"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.ts_staging_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ts_staging_buf"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            self.ts_period = queue.get_timestamp_period();
        }

        if !frame.effects.post_process.enabled {
            // LDR fallback. When dynamic resolution is active and render_scale < 1.0,
            // draw into a scaled intermediate texture and upscale-blit to output_view.
            // Otherwise render directly to output_view.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ldr_encoder"),
            });

            let use_dyn_res = self.current_render_scale < 1.0 - 0.001;
            let needs_blur = self.has_backdrop_blur_shapes();

            if use_dyn_res {
                let sw = ((w as f32 * self.current_render_scale) as u32).max(1);
                let sh = ((h as f32 * self.current_render_scale) as u32).max(1);
                self.ensure_dyn_res_target(device, vp_idx, [sw, sh], [w.max(1), h.max(1)]);
            }

            // When blur backdrops are needed and we'd otherwise render directly
            // to the surface (no dyn_res), force an intermediate so it can be
            // sampled for the blur passes.
            if needs_blur && !use_dyn_res {
                self.ensure_backdrop_blur_state(device, w.max(1), h.max(1));
            }

            {
                let slot = &self.viewport_slots[vp_idx];
                let slot_hdr = slot.hdr.as_ref().expect(
                    "HDR state missing in LDR path; ensure_viewport_hdr must have been called",
                );
                let camera_bg = &slot.camera_bind_group;
                let grid_bg = &slot.grid_bind_group;
                // Choose render target: dyn_res intermediate, backdrop intermediate, or output_view.
                let (scene_colour_view, scene_depth_view): (&wgpu::TextureView, &wgpu::TextureView) =
                    if use_dyn_res {
                        let dr = slot.dyn_res.as_ref().unwrap();
                        (&dr.colour_view, &dr.depth_view)
                    } else if needs_blur {
                        let bs = self.backdrop_blur_state.as_ref().unwrap();
                        (&bs.intermediate_view, &slot_hdr.outline_depth_view)
                    } else {
                        (output_view, &slot_hdr.outline_depth_view)
                    };
                let ts_writes =
                    self.ts_query_set
                        .as_ref()
                        .map(|qs| wgpu::RenderPassTimestampWrites {
                            query_set: qs,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        });
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ldr_render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: scene_colour_view,
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
                        view: scene_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: ts_writes,
                    occlusion_query_set: None,
                });
                emit_draw_calls!(
                    &self.resources,
                    &mut render_pass,
                    frame,
                    self.use_instancing,
                    &self.instanced_batches,
                    camera_bg,
                    grid_bg,
                    &self.compute_filter_results,
                    Some(slot),
                    &self.wireframe_bind_groups
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
                    false
                );
                // TransparentVolumeMesh boundary wireframe overlay.
                if !self.tvm_wireframe_draws.is_empty() {
                    if let Some(ref tvm_bg) = self.tvm_wireframe_bg {
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        for mesh_id in &self.tvm_wireframe_draws {
                            if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                                render_pass.set_pipeline(&self.resources.wireframe_pipeline);
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
                // Phase 16 : GPU implicit surface.
                if !self.implicit_gpu_data.is_empty() {
                    if let Some(ref dual) = self.resources.implicit_pipeline {
                        render_pass.set_pipeline(dual.for_format(false));
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        for gpu in &self.implicit_gpu_data {
                            render_pass.set_bind_group(1, &gpu.bind_group, &[]);
                            render_pass.draw(0..6, 0..1);
                        }
                    }
                }
                // Phase 17 : GPU marching cubes indirect draw.
                if !self.mc_gpu_data.is_empty() {
                    if let Some(ref dual) = self.resources.mc_surface_pipeline {
                        render_pass.set_pipeline(dual.for_format(false));
                        render_pass.set_bind_group(0, camera_bg, &[]);
                        for mc in &self.mc_gpu_data {
                            let vol = &self.resources.mc_volumes[mc.volume_idx];
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
                // Phase 10B / Phase 12 : screen-space image overlays.
                // Regular items drawn with depth_compare: Always (always on top).
                // Depth-composite items drawn with depth_compare: LessEqual (occluded by
                // scene geometry whose depth was already written to the depth attachment).
                if !self.screen_image_gpu_data.is_empty() {
                    if let Some(overlay_pipeline) = &self.resources.screen_image_pipeline {
                        let dc_pipeline = self.resources.screen_image_dc_pipeline.as_ref();
                        for gpu in &self.screen_image_gpu_data {
                            if let (Some(dc_bg), Some(dc_pipe)) =
                                (&gpu.depth_bind_group, dc_pipeline)
                            {
                                render_pass.set_pipeline(dc_pipe);
                                render_pass.set_bind_group(0, dc_bg, &[]);
                            } else {
                                render_pass.set_pipeline(overlay_pipeline);
                                render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                            }
                            render_pass.draw(0..6, 0..1);
                        }
                    }
                }
                // When blur backdrops are needed, skip overlays here. They'll
                // be drawn in a second pass after the blur is applied.
                if !needs_blur {
                    // SDF overlay shapes (LDR fallback).
                    if let Some(ref sd) = self.overlay_shape_gpu_data {
                        if sd.vertex_count > 0 {
                            if let Some(pipeline) = &self.resources.overlay_shape_pipeline {
                                if let Some(vbuf) = &sd.vertex_buf {
                                    render_pass.set_pipeline(pipeline);
                                    render_pass.set_vertex_buffer(0, vbuf.slice(..));
                                    render_pass.draw(0..sd.vertex_count, 0..1);
                                }
                            }
                        }
                        if !sd.tex_batches.is_empty() {
                            if let Some(pipeline) = &self.resources.overlay_shape_tex_pipeline {
                                render_pass.set_pipeline(pipeline);
                                for batch in &sd.tex_batches {
                                    render_pass.set_bind_group(0, &batch.bind_group, &[]);
                                    render_pass.set_vertex_buffer(0, batch.vertex_buf.slice(..));
                                    render_pass.draw(0..batch.vertex_count, 0..1);
                                }
                            }
                        }
                    }
                    // Overlay rects (LDR fallback).
                    if let Some(ref rr) = self.overlay_rect_gpu_data {
                        if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(0, &rr.bind_group, &[]);
                            render_pass.set_vertex_buffer(0, rr.vertex_buf.slice(..));
                            render_pass.draw(0..rr.vertex_count, 0..1);
                        }
                    }
                    // Overlay labels (LDR fallback: inside the same render pass).
                    if let Some(ref ld) = self.label_gpu_data {
                        if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(0, &ld.bind_group, &[]);
                            render_pass.set_vertex_buffer(0, ld.vertex_buf.slice(..));
                            render_pass.draw(0..ld.vertex_count, 0..1);
                        }
                    }
                    // Scalar bars (LDR fallback).
                    if let Some(ref sb) = self.scalar_bar_gpu_data {
                        if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(0, &sb.bind_group, &[]);
                            render_pass.set_vertex_buffer(0, sb.vertex_buf.slice(..));
                            render_pass.draw(0..sb.vertex_count, 0..1);
                        }
                    }
                    // Rulers (LDR fallback).
                    if let Some(ref rd) = self.ruler_gpu_data {
                        if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                            render_pass.set_pipeline(pipeline);
                            render_pass.set_bind_group(0, &rd.bind_group, &[]);
                            render_pass.set_vertex_buffer(0, rd.vertex_buf.slice(..));
                            render_pass.draw(0..rd.vertex_count, 0..1);
                        }
                    }
                    // Phase 7 : overlay images (OverlayFrame, LDR fallback, drawn last).
                    if !self.overlay_image_gpu_data.is_empty() {
                        if let Some(pipeline) = &self.resources.screen_image_pipeline {
                            render_pass.set_pipeline(pipeline);
                            for gpu in &self.overlay_image_gpu_data {
                                render_pass.set_bind_group(0, &gpu.bind_group, &[]);
                                render_pass.draw(0..6, 0..1);
                            }
                        }
                    }
                }
            }
            // -- End of scene render pass (dropped above). ---

            // Backdrop blur: capture scene, run blur, then draw overlays in a
            // second render pass so blur shapes can sample the blurred result.
            if needs_blur {
                let spread = self.overlay_shape_gpu_data.as_ref()
                    .map(|d| d.max_blur_radius)
                    .unwrap_or(1.0);
                let blur_bg = {
                    let source = if use_dyn_res {
                        &self.viewport_slots[vp_idx].dyn_res.as_ref().unwrap().colour_view
                    } else {
                        &self.backdrop_blur_state.as_ref().unwrap().intermediate_view
                    };
                    self.run_backdrop_blur(&mut encoder, device, queue, source, spread)
                };

                // Second render pass for overlays (Load to preserve scene content).
                let slot = &self.viewport_slots[vp_idx];
                let slot_hdr = slot.hdr.as_ref().unwrap();
                let overlay_colour_view: &wgpu::TextureView = if use_dyn_res {
                    &slot.dyn_res.as_ref().unwrap().colour_view
                } else {
                    &self.backdrop_blur_state.as_ref().unwrap().intermediate_view
                };
                let overlay_depth_view: &wgpu::TextureView = if use_dyn_res {
                    &slot.dyn_res.as_ref().unwrap().depth_view
                } else {
                    &slot_hdr.outline_depth_view
                };
                {
                    let mut overlay_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ldr_overlay_blur_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: overlay_colour_view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: overlay_depth_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Discard }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    // Draw blur backdrop shapes first.
                    self.draw_blur_shapes(&mut overlay_pass, &blur_bg);
                    // Then normal shapes.
                    if let Some(ref sd) = self.overlay_shape_gpu_data {
                        if sd.vertex_count > 0 {
                            if let Some(pipeline) = &self.resources.overlay_shape_pipeline {
                                if let Some(vbuf) = &sd.vertex_buf {
                                    overlay_pass.set_pipeline(pipeline);
                                    overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                                    overlay_pass.draw(0..sd.vertex_count, 0..1);
                                }
                            }
                        }
                        if !sd.tex_batches.is_empty() {
                            if let Some(pipeline) = &self.resources.overlay_shape_tex_pipeline {
                                overlay_pass.set_pipeline(pipeline);
                                for batch in &sd.tex_batches {
                                    overlay_pass.set_bind_group(0, &batch.bind_group, &[]);
                                    overlay_pass.set_vertex_buffer(0, batch.vertex_buf.slice(..));
                                    overlay_pass.draw(0..batch.vertex_count, 0..1);
                                }
                            }
                        }
                    }
                    if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                        if let Some(ref rr) = self.overlay_rect_gpu_data {
                            overlay_pass.set_pipeline(pipeline);
                            overlay_pass.set_bind_group(0, &rr.bind_group, &[]);
                            overlay_pass.set_vertex_buffer(0, rr.vertex_buf.slice(..));
                            overlay_pass.draw(0..rr.vertex_count, 0..1);
                        }
                        if let Some(ref ld) = self.label_gpu_data {
                            overlay_pass.set_pipeline(pipeline);
                            overlay_pass.set_bind_group(0, &ld.bind_group, &[]);
                            overlay_pass.set_vertex_buffer(0, ld.vertex_buf.slice(..));
                            overlay_pass.draw(0..ld.vertex_count, 0..1);
                        }
                        if let Some(ref sb) = self.scalar_bar_gpu_data {
                            overlay_pass.set_pipeline(pipeline);
                            overlay_pass.set_bind_group(0, &sb.bind_group, &[]);
                            overlay_pass.set_vertex_buffer(0, sb.vertex_buf.slice(..));
                            overlay_pass.draw(0..sb.vertex_count, 0..1);
                        }
                        if let Some(ref rd) = self.ruler_gpu_data {
                            overlay_pass.set_pipeline(pipeline);
                            overlay_pass.set_bind_group(0, &rd.bind_group, &[]);
                            overlay_pass.set_vertex_buffer(0, rd.vertex_buf.slice(..));
                            overlay_pass.draw(0..rd.vertex_count, 0..1);
                        }
                    }
                    if !self.overlay_image_gpu_data.is_empty() {
                        if let Some(pipeline) = &self.resources.screen_image_pipeline {
                            overlay_pass.set_pipeline(pipeline);
                            for gpu in &self.overlay_image_gpu_data {
                                overlay_pass.set_bind_group(0, &gpu.bind_group, &[]);
                                overlay_pass.draw(0..6, 0..1);
                            }
                        }
                    }
                }
            }

            // Phase 4 : resolve timestamp queries -> staging buffer.
            if let (Some(qs), Some(res_buf), Some(stg_buf)) = (
                self.ts_query_set.as_ref(),
                self.ts_resolve_buf.as_ref(),
                self.ts_staging_buf.as_ref(),
            ) {
                encoder.resolve_query_set(qs, 0..2, res_buf, 0);
                encoder.copy_buffer_to_buffer(res_buf, 0, stg_buf, 0, 16);
                self.ts_needs_readback = true;
            }

            // Phase 3 : upscale blit from dyn_res intermediate to output_view.
            if use_dyn_res {
                let upscale_bg = &self.viewport_slots[vp_idx]
                    .dyn_res
                    .as_ref()
                    .unwrap()
                    .upscale_bind_group;
                let mut upscale_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("dyn_res_upscale_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                if let Some(pipeline) = &self.resources.dyn_res_upscale_pipeline {
                    upscale_pass.set_pipeline(pipeline);
                    upscale_pass.set_bind_group(0, upscale_bg, &[]);
                    upscale_pass.draw(0..3, 0..1);
                }
            } else if needs_blur {
                // Blit backdrop intermediate to the output surface.
                let bs = self.backdrop_blur_state.as_ref().unwrap();
                let blit_bgl = self.resources.dyn_res_upscale_bgl.as_ref().unwrap();
                let blit_sampler = self.resources.dyn_res_linear_sampler.as_ref().unwrap();
                let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("backdrop_blit_bg"),
                    layout: blit_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bs.intermediate_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(blit_sampler) },
                    ],
                });
                let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("backdrop_blit_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                if let Some(pipeline) = &self.resources.dyn_res_upscale_pipeline {
                    blit_pass.set_pipeline(pipeline);
                    blit_pass.set_bind_group(0, &blit_bg, &[]);
                    blit_pass.draw(0..3, 0..1);
                }
            }

            return encoder.finish();
        }

        // HDR path.
        let pp = &frame.effects.post_process;

        let hdr_clear_rgb = [
            bg_colour[0].powf(2.2),
            bg_colour[1].powf(2.2),
            bg_colour[2].powf(2.2),
        ];

        // Upload tone map uniform into the per-viewport buffer.
        let mode = match pp.tone_mapping {
            crate::renderer::ToneMapping::Reinhard => 0u32,
            crate::renderer::ToneMapping::Aces => 1u32,
            crate::renderer::ToneMapping::KhronosNeutral => 2u32,
        };
        let tm_uniform = crate::resources::ToneMapUniform {
            exposure: pp.exposure,
            mode,
            bloom_enabled: if pp.bloom { 1 } else { 0 },
            ssao_enabled: if pp.ssao { 1 } else { 0 },
            contact_shadows_enabled: if pp.contact_shadows { 1 } else { 0 },
            edl_enabled: if pp.edl_enabled { 1 } else { 0 },
            edl_radius: pp.edl_radius,
            edl_strength: pp.edl_strength,
            background_colour: bg_colour,
            near_plane: frame.camera.render_camera.near,
            far_plane: frame.camera.render_camera.far,
            lic_enabled: if frame.scene.lic_items.is_empty() {
                0
            } else {
                1
            },
            lic_strength: frame
                .scene
                .lic_items
                .first()
                .map(|i| i.config.strength)
                .unwrap_or(0.5),
        };
        {
            let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            queue.write_buffer(
                &hdr.tone_map_uniform_buf,
                0,
                bytemuck::cast_slice(&[tm_uniform]),
            );

            // Upload SSAO uniform if needed.
            if pp.ssao {
                let proj = frame.camera.render_camera.projection;
                let inv_proj = proj.inverse();
                let ssao_uniform = crate::resources::SsaoUniform {
                    inv_proj: inv_proj.to_cols_array_2d(),
                    proj: proj.to_cols_array_2d(),
                    radius: 0.5,
                    bias: 0.025,
                    _pad: [0.0; 2],
                };
                queue.write_buffer(
                    &hdr.ssao_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[ssao_uniform]),
                );
            }

            // Upload contact shadow uniform if needed.
            if pp.contact_shadows {
                let proj = frame.camera.render_camera.projection;
                let inv_proj = proj.inverse();
                let light_dir_world: glam::Vec3 =
                    if let Some(l) = frame.effects.lighting.lights.first() {
                        match l.kind {
                            LightKind::Directional { direction } => {
                                glam::Vec3::from(direction).normalize()
                            }
                            LightKind::Spot { direction, .. } => {
                                glam::Vec3::from(direction).normalize()
                            }
                            _ => glam::Vec3::new(0.0, -1.0, 0.0),
                        }
                    } else {
                        glam::Vec3::new(0.0, -1.0, 0.0)
                    };
                let view = frame.camera.render_camera.view;
                let light_dir_view = view.transform_vector3(light_dir_world).normalize();
                let world_up_view = view.transform_vector3(glam::Vec3::Z).normalize();
                let cs_uniform = crate::resources::ContactShadowUniform {
                    inv_proj: inv_proj.to_cols_array_2d(),
                    proj: proj.to_cols_array_2d(),
                    light_dir_view: [light_dir_view.x, light_dir_view.y, light_dir_view.z, 0.0],
                    world_up_view: [world_up_view.x, world_up_view.y, world_up_view.z, 0.0],
                    params: [
                        pp.contact_shadow_max_distance,
                        pp.contact_shadow_steps as f32,
                        pp.contact_shadow_thickness,
                        0.0,
                    ],
                };
                queue.write_buffer(
                    &hdr.contact_shadow_uniform_buf,
                    0,
                    bytemuck::cast_slice(&[cs_uniform]),
                );
            }

            // Upload bloom uniform if needed.
            if pp.bloom {
                let bloom_u = crate::resources::BloomUniform {
                    threshold: pp.bloom_threshold,
                    intensity: pp.bloom_intensity,
                    horizontal: 0,
                    _pad: 0,
                };
                queue.write_buffer(&hdr.bloom_uniform_buf, 0, bytemuck::cast_slice(&[bloom_u]));
            }
        }

        // Upload DoF uniform when enabled.
        if pp.dof_enabled {
            let (w, h) = {
                let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                (hdr.scene_size[0] as f32, hdr.scene_size[1] as f32)
            };
            let dof_uniform = crate::resources::DofUniform {
                focal_distance: pp.dof_focal_distance,
                focal_range: pp.dof_focal_range,
                max_blur_radius: pp.dof_max_blur_radius,
                near_plane: frame.camera.render_camera.near,
                far_plane: frame.camera.render_camera.far,
                viewport_width: w,
                viewport_height: h,
                _pad: 0.0,
            };
            let hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            queue.write_buffer(
                &hdr.dof_uniform_buf,
                0,
                bytemuck::cast_slice(&[dof_uniform]),
            );
        }

        // Rebuild tone-map bind group with correct bloom/AO/DoF texture views.
        {
            let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
            self.resources.rebuild_tone_map_bind_group(
                device,
                hdr,
                pp.bloom,
                pp.ssao,
                pp.contact_shadows,
                !frame.scene.lic_items.is_empty(),
                pp.dof_enabled,
            );
        }

        // -----------------------------------------------------------------------
        // Pre-allocate OIT targets if any transparent items exist.
        // Must happen before camera_bg is borrowed (borrow-checker constraint).
        // -----------------------------------------------------------------------
        {
            let needs_oit = if self.use_instancing && !self.instanced_batches.is_empty() {
                self.instanced_batches.iter().any(|b| b.is_transparent)
            } else {
                scene_items
                    .iter()
                    .any(|i| !i.appearance.hidden && i.appearance.opacity < 1.0)
            } || frame
                .scene
                .transparent_volume_meshes
                .iter()
                .any(|i| !i.appearance.hidden);
            if needs_oit {
                let hdr = self.viewport_slots[vp_idx].hdr.as_mut().unwrap();
                let [sw, sh] = hdr.scene_size;
                self.resources
                    .ensure_viewport_oit(device, hdr, sw, sh);
            }
        }

        // -----------------------------------------------------------------------
        // Build the command encoder.
        // -----------------------------------------------------------------------
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hdr_encoder"),
        });

        // Per-viewport camera bind group and HDR state for the HDR path.
        let slot = &self.viewport_slots[vp_idx];
        let camera_bg = &slot.camera_bind_group;
        let slot_hdr = slot.hdr.as_ref().expect(
            "HDR state missing; ensure_viewport_hdr must be called before render_frame_internal",
        );

        // -----------------------------------------------------------------------
        // HDR scene pass: render geometry into the HDR texture.
        // -----------------------------------------------------------------------
        {
            // Use SSAA target if enabled, otherwise render directly to hdr_texture.
            let use_ssaa = ssaa_factor > 1
                && slot_hdr.ssaa_colour_view.is_some()
                && slot_hdr.ssaa_depth_view.is_some();
            let scene_colour_view = if use_ssaa {
                slot_hdr.ssaa_colour_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_view
            };
            let scene_depth_view = if use_ssaa {
                slot_hdr.ssaa_depth_view.as_ref().unwrap()
            } else {
                &slot_hdr.hdr_depth_view
            };

            let clear_wgpu = wgpu::Color {
                r: hdr_clear_rgb[0] as f64,
                g: hdr_clear_rgb[1] as f64,
                b: hdr_clear_rgb[2] as f64,
                // Clear alpha to 0.0 so OIT composite can signal presence via alpha > 0.
                // Background pixels remain at alpha=0 and are detected in tone_map.wgsl.
                a: 0.0,
            };

            let hdr_ts_writes =
                self.ts_query_set
                    .as_ref()
                    .map(|qs| wgpu::RenderPassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    });
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_scene_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_colour_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_wgpu),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: scene_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0),
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: hdr_ts_writes,
                occlusion_query_set: None,
            });

            let resources = &self.resources;
            render_pass.set_bind_group(0, camera_bg, &[]);

            // Check skybox eligibility early; drawn after all opaques below.
            let show_skybox = frame
                .effects
                .environment
                .as_ref()
                .is_some_and(|e| e.show_skybox)
                && resources.ibl_skybox_view.is_some();

            let use_instancing = self.use_instancing;
            let batches = &self.instanced_batches;
            let compute_filter_results = &self.compute_filter_results;

            if !scene_items.is_empty() {
                if use_instancing && !batches.is_empty() {
                    let excluded_items: Vec<&SceneRenderItem> = scene_items
                        .iter()
                        .filter(|item| {
                            !item.appearance.hidden
                                && (item.active_attribute.is_some()
                                    || item.material.is_two_sided()
                                    || item.material.matcap_id.is_some())
                                && resources.mesh_store.get(item.mesh_id).is_some()
                        })
                        .collect();

                    // Separate opaque and transparent batches.
                    // Carry the global batch index (position in `batches`) alongside each batch
                    // so draw_indexed_indirect can compute the correct buffer offset.
                    let mut opaque_batches: Vec<(usize, &InstancedBatch)> = Vec::new();
                    let mut transparent_batches: Vec<(usize, &InstancedBatch)> = Vec::new();
                    for (batch_global_idx, batch) in batches.iter().enumerate() {
                        if batch.is_transparent {
                            transparent_batches.push((batch_global_idx, batch));
                        } else {
                            opaque_batches.push((batch_global_idx, batch));
                        }
                    }

                    if !opaque_batches.is_empty() && !frame.viewport.wireframe_mode {
                        let use_indirect = self.gpu_culling_enabled
                            && resources.hdr_solid_instanced_cull_pipeline.is_some()
                            && resources.indirect_args_buf.is_some();

                        if use_indirect {
                            if let (Some(pipeline), Some(indirect_buf)) = (
                                &resources.hdr_solid_instanced_cull_pipeline,
                                &resources.indirect_args_buf,
                            ) {
                                render_pass.set_pipeline(pipeline);
                                for (batch_global_idx, batch) in &opaque_batches {
                                    let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                        continue;
                                    };
                                    let mat_key = (
                                        batch.texture_id.unwrap_or(u64::MAX),
                                        batch.normal_map_id.unwrap_or(u64::MAX),
                                        batch.ao_map_id.unwrap_or(u64::MAX),
                                    );
                                    let Some(inst_tex_bg) =
                                        resources.instance_cull_bind_groups.get(&mat_key)
                                    else {
                                        continue;
                                    };
                                    render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                    render_pass.set_index_buffer(
                                        mesh.index_buffer.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    // Each DrawIndexedIndirect entry is 20 bytes; index by global
                                    // batch position so the offset matches write_indirect_args output.
                                    render_pass.draw_indexed_indirect(
                                        indirect_buf,
                                        *batch_global_idx as u64 * 20,
                                    );
                                }
                            }
                        } else if let Some(ref pipeline) = resources.hdr_solid_instanced_pipeline {
                            render_pass.set_pipeline(pipeline);
                            for (_, batch) in &opaque_batches {
                                let Some(mesh) = resources.mesh_store.get(batch.mesh_id) else {
                                    continue;
                                };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) =
                                    resources.instance_bind_groups.get(&mat_key)
                                else {
                                    continue;
                                };
                                render_pass.set_bind_group(1, inst_tex_bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(
                                    0..mesh.index_count,
                                    0,
                                    batch.instance_offset
                                        ..batch.instance_offset + batch.instance_count,
                                );
                            }
                        }
                    }

                    // NOTE: transparent_batches are now rendered in the OIT pass below,
                    // not in the HDR scene pass. This block intentionally left empty.
                    let _ = &transparent_batches; // suppress unused warning

                    if frame.viewport.wireframe_mode {
                        if let Some(ref hdr_wf) = resources.hdr_wireframe_pipeline {
                            let mut wf_idx = 0usize;
                            for item in scene_items {
                                if item.appearance.hidden {
                                    continue;
                                }
                                let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                    continue;
                                };
                                let skin_bg = item.skin_instance.and_then(|inst| {
                                    resources.skin_instance_bind_group(item.mesh_id, inst)
                                });
                                if let (Some(bg), Some(hdr_skinned_wf)) =
                                    (skin_bg, resources.hdr_skinned_wireframe_pipeline.as_ref())
                                {
                                    render_pass.set_pipeline(hdr_skinned_wf);
                                    render_pass.set_bind_group(2, bg, &[]);
                                } else {
                                    render_pass.set_pipeline(hdr_wf);
                                }
                                let bg = self
                                    .wireframe_bind_groups
                                    .get(wf_idx)
                                    .unwrap_or(&mesh.object_bind_group);
                                render_pass.set_bind_group(1, bg, &[]);
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.edge_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                                wf_idx += 1;
                            }
                        }
                    } else if let (Some(hdr_solid), Some(hdr_solid_two_sided)) = (
                        &resources.hdr_solid_pipeline,
                        &resources.hdr_solid_two_sided_pipeline,
                    ) {
                        // Only opaque excluded items are drawn in the scene pass; transparent
                        // excluded items go to the OIT pass below. LDR draws all excluded
                        // items inline (including transparent ones) using the transparent
                        // pipeline -- an intentional divergence since HDR uses OIT for
                        // transparency throughout.
                        for item in excluded_items
                            .into_iter()
                            .filter(|item| item.appearance.opacity >= 1.0 && !item.material.is_blend())
                        {
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };
                            let skin_bg = item.skin_instance.and_then(|inst| {
                                resources.skin_instance_bind_group(item.mesh_id, inst)
                            });
                            if let (Some(bg), Some(hdr_skinned_pl)) = (
                                skin_bg,
                                if item.material.is_two_sided() {
                                    resources.hdr_skinned_solid_two_sided_pipeline.as_ref()
                                } else {
                                    resources.hdr_skinned_solid_pipeline.as_ref()
                                },
                            ) {
                                render_pass.set_pipeline(hdr_skinned_pl);
                                render_pass.set_bind_group(2, bg, &[]);
                            } else {
                                let pipeline = if item.material.is_two_sided() {
                                    hdr_solid_two_sided
                                } else {
                                    hdr_solid
                                };
                                render_pass.set_pipeline(pipeline);
                            }
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            let filter = compute_filter_results
                                .iter()
                                .find(|r| r.mesh_id == item.mesh_id);
                            if let Some(fr) = filter {
                                render_pass.set_index_buffer(
                                    fr.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..fr.index_count, 0, 0..1);
                            } else {
                                render_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            }
                        }
                    }

                    // Normal-line overlays for instanced items with show_normals set.
                    // Instanced batch draws skip per-item logic, so these are drawn
                    // here after all batches finish.
                    if let Some(hdr_wf) = &resources.hdr_wireframe_pipeline {
                        for item in scene_items.iter().filter(|i| i.show_normals && !i.appearance.hidden) {
                            let Some(mesh) = resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };
                            if let Some(ref nl_buf) = mesh.normal_line_buffer {
                                if mesh.normal_line_count > 0 {
                                    render_pass.set_pipeline(hdr_wf);
                                    render_pass.set_bind_group(1, &mesh.normal_bind_group, &[]);
                                    render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                                    render_pass.draw(0..mesh.normal_line_count, 0..1);
                                }
                            }
                        }
                    }
                } else {
                    // Per-object path.
                    let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
                    let dist_from_eye = |item: &&SceneRenderItem| -> f32 {
                        let pos =
                            glam::Vec3::new(item.model[3][0], item.model[3][1], item.model[3][2]);
                        (pos - eye).length()
                    };

                    let mut opaque: Vec<&SceneRenderItem> = Vec::new();
                    let mut transparent: Vec<&SceneRenderItem> = Vec::new();
                    for item in scene_items {
                        if item.appearance.hidden || resources.mesh_store.get(item.mesh_id).is_none() {
                            continue;
                        }
                        if item.appearance.opacity < 1.0 || item.material.is_blend() {
                            transparent.push(item);
                        } else {
                            opaque.push(item);
                        }
                    }
                    opaque.sort_by(|a, b| {
                        dist_from_eye(a)
                            .partial_cmp(&dist_from_eye(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    transparent.sort_by(|a, b| {
                        dist_from_eye(b)
                            .partial_cmp(&dist_from_eye(a))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    let draw_item_hdr =
                        |render_pass: &mut wgpu::RenderPass<'_>,
                         item: &SceneRenderItem,
                         solid_pl: &wgpu::RenderPipeline,
                         trans_pl: &wgpu::RenderPipeline,
                         wf_pl: &wgpu::RenderPipeline| {
                            let mesh = resources.mesh_store.get(item.mesh_id).unwrap();
                            // mesh.object_bind_group (group 1) already carries the object uniform
                            // and the correct texture views.
                            render_pass.set_bind_group(1, &mesh.object_bind_group, &[]);

                            // Skinned routing: if the mesh has a skin sidecar
                            // and this item's instance has a palette uploaded,
                            // switch to the skinned pipeline variant and bind
                            // group 2. Falls back to the static path otherwise.
                            let skin_bg = item.skin_instance.and_then(|inst| {
                                resources.skin_instance_bind_group(item.mesh_id, inst)
                            });
                            let is_face_attr = item.active_attribute.as_ref().map_or(false, |a| {
                                matches!(
                                    a.kind,
                                    crate::resources::AttributeKind::Face
                                        | crate::resources::AttributeKind::FaceColour
                                        | crate::resources::AttributeKind::Halfedge
                                        | crate::resources::AttributeKind::Corner
                                )
                            });
                            if frame.viewport.wireframe_mode {
                                if let (Some(bg), Some(hdr_skinned_wf)) =
                                    (skin_bg, resources.hdr_skinned_wireframe_pipeline.as_ref())
                                {
                                    render_pass.set_pipeline(hdr_skinned_wf);
                                    render_pass.set_bind_group(2, bg, &[]);
                                } else {
                                    render_pass.set_pipeline(wf_pl);
                                }
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                render_pass.set_index_buffer(
                                    mesh.edge_index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                            } else if is_face_attr {
                                if let Some(ref fvb) = mesh.face_vertex_buffer {
                                    let pl = if item.appearance.opacity < 1.0 {
                                        trans_pl
                                    } else {
                                        solid_pl
                                    };
                                    render_pass.set_pipeline(pl);
                                    render_pass.set_vertex_buffer(0, fvb.slice(..));
                                    render_pass.draw(0..mesh.index_count, 0..1);
                                }
                            } else {
                                let filter = compute_filter_results
                                    .iter()
                                    .find(|r| r.mesh_id == item.mesh_id);
                                let pl = if item.appearance.opacity < 1.0 {
                                    trans_pl
                                } else {
                                    solid_pl
                                };
                                let is_blended = item.appearance.opacity < 1.0
                                    || item.material.is_blend();
                                let hdr_skinned_pl = if is_blended {
                                    resources.hdr_skinned_transparent_pipeline.as_ref()
                                } else if item.material.is_two_sided() {
                                    resources.hdr_skinned_solid_two_sided_pipeline.as_ref()
                                } else {
                                    resources.hdr_skinned_solid_pipeline.as_ref()
                                };
                                if let (Some(bg), Some(hdr_skinned)) = (skin_bg, hdr_skinned_pl)
                                {
                                    render_pass.set_pipeline(hdr_skinned);
                                    render_pass.set_bind_group(2, bg, &[]);
                                } else {
                                    render_pass.set_pipeline(pl);
                                }
                                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                if let Some(fr) = filter {
                                    render_pass.set_index_buffer(
                                        fr.index_buffer.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    render_pass.draw_indexed(0..fr.index_count, 0, 0..1);
                                } else {
                                    render_pass.set_index_buffer(
                                        mesh.index_buffer.slice(..),
                                        wgpu::IndexFormat::Uint32,
                                    );
                                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                                }
                            }
                            if item.show_normals {
                                if let Some(ref nl_buf) = mesh.normal_line_buffer {
                                    if mesh.normal_line_count > 0 {
                                        render_pass.set_pipeline(wf_pl);
                                        render_pass.set_bind_group(
                                            1,
                                            &mesh.normal_bind_group,
                                            &[],
                                        );
                                        render_pass.set_vertex_buffer(0, nl_buf.slice(..));
                                        render_pass.draw(0..mesh.normal_line_count, 0..1);
                                    }
                                }
                            }
                        };

                    // NOTE: only opaque items are drawn here. Transparent items are
                    // routed to the OIT pass below.
                    let _ = &transparent; // suppress unused warning
                    if let (
                        Some(hdr_solid),
                        Some(hdr_solid_two_sided),
                        Some(hdr_trans),
                        Some(hdr_wf),
                    ) = (
                        &resources.hdr_solid_pipeline,
                        &resources.hdr_solid_two_sided_pipeline,
                        &resources.hdr_transparent_pipeline,
                        &resources.hdr_wireframe_pipeline,
                    ) {
                        for item in &opaque {
                            let solid_pl = if item.material.is_two_sided() {
                                hdr_solid_two_sided
                            } else {
                                hdr_solid
                            };
                            draw_item_hdr(&mut render_pass, item, solid_pl, hdr_trans, hdr_wf);
                        }
                    }
                }
            }

            // Cap fill pass (HDR path : section view cross-section fill).
            if !slot.cap_buffers.is_empty() {
                if let Some(ref hdr_overlay) = resources.hdr_overlay_pipeline {
                    render_pass.set_pipeline(hdr_overlay);
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.cap_buffers {
                        render_pass.set_bind_group(1, bg, &[]);
                        render_pass.set_vertex_buffer(0, vbuf.slice(..));
                        render_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }
            }

            // SciVis Phase B+D+M8+M: point cloud, glyph, polyline, volume, streamtube (HDR path).
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
                true
            );

            // Phase 16 : GPU implicit surface (HDR path, before skybox).
            if !self.implicit_gpu_data.is_empty() {
                if let Some(ref dual) = self.resources.implicit_pipeline {
                    render_pass.set_pipeline(dual.for_format(true));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for gpu in &self.implicit_gpu_data {
                        render_pass.set_bind_group(1, &gpu.bind_group, &[]);
                        render_pass.draw(0..6, 0..1);
                    }
                }
            }
            // Phase 17 : GPU marching cubes indirect draw (HDR path).
            if !self.mc_gpu_data.is_empty() {
                render_pass.set_bind_group(0, camera_bg, &[]);
                for mc in &self.mc_gpu_data {
                    let vol = &self.resources.mc_volumes[mc.volume_idx];
                    if mc.wireframe || frame.viewport.wireframe_mode {
                        if let Some(ref dual) = self.resources.mc_wireframe_pipeline {
                            render_pass.set_pipeline(dual.for_format(true));
                            for (slab, wire_bg) in
                                vol.slabs.iter().zip(mc.wire_slab_bgs.iter())
                            {
                                render_pass.set_bind_group(1, wire_bg, &[]);
                                render_pass.draw_indirect(&slab.wire_indirect_buf, 0);
                            }
                        }
                    } else if let Some(ref dual) = self.resources.mc_surface_pipeline {
                        render_pass.set_pipeline(dual.for_format(true));
                        render_pass.set_bind_group(1, &mc.render_bg, &[]);
                        for slab in &vol.slabs {
                            render_pass.set_vertex_buffer(0, slab.vertex_buf.slice(..));
                            render_pass.draw_indirect(&slab.indirect_buf, 0);
                        }
                    }
                }
            }

            // Gaussian splats (HDR path).
            if !self.gaussian_splat_draw_data.is_empty() {
                if let Some(ref dual) = self.resources.gaussian_splat_pipeline {
                    render_pass.set_pipeline(dual.for_format(true));
                    render_pass.set_bind_group(0, camera_bg, &[]);
                    for dd in &self.gaussian_splat_draw_data {
                        if dd.wireframe {
                            continue;
                        }
                        if let Some(set) = self.resources.gaussian_splat_store.get(dd.store_index) {
                            if let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) {
                                render_pass.set_bind_group(1, &vp_sort.render_bg, &[]);
                                render_pass.draw(0..6, 0..dd.count);
                            }
                        }
                    }
                }
            }
            // TransparentVolumeMesh boundary wireframe overlay (HDR path).
            if !self.tvm_wireframe_draws.is_empty() {
                if let (Some(tvm_bg), Some(hdr_wf)) = (&self.tvm_wireframe_bg, &resources.hdr_wireframe_pipeline) {
                    for mesh_id in &self.tvm_wireframe_draws {
                        if let Some(mesh) = resources.mesh_store.get(*mesh_id) {
                            render_pass.set_pipeline(hdr_wf);
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

            // Draw skybox last among opaques : only uncovered sky pixels pass depth == 1.0.
            if show_skybox {
                render_pass.set_bind_group(0, camera_bg, &[]);
                render_pass.set_pipeline(&resources.skybox_pipeline);
                render_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // SSAA resolve pass: downsample supersampled scene -> hdr_texture.
        // Only runs when ssaa_factor > 1 and the resolve pipeline is available.
        // -----------------------------------------------------------------------
        if ssaa_factor > 1 {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let (Some(pipeline), Some(bg)) = (
                &self.resources.ssaa_resolve_pipeline,
                &slot_hdr.ssaa_resolve_bind_group,
            ) {
                let mut resolve_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ssaa_resolve_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                resolve_pass.set_pipeline(pipeline);
                resolve_pass.set_bind_group(0, bg, &[]);
                resolve_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Decal pass (D1): projects each decal texture onto opaque surfaces.
        // Reads scene depth as a texture; no depth attachment.
        // Runs after opaque geometry and SSAA resolve, before transparent passes.
        // -----------------------------------------------------------------------
        if !self.decal_gpu_data.is_empty() {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
            let depth_bg = &slot_hdr.decal_depth_bg;
            let replace_pipeline = self.resources.decal_replace_pipeline.as_ref();
            let multiply_pipeline = self.resources.decal_multiply_pipeline.as_ref();
            if replace_pipeline.is_some() || multiply_pipeline.is_some() {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("decal_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_bind_group(0, camera_bg, &[]);
                pass.set_bind_group(1, depth_bg, &[]);
                for gpu in &self.decal_gpu_data {
                    let pipeline = match gpu.blend_mode {
                        crate::renderer::DecalBlendMode::Replace => replace_pipeline,
                        crate::renderer::DecalBlendMode::Multiply => multiply_pipeline,
                    };
                    if let Some(pl) = pipeline {
                        pass.set_pipeline(&pl.hdr);
                        pass.set_bind_group(2, &gpu.bind_group, &[]);
                        pass.draw(0..6, 0..1);
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Sub-object highlight pass: face fill, edge lines, vertex sprites.
        // Runs after opaque geometry (depth buffer is ready) and before OIT so
        // highlights are not occluded by opaque surfaces.
        // -----------------------------------------------------------------------
        if let Some(sub_hl) = self.viewport_slots[vp_idx].sub_highlight.as_ref() {
            let resources = &self.resources;
            if let (Some(fill_pl), Some(edge_pl), Some(sprite_pl)) = (
                &resources.sub_highlight_fill_pipeline,
                &resources.sub_highlight_edge_pipeline,
                &resources.sub_highlight_sprite_pipeline,
            ) {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let camera_bg = &self.viewport_slots[vp_idx].camera_bind_group;
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("sub_highlight_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            // Store even though depth_write_enabled=false on all
                            // sub-highlight pipelines: the values are unchanged, but
                            // StoreOp::Discard would invalidate the tile on Metal and
                            // cause subsequent passes (tone_map, grid, screen images)
                            // to read 0.0, making the background go black.
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if sub_hl.fill_vertex_count > 0 {
                    pass.set_pipeline(fill_pl);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(1, &sub_hl.fill_bind_group, &[]);
                    pass.set_vertex_buffer(0, sub_hl.fill_vertex_buf.slice(..));
                    pass.draw(0..sub_hl.fill_vertex_count, 0..1);
                }
                if sub_hl.edge_segment_count > 0 {
                    pass.set_pipeline(edge_pl);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(1, &sub_hl.edge_bind_group, &[]);
                    pass.set_vertex_buffer(0, sub_hl.edge_vertex_buf.slice(..));
                    pass.draw(0..6, 0..sub_hl.edge_segment_count);
                }
                if sub_hl.sprite_point_count > 0 {
                    pass.set_pipeline(sprite_pl);
                    pass.set_bind_group(0, camera_bg, &[]);
                    pass.set_bind_group(1, &sub_hl.sprite_bind_group, &[]);
                    pass.set_vertex_buffer(0, sub_hl.sprite_vertex_buf.slice(..));
                    pass.draw(0..6, 0..sub_hl.sprite_point_count);
                }
            }
        }

        // -----------------------------------------------------------------------
        // OIT pass: render transparent items into accum + reveal textures.
        // Completely skipped when no transparent items exist (zero overhead).
        // -----------------------------------------------------------------------
        let has_transparent = if self.use_instancing && !self.instanced_batches.is_empty() {
            // Transparent instanced batches go through OIT. Transparent excluded items
            // (two-sided, active-attribute, matcap) are not in any instanced batch, so
            // they must also be checked here -- otherwise the OIT pass is skipped and
            // those items are invisible.
            self.instanced_batches.iter().any(|b| b.is_transparent)
                || scene_items.iter().any(|i| {
                    !i.appearance.hidden
                        && (i.appearance.opacity < 1.0 || i.material.is_blend())
                        && (i.active_attribute.is_some()
                            || i.material.is_two_sided()
                            || i.material.matcap_id.is_some())
                })
        } else {
            scene_items
                .iter()
                .any(|i| !i.appearance.hidden && i.appearance.opacity < 1.0)
        } || frame
            .scene
            .transparent_volume_meshes
            .iter()
            .any(|i| !i.appearance.hidden);

        if has_transparent {
            // OIT targets already allocated in the pre-pass above.
            if let (Some(accum_view), Some(reveal_view)) = (
                slot_hdr.oit_accum_view.as_ref(),
                slot_hdr.oit_reveal_view.as_ref(),
            ) {
                let hdr_depth_view = &slot_hdr.hdr_depth_view;
                // Clear accum to (0,0,0,0), reveal to 1.0 (no contribution yet).
                let mut oit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: accum_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: reveal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 1.0,
                                    g: 1.0,
                                    b: 1.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load, // reuse opaque depth
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                oit_pass.set_bind_group(0, camera_bg, &[]);

                if self.use_instancing && !self.instanced_batches.is_empty() {
                    let use_indirect_oit = self.gpu_culling_enabled
                        && self.resources.oit_instanced_cull_pipeline.is_some()
                        && self.resources.indirect_args_buf.is_some();

                    if use_indirect_oit {
                        if let (Some(pipeline), Some(indirect_buf)) = (
                            &self.resources.oit_instanced_cull_pipeline,
                            &self.resources.indirect_args_buf,
                        ) {
                            oit_pass.set_pipeline(pipeline);
                            for (batch_global_idx, batch) in
                                self.instanced_batches.iter().enumerate()
                            {
                                if !batch.is_transparent {
                                    continue;
                                }
                                let Some(mesh) = self.resources.mesh_store.get(batch.mesh_id)
                                else {
                                    continue;
                                };
                                let mat_key = (
                                    batch.texture_id.unwrap_or(u64::MAX),
                                    batch.normal_map_id.unwrap_or(u64::MAX),
                                    batch.ao_map_id.unwrap_or(u64::MAX),
                                );
                                let Some(inst_tex_bg) =
                                    self.resources.instance_cull_bind_groups.get(&mat_key)
                                else {
                                    continue;
                                };
                                oit_pass.set_bind_group(1, inst_tex_bg, &[]);
                                oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                oit_pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint32,
                                );
                                oit_pass.draw_indexed_indirect(
                                    indirect_buf,
                                    batch_global_idx as u64 * 20,
                                );
                            }
                        }
                    } else if let Some(ref pipeline) = self.resources.oit_instanced_pipeline {
                        oit_pass.set_pipeline(pipeline);
                        for batch in &self.instanced_batches {
                            if !batch.is_transparent {
                                continue;
                            }
                            let Some(mesh) = self.resources.mesh_store.get(batch.mesh_id) else {
                                continue;
                            };
                            let mat_key = (
                                batch.texture_id.unwrap_or(u64::MAX),
                                batch.normal_map_id.unwrap_or(u64::MAX),
                                batch.ao_map_id.unwrap_or(u64::MAX),
                            );
                            let Some(inst_tex_bg) =
                                self.resources.instance_bind_groups.get(&mat_key)
                            else {
                                continue;
                            };
                            oit_pass.set_bind_group(1, inst_tex_bg, &[]);
                            oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            oit_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            oit_pass.draw_indexed(
                                0..mesh.index_count,
                                0,
                                batch.instance_offset..batch.instance_offset + batch.instance_count,
                            );
                        }
                    }

                    // Transparent excluded items (two-sided, active attribute, matcap) are not
                    // in any instanced batch, so the instanced OIT loop above skips them.
                    // Render them here individually so they are not invisible at opacity < 1.
                    if let Some(ref pipeline) = self.resources.oit_pipeline {
                        oit_pass.set_pipeline(pipeline);
                        let mut last_skinned = false;
                        for item in scene_items {
                            if item.appearance.hidden || (item.appearance.opacity >= 1.0 && !item.material.is_blend()) {
                                continue;
                            }
                            let skin_bg = item.skin_instance.and_then(|inst| {
                                self.resources.skin_instance_bind_group(item.mesh_id, inst)
                            });
                            if skin_bg.is_none()
                                && item.active_attribute.is_none()
                                && !item.material.is_two_sided()
                                && item.material.matcap_id.is_none()
                            {
                                continue;
                            }
                            let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                                continue;
                            };
                            let want_skinned = skin_bg.is_some()
                                && self.resources.skinned_oit_pipeline.is_some();
                            if want_skinned != last_skinned {
                                if want_skinned {
                                    oit_pass.set_pipeline(
                                        self.resources.skinned_oit_pipeline.as_ref().unwrap(),
                                    );
                                } else {
                                    oit_pass.set_pipeline(pipeline);
                                }
                                last_skinned = want_skinned;
                            }
                            oit_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                            if let Some(bg) = skin_bg {
                                oit_pass.set_bind_group(2, bg, &[]);
                            }
                            oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                            oit_pass.set_index_buffer(
                                mesh.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            oit_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        }
                    }
                } else if let Some(ref pipeline) = self.resources.oit_pipeline {
                    oit_pass.set_pipeline(pipeline);
                    let mut last_skinned = false;
                    for item in scene_items {
                        if item.appearance.hidden || item.appearance.opacity >= 1.0 {
                            continue;
                        }
                        let Some(mesh) = self.resources.mesh_store.get(item.mesh_id) else {
                            continue;
                        };
                        let skin_bg = item.skin_instance.and_then(|inst| {
                            self.resources.skin_instance_bind_group(item.mesh_id, inst)
                        });
                        let want_skinned = skin_bg.is_some()
                            && self.resources.skinned_oit_pipeline.is_some();
                        if want_skinned != last_skinned {
                            if want_skinned {
                                oit_pass.set_pipeline(
                                    self.resources.skinned_oit_pipeline.as_ref().unwrap(),
                                );
                            } else {
                                oit_pass.set_pipeline(pipeline);
                            }
                            last_skinned = want_skinned;
                        }
                        oit_pass.set_bind_group(1, &mesh.object_bind_group, &[]);
                        if let Some(bg) = skin_bg {
                            oit_pass.set_bind_group(2, bg, &[]);
                        }
                        oit_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        oit_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        oit_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }

                // -----------------------------------------------------------
                // Projected tetrahedra transparent volume meshes (Phase 6).
                // -----------------------------------------------------------
                if !frame.scene.transparent_volume_meshes.is_empty() {
                    self.resources.ensure_pt_pipeline(device);
                    if let Some(pipeline) = self.resources.pt_pipeline.as_ref() {
                        oit_pass.set_pipeline(pipeline);
                        oit_pass.set_bind_group(0, camera_bg, &[]);
                        for item in &frame.scene.transparent_volume_meshes {
                            if item.appearance.hidden {
                                continue;
                            }
                            if item.appearance.wireframe || frame.viewport.wireframe_mode {
                                continue;
                            }
                            let Some(gpu) = self.resources.projected_tet_store.get(item.id.0)
                            else {
                                continue;
                            };
                            let (scalar_min, scalar_max) =
                                item.scalar_range.unwrap_or(gpu.scalar_range);
                            let uniform = crate::resources::ProjectedTetUniform {
                                density: item.density,
                                scalar_min,
                                scalar_max,
                                threshold_min: item.threshold_min,
                                threshold_max: item.threshold_max,
                                _pad: 0.0,
                            };
                            queue.write_buffer(
                                &gpu.uniform_buffer,
                                0,
                                bytemuck::bytes_of(&uniform),
                            );
                            for chunk in &gpu.chunks {
                                oit_pass.set_bind_group(1, &chunk.bind_group, &[]);
                                oit_pass.draw(0..6, 0..chunk.tet_count);
                            }
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // OIT composite pass: blend accum/reveal into HDR buffer.
        // Only executes when transparent items were present.
        // -----------------------------------------------------------------------
        if has_transparent {
            if let (Some(pipeline), Some(bg)) = (
                self.resources.oit_composite_pipeline.as_ref(),
                slot_hdr.oit_composite_bind_group.as_ref(),
            ) {
                let hdr_view = &slot_hdr.hdr_view;
                let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("oit_composite_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                composite_pass.set_pipeline(pipeline);
                composite_pass.set_bind_group(0, bg, &[]);
                composite_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Phase 4: Surface LIC passes.
        // Pass 1: render each LIC mesh into lic_vector_texture (Rgba8Unorm).
        // Pass 2: advect fullscreen triangle into lic_output_texture (R8Unorm).
        // -----------------------------------------------------------------------
        if !self.lic_gpu_data.is_empty() {
            if let (Some(surface_pipeline), Some(advect_pipeline)) = (
                self.resources.lic_surface_pipeline.as_ref(),
                self.resources.lic_advect_pipeline.as_ref(),
            ) {
                let camera_bg = &slot.camera_bind_group;
                // Pass 1: surface vector pass (clears lic_vector_texture first).
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("lic_surface_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &slot_hdr.lic_vector_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(surface_pipeline);
                    pass.set_bind_group(0, camera_bg, &[]);
                    for gpu in &self.lic_gpu_data {
                        let Some(mesh) = self.resources.mesh_store.get(gpu.mesh_id) else {
                            continue;
                        };
                        let Some(vec_buf) =
                            mesh.vector_attribute_buffers.get(&gpu.vector_attribute)
                        else {
                            continue;
                        };
                        pass.set_bind_group(1, &gpu.bind_group, &[]);
                        pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        pass.set_vertex_buffer(1, vec_buf.slice(..));
                        pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
                // Pass 2: advect pass (fullscreen, writes LIC intensity to lic_output_texture).
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("lic_advect_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &slot_hdr.lic_output_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.5,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(advect_pipeline);
                    pass.set_bind_group(0, &slot_hdr.lic_advect_bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
            }
        }

        // -----------------------------------------------------------------------
        // Outline composite pass (HDR path): blit offscreen outline onto hdr_view.
        // Runs after the HDR scene pass (which has depth+stencil) in a separate
        // pass with no depth attachment, so the composite pipeline is compatible.
        // -----------------------------------------------------------------------
        if !slot.outline_object_buffers.is_empty()
            || !slot.splat_outline_buffers.is_empty()
            || !slot.streamtube_outline_items.is_empty()
            || !slot.tube_outline_items.is_empty()
            || !slot.ribbon_outline_items.is_empty()
            || !slot.polyline_outline_indices.is_empty()
            || !slot.volume_outline_indices.is_empty()
            || !slot.glyph_outline_indices.is_empty()
            || !slot.tensor_glyph_outline_indices.is_empty()
            || !slot.sprite_outline_indices.is_empty()
            || !slot.raw_geom_outline_buffers.is_empty()
            || !slot.screen_rect_outline_buffers.is_empty()
            || !slot.implicit_outline_indices.is_empty()
            || !slot.mc_outline_data.is_empty()
        {
            // Prefer the HDR-format pipeline; fall back to LDR single-sample.
            let hdr_pipeline = self
                .resources
                .outline_composite_pipeline_hdr
                .as_ref()
                .or(self.resources.outline_composite_pipeline_single.as_ref());
            if let Some(pipeline) = hdr_pipeline {
                let bg = &slot_hdr.outline_composite_bind_group;
                let hdr_view = &slot_hdr.hdr_view;
                let hdr_depth_view = &slot_hdr.hdr_depth_view;
                let mut outline_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hdr_outline_composite_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: hdr_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                outline_pass.set_pipeline(pipeline);
                outline_pass.set_bind_group(0, bg, &[]);
                outline_pass.draw(0..3, 0..1);
            }
        }

        // Phase 5 : effect throttling. Flag was computed in prepare() so that
        // FrameStats reports exactly what fired rather than an approximation.
        let throttle_effects = self.degradation_effects_throttled;

        // -----------------------------------------------------------------------
        // SSAO pass.
        // -----------------------------------------------------------------------
        if pp.ssao && !throttle_effects {
            if let Some(ssao_pipeline) = &self.resources.ssao_pipeline {
                {
                    let mut ssao_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ssao_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &slot_hdr.ssao_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    ssao_pass.set_pipeline(ssao_pipeline);
                    ssao_pass.set_bind_group(0, &slot_hdr.ssao_bg, &[]);
                    ssao_pass.draw(0..3, 0..1);
                }

                // SSAO blur pass.
                if let Some(ssao_blur_pipeline) = &self.resources.ssao_blur_pipeline {
                    let mut ssao_blur_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("ssao_blur_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &slot_hdr.ssao_blur_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    ssao_blur_pass.set_pipeline(ssao_blur_pipeline);
                    ssao_blur_pass.set_bind_group(0, &slot_hdr.ssao_blur_bg, &[]);
                    ssao_blur_pass.draw(0..3, 0..1);
                }
            }
        }

        // -----------------------------------------------------------------------
        // Contact shadow pass.
        // -----------------------------------------------------------------------
        if pp.contact_shadows && !throttle_effects {
            if let Some(cs_pipeline) = &self.resources.contact_shadow_pipeline {
                let mut cs_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("contact_shadow_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.contact_shadow_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                cs_pass.set_pipeline(cs_pipeline);
                cs_pass.set_bind_group(0, &slot_hdr.contact_shadow_bg, &[]);
                cs_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Bloom passes.
        // -----------------------------------------------------------------------
        if pp.bloom && !throttle_effects {
            // Threshold pass: extract bright pixels into bloom_threshold_texture.
            if let Some(bloom_threshold_pipeline) = &self.resources.bloom_threshold_pipeline {
                {
                    let mut threshold_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("bloom_threshold_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &slot_hdr.bloom_threshold_view,
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
                    threshold_pass.set_pipeline(bloom_threshold_pipeline);
                    threshold_pass.set_bind_group(0, &slot_hdr.bloom_threshold_bg, &[]);
                    threshold_pass.draw(0..3, 0..1);
                }

                // 4 ping-pong H+V blur passes for a wide glow.
                // Pass 1: threshold -> ping -> pong. Passes 2-4: pong -> ping -> pong.
                if let Some(blur_pipeline) = &self.resources.bloom_blur_pipeline {
                    let blur_h_bg = &slot_hdr.bloom_blur_h_bg;
                    let blur_h_pong_bg = &slot_hdr.bloom_blur_h_pong_bg;
                    let blur_v_bg = &slot_hdr.bloom_blur_v_bg;
                    let bloom_ping_view = &slot_hdr.bloom_ping_view;
                    let bloom_pong_view = &slot_hdr.bloom_pong_view;
                    const BLUR_ITERATIONS: usize = 4;
                    for i in 0..BLUR_ITERATIONS {
                        // H pass: pass 0 reads threshold, subsequent passes read pong.
                        let h_bg = if i == 0 { blur_h_bg } else { blur_h_pong_bg };
                        {
                            let mut h_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("bloom_blur_h_pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: bloom_ping_view,
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
                            h_pass.set_pipeline(blur_pipeline);
                            h_pass.set_bind_group(0, h_bg, &[]);
                            h_pass.draw(0..3, 0..1);
                        }
                        // V pass: ping -> pong.
                        {
                            let mut v_pass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: Some("bloom_blur_v_pass"),
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: bloom_pong_view,
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
                            v_pass.set_pipeline(blur_pipeline);
                            v_pass.set_bind_group(0, blur_v_bg, &[]);
                            v_pass.draw(0..3, 0..1);
                        }
                    }
                }
            }
        }

        // -----------------------------------------------------------------------
        // Depth of field pass: HDR + depth -> dof_texture (when enabled).
        // -----------------------------------------------------------------------
        if pp.dof_enabled && !throttle_effects {
            if let Some(dof_pipeline) = &self.resources.dof_pipeline {
                let mut dof_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("dof_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &slot_hdr.dof_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                dof_pass.set_pipeline(dof_pipeline);
                dof_pass.set_bind_group(0, &slot_hdr.dof_bg, &[]);
                dof_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // Tone map pass: HDR + bloom + AO -> tone-mapped LDR.
        //
        // When render_scale < 1.0 the entire post-process chain runs at scene
        // resolution. The result lands in upscale_view (scene-res) and is then
        // upscale-blitted to output_view at native resolution.
        // -----------------------------------------------------------------------
        let use_fxaa = pp.fxaa;
        let use_hdr_upscale = slot_hdr.upscale_bind_group.is_some();
        if let Some(tone_map_pipeline) = &self.resources.tone_map_pipeline {
            let tone_target: &wgpu::TextureView = if use_fxaa {
                &slot_hdr.fxaa_view
            } else if use_hdr_upscale {
                slot_hdr.upscale_view.as_ref().unwrap()
            } else {
                output_view
            };
            let mut tone_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tone_map_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: tone_target,
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
            tone_pass.set_pipeline(tone_map_pipeline);
            tone_pass.set_bind_group(0, &slot_hdr.tone_map_bind_group, &[]);
            tone_pass.draw(0..3, 0..1);
        }

        // -----------------------------------------------------------------------
        // FXAA pass: fxaa_texture -> upscale_view (scaled) or output_view (1:1).
        // -----------------------------------------------------------------------
        if use_fxaa {
            if let Some(fxaa_pipeline) = &self.resources.fxaa_pipeline {
                let fxaa_target: &wgpu::TextureView = if use_hdr_upscale {
                    slot_hdr.upscale_view.as_ref().unwrap()
                } else {
                    output_view
                };
                let mut fxaa_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("fxaa_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: fxaa_target,
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
                fxaa_pass.set_pipeline(fxaa_pipeline);
                fxaa_pass.set_bind_group(0, &slot_hdr.fxaa_bind_group, &[]);
                fxaa_pass.draw(0..3, 0..1);
            }
        }

        // -----------------------------------------------------------------------
        // HDR upscale pass: blit scene-resolution post-processed output to native
        // output_view. Only runs when render_scale < 1.0.
        // -----------------------------------------------------------------------
        if use_hdr_upscale {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let Some(upscale_bg) = &slot_hdr.upscale_bind_group {
                if let Some(pipeline) = &self.resources.dyn_res_upscale_pipeline {
                    let mut upscale_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("hdr_upscale_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: output_view,
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
                    upscale_pass.set_pipeline(pipeline);
                    upscale_pass.set_bind_group(0, upscale_bg, &[]);
                    upscale_pass.draw(0..3, 0..1);
                }
            }
        }

        // Depth blit pass: when render_scale < 1.0, the scene depth texture is
        // smaller than the output surface. Copy it to output_depth_texture (native
        // resolution) so the post-tone-map passes below can attach output_depth_view
        // alongside output_view without a size mismatch. Skipped when render_scale
        // is 1.0 (output_depth_view is just a second view of hdr_depth_texture).
        {
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            if let Some(blit_bg) = &slot_hdr.depth_blit_bind_group {
                if let Some(blit_pipeline) = &self.resources.depth_blit_pipeline {
                    let mut blit_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("depth_blit_pass"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &slot_hdr.output_depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    blit_pass.set_pipeline(blit_pipeline);
                    blit_pass.set_bind_group(0, blit_bg, &[]);
                    blit_pass.draw(0..3, 0..1);
                }
            }
        }

        // Grid pass (HDR path): draw the existing analytical grid on the final
        // output after tone mapping / FXAA, reusing the scene depth buffer so
        // scene geometry still occludes the grid exactly as in the LDR path.
        if frame.viewport.show_grid {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let grid_bg = &slot.grid_bind_group;
            let mut grid_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_grid_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &slot_hdr.output_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            grid_pass.set_pipeline(&self.resources.grid_pipeline);
            grid_pass.set_bind_group(0, grid_bg, &[]);
            grid_pass.draw(0..3, 0..1);
        }

        // Ground plane pass (HDR path): drawn after grid, before editor overlays.
        // Uses the scene depth buffer for correct occlusion against geometry.
        if !matches!(
            frame.effects.ground_plane.mode,
            crate::renderer::types::GroundPlaneMode::None
        ) {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let mut gp_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_ground_plane_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &slot_hdr.output_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            gp_pass.set_pipeline(&self.resources.ground_plane_pipeline);
            gp_pass.set_bind_group(0, &self.resources.ground_plane_bind_group, &[]);
            gp_pass.draw(0..3, 0..1);
        }

        // Phase 10B / Phase 12 : screen-space image overlay pass (HDR path).
        // Must run before the editor overlay and axes passes because those
        // discard hdr_depth_view (StoreOp::Discard). The DC pipeline compares
        // per-pixel image depth against the scene depth buffer; if the buffer
        // has been discarded, Metal returns zeros and all DC fragments fail.
        // Plain overlay items (depth_compare: Always) are unaffected by depth,
        // but ordering them here keeps depth-composite correct.
        if !self.screen_image_gpu_data.is_empty() {
            if let Some(overlay_pipeline) = &self.resources.screen_image_pipeline {
                let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
                let dc_pipeline = self.resources.screen_image_dc_pipeline.as_ref();
                let mut img_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("screen_image_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                for gpu in &self.screen_image_gpu_data {
                    if let (Some(dc_bg), Some(dc_pipe)) = (&gpu.depth_bind_group, dc_pipeline) {
                        img_pass.set_pipeline(dc_pipe);
                        img_pass.set_bind_group(0, dc_bg, &[]);
                    } else {
                        img_pass.set_pipeline(overlay_pipeline);
                        img_pass.set_bind_group(0, &gpu.bind_group, &[]);
                    }
                    img_pass.draw(0..6, 0..1);
                }
            }
        }

        // Editor overlay pass (HDR path): draw viewport/editor overlays on the
        // final output after tone mapping / FXAA, reusing the scene depth
        // buffer so depth-tested helpers still behave correctly.
        {
            let slot = &self.viewport_slots[vp_idx];
            let slot_hdr = slot.hdr.as_ref().unwrap();
            let has_editor_overlays = (frame.interaction.gizmo_model.is_some()
                && slot.gizmo_index_count > 0)
                || !slot.constraint_line_buffers.is_empty()
                || !slot.clip_plane_fill_buffers.is_empty()
                || !slot.clip_plane_line_buffers.is_empty()
                || !slot.xray_object_buffers.is_empty();
            if has_editor_overlays {
                let camera_bg = &slot.camera_bind_group;
                let mut overlay_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hdr_editor_overlay_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                if frame.interaction.gizmo_model.is_some() && slot.gizmo_index_count > 0 {
                    overlay_pass.set_pipeline(&self.resources.gizmo_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    overlay_pass.set_bind_group(1, &slot.gizmo_bind_group, &[]);
                    overlay_pass.set_vertex_buffer(0, slot.gizmo_vertex_buffer.slice(..));
                    overlay_pass.set_index_buffer(
                        slot.gizmo_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    overlay_pass.draw_indexed(0..slot.gizmo_index_count, 0, 0..1);
                }

                if !slot.constraint_line_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.overlay_line_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, index_count, _ubuf, bg) in &slot.constraint_line_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*index_count, 0, 0..1);
                    }
                }

                if !slot.clip_plane_fill_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.overlay_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_fill_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }

                if !slot.clip_plane_line_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.overlay_line_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (vbuf, ibuf, idx_count, _ubuf, bg) in &slot.clip_plane_line_buffers {
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                        overlay_pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        overlay_pass.draw_indexed(0..*idx_count, 0, 0..1);
                    }
                }

                if !slot.xray_object_buffers.is_empty() {
                    overlay_pass.set_pipeline(&self.resources.xray_pipeline);
                    overlay_pass.set_bind_group(0, camera_bg, &[]);
                    for (mesh_id, _buf, bg) in &slot.xray_object_buffers {
                        let Some(mesh) = self.resources.mesh_store.get(*mesh_id) else {
                            continue;
                        };
                        overlay_pass.set_bind_group(1, bg, &[]);
                        overlay_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        overlay_pass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        overlay_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                    }
                }
            }
        }

        // Axes indicator pass (HDR path): draw in screen space on the final
        // output after tone mapping / FXAA so it stays visible in PBR mode.
        if frame.viewport.show_axes_indicator {
            let slot = &self.viewport_slots[vp_idx];
            if slot.axes_vertex_count > 0 {
                let slot_hdr = slot.hdr.as_ref().unwrap();
                let mut axes_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("hdr_axes_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: output_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &slot_hdr.output_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                axes_pass.set_pipeline(&self.resources.axes_pipeline);
                axes_pass.set_vertex_buffer(0, slot.axes_vertex_buffer.slice(..));
                axes_pass.draw(0..slot.axes_vertex_count, 0..1);
            }
        }

        // Overlay shapes, rects, labels, scalar bars, rulers, and overlay images (HDR path): drawn last.
        let has_overlay = self.overlay_shape_gpu_data.is_some()
            || self.overlay_rect_gpu_data.is_some()
            || self.label_gpu_data.is_some()
            || self.scalar_bar_gpu_data.is_some()
            || self.ruler_gpu_data.is_some()
            || self.loading_bar_gpu_data.is_some()
            || !self.overlay_image_gpu_data.is_empty();

        // HDR backdrop blur: the tonemapped scene is on an intermediate so we
        // can sample it for the blur. When blur is needed we redirect the
        // tonemapped output to a managed intermediate, blur it, then draw
        // overlays there and blit to output_view at the end.
        let needs_hdr_blur = self.has_backdrop_blur_shapes();
        let hdr_blur_bg: Option<wgpu::BindGroup> = if needs_hdr_blur && has_overlay {
            self.ensure_backdrop_blur_state(device, w.max(1), h.max(1));
            // Blit output_view content to the intermediate so we have a
            // samplable copy. We already have the tonemapped result on
            // output_view. Unfortunately surface textures can't be sampled,
            // so we blit to our intermediate first.
            //
            // Actually for HDR: the tone-map wrote to output_view (or
            // upscale_view). We need the scene in a samplable texture. The
            // HDR colour texture (hdr_colour_view) is samplable but it's HDR.
            // For simplicity, use the HDR colour texture as the blur source;
            // the blur result will be HDR-ish but clamped by the LDR target
            // format of the blur textures. This looks acceptable in practice.
            let slot_hdr = self.viewport_slots[vp_idx].hdr.as_ref().unwrap();
            let source = &slot_hdr.hdr_view;
            let spread = self.overlay_shape_gpu_data.as_ref()
                .map(|d| d.max_blur_radius)
                .unwrap_or(8.0);
            Some(self.run_backdrop_blur(&mut encoder, device, queue, source, spread))
        } else {
            None
        };

        if has_overlay {
            let hdr_depth_view = &self.viewport_slots[vp_idx]
                .hdr
                .as_ref()
                .unwrap()
                .output_depth_view;
            let mut overlay_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: hdr_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            // Blur backdrop shapes drawn first (behind normal shapes).
            if let Some(ref bg) = hdr_blur_bg {
                self.draw_blur_shapes(&mut overlay_pass, bg);
            }
            // SDF overlay shapes (HDR path).
            if let Some(ref sd) = self.overlay_shape_gpu_data {
                if sd.vertex_count > 0 {
                    if let Some(pipeline) = &self.resources.overlay_shape_pipeline {
                        if let Some(vbuf) = &sd.vertex_buf {
                            overlay_pass.set_pipeline(pipeline);
                            overlay_pass.set_vertex_buffer(0, vbuf.slice(..));
                            overlay_pass.draw(0..sd.vertex_count, 0..1);
                        }
                    }
                }
                if !sd.tex_batches.is_empty() {
                    if let Some(pipeline) = &self.resources.overlay_shape_tex_pipeline {
                        overlay_pass.set_pipeline(pipeline);
                        for batch in &sd.tex_batches {
                            overlay_pass.set_bind_group(0, &batch.bind_group, &[]);
                            overlay_pass.set_vertex_buffer(0, batch.vertex_buf.slice(..));
                            overlay_pass.draw(0..batch.vertex_count, 0..1);
                        }
                    }
                }
            }
            if let Some(pipeline) = &self.resources.overlay_text_pipeline {
                overlay_pass.set_pipeline(pipeline);
                if let Some(ref rr) = self.overlay_rect_gpu_data {
                    overlay_pass.set_bind_group(0, &rr.bind_group, &[]);
                    overlay_pass.set_vertex_buffer(0, rr.vertex_buf.slice(..));
                    overlay_pass.draw(0..rr.vertex_count, 0..1);
                }
                if let Some(ref ld) = self.label_gpu_data {
                    overlay_pass.set_bind_group(0, &ld.bind_group, &[]);
                    overlay_pass.set_vertex_buffer(0, ld.vertex_buf.slice(..));
                    overlay_pass.draw(0..ld.vertex_count, 0..1);
                }
                if let Some(ref sb) = self.scalar_bar_gpu_data {
                    overlay_pass.set_bind_group(0, &sb.bind_group, &[]);
                    overlay_pass.set_vertex_buffer(0, sb.vertex_buf.slice(..));
                    overlay_pass.draw(0..sb.vertex_count, 0..1);
                }
                if let Some(ref rd) = self.ruler_gpu_data {
                    overlay_pass.set_bind_group(0, &rd.bind_group, &[]);
                    overlay_pass.set_vertex_buffer(0, rd.vertex_buf.slice(..));
                    overlay_pass.draw(0..rd.vertex_count, 0..1);
                }
                if let Some(ref lb) = self.loading_bar_gpu_data {
                    overlay_pass.set_bind_group(0, &lb.bind_group, &[]);
                    overlay_pass.set_vertex_buffer(0, lb.vertex_buf.slice(..));
                    overlay_pass.draw(0..lb.vertex_count, 0..1);
                }
            }
            // Phase 7 : overlay images drawn last inside the overlay pass.
            if !self.overlay_image_gpu_data.is_empty() {
                if let Some(pipeline) = &self.resources.screen_image_pipeline {
                    overlay_pass.set_pipeline(pipeline);
                    for gpu in &self.overlay_image_gpu_data {
                        overlay_pass.set_bind_group(0, &gpu.bind_group, &[]);
                        overlay_pass.draw(0..6, 0..1);
                    }
                }
            }
        }

        // Phase 4 : resolve timestamp queries -> staging buffer (HDR path).
        if let (Some(qs), Some(res_buf), Some(stg_buf)) = (
            self.ts_query_set.as_ref(),
            self.ts_resolve_buf.as_ref(),
            self.ts_staging_buf.as_ref(),
        ) {
            encoder.resolve_query_set(qs, 0..2, res_buf, 0);
            encoder.copy_buffer_to_buffer(res_buf, 0, stg_buf, 0, 16);
            self.ts_needs_readback = true;
        }

        encoder.finish()
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
                pixel.swap(0, 2); // B ↔ R
            }
        }

        pixels
    }

    // ------------------------------------------------------------------
    // Backdrop blur helpers
    // ------------------------------------------------------------------

    /// Ensure the backdrop blur state textures exist at the right size.
    fn ensure_backdrop_blur_state(
        &mut self,
        device: &wgpu::Device,
        w: u32,
        h: u32,
    ) {
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
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
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
                size: wgpu::Extent3d { width: blur_w, height: blur_h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
        queue: &wgpu::Queue,
        source_view: &wgpu::TextureView,
        spread: f32,
    ) -> wgpu::BindGroup {
        let bs = self.backdrop_blur_state.as_ref().unwrap();
        let blur_bgl = self.resources.backdrop_blur_bgl.as_ref().unwrap();
        let blur_sampler = self.resources.backdrop_blur_sampler.as_ref().unwrap();
        let blur_pipeline = self.resources.backdrop_blur_pipeline.as_ref().unwrap();
        // Reuse dyn_res blit pipeline and BGL for the downsample pass.
        let blit_pipeline = self.resources.dyn_res_upscale_pipeline.as_ref().unwrap();
        let blit_bgl = self.resources.dyn_res_upscale_bgl.as_ref().unwrap();
        let blit_sampler = self.resources.dyn_res_linear_sampler.as_ref().unwrap();

        // Step 1: downsample source -> blur_a (half-res) using bilinear blit.
        let downsample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop_downsample_bg"),
            layout: blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(blit_sampler) },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("backdrop_downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &bs.blur_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
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
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bs.blur_a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(blur_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: h_uniform.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("backdrop_blur_h"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &bs.blur_b_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
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
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bs.blur_b_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(blur_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: v_uniform.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("backdrop_blur_v"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &bs.blur_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
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
        let tex_bgl = self.resources.overlay_shape_tex_bgl.as_ref().unwrap();
        let tex_sampler = self.resources.overlay_shape_tex_sampler.as_ref().unwrap();
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop_blur_overlay_bg"),
            layout: tex_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bs.blur_a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(tex_sampler) },
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
                    &self.resources.overlay_shape_tex_pipeline,
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
