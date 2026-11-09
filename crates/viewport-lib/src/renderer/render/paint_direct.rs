//! The external-render-pass paint entry. Emits the draw calls for the LDR
//! path directly into a render pass owned by the host (iced, or egui via the
//! paint callback).

use super::*;

impl ViewportRenderer {
    /// The LDR pipeline has no HDR target to meter, so an automatic exposure
    /// draws at the passthrough exposure (the camera uniform carries 1.0).
    /// Report it rather than drawing differently without a word.
    pub(crate) fn report_auto_exposure_on_ldr(&self, frame: &FrameData) {
        if frame.effects.display.exposure.manual_multiplier().is_none() {
            self.resources.notices.report(
                crate::notice::NoticeKey::single(
                    crate::NoticeCategory::Pipeline,
                    "auto_exposure_on_ldr",
                ),
                crate::Severity::Warning,
                || crate::NoticeDetail::AutoExposureOnLdr,
            );
        }
    }

    pub(crate) fn paint_to<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        self.paint_slot(render_pass, frame.camera.viewport_index, frame);
    }

    /// Paint viewport slot `vp_idx` into a host-owned render pass. Shared by
    /// `paint_to` and `paint_viewport_to`, so a viewport painted through the
    /// multi-viewport API draws everything a single viewport does.
    pub(crate) fn paint_slot<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        vp_idx: usize,
        frame: &FrameData,
    ) {
        let _span =
            tracing::debug_span!(target: "viewport_lib::frame", "paint", vp = vp_idx).entered();
        // The foreground pass needs its own cleared depth attachment, which a
        // host-owned render pass cannot provide (attachments are fixed at
        // begin_render_pass). Report it instead of silently drawing nothing.
        if !frame.scene.foreground_items.is_empty() {
            self.resources.notices.report(
                crate::notice::NoticeKey::single(
                    crate::NoticeCategory::Pipeline,
                    "foreground_unsupported_on_path",
                ),
                crate::Severity::Warning,
                || crate::NoticeDetail::ForegroundUnsupportedOnPath,
            );
        }
        self.report_auto_exposure_on_ldr(frame);
        // An overlays-only frame draws its overlays and nothing else: they
        // ignore depth, so the host's depth needs no reset.
        let overlays_only = self.draws_overlays_only(frame);
        // The host cleared this pass's depth to a value of its own; reset the
        // viewport rect to the renderer's clear value before drawing.
        if let (false, Some(reset)) = (
            overlays_only,
            &self.resources.post.host_depth_reset_pipeline,
        ) {
            render_pass.set_pipeline(reset);
            render_pass.draw(0..3, 0..1);
        }
        if overlays_only {
            // Nothing of the scene to draw.
        } else if self.resources.scene_tier.is_built() {
            self.paint_slot_scene(render_pass, vp_idx, frame);
        } else if let Some(slot) = self.viewport_slots.get(vp_idx) {
            self.paint_compatibility_scene(render_pass, &slot.camera_bind_group, frame);
            emit_outline_composite!(&self.resources, &mut *render_pass, Some(slot));
            self.paint_ldr_sub_highlight(render_pass, vp_idx, &slot.camera_bind_group);
        }
        // A pass the host owns cannot be ended at a break: no destination
        // reads, so layers draw with their fallback blend and backdrops are
        // skipped.
        let ov = &self.viewport_slots[vp_idx].overlay;
        emit_overlay_segments(
            &self.resources,
            ov,
            render_pass,
            ov.main_segments(),
            &OverlayEmit::NO_DESTINATION,
        );
        // Shadow atlas viewer overlay.
        if let (true, true, Some(pipeline)) = (
            self.resources.scene_tier.is_built(),
            frame.effects.debug.show_shadow_atlas,
            self.resources
                .guide_pipeline(crate::resources::overlay::guides::GUIDE_ATLAS_VIEWER),
        ) {
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, &self.resources.scene_tier.shadow.atlas_viewer_bg, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }

    /// The scene half of `paint_slot`: everything drawn before the overlays.
    fn paint_slot_scene<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        vp_idx: usize,
        frame: &FrameData,
    ) {
        let camera_bg = self.viewport_camera_bind_group(vp_idx);
        let grid_bg = self.viewport_grid_bind_group(vp_idx);
        let vp_slot = self.viewport_slots.get(vp_idx);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            &self.instancing.batches,
            camera_bg,
            grid_bg,
            vp_slot,
            &self.mesh_uniforms.material_bind_groups,
            &self.prepared_surfaces,
            self.per_object_bundle.as_ref()
        );
        self.draw_mesh_instance_batches(&mut *render_pass, camera_bg);
        // TransparentVolumeMesh boundary wireframe overlay.
        if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
            if let Some(ref tvm_bg) = self.mesh_uniforms.tvm_wireframe_bg {
                render_pass.set_bind_group(0, camera_bg, &[]);
                for (slot, mesh_id) in self.mesh_uniforms.tvm_wireframe_draws.iter().enumerate() {
                    if let (Some(mesh), Some(wf)) = (
                        self.resources.mesh_store.get(*mesh_id),
                        self.resources.scene.wireframe(),
                    ) {
                        render_pass.set_pipeline(wf);
                        render_pass.set_bind_group(1, tvm_bg, &[]);
                        if let Some(records) = &self.mesh_uniforms.tvm_wireframe_buf {
                            render_pass.set_vertex_buffer(1, records.slice(..));
                        }
                        render_pass.set_vertex_buffer(
                            0,
                            self.resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        if let Some(edge_buf) = &mesh.edge_index_buffer {
                            render_pass.set_index_buffer(
                                edge_buf.slice(..),
                                crate::gpu::IndexFormat::Uint32,
                            );
                            let slot = slot as u32;
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, slot..slot + 1);
                        }
                    }
                }
            }
        }
        // Item-type plugin paint (LDR opt-in only): after all built-in scene
        // content, mirroring the HDR scene-pass position.
        self.dispatch_plugin_paint(render_pass, frame, false);
        // Outline composite after all scene content so translucent layers don't overdraw.
        emit_outline_composite!(&self.resources, &mut *render_pass, vp_slot);
        // Sub-object highlight: face fill, edge lines, vertex/point sprites.
        self.paint_ldr_sub_highlight(render_pass, vp_idx, camera_bg);
    }
}

impl crate::renderer::ViewportRenderer {
    /// The sub-object highlight in an LDR scene pass: the selected faces'
    /// fill, their edges as lines and the selected vertices and points as
    /// sprites, drawn over the scene and the outline. `camera_bg` is the
    /// viewport's group 0.
    pub(crate) fn paint_ldr_sub_highlight(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        vp_idx: usize,
        camera_bg: &crate::gpu::BindGroup,
    ) {
        let Some(sub_hl) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.sub_highlight.as_ref())
        else {
            return;
        };
        let (Some(fill_pl), Some(edge_pl), Some(sprite_pl)) = (
            &self.resources.sub_highlight.fill_ldr_pipeline,
            &self.resources.sub_highlight.edge_ldr_pipeline,
            &self.resources.sub_highlight.sprite_ldr_pipeline,
        ) else {
            return;
        };
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
