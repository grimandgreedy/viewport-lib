//! The external-render-pass paint entry. Emits the draw calls for the LDR
//! path directly into a render pass owned by the host (iced, or egui via the
//! paint callback).

use super::*;

impl ViewportRenderer {
    pub(crate) fn paint_to<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &FrameData,
    ) {
        // The foreground pass needs its own cleared depth attachment, which a
        // host-owned render pass cannot provide (attachments are fixed at
        // begin_render_pass). Warn once instead of silently drawing nothing.
        if !frame.scene.foreground_items.is_empty()
            && !self
                .foreground_paint_to_warned
                .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::warn!(
                "foreground_items are not drawn by paint()/paint_viewport(): the foreground \
                 pass needs its own cleared depth attachment, which a host-owned render pass \
                 cannot provide. Use the HDR path (post-processing enabled) or an \
                 owned-encoder render path."
            );
        }
        let vp_idx = frame.camera.viewport_index;
        let camera_bg = self.viewport_camera_bind_group(vp_idx);
        let grid_bg = self.viewport_grid_bind_group(vp_idx);
        let vp_slot = self.viewport_slots.get(vp_idx);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.instancing.use_instancing,
            &self.instancing.batches,
            camera_bg,
            grid_bg,
            &self.compute_filter_results,
            vp_slot,
            &self.mesh_uniforms.wireframe_bind_groups,
            &self.mesh_uniforms.bind_groups,
            &self.mesh_uniforms.submesh_bind_groups,
            &self.mesh_uniforms.object_indices,
            &self.mesh_uniforms.submesh_indices,
            &self.prepared_surfaces,
            self.per_object_bundle.as_ref()
        );
        self.draw_line_and_instance_layers(&mut *render_pass, camera_bg, false);
        // TransparentVolumeMesh boundary wireframe overlay.
        if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
            if let Some(ref tvm_bg) = self.mesh_uniforms.tvm_wireframe_bg {
                render_pass.set_bind_group(0, camera_bg, &[]);
                for mesh_id in &self.mesh_uniforms.tvm_wireframe_draws {
                    if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                        render_pass.set_pipeline(&self.resources.scene.wireframe);
                        bind_deform_group!(
                            render_pass,
                            self.resources,
                            &self.resources.deform.dummy_bind_group
                        );
                        render_pass.set_bind_group(1, tvm_bg, &[]);
                        render_pass.set_vertex_buffer(
                            0,
                            self.resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        if let Some(edge_buf) = &mesh.edge_index_buffer {
                            render_pass.set_index_buffer(
                                edge_buf.slice(..),
                                crate::gpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
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
        // Sub-object highlight (LDR path) : face fill, edge lines, vertex/point sprites.
        if let Some(sub_hl) = self
            .viewport_slots
            .get(vp_idx)
            .and_then(|s| s.sub_highlight.as_ref())
        {
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
        emit_overlay_2d!(self, render_pass);
        // Shadow atlas viewer overlay.
        if frame.effects.debug.show_shadow_atlas {
            render_pass.set_pipeline(&self.resources.shadow.atlas_viewer_pipeline);
            render_pass.set_bind_group(0, &self.resources.shadow.atlas_viewer_bg, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }
}
