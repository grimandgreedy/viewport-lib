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
            &self.mesh_instance_gpu_data,
            false
        );
        // Gaussian splats (alpha-blended, back-to-front sorted, no depth write).
        if !self.gaussian_splat_draw_data.is_empty() {
            if let Some(ref dual) = self.resources.gaussian_splat.pipeline {
                render_pass.set_pipeline(dual.for_format(false));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for dd in &self.gaussian_splat_draw_data {
                    if dd.wireframe {
                        continue;
                    }
                    if let Some(set) = self
                        .resources
                        .content
                        .gaussian_splat_store
                        .get_by_index(dd.store_index)
                    {
                        if let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) {
                            render_pass.set_bind_group(1, &vp_sort.render_bg, &[]);
                            render_pass.draw(0..6, 0..dd.count);
                        }
                    }
                }
            }
        }
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
        // GPU implicit surface (depth-writes enabled, LessEqual compare).
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
            render_pass.set_bind_group(0, camera_bg, &[]);
            for mc in &self.mc_gpu_data {
                let vol = &self.resources.mc.volumes[mc.volume_idx];
                if mc.wireframe || frame.viewport.wireframe_mode {
                    if let Some(ref dual) = self.resources.mc.wireframe_pipeline {
                        render_pass.set_pipeline(dual.for_format(false));
                        for (slab, wire_bg) in vol.slabs.iter().zip(mc.wire_slab_bgs.iter()) {
                            render_pass.set_bind_group(1, wire_bg, &[]);
                            render_pass.draw_indirect(&slab.wire_indirect_buf, 0);
                        }
                    }
                } else if let Some(ref dual) = self.resources.mc.surface_pipeline {
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
        // Screen-space image overlays (always on top, no depth test).
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
        // Shadow atlas viewer overlay.
        if frame.effects.debug.show_shadow_atlas {
            render_pass.set_pipeline(&self.resources.shadow.atlas_viewer_pipeline);
            render_pass.set_bind_group(0, &self.resources.shadow.atlas_viewer_bg, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }
}
