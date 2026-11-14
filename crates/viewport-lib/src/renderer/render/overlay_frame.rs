//! The overlays-only frame: a frame whose scene submits nothing draws its
//! background and its overlay pass, and skips scene prepare and the scene pass.

use super::*;

impl ViewportRenderer {
    /// Whether `frame` is drawn as an overlays-only frame: scene prepare is
    /// skipped and the render step encodes only the overlay pass, cleared to
    /// the frame's background.
    ///
    /// The frame must draw through the LDR path (`PipelineMode::Direct`, or a
    /// renderer below the full tier, which draws every frame that way), and
    /// nothing may want the scene pass:
    ///
    /// - the scene submits no surfaces, volume meshes, mesh instances,
    ///   foreground items or world overlay layers, and every item-type
    ///   collection on it is empty;
    /// - on the full tier, the frame shows no grid, ground plane or shadow
    ///   atlas viewer;
    /// - the interaction frame carries no constraint guides and no sub-object
    ///   selection;
    /// - no runtime GPU plugin reads the scene colour and depth after the
    ///   frame (a runner with GPU plugins keeps the scene depth).
    ///
    /// The LDR path draws such a frame as the background colour under the
    /// overlays, so the two paths produce the same image. An environment
    /// background is not drawn on the LDR path, so it does not stop a frame
    /// from qualifying. `FrameStats::frame_path` reports which path ran.
    pub(crate) fn draws_overlays_only(&self, frame: &FrameData) -> bool {
        if self.renders_hdr(frame) || self.keep_scene_depth {
            return false;
        }
        if Self::has_mesh_content(frame)
            || !frame.scene.world_overlays.is_empty()
            || frame
                .scene
                .plugin_items
                .values()
                .flatten()
                .any(|c| c.len() > 0)
        {
            return false;
        }
        if self.resources.scene_tier.is_built()
            && (frame.viewport.show_grid
                || !matches!(
                    frame.effects.ground_plane.mode,
                    crate::renderer::types::GroundPlaneMode::None
                )
                || frame.effects.debug.show_shadow_atlas)
        {
            return false;
        }
        frame.interaction.constraint_overlays.is_empty()
            && frame.interaction.sub_selection.is_none()
    }

    /// The path `frame` is drawn through, as `FrameStats::frame_path` reports it.
    pub(crate) fn frame_path(&self, frame: &FrameData) -> crate::renderer::stats::FramePath {
        if self.renders_hdr(frame) {
            crate::renderer::stats::FramePath::Hdr
        } else if self.draws_overlays_only(frame) {
            crate::renderer::stats::FramePath::OverlaysOnly
        } else {
            crate::renderer::stats::FramePath::Ldr
        }
    }

    /// Zero the counters scene prepare writes, for a frame that skipped it.
    pub(crate) fn clear_scene_stats(&mut self) {
        self.last_lights_dropped = 0;
        self.last_stats = crate::renderer::stats::FrameStats {
            total_objects: 0,
            visible_objects: 0,
            culled_objects: 0,
            draw_calls: 0,
            instanced_batches: 0,
            per_object_bind_groups_built: 0,
            batches_reuploaded: 0,
            batches_skipped: 0,
            plugin_draw_calls: 0,
            plugin_upload_bytes: 0,
            deform_buffer_reallocations: 0,
            triangles_submitted: 0,
            shadow_draw_calls: 0,
            shadow_draw_commands: 0,
            shadow_buffer_binds: 0,
            shadow_cascade_passes: 0,
            shadow_bounds_rebuilds: 0,
            punctual_shadow_layers: 0,
            punctual_shadow_layer_passes: 0,
            cap_meshes_built: 0,
            per_object_bundle_cached: false,
            lod_items_resolved: 0,
            lod_switches: 0,
            lod_culled: 0,
            lod_items_reduced: 0,
            gpu_culling_active: false,
            gpu_visible_instances: None,
            gpu_culled_total: None,
            gpu_frustum_visible: None,
            ..self.last_stats
        };
    }

    /// Encode an overlays-only frame into `output_view`: one pass that clears
    /// to `bg_colour` and draws the overlays, or, when the overlays read the
    /// destination, the overlay passes with their breaks, the first of them
    /// clearing. The overlays draw at full resolution whatever the render
    /// scale, since there is no scene to scale.
    pub(crate) fn render_frame_overlays(
        &mut self,
        device: &crate::gpu::Device,
        output_view: &crate::gpu::TextureView,
        vp_idx: usize,
        frame: &FrameData,
        bg_colour: [f32; 4],
        w: u32,
        h: u32,
        mut encoder: crate::gpu::CommandEncoder,
    ) -> crate::gpu::CommandEncoder {
        let _span =
            tracing::debug_span!(target: "viewport_lib::frame", "overlay_frame", vp = vp_idx)
                .entered();
        self.report_auto_exposure_on_ldr(frame);
        // Nothing drew into the scene depth this frame, so occlusion culling
        // must not reproject an older one.
        self.viewport_slots[vp_idx].cull.forget_hiz_prev_depth();

        // Screen overlays drawn over a reference presentation's upscale are
        // not drawn here.
        let overlays = frame.overlays_in_scene();
        let has_breaks = overlays && self.viewport_slots[vp_idx].overlay.has_breaks();
        // Overlays that read the destination of an output that cannot be
        // copied draw into the intermediate target, blitted to the output.
        let intermediate = has_breaks
            && !self.viewport_slots[vp_idx]
                .overlay
                .layer_targets
                .can_snapshot(&output_view.texture());
        if intermediate {
            self.ensure_backdrop_blur_state(device, w.max(1), h.max(1));
        }
        let depth = self.ldr_depth_view(vp_idx).clone();
        if has_breaks {
            let (target, scene) = if intermediate {
                let view = self
                    .backdrop_blur_state
                    .as_ref()
                    .unwrap()
                    .intermediate_view
                    .clone();
                (view.clone(), Some(view))
            } else {
                (output_view.clone(), None)
            };
            self.draw_overlay_passes(
                &mut encoder,
                device,
                vp_idx,
                &target,
                &depth,
                crate::gpu::StoreOp::Discard,
                scene.as_ref(),
                Some(bg_colour),
                frame.overlay_space().1,
                "overlay_frame_pass",
            );
        } else {
            let ts = self.ts_writes_for(crate::renderer::GPU_TS_OVERLAY, true, true);
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("overlay_frame_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color {
                            r: bg_colour[0] as f64,
                            g: bg_colour[1] as f64,
                            b: bg_colour[2] as f64,
                            a: bg_colour[3] as f64,
                        }),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(crate::gpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(crate::gpu::Operations {
                        load: crate::gpu::LoadOp::Clear(crate::plugin_api::depth::DEPTH_CLEAR),
                        store: crate::gpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: ts,
                occlusion_query_set: None,
            });
            if overlays {
                let ov = &self.viewport_slots[vp_idx].overlay;
                emit_overlay_2d!(self, ov, pass);
            }
        }
        self.resolve_prev_timestamps(&mut encoder);
        if intermediate {
            self.blit_backdrop_intermediate(&mut encoder, device, output_view);
        }
        encoder
    }
}
