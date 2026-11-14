//! Reference presentation: the scene drawn into a target of a fixed size,
//! then scaled into the output with nearest sampling, and the screen overlays
//! drawn over the result at the output's own resolution.

use super::*;

impl ViewportRenderer {
    /// Ensure `vp_idx` has a reference target of `size` pixels.
    pub(crate) fn ensure_reference_target(
        &mut self,
        device: &crate::gpu::Device,
        vp_idx: usize,
        size: [u32; 2],
    ) {
        self.ensure_viewport_slot(device, vp_idx);
        self.resources.ensure_reference_present_pipeline(device);
        let current = self.viewport_slots[vp_idx]
            .reference
            .as_ref()
            .map(|r| r.size);
        if current != Some(size) {
            let target = self.resources.create_reference_target(device, size);
            self.viewport_slots[vp_idx].reference = Some(target);
        }
    }

    /// Scale `vp_idx`'s reference target into `output_view` by `placement`,
    /// then draw the screen overlays over it when they are not drawn into the
    /// reference target.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_reference_present(
        &mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: &crate::gpu::TextureView,
        vp_idx: usize,
        frame: &FrameData,
        placement: &crate::renderer::ReferencePlacement,
    ) {
        let _span =
            tracing::debug_span!(target: "viewport_lib::frame", "reference_present", vp = vp_idx)
                .entered();
        let (bars, bar_colour) = match frame.viewport.presentation {
            crate::renderer::Presentation::Reference {
                fit: crate::renderer::ReferenceFit::IntegerBars { colour },
                ..
            } => {
                let c = colour.to_linear_rgba();
                (1.0, [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]])
            }
            _ => (0.0, [0.0; 4]),
        };
        let uniform = crate::resources::gpu::reference::PresentUniform {
            origin_px: placement.origin_px,
            scale: placement.scale,
            reference_size: [
                placement.reference_size[0] as f32,
                placement.reference_size[1] as f32,
            ],
            bars,
            _pad: 0.0,
            bar_colour,
        };
        {
            let target = self.viewport_slots[vp_idx]
                .reference
                .as_mut()
                .expect("the scene drew into the reference target");
            if target.written != Some(uniform) {
                queue.write_buffer(&target.uniform_buf, 0, bytemuck::bytes_of(&uniform));
                target.written = Some(uniform);
            }
        }

        let ov = &self.viewport_slots[vp_idx].overlay;
        let overlays = !frame.overlays_in_scene()
            && (ov.overlay_shape_gpu_data.is_some()
                || ov.label_gpu_data.is_some()
                || !ov.overlay_draw_segments.is_empty());
        let has_breaks = overlays && ov.has_breaks();
        // Overlays that read the destination of an output that cannot be
        // copied draw into the intermediate target, blitted to the output.
        let intermediate = has_breaks && !ov.layer_targets.can_snapshot(&output_view.texture());
        let [ow, oh] = placement.viewport_px;
        if intermediate {
            self.ensure_backdrop_blur_state(device, ow, oh);
        }
        let target = if intermediate {
            self.backdrop_blur_state
                .as_ref()
                .unwrap()
                .intermediate_view
                .clone()
        } else {
            output_view.clone()
        };

        {
            let reference = self.viewport_slots[vp_idx].reference.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&crate::gpu::RenderPassDescriptor {
                #[cfg(any(wgpu29, wgpu30))]
                multiview_mask: None,
                label: Some("reference_present_pass"),
                color_attachments: &[Some(crate::gpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    ops: crate::gpu::Operations {
                        // The present covers every pixel.
                        load: crate::gpu::LoadOp::Clear(crate::gpu::Color::TRANSPARENT),
                        store: crate::gpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some(pipeline) = &self.resources.post.reference_present_pipeline {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &reference.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        if overlays {
            self.ensure_reference_output_depth(device, vp_idx, ow, oh);
            let depth = self.viewport_slots[vp_idx]
                .reference
                .as_ref()
                .and_then(|r| r.output_depth.as_ref())
                .expect("just ensured")
                .view
                .clone();
            let scene = intermediate.then(|| target.clone());
            self.draw_overlay_passes(
                encoder,
                device,
                vp_idx,
                &target,
                &depth,
                crate::gpu::StoreOp::Discard,
                scene.as_ref(),
                None,
                frame.camera.pixels_per_point,
                "reference_overlay_pass",
            );
        }
        if intermediate {
            self.blit_backdrop_intermediate(encoder, device, output_view);
        }
    }

    /// Ensure the depth target the overlay pass over a reference
    /// presentation's output draws against is `w`x`h`.
    fn ensure_reference_output_depth(
        &mut self,
        device: &crate::gpu::Device,
        vp_idx: usize,
        w: u32,
        h: u32,
    ) {
        let depth_format = self.resources.scene_depth_format();
        let Some(reference) = self.viewport_slots[vp_idx].reference.as_mut() else {
            return;
        };
        let current = reference
            .output_depth
            .as_ref()
            .map(|d| (d.texture.width(), d.texture.height()));
        if current != Some((w, h)) {
            let alloc = crate::resources::postprocess::targets::ViewportTargetAllocator::new(
                device,
                w,
                h,
                w,
                h,
                1,
                depth_format,
            );
            reference.output_depth = Some(alloc.depth(
                "reference_output_depth",
                crate::resources::postprocess::targets::TargetSize::Scene,
            ));
        }
    }

    /// The reference pixel position under the viewport position `window`
    /// (logical pixels from the viewport's top-left corner, as picks take
    /// them) for `frame`'s presentation.
    ///
    /// `None` when `window` lies in the bars around the reference target or
    /// outside the viewport. Under [`Presentation::Native`](crate::Presentation::Native)
    /// the viewport is its own reference and `window` comes back unchanged.
    pub fn window_to_reference(&self, frame: &FrameData, window: glam::Vec2) -> Option<glam::Vec2> {
        match frame.reference_placement() {
            Some(p) => p.window_to_reference(window),
            None => Some(window),
        }
    }

    /// The viewport position (logical pixels) of the reference pixel position
    /// `reference` for `frame`'s presentation, for placing UI over reference
    /// pixels.
    ///
    /// `None` when `reference` lies outside the reference target or where the
    /// fit crops it out of the viewport. Under
    /// [`Presentation::Native`](crate::Presentation::Native) `reference` comes
    /// back unchanged.
    pub fn reference_to_window(
        &self,
        frame: &FrameData,
        reference: glam::Vec2,
    ) -> Option<glam::Vec2> {
        match frame.reference_placement() {
            Some(p) => p.reference_to_window(reference),
            None => Some(reference),
        }
    }
}
