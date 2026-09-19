//! Per-viewport finalization passes: the debug fragment buffer and the
//! atlas blit uniform.

use super::*;

impl ViewportRenderer {
    pub(super) fn prepare_debug_buffer(&mut self, frame: &FrameData) {
        {
            let vp_idx = frame.camera.viewport_index;
            // Physical pixels: the viewport size the debug readback is bounded
            // by is the one the HDR target was allocated at.
            let ppp = frame.camera.pixels_per_point;
            let vw = (frame.camera.viewport_size[0] * ppp).max(1.0) as u32;
            let vh = (frame.camera.viewport_size[1] * ppp).max(1.0) as u32;
            let debug_active = frame.effects.debug.debug_vis.active;

            // Whether this frame will leave the debug quantity in the HDR
            // texture, where `read_debug_pixel` can read it back after the depth
            // test has chosen a winner. That needs the HDR path (the LDR path
            // renders straight into the caller's target, which the renderer does
            // not own) and `Replace` (the other modes blend the quantity with
            // the shaded colour, so what lands there is not the quantity).
            let readable = debug_active
                && frame.effects.display.mode == crate::renderer::types::PipelineMode::Hdr
                && frame.effects.debug.debug_vis.mode
                    == crate::renderer::types::DebugOutputMode::Replace;
            self.viewport_slots[vp_idx].debug_readback_dims = readable.then_some((vw, vh));
        }
    }

    pub(super) fn prepare_atlas_blit(
        &mut self,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        viewport_fx: &ViewportEffects<'_>,
    ) {
        // Atlas blit uniform: compute NDC rect for the corner overlay.
        if viewport_fx.debug.show_shadow_atlas {
            let vw = frame.camera.viewport_size[0].max(1.0);
            let vh = frame.camera.viewport_size[1].max(1.0);
            let scale = viewport_fx.debug.atlas_viewer_scale.clamp(0.05, 1.0);
            // Atlas is square. Width in NDC = scale * 2. Height preserves pixel aspect.
            let ndc_w = scale * 2.0;
            let ndc_h = ndc_w * (vw / vh);
            let margin_x = 20.0 / vw * 2.0;
            let margin_y = 20.0 / vh * 2.0;
            #[allow(unreachable_patterns)]
            let rect = match viewport_fx.debug.atlas_viewer_corner {
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
