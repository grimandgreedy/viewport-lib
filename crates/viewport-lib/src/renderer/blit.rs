//! Blit a colour texture into a rect of a caller-owned render pass.
//!
//! This completes the [`OffscreenViewportTarget`](crate::OffscreenViewportTarget)
//! story for hosts that do not use a UI framework's texture registration (egui's
//! `register_native_texture`, iced's image widget). Render a viewport into an
//! offscreen target, then draw the target into a sub-rect of the window surface:
//!
//! ```no_run
//! # use viewport_lib::{OffscreenViewportTarget, ViewportRenderer, wgpu};
//! # fn compose(
//! #     renderer: &mut ViewportRenderer,
//! #     device: &wgpu::Device,
//! #     offscreen: &OffscreenViewportTarget,
//! #     rp: &mut wgpu::RenderPass,
//! #     rect: [f32; 4],
//! # ) {
//! // Once per target (rebuild when OffscreenViewportTarget::resize returns true):
//! let blit = renderer.create_blit(device, offscreen.render_view());
//! // Each frame, inside your surface render pass:
//! let [x, y, w, h] = rect;
//! rp.set_viewport(x, y, w, h, 0.0, 1.0);
//! rp.set_scissor_rect(x as u32, y as u32, w as u32, h as u32);
//! renderer.blit(rp, &blit);
//! # }
//! ```
//!
//! Which view to pass: the blit samples the source as linear and lets the render
//! pass's colour target apply the final encode. For an `OffscreenViewportTarget`
//! that means passing its sRGB `render_view()`, so the sampler decodes to linear
//! and the sRGB surface re-encodes exactly once. Passing the non-sRGB
//! `sample_view()` (raw encoded bytes) would double-encode and read too dark;
//! that view is for compositors like egui that sample without re-encoding.
//!
//! The render pass colour target must use the format the renderer was built with
//! (`ViewportRenderer::new`'s format, i.e. `OffscreenViewportTarget::render_format`
//! for the surface): the blit pipeline is compiled for that one format.

use super::ViewportRenderer;

/// A colour texture prepared for [`ViewportRenderer::blit`].
///
/// Build once with [`ViewportRenderer::create_blit`] and reuse it every frame.
/// Rebuild it whenever the source `TextureView` is recreated (for an
/// `OffscreenViewportTarget`, when `resize` returns `true`).
pub struct BlitTexture {
    bind_group: crate::gpu::BindGroup,
}

impl ViewportRenderer {
    /// Prepare `source` for blitting into a render pass.
    ///
    /// `source` is sampled as linear. For an
    /// [`OffscreenViewportTarget`](crate::OffscreenViewportTarget) pass its
    /// `render_view()` (the sRGB view), not `sample_view()`; see the module docs.
    pub fn create_blit(
        &mut self,
        device: &crate::gpu::Device,
        source: &crate::gpu::TextureView,
    ) -> BlitTexture {
        // Ensure both blit pipeline variants exist so blit() and
        // blit_with_depth() are both usable. Both share dyn_res_upscale_bgl and
        // the linear sampler; the ds variant needs the non-ds one built first.
        self.resources.ensure_dyn_res_pipeline(device);
        self.resources.ensure_dyn_res_ds_pipeline(device);

        let bgl = self
            .resources
            .post
            .dyn_res_upscale_bgl
            .as_ref()
            .expect("ensure_dyn_res_pipeline builds dyn_res_upscale_bgl");
        let sampler = self
            .resources
            .post
            .dyn_res_linear_sampler
            .as_ref()
            .expect("ensure_dyn_res_pipeline builds dyn_res_linear_sampler");

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("viewport_blit_bg"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(source),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        BlitTexture { bind_group }
    }

    /// Draw `blit` as a fullscreen triangle into the current viewport/scissor rect
    /// of `rp`. Set the viewport and scissor rect before calling.
    ///
    /// For a render pass without a depth-stencil attachment (a plain surface blit
    /// pass, e.g. winit). For a pass carrying a `Depth24PlusStencil8` attachment
    /// (an eframe paint pass), use [`blit_with_depth`](Self::blit_with_depth).
    pub fn blit<'rp>(&self, rp: &mut crate::gpu::RenderPass<'rp>, blit: &BlitTexture) {
        if let Some(pipeline) = &self.resources.post.dyn_res_upscale_pipeline {
            rp.set_pipeline(pipeline);
            rp.set_bind_group(0, &blit.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
    }

    /// Like [`blit`](Self::blit) but for a render pass that carries a
    /// `Depth24PlusStencil8` depth-stencil attachment (e.g. an eframe paint pass).
    /// The blit itself is identical; only the pipeline variant differs to match
    /// the render pass.
    pub fn blit_with_depth<'rp>(&self, rp: &mut crate::gpu::RenderPass<'rp>, blit: &BlitTexture) {
        if let Some(pipeline) = &self.resources.post.dyn_res_upscale_ds_pipeline {
            rp.set_pipeline(pipeline);
            rp.set_bind_group(0, &blit.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
    }
}
