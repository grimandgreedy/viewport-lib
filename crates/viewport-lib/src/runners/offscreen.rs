//! Offscreen colour target for hosting a viewport inside a 2D UI compositor.
//!
//! When you render a viewport straight to a window surface (the winit path), the
//! surface is an sRGB format and the GPU applies the final linear->sRGB encode on
//! present, so colours are correct. When you instead render into a texture that a
//! UI framework (egui, iced, ...) samples and composites, that encode has to
//! survive the sampler, and the obvious setup gets it wrong: the renderer outputs
//! linear light and relies on the target format for the encode, but an sRGB
//! texture that the compositor samples is decoded straight back to linear on read,
//! cancelling the encode. The image then reads about a gamma-2.2 too dark.
//!
//! [`OffscreenViewportTarget`] holds one texture with two views over it: an sRGB
//! **render view** the renderer draws into (so the encode happens on write) and a
//! non-sRGB **sample view** the compositor samples (so it reads the already-encoded
//! bytes verbatim instead of decoding them). Build your renderer with
//! [`OffscreenViewportTarget::render_format`] so its pipelines target the sRGB
//! format, draw into [`render_view`](OffscreenViewportTarget::render_view), and
//! register [`sample_view`](OffscreenViewportTarget::sample_view) with the
//! compositor.
//!
//! This is a plain `wgpu` helper with no UI-framework dependency; the one
//! framework-specific step, registering the sample view, stays on your side (e.g.
//! `egui_wgpu::Renderer::register_native_texture`).

/// A colour texture rendered by the viewport and sampled by a UI compositor, with
/// the sRGB dual-view wired so the tonemap's linear output is encoded exactly once.
///
/// ```no_run
/// # use viewport_lib::{OffscreenViewportTarget, ViewportInstance, wgpu};
/// # fn setup(device: &wgpu::Device, surface_format: wgpu::TextureFormat) {
/// // Build the renderer with the sRGB render format so the final encode happens.
/// let session = ViewportInstance::new(device, OffscreenViewportTarget::render_format(surface_format));
/// let mut target = OffscreenViewportTarget::new(device, surface_format, [1280, 720]);
/// // Each frame: resize if needed, and re-register with the compositor when the
/// // texture was recreated.
/// if target.resize(device, [1280, 720]) {
///     // re-register `target.sample_view()` with your compositor here
/// }
/// // Render into `target.render_view()`, then draw `target.sample_view()` as an image.
/// # let _ = (session, &target);
/// # }
/// ```
pub struct OffscreenViewportTarget {
    _texture: crate::gpu::Texture,
    render_view: crate::gpu::TextureView,
    sample_view: crate::gpu::TextureView,
    size: [u32; 2],
    render_format: crate::gpu::TextureFormat,
    sample_format: crate::gpu::TextureFormat,
}

impl OffscreenViewportTarget {
    /// The colour format to build your `ViewportInstance` / `ViewportRenderer`
    /// with: the sRGB variant of the compositor's `surface_format`. Rendering into
    /// this format is what makes the GPU apply the linear->sRGB encode on write.
    ///
    /// Pass the same `surface_format` you pass to [`new`](Self::new) (for egui,
    /// `RenderState::target_format`). If it is already sRGB this returns it
    /// unchanged.
    pub fn render_format(surface_format: crate::gpu::TextureFormat) -> crate::gpu::TextureFormat {
        surface_format.add_srgb_suffix()
    }

    /// Create a target of `size` (physical pixels) for a compositor whose surface
    /// is `surface_format`. `size` components are clamped up to at least 1.
    pub fn new(
        device: &crate::gpu::Device,
        surface_format: crate::gpu::TextureFormat,
        size: [u32; 2],
    ) -> Self {
        let render_format = surface_format.add_srgb_suffix();
        let sample_format = surface_format.remove_srgb_suffix();
        let size = [size[0].max(1), size[1].max(1)];

        let texture = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("viewport_offscreen_target"),
            size: crate::gpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: render_format,
            usage: crate::gpu::TextureUsages::RENDER_ATTACHMENT
                | crate::gpu::TextureUsages::TEXTURE_BINDING,
            // Both views are created from this texture; list both formats so the
            // non-sRGB sample view is allowed.
            view_formats: &[render_format, sample_format],
        });
        let render_view = texture.create_view(&crate::gpu::TextureViewDescriptor {
            format: Some(render_format),
            ..Default::default()
        });
        let sample_view = texture.create_view(&crate::gpu::TextureViewDescriptor {
            format: Some(sample_format),
            ..Default::default()
        });

        Self {
            _texture: texture,
            render_view,
            sample_view,
            size,
            render_format,
            sample_format,
        }
    }

    /// Recreate the texture and views if `size` differs from the current size.
    /// Returns `true` when the texture was recreated, in which case
    /// [`sample_view`](Self::sample_view) is a new view and must be re-registered
    /// with the compositor; `false` means the existing registration is still valid.
    pub fn resize(&mut self, device: &crate::gpu::Device, size: [u32; 2]) -> bool {
        let size = [size[0].max(1), size[1].max(1)];
        if size == self.size {
            return false;
        }
        // Rebuild from the same surface format. `render_format` is the sRGB variant
        // already, and `add_srgb_suffix` is idempotent, so it round-trips.
        *self = Self::new(device, self.sample_format, size);
        true
    }

    /// The sRGB view to render into. Pass this to the renderer's paint/render call.
    pub fn render_view(&self) -> &crate::gpu::TextureView {
        &self.render_view
    }

    /// The non-sRGB view to hand the compositor (e.g. via
    /// `egui_wgpu::Renderer::register_native_texture`).
    pub fn sample_view(&self) -> &crate::gpu::TextureView {
        &self.sample_view
    }

    /// Current size in physical pixels.
    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// The sRGB colour format the renderer must target (same value as
    /// [`render_format`](Self::render_format) for this target's surface format).
    pub fn format(&self) -> crate::gpu::TextureFormat {
        self.render_format
    }
}
