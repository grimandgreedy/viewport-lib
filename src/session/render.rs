//! Render forwards and picking.
//!
//! Both render shapes are exposed because the split is dictated by who owns the
//! surface, not by feature richness: [`render`](ViewportSession::render) for a
//! host that owns the swapchain view (winit, iced), and
//! [`prepare`](ViewportSession::prepare) + [`paint`](ViewportSession::paint) for
//! a host that only hands you a render pass (eframe). HDR versus LDR is dispatched
//! internally by the underlying paths off `effects.post_process.enabled`; the
//! caller never chooses a method by path.

use super::ViewportSession;
use crate::{PickBackend, PickHit, PickMask};

impl ViewportSession {
    /// Render the assembled frame into `view` and return the command buffer for
    /// the host to submit. For a host that owns the swapchain texture view.
    pub fn render(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        view: &crate::gpu::TextureView,
    ) -> crate::gpu::CommandBuffer {
        self.renderer
            .owned()
            .render(device, queue, view, &self.frame)
    }

    /// Encode any pre-pass work (HDR chain, dynamic resolution) and return the
    /// command buffers to submit before the surface render pass. Pair with
    /// [`paint`](Self::paint). For a render-pass host such as eframe.
    pub fn prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) -> Vec<crate::gpu::CommandBuffer> {
        self.renderer.pass().prepare(device, queue, &self.frame)
    }

    /// Issue draw calls into a render pass the host provides. Call after
    /// [`prepare`](Self::prepare) for the same frame.
    pub fn paint<'rp>(&self, render_pass: &mut crate::gpu::RenderPass<'rp>) {
        self.renderer.pass_view().paint(render_pass, &self.frame);
    }

    /// Pick the object under `screen` (viewport-local pixels) using the CPU
    /// backend. Device-free and synchronous; returns `None` until the first
    /// frame has been rendered (the CPU pick cache is filled during `prepare`).
    pub fn pick(&self, screen: glam::Vec2, mask: PickMask) -> Option<PickHit> {
        let viewport_size = glam::Vec2::from(self.viewport_size);
        let view_proj = self.camera.view_proj_matrix();
        self.renderer.pick(screen, viewport_size, view_proj, mask)
    }

    /// Pick under `screen` using the GPU backend, which resolves sub-object
    /// levels the CPU backend cannot. Needs the device and queue because it does
    /// a GPU readback.
    pub fn pick_gpu(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        screen: glam::Vec2,
        mask: PickMask,
    ) -> Option<PickHit> {
        self.renderer
            .pick_object(PickBackend::Gpu, screen, &self.frame, device, queue, mask)
    }
}
