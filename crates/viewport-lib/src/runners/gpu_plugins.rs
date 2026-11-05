//! The attached runtime's GPU plugin hooks, called from the render forwards.

use super::ViewportInstance;
use crate::renderer::ViewportId;

impl ViewportInstance {
    /// The viewport this instance renders. Pass it to
    /// [`ViewportRuntime::add_gpu_plugin_for_viewport`](crate::ViewportRuntime::add_gpu_plugin_for_viewport)
    /// to scope a GPU plugin to this instance's viewport.
    ///
    /// [`render`](Self::render) and [`prepare`](Self::prepare) call the
    /// runtime's GPU hooks with this id, so plugins scoped to it run once per
    /// render, as unscoped ones do, and plugins scoped to any other id do not
    /// run. A host that renders more viewports through
    /// [`renderer_mut`](Self::renderer_mut) calls
    /// [`ViewportRuntime::pre_prepare`](crate::ViewportRuntime::pre_prepare)
    /// itself for each of them, with
    /// [`GpuFrameContext::with_viewport`](crate::GpuFrameContext::with_viewport);
    /// unscoped plugins run on those calls as well.
    pub fn viewport_id(&self) -> ViewportId {
        ViewportId(self.frame.camera.viewport_index)
    }

    /// Whether an attached runtime has GPU plugins to run this frame.
    pub(super) fn has_gpu_plugins(&self) -> bool {
        self.runtime
            .as_ref()
            .is_some_and(|runtime| runtime.gpu_plugin_count() > 0)
    }

    /// Run the attached runtime's GPU plugins' `pre_prepare` for the assembled
    /// frame and submit their command buffers on `queue`, so they reach the
    /// GPU ahead of the renderer's prepare passes, which are also submitted
    /// inline. Does nothing without a runtime, without GPU plugins, or once
    /// the device is lost.
    pub(super) fn run_gpu_pre_prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        if !self.has_gpu_plugins() || self.renderer.is_device_lost() {
            return;
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let camera = &self.frame.camera;
        let ctx = crate::GpuFrameContext::new(
            &self.camera,
            physical_size(camera),
            self.runtime_dt,
            self.gpu_frame_index,
        )
        .with_viewport(ViewportId(camera.viewport_index));
        let buffers = runtime.pre_prepare(device, queue, &ctx);
        if !buffers.is_empty() {
            queue.submit(buffers);
        }
    }

    /// Run the attached runtime's GPU plugins' `post_paint` with the scene
    /// colour and depth the frame just drew, and return their command buffers
    /// for submission after the frame's. `output_view` is the view an owned
    /// render drew into, `None` on the pass path. Where the frame left nothing
    /// readable the hook is not called and the renderer raises a notice.
    pub(super) fn run_gpu_post_paint(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        output_view: Option<&crate::gpu::TextureView>,
    ) -> Vec<crate::gpu::CommandBuffer> {
        if !self.has_gpu_plugins() || self.renderer.is_device_lost() {
            return Vec::new();
        }
        if let Some(reason) = self
            .renderer
            .post_paint_skip_reason(&self.frame, output_view)
        {
            self.renderer.report_post_paint_skipped(reason);
            return Vec::new();
        }
        let Some(targets) = self.renderer.scene_target_views(&self.frame, output_view) else {
            return Vec::new();
        };
        let Some(runtime) = self.runtime.as_mut() else {
            return Vec::new();
        };
        let camera = &self.frame.camera;
        let ctx = crate::GpuFrameContext::new(
            &self.camera,
            physical_size(camera),
            self.runtime_dt,
            self.gpu_frame_index,
        )
        .with_viewport(ViewportId(camera.viewport_index));
        runtime.post_paint(device, queue, &targets, &ctx)
    }

    /// Count a rendered frame for the GPU plugins' `frame_index`.
    pub(super) fn end_gpu_frame(&mut self) {
        if self.has_gpu_plugins() {
            self.gpu_frame_index += 1;
        }
    }

    /// The scene colour and depth the last frame drew, as a GPU plugin's
    /// `post_paint` receives them. `output_view` is the view the frame was
    /// rendered into, for an LDR frame.
    #[doc(hidden)]
    pub fn scene_target_views<'a>(
        &'a self,
        output_view: Option<&'a crate::gpu::TextureView>,
    ) -> Option<crate::runtime::PostPaintTargets<'a>> {
        self.renderer.scene_target_views(&self.frame, output_view)
    }
}

/// The viewport size in physical pixels, as the renderer sizes its targets.
fn physical_size(camera: &crate::CameraFrame) -> glam::Vec2 {
    let ppp = camera.pixels_per_point;
    glam::Vec2::new(
        (camera.viewport_size[0] * ppp).round(),
        (camera.viewport_size[1] * ppp).round(),
    )
}
