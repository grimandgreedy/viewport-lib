use super::{FrameData, SceneEffects, ViewportId, ViewportRenderer};

/// Rendering path for applications that own the window loop and wgpu encoder.
///
/// Obtained from [`ViewportRenderer::owned`]. The renderer encodes all GPU work
/// and returns a [`wgpu::CommandBuffer`] for the caller to submit.
///
/// Use this with winit, raw wgpu, or any setup where you control the encoder.
pub struct OwnedPath<'r> {
    pub(super) renderer: &'r mut ViewportRenderer,
}

/// Rendering path for framework callbacks that provide a render pass.
///
/// Obtained from [`ViewportRenderer::pass`]. Prepare encodes pre-pass work
/// and returns command buffers to submit before the surface pass. Paint draws
/// into the caller-provided render pass.
///
/// Use this with eframe (`CallbackTrait`), iced, or any framework that hands
/// you a render pass. For the eframe `paint` callback where only a `&` reference
/// to the renderer is available, use [`ViewportRenderer::pass_view`] instead.
pub struct PassPath<'r> {
    pub(super) renderer: &'r mut ViewportRenderer,
}

/// Read-only paint view returned by [`ViewportRenderer::pass_view`].
///
/// Identical to [`PassPath`] but constructed from a shared reference, making it
/// usable in framework paint callbacks where only `&` access to the renderer is
/// available (e.g. eframe's `CallbackTrait::paint`).
pub struct PassView<'r> {
    pub(super) renderer: &'r ViewportRenderer,
}

impl<'r> OwnedPath<'r> {
    /// Render a single viewport. Runs prepare internally and returns a command buffer to submit.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        self.renderer.render(device, queue, output_view, frame)
    }

    /// Multi-viewport: upload scene-level GPU data shared across all viewports.
    /// Call once per frame before any `prepare_viewport` or `render_viewport` calls.
    pub fn prepare_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        scene_effects: &SceneEffects<'_>,
    ) {
        self.renderer.prepare_scene(device, queue, frame, scene_effects);
    }

    /// Multi-viewport: upload per-viewport GPU data for `id`.
    /// Call after `prepare_scene` and before `render_viewport`.
    pub fn prepare_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ViewportId,
        frame: &FrameData,
    ) {
        self.renderer.prepare_viewport(device, queue, id, frame);
    }

    /// Multi-viewport: encode draw calls for `id` into its own command buffer.
    /// Call after `prepare_scene` and `prepare_viewport` for this id.
    pub fn render_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        id: ViewportId,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        self.renderer.render_viewport(device, queue, output_view, id, frame)
    }
}

impl<'r> PassPath<'r> {
    /// Single viewport: upload uniforms and encode any HDR pre-pass work.
    ///
    /// Returns command buffers that must be submitted before the surface render
    /// pass begins. Returns an empty vec for LDR with no dynamic resolution.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
    ) -> Vec<wgpu::CommandBuffer> {
        self.renderer.prepare_callback(device, queue, frame)
    }

    /// Single viewport: issue draw calls into `render_pass`.
    ///
    /// Call after `prepare` for the same frame. Dispatches to the HDR blit,
    /// dynamic-resolution blit, or direct draw based on which path `prepare`
    /// activated. The render pass is expected to have a depth-stencil attachment,
    /// as provided by eframe and most GUI frameworks.
    pub fn paint<'rp>(&self, render_pass: &mut wgpu::RenderPass<'rp>, frame: &FrameData) {
        self.renderer.paint_callback(render_pass, frame);
    }

    /// Multi-viewport: upload scene-level GPU data shared across all viewports.
    pub fn prepare_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        scene_effects: &SceneEffects<'_>,
    ) {
        self.renderer.prepare_scene(device, queue, frame, scene_effects);
    }

    /// Multi-viewport: upload per-viewport GPU data for `id`.
    pub fn prepare_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ViewportId,
        frame: &FrameData,
    ) {
        self.renderer.prepare_viewport(device, queue, id, frame);
    }

    /// Multi-viewport: issue draw calls for `id` into `render_pass`.
    ///
    /// Set the render pass viewport and scissor rect to the correct region
    /// before calling. Call after `prepare_scene` and `prepare_viewport`.
    pub fn paint_viewport<'rp>(
        &self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        id: ViewportId,
        frame: &FrameData,
    ) {
        self.renderer.paint_viewport_to(render_pass, id, frame);
    }

    /// Multi-viewport HDR: encode the full HDR pipeline for `id` into an intermediate texture.
    ///
    /// Call after `prepare_scene` and `prepare_viewport` for this viewport. Returns a
    /// `CommandBuffer` that eframe must submit before the egui render pass begins.
    ///
    /// Call [`PassView::paint_hdr_blit`] from `CallbackTrait::paint` to composite the
    /// result into the egui render pass. Set the render pass viewport and scissor rect first.
    pub fn prepare_hdr_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ViewportId,
        frame: &FrameData,
    ) -> wgpu::CommandBuffer {
        self.renderer.prepare_hdr_callback_viewport(device, queue, id, frame)
    }
}

impl<'r> PassView<'r> {
    /// Single viewport: issue draw calls into `render_pass`.
    pub fn paint<'rp>(&self, render_pass: &mut wgpu::RenderPass<'rp>, frame: &FrameData) {
        self.renderer.paint_callback(render_pass, frame);
    }

    /// Multi-viewport: issue draw calls for `id` into `render_pass`.
    pub fn paint_viewport<'rp>(
        &self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        id: ViewportId,
        frame: &FrameData,
    ) {
        self.renderer.paint_viewport_to(render_pass, id, frame);
    }

    /// Multi-viewport HDR: blit the intermediate texture for `frame`'s viewport into `render_pass`.
    ///
    /// Call after [`PassPath::prepare_hdr_viewport`] for the same viewport. Set the render
    /// pass viewport and scissor rect to the correct region before calling.
    pub fn paint_hdr_blit<'rp>(&self, render_pass: &mut wgpu::RenderPass<'rp>, frame: &FrameData) {
        self.renderer.paint_hdr_blit(render_pass, frame);
    }
}
