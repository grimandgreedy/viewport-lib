use super::paths::ScenePreparedToken;
use super::{FrameData, OwnedPath, PassPath, SceneEffects, ViewportId};

/// A borrowed `(device, queue)` pair, passed as one argument.
///
/// The renderer's prepare/render calls need both a `Device` and a `Queue`.
/// `GpuContext` bundles the two references so a caller that holds them together
/// can pass one value instead of threading two through every call. Build it from
/// a tuple (`(&device, &queue).into()`) or with [`GpuContext::new`], then use the
/// `_ctx`-suffixed methods on [`OwnedPath`] and [`PassPath`].
///
/// It is a thin borrow, not an owner: cheap to build per call and tied to the
/// lifetime of the two references. The plain `(device, queue)` methods stay
/// available; the `_ctx` variants delegate to them.
#[derive(Clone, Copy)]
pub struct GpuContext<'a> {
    device: &'a crate::gpu::Device,
    queue: &'a crate::gpu::Queue,
}

impl<'a> GpuContext<'a> {
    /// Bundle a device and queue reference.
    pub fn new(device: &'a crate::gpu::Device, queue: &'a crate::gpu::Queue) -> Self {
        Self { device, queue }
    }

    /// The device reference.
    pub fn device(&self) -> &'a crate::gpu::Device {
        self.device
    }

    /// The queue reference.
    pub fn queue(&self) -> &'a crate::gpu::Queue {
        self.queue
    }
}

impl<'a> From<(&'a crate::gpu::Device, &'a crate::gpu::Queue)> for GpuContext<'a> {
    fn from((device, queue): (&'a crate::gpu::Device, &'a crate::gpu::Queue)) -> Self {
        Self { device, queue }
    }
}

impl<'r> OwnedPath<'r> {
    /// [`render`](Self::render) taking a [`GpuContext`] instead of separate
    /// device and queue references.
    pub fn render_ctx<'g>(
        &mut self,
        gpu: impl Into<GpuContext<'g>>,
        output_view: &crate::gpu::TextureView,
        frame: &FrameData,
    ) -> crate::gpu::CommandBuffer {
        let gpu = gpu.into();
        self.render(gpu.device(), gpu.queue(), output_view, frame)
    }

    /// [`prepare_scene`](Self::prepare_scene) taking a [`GpuContext`].
    pub fn prepare_scene_ctx<'g>(
        &mut self,
        gpu: impl Into<GpuContext<'g>>,
        frame: &FrameData,
        scene_effects: &SceneEffects<'_>,
    ) -> ScenePreparedToken {
        let gpu = gpu.into();
        self.prepare_scene(gpu.device(), gpu.queue(), frame, scene_effects)
    }

    /// [`prepare_viewport`](Self::prepare_viewport) taking a [`GpuContext`].
    pub fn prepare_viewport_ctx<'g>(
        &mut self,
        gpu: impl Into<GpuContext<'g>>,
        token: &ScenePreparedToken,
        id: ViewportId,
        frame: &FrameData,
    ) {
        let gpu = gpu.into();
        self.prepare_viewport(gpu.device(), gpu.queue(), token, id, frame);
    }

    /// [`render_viewport`](Self::render_viewport) taking a [`GpuContext`].
    pub fn render_viewport_ctx<'g>(
        &mut self,
        gpu: impl Into<GpuContext<'g>>,
        output_view: &crate::gpu::TextureView,
        id: ViewportId,
        frame: &FrameData,
    ) -> crate::gpu::CommandBuffer {
        let gpu = gpu.into();
        self.render_viewport(gpu.device(), gpu.queue(), output_view, id, frame)
    }
}

impl<'r> PassPath<'r> {
    /// [`prepare`](Self::prepare) taking a [`GpuContext`] instead of separate
    /// device and queue references.
    pub fn prepare_ctx<'g>(
        &mut self,
        gpu: impl Into<GpuContext<'g>>,
        frame: &FrameData,
    ) -> Vec<crate::gpu::CommandBuffer> {
        let gpu = gpu.into();
        self.prepare(gpu.device(), gpu.queue(), frame)
    }

    /// [`prepare_scene`](Self::prepare_scene) taking a [`GpuContext`].
    pub fn prepare_scene_ctx<'g>(
        &mut self,
        gpu: impl Into<GpuContext<'g>>,
        frame: &FrameData,
        scene_effects: &SceneEffects<'_>,
    ) -> ScenePreparedToken {
        let gpu = gpu.into();
        self.prepare_scene(gpu.device(), gpu.queue(), frame, scene_effects)
    }

    /// [`prepare_viewport`](Self::prepare_viewport) taking a [`GpuContext`].
    pub fn prepare_viewport_ctx<'g>(
        &mut self,
        gpu: impl Into<GpuContext<'g>>,
        token: &ScenePreparedToken,
        id: ViewportId,
        frame: &FrameData,
    ) {
        let gpu = gpu.into();
        self.prepare_viewport(gpu.device(), gpu.queue(), token, id, frame);
    }

    /// [`prepare_hdr_viewport`](Self::prepare_hdr_viewport) taking a [`GpuContext`].
    pub fn prepare_hdr_viewport_ctx<'g>(
        &mut self,
        gpu: impl Into<GpuContext<'g>>,
        id: ViewportId,
        frame: &FrameData,
    ) -> crate::gpu::CommandBuffer {
        let gpu = gpu.into();
        self.prepare_hdr_viewport(gpu.device(), gpu.queue(), id, frame)
    }
}
