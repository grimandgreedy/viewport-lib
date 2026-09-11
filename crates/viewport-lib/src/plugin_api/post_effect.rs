//! Post-effect extension points: external producers (and, alongside them,
//! external stages) that plug into the renderer's post-processing chain.
//!
//! Two extension points cover the two shapes a post effect can take:
//!
//! - A [`PostEffectProducer`] runs before tone mapping. It reads the HDR
//!   scene colour and depth, encodes its own passes into the frame, and
//!   contributes the result to one named composite input
//!   ([`PostEffectSlot`]): the same slots the built-in bloom, SSAO, contact
//!   shadows, and surface LIC fill. Registering a producer for a slot while
//!   the corresponding built-in is switched off replaces that effect's
//!   implementation; the composite treats the two identically.
//! - A stage runs after tone mapping, in display space, chained with the
//!   built-in FXAA: each stage reads the previous stage's output and writes
//!   the next stage's input, and the last stage writes the frame's final
//!   target.
//!
//! Producers and stages register on
//! [`ViewportRenderer`](crate::renderer::ViewportRenderer) with
//! [`add_post_effect_producer`](crate::renderer::ViewportRenderer::add_post_effect_producer),
//! because their passes are encoded inside the renderer's own frame, at
//! positions that only exist mid-encode. This is the same placement rule
//! that puts [`ItemTypePlugin`](crate::plugin_api::ItemTypePlugin) on the
//! renderer. Work that only *reads* the finished frame (export taps,
//! host-composited overlays) belongs on
//! [`GpuPlugin::post_paint`](crate::runtime::GpuPlugin::post_paint) instead;
//! work that *changes* the frame belongs here.
//!
//! Producers own their GPU resources: pipelines and shared state are built
//! in [`init_gpu`](PostEffectProducer::init_gpu), per-viewport textures in
//! [`on_viewport_resized`](PostEffectProducer::on_viewport_resized). The
//! renderer never allocates targets on a producer's behalf. Host-driven
//! settings flow through the producer's own state (mutated via whatever
//! handle the host keeps), not through fields on
//! [`PostProcessSettings`](crate::PostProcessSettings):
//! [`enabled`](PostEffectProducer::enabled) is the producer's own gate.
//!
//! Producers run only on the HDR render path (post-processing enabled).
//! They read the scene targets, never another producer's output; effects
//! that need history keep it across frames in their own textures.

/// The named composite inputs an external producer may fill or replace.
///
/// Each variant corresponds to one input of the tone-map composite. A
/// producer declares its slot once via
/// [`PostEffectProducer::slot`]; the view it returns from
/// [`encode`](PostEffectProducer::encode) is bound there for the frame.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PostEffectSlot {
    /// Additive glow, sampled at half resolution and added to the scene
    /// colour before tone mapping.
    Bloom,
    /// Ambient-occlusion factor (single channel, white = unoccluded),
    /// multiplied into the scene colour.
    AmbientOcclusion,
    /// Screen-space contact-shadow factor (single channel, white = lit),
    /// multiplied into the scene colour.
    ContactShadow,
    /// Surface line-integral-convolution intensity, modulated over surfaces
    /// that submitted LIC vector data.
    SurfaceLic,
}

/// Handle returned by
/// [`add_post_effect_producer`](crate::renderer::ViewportRenderer::add_post_effect_producer);
/// pass it to
/// [`remove_post_effect_producer`](crate::renderer::ViewportRenderer::remove_post_effect_producer)
/// to unregister.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PostEffectProducerId(pub(crate) u64);

/// Per-frame, per-viewport inputs handed to a producer's
/// [`prepare`](PostEffectProducer::prepare) and
/// [`encode`](PostEffectProducer::encode).
#[non_exhaustive]
pub struct PostEffectContext<'a> {
    /// Which viewport this frame belongs to. Matches
    /// [`CameraFrame::viewport_index`](crate::CameraFrame); producers key
    /// per-viewport state on it.
    pub viewport_index: usize,
    /// Scene-resolution target size in pixels (render scale applied).
    pub scene_size: [u32; 2],
    /// Output (native) target size in pixels.
    pub output_size: [u32; 2],
    /// This viewport's projection matrix.
    pub proj: glam::Mat4,
    /// This viewport's view matrix.
    pub view: glam::Mat4,
    /// Camera near plane distance.
    pub near: f32,
    /// Camera far plane distance.
    pub far: f32,
    /// The HDR scene colour target (scene resolution, linear).
    pub scene_colour: &'a crate::gpu::TextureView,
    /// Depth-only view of the scene depth target, sampleable.
    pub scene_depth: &'a crate::gpu::TextureView,
    /// The frame's post-process settings, read-only, for coordinating with
    /// the built-in effects. A producer's own switches live on the producer.
    pub post: &'a crate::PostProcessSettings,
}

/// Passed to [`PostEffectProducer::on_viewport_resized`] when a viewport's
/// render targets are (re)created: allocate or resize per-viewport textures
/// against these dimensions.
#[non_exhaustive]
pub struct PostEffectResizeContext<'a> {
    /// Which viewport was (re)created.
    pub viewport_index: usize,
    /// Scene-resolution target size in pixels.
    pub scene_size: [u32; 2],
    /// Output (native) target size in pixels.
    pub output_size: [u32; 2],
    pub(crate) _reserved: std::marker::PhantomData<&'a ()>,
}

/// A pre-tone-map post effect that computes into its own texture and
/// contributes the result to one named composite input.
///
/// Lifecycle: [`init_gpu`](Self::init_gpu) runs once when the renderer first
/// has the device after registration (build pipelines and shared state
/// here); [`on_viewport_resized`](Self::on_viewport_resized) runs for each
/// viewport whose targets are (re)created (allocate per-viewport textures
/// here); then every HDR frame runs [`prepare`](Self::prepare) (uniform
/// writes) followed by [`encode`](Self::encode) (pass encoding) for each
/// viewport where [`enabled`](Self::enabled) returns true.
pub trait PostEffectProducer: Send + 'static {
    /// Stable identifying name, used in diagnostics and pass labels.
    fn type_name(&self) -> &'static str;

    /// The composite input this producer fills. Declared once; a producer
    /// contributes to exactly one slot.
    fn slot(&self) -> PostEffectSlot;

    /// Whether the effect runs this frame. Producers carry their own
    /// settings; the host mutates them through its own handle to the
    /// producer's state.
    fn enabled(&self) -> bool;

    /// Build pipelines, layouts, and shared (viewport-independent) GPU
    /// state. Called once after registration, when the renderer first runs
    /// with the device, and again after device recreation.
    fn init_gpu(&mut self, _device: &crate::gpu::Device) {}

    /// The wgpu device was recreated (device loss, surface re-init). All
    /// previously created GPU resources are invalid; `init_gpu` is called
    /// again after this, followed by `on_viewport_resized` for each live
    /// viewport.
    fn on_device_recreated(&mut self, _device: &crate::gpu::Device, _queue: &crate::gpu::Queue) {}

    /// A viewport's render targets were (re)created. Allocate or resize the
    /// per-viewport textures and bind groups for `ctx.viewport_index` here.
    fn on_viewport_resized(
        &mut self,
        _device: &crate::gpu::Device,
        _ctx: &PostEffectResizeContext<'_>,
    ) {
    }

    /// Write this frame's uniforms. Runs before any pass of the frame is
    /// encoded.
    fn prepare(&mut self, _queue: &crate::gpu::Queue, _ctx: &PostEffectContext<'_>) {}

    /// Encode the producer's passes into the frame, and return the view to
    /// bind at [`slot`](Self::slot) for this viewport this frame. `None`
    /// leaves the slot to the built-in effect (or its disabled placeholder).
    fn encode<'a>(
        &'a mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &PostEffectContext<'_>,
    ) -> Option<&'a crate::gpu::TextureView>;
}
