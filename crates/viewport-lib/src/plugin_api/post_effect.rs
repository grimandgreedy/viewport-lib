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
//! - A [`PostEffectStage`] runs after tone mapping, in display space,
//!   chained with the built-in FXAA by an explicit order key: each stage
//!   reads the previous stage's output and writes the next stage's input,
//!   and the last stage writes the frame's final target.
//!
//! Producers and stages register on
//! [`ViewportRenderer`](crate::renderer::ViewportRenderer) with
//! [`add_post_effect_producer`](crate::renderer::ViewportRenderer::add_post_effect_producer)
//! and
//! [`add_post_effect_stage`](crate::renderer::ViewportRenderer::add_post_effect_stage),
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

/// Handle returned by
/// [`add_post_effect_stage`](crate::renderer::ViewportRenderer::add_post_effect_stage);
/// pass it to
/// [`remove_post_effect_stage`](crate::renderer::ViewportRenderer::remove_post_effect_stage)
/// to unregister.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PostEffectStageId(pub(crate) u64);

/// Well-known order keys for the post-effect stage chain.
///
/// Stages run in ascending order of the `i32` key given at registration;
/// stages sharing a key run in registration order, with built-in stages
/// first. Keys below [`ANTI_ALIASING`] run before the built-in FXAA (still
/// on tone-mapped LDR input); keys above run after it.
pub mod stage_order {
    /// The built-in FXAA stage's position in the chain.
    pub const ANTI_ALIASING: i32 = 0;
    /// The conventional band for external stages: after anti-aliasing, so
    /// FXAA does not soften the stage's output. Space related stages
    /// around this value.
    pub const EXTERNAL_DEFAULT: i32 = 100;
}

/// Per-frame, per-viewport inputs handed to a producer's
/// [`prepare`](PostEffectProducer::prepare) and
/// [`encode`](PostEffectProducer::encode).
#[non_exhaustive]
pub struct PostEffectContext<'a> {
    /// The wgpu device, for bind groups or buffers an effect creates at
    /// frame time (per-viewport allocations belong in
    /// [`on_viewport_resized`](PostEffectProducer::on_viewport_resized)).
    pub device: &'a crate::gpu::Device,
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
/// and bind groups against these dimensions and views.
///
/// The scene views here are the same objects the per-frame
/// [`PostEffectContext`] carries, and they live until the next resize
/// signal for this viewport, so bind groups built against them here stay
/// valid between signals: no per-frame rebuilding is needed.
#[non_exhaustive]
pub struct PostEffectResizeContext<'a> {
    /// Which viewport was (re)created.
    pub viewport_index: usize,
    /// Scene-resolution target size in pixels.
    pub scene_size: [u32; 2],
    /// Output (native) target size in pixels.
    pub output_size: [u32; 2],
    /// The viewport's HDR scene colour target (scene resolution, linear).
    pub scene_colour: &'a crate::gpu::TextureView,
    /// Depth-only view of the viewport's scene depth target, sampleable.
    pub scene_depth: &'a crate::gpu::TextureView,
    /// The renderer's LDR target format. Stage input textures and stage
    /// pipelines must use it; build format-dependent pipelines on the
    /// first resize signal rather than in `init_gpu`.
    pub target_format: crate::gpu::TextureFormat,
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
pub trait PostEffectProducer: Send + Sync + 'static {
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
    ///
    /// The returned view must be sampleable with a filtering sampler (any
    /// filterable float format works; the built-ins use `Rgba16Float` for
    /// bloom and `R8Unorm` for the single-channel slots). When both this
    /// producer and the built-in effect for the slot are active, this view
    /// wins and a once-per-slot debug log records the conflict; switch the
    /// built-in off in [`PostProcessSettings`](crate::PostProcessSettings)
    /// when replacing it.
    fn encode<'a>(
        &'a mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        ctx: &PostEffectContext<'_>,
    ) -> Option<&'a crate::gpu::TextureView>;
}

/// A display-space pass that runs after the tone-map composite, chained
/// with the built-in FXAA by the order key given at registration.
///
/// The chain routes targets: whichever stage (or the composite itself)
/// runs before this one renders into [`input_view`](Self::input_view), and
/// this stage's [`encode`](Self::encode) reads that input and writes the
/// `target` it is handed (the next stage's input, or the frame's final
/// target). No blits and no ping-pong management are needed: owning one
/// input texture per viewport is the whole contract.
///
/// The input texture must match the renderer's LDR target format
/// ([`DeviceResources::target_format`](crate::resources::DeviceResources::target_format)),
/// be scene-sized (`scene_size` from the resize context: the chain runs at
/// scene resolution when dynamic-resolution scaling is active), and carry
/// `RENDER_ATTACHMENT | TEXTURE_BINDING` usage. Allocate it in
/// [`on_viewport_resized`](Self::on_viewport_resized).
///
/// Lifecycle matches [`PostEffectProducer`]: deferred [`init_gpu`]
/// (build pipelines), [`on_viewport_resized`] (allocate the per-viewport
/// input), then per HDR frame [`prepare`] and [`encode`] while
/// [`enabled`](Self::enabled) returns true.
///
/// [`init_gpu`]: Self::init_gpu
/// [`on_viewport_resized`]: Self::on_viewport_resized
/// [`prepare`]: Self::prepare
/// [`encode`]: Self::encode
pub trait PostEffectStage: Send + Sync + 'static {
    /// Stable identifying name, used in diagnostics and pass labels.
    fn type_name(&self) -> &'static str;

    /// Whether the stage runs this frame. Stages carry their own settings;
    /// the host mutates them through its own handle to the stage's state.
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
    /// per-viewport input texture and bind groups for `ctx.viewport_index`
    /// here.
    fn on_viewport_resized(
        &mut self,
        _device: &crate::gpu::Device,
        _ctx: &PostEffectResizeContext<'_>,
    ) {
    }

    /// Write this frame's uniforms. Runs before any pass of the frame is
    /// encoded.
    fn prepare(&mut self, _queue: &crate::gpu::Queue, _ctx: &PostEffectContext<'_>) {}

    /// The stage's input for `viewport_index`: whoever runs before this
    /// stage in the chain renders into this view.
    fn input_view(&self, viewport_index: usize) -> &crate::gpu::TextureView;

    /// Encode the stage's pass: read [`input_view`](Self::input_view) for
    /// `ctx.viewport_index`, write `target`.
    fn encode(
        &mut self,
        encoder: &mut crate::gpu::CommandEncoder,
        target: &crate::gpu::TextureView,
        ctx: &PostEffectContext<'_>,
    );
}

/// Build a fullscreen post-effect pipeline: the fixed shape shared by
/// every post pass (three-vertex fullscreen triangle with `vs_main` /
/// `fs_main` entry points, one bind group layout at group 0, no
/// depth-stencil, single-sampled, no culling).
///
/// A free function taking only the device so it is callable from
/// [`init_gpu`](PostEffectProducer::init_gpu) and
/// [`on_viewport_resized`](PostEffectProducer::on_viewport_resized). See
/// [`shared_wgsl::POST_EFFECT_VS_WGSL`](crate::plugin_api::shared_wgsl::POST_EFFECT_VS_WGSL)
/// for a ready-made vertex stage.
///
/// `target_format` is the format of the view the pass renders into: the
/// producer's own texture format for slot textures, or the resize
/// context's [`target_format`](PostEffectResizeContext::target_format)
/// for a stage pass.
pub fn build_post_effect_pipeline(
    device: &crate::gpu::Device,
    label: &str,
    shader: &crate::gpu::ShaderModule,
    bind_group_layout: &crate::gpu::BindGroupLayout,
    target_format: crate::gpu::TextureFormat,
    blend: Option<crate::gpu::BlendState>,
) -> crate::gpu::RenderPipeline {
    let layout = crate::resources::builders::pipeline_layout(
        device,
        format!("{label}_layout").as_str(),
        &[bind_group_layout],
    );
    crate::resources::builders::build_fullscreen_pipeline(
        device,
        label,
        &layout,
        shader,
        target_format,
        blend,
    )
}
