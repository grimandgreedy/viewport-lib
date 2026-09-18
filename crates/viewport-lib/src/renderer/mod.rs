//! `ViewportRenderer` : the main entry point for the viewport library.
//!
//! Wraps [`DeviceResources`] and provides `prepare()` / `paint()` methods
//! that take raw `wgpu` types. GUI framework adapters (e.g. the egui
//! `CallbackTrait` impl in the application crate) delegate to these methods.

#[macro_use]
mod types;
mod device_lost;
pub use device_lost::{DeviceLostInfo, DeviceLostWatcher};
mod indirect;
mod instancing_state;
use instancing_state::InstancingState;
mod per_object_state;
use per_object_state::PerObjectState;
mod shadow_state;
use shadow_state::ShadowState;
mod blit;
mod paths;
pub use blit::BlitTexture;
pub use capture::{CapturedHdr, CapturedHdrGpu};
pub use paths::{OwnedPath, PassPath, PassView};
mod gpu_context;
pub use gpu_context::GpuContext;
pub(crate) mod item_plugins;
pub(crate) mod picking;
pub use picking::sub_object;
pub use picking::{
    CellSelectionInfo, GpuPickHit, PickBackend, PickHit, PickId, PickMask, PickPoll,
    PickRectResult, PolylineSelectionInfo, SubObjectRef, SubSelection, SubSelectionRef,
    VolumeSelectionInfo,
};
mod capture;
mod overlay_buffers;
mod overlay_draw_order;
mod readback;
pub use readback::ExposureReadback;
// Gaussian splat upload vocabulary lives in `resources`; re-exported here so the
// public `renderer::GaussianSplat*` path and its doc links stay stable.
pub use crate::resources::{GaussianSplatData, GaussianSplatId, ShDegree};
pub(crate) mod pipeline_key;
use pipeline_key::{PipelineKey, select_opaque_solid, select_two_sided};
mod point_shadow_pool;
mod prepare;
mod render;
mod submit;
pub use submit::SubmitSink;
pub mod shader_hashes;
mod shadow_debug_stats;
mod shadows;
pub mod stats;
pub mod tuning;
pub use shadow_debug_stats::ShadowDebugStats;

#[cfg(test)]
mod hidden_tests;
#[cfg(test)]
mod instanced_texture_tests;
#[cfg(test)]
mod lod_instance_tests;

pub use self::types::{
    AnchorX, AnchorY, AnimTrack, AtlasViewerCorner, AutoExposure, BloomSettings, BorderMode,
    CameraFrame, Candela, ClipObject, ClipShape, ComputeFilterItem, ComputeFilterKind,
    ContactShadowSettings, CylindricalFacing, DebugOutputMode, DebugQuantity, DebugVis,
    DecalAnimation, DecalBlendMode, DecalItem, DecalProjection, DisplaySettings, DofSettings,
    EdlSettings, EffectsFrame, EmitterConfig, EnvironmentSettings, ExposureMode, ExposureSettings,
    ExternalInstancesItem, FillRule, FilterMode, ForceField, ForegroundPass, ForegroundProjection,
    FrameData, GaussianSplatItem, GlyphItem, GlyphRunItem, GlyphSetRefItem, GlyphType,
    GpuImplicitItem, GpuImplicitOptions, GpuMarchingCubesItem, GpuParticleSystemItem, GradientStop,
    GroundPlane, GroundPlaneMode, ImageSliceItem, ImplicitBlendMode, ImplicitPrimitive,
    IndirectLightSource, InteractionFrame, LabelAnchor, LabelAnchorY, LabelItem, LerpAnim,
    LicOverlay, LightKind, LightSource, LightingPosture, LightingSettings, LineCap, LineJoin,
    Lumen, Lux, MAX_POINT_SHADOW_LIGHTS, MeshInstanceItem, NineSlice, OVERLAY_MAX_GRADIENT_STOPS,
    OVERLAY_MAX_SHADOW_LAYERS, OverlayAnchor, OverlayAnimation, OverlayAnimations, OverlayEasing,
    OverlayFill, OverlayFrame, OverlayGeometryId, OverlayPolylineItem, OverlayShape,
    OverlayShapeItem, OverlayTextureId, POINT_SHADOW_FACE_SIZE, ParticleMeshAlign, PathSegment,
    PathTrack, PipelineMode, PointCloudItem, PointCloudRefItem, PointRenderMode, PointShadowMode,
    PolylineCap, PolylineItem, PolylineRefItem, PositionedGlyph, PostProcessSettings, RenderCamera,
    RepeatMode, RetainedOverlay, RibbonItem, RibbonRefItem, ScatterQuality, ScatterSettings,
    ScatterVolumeItem, SceneEffects, SceneFrame, SceneRenderItem, ShadowFilter, ShadowLayer,
    ShadowSettings, SliceAxis, SpawnShape, SpriteBlend, SpriteInstanceSetRefItem, SpriteItem,
    SpriteLitParams, SpriteNormalMode, SpriteOrientation, SpriteSetRefItem, SpriteSizeMode,
    StreamtubeItem, StreamtubeRefItem, StrokePattern, SubPath, SurfaceLICConfig, SurfaceSubmission,
    TensorGlyphItem, TensorGlyphSetRefItem, TextureTransform, TileMode, ToneMapping,
    TriangleDirection, TubeItem, TubeRefItem, VelocityDist, ViewportEffects, ViewportFrame,
    VignetteSettings, VolumeItem, VolumeMeshItem, VolumeSurfaceSliceItem, VolumeTransparency,
    aabb_wireframe_polyline, obb_wireframe_polyline, sphere_wireframe_polyline,
};

/// An opaque handle to a per-viewport GPU state slot.
///
/// Obtained from [`ViewportRenderer::create_viewport`] and passed to
/// [`ViewportRenderer::prepare_viewport`], [`ViewportRenderer::paint_viewport`],
/// and [`ViewportRenderer::render_viewport`].
///
/// The slot index is managed internally. To bind a `ViewportId` to a camera frame,
/// use [`CameraFrame::with_viewport_id`]. Single-viewport applications that use
/// the legacy [`ViewportRenderer::prepare`] / [`ViewportRenderer::paint`] API do
/// not need this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewportId(pub(crate) usize);

impl ViewportId {
    /// The slot index this id refers to, matching
    /// [`CameraFrame::viewport_index`](crate::CameraFrame). Useful when tagging a
    /// frame the renderer assembled for you (e.g. via `ViewportInstance`) with the
    /// viewport a readback like [`ViewportRenderer::exposure_state`] should target.
    pub fn index(&self) -> usize {
        self.0
    }
}

use self::shadows::{compute_cascade_matrix, compute_cascade_splits};
use self::types::{INSTANCING_THRESHOLD, InstancedBatch};
use crate::resources::{
    BatchMeta, CLIP_VOLUME_MAX, CameraUniform, ClipPlanesUniform, ClipVolumeEntry,
    ClipVolumesUniform, DeviceResources, GridUniform, InstanceAabb, InstanceData, LightsUniform,
    ObjectUniform, OutlineEdgeUniform, OutlineObjectBuffers, OutlineUniform, PickInstance,
    ShadowAtlasUniform, SingleLightUniform,
};

/// Per-frame selection-outline state for one viewport, rebuilt in prepare().
///
/// Two sources feed the outline mask: the geometry substrate, whose selected
/// surfaces and volume-mesh boundaries get dedicated mask buffers here, and the
/// item-type plugins, which draw their own coverage through
/// [`ItemTypePlugin::outline_mask`](crate::plugin_api::ItemTypePlugin::outline_mask)
/// and are tracked only by the flag below. The mask, edge and composite passes
/// gate on both.
#[derive(Default)]
pub(crate) struct SelectionOutlines {
    /// Per-frame outline buffers for selected objects.
    pub outline_object_buffers: Vec<OutlineObjectBuffers>,
    /// True when an item-type plugin drew selection coverage into the outline
    /// mask this frame. Plugin outline coverage is not tracked in the per-kind
    /// buffers above, so the mask/edge pass and the composite also gate on this.
    pub plugin_outline_present: bool,
}

/// Per-viewport GPU state: uniform buffers and bind groups that differ per viewport.
///
/// Each viewport slot owns its own camera, clip planes, clip volume, shadow info,
/// and grid buffers, plus the bind groups that reference them. Scene-global
/// resources (lights, shadow atlas texture, IBL) are shared via the bind group
/// pointing to buffers on `DeviceResources`.
pub(crate) struct ViewportSlot {
    pub camera_buf: crate::gpu::Buffer,
    pub clip_planes_buf: crate::gpu::Buffer,
    pub clip_volume_buf: crate::gpu::Buffer,
    pub shadow_info_buf: crate::gpu::Buffer,
    pub grid_buf: crate::gpu::Buffer,
    /// Camera bind group (group 0) referencing this slot's per-viewport buffers
    /// plus shared scene-global resources.
    pub camera_bind_group: crate::gpu::BindGroup,
    /// Camera uniform for the foreground pass: the scene view with the
    /// foreground projection (the scene projection, or the override from
    /// `EffectsFrame::foreground`). Written in prepare when foreground work
    /// exists.
    pub foreground_camera_buf: crate::gpu::Buffer,
    /// Zeroed clip uniforms (`count == 0`) so foreground items are never
    /// sliced by scene section planes.
    pub foreground_clip_planes_buf: crate::gpu::Buffer,
    pub foreground_clip_volume_buf: crate::gpu::Buffer,
    /// Group-0 bind group for the foreground pass: same layout and shared
    /// bindings as `camera_bind_group`, but with `foreground_camera_buf` at
    /// binding 0 and the disabled clip buffers at bindings 4/6.
    pub foreground_camera_bind_group: crate::gpu::BindGroup,
    /// Per-item draw resources for this viewport's foreground items,
    /// index-aligned with `SceneFrame::foreground_items`.
    pub foreground_objects: Vec<crate::renderer::per_object_state::ForegroundObjectEntry>,
    /// Grid bind group (group 0 for grid pipeline) referencing this slot's grid buffer.
    pub grid_bind_group: crate::gpu::BindGroup,
    /// Per-viewport HDR post-process render targets.
    ///
    /// Created lazily on first HDR render call and resized when viewport dimensions change.
    pub hdr: Option<crate::resources::ViewportHdrState>,
    /// Per-viewport GPU culling outputs (visibility indices, indirect args,
    /// batch counters, and their bind groups). The cull dispatch for this
    /// viewport's camera writes here; the draw path reads from here.
    pub cull: crate::resources::ViewportCullState,
    /// Viewport dimensions when the last prepared frame left a readable debug
    /// quantity in the HDR texture (debug vis active, HDR path, `Replace`
    /// mode); `None` when it did not. Read by `read_debug_pixel`.
    pub debug_readback_dims: Option<(u32, u32)>,

    // --- Per-viewport interaction state ---
    /// Per-frame selection-outline state, one entry per scene-item kind, rebuilt in prepare().
    pub selection_outlines: SelectionOutlines,
    /// Per-frame x-ray buffers for selected objects, rebuilt in prepare().
    pub xray_object_buffers: Vec<(
        crate::resources::mesh::mesh_store::MeshId,
        crate::gpu::Buffer,
        crate::gpu::BindGroup,
    )>,
    /// Per-frame constraint guide line buffers, rebuilt in prepare().
    pub constraint_line_buffers: Vec<(
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        u32,
        crate::gpu::Buffer,
        crate::gpu::BindGroup,
    )>,
    /// Per-frame cap geometry buffers (section view cross-section fill), rebuilt in prepare().
    pub cap_buffers: Vec<(
        crate::gpu::Buffer,
        crate::gpu::Buffer,
        u32,
        crate::gpu::Buffer,
        crate::gpu::BindGroup,
    )>,
    // --- Sub-object highlight (per-viewport, generation-cached) ---
    /// Per-viewport dynamic resolution intermediate render target.
    /// `None` when render_scale == 1.0 or not yet initialised.
    pub dyn_res: Option<crate::resources::gpu::dyn_res::DynResTarget>,
    /// Per-viewport intermediate render target for the HDR eframe callback path.
    /// `None` until the first `prepare_hdr_callback` call for this viewport.
    pub hdr_callback: Option<crate::resources::gpu::dyn_res::HdrCallbackTarget>,
    /// Cached GPU data for sub-object highlight rendering.
    /// `None` when no sub-object selection is active and no volumes are selected.
    pub sub_highlight: Option<crate::resources::SubHighlightGpuData>,
    /// Version of the last sub-selection snapshot that was uploaded.
    /// `u64::MAX` forces a rebuild on the first frame.
    pub sub_highlight_generation: u64,
}

/// Renderer wrapping all GPU resources and providing `prepare()` and `paint()` methods.

/// GPU timestamp slot for the main opaque HDR scene pass.
pub(crate) const GPU_TS_SCENE: u32 = 0;
/// GPU timestamp slot for the directional shadow depth pass.
pub(crate) const GPU_TS_SHADOW: u32 = 1;
/// GPU timestamp slot for the OIT accumulation pass.
pub(crate) const GPU_TS_OIT: u32 = 2;
/// GPU timestamp slot for the tone-map / resolve pass.
pub(crate) const GPU_TS_POST: u32 = 3;
/// GPU timestamp slot for the main-camera GPU cull dispatch (the
/// `cull_instances` + `write_indirect_args` compute passes). Only the main
/// camera cull is timed; shadow-cascade and single-mesh culls are not.
pub(crate) const GPU_TS_CULL: u32 = 4;
/// Point-light cubemap shadow faces: begin on the first face pass, end on the
/// last, so the slot spans every face rendered this frame.
pub(crate) const GPU_TS_POINT_SHADOW: u32 = 5;
/// Clustered-lighting build compute pass.
pub(crate) const GPU_TS_CLUSTER: u32 = 6;
/// SSAO passes (occlusion + blur), begin on the first and end on the last.
pub(crate) const GPU_TS_SSAO: u32 = 7;
/// Bloom passes (extract + blur chain), begin on the first and end on the last.
pub(crate) const GPU_TS_BLOOM: u32 = 8;
/// FXAA fullscreen pass.
pub(crate) const GPU_TS_FXAA: u32 = 9;
/// The dedicated screen-space overlay pass (shapes, labels, glyph runs,
/// polylines, retained groups). Written only when the overlay runs as its own
/// pass: the HDR path always does, the LDR path only when a backdrop-blur shape
/// forces a second pass, otherwise its draws are inline at the end of the scene
/// pass and counted there.
pub(crate) const GPU_TS_OVERLAY: u32 = 10;
/// The compaction's three dispatches inside the main-camera cull, split out so
/// the cost of packing the visible list in instance order can be attributed to
/// a specific dispatch rather than inferred. These are timestamps taken inside
/// a compute pass, so they need `TIMESTAMP_QUERY_INSIDE_PASSES` on top of
/// `TIMESTAMP_QUERY` and read `0.0` without it. Only the main-camera cull is
/// split; shadow-cascade culls run the same three dispatches untimed.
pub(crate) const GPU_TS_CULL_PLAN: u32 = 11;
pub(crate) const GPU_TS_CULL_COUNT: u32 = 12;
pub(crate) const GPU_TS_CULL_SCATTER: u32 = 13;
/// Number of measured GPU passes; the query set holds `2 * GPU_TS_SLOTS` entries
/// (a begin/end pair per slot).
pub(crate) const GPU_TS_SLOTS: u32 = 14;

/// Whether a `render()` presents the frame the user sees, or is an auxiliary
/// read.
///
/// A `Presented` render owns advancing the per-frame state that only the shown
/// frame should touch: it pumps the upload pipeline, bumps the frame counter,
/// stores HiZ prev-depth for next frame's occlusion reprojection, writes
/// `FrameStats`, and runs item-type plugins' `prepare` / `cull`. A `Derivative`
/// render (a capture / probe bake, and later an offscreen preview) reads the
/// currently resident scene to produce a side output and advances none of that,
/// so it cannot strand a consumer's in-flight upload binds or perturb the
/// presented frame's temporal state.
///
/// Internal, and distinct from the consumer-facing
/// [`RuntimeMode`](crate::renderer::stats::RuntimeMode), which selects render
/// quality, not side-effect behaviour. Consumers never set this; the capture
/// entry points do, for the duration of the capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum RenderMode {
    /// The frame the user sees. Advances per-frame state.
    #[default]
    Presented,
    /// A capture / bake / preview that reads resident state and advances nothing
    /// shared.
    Derivative,
}

/// A registered external post-effect producer plus its lifecycle state.
struct RegisteredPostEffectProducer {
    id: crate::plugin_api::PostEffectProducerId,
    /// `init_gpu` has run against the current device. Registration has no
    /// device parameter, so GPU init is deferred to the next render.
    gpu_ready: bool,
    producer: Box<dyn crate::plugin_api::PostEffectProducer>,
}

/// A registered external post-effect stage plus its chain key and lifecycle
/// state.
struct RegisteredPostEffectStage {
    id: crate::plugin_api::PostEffectStageId,
    /// Chain position; the built-in FXAA sits at
    /// `post_effect::stage_order::ANTI_ALIASING`.
    order: i32,
    /// `init_gpu` has run against the current device (deferred, as for
    /// producers).
    gpu_ready: bool,
    stage: Box<dyn crate::plugin_api::PostEffectStage>,
}

/// Owns the GPU pipelines and per-frame state for rendering a scene. Call
/// `prepare` once per frame to upload data, then `paint_to` (or `render`) to
/// issue draw calls.
pub struct ViewportRenderer {
    resources: DeviceResources,
    /// State for the instanced (GPU-driven) mesh draw path.
    instancing: InstancingState,
    /// Registered item-type plugins keyed by
    /// [`ItemTypePlugin::type_name`](crate::plugin_api::ItemTypePlugin::type_name).
    /// `init_gpu` is invoked once on registration; per-frame `prepare` and
    /// `paint` fire when a matching collection is on `SceneFrame`.
    item_type_plugins: crate::renderer::item_plugins::registry::ItemPluginRegistry,
    /// Externally registered post-effect producers, in registration order.
    /// `init_gpu` is deferred to the first render with the device; per-frame
    /// `prepare` / `encode` run on the HDR path, per viewport.
    post_effect_producers: Vec<RegisteredPostEffectProducer>,
    /// Source for [`PostEffectProducerId`](crate::plugin_api::PostEffectProducerId)s.
    next_post_effect_producer_id: u64,
    /// Externally registered post-effect stages, in registration order; the
    /// chain sorts by each stage's order key at encode time.
    post_effect_stages: Vec<RegisteredPostEffectStage>,
    /// Source for [`PostEffectStageId`](crate::plugin_api::PostEffectStageId)s.
    next_post_effect_stage_id: u64,
    /// This viewport frame's external composite-slot contributions: the
    /// views returned by producer `encode` calls. Cleared per render call.
    frame_external_slot_views: Vec<(crate::plugin_api::PostEffectSlot, crate::gpu::TextureView)>,
    /// Slots that have logged the external-overrides-built-in conflict (one
    /// bit per `PostEffectSlot` variant), so the debug log fires once per
    /// slot per renderer.
    post_effect_slot_warned: u8,
    /// Monotonic frame counter passed to plugin contexts.
    plugin_frame_index: u64,
    /// Performance counters from the last frame.
    last_stats: crate::renderer::stats::FrameStats,
    /// Per-frame glyph GPU data, rebuilt in prepare(), consumed in paint().
    /// Per-frame tensor glyph GPU data, rebuilt in prepare(), consumed in paint().
    /// Per-frame polyline GPU data, rebuilt in prepare(), consumed in paint().
    polyline_gpu_data: Vec<crate::resources::PolylineGpuData>,
    /// Per-frame general tube GPU data, rebuilt in prepare(), consumed in paint().
    /// Per-frame Surface LIC GPU data, rebuilt in prepare(), consumed in paint().
    lic_gpu_data: Vec<crate::resources::LicSurfaceGpuData>,
    /// This frame's decal resource-cache tallies, packed `(uploads << 32) |
    /// reused`. Shared with the decal item type, which is where the cache
    /// lives; the renderer only reads it back into `FrameStats`.
    decal_cache_stats: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Opaque surfaces that opted out of decal projection, resolved at the top
    /// of prepare() and handed to item-type plugins on their frame context.
    /// `receives_decals` lives on mesh items, which no plugin can see.
    decal_excluded_surfaces: Vec<(crate::MeshId, [[f32; 4]; 4])>,
    /// Per-frame mesh-instance batches, rebuilt in prepare(), consumed in paint().
    mesh_instance_gpu_data: Vec<crate::resources::MeshInstanceGpuData>,
    external_instances_gpu_data:
        Vec<crate::resources::gpu::external_instances::ExternalInstancesGpuData>,
    /// Per-frame overlay label GPU data, rebuilt in prepare(), consumed in paint().
    label_gpu_data: Option<crate::resources::LabelGpuData>,
    /// Per-frame SDF overlay shape GPU data, rebuilt in prepare(), consumed in paint().
    overlay_shape_gpu_data: Option<crate::resources::OverlayShapeGpuData>,
    /// Per-frame ordered overlay draw list. The overlay prepare passes record one
    /// segment per contiguous draw, tagged with `z_order` and family rank; the
    /// emit path walks them in sorted order so `z_order` composes across overlay
    /// families. Rebuilt each frame, allocation reused.
    overlay_draw_segments: Vec<overlay_draw_order::OverlayDrawSegment>,
    /// Set when any overlay item this frame carries a non-zero `z_order`. When
    /// false, the segment list is left empty and the emit path uses its fixed
    /// family order, so scenes that never touch `z_order` pay nothing.
    overlay_uses_zorder: bool,
    /// Persistent grow-on-demand vertex buffers for the overlay pass. The prepare
    /// passes write their per-frame geometry into these in place instead of
    /// allocating a fresh buffer every frame; each grows only when a frame needs
    /// more room. One per fixed stream (text, solid shapes, blur) plus a pool for
    /// the variable number of per-texture shape batches.
    overlay_text_vbuf: overlay_buffers::GrowBuffer,
    overlay_shape_vbuf: overlay_buffers::GrowBuffer,
    overlay_shape_blur_vbuf: overlay_buffers::GrowBuffer,
    overlay_shape_tex_vbufs: Vec<overlay_buffers::GrowBuffer>,
    /// Per-frame viewport-size uniform (logical `[w, h, 0, 0]`) shared by the
    /// overlay pipelines. Overlay vertices are stored in local logical pixels; the
    /// overlay vertex shaders read this to map them to NDC, so overlay geometry is
    /// independent of the viewport size (a resize rewrites this uniform, not the
    /// vertices). Created once and overwritten each frame.
    overlay_viewport_buf: Option<crate::gpu::Buffer>,
    /// Per-frame retained overlay text-stream draws, one per submitted
    /// `RetainedOverlay` that resolved to a live compiled group with text-pipeline
    /// geometry. Referenced by `OverlayDrawSource::Retained { draw_index }`.
    overlay_retained_draws: Vec<overlay_buffers::RetainedDraw>,
    /// Per-frame retained overlay shape-stream draws (SDF shapes), referenced by
    /// `OverlayDrawSource::RetainedShape { draw_index }`.
    overlay_retained_shape_draws: Vec<overlay_buffers::RetainedShapeDraw>,
    /// Shared per-draw instance buffer for the overlay pass: slot 0 identity plus
    /// one per retained group. Bound by both the text pipeline (label bind group)
    /// and the shape pipeline (shadow bind group). Rebuilt each frame.
    overlay_instances_buf: Option<crate::gpu::Buffer>,
    /// Set once the label prepare has written `overlay_instances_buf` this frame,
    /// so the shape prepare reuses it instead of building an identity-only fallback.
    overlay_instances_ready: bool,
    /// Cached GPU textures for the backdrop blur effect (frosted glass).
    /// Recreated when the viewport size changes.
    backdrop_blur_state: Option<crate::resources::BackdropBlurState>,
    /// Per-viewport GPU state slots.
    ///
    /// Indexed by `FrameData::camera.viewport_index`. Each slot owns independent
    /// uniform buffers and bind groups for camera, clip planes, clip volume,
    /// shadow info, and grid. Slots are grown lazily in `prepare` via
    /// `ensure_viewport_slot`. There are at most 4 in the current UI.
    viewport_slots: Vec<ViewportSlot>,
    /// GPU compute filter results from the last `prepare()` call.
    ///
    /// Each entry contains a compacted index buffer + count for one filtered mesh.
    /// Consumed during `paint()` to override the mesh's default index buffer.
    /// Cleared and rebuilt each frame.
    compute_filter_results: Vec<crate::resources::ComputeFilterResult>,
    /// State for the non-instanced (per-object) mesh draw path.
    mesh_uniforms: PerObjectState,
    /// Cached render bundle for the opaque per-object draws, rebuilt by
    /// `prepare()` when the item set changes and replayed by the paint path.
    /// `None` when the current frame is ineligible (instanced batches active,
    /// wireframe/attribute/deform features in play, small scenes) or the
    /// churn gate has backed off to immediate draws.
    per_object_bundle: Option<per_object_state::PerObjectBundle>,
    /// Backs the bundle off to immediate draws while the item set churns.
    per_object_bundle_gate: per_object_state::BundleChurnGate,
    /// Scene surface items after the per-frame LOD resolve, in submission order
    /// and length. Filled at the end of `prepare_scene_internal` so the paint
    /// pass draws the resolved level meshes and skips culled items: without
    /// this the draw path re-read the raw `frame.scene.surfaces`, discarding
    /// the LOD level swap and cull for every non-instanced item.
    prepared_surfaces: Vec<SceneRenderItem>,
    /// Cached shadow state carried across frames.
    shadow: ShadowState,
    /// Current runtime mode controlling internal default behaviour.
    runtime_mode: crate::renderer::stats::RuntimeMode,
    /// Whether the current render presents a frame or is an auxiliary read. Set
    /// to `Derivative` for the duration of a capture / bake render and restored
    /// afterwards, so an auxiliary render advances no shared per-frame state.
    render_mode: RenderMode,
    /// Optional cap on how much main-thread time `prepare` is allowed to
    /// spend running apply closures for completed upload jobs.
    ///
    /// `None` means unbounded (apply work runs to completion in one
    /// frame). `Some(d)` spreads the cost across frames so heavy
    /// completions do not produce one fat frame; the deferred applies
    /// run on the next call to `prepare`.
    upload_budget: Option<std::time::Duration>,
    /// Active performance policy: target FPS, render scale bounds, and permitted reductions.
    performance_policy: crate::renderer::stats::PerformancePolicy,
    /// Current render scale tracked by the adaptation controller (or set manually).
    ///
    /// Clamped to `[policy.min_render_scale, policy.max_render_scale]`.
    /// Reported in `FrameStats::render_scale` each frame.
    current_render_scale: f32,
    /// Instant recorded at the start of the most recent `prepare()` call.
    /// Used to compute `total_frame_ms` on the following frame.
    last_prepare_instant: Option<web_time::Instant>,
    /// Frame counter incremented each `prepare()` call. Used for picking throttle in Playback mode.
    frame_counter: u64,
    /// Current LOD level per item, keyed by pick id, carried across frames so
    /// level switches use hysteresis. Items without a pick id are not tracked
    /// here and resolve fresh each frame. Pruned to the items seen each frame.
    lod_levels: std::collections::HashMap<u64, usize>,
    /// Current LOD level per mesh instance, keyed by `(item pick id, instance
    /// index)`. Same role as `lod_levels` but for `MeshInstanceItem`, where each
    /// instance picks its own level. Instances in items without a pick id are
    /// not tracked. Pruned to the instances seen each frame.
    mesh_instance_lod_levels: std::collections::HashMap<(u64, u32), usize>,
    /// Surface items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_scene_items: Vec<SceneRenderItem>,
    /// Broad-phase BVH over the surface `pick_scene_items`, accelerating the CPU
    /// point pick. Lazily (re)built inside `pick()` behind a `Mutex` so `pick(&self)`
    /// still works on the `Send + Sync` egui-callback path; the two revs below (set
    /// in `cache_pick_items`) tell it when to rebuild vs refit.
    pick_bvh: std::sync::Mutex<Option<crate::renderer::picking::point::PickSceneBvh>>,
    /// Rev of the pickable surface set (pick id + mesh + geometry presence). A change
    /// means items were added, removed, or toggled: rebuild the pick BVH.
    pick_bvh_identity_rev: u64,
    /// Identity rev folded with the per-item model transforms. A change while the
    /// identity is stable means objects only moved: refit the pick BVH.
    pick_bvh_transform_rev: u64,
    /// Point cloud items from the last `prepare()` call, retained for `pick()` dispatch.
    /// Opaque volume mesh items from the last `prepare()` call, retained for cell-level `pick()` dispatch.
    pick_volume_mesh_items: Vec<VolumeMeshItem>,
    /// Polyline items from the last `prepare()` call, retained for `pick()` dispatch.
    /// Glyph items from the last `prepare()` call, retained for `pick()` dispatch.
    /// Tensor glyph items from the last `prepare()` call, retained for `pick()` dispatch.
    /// Volume surface slice items from the last `prepare()` call, retained for `pick()` dispatch.
    /// Decal items from the last `prepare()` call, retained for `pick()` dispatch.
    /// When `false`, `prepare()` skips populating the CPU pick caches above, so
    /// scenes that never call `pick()`/`pick_rect()` avoid a per-frame deep copy
    /// of all inline geometry. Enable with `set_cpu_pick_cache(true)`.
    cpu_pick_cache_enabled: bool,

    /// In-flight async GPU pick, if any. `pick_object_begin` submits the id pass
    /// and parks the staging buffers here; `pick_object_poll` reads them back
    /// without blocking on the GPU queue. `None` when no async pick is pending.
    pending_pick: Option<picking::PendingPick>,

    // --- GPU timestamp queries ---
    /// Timestamp query set with `2 * GPU_TS_SLOTS` entries: a begin/end pair per
    /// measured pass (see the `GPU_TS_*` slot constants). `None` when
    /// `TIMESTAMP_QUERY` is unavailable or not yet initialised.
    ///
    /// Double-buffered with `ts_query_set_prev`: passes write this frame's
    /// timestamps here, while the set written last frame is resolved from this
    /// frame's encoder. Resolving in the same command buffer as the pass returns
    /// stale end-of-pass samples on Metal when the pass is short (its
    /// stage-boundary counters have not landed yet), which made short scenes
    /// produce zero-delta samples that were dropped, latching `gpu_frame_ms`
    /// indefinitely. Resolving one submission later reads settled counters.
    ts_query_set: Option<crate::gpu::QuerySet>,
    /// The query set written during the previous frame, resolved this frame.
    /// Swapped with `ts_query_set` at the start of each `prepare()`.
    ts_query_set_prev: Option<crate::gpu::QuerySet>,
    /// Bitmask of `GPU_TS_*` slots written into `ts_query_set_prev` during the
    /// previous frame. Zero when there is nothing to resolve (first frame, or
    /// the sample was already consumed).
    ts_prev_mask: u32,
    /// Resolve buffer: `2 * GPU_TS_SLOTS` x u64, GPU-only (`QUERY_RESOLVE | COPY_SRC`).
    ts_resolve_buf: Option<crate::gpu::Buffer>,
    /// Staging buffer: `2 * GPU_TS_SLOTS` x u64, CPU-readable (`COPY_DST | MAP_READ`).
    ts_staging_buf: Option<crate::gpu::Buffer>,
    /// Bitmask of `GPU_TS_*` slots whose timestamps were written this frame.
    /// Passes are conditional, so unwritten slots hold stale/undefined query
    /// data; only slots set here are read back. Reset at the start of each frame.
    ///
    /// Atomic so a pass method can set its bit through `&self` without colliding
    /// with the immutable viewport-slot borrows live during pass encoding (and
    /// to keep `ViewportRenderer: Sync`).
    ts_written_mask: std::sync::atomic::AtomicU32,
    /// Geometry buffer binds (`set_vertex_buffer` + `set_index_buffer`) issued
    /// by the main-pass instanced draw loops (opaque scene + OIT) this frame.
    /// Atomic so the draw loops can bump it through `&self` while the pass holds
    /// immutable borrows. Reset before paint and latched into
    /// `FrameStats::main_buffer_binds` after. The slab collapses this from ~one
    /// pair per batch to ~one pair per chunk; the counter is the direct proof.
    frame_main_buffer_binds: std::sync::atomic::AtomicU32,
    /// Draw commands the main-pass instanced loops issued this frame: each
    /// `multi_draw_indexed_indirect` counts once regardless of how many batches
    /// it collapsed, each fallback `draw_indexed_indirect` / `draw_indexed`
    /// counts once. Latched into `FrameStats::main_draw_commands`; compare
    /// against `instanced_batches` to read the collapse ratio.
    frame_main_draw_commands: std::sync::atomic::AtomicU32,
    /// One-shot latch for the paint_to foreground warning: a host-owned
    /// render pass cannot host the cleared-depth foreground pass, so
    /// submitted foreground items are reported once instead of every frame.
    foreground_paint_to_warned: std::sync::atomic::AtomicBool,
    /// One-shot latch: item-type plugins are HDR-only, so plugin items submitted
    /// while the LDR pipeline (`PipelineMode::Direct`) is active are dropped and
    /// reported once instead of silently.
    ldr_plugin_items_warned: std::sync::atomic::AtomicBool,
    /// One-shot latch: transparent volume meshes need the OIT pass, which only
    /// exists in the HDR pipeline, so on the LDR pipeline they are reported once.
    ldr_volume_transparency_warned: std::sync::atomic::AtomicBool,
    /// Snapshot of the written mask for the queries currently resolved into the
    /// staging buffer, carried alongside the delayed readback so the reader
    /// knows which slots are valid.
    ts_pending_mask: u32,
    /// Nanoseconds per GPU timestamp tick, from `queue.get_timestamp_period()`.
    ts_period: f32,
    /// True when the staging buffer holds resolved timestamps that have not yet
    /// been mapped for readback.
    ts_data_ready: bool,
    /// True when a map of the timestamp staging buffer is in flight. The render
    /// path skips the resolve/copy while this is set so the single staging
    /// buffer is not overwritten before `prepare()` has read it.
    ts_map_inflight: bool,
    /// In-flight timestamp map status, set from the map callback: 0 = pending,
    /// 1 = mapped, 2 = failed. An `Arc<AtomicU8>` rather than an mpsc channel so
    /// `ViewportRenderer` stays `Sync` (mpsc receivers are not).
    ts_map_status: std::sync::Arc<std::sync::atomic::AtomicU8>,

    /// Per-phase CPU timings accumulated during the current `prepare()` call,
    /// copied into `FrameStats::prepare_breakdown` at the end of the frame.
    prepare_breakdown: crate::renderer::stats::PrepareBreakdown,

    // --- Per-pass degradation state ---
    /// Tiered degradation ladder position (0 = none, 1 = shadows, 2 = volumes, 3 = effects).
    /// Advanced one step per over-budget frame once render scale hits minimum;
    /// reversed one step per comfortably-under-budget frame.
    degradation_tier: u8,
    /// Whether the shadow pass was skipped this frame due to budget pressure.
    /// Computed once per frame at the top of prepare() and used by both
    /// prepare_scene_internal and reported in FrameStats.
    degradation_shadows_skipped: bool,
    /// Whether volume raymarch step size was doubled this frame due to budget pressure.
    degradation_volume_quality_reduced: bool,
    /// Whether SSAO, contact shadows, and bloom were skipped this frame.
    /// Set in prepare(); read by the render path.
    degradation_effects_throttled: bool,

    /// Lights dropped by the CPU frustum cull on the most recent frame.
    /// Surfaced through the cluster debug overlay when enabled.
    pub(crate) last_frustum_culled_lights: u32,
    /// Most recent cluster build readback. Populated when a frame's
    /// `EffectsDebug::cluster_stats_request` was true.
    pub(crate) last_cluster_stats: Option<crate::resources::gpu::clustered::ClusterStats>,
}

/// Warn once when a cull submission needs more compaction scratch than the
/// device will bind as one storage buffer, so the drop to arrival-order
/// submission is visible rather than silent.
///
/// The scratch sizes to the submission, so this needs a scene far past what any
/// device can draw: it takes tens of millions of instances to reach the default
/// 128 MiB binding limit.
pub(crate) fn warn_once_cull_plan_capacity(batches: u32, instances: u32) {
    static WARNED: std::sync::Once = std::sync::Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            batches,
            instances,
            "cull submission needs more compaction scratch than max_storage_buffer_binding_size; \
             draws submit in cull-arrival order, so frames are not bit-reproducible"
        );
    });
}

impl ViewportRenderer {
    /// The optional device features the renderer can take advantage of,
    /// filtered to what `adapter` supports. Pass the result as
    /// `required_features` when requesting the device:
    ///
    /// - `INDIRECT_FIRST_INSTANCE` enables GPU-driven culling and the
    ///   indirect instanced draw path.
    /// - `MULTI_DRAW_INDIRECT_COUNT` signals that `multi_draw_indexed_indirect`
    ///   runs natively rather than emulated as a per-entry loop, so the indirect
    ///   draw path collapses a run of batches that share pipeline, bind group,
    ///   and geometry chunk into one multi-draw (present on Vulkan/DX12; absent
    ///   on Metal, which keeps the per-batch loop).
    /// - `TIMESTAMP_QUERY` enables `FrameStats::gpu_frame_ms` and the
    ///   per-pass GPU breakdown. `TIMESTAMP_QUERY_INSIDE_PASSES` additionally
    ///   splits the cull's compaction into its three dispatches
    ///   (`GpuBreakdown::cull_plan_ms` and friends); without it those read
    ///   `0.0` and the rest of the breakdown is unaffected.
    /// - `PIPELINE_CACHE` enables
    ///   [`pipeline_cache_data`](Self::pipeline_cache_data) /
    ///   [`new_with_pipeline_cache`](Self::new_with_pipeline_cache), so
    ///   pipeline compilation from a previous run can be reused instead of
    ///   redone (startup and first-use hitches).
    /// - `SHADER_PRIMITIVE_INDEX` lets the GPU pick pass read the rasterizer's
    ///   triangle index, so a GPU pick can resolve the hit face / cell / segment
    ///   (not just the object). Without it the GPU pick stays object-level for
    ///   triangle-meshed types; instance- and segment-level picks (glyphs,
    ///   sprites, polylines) do not need it.
    /// - `FLOAT32_FILTERABLE` lets direct-volume (`VolumeItem`) rendering keep the
    ///   scalar field in a full-precision `R32Float` 3D texture and still sample
    ///   it with trilinear interpolation. Without it the field falls back to an
    ///   `R16Float` texture (trilinear at reduced precision, half the bandwidth);
    ///   either way the reconstruction is smooth, never blocky nearest-neighbour.
    /// - The bindless texture-array set (texture `binding_array` + non-uniform
    ///   indexing + partially bound) lets the instanced mesh path bind material
    ///   textures once per frame and index them per material, so instances of one
    ///   mesh with different materials batch together. Present on Vulkan, DX12, and
    ///   Apple Silicon Metal (argument buffers Tier 2); without it (WebGPU, older
    ///   hardware) the path binds textures per batch (the portable default).
    ///
    /// Everything works without them; rendering falls back to direct draws
    /// (with CPU-side shadow-cascade culling), GPU timings read as `None`,
    /// `pipeline_cache_data` returns `None`, and volumes use the `R16Float` path.
    pub fn recommended_device_features(adapter: &crate::gpu::Adapter) -> crate::gpu::Features {
        let mut features = crate::gpu::Features::empty();
        for feature in [
            crate::gpu::Features::INDIRECT_FIRST_INSTANCE,
            crate::gpu::Features::MULTI_DRAW_INDIRECT_COUNT,
            crate::gpu::Features::TIMESTAMP_QUERY,
            crate::gpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
            crate::gpu::Features::PIPELINE_CACHE,
            crate::gpu::PRIMITIVE_INDEX_FEATURE,
            crate::gpu::Features::FLOAT32_FILTERABLE,
        ] {
            if adapter.features().contains(feature) {
                features |= feature;
            }
        }
        // The bindless material-texture path needs the whole texture-array set at
        // once; request it only when the adapter offers every piece, so a device
        // that supports part of it is not asked for a feature it lacks.
        if adapter
            .features()
            .contains(crate::gpu::BINDLESS_TEXTURE_FEATURES)
        {
            features |= crate::gpu::BINDLESS_TEXTURE_FEATURES;
        }
        features
    }

    /// The storage-buffer-per-stage headroom
    /// [`recommended_device_limits`](Self::recommended_device_limits) requests:
    /// the most any single pipeline binds in one stage across all optional
    /// features. The `raytrace` path tracer's compute stage is the high-water
    /// mark at ten; the base lit mesh path needs only
    /// [`MIN_STORAGE_BUFFERS_PER_STAGE`](Self::MIN_STORAGE_BUFFERS_PER_STAGE),
    /// and per-vertex deformers sit between at
    /// [`DEFORM_STORAGE_BUFFERS_PER_STAGE`](Self::DEFORM_STORAGE_BUFFERS_PER_STAGE).
    /// Requesting this much up front lets every optional feature run; it is
    /// clamped to what the adapter actually supports.
    pub const REQUIRED_STORAGE_BUFFERS_PER_STAGE: u32 = 10;

    /// The hard floor the base lit mesh pipeline needs in one shader stage. The
    /// clustered lit fragment binds this many storage buffers (clustered light
    /// data, light probes, shadow data, plus the per-object buffer), so a device
    /// below this cannot render the mesh path at all and [`new`](Self::new)
    /// rejects it. Optional features that need more are gated on the device
    /// providing the headroom (see [`DEFORM_STORAGE_BUFFERS_PER_STAGE`]) rather
    /// than asserted here, so they degrade instead of crashing.
    ///
    /// [`DEFORM_STORAGE_BUFFERS_PER_STAGE`]: Self::DEFORM_STORAGE_BUFFERS_PER_STAGE
    #[cfg(not(feature = "raytrace"))]
    pub const MIN_STORAGE_BUFFERS_PER_STAGE: u32 = 8;
    /// The `raytrace` path tracer's compute stage binds ten storage buffers, so
    /// a build with that feature requires ten up front. See the non-`raytrace`
    /// definition for the base rationale.
    #[cfg(feature = "raytrace")]
    pub const MIN_STORAGE_BUFFERS_PER_STAGE: u32 = 10;

    /// The per-stage storage-buffer count per-vertex deformers need: the base
    /// vertex stage (seven) plus the two the deform sidecar adds. A device below
    /// this still renders the base mesh path, but the deform group is left out
    /// and `register_deformer` reports that deformers are unavailable, rather
    /// than failing pipeline creation.
    pub const DEFORM_STORAGE_BUFFERS_PER_STAGE: u32 = 9;

    /// The device limits viewport-lib runs best with, above wgpu's defaults.
    ///
    /// Raises `max_storage_buffers_per_shader_stage` to
    /// [`REQUIRED_STORAGE_BUFFERS_PER_STAGE`](Self::REQUIRED_STORAGE_BUFFERS_PER_STAGE),
    /// enough headroom for every optional feature (per-vertex deformers, the
    /// `raytrace` path tracer). The base lit mesh path needs only
    /// [`MIN_STORAGE_BUFFERS_PER_STAGE`](Self::MIN_STORAGE_BUFFERS_PER_STAGE) and
    /// renders on wgpu's default limits, so passing this is not required just to
    /// draw: it is what turns the optional features on. Features whose headroom
    /// the device lacks are disabled rather than fatal.
    ///
    /// Pass the result as `required_limits` in the `DeviceDescriptor`. It starts
    /// from [`Limits::default`](crate::gpu::Limits::default) and raises the
    /// storage-buffer count viewport-lib needs (clamped to what the adapter
    /// supports), plus `max_storage_buffer_binding_size` and `max_buffer_size` to
    /// the adapter maximum. The size raises matter for per-vertex deformers: a
    /// large mesh with many morph targets produces a deform-slot storage buffer
    /// past the default 128 MiB binding cap, which the default limits cannot bind.
    /// Consumers that already request `adapter.limits()` (the device's full
    /// capabilities) do not need this. On a device whose maximum is genuinely the
    /// default (some mobile tiers, the web), the size raises are a no-op and an
    /// oversized buffer still cannot be bound.
    pub fn recommended_device_limits(adapter: &crate::gpu::Adapter) -> crate::gpu::Limits {
        let adapter_limits = adapter.limits();
        let mut limits = crate::gpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = limits
            .max_storage_buffers_per_shader_stage
            .max(Self::REQUIRED_STORAGE_BUFFERS_PER_STAGE)
            .min(adapter_limits.max_storage_buffers_per_shader_stage);
        // Assign the adapter's own maximum (never more than it reports, so the
        // request always succeeds). A deform-slot buffer for a big morph mesh can
        // exceed the 128 MiB binding / 256 MiB buffer defaults, which would make
        // the deform bind group invalid; the base draw path stays under both.
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        // The bindless material path binds one texture array; its element count
        // (a binding-array limit that defaults to 0) must be requested alongside
        // the feature or the layout is invalid. Only ask for it when the adapter
        // offers the bindless features, clamped to what it reports.
        if adapter
            .features()
            .contains(crate::gpu::BINDLESS_TEXTURE_FEATURES)
        {
            limits.max_binding_array_elements_per_shader_stage =
                crate::resources::mesh::instanced_bindless::BINDLESS_TEXTURE_CAPACITY
                    .min(adapter_limits.max_binding_array_elements_per_shader_stage);
        }
        limits
    }

    /// Create a new renderer with default settings (no MSAA).
    /// Call once at application startup.
    pub fn new(device: &crate::gpu::Device, target_format: crate::gpu::TextureFormat) -> Self {
        Self::with_sample_count(device, target_format, 1)
    }

    /// Create a new renderer with the specified MSAA sample count (1, 2, or 4).
    ///
    /// When using MSAA (sample_count > 1), the caller must create multisampled
    /// colour and depth textures and use them as render pass attachments with the
    /// final surface texture as the resolve target.
    pub fn with_sample_count(
        device: &crate::gpu::Device,
        target_format: crate::gpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        Self::with_sample_count_and_cache(device, target_format, sample_count, None)
    }

    /// Create a renderer, seeding the GPU pipeline cache from previously saved
    /// data so shader compilation can be skipped on later launches.
    ///
    /// Pass the bytes returned by an earlier [`pipeline_cache_data`](Self::pipeline_cache_data)
    /// call, or `None` on first run. The cache only takes effect when the device
    /// was created with `Features::PIPELINE_CACHE`; otherwise the data is ignored
    /// and this matches [`new`](Self::new).
    pub fn new_with_pipeline_cache(
        device: &crate::gpu::Device,
        target_format: crate::gpu::TextureFormat,
        pipeline_cache_data: Option<&[u8]>,
    ) -> Self {
        Self::with_sample_count_and_cache(device, target_format, 1, pipeline_cache_data)
    }

    /// Returns the current contents of the GPU pipeline cache, suitable for
    /// persisting and feeding back into [`new_with_pipeline_cache`](Self::new_with_pipeline_cache)
    /// on the next launch. `None` when the device lacks `Features::PIPELINE_CACHE`.
    pub fn pipeline_cache_data(&self) -> Option<Vec<u8>> {
        self.resources.pipeline_cache.as_ref()?.get_data()
    }

    /// Like [`with_sample_count`](Self::with_sample_count) with an MSAA count and
    /// an optional saved pipeline cache.
    pub fn with_sample_count_and_cache(
        device: &crate::gpu::Device,
        target_format: crate::gpu::TextureFormat,
        sample_count: u32,
        pipeline_cache_data: Option<&[u8]>,
    ) -> Self {
        // Fail early with an actionable message rather than a cryptic wgpu
        // validation panic deep in mesh-pipeline-layout creation. This is the
        // base lit mesh path's floor; optional features that need more storage
        // buffers (per-vertex deformers, the raytrace path tracer) gate on the
        // device providing the headroom rather than asserting it here, so they
        // degrade instead of crashing.
        let available = device.limits().max_storage_buffers_per_shader_stage;
        assert!(
            available >= Self::MIN_STORAGE_BUFFERS_PER_STAGE,
            "viewport-lib needs max_storage_buffers_per_shader_stage >= {}, but the device was \
             created with {}. Pass ViewportRenderer::recommended_device_limits(&adapter) as \
             required_limits in the DeviceDescriptor (or request a higher limit).",
            Self::MIN_STORAGE_BUFFERS_PER_STAGE,
            available,
        );
        let gpu_culling_supported = device
            .features()
            .contains(crate::gpu::Features::INDIRECT_FIRST_INSTANCE);
        let multi_draw_supported = device
            .features()
            .contains(crate::gpu::Features::MULTI_DRAW_INDIRECT_COUNT);
        // Bindless material textures activate only when the device enabled the
        // whole texture-array feature set AND granted enough binding-array
        // elements for the texture array (both are needed to build the layout).
        // Modern Metal (Apple Silicon, argument buffers Tier 2), Vulkan, and DX12
        // qualify; a device that enabled the feature but not the element limit,
        // or WebGPU, stays on the per-batch binding rather than crashing.
        use crate::resources::mesh::instanced_bindless::{
            BINDLESS_TEXTURE_CAPACITY, MaterialTextureBinding,
        };
        let material_texture_binding = if device
            .features()
            .contains(crate::gpu::BINDLESS_TEXTURE_FEATURES)
            && device.limits().max_binding_array_elements_per_shader_stage
                >= BINDLESS_TEXTURE_CAPACITY
        {
            MaterialTextureBinding::Bindless
        } else {
            MaterialTextureBinding::PerBatch
        };
        let mut resources = DeviceResources::new_with_cache(
            device,
            target_format,
            sample_count,
            pipeline_cache_data,
        );
        resources.instancing.material_texture_binding = material_texture_binding;
        resources
            .material_gpu_builder
            .set_bindless(material_texture_binding == MaterialTextureBinding::Bindless);
        let mut renderer = Self {
            resources,
            instancing: InstancingState::new(gpu_culling_supported, multi_draw_supported),
            item_type_plugins: crate::renderer::item_plugins::registry::ItemPluginRegistry::new(),
            post_effect_producers: Vec::new(),
            next_post_effect_producer_id: 0,
            post_effect_stages: Vec::new(),
            next_post_effect_stage_id: 0,
            frame_external_slot_views: Vec::new(),
            post_effect_slot_warned: 0,
            plugin_frame_index: 0,
            last_stats: crate::renderer::stats::FrameStats::default(),
            prepare_breakdown: crate::renderer::stats::PrepareBreakdown::default(),
            polyline_gpu_data: Vec::new(),
            mesh_instance_gpu_data: Vec::new(),
            external_instances_gpu_data: Vec::new(),
            lic_gpu_data: Vec::new(),
            decal_cache_stats: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            decal_excluded_surfaces: Vec::new(),
            label_gpu_data: None,
            overlay_shape_gpu_data: None,
            overlay_text_vbuf: overlay_buffers::GrowBuffer::vertex("overlay_label_vbuf"),
            overlay_shape_vbuf: overlay_buffers::GrowBuffer::vertex("overlay_shape_vbuf"),
            overlay_shape_blur_vbuf: overlay_buffers::GrowBuffer::vertex("overlay_shape_blur_vbuf"),
            overlay_shape_tex_vbufs: Vec::new(),
            overlay_viewport_buf: None,
            overlay_retained_draws: Vec::new(),
            overlay_retained_shape_draws: Vec::new(),
            overlay_instances_buf: None,
            overlay_instances_ready: false,
            overlay_draw_segments: Vec::new(),
            overlay_uses_zorder: false,
            backdrop_blur_state: None,
            viewport_slots: Vec::new(),
            compute_filter_results: Vec::new(),
            mesh_uniforms: PerObjectState::new(),
            per_object_bundle: None,
            per_object_bundle_gate: Default::default(),
            prepared_surfaces: Vec::new(),
            shadow: ShadowState::new(),
            runtime_mode: crate::renderer::stats::RuntimeMode::Interactive,
            render_mode: RenderMode::Presented,
            performance_policy: crate::renderer::stats::PerformancePolicy::default(),
            upload_budget: None,
            current_render_scale: 1.0,
            last_prepare_instant: None,
            frame_counter: 0,
            lod_levels: std::collections::HashMap::new(),
            mesh_instance_lod_levels: std::collections::HashMap::new(),
            pick_scene_items: Vec::new(),
            pick_bvh: std::sync::Mutex::new(None),
            pick_bvh_identity_rev: 0,
            pick_bvh_transform_rev: 0,
            pick_volume_mesh_items: Vec::new(),
            cpu_pick_cache_enabled: false,
            pending_pick: None,
            ts_query_set: None,
            ts_query_set_prev: None,
            ts_prev_mask: 0,
            ts_resolve_buf: None,
            ts_staging_buf: None,
            ts_period: 1.0,
            ts_data_ready: false,
            ts_map_inflight: false,
            ts_map_status: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
            ts_written_mask: std::sync::atomic::AtomicU32::new(0),
            frame_main_buffer_binds: std::sync::atomic::AtomicU32::new(0),
            frame_main_draw_commands: std::sync::atomic::AtomicU32::new(0),
            foreground_paint_to_warned: std::sync::atomic::AtomicBool::new(false),
            ldr_plugin_items_warned: std::sync::atomic::AtomicBool::new(false),
            ldr_volume_transparency_warned: std::sync::atomic::AtomicBool::new(false),
            ts_pending_mask: 0,
            degradation_tier: 0,
            degradation_shadows_skipped: false,
            degradation_volume_quality_reduced: false,
            degradation_effects_throttled: false,
            last_frustum_culled_lights: 0,
            last_cluster_stats: None,
        };
        renderer.register_internal_item_plugins(device);
        renderer
    }

    /// Access the underlying GPU resources (e.g. for mesh uploads).
    pub fn resources(&self) -> &DeviceResources {
        &self.resources
    }

    /// Resident GPU bytes for the user-uploaded working set, including whatever
    /// registered item-type plugins report holding in stores of their own.
    ///
    /// The same figure as
    /// [`DeviceResources::resident_bytes`](crate::resources::DeviceResources::resident_bytes)
    /// with [`ResidentBytes::plugin_bytes`](crate::resources::ResidentBytes::plugin_bytes)
    /// filled in. Prefer this one: the plugins are registered with the
    /// renderer, so the resources-level call cannot see them and reports
    /// `plugin_bytes` as zero.
    pub fn resident_bytes(&self) -> crate::resources::ResidentBytes {
        let mut bytes = self.resources.resident_bytes();
        bytes.plugin_bytes = self
            .item_type_plugins
            .values()
            .map(|p| p.resident_bytes())
            .sum();
        bytes
    }

    /// Resident GPU bytes per registered item type, in registration order.
    ///
    /// [`ResidentBytes::plugin_bytes`](crate::resources::ResidentBytes::plugin_bytes)
    /// is the sum of these, which is the right figure to budget against but
    /// the wrong one to act on: an eviction policy that is over its ceiling
    /// needs to know which type to free content from. This is the breakdown.
    ///
    /// Every registered type appears, including the ones reporting zero (a
    /// type whose items carry their own geometry holds nothing between
    /// frames, and a type that has not implemented
    /// [`resident_bytes`](crate::plugin_api::ItemTypePlugin::resident_bytes)
    /// reports zero whatever it holds).
    pub fn plugin_resident_bytes(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        self.item_type_plugins
            .iter()
            .map(|(name, plugin)| (name, plugin.resident_bytes()))
    }

    /// Performance counters from the last completed frame.
    pub fn last_frame_stats(&self) -> crate::renderer::stats::FrameStats {
        self.last_stats
    }

    /// The LOD level drawn for the item with this pick id on the last prepared
    /// frame, or `None` when the pick id names no tracked LOD item this frame
    /// (it is not an LOD item, was culled below its threshold, or was not
    /// submitted). Level 0 is the highest-detail mesh; larger indices are
    /// coarser. Only items carrying a non-zero pick id are tracked, since the
    /// level is keyed by it and carried across frames for switch hysteresis.
    pub fn lod_level(&self, pick_id: u64) -> Option<usize> {
        self.lod_levels.get(&pick_id).copied()
    }

    /// Diagnostics from the cluster build pass on the most recent frame that
    /// requested them (`EffectsDebug::cluster_stats_request`). Returns
    /// `None` until a request has been served.
    pub fn cluster_stats(&self) -> Option<crate::resources::gpu::clustered::ClusterStats> {
        self.last_cluster_stats
    }

    /// Read back the current exposure state for a viewport (a blocking GPU
    /// map). Under [`ExposureMode::Automatic`] this exposes the metered target
    /// EV, the adapted EV, and whether adaptation is still settling, for a UI
    /// readout or to decide whether to request another redraw while the
    /// "eye" is still adjusting (`dt > 0`).
    ///
    /// Opt-in diagnostic: it copies a tiny buffer and blocks on a device poll,
    /// so call it for UI, not in the hot path. Returns `None` for an unknown
    /// viewport or before its HDR pipeline has produced a frame.
    pub fn exposure_state(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: ViewportId,
    ) -> Option<ExposureReadback> {
        let slot = self.viewport_slots.get(id.0)?;
        let hdr = slot.hdr.as_ref()?;
        let size = std::mem::size_of::<crate::resources::gpu::exposure::ExposureState>() as u64;
        let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("exposure_readback_staging"),
            size,
            usage: crate::gpu::BufferUsages::COPY_DST | crate::gpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("exposure_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&hdr.exposure_state_buf, 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(crate::gpu::MapMode::Read, |_| {});
        let _ = device.poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        let out = {
            let data = crate::gpu::mapped_range(slice);
            let st: &crate::resources::gpu::exposure::ExposureState =
                &bytemuck::cast_slice(&data)[0];
            ExposureReadback {
                exposure: st.exposure,
                current_ev: st.current_ev,
                target_ev: st.target_ev,
                adapting: st.adapting != 0.0,
            }
        };
        staging.unmap();
        Some(out)
    }

    /// Disable GPU-driven culling, reverting to the direct draw path.
    ///
    /// Has no effect when the device does not support `INDIRECT_FIRST_INSTANCE`
    /// (culling is already disabled on those devices).
    pub fn disable_gpu_driven_culling(&mut self) {
        self.instancing.gpu_culling_enabled = false;
    }

    /// Force a full instance buffer upload on the next frame.
    ///
    /// Normally the renderer skips GPU writes for instanced batches whose data
    /// has not changed since the last upload. Call this when you have mutated
    /// batch-relevant state through a path the renderer cannot observe (for
    /// example, directly modifying GPU buffer contents or scene items after
    /// `collect_render_items` runs). The flag is consumed once and resets
    /// automatically after the next `prepare` call.
    pub fn force_dirty(&mut self) {
        self.instancing.force_full_upload = true;
        // Also invalidate the generation cache so the next prepare is guaranteed
        // to enter the rebuild path even if the scene generation is unchanged.
        self.instancing.last_scene_generation = u64::MAX;
        self.shadow.invalidate_point_shadow_cache();
    }

    /// Re-enable GPU-driven culling after a call to `disable_gpu_driven_culling`.
    ///
    /// Has no effect when the device does not support `INDIRECT_FIRST_INSTANCE`.
    pub fn enable_gpu_driven_culling(&mut self) {
        if self.instancing.gpu_culling_supported {
            self.instancing.gpu_culling_enabled = true;
        }
    }

    /// Enable or disable HiZ occlusion culling on the main-camera cull.
    ///
    /// When on, the GPU cull builds a hierarchical-Z depth pyramid from the
    /// previous frame's scene depth and drops instances whose screen-space box
    /// is entirely behind nearer geometry, on top of the frustum test. Off by
    /// default. Has no effect unless GPU-driven culling is also active.
    ///
    /// The depth source is the previous frame's scene depth reprojected into the
    /// current camera, so the test is one frame stale and assumes a mostly
    /// static world. For a static scene (or static occluders) nothing visible is
    /// culled. With moving or animated occluders, reprojection places last
    /// frame's occluder depth at its old position and can briefly cull an
    /// instance that is actually visible this frame; it self-corrects the next
    /// frame. Treat the "never cull a visible instance" guarantee as holding for
    /// static occluders only, and leave this off for highly dynamic scenes where
    /// that pop is unacceptable.
    ///
    /// Runs on the HDR path and the owned LDR render path (`render` /
    /// `render_viewport`), both of which capture scene depth for the
    /// reprojection. The immediate-mode `paint_to` / `paint_viewport` path does
    /// not capture depth, so occlusion is a no-op there. Single-viewport only:
    /// the cull result and HiZ state are shared, not per-view, so with multiple
    /// viewports on different cameras occlusion can drop geometry that is visible
    /// in another viewport.
    ///
    /// The breakdown is reported in [`FrameStats`]: `gpu_culled_total`,
    /// `gpu_frustum_visible`, and `gpu_visible_instances` give the per-stage
    /// survivor counts.
    pub fn set_occlusion_culling(&mut self, enabled: bool) {
        self.resources.set_occlusion_culling(enabled);
    }

    /// Whether HiZ occlusion culling is currently enabled.
    pub fn occlusion_culling_enabled(&self) -> bool {
        self.resources.occlusion_culling_enabled()
    }

    /// Force the per-object opaque scene-pass draw to keep its discarding
    /// pipeline instead of the discard-free early-Z twin.
    ///
    /// Off by default. A fragment shader that contains `discard` disables
    /// hardware early depth rejection, so an eligible plain-opaque item is
    /// normally drawn with a discard-free pipeline twin and its hidden fragments
    /// are depth-rejected before shading. This forces the discarding pipeline
    /// back on for that path so a benchmark can measure the early-Z difference
    /// on a fill-bound scene in a single process. It does not change rendered
    /// output.
    pub fn set_force_po_discard(&mut self, force: bool) {
        self.resources.set_force_po_discard(force);
    }

    /// Keep the debug-visualisation block compiled into the lit pipelines even
    /// while [`DebugVis`] is off.
    ///
    /// The block sits under a uniform branch that only `DebugVis` takes, so
    /// pixels are identical either way. What changes is what an ordinary draw
    /// pays to carry it: the block declares a 24-element array, and a lit shader
    /// holding that allocation spends registers on it on every draw. The lit
    /// pipelines therefore compile without it by default.
    ///
    /// This exists so a benchmark can measure that cost by rendering one scene
    /// both ways in a single process, which is the only way to compare them
    /// without run-to-run variance swamping the difference. Off by default; not
    /// a rendering mode.
    pub fn set_force_debug_vis_shaders(&mut self, force: bool) {
        self.resources.force_debug_vis_shaders = force;
    }

    /// Force the indirect draw paths to collapse batch runs into
    /// `multi_draw_indexed_indirect` even where the backend emulates it as a
    /// per-entry loop (Metal). The emulated result is identical, so this exists
    /// to exercise and pixel-compare the collapse path on a backend without
    /// native multi-draw. Off by default; on backends that support native
    /// multi-draw the collapse is already active and this is a no-op.
    pub fn set_force_multi_draw(&mut self, force: bool) {
        self.instancing.multi_draw_forced = force;
    }

    /// Cap the per-frame cost of upload-job work on the render thread.
    ///
    /// `None` is the default and matches the historical behaviour:
    /// `prepare` drains every completed upload's apply step and every
    /// queued GPU-job stage in one shot. `Some(d)` switches `prepare`
    /// over to `process_uploads_with_budget` so work that overflows the
    /// budget spills to the next frame. The budget covers both the
    /// apply drain and the deferred GPU stages of texture uploads (the
    /// texture creation and pixel copies), checked between items; a
    /// single large item still runs to completion once started, so the
    /// cap is soft.
    pub fn set_upload_budget(&mut self, budget: Option<std::time::Duration>) {
        self.upload_budget = budget;
    }

    /// Currently configured upload budget. See `set_upload_budget`.
    pub fn upload_budget(&self) -> Option<std::time::Duration> {
        self.upload_budget
    }

    /// Set the runtime mode controlling internal default behaviour.
    ///
    /// - [`RuntimeMode::Interactive`]: full picking rate, full quality (default).
    /// - [`RuntimeMode::Playback`]: picking throttled to reduce CPU overhead during animation.
    /// - [`RuntimeMode::Paused`]: full picking rate, full quality.
    /// - [`RuntimeMode::Capture`]: full quality, intended for screenshot/export workflows.
    pub fn set_runtime_mode(&mut self, mode: crate::renderer::stats::RuntimeMode) {
        self.runtime_mode = mode;
    }

    /// Return the current runtime mode.
    pub fn runtime_mode(&self) -> crate::renderer::stats::RuntimeMode {
        self.runtime_mode
    }

    /// True when the current render presents a frame the user sees, and so
    /// should advance per-frame state (upload pipeline, frame counter, HiZ
    /// prev-depth, stats, plugin `prepare` / `cull`). False for a `Derivative`
    /// capture / bake render, which reads resident state and advances nothing.
    pub(crate) fn render_advances_state(&self) -> bool {
        matches!(self.render_mode, RenderMode::Presented)
    }

    /// Enable or disable the CPU pick cache.
    ///
    /// When enabled, `prepare()` retains a copy of the frame's pickable items so
    /// `pick()` and `pick_rect()` can run later (e.g. on a mouse click) without the
    /// scene data. This copies all inline point/glyph/curve geometry each frame, so it
    /// is disabled by default: turn it on only when using the CPU `pick()`/`pick_rect()`
    /// path. The GPU path (`pick_scene_gpu`) and the renderer-free
    /// `interaction::picking` functions do not need it.
    pub fn set_cpu_pick_cache(&mut self, enabled: bool) {
        if !enabled && self.cpu_pick_cache_enabled {
            self.clear_pick_cache();
        }
        self.cpu_pick_cache_enabled = enabled;
    }

    /// Whether the CPU pick cache is enabled. See `set_cpu_pick_cache`.
    pub fn cpu_pick_cache(&self) -> bool {
        self.cpu_pick_cache_enabled
    }

    /// Set the performance policy controlling target FPS, render scale bounds,
    /// and permitted quality reductions.
    ///
    /// The internal adaptation controller activates when
    /// `policy.allow_dynamic_resolution` is `true` and `policy.target_fps` is
    /// `Some`. It adjusts `render_scale` within `[min_render_scale,
    /// max_render_scale]` each frame based on `total_frame_ms`.
    pub fn set_performance_policy(&mut self, policy: crate::renderer::stats::PerformancePolicy) {
        self.performance_policy = policy;
        // Clamp current scale into the new bounds immediately.
        self.current_render_scale = self
            .current_render_scale
            .clamp(policy.min_render_scale, policy.max_render_scale);
    }

    /// Return the active performance policy.
    pub fn performance_policy(&self) -> crate::renderer::stats::PerformancePolicy {
        self.performance_policy
    }

    /// Apply a full [`RenderTuning`] set in one call: the persistent
    /// performance/behaviour knobs (culling, occlusion, adaptive quality, render
    /// scale, runtime mode, upload budget, CPU pick cache, and the diagnostic
    /// overrides). Equivalent to calling the individual setters, and applies the
    /// same gating: GPU-driven culling only activates on devices that support
    /// it, and the manual render scale is ignored while adaptive resolution is
    /// on. Construction-time choices (MSAA, pipeline cache) and the LOD-group
    /// registry are not part of this and are left unchanged.
    pub fn apply_tuning(&mut self, tuning: &crate::renderer::tuning::RenderTuning) {
        if tuning.gpu_driven_culling {
            self.enable_gpu_driven_culling();
        } else {
            self.disable_gpu_driven_culling();
        }
        self.set_occlusion_culling(tuning.occlusion_culling);
        self.set_performance_policy(tuning.performance);
        self.set_render_scale(tuning.render_scale);
        self.set_runtime_mode(tuning.runtime_mode);
        self.set_upload_budget(tuning.upload_budget);
        self.set_cpu_pick_cache(tuning.cpu_pick_cache);
        self.set_force_multi_draw(tuning.diagnostics.force_multi_draw);
        self.set_force_po_discard(tuning.diagnostics.force_po_discard);
    }

    /// Snapshot the current persistent tuning, so a consumer wanting to tune for
    /// performance can see every live lever in one place and diff it against
    /// [`RenderTuning::default`]. `gpu_driven_culling` reflects the requested
    /// state, which is only actually active when
    /// [`is_gpu_culling_supported`](Self::is_gpu_culling_supported) is true.
    pub fn tuning(&self) -> crate::renderer::tuning::RenderTuning {
        crate::renderer::tuning::RenderTuning {
            gpu_driven_culling: self.instancing.gpu_culling_enabled,
            occlusion_culling: self.occlusion_culling_enabled(),
            performance: self.performance_policy(),
            render_scale: self.current_render_scale,
            runtime_mode: self.runtime_mode(),
            upload_budget: self.upload_budget,
            cpu_pick_cache: self.cpu_pick_cache(),
            diagnostics: crate::renderer::tuning::RenderDiagnostics {
                force_multi_draw: self.instancing.multi_draw_forced,
                force_po_discard: self.resources.force_po_discard,
            },
        }
    }

    /// Manually set the render scale.
    ///
    /// Effective when `performance_policy.allow_dynamic_resolution` is `false`.
    /// When dynamic resolution is enabled the adaptation controller overrides
    /// this value each frame.
    ///
    /// The value is clamped to `[policy.min_render_scale, policy.max_render_scale]`.
    ///
    /// Works on both the LDR and HDR render paths. On the HDR path, the scene,
    /// bloom, SSAO, tone-map, and FXAA all run at the scaled resolution; the
    /// result is upscale-blitted to native resolution before overlays and grid.
    pub fn set_render_scale(&mut self, scale: f32) {
        self.current_render_scale = scale.clamp(
            self.performance_policy.min_render_scale,
            self.performance_policy.max_render_scale,
        );
    }

    /// Set the target frame rate used to compute [`FrameStats::missed_budget`].
    ///
    /// Convenience wrapper that updates `performance_policy.target_fps`.
    pub fn set_target_fps(&mut self, fps: Option<f32>) {
        self.performance_policy.target_fps = fps;
    }

    /// Mutable access to the underlying GPU resources (e.g. for mesh uploads).
    pub fn resources_mut(&mut self) -> &mut DeviceResources {
        &mut self.resources
    }

    /// Returns true when the current frame is rendered via the instanced draw path.
    ///
    /// When true, edits to mesh.wgsl shadow sampling code have no effect - the active
    /// shader is mesh_instanced.wgsl. Check this before testing shader changes.
    pub fn is_using_instanced_path(&self) -> bool {
        self.instancing.use_instancing
    }

    /// Which material-texture path this renderer took: `"bindless"` or
    /// `"per-batch"`.
    ///
    /// Chosen once at construction from the device's enabled features, and it is
    /// not inferable from the backend: an Apple silicon device reports the
    /// bindless feature set through Metal argument buffers and takes that path.
    /// Requesting [`recommended_device_features`](Self::recommended_device_features)
    /// is what enables it when the adapter has it.
    ///
    /// The two paths bind textures differently, so they do not share a bug
    /// surface. Worth logging, and worth putting in a bug report.
    pub fn material_texture_binding(&self) -> &'static str {
        use crate::resources::mesh::instanced_bindless::MaterialTextureBinding;
        match self.resources.instancing.material_texture_binding {
            MaterialTextureBinding::Bindless => "bindless",
            MaterialTextureBinding::PerBatch => "per-batch",
        }
    }

    /// Take the per-batch material-texture binding, whatever the device supports.
    ///
    /// The write side of [`material_texture_binding`](Self::material_texture_binding).
    /// The renderer picks bindless whenever the device offers it, and the two
    /// paths bind textures differently, so when output differs between machines
    /// this is how to hold one of them still. Turning bindless off without it
    /// means building the device with fewer features than the renderer asks for,
    /// which changes more than this one choice.
    ///
    /// Call once, immediately after construction and before the first
    /// [`prepare`](Self::prepare). There is no way back to bindless on the same
    /// renderer: build another one.
    ///
    /// Expect fewer, larger instanced batches on the bindless path and more,
    /// smaller ones here, since a per-batch bind group cannot span materials that
    /// use different textures.
    pub fn use_per_batch_material_textures(&mut self) {
        use crate::resources::mesh::instanced_bindless::MaterialTextureBinding;
        self.resources.instancing.material_texture_binding = MaterialTextureBinding::PerBatch;
        // The interner keys blocks on their bytes, and the texture indices are
        // part of those bytes only on the bindless path, so it has to be told.
        self.resources.material_gpu_builder.set_bindless(false);
    }

    /// Returns the number of instanced batches prepared for the current frame.
    ///
    /// Zero when using the non-instanced path. Each batch corresponds to a distinct
    /// (MeshId, material) combination in the scene.
    pub fn instanced_batch_count(&self) -> usize {
        self.instancing.batches.len()
    }

    /// Run the GPU-driven cull compute against a plugin's
    /// [`CullSubmission`](crate::plugin_api::CullSubmission).
    ///
    /// Encodes two compute passes into `encoder`:
    /// 1. one thread per instance, tests AABB against `frustum`, claims a
    ///    visibility slot via atomic add;
    /// 2. one thread per batch, writes a `DrawIndexedIndirect` entry into
    ///    `sub.indirect_out` with the final visible count and zeroes the
    ///    counter for the next call.
    ///
    /// After the encoder runs, draw each batch with
    /// `pass.draw_indexed_indirect(sub.indirect_out, batch_idx * 20)` using
    /// `sub.visible_out` as the per-instance lookup buffer.
    ///
    /// The cull pipeline is created lazily on the first call. Returns
    /// without dispatching if the device does not support
    /// `INDIRECT_FIRST_INSTANCE` (call
    /// [`is_gpu_culling_supported`](Self::is_gpu_culling_supported) first).
    pub fn submit_cull(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        encoder: &mut crate::gpu::CommandEncoder,
        frustum: &crate::camera::frustum::Frustum,
        sub: &crate::plugin_api::CullSubmission<'_>,
    ) {
        if !self.instancing.gpu_culling_supported {
            return;
        }
        if self.instancing.cull_resources.is_none() {
            self.instancing.cull_resources =
                Some(crate::renderer::indirect::CullResources::new(device));
        }
        let cull = self.instancing.cull_resources.as_ref().unwrap();
        cull.dispatch(encoder, device, queue, frustum, None, sub, None, None);
    }

    /// Same as [`submit_cull`](Self::submit_cull) for one shadow cascade.
    ///
    /// Uploads the frustum to the cascade slot (so a single frame can submit
    /// the main pass plus every cascade without overwriting an in-flight
    /// upload) and forces the cull shader's shadow flag so
    /// `InstanceAabb::cast_shadows = 0` entries are skipped.
    ///
    /// `cascade_idx` must be in `0..4`; values outside that range panic in
    /// debug builds and clamp to 3 in release.
    pub fn submit_cull_shadow(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        encoder: &mut crate::gpu::CommandEncoder,
        cascade_idx: usize,
        cascade_frustum: &crate::camera::frustum::Frustum,
        sub: &crate::plugin_api::CullSubmission<'_>,
    ) {
        if !self.instancing.gpu_culling_supported {
            return;
        }
        debug_assert!(cascade_idx < 4, "cascade_idx must be in 0..4");
        let cascade_idx = cascade_idx.min(3);
        if self.instancing.cull_resources.is_none() {
            self.instancing.cull_resources =
                Some(crate::renderer::indirect::CullResources::new(device));
        }
        let cull = self.instancing.cull_resources.as_ref().unwrap();
        cull.dispatch(
            encoder,
            device,
            queue,
            cascade_frustum,
            Some(cascade_idx),
            sub,
            None,
            None,
        );
    }

    /// Convenience wrapper around [`submit_cull`](Self::submit_cull) for the
    /// common case of one mesh with N instances.
    ///
    /// The renderer fills its scratch [`BatchMeta`] slot from `draw`, zeroes
    /// its scratch counter, seeds the indirect entry, and runs a one-batch
    /// cull. Plugins that only have a single mesh per submission don't have
    /// to allocate either buffer themselves.
    ///
    /// `indirect_out` must hold one `DrawIndexedIndirect` entry (20 bytes).
    pub fn submit_cull_single_mesh(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        encoder: &mut crate::gpu::CommandEncoder,
        frustum: &crate::camera::frustum::Frustum,
        instance_aabbs: &crate::gpu::Buffer,
        instance_count: u32,
        visible_out: &crate::gpu::Buffer,
        indirect_out: &crate::gpu::Buffer,
        draw: crate::plugin_api::SingleMeshDraw,
        shadow_pass: bool,
    ) {
        self.dispatch_cull_single_mesh(
            device,
            queue,
            encoder,
            frustum,
            None,
            instance_aabbs,
            instance_count,
            visible_out,
            indirect_out,
            draw,
            shadow_pass,
        );
    }

    /// Single-mesh shadow variant of
    /// [`submit_cull_single_mesh`](Self::submit_cull_single_mesh).
    pub fn submit_cull_shadow_single_mesh(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        encoder: &mut crate::gpu::CommandEncoder,
        cascade_idx: usize,
        cascade_frustum: &crate::camera::frustum::Frustum,
        instance_aabbs: &crate::gpu::Buffer,
        instance_count: u32,
        visible_out: &crate::gpu::Buffer,
        indirect_out: &crate::gpu::Buffer,
        draw: crate::plugin_api::SingleMeshDraw,
    ) {
        debug_assert!(cascade_idx < 4, "cascade_idx must be in 0..4");
        let cascade_idx = cascade_idx.min(3);
        self.dispatch_cull_single_mesh(
            device,
            queue,
            encoder,
            cascade_frustum,
            Some(cascade_idx),
            instance_aabbs,
            instance_count,
            visible_out,
            indirect_out,
            draw,
            true,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_cull_single_mesh(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        encoder: &mut crate::gpu::CommandEncoder,
        frustum: &crate::camera::frustum::Frustum,
        cascade: Option<usize>,
        instance_aabbs: &crate::gpu::Buffer,
        instance_count: u32,
        visible_out: &crate::gpu::Buffer,
        indirect_out: &crate::gpu::Buffer,
        draw: crate::plugin_api::SingleMeshDraw,
        shadow_pass: bool,
    ) {
        if !self.instancing.gpu_culling_supported {
            return;
        }
        if self.instancing.cull_resources.is_none() {
            self.instancing.cull_resources =
                Some(crate::renderer::indirect::CullResources::new(device));
        }
        let cull = self.instancing.cull_resources.as_ref().unwrap();
        let (meta_buf, counter_buf) = cull.scratch_single_mesh_buffers();
        let meta = crate::plugin_api::BatchMeta {
            index_count: draw.index_count,
            first_index: draw.first_index,
            instance_offset: 0,
            instance_count,
            vis_offset: 0,
            is_transparent: 0,
            base_vertex: draw.base_vertex,
            _reserved_flags: 0,
        };
        queue.write_buffer(meta_buf, 0, bytemuck::bytes_of(&meta));
        queue.write_buffer(counter_buf, 0, &[0u8; 4]);
        // Seed the static fields of the indirect entry; the compute pass
        // overwrites `instance_count` with the final visible count.
        let seed: [u32; 5] = [
            draw.index_count,
            0,
            draw.first_index,
            draw.base_vertex as u32,
            draw.first_instance,
        ];
        queue.write_buffer(indirect_out, 0, bytemuck::cast_slice(&seed));

        let sub = crate::plugin_api::CullSubmission {
            instance_aabbs,
            instance_count,
            batch_meta: meta_buf,
            batch_count: 1,
            counter: counter_buf,
            visible_out,
            indirect_out,
            shadow_pass,
        };
        cull.dispatch(encoder, device, queue, frustum, cascade, &sub, None, None);
    }

    /// Register an [`ItemTypePlugin`](crate::plugin_api::ItemTypePlugin).
    ///
    /// Invokes the plugin's `init_gpu` against the current device and
    /// shared bind layout, then stores it keyed by `type_name()` for the
    /// remainder of the renderer's lifetime. Registering a second plugin
    /// with the same `type_name` replaces the first.
    ///
    /// The renderer will dispatch `prepare` and `paint` to the plugin on
    /// every frame where
    /// [`SceneFrame::submit_plugin_items`](crate::renderer::SceneFrame::submit_plugin_items)
    /// has populated a collection under the same name.
    pub fn with_item_type_plugin(
        &mut self,
        device: &crate::gpu::Device,
        mut plugin: Box<dyn crate::plugin_api::ItemTypePlugin>,
    ) {
        let shared = self.resources.shared_bindings();
        plugin.init_gpu(device, &shared);
        let name = plugin.type_name();
        self.item_type_plugins.insert(name, plugin);
    }

    /// Returns true when an item-type plugin with `type_name` is
    /// registered.
    pub fn has_item_type_plugin(&self, type_name: &str) -> bool {
        self.item_type_plugins.contains_key(type_name)
    }

    /// Borrow a registered item-type plugin back as its concrete type.
    ///
    /// `with_item_type_plugin` takes the plugin by box and the renderer owns it
    /// from then on, so this is how a host reaches it again: to upload content
    /// the plugin stores itself, to change its settings, or to read state it
    /// accumulated during a frame.
    ///
    /// `None` when nothing is registered under `type_name`, or when something
    /// is but it is not a `T`.
    pub fn item_type_plugin<T: crate::plugin_api::ItemTypePlugin>(
        &self,
        type_name: &str,
    ) -> Option<&T> {
        self.item_type_plugins
            .get(type_name)?
            .as_any_plugin()
            .downcast_ref::<T>()
    }

    /// Mutably borrow a registered item-type plugin back as its concrete type,
    /// the same lookup as [`item_type_plugin`](Self::item_type_plugin).
    ///
    /// This is the upload route for an item type that owns its own content:
    /// take the plugin and call its own upload method on it, the way content
    /// held by the renderer is uploaded through
    /// [`resources_mut`](Self::resources_mut).
    pub fn item_type_plugin_mut<T: crate::plugin_api::ItemTypePlugin>(
        &mut self,
        type_name: &str,
    ) -> Option<&mut T> {
        self.item_type_plugins
            .get_mut(type_name)?
            .as_any_plugin_mut()
            .downcast_mut::<T>()
    }

    /// Borrow a registered item-type plugin together with the renderer-owned
    /// services an upload into it needs: the job runner and read access to the
    /// shared content arenas.
    ///
    /// Use this rather than [`item_type_plugin_mut`](Self::item_type_plugin_mut)
    /// whenever the call being made on the plugin needs more than the plugin
    /// itself. `item_type_plugin_mut` borrows the whole renderer, so
    /// `renderer.resources()` and the job runner are out of reach for as long
    /// as the plugin is held; these are separate fields, and only the renderer
    /// can lend them out at the same time. The built-in types that hold their
    /// own content upload through this.
    ///
    /// `None` when nothing is registered under `type_name`, or when something
    /// is but it is not a `T`.
    pub fn item_type_plugin_host<T: crate::plugin_api::ItemTypePlugin>(
        &mut self,
        type_name: &str,
    ) -> Option<crate::plugin_api::ItemTypeHost<'_, T>> {
        let jobs = crate::resources::Jobs::new(&self.resources);
        let plugin = self
            .item_type_plugins
            .get_mut(type_name)?
            .as_any_plugin_mut()
            .downcast_mut::<T>()?;
        Some(crate::plugin_api::ItemTypeHost {
            plugin,
            jobs,
            resources: &self.resources,
        })
    }

    /// Register a [`PostEffectProducer`](crate::plugin_api::PostEffectProducer).
    ///
    /// The producer's `init_gpu` runs on the next render (registration takes
    /// no device), followed by `on_viewport_resized` for every viewport that
    /// already has render targets. From then on, each HDR frame runs
    /// `prepare` and `encode` per viewport while `enabled` returns true.
    /// Producers run after the built-in effects, in registration order.
    ///
    /// Returns an id for [`remove_post_effect_producer`](Self::remove_post_effect_producer).
    pub fn add_post_effect_producer(
        &mut self,
        producer: Box<dyn crate::plugin_api::PostEffectProducer>,
    ) -> crate::plugin_api::PostEffectProducerId {
        self.next_post_effect_producer_id += 1;
        let id = crate::plugin_api::PostEffectProducerId(self.next_post_effect_producer_id);
        self.post_effect_producers
            .push(RegisteredPostEffectProducer {
                id,
                gpu_ready: false,
                producer,
            });
        id
    }

    /// Unregister a post-effect producer. Unknown ids are ignored.
    pub fn remove_post_effect_producer(&mut self, id: crate::plugin_api::PostEffectProducerId) {
        self.post_effect_producers.retain(|p| p.id != id);
    }

    /// Register a [`PostEffectStage`](crate::plugin_api::PostEffectStage) at
    /// `order` in the post-composite chain.
    ///
    /// The chain runs in ascending order of the key; the built-in FXAA sits
    /// at [`stage_order::ANTI_ALIASING`](crate::plugin_api::post_effect::stage_order::ANTI_ALIASING)
    /// (0), and [`stage_order::EXTERNAL_DEFAULT`](crate::plugin_api::post_effect::stage_order::EXTERNAL_DEFAULT)
    /// (100) is the conventional post-AA band. Negative keys run before AA.
    /// Stages sharing a key run in registration order, built-ins first. The
    /// stage's `init_gpu` runs on the next render, followed by
    /// `on_viewport_resized` for every viewport that already has render
    /// targets; each HDR frame then runs `prepare` and `encode` per viewport
    /// while `enabled` returns true.
    ///
    /// Returns an id for [`remove_post_effect_stage`](Self::remove_post_effect_stage).
    pub fn add_post_effect_stage(
        &mut self,
        stage: Box<dyn crate::plugin_api::PostEffectStage>,
        order: i32,
    ) -> crate::plugin_api::PostEffectStageId {
        self.next_post_effect_stage_id += 1;
        let id = crate::plugin_api::PostEffectStageId(self.next_post_effect_stage_id);
        self.post_effect_stages.push(RegisteredPostEffectStage {
            id,
            order,
            gpu_ready: false,
            stage,
        });
        id
    }

    /// Unregister a post-effect stage. Unknown ids are ignored.
    pub fn remove_post_effect_stage(&mut self, id: crate::plugin_api::PostEffectStageId) {
        self.post_effect_stages.retain(|s| s.id != id);
    }

    /// Run deferred GPU init for post-effect producers and stages registered
    /// since the last render: `init_gpu`, then `on_viewport_resized` for
    /// each viewport that already has render targets.
    pub(crate) fn init_pending_post_effect_producers(&mut self, device: &crate::gpu::Device) {
        if self.post_effect_producers.iter().all(|p| p.gpu_ready)
            && self.post_effect_stages.iter().all(|s| s.gpu_ready)
        {
            return;
        }
        let target_format = self.resources.target_format;
        let live: Vec<crate::plugin_api::PostEffectResizeContext<'_>> = self
            .viewport_slots
            .iter()
            .enumerate()
            .filter_map(|(vp_idx, slot)| {
                slot.hdr
                    .as_ref()
                    .map(|hdr| crate::plugin_api::PostEffectResizeContext {
                        viewport_index: vp_idx,
                        scene_size: hdr.scene_size,
                        output_size: hdr.output_size,
                        scene_colour: &hdr.hdr_view,
                        scene_depth: &hdr.hdr_depth_only_view,
                        target_format,
                    })
            })
            .collect();
        for entry in &mut self.post_effect_producers {
            if entry.gpu_ready {
                continue;
            }
            entry.producer.init_gpu(device);
            for ctx in &live {
                entry.producer.on_viewport_resized(device, ctx);
            }
            entry.gpu_ready = true;
        }
        for entry in &mut self.post_effect_stages {
            if entry.gpu_ready {
                continue;
            }
            entry.stage.init_gpu(device);
            for ctx in &live {
                entry.stage.on_viewport_resized(device, ctx);
            }
            entry.gpu_ready = true;
        }
    }

    /// Notify every registered item-type plugin that the wgpu device has been
    /// recreated (device loss, surface re-init, host-driven reset).
    ///
    /// Calls [`ItemTypePlugin::on_device_recreated`](crate::plugin_api::ItemTypePlugin::on_device_recreated)
    /// on each plugin, then re-runs its `init_gpu` against the new device and
    /// the current shared bind layout, mirroring registration. The renderer
    /// does not detect device loss on its own; the host invokes this after it
    /// recreates the device. Matches
    /// [`ViewportRuntime::notify_device_recreated`](crate::runtime::ViewportRuntime::notify_device_recreated)
    /// on the GPU-plugin side.
    pub fn notify_device_recreated(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        let shared = self.resources.shared_bindings();
        for plugin in self.item_type_plugins.values_mut() {
            plugin.on_device_recreated(device, queue);
            plugin.init_gpu(device, &shared);
        }
        // Post-effect producers and stages follow the same shape, then
        // re-receive the per-viewport resize signal so their targets are
        // rebuilt against the new device.
        for entry in &mut self.post_effect_producers {
            entry.producer.on_device_recreated(device, queue);
            entry.producer.init_gpu(device);
            entry.gpu_ready = true;
        }
        for entry in &mut self.post_effect_stages {
            entry.stage.on_device_recreated(device, queue);
            entry.stage.init_gpu(device);
            entry.gpu_ready = true;
        }
        let target_format = self.resources.target_format;
        for (vp_idx, slot) in self.viewport_slots.iter().enumerate() {
            let Some(hdr) = slot.hdr.as_ref() else {
                continue;
            };
            let ctx = crate::plugin_api::PostEffectResizeContext {
                viewport_index: vp_idx,
                scene_size: hdr.scene_size,
                output_size: hdr.output_size,
                scene_colour: &hdr.hdr_view,
                scene_depth: &hdr.hdr_depth_only_view,
                target_format,
            };
            for entry in &mut self.post_effect_producers {
                entry.producer.on_viewport_resized(device, &ctx);
            }
            for entry in &mut self.post_effect_stages {
                entry.stage.on_viewport_resized(device, &ctx);
            }
        }
    }

    /// Walk registered item-type plugins, invoke `prepare` for each one
    /// that has a matching collection submitted on `frame.scene`, and
    /// return the concatenated command buffers.
    ///
    /// Called internally from the lib's prepare paths; not part of the
    /// consumer-facing API.
    pub(crate) fn dispatch_plugin_prepare(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) -> Vec<crate::gpu::CommandBuffer> {
        if self.item_type_plugins.is_empty() {
            return Vec::new();
        }
        // Derivative renders (captures / bakes) dispatch too: they run their
        // own prepare and cull against the capture camera, so the matching
        // paint draws current, right-camera geometry, the same as built-in
        // item types. The presented frame re-prepares afterwards anyway.
        //
        // Plugin prepare runs before the rest of the lib's prepare, so the
        // shared LUT set a plugin may resolve through the context has to be
        // resident already; the call is a no-op after the first frame.
        self.resources.ensure_colourmaps_initialized(device, queue);
        self.plugin_frame_index = self.plugin_frame_index.wrapping_add(1);
        let mut bufs: Vec<crate::gpu::CommandBuffer> = Vec::new();
        for (name, plugin) in self.item_type_plugins.iter_mut() {
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                // Constructed per plugin because `Jobs` borrows `&resources`
                // and the borrow only needs to live for this iteration.
                let ctx = crate::plugin_api::ItemFrameContext {
                    camera: &frame.camera.render_camera,
                    viewport_size: glam::Vec2::from(frame.camera.viewport_size),
                    viewport_index: frame.camera.viewport_index,
                    frame_index: self.plugin_frame_index,
                    jobs: crate::resources::Jobs::new(&self.resources),
                    resources: &self.resources,
                    wireframe_mode: frame.viewport.wireframe_mode,
                    outline_selected: frame.interaction.outline_selected,
                    sub_selection: frame.interaction.sub_selection.as_ref(),
                    clip_objects: &frame.effects.clip.objects,
                    quality_reduced: self.degradation_volume_quality_reduced,
                    decal_excluded_surfaces: &self.decal_excluded_surfaces,
                    ref_items: crate::renderer::item_plugins::plugin_ref_items_for(frame, name),
                };
                bufs.extend(plugin.prepare(device, queue, &ctx, items));
            }
        }
        bufs
    }

    /// Collect every plugin's wireframe polylines and upload them into the
    /// shared line substrate.
    ///
    /// Runs in scene prepare, right after the substrate's own producers
    /// (isolines, clip outlines), so a plugin's wireframe draws in the same
    /// pass and the same order relative to scene geometry as before these
    /// moved behind the seam. Two phases because the context borrows
    /// `resources` while the upload needs it mutably.
    pub(crate) fn dispatch_plugin_wireframes(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        let mut polylines: Vec<crate::renderer::PolylineItem> = Vec::new();
        for (name, plugin) in self.item_type_plugins.iter() {
            let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) else {
                continue;
            };
            if items.is_empty() {
                continue;
            }
            let ctx = crate::plugin_api::ItemFrameContext {
                camera: &frame.camera.render_camera,
                viewport_size: glam::Vec2::from(frame.camera.viewport_size),
                viewport_index: frame.camera.viewport_index,
                frame_index: self.plugin_frame_index,
                jobs: crate::resources::Jobs::new(&self.resources),
                resources: &self.resources,
                wireframe_mode: frame.viewport.wireframe_mode,
                outline_selected: frame.interaction.outline_selected,
                sub_selection: frame.interaction.sub_selection.as_ref(),
                clip_objects: &frame.effects.clip.objects,
                quality_reduced: self.degradation_volume_quality_reduced,
                decal_excluded_surfaces: &self.decal_excluded_surfaces,
                ref_items: crate::renderer::item_plugins::plugin_ref_items_for(frame, name),
            };
            polylines.extend(plugin.wireframe_polylines(items, &ctx));
        }
        if polylines.is_empty() {
            return;
        }
        self.resources.ensure_polyline_pipeline(device);
        let vp_size = frame.camera.viewport_size;
        for item in &polylines {
            if item.positions.is_empty() {
                continue;
            }
            let mut gpu = self
                .resources
                .upload_polyline_per_frame(device, queue, item, vp_size);
            // A plugin asks for the thin single-pixel line by setting the flag
            // on the item it returns; the substrate keys its pipeline on the
            // uploaded data rather than on the item.
            gpu.wireframe = item.settings.wireframe;
            self.polyline_gpu_data.push(gpu);
        }
    }

    /// Walk registered item-type plugins and invoke `paint` for each one
    /// that has a matching collection submitted on `frame.scene`.
    ///
    /// Called from inside the lib's HDR scene pass after built-in opaques
    /// and the skybox (`is_hdr` true), and from the LDR scene positions
    /// after the built-in scene content (`is_hdr` false). On the LDR path
    /// only plugins that opt in via `draws_ldr` are invoked: the pass
    /// targets the renderer's output format, and a plugin without a
    /// pipeline for it must be skipped, not handed an incompatible pass.
    pub(crate) fn dispatch_plugin_paint(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        frame: &FrameData,
        is_hdr: bool,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        let ctx = crate::plugin_api::PaintContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
            target_format: if is_hdr {
                crate::resources::HDR_COLOR_FORMAT
            } else {
                self.resources.target_format
            },
            meshes: crate::resources::MeshDraw::new(&self.resources),
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if !is_hdr && !plugin.draws_ldr() {
                continue;
            }
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                plugin.paint(pass, &ctx, items);
            }
        }
    }

    /// `true` when at least one registered plugin has a non-empty collection
    /// submitted this frame under its own type name. A collection submitted
    /// for a type name with no matching registered plugin does not count,
    /// since no plugin will ever draw it. Used to gate the pick pass and the
    /// OIT pass; does not inspect per-item `pick_id`, `render_pick` skips
    /// non-pickable items.
    pub(crate) fn any_plugin_items_submitted(&self, frame: &FrameData) -> bool {
        !self.item_type_plugins.is_empty()
            && self.item_type_plugins.keys().any(|name| {
                crate::renderer::item_plugins::plugin_collections_for(frame, name)
                    .any(|items| !items.is_empty())
            })
    }

    /// `true` when the foreground pass has work this frame: submitted
    /// foreground items, or a registered foreground-drawing plugin with a
    /// non-empty collection.
    pub(crate) fn foreground_active(&self, frame: &FrameData) -> bool {
        !frame.scene.foreground_items.is_empty()
            || self.item_type_plugins.iter().any(|(name, plugin)| {
                plugin.draws_foreground()
                    && crate::renderer::item_plugins::plugin_items_for(frame, name)
                        .is_some_and(|items| !items.is_empty())
            })
    }

    /// Walk registered plugins and invoke `paint_foreground` for each
    /// foreground-drawing plugin whose collection is on `frame.scene`.
    ///
    /// Called from inside the foreground pass after the built-in item
    /// draws. `camera` carries the foreground projection so plugin-side
    /// math agrees with the bound group-0 camera.
    pub(crate) fn dispatch_plugin_paint_foreground(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        frame: &FrameData,
        camera: &RenderCamera,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        let ctx = crate::plugin_api::PaintContext {
            camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
            target_format: crate::resources::HDR_COLOR_FORMAT,
            meshes: crate::resources::MeshDraw::new(&self.resources),
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if !plugin.draws_foreground() {
                continue;
            }
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                plugin.paint_foreground(pass, &ctx, items);
            }
        }
    }

    /// `true` when the read-only-depth pass has work this frame: a registered
    /// depth-read-drawing plugin with a non-empty collection. Gates
    /// `hdr_depth_read_pass` so a frame with no opted-in plugin begins no extra
    /// pass and triggers no depth-attachment transition.
    pub(crate) fn any_plugin_draws_depth_read(&self, frame: &FrameData) -> bool {
        !self.item_type_plugins.is_empty()
            && self.item_type_plugins.iter().any(|(name, plugin)| {
                plugin.draws_depth_read()
                    && crate::renderer::item_plugins::plugin_items_for(frame, name)
                        .is_some_and(|items| !items.is_empty())
            })
    }

    /// Walk registered plugins and invoke `paint_depth_read` for each
    /// depth-read-drawing plugin whose collection is on `frame.scene`.
    ///
    /// Called from inside the read-only-depth pass after the built-in draws.
    /// The scene depth is bound read-only as the pass's depth attachment, so
    /// the plugin samples `scene_depth` (a depth-only view of the same buffer)
    /// instead of writing it. The caller hands over the view + sampler and a
    /// prebuilt bind group so the plugin can either bake the depth into a group
    /// of its own or bind the ready-made group at a spare slot.
    pub(crate) fn dispatch_plugin_paint_depth_read(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        frame: &FrameData,
        scene_depth: &crate::gpu::TextureView,
        scene_depth_sampler: &crate::gpu::Sampler,
        scene_depth_bind_group: &crate::gpu::BindGroup,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        let ctx = crate::plugin_api::DepthReadContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
            scene_depth,
            scene_depth_sampler,
            scene_depth_bind_group,
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if !plugin.draws_depth_read() {
                continue;
            }
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                plugin.paint_depth_read(pass, &ctx, items);
            }
        }
    }

    /// Walk registered plugins and invoke `encode` for each one that asked for
    /// `scope`, handing over the frame's command encoder rather than a begun
    /// render pass.
    ///
    /// Called from the HDR path at each [`EncoderScope`] point. A plugin that
    /// lists no scopes is skipped without building the context.
    ///
    /// [`EncoderScope`]: crate::plugin_api::EncoderScope
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_plugin_encode(
        &self,
        encoder: &mut crate::gpu::CommandEncoder,
        frame: &FrameData,
        scope: crate::plugin_api::EncoderScope,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vp_idx: usize,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        if !self
            .item_type_plugins
            .values()
            .any(|p| p.encoder_scopes().contains(&scope))
        {
            return;
        }
        let Some(slot) = self.viewport_slots.get(vp_idx) else {
            return;
        };
        let Some(slot_hdr) = slot.hdr.as_ref() else {
            return;
        };
        // The HDR attachments, not the supersampled ones: under SSAA the
        // resolve has already run by the time either scope fires, so from here
        // on the frame draws into the HDR target like every other frame.
        let ctx = crate::plugin_api::EncoderScopeContext {
            scope,
            device,
            queue,
            camera: &frame.camera.render_camera,
            viewport_index: vp_idx,
            frame_index: self.plugin_frame_index,
            scene_size: slot_hdr.scene_size,
            camera_bind_group: &slot.camera_bind_group,
            scene_colour: &slot_hdr.hdr_view,
            scene_colour_texture: &slot_hdr.hdr_texture,
            scene_depth: &slot_hdr.hdr_depth_view,
            scene_depth_only: &slot_hdr.hdr_depth_only_view,
            scene_stencil_only: &slot_hdr.hdr_stencil_only_view,
            effects: &frame.effects,
            outline_colour: frame.interaction.outline_colour,
            outline_width_px: frame.interaction.outline_width_px,
            meshes: crate::resources::MeshDraw::new(&self.resources),
            quality_reduced: self.last_stats.volume_quality_reduced,
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if !plugin.encoder_scopes().contains(&scope) {
                continue;
            }
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                plugin.encode(encoder, &ctx, items);
            }
        }
    }

    /// Walk registered plugins and invoke `render_pick` for each one whose
    /// collection is on `frame.scene`.
    ///
    /// Called from inside the GPU pick pass after the built-in draws. The
    /// caller has bound the shared group-0 camera bind group; plugins that
    /// rebind group 0 must restore it.
    pub(crate) fn dispatch_plugin_pick(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        frame: &FrameData,
        mask: crate::renderer::picking::PickMask,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        let ctx = crate::plugin_api::PickPassContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
            mask,
            meshes: crate::resources::MeshDraw::new(&self.resources),
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                plugin.render_pick(pass, &ctx, items);
            }
        }
    }

    /// Walk registered plugins and invoke `paint_transparent` for each
    /// one whose collection is on `frame.scene`.
    ///
    /// Called from inside the lib's OIT render pass, after built-in
    /// transparent draws.
    pub(crate) fn dispatch_plugin_paint_transparent(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        frame: &FrameData,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        let ctx = crate::plugin_api::PaintContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
            target_format: crate::resources::HDR_COLOR_FORMAT,
            meshes: crate::resources::MeshDraw::new(&self.resources),
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                plugin.paint_transparent(pass, &ctx, items);
            }
        }
    }

    /// Walk registered plugins and invoke `cast_shadow_pass` for the
    /// given cascade.
    ///
    /// Currently unused: the shadow-pass call site inlines the plugin
    /// dispatch because the surrounding scope holds a mutable borrow of
    /// `self.resources` that blocks a normal `&self` method call. Kept
    /// alongside the other dispatchers as the natural shape; a future
    /// refactor that splits the resources borrow can switch back.
    #[allow(dead_code)]
    pub(crate) fn dispatch_plugin_shadow(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        frame: &FrameData,
        cascade_idx: u32,
        light_view_proj: glam::Mat4,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        let ctx = crate::plugin_api::ShadowCastContext {
            cascade_idx,
            light_view_proj,
            camera: &frame.camera.render_camera,
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                plugin.cast_shadow_pass(pass, &ctx, items);
            }
        }
    }

    /// Walk registered plugins and invoke `cull` for each one whose
    /// collection is on `frame.scene`.
    ///
    /// Called from the lib's prepare path once the camera frustum for
    /// the frame is known.
    pub(crate) fn dispatch_plugin_cull(
        &mut self,
        frustum: &crate::camera::frustum::Frustum,
        frame: &FrameData,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        for (name, plugin) in self.item_type_plugins.iter_mut() {
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                let ctx = crate::plugin_api::ItemFrameContext {
                    camera: &frame.camera.render_camera,
                    viewport_size: glam::Vec2::from(frame.camera.viewport_size),
                    viewport_index: frame.camera.viewport_index,
                    frame_index: self.plugin_frame_index,
                    jobs: crate::resources::Jobs::new(&self.resources),
                    resources: &self.resources,
                    wireframe_mode: frame.viewport.wireframe_mode,
                    outline_selected: frame.interaction.outline_selected,
                    sub_selection: frame.interaction.sub_selection.as_ref(),
                    clip_objects: &frame.effects.clip.objects,
                    quality_reduced: self.degradation_volume_quality_reduced,
                    decal_excluded_surfaces: &self.decal_excluded_surfaces,
                    ref_items: crate::renderer::item_plugins::plugin_ref_items_for(frame, name),
                };
                plugin.cull(frustum, &ctx, items);
            }
        }
    }

    /// Walk registered item-type plugins and invoke `outline_mask` for
    /// each one whose collection is on `frame.scene`.
    ///
    /// Called from inside the lib's outline-mask render pass.
    /// True when any registered item-type plugin has a selected, non-hidden
    /// item on this frame. The outline offscreen pass gates on this (alongside
    /// the built-in outline buffers) so plugin items can drive the selection
    /// outline the same way built-in items do.
    pub(crate) fn any_plugin_item_selected(&self, frame: &FrameData) -> bool {
        if self.item_type_plugins.is_empty() {
            return false;
        }
        self.item_type_plugins.keys().any(|name| {
            crate::renderer::item_plugins::plugin_collections_for(frame, name).any(|items| {
                (0..items.len()).any(|i| {
                    let s = items.item_settings(i);
                    if s.hidden {
                        return false;
                    }
                    // A sub-object selection on a plugin item drives the
                    // outline machinery the same way whole-item selection
                    // does; without this, sub-level outlines would never
                    // open the mask pass.
                    s.selected
                        || (s.pick_id != PickId::NONE
                            && frame.interaction.sub_selection.as_ref().is_some_and(|sel| {
                                sel.items.iter().any(|(node_id, _)| *node_id == s.pick_id.0)
                            }))
                })
            })
        })
    }

    pub(crate) fn dispatch_plugin_outline_mask(
        &self,
        pass: &mut crate::gpu::RenderPass<'_>,
        frame: &FrameData,
    ) {
        if self.item_type_plugins.is_empty() {
            return;
        }
        let ctx = crate::plugin_api::OutlineMaskContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
            meshes: crate::resources::MeshDraw::new(&self.resources),
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if let Some(items) = crate::renderer::item_plugins::plugin_items_for(frame, name) {
                plugin.outline_mask(pass, &ctx, items);
            }
        }
    }

    /// True when the device supports the features GPU-driven culling needs.
    ///
    /// Plugins should gate `submit_cull` calls on this. If false, the lib
    /// silently no-ops the submission and the plugin must fall back to
    /// direct draws.
    pub fn is_gpu_culling_supported(&self) -> bool {
        self.instancing.gpu_culling_supported
    }

    /// Returns per-frame shadow and lighting pipeline statistics for debug inspection.
    ///
    /// All fields reflect the most recently completed `prepare` call (one frame
    /// behind the display). Returns default values before the first `prepare` call.
    pub fn shadow_debug_stats(&self) -> ShadowDebugStats {
        ShadowDebugStats {
            using_instanced_path: self.instancing.use_instancing,
            instanced_batch_count: self.instancing.batches.len(),
            cascade_count: self.shadow.last_cascade_count,
            cascade_splits: self.shadow.last_cascade_splits,
            shadow_atlas_resolution: self.shadow.last_shadow_atlas_resolution,
            shadow_extent_world: self.shadow.last_shadow_extent,
            contact_shadow_active: self.shadow.last_contact_shadow_active,
        }
    }

    /// Read the debug quantity at a specific pixel, as the visible surface there
    /// resolved it.
    ///
    /// The value comes out of the HDR target after the depth test, so it belongs
    /// to the surface you can see rather than to whichever fragment happened to
    /// shade last. The three colour channels hold the current R/G/B channel
    /// selectors; alpha comes from the target and carries nothing useful.
    ///
    /// Returns `None` unless the last prepared frame could leave the quantity
    /// there, which needs all of:
    ///
    /// - [`DebugVis::active`](crate::DebugVis) set,
    /// - [`PipelineMode::Hdr`](crate::PipelineMode) : the LDR path renders
    ///   straight into your target, which the renderer does not own,
    /// - [`DebugOutputMode::Replace`](crate::DebugOutputMode) : the other modes
    ///   mix the quantity with the shaded colour, so what lands in the target is
    ///   not the quantity,
    ///
    /// and `(x, y)` inside the viewport. `None` means the configuration cannot
    /// be answered, never that the pixel had no value.
    ///
    /// Values are half-float, so expect about three decimal digits. With
    /// supersampling on, the texel has been resolved from several samples and is
    /// a filtered average rather than one surface's value.
    ///
    /// This submits a GPU-to-CPU copy and waits synchronously. Only call from outside
    /// a render pass (e.g., in the next frame's prepare step), not inside paint callbacks.
    ///
    /// The returned values are from the previous rendered frame.
    pub fn read_debug_pixel(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        x: u32,
        y: u32,
    ) -> Option<[f32; 4]> {
        // Use the primary viewport slot (index 0).
        let slot = self.viewport_slots.first()?;
        let (vw, vh) = slot.debug_readback_dims?;
        if x >= vw || y >= vh {
            return None;
        }
        let texture = &slot.hdr.as_ref()?.hdr_texture;

        // One texel out of the HDR target. A buffer copy needs its rows aligned
        // even for a single row, so the staging buffer is a whole aligned row
        // and only its first texel is read.
        let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("debug_pixel_staging"),
            size: u64::from(crate::gpu::COPY_BYTES_PER_ROW_ALIGNMENT),
            usage: crate::gpu::BufferUsages::MAP_READ | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            crate::gpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: crate::gpu::Origin3d { x, y, z: 0 },
                aspect: crate::gpu::TextureAspect::All,
            },
            crate::gpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: crate::gpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(crate::gpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
                },
            },
            crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), crate::gpu::BufferAsyncError>>();
        slice.map_async(crate::gpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(crate::gpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        });
        rx.recv().ok()?.ok()?;
        let data = crate::gpu::mapped_range(slice);
        // Rgba16Float: four halves.
        Some(std::array::from_fn(|c| {
            let bits = u16::from_le_bytes([data[c * 2], data[c * 2 + 1]]);
            half::f16::from_bits(bits).to_f32()
        }))
    }

    /// Upload a Gaussian splat set to the GPU.
    ///
    /// Call once per splat set at startup or when it changes. The returned
    /// [`GaussianSplatId`] is valid until [`free_gaussian_splat`](Self::free_gaussian_splat) is called.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// if `data.positions` is empty or if `positions`, `scales`, `rotations`, and `opacities`
    /// differ in length.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use viewport_lib::error::ViewportError;
    /// # use viewport_lib::renderer::{GaussianSplatData, ViewportRenderer};
    /// # fn demo(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device, queue: &viewport_lib::wgpu::Queue) {
    /// let result = renderer.upload_gaussian_splat(device, queue, &GaussianSplatData::default());
    /// assert!(matches!(result, Err(ViewportError::InvalidGaussianSplatData { .. })));
    /// # }
    /// ```
    pub fn upload_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: &GaussianSplatData,
    ) -> crate::error::ViewportResult<GaussianSplatId> {
        self.gaussian_splat_plugin_mut()?
            .upload(device, queue, data)
    }

    /// Replace the splats behind a live [`GaussianSplatId`], keeping the handle.
    ///
    /// Items already holding the handle pick up the new set on the next frame.
    /// Use this for content that changes over time, such as a streamed or
    /// re-trained splat set.
    ///
    /// # Errors
    ///
    /// [`InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// when `data` is empty or its per-attribute vectors disagree in length, or
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` no
    /// longer resolves to a live set.
    pub fn replace_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: GaussianSplatId,
        data: &GaussianSplatData,
    ) -> crate::error::ViewportResult<()> {
        self.gaussian_splat_plugin_mut()?
            .replace(device, queue, id, data)
    }

    /// Remove an uploaded Gaussian splat set by handle.
    ///
    /// After this call the `id` is invalid and must not be submitted in `SceneFrame`.
    pub fn free_gaussian_splat(&mut self, id: GaussianSplatId) {
        if let Ok(plugin) = self.gaussian_splat_plugin_mut() {
            plugin.free(id);
        }
    }

    /// The registered Gaussian splat item type, which holds the uploaded sets.
    fn gaussian_splat_plugin_mut(
        &mut self,
    ) -> crate::error::ViewportResult<
        &mut crate::renderer::item_plugins::gaussian_splat::GaussianSplatPlugin,
    > {
        let name = crate::renderer::item_plugins::gaussian_splat::TYPE_NAME;
        self.item_type_plugin_mut(name)
            .ok_or(crate::error::ViewportError::ItemTypePluginMissing { type_name: name })
    }

    // -------------------------------------------------------------------------
    // Uploads for the item types that hold their own content.
    //
    // Each of these forwards to the `DeviceResources` method of the same name.
    // They live here because the renderer, not `DeviceResources`, owns the
    // registered item types: an upload that has to reach a type's own storage
    // can only be reached from this level.
    // -------------------------------------------------------------------------

    /// Upload a polyline for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::PolylineItem,
    ) -> crate::resources::PolylineId {
        self.resources.upload_polyline(device, queue, item)
    }

    /// Start an off-thread upload of a polyline. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_polyline`](Self::upload_result_polyline).
    pub fn begin_upload_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::PolylineItem,
    ) -> crate::resources::JobId {
        self.resources.begin_upload_polyline(device, queue, item)
    }

    /// Take the handle from a finished [`begin_upload_polyline`](Self::begin_upload_polyline) job.
    pub fn upload_result_polyline(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::PolylineId> {
        self.resources.upload_result_polyline(id)
    }

    /// Replace the geometry behind a polyline handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::PolylineId,
        item: &crate::renderer::PolylineItem,
    ) -> bool {
        self.resources.replace_polyline(device, queue, id, item)
    }

    /// Release a polyline. `false` if the handle does not resolve.
    pub fn drop_polyline(&mut self, id: crate::resources::PolylineId) -> bool {
        self.resources.drop_polyline(id)
    }

    /// Upload a streamtube for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::StreamtubeItem,
    ) -> crate::resources::StreamtubeId {
        let host = self.streamtube_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a streamtube. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_streamtube`](Self::upload_result_streamtube).
    pub fn begin_upload_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::StreamtubeItem,
    ) -> crate::resources::JobId {
        let host = self.streamtube_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_streamtube`](Self::begin_upload_streamtube) job.
    pub fn upload_result_streamtube(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::StreamtubeId> {
        let host = self.streamtube_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a streamtube handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_streamtube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::StreamtubeId,
        item: &crate::renderer::StreamtubeItem,
    ) -> bool {
        let host = self.streamtube_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a streamtube. `false` if the handle does not resolve.
    pub fn drop_streamtube(&mut self, id: crate::resources::StreamtubeId) -> bool {
        self.streamtube_host().plugin.drop_stored(id)
    }

    /// Upload a tube for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::TubeItem,
    ) -> crate::resources::TubeId {
        let host = self.tube_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a tube. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_tube`](Self::upload_result_tube).
    pub fn begin_upload_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::TubeItem,
    ) -> crate::resources::JobId {
        let host = self.tube_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_tube`](Self::begin_upload_tube) job.
    pub fn upload_result_tube(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TubeId> {
        let host = self.tube_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a tube handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_tube(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::TubeId,
        item: &crate::renderer::TubeItem,
    ) -> bool {
        let host = self.tube_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a tube. `false` if the handle does not resolve.
    pub fn drop_tube(&mut self, id: crate::resources::TubeId) -> bool {
        self.tube_host().plugin.drop_stored(id)
    }

    /// Upload a ribbon for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::RibbonItem,
    ) -> crate::resources::RibbonId {
        let host = self.ribbon_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a ribbon. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_ribbon`](Self::upload_result_ribbon).
    pub fn begin_upload_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::RibbonItem,
    ) -> crate::resources::JobId {
        let host = self.ribbon_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_ribbon`](Self::begin_upload_ribbon) job.
    pub fn upload_result_ribbon(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::RibbonId> {
        let host = self.ribbon_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a ribbon handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_ribbon(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::RibbonId,
        item: &crate::renderer::RibbonItem,
    ) -> bool {
        let host = self.ribbon_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a ribbon. `false` if the handle does not resolve.
    pub fn drop_ribbon(&mut self, id: crate::resources::RibbonId) -> bool {
        self.ribbon_host().plugin.drop_stored(id)
    }

    /// Upload a point cloud for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::PointCloudItem,
    ) -> crate::resources::PointCloudId {
        let host = self.point_cloud_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a point cloud. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_point_cloud`](Self::upload_result_point_cloud).
    pub fn begin_upload_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::PointCloudItem,
    ) -> crate::resources::JobId {
        let host = self.point_cloud_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_point_cloud`](Self::begin_upload_point_cloud) job.
    pub fn upload_result_point_cloud(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::PointCloudId> {
        let host = self.point_cloud_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a point cloud handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_point_cloud(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::PointCloudId,
        item: &crate::renderer::PointCloudItem,
    ) -> bool {
        let host = self.point_cloud_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a point cloud. `false` if the handle does not resolve.
    pub fn drop_point_cloud(&mut self, id: crate::resources::PointCloudId) -> bool {
        self.point_cloud_host().plugin.drop_stored(id)
    }

    /// The registered streamtube item type, which holds the uploaded curves.
    fn streamtube_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::curves::StreamtubePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::curves::STREAMTUBE_TYPE_NAME)
            .expect("the built-in streamtube item type is registered at construction")
    }

    /// The registered tube item type, which holds the uploaded curves.
    fn tube_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::curves::TubePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::curves::TUBE_TYPE_NAME)
            .expect("the built-in tube item type is registered at construction")
    }

    /// The registered ribbon item type, which holds the uploaded curves.
    fn ribbon_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::curves::RibbonPlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::curves::RIBBON_TYPE_NAME)
            .expect("the built-in ribbon item type is registered at construction")
    }

    /// The registered glyph item type, which holds the uploaded sets.
    fn glyph_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::glyph::GlyphPlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::glyph::TYPE_NAME)
            .expect("the built-in glyph item type is registered at construction")
    }

    /// The registered tensor glyph item type, which holds the uploaded sets.
    fn tensor_glyph_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<
        '_,
        crate::renderer::item_plugins::tensor_glyph::TensorGlyphPlugin,
    > {
        self.item_type_plugin_host(crate::renderer::item_plugins::tensor_glyph::TYPE_NAME)
            .expect("the built-in tensor glyph item type is registered at construction")
    }

    /// The registered sprite item type, which holds the uploaded batches.
    fn sprite_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<'_, crate::renderer::item_plugins::sprite::SpritePlugin>
    {
        self.item_type_plugin_host(crate::renderer::item_plugins::sprite::TYPE_NAME)
            .expect("the built-in sprite item type is registered at construction")
    }

    /// The registered point cloud item type, which holds the uploaded clouds.
    fn point_cloud_host(
        &mut self,
    ) -> crate::plugin_api::ItemTypeHost<
        '_,
        crate::renderer::item_plugins::point_cloud::PointCloudPlugin,
    > {
        self.item_type_plugin_host(crate::renderer::item_plugins::point_cloud::TYPE_NAME)
            .expect("the built-in point cloud item type is registered at construction")
    }

    /// Upload a glyph set for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::GlyphItem,
    ) -> crate::resources::GlyphSetId {
        let host = self.glyph_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a glyph set. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_glyph_set`](Self::upload_result_glyph_set).
    pub fn begin_upload_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::GlyphItem,
    ) -> crate::resources::JobId {
        let host = self.glyph_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_glyph_set`](Self::begin_upload_glyph_set) job.
    pub fn upload_result_glyph_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::GlyphSetId> {
        let host = self.glyph_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a glyph set handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::GlyphSetId,
        item: &crate::renderer::GlyphItem,
    ) -> bool {
        let host = self.glyph_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a glyph set. `false` if the handle does not resolve.
    pub fn drop_glyph_set(&mut self, id: crate::resources::GlyphSetId) -> bool {
        self.glyph_host().plugin.drop_stored(id)
    }

    /// Upload a tensor glyph set for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::TensorGlyphItem,
    ) -> crate::resources::TensorGlyphSetId {
        let host = self.tensor_glyph_host();
        host.plugin.upload(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a tensor glyph set. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_tensor_glyph_set`](Self::upload_result_tensor_glyph_set).
    pub fn begin_upload_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::TensorGlyphItem,
    ) -> crate::resources::JobId {
        let host = self.tensor_glyph_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_tensor_glyph_set`](Self::begin_upload_tensor_glyph_set) job.
    pub fn upload_result_tensor_glyph_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TensorGlyphSetId> {
        let host = self.tensor_glyph_host();
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Replace the geometry behind a tensor glyph set handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_tensor_glyph_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::TensorGlyphSetId,
        item: &crate::renderer::TensorGlyphItem,
    ) -> bool {
        let host = self.tensor_glyph_host();
        host.plugin.replace(device, queue, host.resources, id, item)
    }

    /// Release a tensor glyph set. `false` if the handle does not resolve.
    pub fn drop_tensor_glyph_set(&mut self, id: crate::resources::TensorGlyphSetId) -> bool {
        self.tensor_glyph_host().plugin.drop_stored(id)
    }

    /// Upload a sprite set for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::SpriteItem,
    ) -> crate::resources::SpriteSetId {
        let host = self.sprite_host();
        host.plugin.upload_set(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a sprite set. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_sprite_set`](Self::upload_result_sprite_set).
    pub fn begin_upload_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::SpriteItem,
    ) -> crate::resources::JobId {
        let host = self.sprite_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_sprite_set`](Self::begin_upload_sprite_set) job.
    pub fn upload_result_sprite_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::SpriteSetId> {
        let host = self.sprite_host();
        host.plugin.take_set_result(&host.jobs, id)
    }

    /// Replace the geometry behind a sprite set handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::SpriteSetId,
        item: &crate::renderer::SpriteItem,
    ) -> bool {
        let host = self.sprite_host();
        host.plugin
            .replace_set(device, queue, host.resources, id, item)
    }

    /// Release a sprite set. `false` if the handle does not resolve.
    pub fn drop_sprite_set(&mut self, id: crate::resources::SpriteSetId) -> bool {
        self.sprite_host().plugin.drop_set(id)
    }

    /// Upload a sprite instance set for reuse across frames, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::SpriteItem,
    ) -> crate::resources::SpriteInstanceSetId {
        let host = self.sprite_host();
        host.plugin
            .upload_instance_set(device, queue, host.resources, item)
    }

    /// Start an off-thread upload of a sprite instance set. Poll the returned job with
    /// [`upload_status`](Self::upload_status) and take the handle from
    /// [`upload_result_sprite_instance_set`](Self::upload_result_sprite_instance_set).
    pub fn begin_upload_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::SpriteItem,
    ) -> crate::resources::JobId {
        let host = self.sprite_host();
        host.plugin
            .begin_upload(&host.jobs, device, queue, host.resources, item)
    }

    /// Take the handle from a finished [`begin_upload_sprite_instance_set`](Self::begin_upload_sprite_instance_set) job.
    pub fn upload_result_sprite_instance_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::SpriteInstanceSetId> {
        let host = self.sprite_host();
        host.plugin.take_instance_set_result(&host.jobs, id)
    }

    /// Replace the geometry behind a sprite instance set handle, keeping the handle valid.
    /// `false` if the handle does not resolve.
    pub fn replace_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::SpriteInstanceSetId,
        item: &crate::renderer::SpriteItem,
    ) -> bool {
        let host = self.sprite_host();
        host.plugin
            .replace_instance_set(device, queue, host.resources, id, item)
    }

    /// Release a sprite instance set. `false` if the handle does not resolve.
    pub fn drop_sprite_instance_set(&mut self, id: crate::resources::SpriteInstanceSetId) -> bool {
        self.sprite_host().plugin.drop_instance_set(id)
    }

    /// Upload a scalar volume for GPU marching cubes, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn upload_volume_for_mc(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vol: &crate::geometry::marching_cubes::VolumeData,
    ) -> crate::ViewportResult<crate::resources::McVolumeId> {
        self.gpu_marching_cubes_plugin_mut()?
            .upload(device, queue, vol)
    }

    /// Release a marching-cubes volume and its slab buffers.
    ///
    /// Dropping the buffers takes the volume out of
    /// [`resident_bytes`](Self::resident_bytes) immediately; wgpu defers the
    /// real GPU free until in-flight commands referencing them complete. The
    /// emptied slot is reused by a later upload, at a new generation, so the
    /// freed handle cannot alias its successor.
    pub fn free_mc_volume(&mut self, id: crate::resources::McVolumeId) {
        if let Ok(plugin) = self.gpu_marching_cubes_plugin_mut() {
            plugin.free(id);
        }
    }

    /// Feed a marching-cubes volume from a caller-supplied buffer, refreshed
    /// before every dispatch so the isosurface tracks it with no CPU upload.
    ///
    /// The buffer holds one `f32` per volume node in x-fastest order
    /// (`index = x + y * nx + z * nx * ny`), matching `VolumeData::data`,
    /// starting at `offset_bytes`. It needs `COPY_SRC` usage and
    /// `offset_bytes` must be a multiple of 4. The renderer keeps a clone of
    /// the buffer handle; if the consumer reallocates it, call this again with
    /// the new buffer.
    ///
    /// # Errors
    ///
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` does
    /// not resolve to a live volume,
    /// [`ExternalBufferUsageMissing`](crate::error::ViewportError::ExternalBufferUsageMissing)
    /// if the buffer lacks `COPY_SRC`, or
    /// [`McScalarSourceMismatch`](crate::error::ViewportError::McScalarSourceMismatch)
    /// if the offset is misaligned or the volume's scalars do not fit in the
    /// buffer past `offset_bytes`.
    pub fn set_mc_scalar_source_buffer(
        &mut self,
        id: crate::resources::McVolumeId,
        buffer: crate::gpu::Buffer,
        offset_bytes: u64,
    ) -> crate::ViewportResult<()> {
        self.gpu_marching_cubes_plugin_mut()?
            .set_scalar_source(id, buffer, offset_bytes)
    }

    /// Detach the external scalar source, freezing the isosurface at the last
    /// field copied in.
    ///
    /// # Errors
    ///
    /// [`StaleHandle`](crate::error::ViewportError::StaleHandle) if `id` does
    /// not resolve to a live volume.
    pub fn clear_mc_scalar_source(
        &mut self,
        id: crate::resources::McVolumeId,
    ) -> crate::ViewportResult<()> {
        self.gpu_marching_cubes_plugin_mut()?
            .clear_scalar_source(id)
    }

    /// The registered GPU marching cubes item type, which holds the uploaded
    /// volumes.
    fn gpu_marching_cubes_plugin_mut(
        &mut self,
    ) -> crate::error::ViewportResult<
        &mut crate::renderer::item_plugins::gpu_marching_cubes::GpuMarchingCubesPlugin,
    > {
        let name = crate::renderer::item_plugins::gpu_marching_cubes::TYPE_NAME;
        self.item_type_plugin_mut(name)
            .ok_or(crate::error::ViewportError::ItemTypePluginMissing { type_name: name })
    }

    /// Create a persistent GPU particle system, returning its handle.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn create_gpu_particle_system(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        config: &crate::resources::GpuParticleSystemConfig,
    ) -> crate::resources::GpuParticleSystemId {
        self.resources
            .create_gpu_particle_system(device, queue, config)
    }

    /// Release a GPU particle system. The handle stops resolving.
    pub fn drop_gpu_particle_system(&mut self, id: crate::resources::GpuParticleSystemId) {
        self.resources.drop_gpu_particle_system(id)
    }

    /// Create an instance set drawn from a caller-owned positions buffer.
    ///
    /// Prefer this over the [`DeviceResources`] method of the same name: it is
    /// the call that keeps working once an item type owns its own storage.
    pub fn create_external_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        config: &crate::resources::ExternalInstanceSetConfig,
    ) -> crate::error::ViewportResult<crate::resources::ExternalInstanceSetId> {
        self.resources.create_external_instance_set(device, config)
    }

    /// Release an external instance set. Items still naming it are skipped.
    pub fn drop_external_instance_set(&mut self, id: crate::resources::ExternalInstanceSetId) {
        self.resources.drop_external_instance_set(id)
    }

    /// Re-point an external instance set at a different positions buffer.
    pub fn set_external_instance_set_buffer(
        &mut self,
        id: crate::resources::ExternalInstanceSetId,
        positions: crate::gpu::Buffer,
    ) -> crate::error::ViewportResult<()> {
        self.resources
            .set_external_instance_set_buffer(id, positions)
    }

    /// Upload an equirectangular HDR environment map and precompute IBL textures.
    ///
    /// `pixels` is row-major RGBA f32 data (4 floats per texel), `width`x`height`.
    /// This rebuilds camera bind groups so shaders immediately see the new textures.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// if `pixels.len()` does not equal `width * height * 4`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use viewport_lib::error::ViewportError;
    /// # use viewport_lib::renderer::ViewportRenderer;
    /// # fn demo(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device, queue: &viewport_lib::wgpu::Queue) {
    /// // 2x2 RGBA image requires exactly 16 floats.
    /// let result = renderer.upload_environment_map(device, queue, &[0.0f32; 12], 2, 2);
    /// assert!(matches!(result, Err(ViewportError::InvalidTextureData { expected: 16, actual: 12 })));
    /// # }
    /// ```
    pub fn upload_environment_map(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        pixels: &[f32],
        width: u32,
        height: u32,
    ) -> crate::error::ViewportResult<()> {
        crate::resources::material::environment::upload_environment_map(
            &mut self.resources,
            device,
            queue,
            pixels,
            width,
            height,
        )?;
        self.rebuild_camera_bind_groups(device);
        Ok(())
    }

    /// Upload an extra environment into the indexed set and return its handle.
    ///
    /// Unlike [`upload_environment_map`](Self::upload_environment_map), this does
    /// not replace the scene default or the skybox: the environment takes its own
    /// array layer, to be selected per fragment once zone selection lands. Blocks
    /// until the bake finishes, then rebuilds the camera bind groups.
    ///
    /// # Errors
    ///
    /// [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// if `pixels.len()` does not equal `width * height * 4`, or
    /// [`ViewportError::TooManyEnvironments`](crate::error::ViewportError::TooManyEnvironments)
    /// once the environment set is full.
    pub fn upload_environment(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        pixels: &[f32],
        width: u32,
        height: u32,
    ) -> crate::error::ViewportResult<crate::resources::EnvironmentMapId> {
        let env = crate::resources::material::environment::upload_environment(
            &mut self.resources,
            device,
            queue,
            pixels,
            width,
            height,
        )?;
        self.rebuild_camera_bind_groups(device);
        Ok(env)
    }

    /// Set the environment-selection zones. Fragments inside a zone are lit by
    /// that zone's environment (from [`upload_environment`](Self::upload_environment)),
    /// blended by influence weight in overlaps; fragments outside every zone use
    /// the default environment. Replaces any previous set; an empty slice clears
    /// them. Zones past [`MAX_ENV_ZONES`](crate::resources::material::environment::MAX_ENV_ZONES)
    /// are dropped.
    pub fn set_environment_zones(
        &mut self,
        queue: &crate::gpu::Queue,
        zones: &[crate::resources::material::environment::EnvironmentZone],
    ) {
        crate::resources::material::environment::set_environment_zones(
            &mut self.resources,
            queue,
            zones,
        );
    }

    /// Clear all environment-selection zones; every fragment reverts to the
    /// default environment.
    pub fn clear_environment_zones(&mut self) {
        crate::resources::material::environment::clear_environment_zones(&mut self.resources);
    }

    /// Current state of an in-flight upload job.
    pub fn upload_status(&self, id: crate::resources::JobId) -> crate::resources::UploadStatus {
        self.resources.upload_status(id)
    }

    /// Count of upload jobs still in flight.
    pub fn uploads_pending(&self) -> usize {
        self.resources.uploads_pending()
    }

    /// True when the mesh for `id` is uploaded and resident in the store.
    ///
    /// A level query: it reports whether the mesh is present right now, not
    /// whether a promotion edge just fired, so it is unaffected by upload-pipeline
    /// pumping (a capture / bake never changes the answer). Referencing a
    /// non-resident `MeshId` on a scene node is a graceful skip: the item draws
    /// once the mesh lands, so this is for lifecycle decisions (load gating,
    /// eviction, a bake precondition), not for gating a bind. `MeshId` is
    /// generational, so a stale handle for an evicted-and-reused slot reports
    /// `false` rather than aliasing the new mesh.
    pub fn mesh_resident(&self, id: crate::MeshId) -> bool {
        self.resources.mesh(id).is_some()
    }

    /// True when every mesh referenced by `frame`'s surface items is resident.
    ///
    /// Walks the frame's `surfaces` submission and checks each item's `mesh_id`.
    /// Use it to gate a capture / bake on the scene being streamed in, since the
    /// `bake_*` and `capture_*` entries read the currently resident scene. This
    /// looks only at surface (mesh) items; it does not consider non-mesh items,
    /// plugin items, or LOD levels other than the referenced id. For a
    /// whole-scene readiness check, combine it with `uploads_pending() == 0`.
    pub fn frame_fully_resident(&self, frame: &FrameData) -> bool {
        match &frame.scene.surfaces {
            SurfaceSubmission::Flat(items) => items
                .iter()
                .all(|item| self.resources.mesh(item.mesh_id).is_some()),
        }
    }

    /// Wall-clock work duration recorded for an async upload job. See
    /// [`DeviceResources::job_duration`].
    pub fn job_duration(&self, id: crate::resources::JobId) -> Option<std::time::Duration> {
        self.resources.job_duration(id)
    }

    /// Drop the recorded duration for `id` after reading it. See
    /// [`DeviceResources::drop_job_duration`].
    pub fn drop_job_duration(&mut self, id: crate::resources::JobId) {
        self.resources.drop_job_duration(id);
    }

    /// Start an asynchronous 3D volume texture upload. See
    /// [`DeviceResources::begin_upload_volume`].
    pub fn begin_upload_volume(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: Vec<f32>,
        dims: [u32; 3],
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.resources
            .begin_upload_volume(device, queue, data, dims)
    }

    /// Take the volume id produced by a completed
    /// [`begin_upload_volume`](Self::begin_upload_volume) job.
    pub fn upload_result_volume(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::VolumeId> {
        self.resources.upload_result_volume(id)
    }

    /// Overwrite the 3D texture behind `id` in place. See
    /// [`DeviceResources::replace_volume`]. Use this for time-series playback so
    /// resident volume memory stays flat instead of leaking a texture per step.
    pub fn replace_volume(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::VolumeId,
        data: &[f32],
        dims: [u32; 3],
    ) -> bool {
        self.resources.replace_volume(device, queue, id, data, dims)
    }

    /// Free the 3D texture behind `id`, reclaiming its slot. See
    /// [`DeviceResources::free_volume`].
    pub fn free_volume(&mut self, id: crate::resources::VolumeId) -> bool {
        self.resources.free_volume(id)
    }

    /// Start an asynchronous marching-cubes-ready volume upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Slab sizing
    /// and the scalar, intermediate and output buffer allocation run on a
    /// worker thread against cloned `Device` and `Queue` handles. Ownership of
    /// `vol` transfers into the worker. The worker surfaces
    /// [`McBufferTooLarge`](crate::error::ViewportError::McBufferTooLarge)
    /// through `UploadStatus::Failed` when the device's
    /// `max_storage_buffer_binding_size` cannot fit a single Z-cell layer.
    pub fn begin_upload_volume_for_mc(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vol: crate::geometry::marching_cubes::VolumeData,
    ) -> crate::resources::JobId {
        let name = crate::renderer::item_plugins::gpu_marching_cubes::TYPE_NAME;
        let host = self
            .item_type_plugin_host::<crate::renderer::item_plugins::gpu_marching_cubes::GpuMarchingCubesPlugin>(
                name,
            )
            .expect("the built-in marching cubes item type is registered at construction");
        host.plugin.begin_upload(&host.jobs, device, queue, vol)
    }

    /// Take the [`McVolumeId`](crate::resources::McVolumeId) produced by a
    /// completed [`begin_upload_volume_for_mc`](Self::begin_upload_volume_for_mc) job.
    ///
    /// The volume enters the store here, so a handle is minted on the call that
    /// collects the job rather than on a background thread.
    pub fn upload_result_volume_mc(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::McVolumeId> {
        let name = crate::renderer::item_plugins::gpu_marching_cubes::TYPE_NAME;
        let host = self
            .item_type_plugin_host::<crate::renderer::item_plugins::gpu_marching_cubes::GpuMarchingCubesPlugin>(
                name,
            )
            .ok_or(crate::error::ViewportError::ItemTypePluginMissing { type_name: name })?;
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Start an asynchronous boundary-only volume mesh upload. See
    /// [`DeviceResources::begin_upload_volume_mesh`].
    pub fn begin_upload_volume_mesh(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::volume::volume_mesh::VolumeMeshData,
    ) -> crate::resources::JobId {
        self.resources.begin_upload_volume_mesh(device, data)
    }

    /// Take the [`VolumeMeshItem`](crate::VolumeMeshItem)
    /// produced by a completed
    /// [`begin_upload_volume_mesh`](Self::begin_upload_volume_mesh) job.
    pub fn upload_result_volume_mesh(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::VolumeMeshItem> {
        self.resources.upload_result_volume_mesh(id)
    }

    /// Start an asynchronous clipped volume mesh upload. See
    /// [`DeviceResources::begin_upload_clipped_volume_mesh`].
    pub fn begin_upload_clipped_volume_mesh(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::volume::volume_mesh::VolumeMeshData,
        clip_planes: Vec<[f32; 4]>,
    ) -> crate::resources::JobId {
        self.resources
            .begin_upload_clipped_volume_mesh(device, data, clip_planes)
    }

    /// Take the [`VolumeMeshItem`](crate::VolumeMeshItem)
    /// produced by a completed
    /// [`begin_upload_clipped_volume_mesh`](Self::begin_upload_clipped_volume_mesh) job.
    pub fn upload_result_clipped_volume_mesh(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::VolumeMeshItem> {
        self.resources.upload_result_clipped_volume_mesh(id)
    }

    /// Start an asynchronous sparse voxel grid upload. See
    /// [`DeviceResources::begin_upload_sparse_volume_grid_data`].
    pub fn begin_upload_sparse_volume_grid_data(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::SparseVolumeGridData,
    ) -> crate::resources::JobId {
        self.resources
            .begin_upload_sparse_volume_grid_data(device, data)
    }

    /// Take the [`MeshId`](crate::resources::mesh::mesh_store::MeshId) produced by a completed
    /// [`begin_upload_sparse_volume_grid_data`](Self::begin_upload_sparse_volume_grid_data)
    /// job.
    pub fn upload_result_sparse_volume_grid(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::mesh::mesh_store::MeshId> {
        self.resources.upload_result_sparse_volume_grid(id)
    }

    /// Start an asynchronous Gaussian splat upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. The vec4
    /// padding and the storage-buffer writes run on a worker thread against
    /// cloned `Device` and `Queue` handles. Poll
    /// [`upload_status`](crate::resources::DeviceResources::upload_status) and
    /// call [`upload_result_gaussian_splat`](Self::upload_result_gaussian_splat)
    /// once it reads `Ready`.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidGaussianSplatData`](crate::error::ViewportError::InvalidGaussianSplatData)
    /// before any job is submitted when `data.positions` is empty or the
    /// per-attribute vectors disagree in length.
    pub fn begin_upload_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: crate::renderer::GaussianSplatData,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        let name = crate::renderer::item_plugins::gaussian_splat::TYPE_NAME;
        let host = self
            .item_type_plugin_host::<crate::renderer::item_plugins::gaussian_splat::GaussianSplatPlugin>(
                name,
            )
            .ok_or(crate::error::ViewportError::ItemTypePluginMissing { type_name: name })?;
        host.plugin.begin_upload(&host.jobs, device, queue, data)
    }

    /// Take the [`GaussianSplatId`](crate::renderer::GaussianSplatId) produced by a
    /// completed [`begin_upload_gaussian_splat`](Self::begin_upload_gaussian_splat) job.
    ///
    /// The set enters the store here, so a handle is minted on the call that
    /// collects the job rather than on a background thread.
    pub fn upload_result_gaussian_splat(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::renderer::GaussianSplatId> {
        let name = crate::renderer::item_plugins::gaussian_splat::TYPE_NAME;
        let host = self
            .item_type_plugin_host::<crate::renderer::item_plugins::gaussian_splat::GaussianSplatPlugin>(
                name,
            )
            .ok_or(crate::error::ViewportError::ItemTypePluginMissing { type_name: name })?;
        host.plugin.take_upload_result(&host.jobs, id)
    }

    /// Start an asynchronous overlay texture upload. See
    /// [`DeviceResources::begin_upload_overlay_texture`].
    pub fn begin_upload_overlay_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba_data: Vec<u8>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.resources
            .begin_upload_overlay_texture(device, queue, width, height, rgba_data)
    }

    /// Take the [`OverlayTextureId`](crate::renderer::OverlayTextureId) produced by a
    /// completed [`begin_upload_overlay_texture`](Self::begin_upload_overlay_texture) job.
    pub fn upload_result_overlay_texture(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::renderer::OverlayTextureId> {
        self.resources.upload_result_overlay_texture(id)
    }

    /// Register an external, caller-owned colour `TextureView` as an
    /// [`OverlayTextureId`](crate::renderer::OverlayTextureId). See
    /// [`DeviceResources::register_overlay_texture_view`]. Draw an offscreen
    /// viewport as an overlay image (an
    /// [`OverlayShapeItem`](crate::vplt::overlay::OverlayShapeItem) rect with
    /// `with_texture`) in the overlay z-order, no CPU round-trip. Pass the source's
    /// sRGB `render_view()`.
    pub fn register_overlay_texture_view(
        &mut self,
        view: &crate::gpu::TextureView,
        width: u32,
        height: u32,
    ) -> crate::renderer::OverlayTextureId {
        self.resources
            .register_overlay_texture_view(view, width, height)
    }

    /// Re-point an external overlay texture id at a new `TextureView`, keeping the
    /// id stable. See [`DeviceResources::update_overlay_texture_view`].
    pub fn update_overlay_texture_view(
        &mut self,
        id: crate::renderer::OverlayTextureId,
        view: &crate::gpu::TextureView,
        width: u32,
        height: u32,
    ) -> bool {
        self.resources
            .update_overlay_texture_view(id, view, width, height)
    }

    /// True when no upload jobs are in flight.
    pub fn all_uploads_complete(&self) -> bool {
        self.resources.all_uploads_complete()
    }

    /// Register a callback to fire when an upload job finishes. See
    /// [`DeviceResources::on_upload_complete`] for the semantics.
    pub fn on_upload_complete<F>(&mut self, id: crate::resources::JobId, cb: F)
    where
        F: FnOnce(&crate::resources::UploadStatus) + Send + 'static,
    {
        self.resources.on_upload_complete(id, cb);
    }

    /// Start an asynchronous texture upload. See
    /// [`DeviceResources::begin_upload_texture`] for the semantics.
    pub fn begin_upload_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: crate::resources::TextureData,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.resources.begin_upload_texture(device, queue, data)
    }

    /// Start an asynchronous normal-map upload.
    #[deprecated(
        since = "0.23.0",
        note = "build the payload instead: begin_upload_texture(device, queue, TextureData::normal_map(w, h, rgba))"
    )]
    pub fn begin_upload_normal_map(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.begin_upload_texture(
            device,
            queue,
            crate::resources::TextureData::normal_map(width, height, rgba),
        )
    }

    /// Take the texture id from a completed async texture upload. See
    /// [`DeviceResources::upload_result_texture`] for the error
    /// semantics.
    pub fn upload_result_texture(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TextureId> {
        self.resources.upload_result_texture(id)
    }

    /// Start an asynchronous mesh upload.
    ///
    /// Returns a `JobId` immediately. The CPU prep (tangent computation,
    /// vertex repack, normal-line build) runs on a worker thread; GPU
    /// buffer creation and store insertion run on the main thread during
    /// the next `process_uploads` call after the worker finishes. Once the
    /// status is `Ready`, take the produced `MeshId` with
    /// `upload_result_mesh`.
    ///
    /// Ownership of `data` transfers into the worker; clone at the call
    /// site if you need to retain it.
    ///
    /// # Errors
    ///
    /// Same validation errors as `upload_mesh_data` (empty mesh, length
    /// mismatch, invalid vertex index), all reported before the job is
    /// submitted.
    pub fn begin_upload_mesh_data(
        &mut self,
        device: &crate::gpu::Device,
        data: crate::resources::MeshData,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.resources.begin_upload_mesh_data(device, data)
    }

    /// Take the `MeshId` produced by a completed `begin_upload_mesh_data`
    /// job. See [`DeviceResources::upload_result_mesh`] for the error
    /// semantics.
    pub fn upload_result_mesh(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::mesh::mesh_store::MeshId> {
        self.resources.upload_result_mesh(id)
    }

    /// Start an asynchronous environment-map upload.
    ///
    /// Returns immediately with a `JobId`. The caller drives the upload-job
    /// runner from the renderer's prepare path each frame; once the job
    /// reports `Ready`, the IBL textures are live on the renderer and a
    /// subsequent call to `rebuild_camera_bind_groups` makes them visible
    /// to shaders.
    ///
    /// Ownership of `pixels` transfers into the background worker.
    ///
    /// # Errors
    ///
    /// Returns [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// if `pixels.len() != width * height * 4`.
    pub fn begin_upload_environment_map(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        pixels: Vec<f32>,
        width: u32,
        height: u32,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        crate::resources::material::environment::begin_upload_environment_map(
            &mut self.resources,
            device,
            queue,
            pixels,
            width,
            height,
        )
    }

    /// Rebuild the primary and per-viewport camera bind groups.
    ///
    /// Call after IBL textures are uploaded so the shaders see the new
    /// environment. The synchronous `upload_environment_map` does this
    /// internally; consumers driving the async path through
    /// `begin_upload_environment_map` should call this themselves once the
    /// matching job reports `Ready`.
    pub fn rebuild_camera_bind_groups(&mut self, device: &crate::gpu::Device) {
        self.resources.binds.camera_bg = self.resources.create_camera_bind_group(
            device,
            &self.resources.binds.camera_uniform_buf,
            &self.resources.binds.clip_planes_buf,
            &self.resources.shadow.info_buf,
            &self.resources.binds.clip_volume_buf,
            "camera_bind_group",
        );

        for slot in &mut self.viewport_slots {
            slot.camera_bind_group = self.resources.create_camera_bind_group(
                device,
                &slot.camera_buf,
                &slot.clip_planes_buf,
                &slot.shadow_info_buf,
                &slot.clip_volume_buf,
                "per_viewport_camera_bg",
            );
            slot.foreground_camera_bind_group = self.resources.create_camera_bind_group(
                device,
                &slot.foreground_camera_buf,
                &slot.foreground_clip_planes_buf,
                &slot.shadow_info_buf,
                &slot.foreground_clip_volume_buf,
                "per_viewport_foreground_camera_bg",
            );
        }
    }

    /// Ensure a per-viewport slot exists for `viewport_index`.
    ///
    /// Creates a full `ViewportSlot` with independent uniform buffers for camera,
    /// clip planes, clip volume, shadow info, and grid. The camera bind group
    /// references this slot's per-viewport buffers plus shared scene-global
    /// resources. Slots are created lazily and never destroyed.
    fn ensure_viewport_slot(&mut self, device: &crate::gpu::Device, viewport_index: usize) {
        while self.viewport_slots.len() <= viewport_index {
            let camera_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("vp_camera_buf"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let clip_planes_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("vp_clip_planes_buf"),
                size: std::mem::size_of::<ClipPlanesUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let clip_volume_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("vp_clip_volume_buf"),
                size: std::mem::size_of::<ClipVolumesUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // Seeded with the latest shadow atlas uniform rather than zeros:
            // prepare_scene_internal writes shadow info only to slots that
            // exist at that point, so a slot created later in the same frame
            // would otherwise render its first frame with zeroed cascade
            // matrices (NaN shadow UVs, everything shadowed).
            let shadow_info_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("vp_shadow_info_buf"),
                size: std::mem::size_of::<ShadowAtlasUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            crate::resources::builders::write_mapped(
                shadow_info_buf.slice(..),
                bytemuck::cast_slice(&[self.shadow.last_shadow_atlas_uniform]),
            );
            shadow_info_buf.unmap();
            let grid_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("vp_grid_buf"),
                size: std::mem::size_of::<GridUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let camera_bind_group = self.resources.create_camera_bind_group(
                device,
                &camera_buf,
                &clip_planes_buf,
                &shadow_info_buf,
                &clip_volume_buf,
                "per_viewport_camera_bg",
            );

            // Foreground pass group 0: own camera uniform, clip disabled
            // (zeroed uniforms mean count == 0 for both planes and volumes).
            let foreground_camera_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("vp_foreground_camera_buf"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let foreground_clip_planes_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("vp_foreground_clip_planes_buf"),
                size: std::mem::size_of::<ClipPlanesUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });
            let foreground_clip_volume_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("vp_foreground_clip_volume_buf"),
                size: std::mem::size_of::<ClipVolumesUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });
            let foreground_camera_bind_group = self.resources.create_camera_bind_group(
                device,
                &foreground_camera_buf,
                &foreground_clip_planes_buf,
                &shadow_info_buf,
                &foreground_clip_volume_buf,
                "per_viewport_foreground_camera_bg",
            );

            let grid_bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("vp_grid_bind_group"),
                layout: &self.resources.guides.grid_bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_buf.as_entire_binding(),
                }],
            });

            // The transform gizmo and the axes indicator now draw through the 2D
            // overlay system, so no per-viewport geometry buffers are needed for them.

            self.viewport_slots.push(ViewportSlot {
                camera_buf,
                clip_planes_buf,
                clip_volume_buf,
                shadow_info_buf,
                grid_buf,
                camera_bind_group,
                foreground_camera_buf,
                foreground_clip_planes_buf,
                foreground_clip_volume_buf,
                foreground_camera_bind_group,
                foreground_objects: Vec::new(),
                grid_bind_group,
                hdr: None,
                cull: crate::resources::ViewportCullState::new(),
                debug_readback_dims: None,
                selection_outlines: SelectionOutlines::default(),
                xray_object_buffers: Vec::new(),
                constraint_line_buffers: Vec::new(),
                cap_buffers: Vec::new(),
                sub_highlight: None,
                sub_highlight_generation: u64::MAX,
                dyn_res: None,
                hdr_callback: None,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Multi-viewport public API
    // -----------------------------------------------------------------------

    /// Create a new viewport slot and return its handle.
    ///
    /// The returned [`ViewportId`] is stable for the lifetime of the renderer.
    /// Pass it to [`prepare_viewport`](Self::prepare_viewport),
    /// [`paint_viewport`](Self::paint_viewport), and
    /// [`render_viewport`](Self::render_viewport) each frame.
    ///
    /// Also set the viewport slot on the camera frame when building the
    /// [`FrameData`] for this viewport:
    /// ```rust,ignore
    /// let id = renderer.create_viewport(&device);
    /// let frame = FrameData {
    ///     camera: CameraFrame::from_camera(&cam, size).with_viewport_id(id),
    ///     ..Default::default()
    /// };
    /// ```
    pub fn create_viewport(&mut self, device: &crate::gpu::Device) -> ViewportId {
        let idx = self.viewport_slots.len();
        self.ensure_viewport_slot(device, idx);
        ViewportId(idx)
    }

    /// Release the heavy GPU texture memory (HDR targets, OIT, bloom, SSAO) held
    /// by `id`.
    ///
    /// The slot index is not reclaimed : future calls with this `ViewportId` will
    /// lazily recreate the texture resources as needed.  This is useful when a
    /// viewport is hidden or minimised and you want to reduce VRAM pressure without
    /// invalidating the handle.
    pub fn destroy_viewport(&mut self, id: ViewportId) {
        if let Some(slot) = self.viewport_slots.get_mut(id.0) {
            slot.hdr = None;
        }
    }

    /// Returns the owned-encoder rendering path.
    ///
    /// Use when you own the window loop and wgpu encoder (winit, raw wgpu).
    /// See [`OwnedPath`] for available methods.
    pub fn owned(&mut self) -> OwnedPath<'_> {
        OwnedPath { renderer: self }
    }

    /// Returns the pass-based rendering path.
    ///
    /// Use when a framework provides you with a render pass (eframe, iced).
    /// See [`PassPath`] for available methods.
    pub fn pass(&mut self) -> PassPath<'_> {
        PassPath { renderer: self }
    }

    /// Returns a read-only paint view for framework paint callbacks.
    ///
    /// Use this in callbacks where only a shared reference to the renderer is
    /// available (e.g. eframe's `CallbackTrait::paint` where `callback_resources`
    /// is `&CallbackResources`). Exposes only the paint methods, not prepare.
    pub fn pass_view(&self) -> PassView<'_> {
        PassView { renderer: self }
    }

    /// Prepare shared scene data.  Call **once per frame**, before any
    /// [`prepare_viewport`](Self::prepare_viewport) calls.
    ///
    /// `frame` provides the scene content (`frame.scene`) and the primary camera
    /// used for shadow cascade framing (`frame.camera`).  In a multi-viewport
    /// setup use any one viewport's `FrameData` here : typically the perspective
    /// view : as the shadow framing reference.
    ///
    /// `scene_effects` carries the scene-global effects: lighting, environment
    /// map, and scatter settings.  Obtain it by constructing [`SceneEffects`]
    /// directly or via [`EffectsFrame::split`]. Compute filter items are read
    /// from `frame.scene.compute_filter_items`.
    pub(crate) fn prepare_scene(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        scene_effects: &SceneEffects<'_>,
    ) {
        let mut sink = SubmitSink::inline(queue);
        self.prepare_scene_internal(device, queue, frame, scene_effects, &mut sink);
    }

    /// Prepare per-viewport GPU state (camera, clip planes, overlays, axes).
    ///
    /// Call once per viewport per frame, **after** [`prepare_scene`](Self::prepare_scene).
    ///
    /// `id` must have been obtained from [`create_viewport`](Self::create_viewport).
    /// `frame.camera.viewport_index` must equal the slot for `id`; use
    /// [`CameraFrame::with_viewport_id`] when building the frame.
    pub(crate) fn prepare_viewport(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: ViewportId,
        frame: &FrameData,
    ) {
        debug_assert_eq!(
            frame.camera.viewport_index, id.0,
            "frame.camera.viewport_index ({}) must equal the ViewportId ({}); \
             use CameraFrame::with_viewport_id(id)",
            frame.camera.viewport_index, id.0,
        );
        let (_, viewport_fx) = frame.effects.split();
        let mut sink = SubmitSink::inline(queue);
        // The split API never routes through `prepare_into`, so plugin
        // prepare and cull dispatch here, once per viewport, with this
        // viewport's camera in the context. The single-call `prepare` path
        // dispatches from `prepare_into` instead and does not reach this
        // wrapper.
        let plugin_bufs = self.dispatch_plugin_prepare(device, queue, frame);
        if !plugin_bufs.is_empty() {
            sink.extend(plugin_bufs);
        }
        if !self.item_type_plugins.is_empty() {
            let vp = frame.camera.render_camera.view_proj();
            let frustum = crate::camera::frustum::Frustum::from_view_proj(&vp);
            self.dispatch_plugin_cull(&frustum, frame);
        }
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx, &mut sink);
    }

    /// Issue draw calls for `id` into a render pass with any lifetime.
    ///
    /// Identical to [`paint_viewport`](Self::paint_viewport) but accepts a render pass with a
    /// non-`'static` lifetime, making it usable from winit, iced, or raw wgpu where the encoder
    /// creates its own render pass.
    pub(crate) fn paint_viewport_to<'rp>(
        &self,
        render_pass: &mut crate::gpu::RenderPass<'rp>,
        id: ViewportId,
        frame: &FrameData,
    ) {
        let vp_idx = id.0;
        let camera_bg = self.viewport_camera_bind_group(vp_idx);
        let grid_bg = self.viewport_grid_bind_group(vp_idx);
        let vp_slot = self.viewport_slots.get(vp_idx);
        emit_draw_calls!(
            &self.resources,
            &mut *render_pass,
            frame,
            self.instancing.use_instancing,
            &self.instancing.batches,
            camera_bg,
            grid_bg,
            &self.compute_filter_results,
            vp_slot,
            &self.mesh_uniforms.wireframe_bind_groups,
            &self.mesh_uniforms.bind_groups,
            &self.mesh_uniforms.submesh_bind_groups,
            &self.mesh_uniforms.object_indices,
            &self.mesh_uniforms.submesh_indices,
            &self.prepared_surfaces,
            self.per_object_bundle.as_ref()
        );
        self.draw_line_and_instance_layers(&mut *render_pass, camera_bg, false);
        // TransparentVolumeMesh boundary wireframe overlay.
        if !self.mesh_uniforms.tvm_wireframe_draws.is_empty() {
            if let Some(ref tvm_bg) = self.mesh_uniforms.tvm_wireframe_bg {
                render_pass.set_bind_group(0, camera_bg, &[]);
                for mesh_id in &self.mesh_uniforms.tvm_wireframe_draws {
                    if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                        render_pass.set_pipeline(&self.resources.scene.wireframe);
                        bind_deform_group!(
                            render_pass,
                            self.resources,
                            &self.resources.deform.dummy_bind_group
                        );
                        render_pass.set_bind_group(1, tvm_bg, &[]);
                        render_pass.set_vertex_buffer(
                            0,
                            self.resources.geometry.vertex_slice(mesh.vertex_span),
                        );
                        if let Some(edge_buf) = &mesh.edge_index_buffer {
                            render_pass.set_index_buffer(
                                edge_buf.slice(..),
                                crate::gpu::IndexFormat::Uint32,
                            );
                            render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                        }
                    }
                }
            }
        }
        // Item-type plugin paint (LDR opt-in only): after all built-in scene
        // content, mirroring the HDR scene-pass position.
        self.dispatch_plugin_paint(render_pass, frame, false);
        // Shadow atlas viewer overlay.
        if frame.effects.debug.show_shadow_atlas {
            render_pass.set_pipeline(&self.resources.shadow.atlas_viewer_pipeline);
            render_pass.set_bind_group(0, &self.resources.shadow.atlas_viewer_bg, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }

    /// Return a reference to the camera bind group for the given viewport slot.
    ///
    /// Falls back to `resources.camera_bind_group` if no per-viewport slot
    /// exists (e.g. in single-viewport mode before the first prepare call).
    fn viewport_camera_bind_group(&self, viewport_index: usize) -> &crate::gpu::BindGroup {
        self.viewport_slots
            .get(viewport_index)
            .map(|slot| &slot.camera_bind_group)
            .unwrap_or(&self.resources.binds.camera_bg)
    }

    /// Return a reference to the grid bind group for the given viewport slot.
    ///
    /// Falls back to `resources.grid_bind_group` if no per-viewport slot exists.
    fn viewport_grid_bind_group(&self, viewport_index: usize) -> &crate::gpu::BindGroup {
        self.viewport_slots
            .get(viewport_index)
            .map(|slot| &slot.grid_bind_group)
            .unwrap_or(&self.resources.guides.grid_bind_group)
    }

    /// Ensure the dyn-res intermediate render target exists for `vp_idx` at the
    /// given `scaled_size`, creating or recreating it when size changes.
    ///
    /// `surface_size` is the native output dimensions (used to size the upscale
    /// blit correctly). `ensure_dyn_res_pipeline` is called automatically.
    pub(crate) fn ensure_dyn_res_target(
        &mut self,
        device: &crate::gpu::Device,
        vp_idx: usize,
        scaled_size: [u32; 2],
        surface_size: [u32; 2],
    ) {
        self.resources.ensure_dyn_res_pipeline(device);
        let needs_create = match &self.viewport_slots[vp_idx].dyn_res {
            None => true,
            Some(dr) => dr.scaled_size != scaled_size || dr.surface_size != surface_size,
        };
        if needs_create {
            let target = self
                .resources
                .create_dyn_res_target(device, scaled_size, surface_size);
            self.viewport_slots[vp_idx].dyn_res = Some(target);
        }
    }

    /// Ensure per-viewport HDR state exists for `viewport_index` at dimensions `w`x`h`.
    ///
    /// Calls `ensure_hdr_shared` once to initialise shared pipelines/BGLs/samplers, then
    /// lazily creates or resizes the `ViewportHdrState` inside the slot. Idempotent: if the
    /// slot already has HDR state at the correct size nothing is recreated.
    pub(crate) fn ensure_viewport_hdr(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        viewport_index: usize,
        w: u32,
        h: u32,
        ssaa_factor: u32,
        render_scale: f32,
    ) {
        let format = self.resources.target_format;
        // Ensure shared infrastructure (pipelines, BGLs, samplers) exists.
        self.resources.ensure_hdr_shared(device, queue, format);
        // When render_scale < 1.0, the HDR upscale path needs the dyn_res
        // pipeline and sampler for the final upscale-blit to output resolution.
        if render_scale < 1.0 - 0.001 {
            self.resources.ensure_dyn_res_pipeline(device);
        }
        // Compute the scene-resolution render target size.
        let scale = render_scale.clamp(0.1, 1.0);
        let scene_w = ((w as f32) * scale).round() as u32;
        let scene_h = ((h as f32) * scale).round() as u32;
        // Ensure the slot exists.
        self.ensure_viewport_slot(device, viewport_index);
        let slot = &mut self.viewport_slots[viewport_index];
        // Create or resize the per-viewport HDR state.
        let needs_create = match &slot.hdr {
            None => true,
            Some(s) => {
                s.output_size != [w, h]
                    || s.scene_size != [scene_w.max(1), scene_h.max(1)]
                    || s.ssaa_factor != ssaa_factor
            }
        };
        if needs_create {
            slot.hdr = Some(self.resources.create_hdr_viewport_state(
                device,
                queue,
                format,
                w,
                h,
                scene_w.max(1),
                scene_h.max(1),
                ssaa_factor,
            ));
            // Tell post-effect producers and stages this viewport's targets
            // changed so they can reallocate their own. Any still awaiting
            // deferred `init_gpu` get this signal during that init instead.
            let hdr = slot.hdr.as_ref().unwrap();
            let ctx = crate::plugin_api::PostEffectResizeContext {
                viewport_index,
                scene_size: hdr.scene_size,
                output_size: hdr.output_size,
                scene_colour: &hdr.hdr_view,
                scene_depth: &hdr.hdr_depth_only_view,
                target_format: format,
            };
            for entry in &mut self.post_effect_producers {
                if entry.gpu_ready {
                    entry.producer.on_viewport_resized(device, &ctx);
                }
            }
            for entry in &mut self.post_effect_stages {
                if entry.gpu_ready {
                    entry.stage.on_viewport_resized(device, &ctx);
                }
            }
        }
    }
}
