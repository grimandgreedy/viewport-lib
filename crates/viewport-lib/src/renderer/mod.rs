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
mod paths;
pub use capture::{CapturedHdr, CapturedHdrGpu};
pub use paths::{OwnedPath, PassPath, PassView};
mod gpu_context;
pub use gpu_context::GpuContext;
mod picking;
pub use picking::sub_object;
pub use picking::{
    CellSelectionInfo, GpuPickHit, PickBackend, PickHit, PickMask, PickPoll, PickRectResult,
    PolylineSelectionInfo, SubObjectRef, SubSelection, SubSelectionRef, VolumeSelectionInfo,
};
mod capture;
mod overlay_draw_order;
mod readback;
pub use readback::ExposureReadback;
// Gaussian splat upload vocabulary lives in `resources`; re-exported here so the
// public `renderer::GaussianSplat*` path and its doc links stay stable.
pub use crate::resources::{GaussianSplatData, GaussianSplatId, ShDegree};
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
mod lod_instance_tests;

pub use self::types::{
    AnchorX, AnchorY, AnimTrack, AtlasViewerCorner, AutoExposure, BloomSettings, BorderMode,
    CameraFrame, Candela, ClipObject, ClipShape, ComputeFilterItem, ComputeFilterKind,
    ContactShadowSettings, CylindricalFacing, DebugOutputMode, DebugQuantity, DebugVis,
    DecalAnimation, DecalBlendMode, DecalItem, DecalProjection, DisplaySettings, DofSettings,
    EdlSettings, EffectsFrame, EmitterConfig, EnvironmentSettings, ExposureMode, ExposureSettings,
    ExternalInstancesItem, FillRule, FilterMode, ForceField, ForegroundPass, ForegroundProjection,
    FrameData, GaussianSplatItem, GlyphItem, GlyphRunItem, GlyphSetRefItem, GlyphType,
    GpuImplicitItem, GpuMarchingCubesItem, GpuParticleSystemItem, GradientStop, GroundPlane,
    GroundPlaneMode, ImageSliceItem, IndirectLightSource, InteractionFrame, LabelAnchor,
    LabelAnchorY, LabelItem, LerpAnim, LicOverlay, LightKind, LightSource, LightingPosture,
    LightingSettings, LineCap, LineJoin, Lumen, Lux, MAX_POINT_SHADOW_LIGHTS, MeshInstanceItem,
    NineSlice, OVERLAY_MAX_GRADIENT_STOPS, OVERLAY_MAX_SHADOW_LAYERS, OverlayAnchor,
    OverlayAnimation, OverlayAnimations, OverlayEasing, OverlayFill, OverlayFrame,
    OverlayPolylineItem, OverlayShape, OverlayShapeItem, OverlayTextureId, POINT_SHADOW_FACE_SIZE,
    ParticleMeshAlign, PathSegment, PathTrack, PickId, PipelineMode, PointCloudItem,
    PointCloudRefItem, PointRenderMode, PointShadowMode, PolylineCap, PolylineItem,
    PolylineRefItem, PositionedGlyph, PostProcessSettings, RenderCamera, RepeatMode, RibbonItem,
    RibbonRefItem, ScatterQuality, ScatterSettings, ScatterVolumeItem, SceneEffects, SceneFrame,
    SceneRenderItem, ScreenImageItem, ShadowFilter, ShadowLayer, ShadowSettings, SliceAxis,
    SpawnShape, SpriteBlend, SpriteInstanceSetRefItem, SpriteItem, SpriteLitParams,
    SpriteNormalMode, SpriteOrientation, SpriteSetRefItem, SpriteSizeMode, StreamtubeItem,
    StreamtubeRefItem, StrokePattern, SubPath, SurfaceLICConfig, SurfaceSubmission,
    TensorGlyphItem, TensorGlyphSetRefItem, TextureTransform, TileMode, ToneMapping,
    TriangleDirection, TubeItem, TubeRefItem, VelocityDist, ViewportEffects, ViewportFrame,
    VolumeItem, VolumeMeshItem, VolumeSurfaceSliceItem, VolumeTransparency,
    aabb_wireframe_polyline, sphere_wireframe_polyline,
};

// Crate-internal anchor resolution helpers (viewport corner placement), used by
// the screen-image prepare/upload and picking paths outside `renderer::types`.
pub(crate) use self::types::viewport_anchored_ndc;

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
    ShadowAtlasUniform, SingleLightUniform, SplatOutlineMaskUniform,
};

/// Per-frame selection-outline state for one viewport, one entry per scene-item
/// kind, rebuilt in prepare(). Each kind either owns dedicated outline buffers or
/// records indices into that kind's `*_gpu_data`; the outline mask/edge passes and
/// the composite walk these. Grouped so `ViewportSlot` carries one field instead of
/// a dozen parallel ones.
#[derive(Default)]
pub(crate) struct SelectionOutlines {
    /// Per-frame outline buffers for selected objects.
    pub outline_object_buffers: Vec<OutlineObjectBuffers>,
    /// Per-frame outline buffers for selected Gaussian splat sets.
    pub splat_outline_buffers: Vec<crate::resources::SplatOutlineBuffers>,
    /// Indices into `volume_gpu_data` for selected volumes.
    pub volume_outline_indices: Vec<usize>,
    /// Indices into `glyph_gpu_data` for selected glyph sets. Each entry is
    /// (gpu_data_index, instance_filter): None draws all instances, Some(indices)
    /// draws only those specific instance indices.
    pub glyph_outline_indices: Vec<(usize, Option<Vec<u32>>)>,
    /// Indices into `tensor_glyph_gpu_data` for selected tensor glyph sets.
    pub tensor_glyph_outline_indices: Vec<(usize, Option<Vec<u32>>)>,
    /// Indices into `sprite_gpu_data` for selected sprite sets.
    pub sprite_outline_indices: Vec<(usize, Option<Vec<u32>>)>,
    /// Per-frame inline quad outline buffers for selected image slices.
    pub raw_geom_outline_buffers: Vec<crate::resources::RawGeomOutlineBuffers>,
    /// Per-frame NDC rect outline buffers for selected screen images.
    pub screen_rect_outline_buffers: Vec<crate::resources::ScreenRectOutlineBuffers>,
    /// Indices into `implicit_gpu_data` for selected GPU implicit items.
    pub implicit_outline_indices: Vec<usize>,
    /// Per-frame outline data for selected GPU marching cubes items.
    pub mc_outline_data: Vec<crate::resources::volume::gpu_marching_cubes::McOutlineItem>,
    /// Outline items for selected streamtubes (index into streamtube_gpu_data + mask bind group).
    pub streamtube_outline_items: Vec<crate::resources::CurveMeshOutlineItem>,
    /// Outline items for selected tubes.
    pub tube_outline_items: Vec<crate::resources::CurveMeshOutlineItem>,
    /// Outline items for selected ribbons.
    pub ribbon_outline_items: Vec<crate::resources::CurveMeshOutlineItem>,
    /// Indices into polyline_gpu_data for selected user polylines.
    pub polyline_outline_indices: Vec<usize>,
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
    /// Per-fragment debug storage buffer (group 0 binding 12). Allocated at
    /// `width * height * 16` bytes when debug_vis is active; None otherwise.
    pub debug_frag_buf: Option<crate::gpu::Buffer>,
    /// Viewport dimensions for which `debug_frag_buf` was allocated.
    pub debug_frag_dims: (u32, u32),

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

/// Retained pick state for one GPU implicit surface, built during `prepare()`.
struct GpuImplicitPickItem {
    id: u64,
    primitives: Vec<crate::resources::ImplicitPrimitive>,
    blend_mode: crate::resources::ImplicitBlendMode,
    max_steps: u32,
    step_scale: f32,
    hit_threshold: f32,
    max_distance: f32,
}

/// Retained pick state for one GPU marching cubes item, built during `prepare()`.
struct GpuMcPickItem {
    id: u64,
    isovalue: f32,
    volume_data: std::sync::Arc<crate::geometry::marching_cubes::VolumeData>,
}

/// Renderer wrapping all GPU resources and providing `prepare()` and `paint()` methods.
/// Per-viewport scene-colour resolve sampled by the refractive sprite pass.
///
/// Lazily allocated on the first frame containing a refractive sprite; resized
/// whenever the HDR target dimensions change. The bind group is rebuilt with
/// the resolve when either changes.
struct SpriteRefractionResolve {
    texture: crate::gpu::Texture,
    view: crate::gpu::TextureView,
    size: [u32; 2],
}

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
/// Number of measured GPU passes; the query set holds `2 * GPU_TS_SLOTS` entries
/// (a begin/end pair per slot).
pub(crate) const GPU_TS_SLOTS: u32 = 10;

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
    item_type_plugins:
        std::collections::HashMap<&'static str, Box<dyn crate::plugin_api::ItemTypePlugin>>,
    /// Monotonic frame counter passed to plugin contexts.
    plugin_frame_index: u64,
    /// Performance counters from the last frame.
    last_stats: crate::renderer::stats::FrameStats,
    /// Per-frame point cloud GPU data, rebuilt in prepare(), consumed in paint().
    point_cloud_gpu_data: Vec<crate::resources::PointCloudGpuData>,
    /// Per-frame glyph GPU data, rebuilt in prepare(), consumed in paint().
    glyph_gpu_data: Vec<crate::resources::GlyphGpuData>,
    /// Per-frame tensor glyph GPU data, rebuilt in prepare(), consumed in paint().
    tensor_glyph_gpu_data: Vec<crate::resources::TensorGlyphGpuData>,
    /// Per-frame polyline GPU data, rebuilt in prepare(), consumed in paint().
    polyline_gpu_data: Vec<crate::resources::PolylineGpuData>,
    /// Per-frame volume GPU data, rebuilt in prepare(), consumed in paint().
    volume_gpu_data: Vec<crate::resources::VolumeGpuData>,
    /// Per-frame streamtube GPU data, rebuilt in prepare(), consumed in paint().
    streamtube_gpu_data: Vec<crate::resources::StreamtubeGpuData>,
    /// Per-frame general tube GPU data, rebuilt in prepare(), consumed in paint().
    tube_gpu_data: Vec<crate::resources::StreamtubeGpuData>,
    /// Per-frame ribbon GPU data, rebuilt in prepare(), consumed in paint().
    ribbon_gpu_data: Vec<crate::resources::StreamtubeGpuData>,
    /// Indices into streamtube_gpu_data for selected streamtubes (set in prepare_scene, consumed in prepare_viewport).
    streamtube_selected_gpu_indices: Vec<usize>,
    /// Indices into tube_gpu_data for selected tubes (set in prepare_scene, consumed in prepare_viewport).
    tube_selected_gpu_indices: Vec<usize>,
    /// Indices into ribbon_gpu_data for selected ribbons (set in prepare_scene, consumed in prepare_viewport).
    ribbon_selected_gpu_indices: Vec<usize>,
    /// Indices into polyline_gpu_data for selected user polylines (set in prepare_scene, consumed in prepare_viewport).
    polyline_selected_gpu_indices: Vec<usize>,
    /// Per-frame image slice GPU data, rebuilt in prepare(), consumed in paint().
    image_slice_gpu_data: Vec<crate::resources::ImageSliceGpuData>,
    /// Per-frame volume surface slice GPU data, rebuilt in prepare(), consumed in paint().
    volume_surface_slice_gpu_data: Vec<crate::resources::VolumeSurfaceSliceGpuData>,
    /// Per-frame Surface LIC GPU data, rebuilt in prepare(), consumed in paint().
    lic_gpu_data: Vec<crate::resources::LicSurfaceGpuData>,
    /// Per-frame GPU implicit surface data, rebuilt in prepare(), consumed in paint().
    implicit_gpu_data: Vec<crate::resources::volume::implicit::ImplicitGpuItem>,
    /// Per-frame decal draw list, rebuilt in prepare(), consumed in paint() (D1).
    /// Entries are cheap clones of cached GPU handles from `decal_cache`.
    decal_gpu_data: Vec<crate::resources::decal::DecalGpuItem>,
    /// Decal GPU resources cached across frames, keyed by decal content hash.
    /// Decals are static per submission, so this skips rebuilding a uniform
    /// buffer and bind group for each decal every frame. Entries not seen in a
    /// frame are evicted so removed decals do not leak.
    decal_cache: std::collections::HashMap<u64, crate::resources::decal::DecalGpuItem>,
    /// Per-frame decal exclude GPU data, rebuilt in prepare(), consumed in paint() (D5).
    decal_exclude_items: Vec<crate::resources::decal::DecalExcludeGpuItem>,
    /// Per-frame GPU marching cubes render data, rebuilt in prepare(), consumed in paint().
    mc_gpu_data: Vec<crate::resources::volume::gpu_marching_cubes::McFrameData>,
    /// Per-frame sprite GPU data, rebuilt in prepare(), consumed in paint().
    sprite_gpu_data: Vec<crate::resources::SpriteGpuData>,
    /// Per-frame mesh-instance batches, rebuilt in prepare(), consumed in paint().
    mesh_instance_gpu_data: Vec<crate::resources::MeshInstanceGpuData>,
    /// Per-frame GPU particle systems, dispatched in prepare(), consumed in paint().
    particle_gpu_data: Vec<crate::resources::gpu::gpu_particles::ParticleFrameData>,
    external_instances_gpu_data:
        Vec<crate::resources::gpu::external_instances::ExternalInstancesGpuData>,
    /// Scene-colour resolve textures for the refractive sprite pass, indexed
    /// alongside `viewport_slots`. Lazily allocated when the first refractive
    /// sprite appears for a viewport.
    sprite_refraction_resolves: Vec<Option<SpriteRefractionResolve>>,
    /// Per-frame Gaussian splat draw data, rebuilt in prepare_viewport_internal(), consumed in paint().
    gaussian_splat_draw_data: Vec<crate::resources::GaussianSplatDrawData>,
    /// Per-frame screen-image GPU data, rebuilt in prepare(), consumed in paint().
    screen_image_gpu_data: Vec<crate::resources::ScreenImageGpuData>,
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
    /// Current runtime mode controlling internal default behavior.
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
    /// Instant the renderer was constructed. Used as the t=0 reference for
    /// per-frame animated effects (e.g. `ScatterVolume::noise` time scrolling).
    start_instant: web_time::Instant,
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
    pick_point_cloud_items: Vec<PointCloudItem>,
    /// Gaussian splat items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_splat_items: Vec<GaussianSplatItem>,
    /// Volume items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_volume_items: Vec<VolumeItem>,
    /// Scatter volume items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_scatter_volume_items: Vec<crate::renderer::types::ScatterVolumeItem>,
    /// Volumes packed into the GPU storage buffer this frame
    /// (volume, density_multiplier, flag bits). Stored so `render_viewport`
    /// can re-upload as needed without re-walking the scene frame.
    pub(crate) prepared_scatter_volumes:
        Vec<(crate::scene::scatter_volume::ScatterVolume, f32, u32)>,
    /// Subset of the prepared scatter volumes that carry `RefractionParams`.
    /// Cleared and refilled each frame by `prepare_viewport`. The refraction
    /// pass walks this list; an empty list skips the pass entirely.
    pub(crate) prepared_refraction_volumes: Vec<(crate::scene::scatter_volume::ScatterVolume, f32)>,
    /// Per-viewport scatter intermediates and temporal history. Indexed by
    /// `vp_idx`. Grown lazily inside the scatter pass; each entry is
    /// reallocated when the requested scatter target size or downsample mode
    /// changes.
    pub(crate) scatter_viewport_states: Vec<Option<crate::resources::ScatterViewportState>>,
    /// Opaque volume mesh items from the last `prepare()` call, retained for cell-level `pick()` dispatch.
    pick_volume_mesh_items: Vec<VolumeMeshItem>,
    /// Polyline items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_polyline_items: Vec<PolylineItem>,
    /// Glyph items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_glyph_items: Vec<GlyphItem>,
    /// Tensor glyph items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_tensor_glyph_items: Vec<TensorGlyphItem>,
    /// Sprite items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_sprite_items: Vec<SpriteItem>,
    /// Streamtube items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_streamtube_items: Vec<StreamtubeItem>,
    /// Tube items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_tube_items: Vec<TubeItem>,
    /// Ribbon items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_ribbon_items: Vec<RibbonItem>,
    /// Image slice items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_image_slice_items: Vec<ImageSliceItem>,
    /// Volume surface slice items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_volume_surface_slice_items: Vec<VolumeSurfaceSliceItem>,
    /// Screen image items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_screen_image_items: Vec<ScreenImageItem>,
    /// Decal items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_decal_items: Vec<DecalItem>,
    /// GPU implicit surface items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_implicit_items: Vec<GpuImplicitPickItem>,
    /// GPU marching cubes items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_mc_items: Vec<GpuMcPickItem>,
    /// When `false`, `prepare()` skips populating the CPU pick caches above, so
    /// scenes that never call `pick()`/`pick_rect()` avoid a per-frame deep copy
    /// of all inline geometry. Enable with `set_cpu_pick_cache(true)`.
    cpu_pick_cache_enabled: bool,

    /// In-flight async GPU pick, if any. `pick_object_begin` submits the id pass
    /// and parks the staging buffers here; `pick_object_poll` reads them back
    /// without blocking on the GPU queue. `None` when no async pick is pending.
    pending_pick: Option<picking::PendingPick>,

    /// Shared unit-cube mesh (`[-0.5, 0.5]^3`) used as the GPU pick proxy for
    /// box-shaped items: decals (under each decal's `transform`) and box scatter
    /// volumes (under a translate+scale to the box). Uploaded lazily on first
    /// use, then reused. `None` until then.
    decal_pick_cube: Option<crate::resources::mesh::mesh_store::MeshId>,

    /// Shared unit-radius icosphere used as the GPU pick proxy for sphere scatter
    /// volumes, scaled to each volume's radius. Uploaded lazily on first use.
    /// `None` until then.
    scatter_pick_sphere: Option<crate::resources::mesh::mesh_store::MeshId>,

    // --- GPU timestamp queries ---
    /// Timestamp query set with `2 * GPU_TS_SLOTS` entries: a begin/end pair per
    /// measured pass (see the `GPU_TS_*` slot constants). `None` when
    /// `TIMESTAMP_QUERY` is unavailable or not yet initialized.
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
    ///   per-pass GPU breakdown.
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
    ///   either way the reconstruction is smooth, never blocky nearest-neighbor.
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
            crate::gpu::Features::PIPELINE_CACHE,
            crate::gpu::PRIMITIVE_INDEX_FEATURE,
            crate::gpu::Features::FLOAT32_FILTERABLE,
        ] {
            if adapter.features().contains(feature) {
                features |= feature;
            }
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
        Self {
            resources: DeviceResources::new_with_cache(
                device,
                target_format,
                sample_count,
                pipeline_cache_data,
            ),
            instancing: InstancingState::new(gpu_culling_supported, multi_draw_supported),
            item_type_plugins: std::collections::HashMap::new(),
            plugin_frame_index: 0,
            last_stats: crate::renderer::stats::FrameStats::default(),
            prepare_breakdown: crate::renderer::stats::PrepareBreakdown::default(),
            point_cloud_gpu_data: Vec::new(),
            glyph_gpu_data: Vec::new(),
            tensor_glyph_gpu_data: Vec::new(),
            polyline_gpu_data: Vec::new(),
            volume_gpu_data: Vec::new(),
            streamtube_gpu_data: Vec::new(),
            tube_gpu_data: Vec::new(),
            ribbon_gpu_data: Vec::new(),
            streamtube_selected_gpu_indices: Vec::new(),
            tube_selected_gpu_indices: Vec::new(),
            ribbon_selected_gpu_indices: Vec::new(),
            polyline_selected_gpu_indices: Vec::new(),
            image_slice_gpu_data: Vec::new(),
            volume_surface_slice_gpu_data: Vec::new(),
            sprite_gpu_data: Vec::new(),
            mesh_instance_gpu_data: Vec::new(),
            particle_gpu_data: Vec::new(),
            external_instances_gpu_data: Vec::new(),
            sprite_refraction_resolves: Vec::new(),
            gaussian_splat_draw_data: Vec::new(),
            lic_gpu_data: Vec::new(),
            implicit_gpu_data: Vec::new(),
            decal_gpu_data: Vec::new(),
            decal_cache: std::collections::HashMap::new(),
            decal_exclude_items: Vec::new(),
            mc_gpu_data: Vec::new(),
            screen_image_gpu_data: Vec::new(),
            label_gpu_data: None,
            overlay_shape_gpu_data: None,
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
            start_instant: web_time::Instant::now(),
            last_prepare_instant: None,
            frame_counter: 0,
            lod_levels: std::collections::HashMap::new(),
            mesh_instance_lod_levels: std::collections::HashMap::new(),
            pick_scene_items: Vec::new(),
            pick_bvh: std::sync::Mutex::new(None),
            pick_bvh_identity_rev: 0,
            pick_bvh_transform_rev: 0,
            pick_point_cloud_items: Vec::new(),
            pick_splat_items: Vec::new(),
            pick_volume_items: Vec::new(),
            pick_scatter_volume_items: Vec::new(),
            prepared_scatter_volumes: Vec::new(),
            prepared_refraction_volumes: Vec::new(),
            scatter_viewport_states: Vec::new(),
            pick_volume_mesh_items: Vec::new(),
            pick_polyline_items: Vec::new(),
            pick_glyph_items: Vec::new(),
            pick_tensor_glyph_items: Vec::new(),
            pick_sprite_items: Vec::new(),
            pick_streamtube_items: Vec::new(),
            pick_tube_items: Vec::new(),
            pick_ribbon_items: Vec::new(),
            pick_image_slice_items: Vec::new(),
            pick_volume_surface_slice_items: Vec::new(),
            pick_screen_image_items: Vec::new(),
            pick_decal_items: Vec::new(),
            pick_implicit_items: Vec::new(),
            pick_mc_items: Vec::new(),
            cpu_pick_cache_enabled: false,
            pending_pick: None,
            decal_pick_cube: None,
            scatter_pick_sphere: None,
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
        }
    }

    /// Access the underlying GPU resources (e.g. for mesh uploads).
    pub fn resources(&self) -> &DeviceResources {
        &self.resources
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
            let data = slice.get_mapped_range();
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

    /// Set the runtime mode controlling internal default behavior.
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
    /// performance/behavior knobs (culling, occlusion, adaptive quality, render
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
            _pad: 0,
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
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return Vec::new();
        }
        // A derivative render (capture / bake) reads resident state: it must not
        // run a plugin's `&mut self` prepare, which would advance the plugin's
        // own per-frame state, nor bump the plugin frame index. The draw passes
        // are skipped in the same way (see `dispatch_plugin_paint`), so plugin
        // geometry is absent from a bake rather than drawn stale.
        if !self.render_advances_state() {
            return Vec::new();
        }
        self.plugin_frame_index = self.plugin_frame_index.wrapping_add(1);
        let mut bufs: Vec<crate::gpu::CommandBuffer> = Vec::new();
        for (name, plugin) in self.item_type_plugins.iter_mut() {
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                // Constructed per plugin because `Jobs` borrows `&resources`
                // and the borrow only needs to live for this iteration.
                let ctx = crate::plugin_api::ItemFrameContext {
                    camera: &frame.camera.render_camera,
                    viewport_size: glam::Vec2::from(frame.camera.viewport_size),
                    viewport_index: frame.camera.viewport_index,
                    frame_index: self.plugin_frame_index,
                    jobs: crate::resources::Jobs::new(&self.resources),
                };
                bufs.extend(plugin.prepare(device, queue, &ctx, items.as_ref()));
            }
        }
        bufs
    }

    /// Walk registered item-type plugins and invoke `paint` for each one
    /// that has a matching collection submitted on `frame.scene`.
    ///
    /// Called from inside the lib's HDR scene pass between built-in
    /// opaques and the skybox.
    pub(crate) fn dispatch_plugin_paint<'rp>(
        &'rp self,
        pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &'rp FrameData,
    ) {
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return;
        }
        // A derivative render (capture / bake) does not draw plugin items. Their
        // draw passes read visibility / LOD that `cull` sets against the
        // presented camera (skipped here), so painting them would render stale,
        // wrong-camera geometry into the bake. Plugin geometry is absent from
        // bakes rather than partially present.
        if !self.render_advances_state() {
            return;
        }
        let ctx = crate::plugin_api::PaintContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                plugin.paint(pass, &ctx, items.as_ref());
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
                frame
                    .scene
                    .plugin_items
                    .get(*name)
                    .is_some_and(|items| !items.is_empty())
            })
    }

    /// `true` when the foreground pass has work this frame: submitted
    /// foreground items, or a registered foreground-drawing plugin with a
    /// non-empty collection.
    pub(crate) fn foreground_active(&self, frame: &FrameData) -> bool {
        !frame.scene.foreground_items.is_empty()
            || self.item_type_plugins.iter().any(|(name, plugin)| {
                plugin.draws_foreground()
                    && frame
                        .scene
                        .plugin_items
                        .get(*name)
                        .is_some_and(|items| !items.is_empty())
            })
    }

    /// Walk registered plugins and invoke `paint_foreground` for each
    /// foreground-drawing plugin whose collection is on `frame.scene`.
    ///
    /// Called from inside the foreground pass after the built-in item
    /// draws. `camera` carries the foreground projection so plugin-side
    /// math agrees with the bound group-0 camera.
    pub(crate) fn dispatch_plugin_paint_foreground<'rp>(
        &'rp self,
        pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &'rp FrameData,
        camera: &'rp RenderCamera,
    ) {
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return;
        }
        // See `dispatch_plugin_paint`: a derivative render draws no plugin items.
        if !self.render_advances_state() {
            return;
        }
        let ctx = crate::plugin_api::PaintContext {
            camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if !plugin.draws_foreground() {
                continue;
            }
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                plugin.paint_foreground(pass, &ctx, items.as_ref());
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
                    && frame
                        .scene
                        .plugin_items
                        .get(*name)
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
    pub(crate) fn dispatch_plugin_paint_depth_read<'rp>(
        &'rp self,
        pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &'rp FrameData,
        scene_depth: &'rp crate::gpu::TextureView,
        scene_depth_sampler: &'rp crate::gpu::Sampler,
        scene_depth_bind_group: &'rp crate::gpu::BindGroup,
    ) {
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return;
        }
        // See `dispatch_plugin_paint`: a derivative render draws no plugin items.
        if !self.render_advances_state() {
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
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                plugin.paint_depth_read(pass, &ctx, items.as_ref());
            }
        }
    }

    /// Walk registered plugins and invoke `render_pick` for each one whose
    /// collection is on `frame.scene`.
    ///
    /// Called from inside the GPU pick pass after the built-in draws. The
    /// caller has bound the shared group-0 camera bind group; plugins that
    /// rebind group 0 must restore it.
    pub(crate) fn dispatch_plugin_pick<'rp>(
        &'rp self,
        pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &'rp FrameData,
        mask: crate::renderer::picking::PickMask,
    ) {
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return;
        }
        let ctx = crate::plugin_api::PickPassContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
            mask,
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                plugin.render_pick(pass, &ctx, items.as_ref());
            }
        }
    }

    /// Walk registered plugins and invoke `paint_transparent` for each
    /// one whose collection is on `frame.scene`.
    ///
    /// Called from inside the lib's OIT render pass, after built-in
    /// transparent draws.
    pub(crate) fn dispatch_plugin_paint_transparent<'rp>(
        &'rp self,
        pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &'rp FrameData,
    ) {
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return;
        }
        // See `dispatch_plugin_paint`: a derivative render draws no plugin items.
        if !self.render_advances_state() {
            return;
        }
        let ctx = crate::plugin_api::PaintContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                plugin.paint_transparent(pass, &ctx, items.as_ref());
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
    pub(crate) fn dispatch_plugin_shadow<'rp>(
        &'rp self,
        pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &'rp FrameData,
        cascade_idx: u32,
        light_view_proj: glam::Mat4,
    ) {
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return;
        }
        // See `dispatch_plugin_paint`: a derivative render draws no plugin items.
        if !self.render_advances_state() {
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
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                plugin.cast_shadow_pass(pass, &ctx, items.as_ref());
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
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return;
        }
        // Skip during a derivative render: `cull` is `&mut self` and would
        // advance the plugin's own per-frame visibility state. See
        // `dispatch_plugin_prepare`.
        if !self.render_advances_state() {
            return;
        }
        for (name, plugin) in self.item_type_plugins.iter_mut() {
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                let ctx = crate::plugin_api::ItemFrameContext {
                    camera: &frame.camera.render_camera,
                    viewport_size: glam::Vec2::from(frame.camera.viewport_size),
                    viewport_index: frame.camera.viewport_index,
                    frame_index: self.plugin_frame_index,
                    jobs: crate::resources::Jobs::new(&self.resources),
                };
                plugin.cull(frustum, &ctx, items.as_ref());
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
            frame.scene.plugin_items.get(*name).is_some_and(|items| {
                (0..items.len()).any(|i| {
                    let s = items.item_settings(i);
                    s.selected && !s.hidden
                })
            })
        })
    }

    pub(crate) fn dispatch_plugin_outline_mask<'rp>(
        &'rp self,
        pass: &mut crate::gpu::RenderPass<'rp>,
        frame: &'rp FrameData,
    ) {
        if self.item_type_plugins.is_empty() || frame.scene.plugin_items.is_empty() {
            return;
        }
        // See `dispatch_plugin_paint`: a derivative render draws no plugin items.
        if !self.render_advances_state() {
            return;
        }
        let ctx = crate::plugin_api::OutlineMaskContext {
            camera: &frame.camera.render_camera,
            viewport_size: glam::Vec2::from(frame.camera.viewport_size),
            viewport_index: frame.camera.viewport_index,
            frame_index: self.plugin_frame_index,
        };
        for (name, plugin) in self.item_type_plugins.iter() {
            if let Some(items) = frame.scene.plugin_items.get(*name) {
                plugin.outline_mask(pass, &ctx, items.as_ref());
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

    /// Read the debug values at a specific pixel from the per-fragment storage buffer.
    ///
    /// Returns `None` when debug_vis is inactive (no buffer allocated) or when `(x, y)`
    /// is outside the viewport. The four channels correspond to the current R/G/B channel
    /// selectors plus 1.0 for alpha.
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
        let buf = slot.debug_frag_buf.as_ref()?;
        let (vw, vh) = slot.debug_frag_dims;
        if x >= vw || y >= vh {
            return None;
        }
        let byte_offset = ((y as u64) * (vw as u64) + (x as u64)) * 16;
        let staging = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: crate::gpu::BufferUsages::MAP_READ | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buf, byte_offset, &staging, 0, 16);
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
        let data = slice.get_mapped_range();
        Some(bytemuck::pod_read_unaligned::<[f32; 4]>(&data))
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
        self.resources.upload_gaussian_splat(device, queue, data)
    }

    /// Remove an uploaded Gaussian splat set by handle.
    ///
    /// After this call the `id` is invalid and must not be submitted in `SceneFrame`.
    pub fn free_gaussian_splat(&mut self, id: GaussianSplatId) {
        self.resources.free_gaussian_splat(id);
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

    /// Start an asynchronous marching-cubes-ready volume upload. See
    /// [`DeviceResources::begin_upload_volume_for_mc`].
    pub fn begin_upload_volume_for_mc(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vol: crate::geometry::marching_cubes::VolumeData,
    ) -> crate::resources::JobId {
        self.resources
            .begin_upload_volume_for_mc(device, queue, vol)
    }

    /// Take the [`McVolumeId`](crate::resources::McVolumeId) produced by a
    /// completed [`begin_upload_volume_for_mc`](Self::begin_upload_volume_for_mc) job.
    pub fn upload_result_volume_mc(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::McVolumeId> {
        self.resources.upload_result_volume_mc(id)
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

    /// Start an asynchronous Gaussian splat upload. See
    /// [`DeviceResources::begin_upload_gaussian_splat`].
    pub fn begin_upload_gaussian_splat(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        data: crate::renderer::GaussianSplatData,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.resources
            .begin_upload_gaussian_splat(device, queue, data)
    }

    /// Take the [`GaussianSplatId`](crate::renderer::GaussianSplatId) produced by a
    /// completed [`begin_upload_gaussian_splat`](Self::begin_upload_gaussian_splat) job.
    pub fn upload_result_gaussian_splat(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::renderer::GaussianSplatId> {
        self.resources.upload_result_gaussian_splat(id)
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

    /// Start an asynchronous albedo texture upload. See
    /// [`DeviceResources::begin_upload_texture`] for the semantics.
    pub fn begin_upload_texture(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.resources
            .begin_upload_texture(device, queue, width, height, rgba)
    }

    /// Start an asynchronous normal-map upload. See
    /// [`DeviceResources::begin_upload_normal_map`] for the semantics.
    pub fn begin_upload_normal_map(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    ) -> crate::error::ViewportResult<crate::resources::JobId> {
        self.resources
            .begin_upload_normal_map(device, queue, width, height, rgba)
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
            &self.resources.binds.debug_frag_sentinel_buf,
            "camera_bind_group",
        );

        for slot in &mut self.viewport_slots {
            let dbg_buf = slot
                .debug_frag_buf
                .as_ref()
                .unwrap_or(&self.resources.binds.debug_frag_sentinel_buf);
            slot.camera_bind_group = self.resources.create_camera_bind_group(
                device,
                &slot.camera_buf,
                &slot.clip_planes_buf,
                &slot.shadow_info_buf,
                &slot.clip_volume_buf,
                dbg_buf,
                "per_viewport_camera_bg",
            );
            slot.foreground_camera_bind_group = self.resources.create_camera_bind_group(
                device,
                &slot.foreground_camera_buf,
                &slot.foreground_clip_planes_buf,
                &slot.shadow_info_buf,
                &slot.foreground_clip_volume_buf,
                dbg_buf,
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
                &self.resources.binds.debug_frag_sentinel_buf,
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
                &self.resources.binds.debug_frag_sentinel_buf,
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
                debug_frag_buf: None,
                debug_frag_dims: (0, 0),
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
        emit_scivis_draw_calls!(
            &self.resources,
            &mut *render_pass,
            &self.point_cloud_gpu_data,
            &self.glyph_gpu_data,
            &self.polyline_gpu_data,
            &self.volume_gpu_data,
            &self.streamtube_gpu_data,
            camera_bg,
            &self.tube_gpu_data,
            &self.image_slice_gpu_data,
            &self.tensor_glyph_gpu_data,
            &self.ribbon_gpu_data,
            &self.volume_surface_slice_gpu_data,
            &self.sprite_gpu_data,
            &self.mesh_instance_gpu_data,
            false
        );
        // Gaussian splats (alpha-blended, back-to-front sorted, no depth write).
        if !self.gaussian_splat_draw_data.is_empty() {
            if let Some(ref dual) = self.resources.gaussian_splat.pipeline {
                render_pass.set_pipeline(dual.for_format(false));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for dd in &self.gaussian_splat_draw_data {
                    if dd.wireframe {
                        continue;
                    }
                    if let Some(set) = self
                        .resources
                        .content
                        .gaussian_splat_store
                        .get_by_index(dd.store_index)
                    {
                        if let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) {
                            render_pass.set_bind_group(1, &vp_sort.render_bg, &[]);
                            render_pass.draw(0..6, 0..dd.count);
                        }
                    }
                }
            }
        }
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
        }
    }
}
