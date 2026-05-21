//! `ViewportRenderer` : the main entry point for the viewport library.
//!
//! Wraps [`ViewportGpuResources`] and provides `prepare()` / `paint()` methods
//! that take raw `wgpu` types. GUI framework adapters (e.g. the egui
//! `CallbackTrait` impl in the application crate) delegate to these methods.

#[macro_use]
mod types;
mod indirect;
mod picking;
pub use picking::PickRectResult;
mod prepare;
mod render;
pub mod shader_hashes;
mod shadows;
pub mod stats;

pub use self::types::{
    CameraFrame, CameraFrustumItem, ClipObject, ClipShape, ComputeFilterItem, ComputeFilterKind,
    EffectsFrame, EnvironmentMap, FilterMode, FrameData, GaussianSplatData, GaussianSplatId,
    GaussianSplatItem, GlyphItem, GlyphType, GroundPlane, GroundPlaneMode, ImageAnchor,
    ImageSliceItem, InteractionFrame, LabelAnchor, LabelItem, LightKind, LightSource,
    LightingSettings, LoadingBarAnchor, LoadingBarItem, OverlayFrame, OverlayImageItem,
    OverlayRectItem, OverlayShape, OverlayShapeItem, PickId, TriangleDirection,
    PointCloudItem, PointRenderMode, PolylineItem, PostProcessSettings, RenderCamera, RibbonItem,
    RulerItem, ScalarBarAnchor, ScalarBarItem, ScalarBarOrientation, SceneEffects, SceneFrame,
    SceneRenderItem, ScreenImageItem, ShDegree, ShadowFilter, SliceAxis, SpriteItem,
    SpriteSizeMode, StreamtubeItem, SurfaceLICConfig, SurfaceLICItem, SurfaceSubmission,
    TensorGlyphItem, ToneMapping, TransparentVolumeMeshItem, TubeItem, ViewportEffects,
    ViewportFrame, VolumeItem, VolumeMeshItem, VolumeSurfaceSliceItem, aabb_wireframe_polyline,
};

/// An opaque handle to a per-viewport GPU state slot.
///
/// Obtained from [`ViewportRenderer::create_viewport`] and passed to
/// [`ViewportRenderer::prepare_viewport`], [`ViewportRenderer::paint_viewport`],
/// and [`ViewportRenderer::render_viewport`].
///
/// The inner `usize` is the slot index and doubles as the value for
/// [`CameraFrame::with_viewport_index`].  Single-viewport applications that use
/// the legacy [`ViewportRenderer::prepare`] / [`ViewportRenderer::paint`] API do
/// not need this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewportId(pub usize);

use self::shadows::{compute_cascade_matrix, compute_cascade_splits};
use self::types::{INSTANCING_THRESHOLD, InstancedBatch};
use crate::resources::{
    BatchMeta, CLIP_VOLUME_MAX, CameraUniform, ClipPlanesUniform, ClipVolumeEntry,
    ClipVolumesUniform, GridUniform, InstanceAabb, InstanceData, LightsUniform, ObjectUniform,
    OutlineEdgeUniform, OutlineObjectBuffers, OutlineUniform, PickInstance, ShadowAtlasUniform,
    SingleLightUniform, SplatOutlineMaskUniform, ViewportGpuResources,
};

/// Per-viewport GPU state: uniform buffers and bind groups that differ per viewport.
///
/// Each viewport slot owns its own camera, clip planes, clip volume, shadow info,
/// and grid buffers, plus the bind groups that reference them. Scene-global
/// resources (lights, shadow atlas texture, IBL) are shared via the bind group
/// pointing to buffers on `ViewportGpuResources`.
pub(crate) struct ViewportSlot {
    pub camera_buf: wgpu::Buffer,
    pub clip_planes_buf: wgpu::Buffer,
    pub clip_volume_buf: wgpu::Buffer,
    pub shadow_info_buf: wgpu::Buffer,
    pub grid_buf: wgpu::Buffer,
    /// Camera bind group (group 0) referencing this slot's per-viewport buffers
    /// plus shared scene-global resources.
    pub camera_bind_group: wgpu::BindGroup,
    /// Grid bind group (group 0 for grid pipeline) referencing this slot's grid buffer.
    pub grid_bind_group: wgpu::BindGroup,
    /// Per-viewport HDR post-process render targets.
    ///
    /// Created lazily on first HDR render call and resized when viewport dimensions change.
    pub hdr: Option<crate::resources::ViewportHdrState>,

    // --- Per-viewport interaction state (Phase 4) ---
    /// Per-frame outline buffers for selected objects, rebuilt in prepare().
    pub outline_object_buffers: Vec<OutlineObjectBuffers>,
    /// Per-frame outline buffers for selected Gaussian splat sets, rebuilt in prepare().
    pub splat_outline_buffers: Vec<crate::resources::SplatOutlineBuffers>,
    /// Indices into `volume_gpu_data` for selected volumes, rebuilt in prepare().
    pub volume_outline_indices: Vec<usize>,
    /// Indices into `glyph_gpu_data` for selected glyph sets, rebuilt in prepare().
    /// Each entry is (gpu_data_index, instance_filter): None draws all instances,
    /// Some(indices) draws only those specific instance indices.
    pub glyph_outline_indices: Vec<(usize, Option<Vec<u32>>)>,
    /// Indices into `tensor_glyph_gpu_data` for selected tensor glyph sets, rebuilt in prepare().
    pub tensor_glyph_outline_indices: Vec<(usize, Option<Vec<u32>>)>,
    /// Indices into `sprite_gpu_data` for selected sprite sets, rebuilt in prepare().
    pub sprite_outline_indices: Vec<(usize, Option<Vec<u32>>)>,
    /// Per-frame inline quad outline buffers for selected image slices, rebuilt in prepare().
    pub raw_geom_outline_buffers: Vec<crate::resources::RawGeomOutlineBuffers>,
    /// Per-frame NDC rect outline buffers for selected screen images, rebuilt in prepare().
    pub screen_rect_outline_buffers: Vec<crate::resources::ScreenRectOutlineBuffers>,
    /// Indices into `implicit_gpu_data` for selected GPU implicit items, rebuilt in prepare().
    pub implicit_outline_indices: Vec<usize>,
    /// Per-frame outline data for selected GPU marching cubes jobs, rebuilt in prepare().
    pub mc_outline_data: Vec<crate::resources::gpu_marching_cubes::McOutlineItem>,
    /// Outline items for selected streamtubes (index into streamtube_gpu_data + mask bind group).
    pub streamtube_outline_items: Vec<crate::resources::CurveMeshOutlineItem>,
    /// Outline items for selected tubes.
    pub tube_outline_items: Vec<crate::resources::CurveMeshOutlineItem>,
    /// Outline items for selected ribbons.
    pub ribbon_outline_items: Vec<crate::resources::CurveMeshOutlineItem>,
    /// Indices into polyline_gpu_data for selected user polylines.
    pub polyline_outline_indices: Vec<usize>,
    /// Per-frame x-ray buffers for selected objects, rebuilt in prepare().
    pub xray_object_buffers: Vec<(
        crate::resources::mesh_store::MeshId,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Per-frame constraint guide line buffers, rebuilt in prepare().
    pub constraint_line_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Per-frame cap geometry buffers (section view cross-section fill), rebuilt in prepare().
    pub cap_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Per-frame clip plane fill overlay buffers, rebuilt in prepare().
    pub clip_plane_fill_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Per-frame clip plane line overlay buffers, rebuilt in prepare().
    pub clip_plane_line_buffers: Vec<(
        wgpu::Buffer,
        wgpu::Buffer,
        u32,
        wgpu::Buffer,
        wgpu::BindGroup,
    )>,
    /// Vertex buffer for axes indicator geometry (rebuilt each frame).
    pub axes_vertex_buffer: wgpu::Buffer,
    /// Number of vertices in the axes indicator buffer.
    pub axes_vertex_count: u32,
    /// Gizmo model-matrix uniform buffer.
    pub gizmo_uniform_buf: wgpu::Buffer,
    /// Gizmo bind group (group 1: model matrix uniform).
    pub gizmo_bind_group: wgpu::BindGroup,
    /// Gizmo vertex buffer.
    pub gizmo_vertex_buffer: wgpu::Buffer,
    /// Gizmo index buffer.
    pub gizmo_index_buffer: wgpu::Buffer,
    /// Number of indices in the current gizmo mesh.
    pub gizmo_index_count: u32,

    // --- Sub-object highlight (per-viewport, generation-cached) ---
    /// Per-viewport dynamic resolution intermediate render target.
    /// `None` when render_scale == 1.0 or not yet initialised.
    pub dyn_res: Option<crate::resources::dyn_res::DynResTarget>,
    /// Per-viewport intermediate render target for the HDR eframe callback path.
    /// `None` until the first `prepare_hdr_callback` call for this viewport.
    pub hdr_callback: Option<crate::resources::dyn_res::HdrCallbackTarget>,
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

/// Retained pick state for one GPU marching cubes job, built during `prepare()`.
struct GpuMcPickItem {
    id: u64,
    isovalue: f32,
    volume_data: std::sync::Arc<crate::geometry::marching_cubes::VolumeData>,
}

/// Renderer wrapping all GPU resources and providing `prepare()` and `paint()` methods.
pub struct ViewportRenderer {
    resources: ViewportGpuResources,
    /// Instanced batches prepared for the current frame. Empty when using per-object path.
    instanced_batches: Vec<InstancedBatch>,
    /// Whether the current frame uses the instanced draw path.
    use_instancing: bool,
    /// True when the device supports `INDIRECT_FIRST_INSTANCE`.
    gpu_culling_supported: bool,
    /// True when GPU-driven culling is active (supported and not disabled by the caller).
    gpu_culling_enabled: bool,
    /// GPU culling compute pipelines and frustum buffer. Created lazily on the first
    /// frame where `gpu_culling_enabled` is true and instance buffers are present.
    cull_resources: Option<indirect::CullResources>,
    /// Performance counters from the last frame.
    last_stats: crate::renderer::stats::FrameStats,
    /// Last scene generation seen during prepare(). u64::MAX forces rebuild on first frame.
    last_scene_generation: u64,
    /// Last selection generation seen during prepare(). u64::MAX forces rebuild on first frame.
    last_selection_generation: u64,
    /// Last scene_items count seen during prepare(). usize::MAX forces rebuild on first frame.
    /// Included in cache key so that frustum-culling changes (different visible set, different
    /// count) correctly invalidate the instance buffer even when scene_generation is stable.
    last_scene_items_count: usize,
    /// Count of items that passed the instanced-path filter on the last rebuild.
    /// Used in place of has_per_frame_mutations so scenes that mix instanced and
    /// non-instanced items (e.g. one two-sided mesh + 10k static boxes) still hit
    /// the instanced batch cache on frames where the filtered set is unchanged.
    last_instancable_count: usize,
    /// Total instance count from the last rebuild. Used as a fast length check
    /// in `structure_preserved` and as `instance_count` for GPU cull dispatches.
    cached_instance_count: usize,
    /// Per-batch content hash from the last rebuild, indexed by batch position.
    /// A hash mismatch triggers a `write_buffer` for that batch; a match skips it.
    cached_instance_hashes: Vec<u64>,
    /// Cached instanced batch descriptors from last rebuild.
    cached_instanced_batches: Vec<InstancedBatch>,
    /// When true, the next cache-miss forces a full buffer upload instead of the
    /// per-batch partial-upload path. Set by `force_dirty()` and consumed once.
    force_full_upload: bool,
    /// Per-frame point cloud GPU data, rebuilt in prepare(), consumed in paint().
    point_cloud_gpu_data: Vec<crate::resources::PointCloudGpuData>,
    /// Per-frame glyph GPU data, rebuilt in prepare(), consumed in paint().
    glyph_gpu_data: Vec<crate::resources::GlyphGpuData>,
    /// Per-frame tensor glyph GPU data, rebuilt in prepare(), consumed in paint() (Phase 5).
    tensor_glyph_gpu_data: Vec<crate::resources::TensorGlyphGpuData>,
    /// Per-frame polyline GPU data, rebuilt in prepare(), consumed in paint().
    polyline_gpu_data: Vec<crate::resources::PolylineGpuData>,
    /// Per-frame volume GPU data, rebuilt in prepare(), consumed in paint().
    volume_gpu_data: Vec<crate::resources::VolumeGpuData>,
    /// Per-frame streamtube GPU data, rebuilt in prepare(), consumed in paint().
    streamtube_gpu_data: Vec<crate::resources::StreamtubeGpuData>,
    /// Per-frame general tube GPU data, rebuilt in prepare(), consumed in paint() (Phase 3).
    tube_gpu_data: Vec<crate::resources::StreamtubeGpuData>,
    /// Per-frame ribbon GPU data, rebuilt in prepare(), consumed in paint() (Phase 8.1).
    ribbon_gpu_data: Vec<crate::resources::StreamtubeGpuData>,
    /// Indices into streamtube_gpu_data for selected streamtubes (set in prepare_scene, consumed in prepare_viewport).
    streamtube_selected_gpu_indices: Vec<usize>,
    /// Indices into tube_gpu_data for selected tubes (set in prepare_scene, consumed in prepare_viewport).
    tube_selected_gpu_indices: Vec<usize>,
    /// Indices into ribbon_gpu_data for selected ribbons (set in prepare_scene, consumed in prepare_viewport).
    ribbon_selected_gpu_indices: Vec<usize>,
    /// Indices into polyline_gpu_data for selected user polylines (set in prepare_scene, consumed in prepare_viewport).
    polyline_selected_gpu_indices: Vec<usize>,
    /// Per-frame image slice GPU data, rebuilt in prepare(), consumed in paint() (Phase 3).
    image_slice_gpu_data: Vec<crate::resources::ImageSliceGpuData>,
    /// Per-frame volume surface slice GPU data, rebuilt in prepare(), consumed in paint() (Phase 10).
    volume_surface_slice_gpu_data: Vec<crate::resources::VolumeSurfaceSliceGpuData>,
    /// Per-frame Surface LIC GPU data, rebuilt in prepare(), consumed in paint() (Phase 4).
    lic_gpu_data: Vec<crate::resources::LicSurfaceGpuData>,
    /// Per-frame GPU implicit surface data, rebuilt in prepare(), consumed in paint() (Phase 16).
    implicit_gpu_data: Vec<crate::resources::implicit::ImplicitGpuItem>,
    /// Per-frame GPU marching cubes render data, rebuilt in prepare(), consumed in paint() (Phase 17).
    mc_gpu_data: Vec<crate::resources::gpu_marching_cubes::McFrameData>,
    /// Per-frame sprite GPU data, rebuilt in prepare(), consumed in paint().
    sprite_gpu_data: Vec<crate::resources::SpriteGpuData>,
    /// Per-frame Gaussian splat draw data, rebuilt in prepare_viewport_internal(), consumed in paint().
    gaussian_splat_draw_data: Vec<crate::resources::GaussianSplatDrawData>,
    /// Per-frame screen-image GPU data, rebuilt in prepare(), consumed in paint() (Phase 10B).
    screen_image_gpu_data: Vec<crate::resources::ScreenImageGpuData>,
    /// Per-frame overlay image GPU data, rebuilt in prepare(), consumed in paint() (Phase 7).
    overlay_image_gpu_data: Vec<crate::resources::ScreenImageGpuData>,
    /// Per-frame overlay label GPU data, rebuilt in prepare(), consumed in paint().
    label_gpu_data: Option<crate::resources::LabelGpuData>,
    /// Per-frame scalar bar GPU data, rebuilt in prepare(), consumed in paint().
    scalar_bar_gpu_data: Option<crate::resources::LabelGpuData>,
    /// Per-frame ruler GPU data, rebuilt in prepare(), consumed in paint().
    ruler_gpu_data: Option<crate::resources::LabelGpuData>,
    /// Per-frame loading bar GPU data, rebuilt in prepare(), consumed in paint().
    loading_bar_gpu_data: Option<crate::resources::LabelGpuData>,
    /// Per-frame overlay rect GPU data, rebuilt in prepare(), consumed in paint().
    overlay_rect_gpu_data: Option<crate::resources::LabelGpuData>,
    /// Per-frame SDF overlay shape GPU data, rebuilt in prepare(), consumed in paint().
    overlay_shape_gpu_data: Option<crate::resources::OverlayShapeGpuData>,
    /// Per-viewport GPU state slots.
    ///
    /// Indexed by `FrameData::camera.viewport_index`. Each slot owns independent
    /// uniform buffers and bind groups for camera, clip planes, clip volume,
    /// shadow info, and grid. Slots are grown lazily in `prepare` via
    /// `ensure_viewport_slot`. There are at most 4 in the current UI.
    viewport_slots: Vec<ViewportSlot>,
    /// Phase G : GPU compute filter results from the last `prepare()` call.
    ///
    /// Each entry contains a compacted index buffer + count for one filtered mesh.
    /// Consumed during `paint()` to override the mesh's default index buffer.
    /// Cleared and rebuilt each frame.
    compute_filter_results: Vec<crate::resources::ComputeFilterResult>,
    /// Per-item uniform buffers for wireframe mode. In wireframe mode multiple scene
    /// items can share the same MeshId, but each needs its own object uniform (model
    /// matrix, colour, etc.). The mesh's single `object_uniform_buf` gets overwritten
    /// by the last item prepared, so we maintain a separate pool here. Indexed in the
    /// same order as the visible scene items. Grown lazily, never shrunk.
    wireframe_uniform_bufs: Vec<wgpu::Buffer>,
    /// Bind groups corresponding to `wireframe_uniform_bufs`. Each bind group pairs
    /// the per-item uniform buffer with the mesh's fallback textures so it is
    /// compatible with the object bind group layout.
    wireframe_bind_groups: Vec<wgpu::BindGroup>,
    /// Per-frame list of boundary mesh IDs to draw in wireframe for
    /// TransparentVolumeMeshItems with `appearance.wireframe = true`.
    /// Cleared and rebuilt each frame in prepare_scene_internal.
    tvm_wireframe_draws: Vec<crate::resources::mesh_store::MeshId>,
    /// Shared uniform buffer for TVM boundary wireframe draws (wireframe=1, model=identity).
    /// Created once on first use, reused every frame.
    tvm_wireframe_buf: Option<wgpu::Buffer>,
    /// Bind group for TVM boundary wireframe draws. Pairs `tvm_wireframe_buf` with
    /// fallback textures matching the object bind group layout.
    tvm_wireframe_bg: Option<wgpu::BindGroup>,
    /// Cascade-0 light-space view-projection matrix from the last shadow prepare.
    /// Cached here so `prepare_viewport_internal` can copy it into the ground plane uniform.
    last_cascade0_shadow_mat: glam::Mat4,
    /// Current runtime mode controlling internal default behavior.
    runtime_mode: crate::renderer::stats::RuntimeMode,
    /// Active performance policy: target FPS, render scale bounds, and permitted reductions.
    performance_policy: crate::renderer::stats::PerformancePolicy,
    /// Current render scale tracked by the adaptation controller (or set manually).
    ///
    /// Clamped to `[policy.min_render_scale, policy.max_render_scale]`.
    /// Reported in `FrameStats::render_scale` each frame.
    current_render_scale: f32,
    /// Instant recorded at the start of the most recent `prepare()` call.
    /// Used to compute `total_frame_ms` on the following frame.
    last_prepare_instant: Option<std::time::Instant>,
    /// Frame counter incremented each `prepare()` call. Used for picking throttle in Playback mode.
    frame_counter: u64,
    /// Surface items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_scene_items: Vec<SceneRenderItem>,
    /// Point cloud items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_point_cloud_items: Vec<PointCloudItem>,
    /// Gaussian splat items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_splat_items: Vec<GaussianSplatItem>,
    /// Volume items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_volume_items: Vec<VolumeItem>,
    /// Transparent volume mesh items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_tvm_items: Vec<TransparentVolumeMeshItem>,
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
    /// GPU implicit surface items from the last `prepare()` call, retained for `pick()` dispatch.
    pick_implicit_items: Vec<GpuImplicitPickItem>,
    /// GPU marching cubes jobs from the last `prepare()` call, retained for `pick()` dispatch.
    pick_mc_items: Vec<GpuMcPickItem>,

    // --- Phase 4 : GPU timestamp queries ---
    /// Timestamp query set with 2 entries (scene-pass begin + end).
    /// `None` when `TIMESTAMP_QUERY` is unavailable or not yet initialized.
    ts_query_set: Option<wgpu::QuerySet>,
    /// Resolve buffer: 2 × u64, GPU-only (`QUERY_RESOLVE | COPY_SRC`).
    ts_resolve_buf: Option<wgpu::Buffer>,
    /// Staging buffer: 2 × u64, CPU-readable (`COPY_DST | MAP_READ`).
    ts_staging_buf: Option<wgpu::Buffer>,
    /// Nanoseconds per GPU timestamp tick, from `queue.get_timestamp_period()`.
    ts_period: f32,
    /// Whether the staging buffer holds unread timestamp data from the previous frame.
    ts_needs_readback: bool,

    // --- Indirect-args readback (GPU-driven culling visible instance count) ---
    /// CPU-readable staging buffer for `indirect_args_buf` (batch_count × 20 bytes).
    /// Grown lazily; never shrunk.
    indirect_readback_buf: Option<wgpu::Buffer>,
    /// Number of batches whose data was copied into `indirect_readback_buf` last frame.
    indirect_readback_batch_count: u32,
    /// True when `indirect_readback_buf` holds unread data from the previous cull pass.
    indirect_readback_pending: bool,

    // --- Per-pass degradation state (Phases 6 + 11) ---
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
}

impl ViewportRenderer {
    /// Create a new renderer with default settings (no MSAA).
    /// Call once at application startup.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        Self::with_sample_count(device, target_format, 1)
    }

    /// Create a new renderer with the specified MSAA sample count (1, 2, or 4).
    ///
    /// When using MSAA (sample_count > 1), the caller must create multisampled
    /// colour and depth textures and use them as render pass attachments with the
    /// final surface texture as the resolve target.
    pub fn with_sample_count(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let gpu_culling_supported = device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE);
        Self {
            resources: ViewportGpuResources::new(device, target_format, sample_count),
            instanced_batches: Vec::new(),
            use_instancing: false,
            gpu_culling_supported,
            gpu_culling_enabled: gpu_culling_supported,
            cull_resources: None,
            last_stats: crate::renderer::stats::FrameStats::default(),
            last_scene_generation: u64::MAX,
            last_selection_generation: u64::MAX,
            last_scene_items_count: usize::MAX,
            last_instancable_count: usize::MAX,
            cached_instance_count: 0,
            cached_instance_hashes: Vec::new(),
            cached_instanced_batches: Vec::new(),
            force_full_upload: false,
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
            gaussian_splat_draw_data: Vec::new(),
            lic_gpu_data: Vec::new(),
            implicit_gpu_data: Vec::new(),
            mc_gpu_data: Vec::new(),
            screen_image_gpu_data: Vec::new(),
            overlay_image_gpu_data: Vec::new(),
            label_gpu_data: None,
            scalar_bar_gpu_data: None,
            ruler_gpu_data: None,
            loading_bar_gpu_data: None,
            overlay_rect_gpu_data: None,
            overlay_shape_gpu_data: None,
            viewport_slots: Vec::new(),
            compute_filter_results: Vec::new(),
            wireframe_uniform_bufs: Vec::new(),
            wireframe_bind_groups: Vec::new(),
            tvm_wireframe_draws: Vec::new(),
            tvm_wireframe_buf: None,
            tvm_wireframe_bg: None,
            last_cascade0_shadow_mat: glam::Mat4::IDENTITY,
            runtime_mode: crate::renderer::stats::RuntimeMode::Interactive,
            performance_policy: crate::renderer::stats::PerformancePolicy::default(),
            current_render_scale: 1.0,
            last_prepare_instant: None,
            frame_counter: 0,
            pick_scene_items: Vec::new(),
            pick_point_cloud_items: Vec::new(),
            pick_splat_items: Vec::new(),
            pick_volume_items: Vec::new(),
            pick_tvm_items: Vec::new(),
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
            pick_implicit_items: Vec::new(),
            pick_mc_items: Vec::new(),
            ts_query_set: None,
            ts_resolve_buf: None,
            ts_staging_buf: None,
            ts_period: 1.0,
            ts_needs_readback: false,
            indirect_readback_buf: None,
            indirect_readback_batch_count: 0,
            indirect_readback_pending: false,
            degradation_tier: 0,
            degradation_shadows_skipped: false,
            degradation_volume_quality_reduced: false,
            degradation_effects_throttled: false,
        }
    }

    /// Access the underlying GPU resources (e.g. for mesh uploads).
    pub fn resources(&self) -> &ViewportGpuResources {
        &self.resources
    }

    /// Performance counters from the last completed frame.
    pub fn last_frame_stats(&self) -> crate::renderer::stats::FrameStats {
        self.last_stats
    }

    /// Disable GPU-driven culling, reverting to the direct draw path.
    ///
    /// Has no effect when the device does not support `INDIRECT_FIRST_INSTANCE`
    /// (culling is already disabled on those devices).
    pub fn disable_gpu_driven_culling(&mut self) {
        self.gpu_culling_enabled = false;
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
        self.force_full_upload = true;
        // Also invalidate the generation cache so the next prepare is guaranteed
        // to enter the rebuild path even if the scene generation is unchanged.
        self.last_scene_generation = u64::MAX;
    }

    /// Re-enable GPU-driven culling after a call to `disable_gpu_driven_culling`.
    ///
    /// Has no effect when the device does not support `INDIRECT_FIRST_INSTANCE`.
    pub fn enable_gpu_driven_culling(&mut self) {
        if self.gpu_culling_supported {
            self.gpu_culling_enabled = true;
        }
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
    pub fn resources_mut(&mut self) -> &mut ViewportGpuResources {
        &mut self.resources
    }

    /// Upload a Gaussian splat set to the GPU.
    ///
    /// Call once per splat set at startup or when it changes. The returned
    /// [`GaussianSplatId`] is valid until [`remove_gaussian_splats`](Self::remove_gaussian_splats) is called.
    pub fn upload_gaussian_splats(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &GaussianSplatData,
    ) -> GaussianSplatId {
        self.resources.upload_gaussian_splats(device, queue, data)
    }

    /// Remove an uploaded Gaussian splat set by handle.
    ///
    /// After this call the `id` is invalid and must not be submitted in `SceneFrame`.
    pub fn remove_gaussian_splats(&mut self, id: GaussianSplatId) {
        self.resources.remove_gaussian_splats(id);
    }

    /// Upload an equirectangular HDR environment map and precompute IBL textures.
    ///
    /// `pixels` is row-major RGBA f32 data (4 floats per texel), `width`×`height`.
    /// This rebuilds camera bind groups so shaders immediately see the new textures.
    pub fn upload_environment_map(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pixels: &[f32],
        width: u32,
        height: u32,
    ) {
        crate::resources::environment::upload_environment_map(
            &mut self.resources,
            device,
            queue,
            pixels,
            width,
            height,
        );
        self.rebuild_camera_bind_groups(device);
    }

    /// Rebuild the primary + per-viewport camera bind groups.
    ///
    /// Call after IBL textures are uploaded so shaders see the new environment.
    fn rebuild_camera_bind_groups(&mut self, device: &wgpu::Device) {
        self.resources.camera_bind_group = self.resources.create_camera_bind_group(
            device,
            &self.resources.camera_uniform_buf,
            &self.resources.clip_planes_uniform_buf,
            &self.resources.shadow_info_buf,
            &self.resources.clip_volume_uniform_buf,
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
        }
    }

    /// Ensure a per-viewport slot exists for `viewport_index`.
    ///
    /// Creates a full `ViewportSlot` with independent uniform buffers for camera,
    /// clip planes, clip volume, shadow info, and grid. The camera bind group
    /// references this slot's per-viewport buffers plus shared scene-global
    /// resources. Slots are created lazily and never destroyed.
    fn ensure_viewport_slot(&mut self, device: &wgpu::Device, viewport_index: usize) {
        while self.viewport_slots.len() <= viewport_index {
            let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_camera_buf"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let clip_planes_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_clip_planes_buf"),
                size: std::mem::size_of::<ClipPlanesUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let clip_volume_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_clip_volume_buf"),
                size: std::mem::size_of::<ClipVolumesUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let shadow_info_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_shadow_info_buf"),
                size: std::mem::size_of::<ShadowAtlasUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_grid_buf"),
                size: std::mem::size_of::<GridUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

            let grid_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vp_grid_bind_group"),
                layout: &self.resources.grid_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_buf.as_entire_binding(),
                }],
            });

            // Per-viewport gizmo buffers (initial mesh: Translate, no hover, identity orientation).
            let (gizmo_verts, gizmo_indices) = crate::interaction::gizmo::build_gizmo_mesh(
                crate::interaction::gizmo::GizmoMode::Translate,
                crate::interaction::gizmo::GizmoAxis::None,
                glam::Quat::IDENTITY,
            );
            let gizmo_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_gizmo_vertex_buf"),
                size: (std::mem::size_of::<crate::resources::Vertex>() * gizmo_verts.len().max(1))
                    as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            gizmo_vertex_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&gizmo_verts));
            gizmo_vertex_buffer.unmap();
            let gizmo_index_count = gizmo_indices.len() as u32;
            let gizmo_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_gizmo_index_buf"),
                size: (std::mem::size_of::<u32>() * gizmo_indices.len().max(1)) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            gizmo_index_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&gizmo_indices));
            gizmo_index_buffer.unmap();
            let gizmo_uniform = crate::interaction::gizmo::GizmoUniform {
                model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            };
            let gizmo_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_gizmo_uniform_buf"),
                size: std::mem::size_of::<crate::interaction::gizmo::GizmoUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            gizmo_uniform_buf
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&[gizmo_uniform]));
            gizmo_uniform_buf.unmap();
            let gizmo_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vp_gizmo_bind_group"),
                layout: &self.resources.gizmo_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gizmo_uniform_buf.as_entire_binding(),
                }],
            });

            // Per-viewport axes vertex buffer (2048 vertices = enough for all axes geometry).
            let axes_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vp_axes_vertex_buf"),
                size: (std::mem::size_of::<crate::widgets::axes_indicator::AxesVertex>() * 2048)
                    as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.viewport_slots.push(ViewportSlot {
                camera_buf,
                clip_planes_buf,
                clip_volume_buf,
                shadow_info_buf,
                grid_buf,
                camera_bind_group,
                grid_bind_group,
                hdr: None,
                outline_object_buffers: Vec::new(),
                splat_outline_buffers: Vec::new(),
                volume_outline_indices: Vec::new(),
                glyph_outline_indices: Vec::new(),
                tensor_glyph_outline_indices: Vec::new(),
                sprite_outline_indices: Vec::new(),
                raw_geom_outline_buffers: Vec::new(),
                screen_rect_outline_buffers: Vec::new(),
                implicit_outline_indices: Vec::new(),
                mc_outline_data: Vec::new(),
                streamtube_outline_items: Vec::new(),
                tube_outline_items: Vec::new(),
                ribbon_outline_items: Vec::new(),
                polyline_outline_indices: Vec::new(),
                xray_object_buffers: Vec::new(),
                constraint_line_buffers: Vec::new(),
                cap_buffers: Vec::new(),
                clip_plane_fill_buffers: Vec::new(),
                clip_plane_line_buffers: Vec::new(),
                axes_vertex_buffer,
                axes_vertex_count: 0,
                gizmo_uniform_buf,
                gizmo_bind_group,
                gizmo_vertex_buffer,
                gizmo_index_buffer,
                gizmo_index_count,
                sub_highlight: None,
                sub_highlight_generation: u64::MAX,
                dyn_res: None,
                hdr_callback: None,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Multi-viewport public API (Phase 5)
    // -----------------------------------------------------------------------

    /// Create a new viewport slot and return its handle.
    ///
    /// The returned [`ViewportId`] is stable for the lifetime of the renderer.
    /// Pass it to [`prepare_viewport`](Self::prepare_viewport),
    /// [`paint_viewport`](Self::paint_viewport), and
    /// [`render_viewport`](Self::render_viewport) each frame.
    ///
    /// Also set `CameraFrame::viewport_index` to `id.0` when building the
    /// [`FrameData`] for this viewport:
    /// ```rust,ignore
    /// let id = renderer.create_viewport(&device);
    /// let frame = FrameData {
    ///     camera: CameraFrame::from_camera(&cam, size).with_viewport_index(id.0),
    ///     ..Default::default()
    /// };
    /// ```
    pub fn create_viewport(&mut self, device: &wgpu::Device) -> ViewportId {
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

    /// Prepare shared scene data.  Call **once per frame**, before any
    /// [`prepare_viewport`](Self::prepare_viewport) calls.
    ///
    /// `frame` provides the scene content (`frame.scene`) and the primary camera
    /// used for shadow cascade framing (`frame.camera`).  In a multi-viewport
    /// setup use any one viewport's `FrameData` here : typically the perspective
    /// view : as the shadow framing reference.
    ///
    /// `scene_effects` carries the scene-global effects: lighting, environment
    /// map, and compute filters.  Obtain it by constructing [`SceneEffects`]
    /// directly or via [`EffectsFrame::split`].
    pub fn prepare_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &FrameData,
        scene_effects: &SceneEffects<'_>,
    ) {
        self.prepare_scene_internal(device, queue, frame, scene_effects);
    }

    /// Prepare per-viewport GPU state (camera, clip planes, overlays, axes).
    ///
    /// Call once per viewport per frame, **after** [`prepare_scene`](Self::prepare_scene).
    ///
    /// `id` must have been obtained from [`create_viewport`](Self::create_viewport).
    /// `frame.camera.viewport_index` must equal `id.0`; use
    /// [`CameraFrame::with_viewport_index`] when building the frame.
    pub fn prepare_viewport(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: ViewportId,
        frame: &FrameData,
    ) {
        debug_assert_eq!(
            frame.camera.viewport_index, id.0,
            "frame.camera.viewport_index ({}) must equal the ViewportId ({}); \
             use CameraFrame::with_viewport_index(id.0)",
            frame.camera.viewport_index, id.0,
        );
        let (_, viewport_fx) = frame.effects.split();
        self.prepare_viewport_internal(device, queue, frame, &viewport_fx);
    }

    /// Issue draw calls for `id` into a `'static` render pass (as provided by egui callbacks).
    ///
    /// This is the method to use from an egui/eframe `CallbackTrait::paint` implementation.
    /// Call [`prepare_scene`](Self::prepare_scene) and [`prepare_viewport`](Self::prepare_viewport)
    /// first (in `CallbackTrait::prepare`), then set the render pass viewport/scissor to confine
    /// drawing to the correct quadrant, and call this method.
    ///
    /// For non-`'static` render passes (winit, iced, manual wgpu), use
    /// [`paint_viewport_to`](Self::paint_viewport_to).
    pub fn paint_viewport(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
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
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            grid_bg,
            &self.compute_filter_results,
            vp_slot,
            &self.wireframe_bind_groups
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
            false
        );
        // Gaussian splats (alpha-blended, back-to-front sorted, no depth write).
        if !self.gaussian_splat_draw_data.is_empty() {
            if let Some(ref dual) = self.resources.gaussian_splat_pipeline {
                render_pass.set_pipeline(dual.for_format(false));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for dd in &self.gaussian_splat_draw_data {
                    if dd.wireframe {
                        continue;
                    }
                    if let Some(set) = self.resources.gaussian_splat_store.get(dd.store_index) {
                        if let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) {
                            render_pass.set_bind_group(1, &vp_sort.render_bg, &[]);
                            render_pass.draw(0..6, 0..dd.count);
                        }
                    }
                }
            }
        }
        // TransparentVolumeMesh boundary wireframe overlay.
        if !self.tvm_wireframe_draws.is_empty() {
            if let Some(ref tvm_bg) = self.tvm_wireframe_bg {
                render_pass.set_bind_group(0, camera_bg, &[]);
                for mesh_id in &self.tvm_wireframe_draws {
                    if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                        render_pass.set_pipeline(&self.resources.wireframe_pipeline);
                        render_pass.set_bind_group(1, tvm_bg, &[]);
                        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            mesh.edge_index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                    }
                }
            }
        }
    }

    /// Issue draw calls for `id` into a render pass with any lifetime.
    ///
    /// Identical to [`paint_viewport`](Self::paint_viewport) but accepts a render pass with a
    /// non-`'static` lifetime, making it usable from winit, iced, or raw wgpu where the encoder
    /// creates its own render pass.
    pub fn paint_viewport_to<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
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
            self.use_instancing,
            &self.instanced_batches,
            camera_bg,
            grid_bg,
            &self.compute_filter_results,
            vp_slot,
            &self.wireframe_bind_groups
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
            false
        );
        // Gaussian splats (alpha-blended, back-to-front sorted, no depth write).
        if !self.gaussian_splat_draw_data.is_empty() {
            if let Some(ref dual) = self.resources.gaussian_splat_pipeline {
                render_pass.set_pipeline(dual.for_format(false));
                render_pass.set_bind_group(0, camera_bg, &[]);
                for dd in &self.gaussian_splat_draw_data {
                    if dd.wireframe {
                        continue;
                    }
                    if let Some(set) = self.resources.gaussian_splat_store.get(dd.store_index) {
                        if let Some(Some(vp_sort)) = set.viewport_sort.get(dd.viewport_index) {
                            render_pass.set_bind_group(1, &vp_sort.render_bg, &[]);
                            render_pass.draw(0..6, 0..dd.count);
                        }
                    }
                }
            }
        }
        // TransparentVolumeMesh boundary wireframe overlay.
        if !self.tvm_wireframe_draws.is_empty() {
            if let Some(ref tvm_bg) = self.tvm_wireframe_bg {
                render_pass.set_bind_group(0, camera_bg, &[]);
                for mesh_id in &self.tvm_wireframe_draws {
                    if let Some(mesh) = self.resources.mesh_store.get(*mesh_id) {
                        render_pass.set_pipeline(&self.resources.wireframe_pipeline);
                        render_pass.set_bind_group(1, tvm_bg, &[]);
                        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            mesh.edge_index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        render_pass.draw_indexed(0..mesh.edge_index_count, 0, 0..1);
                    }
                }
            }
        }
    }

    /// Return a reference to the camera bind group for the given viewport slot.
    ///
    /// Falls back to `resources.camera_bind_group` if no per-viewport slot
    /// exists (e.g. in single-viewport mode before the first prepare call).
    fn viewport_camera_bind_group(&self, viewport_index: usize) -> &wgpu::BindGroup {
        self.viewport_slots
            .get(viewport_index)
            .map(|slot| &slot.camera_bind_group)
            .unwrap_or(&self.resources.camera_bind_group)
    }

    /// Return a reference to the grid bind group for the given viewport slot.
    ///
    /// Falls back to `resources.grid_bind_group` if no per-viewport slot exists.
    fn viewport_grid_bind_group(&self, viewport_index: usize) -> &wgpu::BindGroup {
        self.viewport_slots
            .get(viewport_index)
            .map(|slot| &slot.grid_bind_group)
            .unwrap_or(&self.resources.grid_bind_group)
    }

    /// Ensure the dyn-res intermediate render target exists for `vp_idx` at the
    /// given `scaled_size`, creating or recreating it when size changes.
    ///
    /// `surface_size` is the native output dimensions (used to size the upscale
    /// blit correctly). `ensure_dyn_res_pipeline` is called automatically.
    pub(crate) fn ensure_dyn_res_target(
        &mut self,
        device: &wgpu::Device,
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

    /// Ensure per-viewport HDR state exists for `viewport_index` at dimensions `w`×`h`.
    ///
    /// Calls `ensure_hdr_shared` once to initialise shared pipelines/BGLs/samplers, then
    /// lazily creates or resizes the `ViewportHdrState` inside the slot. Idempotent: if the
    /// slot already has HDR state at the correct size nothing is recreated.
    pub(crate) fn ensure_viewport_hdr(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
