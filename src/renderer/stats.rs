//! Per-frame performance counters for the viewport renderer.

/// Controls the renderer's internal default behavior.
///
/// The host application owns playback state, time, and scene content.
/// `RuntimeMode` tells the renderer what workload to expect so it can adjust
/// internal defaults (e.g. picking rate) accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RuntimeMode {
    /// Prioritize responsiveness and picking accuracy.
    #[default]
    Interactive,
    /// Prioritize steady frame pacing; picking may be throttled.
    Playback,
    /// Restore full quality; picking runs at full rate.
    Paused,
    /// Full-quality render intended for screenshot or export workflows.
    ///
    /// Forces `render_scale` to `max_render_scale` for the duration of the frame
    /// and suppresses all pass-level degradation regardless of `missed_budget`.
    /// The adaptation controller is paused; render scale resumes from
    /// `max_render_scale` on the next non-Capture frame.
    ///
    /// Note: blocking until GPU work is complete (for pixel readback) is the
    /// caller's responsibility. This mode ensures full-quality CPU-side decisions
    /// only; it does not insert any GPU synchronisation inside the renderer.
    Capture,
}

/// A coarse quality tier for [`PerformancePolicy`].
///
/// When `PerformancePolicy::preset` is `Some`, the renderer derives render scale
/// bounds and pass-level degradation flags from the preset instead of the individual
/// policy fields. The individual fields are still persisted so that switching back
/// to `None` restores the previous custom configuration.
///
/// `target_fps` and `allow_dynamic_resolution` are always taken from the policy
/// fields regardless of the preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum QualityPreset {
    /// Native resolution, all passes enabled, no degradation allowed.
    ///
    /// Equivalent to `min_render_scale = 1.0`, `max_render_scale = 1.0`,
    /// all `allow_*` flags false.
    High,
    /// Render scale [0.75, 1.0]; shadow reduction and effect throttling allowed.
    ///
    /// Equivalent to `min_render_scale = 0.75`, `max_render_scale = 1.0`,
    /// `allow_shadow_reduction = true`, `allow_effect_throttling = true`.
    Medium,
    /// Render scale [0.5, 0.75]; all degradation paths allowed.
    ///
    /// Equivalent to `min_render_scale = 0.5`, `max_render_scale = 0.75`,
    /// all `allow_*` flags true.
    Low,
}

/// Controls what quality reductions the viewport is allowed to apply under load.
///
/// Set once via [`crate::ViewportRenderer::set_performance_policy`]. The internal
/// adaptation controller reads `target_fps` and adjusts render scale within
/// `[min_render_scale, max_render_scale]` when `allow_dynamic_resolution` is true.
///
/// Pass-specific flags (`allow_shadow_reduction`, `allow_volume_quality_reduction`,
/// `allow_effect_throttling`) gate concrete quality reductions that kick in when
/// the previous frame missed the target budget, applied in order via a tiered
/// degradation ladder (render scale first, then shadows, then volumes, then effects).
///
/// Set `preset` to a [`QualityPreset`] to configure bounds and flags as a unit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerformancePolicy {
    /// Coarse quality tier. When `Some`, overrides `min_render_scale`,
    /// `max_render_scale`, and all `allow_*` flags for the duration of the frame.
    /// Set to `None` to use the individual fields directly.
    pub preset: Option<QualityPreset>,
    /// Target frames per second. `None` means uncapped; `missed_budget` is always `false`.
    pub target_fps: Option<f32>,
    /// Lower bound for dynamic render scale (e.g. 0.5 = half resolution).
    ///
    /// Ignored when `preset` is `Some`; the preset's bounds are used instead.
    pub min_render_scale: f32,
    /// Upper bound for dynamic render scale (1.0 = native).
    ///
    /// Ignored when `preset` is `Some`; the preset's bounds are used instead.
    pub max_render_scale: f32,
    /// Allow the viewport to adjust render scale automatically when budget is exceeded.
    ///
    /// When `false`, the internal controller is inactive and render scale can be
    /// set manually via [`crate::ViewportRenderer::set_render_scale`].
    ///
    /// The controller uses `FrameStats::gpu_frame_ms` as the cost signal when GPU
    /// timestamp queries are available (excludes vsync wait). When GPU timestamps are
    /// unavailable it falls back to `total_frame_ms`, which reflects wall-clock frame
    /// duration and correctly fires over-budget at low frame rates.
    ///
    /// Works in both the LDR path (`paint_to` / `paint_viewport`) and the HDR
    /// path (`render` / `render_viewport`). In HDR mode, the scene textures
    /// (HDR colour, depth, bloom, SSAO, DoF, etc.) are allocated at
    /// `render_scale * resolution`; the tone-map pass upscales to native
    /// resolution. `FrameStats::render_scale` reflects the actual scale in both
    /// modes.
    pub allow_dynamic_resolution: bool,
    /// Allow the viewport to skip the shadow pass under load.
    ///
    /// When `true` and the previous frame exceeded the target budget, the shadow depth
    /// pass is skipped entirely. Shadows reappear as soon as the frame is within budget.
    /// Ignored when `preset` is `Some`.
    pub allow_shadow_reduction: bool,
    /// Allow the viewport to reduce volume raymarch quality under load.
    ///
    /// When `true` and the previous frame exceeded the target budget, the per-volume
    /// step size is doubled (half the number of samples), reducing GPU cost at the
    /// cost of coarser volume appearance. Ignored when `preset` is `Some`.
    pub allow_volume_quality_reduction: bool,
    /// Allow the viewport to skip non-essential HDR effect passes under load.
    ///
    /// When `true` and the previous frame exceeded the target budget, the SSAO,
    /// contact shadow, and bloom passes are skipped for that frame.
    /// Ignored when `preset` is `Some`.
    pub allow_effect_throttling: bool,
}

impl Default for PerformancePolicy {
    fn default() -> Self {
        Self {
            preset: None,
            target_fps: None,
            min_render_scale: 0.5,
            max_render_scale: 1.0,
            allow_dynamic_resolution: false,
            allow_shadow_reduction: false,
            allow_volume_quality_reduction: false,
            allow_effect_throttling: false,
        }
    }
}

/// CPU time spent in each phase of `prepare()`, in milliseconds.
///
/// `FrameStats::cpu_prepare_ms` is the total. This splits that total across the
/// main phases so a slow `prepare` can be attributed to a specific one instead
/// of guessing. The fields sum to roughly `cpu_prepare_ms`; `other_ms` carries
/// the remainder (timestamp readback, scatter-volume sort, stats assembly, and
/// other small per-frame work).
///
/// All values are wall-clock CPU time on the thread calling `prepare`, measured
/// with `Instant`. They measure how long it takes to record and submit the work,
/// not how long the GPU spends running it; use `gpu_frame_ms` for GPU cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrepareBreakdown {
    /// Item-type plugin prepare and cull dispatch. Skinning and other vertex
    /// deformers run here, so a heavy skinned crowd shows up in this field.
    pub plugin_ms: f32,
    /// Lighting setup: directional shadow cascade matrices, point-light cube-map
    /// faces, and light clustering.
    pub lighting_ms: f32,
    /// Per-object uniform writes on the non-instanced path. Zero when the scene
    /// is above the instancing threshold and all meshes batch.
    pub uniforms_ms: f32,
    /// Building and uploading instanced batches (grouping items by mesh and
    /// material, writing instance data).
    pub instancing_ms: f32,
    /// Uploading non-mesh geometry: glyphs, point clouds, polylines, decals,
    /// images, tubes, ribbons, slices, and volumes.
    pub geometry_ms: f32,
    /// Recording the shadow depth pass (directional cascades and point-light
    /// cube-map faces) into the atlas.
    pub shadow_ms: f32,
    /// Per-viewport prepare: camera and clip uniforms, grid, overlays, and
    /// interaction state. Runs once per viewport.
    pub viewport_ms: f32,
    /// Remainder of `prepare` not covered by the fields above.
    pub other_ms: f32,
}

/// Per-pass split of GPU frame time, measured with `TIMESTAMP_QUERY`.
///
/// Each field is the GPU duration of one render pass in milliseconds, captured
/// with timestamp queries around the pass. A field is `0.0` when that pass did
/// not run this frame (for example `shadow_ms` when shadows are off or
/// `oit_ms` when nothing transparent is drawn) or when the backend does not
/// support `TIMESTAMP_QUERY` (the same condition that leaves
/// [`FrameStats::gpu_frame_ms`] at `None`).
///
/// These passes are submitted in separate command buffers but resolved
/// together, so the values are comparable. Like `gpu_frame_ms`, they lag one
/// frame behind due to async readback. The passes measured here do not cover
/// every GPU pass (decals, scatter, bloom, depth of field, and the final
/// overlays are not split out), so the fields do not sum to the full frame GPU
/// time; treat the remainder as those un-instrumented passes plus present.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBreakdown {
    /// Main opaque HDR scene pass.
    pub scene_ms: f32,
    /// Directional shadow depth pass (all cascades rendered into the atlas).
    pub shadow_ms: f32,
    /// Order-independent transparency accumulation pass.
    pub oit_ms: f32,
    /// Tone-map and resolve pass that writes the final LDR image.
    pub post_ms: f32,
    /// Main-camera GPU cull dispatch: the `cull_instances` + `write_indirect_args`
    /// compute passes that produce the indirect draw args. `0.0` when GPU culling
    /// is off or the scene has no instanced batches. Shadow-cascade culls are not
    /// included here. Useful for checking whether the cull pass costs more than it
    /// saves on a given scene.
    pub cull_ms: f32,
    /// Point-light cubemap shadow faces, spanning the first to the last face
    /// pass rendered this frame (up to casters x 6 depth passes). This work
    /// used to be invisible to the breakdown and can dominate frames with
    /// several shadow-casting point lights over heavy geometry.
    pub point_shadow_ms: f32,
    /// Clustered-lighting build compute pass. `0.0` when the frame ran the
    /// small-N fallback (under 16 active lights).
    pub cluster_ms: f32,
    /// SSAO passes (occlusion + blur). `0.0` when SSAO is off.
    pub ssao_ms: f32,
    /// Bloom passes (threshold extract + blur chain). `0.0` when bloom is off.
    pub bloom_ms: f32,
    /// FXAA fullscreen pass. `0.0` when FXAA is off.
    pub fxaa_ms: f32,
}

/// Per-frame rendering statistics returned by [`crate::ViewportRenderer::prepare`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameStats {
    /// Total objects considered for rendering.
    pub total_objects: u32,
    /// Objects that passed visibility and frustum tests.
    pub visible_objects: u32,
    /// Objects culled by frustum or visibility.
    pub culled_objects: u32,
    /// Number of draw calls issued in the main pass.
    pub draw_calls: u32,
    /// Number of instanced batches (0 when using per-object path).
    pub instanced_batches: u32,
    /// True when `prepare()` holds a cached render bundle for the opaque
    /// per-object draws this frame (large all-per-object scenes on the LDR
    /// path). The paint pass replays the bundle instead of recording one draw
    /// per item.
    pub per_object_bundle_cached: bool,
    /// Visible items that miss the instanced fast path and are drawn one at a
    /// time through the per-object path.
    ///
    /// An item is non-instanceable when it uses a styled back-face policy
    /// (different-colour/tint/pattern) or is two-sided and transparent, uses a
    /// matcap, has a scalar attribute or parameter visualization, carries a
    /// position/normal override, has per-instance deform data (skinning), or is
    /// hit by a compute filter. Each such item costs a uniform write and a bind-group
    /// build in `prepare`, so a large count here means `prepare` is paying
    /// per-object cost across much of the scene rather than batching it.
    pub per_object_items: u32,
    /// Per-object bind groups actually (re)built this frame.
    ///
    /// The per-object path caches bind groups and skips the rebuild when the
    /// item's textures, attributes, and overrides are unchanged. A value near
    /// `per_object_items` means the cache is missing most frames (for example
    /// because items move between slots); a value near zero means the remaining
    /// per-object cost is the uniform writes, not the bind-group builds.
    pub per_object_bind_groups_built: u32,
    /// Instanced batches whose GPU sub-range was re-uploaded this frame.
    ///
    /// Non-zero only on cache-miss frames where at least one batch's instance
    /// data changed. Zero on cache-hit frames and when the per-object path is
    /// active. Together with `batches_skipped`, this lets callers measure how
    /// effective the dirty-flagging optimisation is for their scene.
    pub batches_reuploaded: u32,
    /// Instanced batches whose GPU sub-range was reused unchanged this frame.
    ///
    /// Non-zero only on cache-miss frames where at least one batch compared
    /// equal to the cached data and its write was skipped. Zero on cache-hit
    /// frames (no upload attempted at all) and when the per-object path is
    /// active.
    pub batches_skipped: u32,
    /// Decals whose GPU buffer + bind group were built this frame (cache miss).
    ///
    /// Non-zero only when a decal is new or its content changed. In steady state
    /// with static decals this is zero after the first frame: a persistent value
    /// near the decal count means the cache is missing every frame.
    pub decal_uploads: u32,
    /// Decals served from the cross-frame cache this frame (cache hit).
    ///
    /// Together with `decal_uploads` this shows how effective the decal resource
    /// cache is: `decal_reused` should equal the visible decal count and
    /// `decal_uploads` should be zero once decals are settled.
    pub decal_reused: u32,
    /// Total triangles submitted to the GPU.
    pub triangles_submitted: u64,
    /// Number of draw calls in the shadow pass.
    pub shadow_draw_calls: u32,
    /// CPU time spent in `prepare()`, in milliseconds.
    pub cpu_prepare_ms: f32,
    /// Per-phase split of `cpu_prepare_ms`. See [`PrepareBreakdown`].
    pub prepare_breakdown: PrepareBreakdown,
    /// CPU time spent encoding the paint pass (draw-call recording) in
    /// milliseconds, separate from `cpu_prepare_ms`.
    ///
    /// Measured around the render/paint encode and latched after it, so
    /// `last_frame_stats()` reflects the current frame once `render` has run.
    /// The value carried in the `FrameStats` returned by `prepare()` is the
    /// previous frame's paint, since paint runs after prepare. For
    /// multi-viewport rendering this reflects the most recently painted
    /// viewport, not the sum.
    pub cpu_paint_ms: f32,
    /// GPU frame time in milliseconds, if timestamp queries are available.
    ///
    /// The span from the first to the last measured pass this frame (shadow
    /// cascades, point-shadow cubemap faces, cluster build, cull, scene, OIT,
    /// and post; see [`GpuBreakdown`] for the split). Work between and around
    /// those passes on the same queue is included in the span; passes with no
    /// timestamp slot that run outside it are not. Previously this field
    /// reported only the scene pass, which badly under-reported frames
    /// dominated by shadow or compute work.
    /// `None` on backends that do not support `TIMESTAMP_QUERY` (e.g. WebGL).
    ///
    /// Note: this value reflects a recent frame's GPU cost, not the current
    /// one: the queries are resolved one frame after the passes that wrote
    /// them and read back asynchronously, so the value lags by about two
    /// frames and should not be used by the adaptation controller across mode
    /// transitions.
    ///
    /// Between readbacks the last measured value is carried over rather than
    /// cleared. Compare `gpu_sample_generation` across frames to tell a fresh
    /// measurement from a carried-over one.
    pub gpu_frame_ms: Option<f32>,
    /// Incremented each time `gpu_frame_ms` and [`GpuBreakdown`] are refreshed
    /// from a completed GPU timestamp readback. Unchanged between readbacks,
    /// where `gpu_frame_ms` re-surfaces its previous value; consumers building
    /// histograms or percentiles from `gpu_frame_ms` should drop frames whose
    /// generation matches the previous frame's.
    pub gpu_sample_generation: u64,
    /// Per-pass split of GPU frame time. See [`GpuBreakdown`].
    ///
    /// Populated under the same conditions as `gpu_frame_ms` (requires
    /// `TIMESTAMP_QUERY`); all fields are `0.0` otherwise.
    pub gpu_breakdown: GpuBreakdown,
    /// Wall-clock duration since the previous `prepare()` call, in milliseconds.
    ///
    /// Approximates the full frame interval. Zero on the first frame.
    pub total_frame_ms: f32,
    /// Current internal render scale (1.0 = native resolution).
    ///
    /// Reflects the value tracked by the adaptation controller. Values below 1.0
    /// cause the scene to render into a scaled intermediate texture that is
    /// bilinearly upscaled to the surface (requires `allow_dynamic_resolution`).
    pub render_scale: f32,
    /// True if the last frame exceeded the target frame budget.
    ///
    /// Requires `target_fps` to be set in the [`PerformancePolicy`]. Always
    /// `false` when no target is configured.
    pub missed_budget: bool,
    /// Bytes written to GPU buffers for scene content since the previous
    /// `prepare()` call.
    ///
    /// Counts mesh data (initial uploads via `upload_mesh_data` /
    /// `upload_mesh` and reallocations via
    /// [`crate::DeviceResources::replace_mesh_data`]), instanced batch
    /// data (instance/AABB/batch-meta writes, full and partial), and
    /// per-object uniform writes for items whose transform or material
    /// changed. A steady frame over a static scene reads 0; a non-zero
    /// value attributes frame cost to CPU-to-GPU transfer.
    pub upload_bytes: u64,
    /// GPU pipelines compiled lazily since the previous `prepare()` call.
    ///
    /// Most pipelines are built at renderer creation or when a resource is
    /// uploaded; the rest compile on the first frame that uses a feature. A
    /// non-zero value on a frame that hitched points at one of those compiles.
    /// Counts calls to the internal pipeline builders, so one increment can
    /// cover a small family of pipeline variants built together.
    pub pipelines_built_this_frame: u32,
    /// True when GPU-driven culling is active this frame.
    ///
    /// False when the device does not support `INDIRECT_FIRST_INSTANCE` or
    /// culling has been disabled via `disable_gpu_driven_culling()`.
    pub gpu_culling_active: bool,
    /// Number of instances that passed GPU culling and were submitted for drawing.
    ///
    /// This is the count after both the frustum and occlusion tests. Populated
    /// only when `gpu_culling_active` is true. `None` on the first frame or when
    /// GPU culling is off. Lags by one frame due to async readback.
    pub gpu_visible_instances: Option<u32>,
    /// Total instances considered by the main-camera cull this frame, before
    /// any test. Same readback and `None` conditions as `gpu_visible_instances`.
    pub gpu_culled_total: Option<u32>,
    /// Instances that survived the frustum test, before the HiZ occlusion test.
    ///
    /// `gpu_culled_total - gpu_frustum_visible` is the frustum-culled count, and
    /// `gpu_frustum_visible - gpu_visible_instances` is the occlusion-culled
    /// count. When occlusion culling is off, this equals `gpu_visible_instances`.
    /// Same readback and `None` conditions as `gpu_visible_instances`.
    pub gpu_frustum_visible: Option<u32>,

    // --- Per-pass degradation flags ---
    /// True when the shadow depth pass was skipped this frame.
    ///
    /// Set when `allow_shadow_reduction` is true and the previous frame missed
    /// the target budget. Always false in [`RuntimeMode::Capture`].
    pub shadows_skipped: bool,
    /// True when volume raymarch step size was doubled this frame.
    ///
    /// Set when `allow_volume_quality_reduction` is true and the previous frame
    /// missed the target budget. Always false in [`RuntimeMode::Capture`].
    pub volume_quality_reduced: bool,
    /// True when SSAO, contact shadows, and bloom were skipped this frame.
    ///
    /// Set when `allow_effect_throttling` is true and the previous frame missed
    /// the target budget. Always false in [`RuntimeMode::Capture`].
    pub effects_throttled: bool,

    /// Objects that resolved through a LOD group this frame.
    ///
    /// Counts `SceneRenderItem`s with a `lod_group` set, plus each individual
    /// `MeshInstanceItem` instance whose level was selected. Zero when nothing
    /// uses LOD.
    pub lod_items_resolved: u32,
    /// LOD objects whose chosen level changed from the previous frame.
    ///
    /// Counts both scene items and individual mesh instances that moved to a
    /// different level. Only objects with a `pick_id` are tracked across frames,
    /// so this counts switches among those. A high value relative to
    /// `lod_items_resolved` suggests thresholds are being crossed often (camera
    /// motion, or thresholds spaced too tightly for the hysteresis margin).
    pub lod_switches: u32,
    /// Objects dropped this frame because they fell below their LOD group's cull
    /// size. Counts hidden `SceneRenderItem`s plus individual mesh instances that
    /// were left out of every batch. Zero unless a group sets `cull_below`.
    pub lod_culled: u32,
    /// LOD objects drawn this frame at a reduced level (level index > 0, i.e.
    /// below full detail). Counts scene items plus individual mesh instances.
    ///
    /// Unlike [`lod_switches`](Self::lod_switches), this does not need per-object
    /// tracking, so it is accurate for objects with no `pick_id`. It is the
    /// reliable signal for "is LOD actually reducing geometry": if it stays near
    /// zero while distant objects fill the view, either the heavy meshes are not
    /// in LOD groups or the `min_screen_size` thresholds never trigger, and the
    /// triangle count is not coming down.
    pub lod_items_reduced: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_stats_default_is_zero() {
        let stats = FrameStats::default();
        assert_eq!(stats.total_objects, 0);
        assert_eq!(stats.visible_objects, 0);
        assert_eq!(stats.culled_objects, 0);
        assert_eq!(stats.draw_calls, 0);
        assert_eq!(stats.instanced_batches, 0);
        assert_eq!(stats.batches_reuploaded, 0);
        assert_eq!(stats.batches_skipped, 0);
        assert_eq!(stats.triangles_submitted, 0);
        assert_eq!(stats.shadow_draw_calls, 0);
        assert_eq!(stats.cpu_prepare_ms, 0.0);
        assert!(stats.gpu_frame_ms.is_none());
        assert_eq!(stats.total_frame_ms, 0.0);
        assert_eq!(stats.render_scale, 0.0);
        assert!(!stats.missed_budget);
        assert_eq!(stats.upload_bytes, 0);
        assert!(!stats.gpu_culling_active);
        assert!(stats.gpu_visible_instances.is_none());
        assert!(!stats.shadows_skipped);
        assert!(!stats.volume_quality_reduced);
        assert!(!stats.effects_throttled);
    }

    #[test]
    fn test_runtime_mode_default_is_interactive() {
        assert_eq!(RuntimeMode::default(), RuntimeMode::Interactive);
    }

    #[test]
    fn test_performance_policy_default() {
        let p = PerformancePolicy::default();
        assert!(p.preset.is_none());
        assert!(p.target_fps.is_none());
        assert!(!p.allow_dynamic_resolution);
        assert!(p.min_render_scale <= p.max_render_scale);
        assert_eq!(p.max_render_scale, 1.0);
    }
}
