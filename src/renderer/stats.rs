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
    /// **LDR path only.** Dynamic resolution applies when the scene is rendered
    /// via [`crate::ViewportRenderer::paint_to`] or
    /// [`crate::ViewportRenderer::paint_viewport`] (i.e.
    /// `PostProcessSettings::enabled` is `false`). When the HDR post-processing
    /// path is active (`render` / `render_viewport`), render scale has no effect
    /// on output quality and the adaptation controller is suppressed.
    /// `FrameStats::render_scale` will report 1.0 in that case.
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
    /// Total triangles submitted to the GPU.
    pub triangles_submitted: u64,
    /// Number of draw calls in the shadow pass.
    pub shadow_draw_calls: u32,
    /// CPU time spent in `prepare()`, in milliseconds.
    pub cpu_prepare_ms: f32,
    /// GPU scene-pass time in milliseconds, if timestamp queries are available.
    ///
    /// Measured with `TIMESTAMP_QUERY` around the main scene render pass.
    /// `None` on backends that do not support `TIMESTAMP_QUERY` (e.g. WebGL).
    ///
    /// Note: this value reflects the *previous* frame's GPU cost due to async
    /// readback. The value lags by one frame and should not be used by the
    /// adaptation controller across mode transitions.
    pub gpu_frame_ms: Option<f32>,
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
    /// Bytes of geometry data uploaded to the GPU since the previous
    /// `prepare()` call.
    ///
    /// Counts full buffer reallocations triggered by
    /// [`crate::ViewportGpuResources::replace_mesh_data`] and initial uploads
    /// via `upload_mesh_data` / `upload_mesh`. Uniform buffer writes are not
    /// counted.
    pub upload_bytes: u64,
    /// True when GPU-driven culling is active this frame.
    ///
    /// False when the device does not support `INDIRECT_FIRST_INSTANCE` or
    /// culling has been disabled via `disable_gpu_driven_culling()`.
    pub gpu_culling_active: bool,
    /// Number of instances that passed GPU culling and were submitted for drawing.
    ///
    /// Populated only when `gpu_culling_active` is true. `None` on the first frame
    /// or when GPU culling is off. Lags by one frame due to async readback.
    pub gpu_visible_instances: Option<u32>,

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
