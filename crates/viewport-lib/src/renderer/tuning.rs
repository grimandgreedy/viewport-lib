//! Persistent performance/behavior tuning for [`crate::ViewportRenderer`].
//!
//! [`RenderTuning`] gathers the persistent (set-once, live-on-the-renderer)
//! performance knobs in one place so they are discoverable and their defaults
//! live in one [`Default`] impl instead of scattered across constructors. Apply
//! a whole set with [`crate::ViewportRenderer::apply_tuning`] and read the live
//! values with [`crate::ViewportRenderer::tuning`]; both are equivalent to the
//! individual setter methods and apply the same gating.

use crate::renderer::stats::{PerformancePolicy, RuntimeMode};

/// Persistent performance/behavior knobs for a [`crate::ViewportRenderer`],
/// gathered in one place so they are discoverable and their defaults live in one
/// [`Default`] impl instead of scattered across constructors. Apply with
/// [`crate::ViewportRenderer::apply_tuning`]; read the live values with
/// [`crate::ViewportRenderer::tuning`].
///
/// Not included here (they are not persistent runtime knobs):
/// - Per-frame settings you rebuild each frame on `EffectsFrame`
///   (`LightingSettings` shadows, `PostProcessSettings`, `ScatterSettings`); they
///   can differ per frame/viewport.
/// - MSAA sample count and the GPU pipeline cache: construction-time, see
///   [`crate::ViewportRenderer::with_sample_count`] /
///   [`crate::ViewportRenderer::new_with_pipeline_cache`].
/// - LOD groups: a registration API keyed by mesh, see
///   [`crate::ViewportRenderer::register_lod_group`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct RenderTuning {
    /// Request GPU-driven culling (frustum cull + indirect draw). Reconciled
    /// against device support on apply: no effect unless the device offers
    /// `INDIRECT_FIRST_INSTANCE`. Default: `true` (active where supported).
    pub gpu_driven_culling: bool,
    /// HiZ occlusion culling on top of the frustum cull. No effect unless
    /// `gpu_driven_culling` is active and the pass captures scene depth
    /// (HDR / owned-LDR paths, single viewport). Default: `false`.
    pub occlusion_culling: bool,
    /// Adaptive resolution + load-shedding policy (target FPS, render-scale
    /// bounds, quality preset, `allow_*` degradation flags). See
    /// [`PerformancePolicy`]. Default: all off (fixed quality).
    pub performance: PerformancePolicy,
    /// Manual render scale, applied only when adaptive resolution is off
    /// (`performance.allow_dynamic_resolution == false`); clamped to the policy
    /// bounds. Default: `1.0`.
    pub render_scale: f32,
    /// Runtime mode: `Interactive` (full quality), `Playback` (throttles
    /// picking), `Capture` (forces max render scale). Default: `Interactive`.
    pub runtime_mode: RuntimeMode,
    /// Per-frame upload time budget: spill mesh/texture uploads across frames
    /// instead of draining them in one `prepare`. `None` drains everything. Soft
    /// cap (one oversized item still runs to completion). Default: `None`.
    pub upload_budget: Option<std::time::Duration>,
    /// Keep a CPU-side copy of pickable geometry each frame for CPU `pick` /
    /// `pick_rect`. Costs a per-frame geometry copy; leave off for GPU-only
    /// picking. Default: `false`.
    pub cpu_pick_cache: bool,
    /// Diagnostic / benchmark overrides. Not for normal tuning.
    pub diagnostics: RenderDiagnostics,
}

impl Default for RenderTuning {
    fn default() -> Self {
        Self {
            gpu_driven_culling: true,
            occlusion_culling: false,
            performance: PerformancePolicy::default(),
            render_scale: 1.0,
            runtime_mode: RuntimeMode::default(),
            upload_budget: None,
            cpu_pick_cache: false,
            diagnostics: RenderDiagnostics::default(),
        }
    }
}

/// Benchmark-only overrides on [`RenderTuning`], separated so they read as
/// diagnostics rather than tuning. Both default off and do not change output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct RenderDiagnostics {
    /// Force the multi-draw collapse even where the backend emulates it (Metal),
    /// to exercise/measure the collapsed path. No-op where native. Default off.
    pub force_multi_draw: bool,
    /// Force the per-object opaque pass to keep its discarding pipeline instead
    /// of the discard-free early-Z twin (A/B the early-Z win). Default off.
    pub force_po_discard: bool,
}
