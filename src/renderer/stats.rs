//! Per-frame performance counters for the viewport renderer.

/// Controls the renderer's internal default behavior.
///
/// The host application owns playback state, time, and scene content.
/// `RuntimeMode` tells the renderer what workload to expect so it can adjust
/// internal defaults (e.g. picking rate) accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuntimeMode {
    /// Prioritize responsiveness and picking accuracy.
    #[default]
    Interactive,
    /// Prioritize steady frame pacing; picking may be throttled.
    Playback,
    /// Restore full quality; picking runs at full rate.
    Paused,
    /// Deterministic full-quality render intended for screenshot or export.
    Capture,
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
    /// Total triangles submitted to the GPU.
    pub triangles_submitted: u64,
    /// Number of draw calls in the shadow pass.
    pub shadow_draw_calls: u32,
    /// CPU time spent in `prepare()`, in milliseconds.
    pub cpu_prepare_ms: f32,
    /// GPU frame time in milliseconds, if timestamp queries are available.
    ///
    /// Note: this reflects the *previous* frame's GPU cost due to async
    /// readback; the value lags by one frame. Always `None` until Phase 4
    /// (GPU timestamp queries) is implemented.
    pub gpu_frame_ms: Option<f32>,
    /// Wall-clock duration since the previous `prepare()` call, in milliseconds.
    ///
    /// Approximates the full frame interval. Zero on the first frame.
    pub total_frame_ms: f32,
    /// Current internal render scale (1.0 = native resolution).
    ///
    /// Always 1.0 until Phase 3 (dynamic resolution render target) is implemented.
    pub render_scale: f32,
    /// True if the last frame exceeded the target frame budget.
    ///
    /// Requires a target FPS to be set via
    /// [`crate::ViewportRenderer::set_target_fps`]. Always `false` when no
    /// target is configured.
    pub missed_budget: bool,
    /// Bytes of geometry data uploaded to the GPU since the previous
    /// `prepare()` call.
    ///
    /// Counts full buffer reallocations triggered by
    /// [`crate::ViewportGpuResources::replace_mesh_data`] and initial uploads
    /// via `upload_mesh_data` / `upload_mesh`. Uniform buffer writes are not
    /// counted.
    pub upload_bytes: u64,
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
        assert_eq!(stats.triangles_submitted, 0);
        assert_eq!(stats.shadow_draw_calls, 0);
        assert_eq!(stats.cpu_prepare_ms, 0.0);
        assert!(stats.gpu_frame_ms.is_none());
        assert_eq!(stats.total_frame_ms, 0.0);
        assert_eq!(stats.render_scale, 0.0);
        assert!(!stats.missed_budget);
        assert_eq!(stats.upload_bytes, 0);
    }

    #[test]
    fn test_runtime_mode_default_is_interactive() {
        assert_eq!(RuntimeMode::default(), RuntimeMode::Interactive);
    }
}
