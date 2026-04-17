//! Per-frame performance counters for the viewport renderer.

/// Per-frame rendering statistics.
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
    }
}
