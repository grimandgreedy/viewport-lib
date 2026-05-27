/// Per-frame shadow and lighting pipeline statistics for debug inspection.
///
/// Returned by [`crate::ViewportRenderer::shadow_debug_stats`]. All values reflect
/// the most recently completed `prepare` call (one frame behind the display).
#[derive(Clone, Copy, Debug)]
pub struct ShadowDebugStats {
    /// True when the current frame uses the instanced draw path.
    ///
    /// When true, edits to `mesh.wgsl` shadow sampling code have no effect --
    /// the active shader is `mesh_instanced.wgsl`.
    pub using_instanced_path: bool,
    /// Number of instanced batches in the current frame. Zero when non-instanced.
    pub instanced_batch_count: usize,
    /// Number of active shadow cascades (1-4).
    pub cascade_count: u32,
    /// Distance to each cascade split in camera space. Unused slots are 0.
    pub cascade_splits: [f32; 4],
    /// Shadow atlas resolution (width = height) in pixels.
    pub shadow_atlas_resolution: u32,
    /// Shadow frustum half-extent in world units.
    ///
    /// Auto-computed as 20.0 unless overridden via `shadow_extent_override`.
    pub shadow_extent_world: f32,
    /// True when screen-space contact shadows are enabled in post-process settings.
    pub contact_shadow_active: bool,
}

impl Default for ShadowDebugStats {
    fn default() -> Self {
        Self {
            using_instanced_path: false,
            instanced_batch_count: 0,
            cascade_count: 0,
            cascade_splits: [0.0; 4],
            shadow_atlas_resolution: 4096,
            shadow_extent_world: 20.0,
            contact_shadow_active: false,
        }
    }
}
