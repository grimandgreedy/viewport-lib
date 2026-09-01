//! Retained overlay geometry: compiled vertex buffers keyed by
//! [`OverlayGeometryId`], plus the per-draw instance struct that carries a
//! group's per-frame translate, opacity, and clip to the overlay shaders.

/// A compiled overlay group's GPU geometry.
///
/// Holds the text-pipeline vertex stream (`OverlayTextVertex`: polylines and
/// vector fills) in a persistent buffer that lives until the group is freed. The
/// SDF-shape stream is added later; a group may then carry either or both.
pub(crate) struct CompiledOverlay {
    /// Text-pipeline vertices in local logical-pixel space, uploaded once.
    pub vertex_buf: crate::gpu::Buffer,
    /// Number of vertices in `vertex_buf`.
    pub vertex_count: u32,
    /// GPU bytes charged for this entry (the vertex buffer size).
    pub bytes: u64,
}

/// Per-draw instance data shared by immediate and retained overlay draws.
///
/// Slot 0 is the identity used by every immediate draw (no translate, full
/// opacity, no clip); each retained group gets its own slot. The overlay vertex
/// shader reads the slot named by `@builtin(instance_index)`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OverlayInstance {
    /// Logical-pixel offset applied to the vertex position before pixel-to-NDC.
    pub translate: [f32; 2],
    /// Opacity multiplier applied to the fragment alpha.
    pub opacity: f32,
    pub _pad: f32,
    /// Outer clip bounding box in framebuffer pixels `[x0, y0, x1, y1]`; all-zero
    /// means no clip.
    pub clip_rect: [f32; 4],
}

impl OverlayInstance {
    /// The identity instance immediate draws use: no offset, full opacity, no clip.
    pub const IDENTITY: OverlayInstance = OverlayInstance {
        translate: [0.0, 0.0],
        opacity: 1.0,
        _pad: 0.0,
        clip_rect: [0.0, 0.0, 0.0, 0.0],
    };
}
