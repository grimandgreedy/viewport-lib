//! Retained overlay geometry: compiled vertex buffers keyed by
//! [`OverlayGeometryId`], plus the per-draw instance struct that carries a
//! group's per-frame translate, opacity, and clip to the overlay shaders.

/// A compiled overlay group's GPU geometry.
///
/// Holds the text-pipeline vertex stream (`OverlayTextVertex`: polylines, vector
/// fills, and glyph runs) in a persistent buffer that lives until the group is
/// freed. The SDF-shape stream is added later; a group may then carry either or
/// both.
pub(crate) struct CompiledOverlay {
    /// Text-pipeline vertices in local logical-pixel space, uploaded once.
    pub vertex_buf: crate::gpu::Buffer,
    /// Number of vertices in `vertex_buf`.
    pub vertex_count: u32,
    /// GPU bytes charged for this entry (both vertex buffers plus shadows).
    pub bytes: u64,
    /// Analytic SDF shape vertices (`OverlayShapeVertex`), drawn through the shape
    /// pipeline. `None` when the group has no SDF shapes.
    pub shape_vertex_buf: Option<crate::gpu::Buffer>,
    /// Number of vertices in `shape_vertex_buf`.
    pub shape_vertex_count: u32,
    /// The stacked shadow-layer storage buffer the SDF shapes reference. Kept
    /// alive alongside `shape_vertex_buf`; always at least one (dummy) entry when
    /// the shape stream is present.
    pub shadow_buf: Option<crate::gpu::Buffer>,
    /// Retained source items for re-emission, present only when the group carries
    /// glyphs. Glyph geometry bakes atlas UVs at a physical size, so it goes stale
    /// when the atlas grows or `pixels_per_point` changes; the renderer re-emits
    /// from this source on that event. `None` for polyline/vector-only groups,
    /// whose geometry is viewport-independent and never invalidates.
    pub source: Option<CompiledSource>,
    /// The anchor a compiled label resolves each frame. `Some` only for a group
    /// compiled from a `LabelItem`: the renderer resolves this to a screen origin
    /// (a viewport corner, or a world point projected through the camera) and adds
    /// it to the per-frame translate, and skips the group when a world anchor is
    /// culled. `None` for every other group, whose translate is taken as-is.
    pub anchor: Option<viewport_lib_types::overlay::OverlayAnchor>,
}

/// The source items a glyph-bearing group retains so its geometry can be
/// re-emitted when its baked atlas UVs go stale.
#[derive(Clone)]
pub(crate) struct CompiledSource {
    pub polylines: Vec<viewport_lib_types::overlay::OverlayPolylineItem>,
    pub vector_shapes: Vec<viewport_lib_types::overlay::OverlayShapeItem>,
    pub glyph_runs: Vec<viewport_lib_types::overlay::GlyphRunItem>,
    /// Labels retained for re-emission. A label lays its text out at compile time
    /// (glyphs, background box, and leader line, all on the text stream); a DPI or
    /// atlas change re-lays it out from here.
    pub labels: Vec<viewport_lib_types::overlay::LabelItem>,
    /// The atlas growth version the current geometry baked UVs against.
    pub baked_atlas_version: u64,
    /// The `pixels_per_point` the current geometry baked glyphs at.
    pub baked_ppp: f32,
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
