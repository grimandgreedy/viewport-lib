//! The polyline item type, and the wireframe builders that produce one.
//!
//! The builders live here because a bounds wireframe *is* a polyline: they are
//! constructors for this type, used by the item types whose geometry has no
//! edges of its own to draw and by consumers drawing their own bounds.

use crate::renderer::types::items::IDENTITY_MAT4;
use crate::resources::ColourmapId;
use crate::scene::material::ItemSettings;

/// A polyline (stream tracer) item to render in the viewport.
///
/// All streamlines for one source are concatenated into a single vertex buffer.
/// `strip_lengths` records how many vertices belong to each individual streamline.
///
/// # Curve network quantities
///
/// In addition to the existing per-node scalar path (`scalars`/`colourmap_id`), this
/// item supports several curve-network quantities:
///
/// - **Per-edge scalars** (`edge_scalars`): one value per segment; rendered as a flat
///   constant colour per edge (both endpoints share the same LUT value).
/// - **Per-node colours** (`node_colours`): direct RGBA per node; takes priority over
///   scalar-driven colouring.
/// - **Per-edge colours** (`edge_colours`): direct RGBA per segment; takes priority over
///   edge scalars.
/// - **Per-node radius** (`node_radii`): per-node line width in pixels; overrides the
///   global `line_width`.
/// - **Node vectors** (`node_vectors`): world-space 3-D arrows at each node, rendered
///   automatically as `GlyphItem` arrows.
/// - **Edge vectors** (`edge_vectors`): world-space 3-D arrows at each segment midpoint,
///   also rendered as `GlyphItem` arrows.
///
/// Colour priority per segment: `node_colours`/`edge_colours` (direct) > `edge_scalars` >
/// `scalars` (per-node) > `default_colour`.
#[derive(Clone)]
#[non_exhaustive]
pub struct PolylineItem {
    /// World-space positions for all streamlines, concatenated.
    pub positions: Vec<[f32; 3]>,
    /// Per-node scalar values (same length as `positions`). Empty = no scalar colouring.
    pub scalars: Vec<f32>,
    /// Number of vertices per individual streamline strip.
    pub strip_lengths: Vec<u32>,
    /// Scalar range for LUT mapping. None = auto from min/max of `scalars` or `edge_scalars`.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. None = viridis.
    pub colourmap_id: Option<ColourmapId>,
    /// Fallback colour when no scalar or direct-colour data is provided.
    pub default_colour: crate::Colour,
    /// Global line width in pixels. Used when `node_radii` is empty.
    pub line_width: f32,
    /// Per-node direct RGBA colours. Length must match `positions`. Empty = not used.
    /// Takes priority over scalar-driven colouring when non-empty.
    pub node_colours: Vec<crate::Colour>,
    /// Per-edge scalar values. Length = total segment count across all strips (sum of
    /// `strip_lengths[i] - 1`). Used when `scalars` is empty; both endpoints of each
    /// segment share the same LUT value (flat constant colour per edge).
    pub edge_scalars: Vec<f32>,
    /// Per-edge direct RGBA colours. Length = total segment count. Takes priority over
    /// `edge_scalars` when non-empty.
    pub edge_colours: Vec<crate::Colour>,
    /// Per-node line width in pixels. Length must match `positions`. When non-empty,
    /// overrides the global `line_width`; adjacent endpoints are linearly interpolated
    /// along each segment.
    pub node_radii: Vec<f32>,
    /// Per-node world-space vectors. Length must match `positions`. When non-empty the
    /// renderer automatically generates a [`GlyphItem`](super::GlyphItem) (arrows at node positions).
    pub node_vectors: Vec<[f32; 3]>,
    /// Per-edge world-space vectors. Length = total segment count. When non-empty the
    /// renderer automatically generates a [`GlyphItem`](super::GlyphItem) (arrows at segment midpoints).
    pub edge_vectors: Vec<[f32; 3]>,
    /// Scale applied to generated arrow glyphs from `node_vectors`/`edge_vectors`.
    pub vector_scale: f32,
    /// Per-frame model matrix applied to `positions` in the vertex shader.
    /// Identity (the default) renders `positions` as world-space coordinates,
    /// preserving the historical behaviour. Set this to a translation, rotation,
    /// or scale to move a pre-uploaded polyline without rebuilding its vertex data.
    pub model: [[f32; 4]; 4],
    /// Solid, dashed, or dotted stroke. Cadence is measured in world-space arc
    /// length along the line (not screen pixels), so the pattern is view
    /// independent and stays fixed to the geometry. `Solid` (the default) leaves
    /// the historical continuous stroke unchanged. Dashing ignores any scale in
    /// `model`; the run lengths are in the units of `positions`.
    pub stroke_pattern: crate::StrokePattern,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for PolylineItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            scalars: Vec::new(),
            strip_lengths: Vec::new(),
            scalar_range: None,
            colourmap_id: None,
            default_colour: [0.9, 0.92, 0.96, 1.0].into(),
            line_width: 2.0,
            node_colours: Vec::new(),
            edge_scalars: Vec::new(),
            edge_colours: Vec::new(),
            node_radii: Vec::new(),
            node_vectors: Vec::new(),
            edge_vectors: Vec::new(),
            vector_scale: 1.0,
            model: IDENTITY_MAT4,
            stroke_pattern: crate::StrokePattern::Solid,
            settings: ItemSettings::default(),
        }
    }
}

/// Build a `PolylineItem` that draws the 12 edges of an axis-aligned bounding box.
///
/// Produces 6 strips: bottom face loop (5 pts), top face loop (5 pts), and
/// 4 vertical edges (2 pts each). Pass `colour` as RGBA in linear space.
pub fn aabb_wireframe_polyline(
    aabb: &crate::scene::aabb::Aabb,
    colour: impl Into<crate::Colour>,
) -> PolylineItem {
    let colour = colour.into();
    let mn = aabb.min;
    let mx = aabb.max;
    PolylineItem {
        positions: vec![
            // Bottom face loop
            [mn.x, mn.y, mn.z],
            [mx.x, mn.y, mn.z],
            [mx.x, mx.y, mn.z],
            [mn.x, mx.y, mn.z],
            [mn.x, mn.y, mn.z],
            // Top face loop
            [mn.x, mn.y, mx.z],
            [mx.x, mn.y, mx.z],
            [mx.x, mx.y, mx.z],
            [mn.x, mx.y, mx.z],
            [mn.x, mn.y, mx.z],
            // Vertical edges
            [mn.x, mn.y, mn.z],
            [mn.x, mn.y, mx.z],
            [mx.x, mn.y, mn.z],
            [mx.x, mn.y, mx.z],
            [mx.x, mx.y, mn.z],
            [mx.x, mx.y, mx.z],
            [mn.x, mx.y, mn.z],
            [mn.x, mx.y, mx.z],
        ],
        strip_lengths: vec![5, 5, 2, 2, 2, 2],
        default_colour: colour,
        ..Default::default()
    }
}

/// Build a `PolylineItem` that draws the 12 edges of a box from its 8 corners.
///
/// The box need not be axis-aligned, which is what separates this from
/// [`aabb_wireframe_polyline`]: pass corners already transformed into world
/// space and any orientation works. Corner order is bit 0 = x, bit 1 = y,
/// bit 2 = z, with 0 the minimum and 1 the maximum along that axis, so corner 5
/// is (max x, min y, max z). Produces 6 strips: the two z faces as closed
/// loops, then the four lateral edges. Pass `colour` as RGBA in linear space.
pub fn obb_wireframe_polyline(
    corners: &[[f32; 3]; 8],
    colour: impl Into<crate::Colour>,
) -> PolylineItem {
    let c = corners;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut strip_lengths: Vec<u32> = Vec::new();
    // Bottom face (z = min): 0, 1, 3, 2, 0
    positions.extend_from_slice(&[c[0], c[1], c[3], c[2], c[0]]);
    strip_lengths.push(5);
    // Top face (z = max): 4, 5, 7, 6, 4
    positions.extend_from_slice(&[c[4], c[5], c[7], c[6], c[4]]);
    strip_lengths.push(5);
    for (lo, hi) in [(0usize, 4usize), (1, 5), (2, 6), (3, 7)] {
        positions.extend_from_slice(&[c[lo], c[hi]]);
        strip_lengths.push(2);
    }
    PolylineItem {
        positions,
        strip_lengths,
        default_colour: colour.into(),
        line_width: 1.0,
        ..PolylineItem::default()
    }
}

/// Build a `PolylineItem` that draws three great-circle outlines for a sphere.
///
/// Produces three closed loops in the XY, XZ, and YZ planes through the given
/// centre, each sampled at `segments` points. Pass `colour` as RGBA in linear
/// space. Used as the selection outline for `ScatterShape::Sphere`.
pub fn sphere_wireframe_polyline(
    center: [f32; 3],
    radius: f32,
    segments: u32,
    colour: impl Into<crate::Colour>,
) -> PolylineItem {
    let colour = colour.into();
    let n = segments.max(8) as usize;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(3 * (n + 1));
    let two_pi = std::f32::consts::TAU;
    let cx = center[0];
    let cy = center[1];
    let cz = center[2];
    // XY plane circle
    for i in 0..=n {
        let t = i as f32 / n as f32 * two_pi;
        positions.push([cx + radius * t.cos(), cy + radius * t.sin(), cz]);
    }
    // XZ plane circle
    for i in 0..=n {
        let t = i as f32 / n as f32 * two_pi;
        positions.push([cx + radius * t.cos(), cy, cz + radius * t.sin()]);
    }
    // YZ plane circle
    for i in 0..=n {
        let t = i as f32 / n as f32 * two_pi;
        positions.push([cx, cy + radius * t.cos(), cz + radius * t.sin()]);
    }
    let strip = (n + 1) as u32;
    PolylineItem {
        positions,
        strip_lengths: vec![strip, strip, strip],
        default_colour: colour,
        ..Default::default()
    }
}

/// Per-frame reference to a pre-uploaded polyline.
///
/// Submit one of these on `SceneFrame::polyline_refs` instead of pushing the
/// full `PolylineItem` on `polylines` every frame. The renderer looks up the
/// stored GPU buffers by `id` and applies the per-frame `model` and
/// `settings` without rebuilding the segment buffer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PolylineRefItem {
    /// Handle to GPU buffers produced by
    /// [`DeviceResources::upload_polyline`](crate::resources::DeviceResources::upload_polyline)
    /// or `begin_upload_polyline`.
    pub source: crate::resources::PolylineId,
    /// Per-frame model matrix. Identity uses the polyline's own world-space
    /// positions.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, wireframe, selection, picking).
    pub settings: ItemSettings,
}

impl PolylineRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::PolylineId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}
