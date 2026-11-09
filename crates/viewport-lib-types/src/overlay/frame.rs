//! The per-frame collection of overlay items a consumer submits for a frame.

use crate::overlay::*;

/// Screen-space overlays rendered after post-processing: shapes, labels, glyph
/// runs, and polylines.
///
/// This frame section is the right place for any visual element that belongs
/// in front of the 3D scene and must not be affected by tone-mapping or bloom.
///
/// Items paint in `z_order` order, lowest first. At an equal `z_order`
/// analytic shapes paint under everything else (labels, glyph runs, polylines
/// and vector shapes), and each list paints in its own order. The order is kept between items that overlap on screen; items
/// that do not overlap may be drawn in any order, which is what lets the
/// renderer batch them across `z_order` values and families.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct OverlayFrame {
    /// Current time in seconds, used to resolve animation tracks on
    /// shapes. Use the same epoch as the `start_time` values in your
    /// animations (e.g. seconds since app launch). Default: `0.0`.
    pub time: f64,
    /// SDF-based shapes rendered before labels. Supports rounded rects,
    /// circles, ellipses, and capsules with anti-aliased edges and borders.
    /// A solid rectangle is `OverlayShape::Rect` with an `OverlayFill::Solid`.
    pub shapes: Vec<OverlayShapeItem>,
    /// Text labels anchored to world-space or screen-space positions.
    pub labels: Vec<LabelItem>,
    /// Pre-positioned glyph runs. The low-level text path: the caller supplies
    /// glyph ids and positions (typically from a shaping engine) and the
    /// renderer only rasterises and draws them. `labels` is the string-in,
    /// laid-out-here path; this is the glyphs-in, drawn-as-given path.
    pub glyph_runs: Vec<GlyphRunItem>,
    /// Stroked polylines. Rendered through the same pipeline as labels; share
    /// their z-order space.
    pub polylines: Vec<OverlayPolylineItem>,
    /// Retained overlay groups: geometry compiled once with
    /// `compile_overlay_geometry` and re-drawn from cached buffers, each carrying
    /// its per-frame translate, opacity, z-order, and clip. Empty by default and
    /// costs nothing when unused; this is the opt-in path for a UI that draws many
    /// mostly-static overlays without re-tessellating them every frame.
    pub retained: Vec<RetainedOverlay>,
    /// Layers: runs of the items above composited as one, with an opacity,
    /// blend, colour filter, blur, drop shadow or backdrop applied to the
    /// whole. Empty by default. Build one with [`layer_mark`](Self::layer_mark)
    /// and [`push_layer`](Self::push_layer).
    pub layers: Vec<OverlayLayer>,
}
