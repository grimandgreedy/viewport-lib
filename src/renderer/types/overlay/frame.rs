use crate::renderer::types::*;

/// Semantic overlays rendered after post-processing: labels, scalar bars,
/// rulers, screen-space images, and loading bars.
///
/// This frame section is the right place for any visual element that belongs
/// in front of the 3D scene and must not be affected by tone-mapping or bloom.
#[derive(Debug, Clone, Default)]
pub struct OverlayFrame {
    /// Current time in seconds, used to resolve [`OverlayAnimation`] on
    /// shapes. Use the same epoch as the `start_time` values in your
    /// animations (e.g. seconds since app launch). Default: `0.0`.
    pub time: f64,
    /// SDF-based shapes rendered before labels. Supports rounded rects,
    /// circles, ellipses, and capsules with anti-aliased edges and borders.
    pub shapes: Vec<OverlayShapeItem>,
    /// Solid filled rectangles rendered before labels. Useful for panel
    /// backgrounds, scrims, and full-screen fades.
    pub rects: Vec<OverlayRectItem>,
    /// Text labels anchored to world-space or screen-space positions.
    pub labels: Vec<LabelItem>,
    /// Pre-positioned glyph runs. The low-level text path: the caller supplies
    /// glyph ids and positions (typically from a shaping engine) and the
    /// renderer only rasterizes and draws them. `labels` is the string-in,
    /// laid-out-here path; this is the glyphs-in, drawn-as-given path.
    pub glyph_runs: Vec<GlyphRunItem>,
    /// Colour-legend (scalar bar) overlays.
    pub scalar_bars: Vec<ScalarBarItem>,
    /// Two-point distance measurement overlays.
    pub rulers: Vec<RulerItem>,
    /// Pixel images composited over the viewport in screen space.
    pub images: Vec<OverlayImageItem>,
    /// Progress bar overlays (loading indicators, progress feedback).
    pub loading_bars: Vec<LoadingBarItem>,
    /// Stroked polylines. Rendered through the same pipeline as overlay
    /// rects and labels; share their z-order space.
    pub polylines: Vec<OverlayPolylineItem>,
}

impl OverlayFrame {
    /// Whether any overlay item carries a non-zero `z_order`. When every item is
    /// at the default `0`, the renderer keeps its fixed family draw order and
    /// skips building the cross-family ordering list, so the common case pays
    /// nothing for the feature.
    pub(crate) fn uses_nonzero_z_order(&self) -> bool {
        self.shapes.iter().any(|i| i.z_order != 0)
            || self.rects.iter().any(|i| i.z_order != 0)
            || self.labels.iter().any(|i| i.z_order != 0)
            || self.glyph_runs.iter().any(|i| i.z_order != 0)
            || self.scalar_bars.iter().any(|i| i.z_order != 0)
            || self.rulers.iter().any(|i| i.z_order != 0)
            || self.images.iter().any(|i| i.z_order != 0)
            || self.loading_bars.iter().any(|i| i.z_order != 0)
            || self.polylines.iter().any(|i| i.z_order != 0)
    }
}
