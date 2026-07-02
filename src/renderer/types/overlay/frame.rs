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
