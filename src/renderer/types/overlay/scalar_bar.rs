/// Corner of the viewport where a [`ScalarBarItem`] is anchored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalarBarAnchor {
    /// Top-left corner of the viewport.
    TopLeft,
    /// Top-right corner of the viewport.
    TopRight,
    /// Bottom-left corner of the viewport.
    BottomLeft,
    /// Bottom-right corner of the viewport (default).
    #[default]
    BottomRight,
}

/// Long-axis orientation of a [`ScalarBarItem`] gradient strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalarBarOrientation {
    /// Gradient runs top (max) to bottom (min).  Default.
    #[default]
    Vertical,
    /// Gradient runs left (min) to right (max).
    Horizontal,
}

/// A colour-legend (scalar bar) rendered as a screen-space overlay.
///
/// References an already-uploaded [`crate::resources::ColourmapId`] and draws a
/// gradient strip with evenly-spaced tick labels directly in the overlay pass,
/// without requiring any application-side painting.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::{ScalarBarItem, ScalarBarAnchor, ScalarBarOrientation};
/// let bar = ScalarBarItem {
///     colourmap_id: viewport_lib::ColourmapId(0),
///     scalar_min: 0.0,
///     scalar_max: 1.0,
///     title: Some("Height (m)".into()),
///     anchor: ScalarBarAnchor::BottomRight,
///     orientation: ScalarBarOrientation::Vertical,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ScalarBarItem {
    /// Colourmap to sample for the gradient strip.
    pub colourmap_id: crate::resources::ColourmapId,

    /// Scalar value at the low end (bottom or left) of the gradient.
    pub scalar_min: f32,

    /// Scalar value at the high end (top or right) of the gradient.
    pub scalar_max: f32,

    /// Optional title drawn above the gradient strip.
    pub title: Option<String>,

    /// Viewport corner to anchor the bar to.
    pub anchor: ScalarBarAnchor,

    /// Long-axis orientation of the gradient strip.
    pub orientation: ScalarBarOrientation,

    /// Short-axis size of the gradient strip in logical pixels.  Default: `20.0`.
    pub bar_width_px: f32,

    /// Long-axis size of the gradient strip in logical pixels.  Default: `200.0`.
    pub bar_length_px: f32,

    /// Distance from the viewport edge in logical pixels.  Default: `16.0`.
    pub margin_px: f32,

    /// Font to use for tick labels and title.  `None` uses the built-in default.
    pub font: Option<crate::resources::overlay::font::FontHandle>,

    /// Font size for tick labels and title in logical pixels.  Default: `12.0`.
    pub font_size: f32,

    /// RGBA colour for tick labels and title.  Default: white.
    pub label_colour: [f32; 4],

    /// Number of evenly-spaced labelled ticks (including min and max).  Default: `5`.
    pub tick_count: u32,

    /// RGBA background box colour (including alpha).
    ///
    /// Default: semi-transparent black `[0.0, 0.0, 0.0, 0.63]`.
    pub background_colour: [f32; 4],

    /// Reverse the value direction of the gradient.
    ///
    /// When `false` (default): vertical bars run max-at-top / min-at-bottom;
    /// horizontal bars run min-at-left / max-at-right.
    /// When `true` the direction is flipped for both orientations.
    pub ticks_reversed: bool,

    /// Font size used exclusively for the title text.
    ///
    /// `None` (default) falls back to `font_size`.
    pub title_font_size: Option<f32>,
}

impl Default for ScalarBarItem {
    fn default() -> Self {
        Self {
            colourmap_id: crate::resources::ColourmapId(0),
            scalar_min: 0.0,
            scalar_max: 1.0,
            title: None,
            anchor: ScalarBarAnchor::BottomRight,
            orientation: ScalarBarOrientation::Vertical,
            bar_width_px: 20.0,
            bar_length_px: 200.0,
            margin_px: 16.0,
            font: None,
            font_size: 12.0,
            label_colour: [1.0, 1.0, 1.0, 1.0],
            tick_count: 5,
            background_colour: [0.0, 0.0, 0.0, 0.63],
            ticks_reversed: false,
            title_font_size: None,
        }
    }
}
