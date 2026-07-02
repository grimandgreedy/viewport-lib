/// A two-point measurement overlay that displays the distance between two
/// world-space positions, with a distance readout at the segment midpoint.
///
/// Both endpoints are projected each frame by the renderer; the item culls
/// cleanly when both endpoints are behind the camera.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::RulerItem;
/// let ruler = RulerItem {
///     start: [0.0, 0.0, 0.0],
///     end: [2.5, 0.0, 0.0],
///     colour: [1.0, 1.0, 1.0, 1.0],
///     label_format: Some("{:.2} m".into()),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RulerItem {
    /// World-space start endpoint.
    pub start: [f32; 3],
    /// World-space end endpoint.
    pub end: [f32; 3],
    /// RGBA colour for the ruler line and end caps. Default: white.
    pub colour: [f32; 4],
    /// Line thickness in screen pixels. Default: `1.5`.
    pub line_width_px: f32,
    /// Font for the distance label. `None` = built-in default.
    pub font: Option<crate::resources::FontHandle>,
    /// Font size for the distance label in logical pixels. Default: `13.0`.
    pub font_size: f32,
    /// RGBA colour for the distance label text. Default: white.
    pub label_colour: [f32; 4],
    /// Format string for the distance value using Rust `format!` syntax.
    ///
    /// The `{}` placeholder is replaced with the computed distance.
    /// Accepts precision specifiers like `"{:.3}"` or unit suffixes like
    /// `"{:.2} m"`. Default (`None`): `"{:.3}"` (3 decimal places).
    pub label_format: Option<String>,
    /// Draw small perpendicular tick marks at each endpoint. Default: `true`.
    pub end_caps: bool,
}

impl Default for RulerItem {
    fn default() -> Self {
        Self {
            start: [0.0; 3],
            end: [1.0, 0.0, 0.0],
            colour: [1.0, 1.0, 1.0, 1.0],
            line_width_px: 1.5,
            font: None,
            font_size: 13.0,
            label_colour: [1.0, 1.0, 1.0, 1.0],
            label_format: None,
            end_caps: true,
        }
    }
}
