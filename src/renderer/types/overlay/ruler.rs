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
/// let ruler = RulerItem::new([0.0, 0.0, 0.0], [2.5, 0.0, 0.0])
///     .with_colour([1.0, 1.0, 1.0, 1.0])
///     .with_label_format("{:.2} m");
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
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

impl RulerItem {
    /// Create a ruler between the world-space `start` and `end` points. All
    /// other fields take their defaults; set them with the `with_*` methods
    /// below.
    pub fn new(start: [f32; 3], end: [f32; 3]) -> Self {
        Self {
            start,
            end,
            ..Default::default()
        }
    }

    /// Set the colour of the ruler line and end caps.
    pub fn with_colour(mut self, colour: [f32; 4]) -> Self {
        self.colour = colour;
        self
    }

    /// Set the line thickness in screen pixels.
    pub fn with_line_width(mut self, line_width: f32) -> Self {
        self.line_width_px = line_width;
        self
    }

    /// Set the font for the distance label. Without this the built-in default
    /// font is used.
    pub fn with_font(mut self, font: crate::resources::FontHandle) -> Self {
        self.font = Some(font);
        self
    }

    /// Set the font size for the distance label in logical pixels.
    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.font_size = font_size;
        self
    }

    /// Set the distance label text colour.
    pub fn with_label_colour(mut self, colour: [f32; 4]) -> Self {
        self.label_colour = colour;
        self
    }

    /// Set the `format!`-style string used to render the distance value.
    pub fn with_label_format(mut self, label_format: impl Into<String>) -> Self {
        self.label_format = Some(label_format.into());
        self
    }

    /// Set whether to draw perpendicular tick marks at each endpoint.
    pub fn with_end_caps(mut self, end_caps: bool) -> Self {
        self.end_caps = end_caps;
        self
    }
}
