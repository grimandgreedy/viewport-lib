/// Anchor position for a [`LoadingBarItem`] overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadingBarAnchor {
    /// Anchored at the top center of the viewport.
    TopCenter,
    /// Anchored at the center of the viewport.
    Center,
    /// Anchored at the bottom center of the viewport (default).
    #[default]
    BottomCenter,
}

/// A progress bar drawn over the viewport in screen space.
///
/// Render via [`OverlayFrame::loading_bars`].
///
/// ```no_run
/// use viewport_lib::{LoadingBarItem, LoadingBarAnchor};
/// let bar = LoadingBarItem::new(0.42)
///     .with_label("Building scene... 420 000 / 1 000 000")
///     .with_anchor(LoadingBarAnchor::BottomCenter);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoadingBarItem {
    /// Progress fraction in [0.0, 1.0].
    pub progress: f32,
    /// Optional label displayed above (or below for `TopCenter`) the bar.
    pub label: Option<String>,
    /// Viewport anchor for the bar.
    pub anchor: LoadingBarAnchor,
    /// Bar width in logical pixels.
    pub width_px: f32,
    /// Bar height in logical pixels.
    pub height_px: f32,
    /// Distance from the anchored viewport edge in logical pixels.
    pub margin_px: f32,
    /// Background (unfilled) colour.
    pub background_colour: [f32; 4],
    /// Fill (progress) colour.
    pub fill_colour: [f32; 4],
    /// Label text colour.
    pub label_colour: [f32; 4],
    /// Font size for the label in logical pixels.
    pub font_size: f32,
    /// Corner radius of the bar rectangles in logical pixels.
    pub corner_radius: f32,
    /// Font for the label text. `None` uses the built-in default font.
    pub font: Option<crate::resources::overlay::font::FontHandle>,
    /// Draw order relative to other overlay items. Lower values render first
    /// (further back).
    pub z_order: i32,
}

impl Default for LoadingBarItem {
    fn default() -> Self {
        Self {
            progress: 0.0,
            label: None,
            anchor: LoadingBarAnchor::default(),
            width_px: 300.0,
            height_px: 16.0,
            margin_px: 24.0,
            background_colour: [0.12, 0.12, 0.12, 0.88],
            fill_colour: [0.22, 0.60, 1.0, 1.0],
            label_colour: [1.0, 1.0, 1.0, 1.0],
            font_size: 13.0,
            corner_radius: 4.0,
            font: None,
            z_order: 0,
        }
    }
}

impl LoadingBarItem {
    /// Create a loading bar showing `progress` (a fraction in [0.0, 1.0]). All
    /// other fields take their defaults; set them with the `with_*` methods
    /// below.
    pub fn new(progress: f32) -> Self {
        Self {
            progress,
            ..Default::default()
        }
    }

    /// Set the label displayed alongside the bar.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the viewport anchor for the bar.
    pub fn with_anchor(mut self, anchor: LoadingBarAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Set the bar width in logical pixels.
    pub fn with_width(mut self, width: f32) -> Self {
        self.width_px = width;
        self
    }

    /// Set the bar height in logical pixels.
    pub fn with_height(mut self, height: f32) -> Self {
        self.height_px = height;
        self
    }

    /// Set the distance from the anchored viewport edge in logical pixels.
    pub fn with_margin(mut self, margin: f32) -> Self {
        self.margin_px = margin;
        self
    }

    /// Set the background (unfilled) colour.
    pub fn with_background_colour(mut self, colour: [f32; 4]) -> Self {
        self.background_colour = colour;
        self
    }

    /// Set the fill (progress) colour.
    pub fn with_fill_colour(mut self, colour: [f32; 4]) -> Self {
        self.fill_colour = colour;
        self
    }

    /// Set the label text colour.
    pub fn with_label_colour(mut self, colour: [f32; 4]) -> Self {
        self.label_colour = colour;
        self
    }

    /// Set the label font size in logical pixels.
    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.font_size = font_size;
        self
    }

    /// Set the corner radius of the bar rectangles in logical pixels.
    pub fn with_corner_radius(mut self, corner_radius: f32) -> Self {
        self.corner_radius = corner_radius;
        self
    }

    /// Set the font for the label text. Without this the built-in default font
    /// is used.
    pub fn with_font(mut self, font: crate::resources::overlay::font::FontHandle) -> Self {
        self.font = Some(font);
        self
    }

    /// Set the draw order relative to other overlay items. Lower values render
    /// first (further back).
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }
}
