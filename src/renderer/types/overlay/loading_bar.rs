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
/// let bar = LoadingBarItem {
///     progress: 0.42,
///     label: Some("Building scene... 420 000 / 1 000 000".into()),
///     anchor: LoadingBarAnchor::BottomCenter,
///     ..LoadingBarItem::default()
/// };
/// ```
#[derive(Debug, Clone)]
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
    pub font: Option<crate::resources::font::FontHandle>,
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
        }
    }
}
