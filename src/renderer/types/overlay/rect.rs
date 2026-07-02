/// A solid filled rectangle rendered in screen space.
///
/// Position and size are in logical pixels from the top-left corner of the
/// viewport. Renders after post-processing, in the same pass as other overlay
/// items, before labels so it can serve as a background panel.
///
/// The border is drawn as a slightly expanded rect behind the fill, so
/// `border_width` pixels of it are visible on each side of the fill rect.
/// Set `border_width` to `0.0` for no border.
#[derive(Debug, Clone)]
pub struct OverlayRectItem {
    /// Top-left position in logical pixels from the viewport top-left.
    pub position: [f32; 2],
    /// Width and height in logical pixels.
    pub size: [f32; 2],
    /// RGBA fill colour in linear float format.
    pub colour: [f32; 4],
    /// Overall opacity multiplier applied to both fill and border. Range 0.0-1.0.
    pub opacity: f32,
    /// Corner radius in logical pixels. `0.0` produces sharp corners.
    pub corner_radius: f32,
    /// RGBA border colour in linear float format.
    pub border_colour: [f32; 4],
    /// Border thickness in logical pixels. `0.0` disables the border.
    pub border_width: f32,
    /// Draw order relative to other rects. Lower values render first (further back).
    pub z_order: i32,
}

impl Default for OverlayRectItem {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            size: [100.0, 100.0],
            colour: [0.0, 0.0, 0.0, 0.55],
            opacity: 1.0,
            corner_radius: 0.0,
            border_colour: [1.0, 1.0, 1.0, 1.0],
            border_width: 0.0,
            z_order: 0,
        }
    }
}
