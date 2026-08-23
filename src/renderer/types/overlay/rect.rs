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
#[non_exhaustive]
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
    /// When set, this rect is clipped to the mask shape whose `clip_mask_id`
    /// matches this value: fragments outside the mask are discarded, so the rect
    /// is contained within a scroll well or masked panel. The mask can be any
    /// overlay shape, and masks may nest. `None` (the default) draws it
    /// unclipped, as does a missing mask.
    pub clip_id: Option<u32>,
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
            clip_id: None,
        }
    }
}

impl OverlayRectItem {
    /// Create a rect at `position` (top-left, logical pixels) with `size`
    /// (width, height) and fill `colour`. All other fields take their defaults;
    /// set them with the `with_*` methods below.
    pub fn new(position: [f32; 2], size: [f32; 2], colour: [f32; 4]) -> Self {
        Self {
            position,
            size,
            colour,
            ..Default::default()
        }
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set the corner radius in logical pixels. `0.0` produces sharp corners.
    pub fn with_corner_radius(mut self, corner_radius: f32) -> Self {
        self.corner_radius = corner_radius;
        self
    }

    /// Set the border colour and width. A width of `0.0` disables the border.
    pub fn with_border(mut self, colour: [f32; 4], width: f32) -> Self {
        self.border_colour = colour;
        self.border_width = width;
        self
    }

    /// Set the draw order. Lower values render first (further back).
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Clip the rect to the mask shape with this id (registered via
    /// [`crate::OverlayShapeItem::with_clip_mask`]). Fragments outside the mask
    /// are discarded.
    pub fn with_clip(mut self, clip_id: u32) -> Self {
        self.clip_id = Some(clip_id);
        self
    }
}
