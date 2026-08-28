use super::anchor::{AnchorX, AnchorY};

/// A text label rendered as a screen-space overlay.
///
/// Anchored to a world-space or screen-space position with optional leader line
/// and background box.
///
/// # Anchoring
///
/// Set `world_anchor` to pin the label to a 3D position that is reprojected
/// each frame.  Set `screen_anchor` for a fixed screen position in logical
/// pixels from the top-left corner.  When both are set, `screen_anchor` takes
/// precedence.  World-anchored labels are frustum-culled: they are not drawn
/// when the anchor is behind the camera or outside the viewport.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib::LabelItem;
/// let label = LabelItem::new("Peak Pressure: 101.3 kPa")
///     .with_world_anchor([2.0, 3.0, 0.0])
///     .with_leader_line(true);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LabelItem {
    /// World-space anchor.  Projected to screen by the renderer each frame.
    /// Set `screen_anchor` instead for fixed screen positions.
    pub world_anchor: Option<[f32; 3]>,

    /// Screen-space anchor in logical pixels from top-left.
    /// Takes precedence over `world_anchor` when both are set.
    pub screen_anchor: Option<[f32; 2]>,

    /// Text content to display.
    pub text: String,

    /// RGBA text colour in linear float format.
    pub colour: [f32; 4],

    /// Font size in logical pixels.
    pub font_size: f32,

    /// Font to use.  `None` uses the built-in default font.
    pub font: Option<crate::resources::overlay::font::FontHandle>,

    /// Draw a filled rectangle behind the text.
    pub background: bool,

    /// RGBA colour of the background rectangle.
    pub background_colour: [f32; 4],

    /// Padding between the text and the background rectangle edge in logical
    /// pixels.  Only used when `background` is `true`.  Default: `3.0`.
    pub padding: f32,

    /// Draw a line from the projected `world_anchor` to the label text origin.
    /// Only drawn when `world_anchor` is set.
    pub leader_line: bool,

    /// RGBA colour of the leader line.
    pub leader_colour: [f32; 4],

    /// Horizontal alignment of the label text relative to its anchor.
    pub align_x: AnchorX,

    /// Vertical alignment of the label text relative to its anchor. Default:
    /// `Middle`, which centers the text on the anchor. Use `Top` to place the
    /// top edge of the text at the anchor when laying out screen-space UI.
    pub align_y: AnchorY,

    /// Gap in logical pixels between the anchor and the near edge of the text,
    /// applied in the anchor-facing direction: `Left` text is pushed this far
    /// right of the anchor, `Right` text this far left. `Middle` is
    /// unaffected. Default: `6.0`, which keeps text clear of a leader line.
    /// Set to `0.0` for anchor-exact placement when laying out screen-space UI.
    pub anchor_padding: f32,

    /// Pixel offset from the anchor, applied after anchor resolution and
    /// alignment. Useful for nudging a label away from its anchor without moving
    /// the leader line endpoint.  Default: `[0.0, 0.0]`.
    pub anchor_offset: [f32; 2],

    /// Overall opacity multiplier applied to text, background, and leader
    /// line colours.  Range 0.0 (invisible) to 1.0 (fully opaque).
    pub opacity: f32,

    /// Maximum text width in logical pixels.  When set, text that exceeds
    /// this width is wrapped to multiple lines.  `None` disables wrapping.
    pub max_width: Option<f32>,

    /// Corner radius of the background rectangle in logical pixels.
    /// Only used when `background` is `true`.  Default: `0.0` (sharp corners).
    pub border_radius: f32,

    /// Explicit draw order.  Labels with lower values are drawn first
    /// (further back).  Labels with equal `z_order` are drawn in list order.
    pub z_order: i32,

    /// Reserved for depth-based occlusion.  Not implemented yet: when `true`
    /// the label is still rendered; behaviour will be defined in a follow-up.
    pub occlude: bool,

    /// When set, this label is clipped to the mask shape whose `clip_mask_id`
    /// matches this value: glyph and background fragments outside the mask are
    /// discarded, so text scrolled inside a region is contained. The mask can be
    /// any overlay shape (rect, rounded rect, circle, ...), and masks may nest.
    /// `None` (the default) draws the label unclipped, as does a missing mask.
    pub clip_id: Option<u32>,
}

impl Default for LabelItem {
    fn default() -> Self {
        Self {
            world_anchor: None,
            screen_anchor: None,
            text: String::new(),
            colour: [1.0, 1.0, 1.0, 1.0],
            font_size: 14.0,
            font: None,
            background: false,
            background_colour: [0.0, 0.0, 0.0, 0.55],
            padding: 3.0,
            leader_line: false,
            leader_colour: [1.0, 1.0, 1.0, 0.6],
            align_x: AnchorX::Left,
            align_y: AnchorY::Middle,
            anchor_padding: 6.0,
            anchor_offset: [0.0, 0.0],
            opacity: 1.0,
            max_width: None,
            border_radius: 0.0,
            z_order: 0,
            occlude: false,
            clip_id: None,
        }
    }
}

impl LabelItem {
    /// Create a label with the given `text`. All other fields take their
    /// defaults; set them with the `with_*` methods below.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    /// Pin the label to a world-space position, reprojected each frame.
    pub fn with_world_anchor(mut self, anchor: [f32; 3]) -> Self {
        self.world_anchor = Some(anchor);
        self
    }

    /// Pin the label to a fixed screen position in logical pixels from
    /// top-left. Takes precedence over the world anchor when both are set.
    pub fn with_screen_anchor(mut self, anchor: [f32; 2]) -> Self {
        self.screen_anchor = Some(anchor);
        self
    }

    /// Set the text colour.
    pub fn with_colour(mut self, colour: [f32; 4]) -> Self {
        self.colour = colour;
        self
    }

    /// Set the font size in logical pixels.
    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.font_size = font_size;
        self
    }

    /// Set the font. Without this the built-in default font is used.
    pub fn with_font(mut self, font: crate::resources::overlay::font::FontHandle) -> Self {
        self.font = Some(font);
        self
    }

    /// Draw a filled rectangle behind the text.
    pub fn with_background(mut self, background: bool) -> Self {
        self.background = background;
        self
    }

    /// Set the background rectangle colour.
    pub fn with_background_colour(mut self, colour: [f32; 4]) -> Self {
        self.background_colour = colour;
        self
    }

    /// Set the padding between the text and the background rectangle edge.
    pub fn with_padding(mut self, padding: f32) -> Self {
        self.padding = padding;
        self
    }

    /// Draw a leader line from the projected world anchor to the text origin.
    pub fn with_leader_line(mut self, leader_line: bool) -> Self {
        self.leader_line = leader_line;
        self
    }

    /// Set the leader line colour.
    pub fn with_leader_colour(mut self, colour: [f32; 4]) -> Self {
        self.leader_colour = colour;
        self
    }

    /// Set the horizontal alignment of the text relative to its anchor.
    pub fn with_align_x(mut self, align_x: AnchorX) -> Self {
        self.align_x = align_x;
        self
    }

    /// Set the vertical alignment of the text relative to its anchor. Defaults
    /// to `Middle`; pass `Top` to place the top edge of the text at the anchor.
    pub fn with_align_y(mut self, align_y: AnchorY) -> Self {
        self.align_y = align_y;
        self
    }

    /// Set the gap between the anchor and the near edge of the text. Defaults to
    /// `6.0`; pass `0.0` for anchor-exact placement in screen-space UI.
    pub fn with_anchor_padding(mut self, anchor_padding: f32) -> Self {
        self.anchor_padding = anchor_padding;
        self
    }

    /// Set the pixel offset from the anchor, applied after anchor resolution and
    /// alignment.
    pub fn with_anchor_offset(mut self, anchor_offset: [f32; 2]) -> Self {
        self.anchor_offset = anchor_offset;
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    /// Set the maximum text width in logical pixels. Text wider than this wraps
    /// to multiple lines.
    pub fn with_max_width(mut self, max_width: f32) -> Self {
        self.max_width = Some(max_width);
        self
    }

    /// Set the corner radius of the background rectangle in logical pixels.
    pub fn with_border_radius(mut self, border_radius: f32) -> Self {
        self.border_radius = border_radius;
        self
    }

    /// Set the draw order. Lower values render first (further back).
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Set the depth-occlusion flag (reserved; not yet implemented).
    pub fn with_occlude(mut self, occlude: bool) -> Self {
        self.occlude = occlude;
        self
    }

    /// Clip the label to the mask shape with this id (registered via
    /// [`OverlayShapeItem::with_clip_mask`]). Fragments outside the mask are
    /// discarded, so text scrolled inside a region is contained.
    pub fn with_clip(mut self, clip_id: u32) -> Self {
        self.clip_id = Some(clip_id);
        self
    }
}
