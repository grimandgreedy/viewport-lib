/// Horizontal alignment of a label relative to its anchor point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LabelAnchor {
    /// Text positioned to the right of the anchor (default).
    #[default]
    Leading,
    /// Text centered horizontally on the anchor.
    Center,
    /// Text positioned to the left of the anchor.
    Trailing,
}

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
/// let label = LabelItem {
///     world_anchor: Some([2.0, 3.0, 0.0]),
///     text: "Peak Pressure: 101.3 kPa".into(),
///     leader_line: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
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
    pub font: Option<crate::resources::font::FontHandle>,

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
    pub anchor_align: LabelAnchor,

    /// Pixel offset applied after anchor resolution and alignment.
    /// Useful for nudging a label away from its anchor without moving the
    /// leader line endpoint.  Default: `[0.0, 0.0]`.
    pub offset: [f32; 2],

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
            anchor_align: LabelAnchor::Leading,
            offset: [0.0, 0.0],
            opacity: 1.0,
            max_width: None,
            border_radius: 0.0,
            z_order: 0,
            occlude: false,
        }
    }
}
