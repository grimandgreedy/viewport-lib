//! Text label overlay items.

use super::anchor::{AnchorX, AnchorY, OverlayAnchor};

/// A text label rendered as a screen-space overlay.
///
/// Anchored to a viewport corner or a projected world point with an optional
/// leader line and background box.
///
/// # Anchoring
///
/// `anchor` sets the origin the label hangs from: an [`OverlayAnchor::Viewport`]
/// corner (the default is the top-left) or an [`OverlayAnchor::World`] point that
/// is reprojected each frame.  `position` nudges the text from that origin in
/// logical pixels, and `align_x` / `align_y` place the text box on it.  A
/// world-anchored label is frustum-culled: it is not drawn when the point is
/// behind the camera or outside the viewport, and it draws a leader line when
/// `leader_line` is set.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib_types::overlay::LabelItem;
/// let label = LabelItem::new("Peak Pressure: 101.3 kPa")
///     .with_world_anchor([2.0, 3.0, 0.0])
///     .with_leader_line(true);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LabelItem {
    /// Origin the label hangs from: a viewport corner (default top-left) or a
    /// projected world point.  `position` nudges the text from here and the
    /// leader line draws for an [`OverlayAnchor::World`] anchor.
    pub anchor: OverlayAnchor,

    /// Text content to display.
    pub text: String,

    /// RGBA text colour in linear float format.
    pub colour: crate::colour::Colour,

    /// Font size in logical pixels.
    pub font_size: f32,

    /// Font to use.  `None` uses the built-in default font.
    pub font: Option<crate::overlay::font::FontHandle>,

    /// Draw a filled rectangle behind the text.
    pub background: bool,

    /// RGBA colour of the background rectangle.
    pub background_colour: crate::colour::Colour,

    /// Padding between the text and the background rectangle edge in logical
    /// pixels.  Only used when `background` is `true`.  Default: `3.0`.
    pub padding: f32,

    /// Draw a line from the projected `world_anchor` to the label text origin.
    /// Only drawn when `world_anchor` is set.
    pub leader_line: bool,

    /// RGBA colour of the leader line.
    pub leader_colour: crate::colour::Colour,

    /// Horizontal alignment of the label text relative to its anchor.
    pub align_x: AnchorX,

    /// Vertical alignment of the label text relative to its anchor. Default:
    /// `Middle`, which centres the text on the anchor. Use `Top` to place the
    /// top edge of the text at the anchor when laying out screen-space UI.
    pub align_y: AnchorY,

    /// Gap in logical pixels between the anchor and the near edge of the text,
    /// applied in the anchor-facing direction: `Left` text is pushed this far
    /// right of the anchor, `Right` text this far left. `Middle` is
    /// unaffected. Default: `6.0`, which keeps text clear of a leader line.
    /// Set to `0.0` for anchor-exact placement when laying out screen-space UI.
    pub anchor_padding: f32,

    /// Placement relative to the resolved `anchor` origin, in logical pixels,
    /// applied after anchor resolution and alignment. Nudges the label away from
    /// its anchor without moving the leader line endpoint.  Default: `[0.0, 0.0]`.
    pub position: [f32; 2],

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

    /// Reserved for depth-based occlusion.  Currently a no-op: the label
    /// renders the same whether this is `true` or `false`.
    pub occlude: bool,

    /// When set, this label is clipped to the mask shape whose `clip_mask_id`
    /// matches this value: glyph and background fragments outside the mask are
    /// discarded, so text scrolled inside a region is contained. The mask can be
    /// any overlay shape (rect, rounded rect, circle, ...), and masks may nest.
    /// `None` (the default) draws the label unclipped, as does a missing mask.
    pub clip_id: Option<u32>,
    /// Stacked drop shadows and contours drawn behind this item, first entry
    /// furthest back. Up to [`OVERLAY_MAX_SHADOW_LAYERS`] are honoured.
    ///
    /// Empty by default, which draws none. A contour that keeps the item legible
    /// over an unpredictable background is one
    /// [`ShadowLayer::outline`] entry; a soft drop shadow is a blurred entry.
    ///
    /// [`OVERLAY_MAX_SHADOW_LAYERS`]: crate::overlay::OVERLAY_MAX_SHADOW_LAYERS
    /// [`ShadowLayer::outline`]: crate::overlay::ShadowLayer::outline
    pub shadows: Vec<crate::overlay::ShadowLayer>,
    /// Rotation around the text-box centre in radians. Positive rotates
    /// counter-clockwise in math coordinates, which reads as clockwise on
    /// screen because the Y axis points down. `0.0` keeps the default
    /// orientation.
    ///
    /// The extent box stays axis-aligned: `align_x` / `align_y` place the
    /// unrotated box on the anchor and the text turns inside it, matching
    /// [`OverlayShapeItem::rotation`].
    ///
    /// [`OverlayShapeItem::rotation`]: crate::overlay::OverlayShapeItem::rotation
    pub rotation: f32,

    /// Point to rotate around, in logical pixels measured from the text-box
    /// centre. `[0.0, 0.0]` (default) rotates around the centre. Positive X is
    /// right, positive Y is down, matching the screen-space axes.
    ///
    /// The box is the laid-out text, so it moves when the text, font, or wrap
    /// width changes. To turn about a fixed corner instead, measure the text
    /// and offset the pivot by half its extent.
    pub rotation_pivot: [f32; 2],
}

impl Default for LabelItem {
    fn default() -> Self {
        Self {
            anchor: OverlayAnchor::default(),
            text: String::new(),
            colour: [1.0, 1.0, 1.0, 1.0].into(),
            font_size: 14.0,
            font: None,
            background: false,
            background_colour: [0.0, 0.0, 0.0, 0.55].into(),
            padding: 3.0,
            leader_line: false,
            leader_colour: [1.0, 1.0, 1.0, 0.6].into(),
            align_x: AnchorX::Left,
            align_y: AnchorY::Middle,
            anchor_padding: 6.0,
            position: [0.0, 0.0],
            opacity: 1.0,
            max_width: None,
            border_radius: 0.0,
            z_order: 0,
            occlude: false,
            clip_id: None,
            shadows: Vec::new(),
            rotation: 0.0,
            rotation_pivot: [0.0, 0.0],
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

    /// Set the origin the label hangs from (a viewport corner or a world point).
    pub fn with_anchor(mut self, anchor: OverlayAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Pin the label to a world-space position, reprojected each frame. Sugar
    /// for `with_anchor(OverlayAnchor::World(pos))`.
    pub fn with_world_anchor(mut self, pos: [f32; 3]) -> Self {
        self.anchor = OverlayAnchor::World(pos);
        self
    }

    /// Pin the label to a fixed screen position in logical pixels from the
    /// top-left. Sugar for the default viewport-top-left anchor with `position`
    /// set to `pos`.
    pub fn with_screen_anchor(mut self, pos: [f32; 2]) -> Self {
        self.anchor = OverlayAnchor::Viewport {
            x: AnchorX::Left,
            y: AnchorY::Top,
        };
        self.position = pos;
        self
    }

    /// Set the text colour.
    pub fn with_colour(mut self, colour: impl Into<crate::colour::Colour>) -> Self {
        self.colour = colour.into();
        self
    }

    /// Set the font size in logical pixels.
    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.font_size = font_size;
        self
    }

    /// Set the font. Without this the built-in default font is used.
    pub fn with_font(mut self, font: crate::overlay::font::FontHandle) -> Self {
        self.font = Some(font);
        self
    }

    /// Draw a filled rectangle behind the text.
    pub fn with_background(mut self, background: bool) -> Self {
        self.background = background;
        self
    }

    /// Set the background rectangle colour.
    pub fn with_background_colour(mut self, colour: impl Into<crate::colour::Colour>) -> Self {
        self.background_colour = colour.into();
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
    pub fn with_leader_colour(mut self, colour: impl Into<crate::colour::Colour>) -> Self {
        self.leader_colour = colour.into();
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

    /// Set the placement relative to the anchor, applied after anchor resolution
    /// and alignment.
    pub fn with_position(mut self, position: [f32; 2]) -> Self {
        self.position = position;
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
    /// [`OverlayShapeItem::with_clip_mask`](crate::overlay::OverlayShapeItem::with_clip_mask)). Fragments outside the mask are
    /// discarded, so text scrolled inside a region is contained.
    pub fn with_clip(mut self, clip_id: u32) -> Self {
        self.clip_id = Some(clip_id);
        self
    }

    /// Set the stacked shadow layers drawn behind this item.
    pub fn with_shadows(mut self, shadows: Vec<crate::overlay::ShadowLayer>) -> Self {
        self.shadows = shadows;
        self
    }

    /// Add one shadow layer, in front of any already set.
    pub fn with_shadow(mut self, shadow: crate::overlay::ShadowLayer) -> Self {
        self.shadows.push(shadow);
        self
    }

    /// Add a contour of `width` logical pixels in `colour` behind this item.
    /// Shorthand for pushing a [`ShadowLayer::outline`].
    ///
    /// [`ShadowLayer::outline`]: crate::overlay::ShadowLayer::outline
    pub fn with_outline(mut self, colour: impl Into<crate::colour::Colour>, width: f32) -> Self {
        self.shadows
            .push(crate::overlay::ShadowLayer::outline(colour, width));
        self
    }

    /// Set the rotation in radians about the text-box centre.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }

    /// Set the point to rotate about, in logical pixels from the text-box centre.
    pub fn with_rotation_pivot(mut self, pivot: [f32; 2]) -> Self {
        self.rotation_pivot = pivot;
        self
    }
}
