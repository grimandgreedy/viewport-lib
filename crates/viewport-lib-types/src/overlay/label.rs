//! Text label overlay items.

use super::anchor::{
    Alignment, AnchorX, AnchorY, OverlayAnchoring, OverlayOrigin, resolve_anchor_origin,
};
use super::animation::OverlayAnimations;
use super::clip::OverlayClip;
use super::transform::OverlayTransform;

/// A text label rendered as a screen-space overlay.
///
/// Anchored to a viewport corner or a projected world point with an optional
/// leader line and background box. The text colour is `style.fill`, which can
/// be a gradient across the text box as well as a flat colour.
///
/// # Anchoring
///
/// `anchoring.origin` sets the point the label hangs from: an
/// [`OverlayOrigin::Viewport`]
/// corner (the default is the top-left) or an [`OverlayOrigin::World`] point
/// that is reprojected each frame.  `position` nudges the text from that origin in
/// logical pixels, and `anchoring.align` places the text box on it.  A
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
    /// Where the item hangs from and which point of its own box lands there.
    ///
    /// `transform.translate` is a nudge from the resolved origin, and a world
    /// origin behind the camera or off screen culls the item for the frame.
    pub anchoring: OverlayAnchoring,
    /// Translate, rotate, and scale, in logical pixels and radians.
    ///
    /// `translate` is the nudge from the resolved `anchor` origin, so with the
    /// default anchor and alignment it is the absolute screen placement.
    /// Rotation turns the item inside its extent box, which stays
    /// axis-aligned, so `anchoring.align` places the unrotated box and the
    /// content turns within it. See [`OverlayTransform`] for how an item's
    /// transform composes with the transform of a retained group containing
    /// it.
    pub transform: OverlayTransform,
    /// Baked appearance: fill, shadow layers, and backdrop effects.
    ///
    /// Shared across the overlay item types, so a field can be present and
    /// inert here. Ask
    /// [`OverlayStyleSupport`](crate::overlay::OverlayStyleSupport) for what
    /// this family draws.
    pub style: crate::overlay::OverlayStyle,
    /// Animation tracks resolved each frame against `OverlayFrame::time`.
    ///
    /// Boxed and `None` for a static item: the track block is several times
    /// the size of the rest of the item, so only items that animate pay for
    /// it. See [`OverlayAnimations`] for why the channel list is what it is.
    pub animations: Option<Box<OverlayAnimations>>,
    /// What this item is clipped to: an axis-aligned box, a mask shape, or
    /// both. The default clips nothing.
    pub clip: OverlayClip,

    /// Text content to display.
    pub text: String,

    /// Which font to draw the text in, and at what size.
    pub text_style: crate::overlay::TextStyle,

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

    /// Gap in logical pixels between the anchor and the near edge of the text,
    /// applied in the anchor-facing direction: `Left` text is pushed this far
    /// right of the anchor, `Right` text this far left. `Middle` is
    /// unaffected. Default: `6.0`, which keeps text clear of a leader line.
    /// Set to `0.0` for anchor-exact placement when laying out screen-space UI.
    pub anchor_padding: f32,

    /// Maximum text width in logical pixels.  When set, text that exceeds
    /// this width is wrapped to multiple lines.  `None` disables wrapping.
    pub max_width: Option<f32>,

    /// Corner radius of the background rectangle in logical pixels.
    /// Only used when `background` is `true`.  Default: `0.0` (sharp corners).
    pub border_radius: f32,

    /// Explicit draw order.  Labels with lower values are drawn first
    /// (further back).  Labels with equal `z_order` are drawn in list order.
    pub z_order: i32,
}

impl Default for LabelItem {
    fn default() -> Self {
        Self {
            anchoring: crate::overlay::OverlayAnchoring::default().with_align(
                crate::overlay::Alignment::new(AnchorX::Left, AnchorY::Middle),
            ),
            text: String::new(),
            text_style: crate::overlay::TextStyle::default(),
            background: false,
            background_colour: [0.0, 0.0, 0.0, 0.55].into(),
            padding: 3.0,
            leader_line: false,
            leader_colour: [1.0, 1.0, 1.0, 0.6].into(),
            anchor_padding: 6.0,
            transform: OverlayTransform::IDENTITY,
            style: crate::overlay::OverlayStyle::default(),
            animations: None,
            clip: OverlayClip::default(),
            max_width: None,
            border_radius: 0.0,
            z_order: 0,
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
    pub fn with_anchor(mut self, anchor: OverlayOrigin) -> Self {
        self.anchoring.origin = anchor;
        self
    }

    /// Pin the label to a world-space position, reprojected each frame. Sugar
    /// for `with_anchor(OverlayOrigin::World(pos))`.
    pub fn with_world_anchor(mut self, pos: [f32; 3]) -> Self {
        self.anchoring.origin = OverlayOrigin::World(pos);
        self
    }

    /// Pin the label to a fixed screen position in logical pixels from the
    /// top-left. Sugar for the default viewport-top-left anchor with `position`
    /// set to `pos`.
    pub fn with_screen_anchor(mut self, pos: [f32; 2]) -> Self {
        self.anchoring.origin =
            OverlayOrigin::Viewport(Alignment::new(AnchorX::Left, AnchorY::Top));
        self.transform.translate = pos;
        self
    }

    /// Set what the glyphs are filled with: a colour, or a gradient across the
    /// text box. This is the text colour, and the only source of it.
    pub fn with_fill(mut self, fill: crate::overlay::OverlayFill) -> Self {
        self.style.fill = fill;
        self
    }

    /// Set the text colour. Sugar for a solid [`with_fill`](Self::with_fill).
    pub fn with_colour(mut self, colour: impl Into<crate::colour::Colour>) -> Self {
        self.style.fill = crate::overlay::OverlayFill::Solid(colour.into());
        self
    }

    /// Set the whole text style: the font and its size.
    pub fn with_text_style(mut self, text_style: crate::overlay::TextStyle) -> Self {
        self.text_style = text_style;
        self
    }

    /// Set the font size in logical pixels.
    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.text_style.size = font_size;
        self
    }

    /// Set the font. Without this the built-in default font is used.
    pub fn with_font(mut self, font: crate::overlay::font::FontHandle) -> Self {
        self.text_style.font = Some(font);
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
        self.anchoring.align.x = align_x;
        self
    }

    /// Set the vertical alignment of the text relative to its anchor. Defaults
    /// to `Middle`; pass `Top` to place the top edge of the text at the anchor.
    pub fn with_align_y(mut self, align_y: AnchorY) -> Self {
        self.anchoring.align.y = align_y;
        self
    }

    /// Set both alignments at once: which point of the text box lands on the
    /// resolved origin.
    pub fn with_align(mut self, align: Alignment) -> Self {
        self.anchoring.align = align;
        self
    }

    /// Set the gap between the anchor and the near edge of the text. Defaults to
    /// `6.0`; pass `0.0` for anchor-exact placement in screen-space UI.
    pub fn with_anchor_padding(mut self, anchor_padding: f32) -> Self {
        self.anchor_padding = anchor_padding;
        self
    }

    /// Resolve the top-left pixel of the laid-out text for a frame: the
    /// origin, plus `position`, shifted by `anchoring.align` for a
    /// text box of `size`, plus `anchor_padding` on the horizontal edge the
    /// text is aligned to. Returns `None` when a `World` anchor projects behind
    /// the camera or off-screen, which is the frame the label is skipped on.
    ///
    /// `size` is the measured text, in logical pixels: `[width, height]` from
    /// `DeviceResources::measure_overlay_text`, or from the wrapped measure
    /// when `max_width` is set. The renderer lays the text out and then places
    /// it the same way, so a backing shape or a leader line built on this lands
    /// where the text does.
    ///
    /// The box returned is the unrotated one. `transform.rotation` turns the
    /// text inside it about the pivot, so a rotated label's glyphs leave this
    /// box while its placement does not.
    pub fn resolve_top_left(
        &self,
        size: [f32; 2],
        viewport_size: [f32; 2],
        view: &glam::Mat4,
        proj: &glam::Mat4,
    ) -> Option<[f32; 2]> {
        let origin = resolve_anchor_origin(&self.anchoring.origin, viewport_size, view, proj)?;
        // The horizontal rule is the shared align shift plus the anchor gap,
        // which pushes the text away from the anchor on whichever side it sits.
        let shift_x = match self.anchoring.align.x {
            AnchorX::Left => self.anchor_padding,
            AnchorX::Middle => -size[0] * 0.5,
            AnchorX::Right => -size[0] - self.anchor_padding,
        };
        Some([
            origin[0] + self.transform.translate[0] + shift_x,
            origin[1] + self.transform.translate[1] + self.anchoring.align.y.align_shift(size[1]),
        ])
    }

    /// Set the placement relative to the anchor, applied after anchor resolution
    /// and alignment.
    pub fn with_position(mut self, position: [f32; 2]) -> Self {
        self.transform.translate = position;
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.style.opacity = opacity;
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

    /// Clip the label to the mask shape with this id (registered via
    /// [`OverlayShapeItem::with_clip_mask`](crate::overlay::OverlayShapeItem::with_clip_mask)). Fragments outside the mask are
    /// discarded, so text scrolled inside a region is contained.
    pub fn with_clip(mut self, clip_id: u32) -> Self {
        self.clip.mask = Some(clip_id);
        self
    }

    /// Set the stacked shadow layers drawn behind this item.
    pub fn with_shadows(mut self, shadows: Vec<crate::overlay::ShadowLayer>) -> Self {
        self.style.shadows = shadows;
        self
    }

    /// Add one shadow layer, in front of any already set.
    pub fn with_shadow(mut self, shadow: crate::overlay::ShadowLayer) -> Self {
        self.style.shadows.push(shadow);
        self
    }

    /// Add a contour of `width` logical pixels in `colour` behind this item.
    /// Shorthand for pushing a [`ShadowLayer::outline`].
    ///
    /// [`ShadowLayer::outline`]: crate::overlay::ShadowLayer::outline
    pub fn with_outline(mut self, colour: impl Into<crate::colour::Colour>, width: f32) -> Self {
        self.style
            .shadows
            .push(crate::overlay::ShadowLayer::outline(colour, width));
        self
    }

    /// Set the rotation in radians about the text-box centre.
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.transform.rotation = radians;
        self
    }

    /// Set the point to rotate about, in logical pixels from the text-box centre.
    pub fn with_rotation_pivot(mut self, pivot: [f32; 2]) -> Self {
        self.transform.pivot = pivot;
        self
    }
}

impl LabelItem {
    /// Set the transform: translate, rotate, scale, and pivot at once.
    pub fn with_transform(mut self, transform: OverlayTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Set the uniform scale about the transform pivot.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.transform.scale = scale;
        self
    }

    /// Set the per-frame colour multiplier (identity `[1, 1, 1, 1]`).
    pub fn with_tint(mut self, tint: [f32; 4]) -> Self {
        self.style.tint = tint;
        self
    }

    /// Set the animation tracks.
    pub fn with_animations(mut self, animations: OverlayAnimations) -> Self {
        self.animations = Some(Box::new(animations));
        self
    }

    /// Clip to an axis-aligned box in logical pixels `[x0, y0, x1, y1]`.
    pub fn with_clip_rect(mut self, clip_rect: [f32; 4]) -> Self {
        self.clip.rect = Some(clip_rect);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A screen-space label with the anchor gap zeroed resolves to its position,
    /// with no camera involved, so plain UI placement is exact.
    #[test]
    fn resolve_top_left_default_is_the_position() {
        let label = LabelItem::new("hello")
            .with_screen_anchor([40.0, 25.0])
            .with_align_x(AnchorX::Left)
            .with_align_y(AnchorY::Top)
            .with_anchor_padding(0.0);
        let tl = label
            .resolve_top_left(
                [60.0, 14.0],
                [800.0, 600.0],
                &glam::Mat4::IDENTITY,
                &glam::Mat4::IDENTITY,
            )
            .unwrap();
        assert_eq!(tl, [40.0, 25.0]);
    }

    /// Alignment shifts the measured box onto the anchor, and the anchor gap
    /// pushes the text away from the anchor on the side it is aligned to.
    #[test]
    fn resolve_top_left_shifts_by_alignment_and_anchor_padding() {
        let base = LabelItem::new("hello").with_anchor(OverlayOrigin::Viewport(Alignment::new(
            AnchorX::Right,
            AnchorY::Bottom,
        )));
        let size = [60.0, 14.0];

        let right = base
            .clone()
            .with_align_x(AnchorX::Right)
            .with_align_y(AnchorY::Bottom)
            .with_anchor_padding(6.0)
            .resolve_top_left(
                size,
                [800.0, 600.0],
                &glam::Mat4::IDENTITY,
                &glam::Mat4::IDENTITY,
            )
            .unwrap();
        assert_eq!(right, [800.0 - 60.0 - 6.0, 600.0 - 14.0]);

        let centred = base
            .with_align_x(AnchorX::Middle)
            .with_align_y(AnchorY::Middle)
            .with_anchor_padding(6.0)
            .resolve_top_left(
                size,
                [800.0, 600.0],
                &glam::Mat4::IDENTITY,
                &glam::Mat4::IDENTITY,
            )
            .unwrap();
        // Middle alignment centres the box on the anchor and ignores the gap,
        // which has no side to push away from.
        assert_eq!(centred, [800.0 - 30.0, 600.0 - 7.0]);
    }

    /// A world anchor behind the camera resolves to nothing, which is the frame
    /// the label is skipped on.
    #[test]
    fn resolve_top_left_culls_behind_the_camera() {
        let label = LabelItem::new("hello").with_world_anchor([0.0, 0.0, 10.0]);
        // A projection that sends everything to a negative w: nothing survives.
        let behind = glam::Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, -1.0, //
            0.0, 0.0, 0.0, 0.0,
        ]);
        assert!(
            label
                .resolve_top_left([60.0, 14.0], [800.0, 600.0], &glam::Mat4::IDENTITY, &behind)
                .is_none()
        );
    }
}
