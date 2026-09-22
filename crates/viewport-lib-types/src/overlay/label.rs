//! Text label overlay items.

use super::anchor::{
    Alignment, AnchorX, AnchorY, OverlayAnchoring, OverlayOrigin, resolve_anchor_origin,
};
use super::animation::OverlayAnimations;
use super::clip::OverlayClip;
use super::transform::OverlayTransform;

/// A text label rendered as a screen-space overlay.
///
/// Anchored to a viewport corner or a projected world point. The text colour is
/// `style.fill`, which can be a gradient across the text box as well as a flat
/// colour.
///
/// A label draws text and nothing else. A backing panel behind it is an
/// [`OverlayShapeItem`](crate::overlay::OverlayShapeItem) with the same
/// anchoring and the measured text plus padding for its size, which gets the
/// SDF pipeline's rounded corners, gradients, textures and shadow layers
/// instead of a flat quad in the text stream. Shapes draw under text at equal
/// `z_order`, so the two need no ordering work.
///
/// # Anchoring
///
/// `anchoring.origin` sets the point the label hangs from: an
/// [`OverlayOrigin::Viewport`]
/// corner (the default is the top-left) or an [`OverlayOrigin::World`] point
/// that is reprojected each frame.  `position` nudges the text from that origin in
/// logical pixels, and `anchoring.align` places the text box on it.  A
/// world-anchored label is frustum-culled: it is not drawn when the point is
/// behind the camera or outside the viewport.
///
/// # Examples
///
/// ```rust
/// # use viewport_lib_types::overlay::LabelItem;
/// let label = LabelItem::new("Peak Pressure: 101.3 kPa")
///     .with_world_anchor([2.0, 3.0, 0.0])
///     .with_colour([1.0, 0.9, 0.4, 1.0]);
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

    /// Maximum text width in logical pixels.  When set, text that exceeds
    /// this width is wrapped to multiple lines.  `None` disables wrapping.
    pub max_width: Option<f32>,

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
            transform: OverlayTransform::IDENTITY,
            style: crate::overlay::OverlayStyle::default(),
            animations: None,
            clip: OverlayClip::default(),
            max_width: None,
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

    /// Set the whole baked appearance at once: fill, shadow layers, backdrop,
    /// tint and opacity. The escape hatch for any cell without a dedicated
    /// builder.
    pub fn with_style(mut self, style: crate::overlay::OverlayStyle) -> Self {
        self.style = style;
        self
    }

    /// Set the stacked inner (inset) shadow layers, drawn over the item and
    /// eroding inward from its boundary.
    pub fn with_inner_shadows(mut self, shadows: Vec<crate::overlay::ShadowLayer>) -> Self {
        self.style.inner_shadows = shadows;
        self
    }

    /// Add one inner shadow layer, over any already set.
    pub fn with_inner_shadow(mut self, shadow: crate::overlay::ShadowLayer) -> Self {
        self.style.inner_shadows.push(shadow);
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

    /// Resolve the top-left pixel of the laid-out text for a frame: the
    /// origin, plus `position`, shifted by `anchoring.align` for a
    /// text box of `size`. Returns `None` when a `World` anchor projects behind
    /// the camera or off-screen, which is the frame the label is skipped on.
    ///
    /// `size` is the measured text, in logical pixels: `[width, height]` from
    /// `DeviceResources::measure_overlay_text`, or from the wrapped measure
    /// when `max_width` is set. The renderer lays the text out and then places
    /// it the same way, so a backing shape built on this lands where the text
    /// does.
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
        Some([
            origin[0] + self.transform.translate[0] + self.anchoring.align.x.align_shift(size[0]),
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

    /// Add an outline: a band of `width` logical pixels on the item's edge,
    /// placed by `mode`. A width of `0.0` adds nothing.
    ///
    /// An outline is a shadow layer with no blur, so this pushes one (or two,
    /// for [`OutlineMode::Centre`]) onto the style's shadow lists, and costs a
    /// layer out of [`OVERLAY_MAX_SHADOW_LAYERS`] per list.
    ///
    /// [`OutlineMode::Centre`]: crate::overlay::OutlineMode::Centre
    /// [`OVERLAY_MAX_SHADOW_LAYERS`]: crate::overlay::OVERLAY_MAX_SHADOW_LAYERS
    pub fn with_outline(
        mut self,
        colour: impl Into<crate::colour::Colour>,
        width: f32,
        mode: crate::overlay::OutlineMode,
    ) -> Self {
        if width <= 0.0 {
            return self;
        }
        let colour = colour.into();
        let band = |spread: f32| {
            crate::overlay::ShadowLayer::new(colour, 0.0, [0.0, 0.0]).with_spread(spread)
        };
        match mode {
            crate::overlay::OutlineMode::Inset => self.style.inner_shadows.push(band(width)),
            crate::overlay::OutlineMode::Outer => self.style.shadows.push(band(width)),
            crate::overlay::OutlineMode::Centre => {
                self.style.inner_shadows.push(band(width * 0.5));
                self.style.shadows.push(band(width * 0.5));
            }
        }
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
            .with_align_y(AnchorY::Top);
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
    fn resolve_top_left_shifts_by_alignment() {
        let base = LabelItem::new("hello").with_anchor(OverlayOrigin::Viewport(Alignment::new(
            AnchorX::Right,
            AnchorY::Bottom,
        )));
        let size = [60.0, 14.0];

        let right = base
            .clone()
            .with_align_x(AnchorX::Right)
            .with_align_y(AnchorY::Bottom)
            .resolve_top_left(
                size,
                [800.0, 600.0],
                &glam::Mat4::IDENTITY,
                &glam::Mat4::IDENTITY,
            )
            .unwrap();
        assert_eq!(right, [800.0 - 60.0, 600.0 - 14.0]);

        let centred = base
            .with_align_x(AnchorX::Middle)
            .with_align_y(AnchorY::Middle)
            .resolve_top_left(
                size,
                [800.0, 600.0],
                &glam::Mat4::IDENTITY,
                &glam::Mat4::IDENTITY,
            )
            .unwrap();
        // Middle alignment centres the box on the origin.
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
