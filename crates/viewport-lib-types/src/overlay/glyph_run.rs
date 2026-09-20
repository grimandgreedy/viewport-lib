//! Positioned glyph runs: shaped, laid-out text ready for the overlay atlas.

use crate::overlay::OverlayClip;

/// One glyph placed at an explicit position within a [`GlyphRunItem`].
///
/// `glyph_id` is an index into the font's glyph table, not a Unicode codepoint.
/// It is what a shaping engine emits after applying the font's substitution and
/// positioning tables, so a ligature or a joined Arabic form is a single glyph
/// id that may correspond to no single character. viewport-lib rasterises the
/// glyph straight from this id; it never sees the source text.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct PositionedGlyph {
    /// Index into the run's font glyph table.
    pub glyph_id: u16,
    /// Pen position in logical pixels, relative to the run origin. This is the
    /// position before the glyph's own bitmap bearing, which the draw path adds,
    /// matching how [`LabelItem`] places glyphs.
    ///
    /// [`LabelItem`]: crate::overlay::LabelItem
    pub x: f32,
    /// Vertical pen position in logical pixels, relative to the run origin.
    pub y: f32,
}

impl PositionedGlyph {
    /// A glyph placed at `(x, y)` logical pixels from the run origin.
    pub fn new(glyph_id: u16, x: f32, y: f32) -> Self {
        Self { glyph_id, x, y }
    }
}

/// A run of pre-positioned glyphs drawn as a screen-space overlay.
///
/// This is the low-level counterpart to [`LabelItem`]. A `LabelItem` takes a
/// `String` and lays it out internally (fontdue, one glyph per codepoint,
/// left to right). A `GlyphRunItem` takes glyphs that are already positioned and
/// only rasterises and draws them, so the caller decides the layout. That is the
/// hook a shaping / bidi engine uses: it shapes a run of text into positioned
/// glyph ids for a font and submits them here, keeping the shaper itself out of
/// viewport-lib.
///
/// One run carries one font. A line that spans several fonts (script fallback,
/// or mixing a text font with an icon font) is submitted as several runs sharing
/// a baseline, one per font. Glyph positions are relative to the resolved
/// origin, so moving a whole run is a change to `anchoring` or `position`.
///
/// The run's colour is `style.fill`, with `glyph_tints` multiplying over it per
/// glyph.
///
/// [`LabelItem`]: crate::overlay::LabelItem
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GlyphRunItem {
    /// Which font the glyph ids index into, and the size the bitmaps are
    /// rasterised at. The glyph positions themselves come from `glyphs`.
    pub text_style: crate::overlay::TextStyle,

    /// Where the item hangs from and which point of its own box lands there.
    ///
    /// `transform.translate` is a nudge from the resolved origin, and a world
    /// origin behind the camera or off screen culls the item for the frame.
    pub anchoring: crate::overlay::OverlayAnchoring,
    /// Translate, rotate, and scale, in logical pixels and radians.
    ///
    /// `translate` is the nudge from the resolved `anchor` origin, so with the
    /// default anchor and alignment it is the absolute screen placement.
    /// Rotation turns the item inside its extent box, which stays
    /// axis-aligned, so `anchoring.align` places the unrotated box and the
    /// content turns within it. See [`OverlayTransform`] for how an item's
    /// transform composes with the transform of a retained group containing
    /// it.
    pub transform: crate::overlay::OverlayTransform,
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
    /// it. See [`crate::overlay::OverlayAnimations`] for why the channel list is what it is.
    pub animations: Option<Box<crate::overlay::OverlayAnimations>>,
    /// What this item is clipped to: an axis-aligned box, a mask shape, or
    /// both. The default clips nothing.
    pub clip: OverlayClip,

    /// Positioned glyphs, in draw order.
    pub glyphs: Vec<PositionedGlyph>,

    /// Optional per-glyph colour multiplier over `style.fill`, parallel to
    /// `glyphs`. When non-empty, glyph `i` is multiplied by `glyph_tints[i]`;
    /// glyphs past the end of this list (and every glyph when it is empty) draw
    /// the fill unmodified. Use it for runs where glyphs differ in colour, such
    /// as syntax highlighting.
    ///
    /// It is a multiplier and not a colour, and it behaves like the item's own
    /// `style.tint`: it never reaches a shadow layer, so a run's contour stays
    /// one colour however many the glyphs are.
    pub glyph_tints: Vec<[f32; 4]>,

    /// Explicit draw order. Runs with lower values are drawn first (further
    /// back). Shares the cross-family z-order space with labels and shapes.
    pub z_order: i32,
}

impl Default for GlyphRunItem {
    fn default() -> Self {
        Self {
            text_style: crate::overlay::TextStyle::default(),
            anchoring: crate::overlay::OverlayAnchoring::default(),
            transform: crate::overlay::OverlayTransform::IDENTITY,
            style: crate::overlay::OverlayStyle::default(),
            animations: None,
            clip: OverlayClip::default(),
            glyphs: Vec::new(),
            glyph_tints: Vec::new(),
            z_order: 0,
        }
    }
}

impl GlyphRunItem {
    /// Create a run from a list of positioned glyphs. All other fields take their
    /// defaults; set them with the `with_*` methods below.
    pub fn new(glyphs: impl Into<Vec<PositionedGlyph>>) -> Self {
        Self {
            glyphs: glyphs.into(),
            ..Default::default()
        }
    }

    /// Set the font the glyph ids index into. Without this the built-in default
    /// font is used.
    pub fn with_font(mut self, font: crate::overlay::font::FontHandle) -> Self {
        self.text_style.font = Some(font);
        self
    }

    /// Set the font size in logical pixels.
    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.text_style.size = font_size;
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

    /// Set the origin the run hangs from (a viewport corner or a world point).
    pub fn with_anchor(mut self, anchor: crate::overlay::OverlayOrigin) -> Self {
        self.anchoring.origin = anchor;
        self
    }

    /// Pin the run to a world-space position, reprojected each frame. Sugar for
    /// `with_anchor(OverlayOrigin::World(pos))`.
    pub fn with_world_anchor(mut self, pos: [f32; 3]) -> Self {
        self.anchoring.origin = crate::overlay::OverlayOrigin::World(pos);
        self
    }

    /// Set the placement in logical pixels relative to the resolved anchor
    /// origin. With the default anchor this is the run's screen position from
    /// the viewport top-left.
    pub fn with_position(mut self, position: [f32; 2]) -> Self {
        self.transform.translate = position;
        self
    }

    /// Set how the run's glyph-extent box aligns onto the resolved anchor origin.
    pub fn with_align(mut self, align: crate::overlay::Alignment) -> Self {
        self.anchoring.align = align;
        self
    }

    /// Replace the positioned glyphs.
    pub fn with_glyphs(mut self, glyphs: impl Into<Vec<PositionedGlyph>>) -> Self {
        self.glyphs = glyphs.into();
        self
    }

    /// Set the per-glyph colour multipliers over `style.fill`, parallel to the
    /// glyphs. Glyphs past the end of this list draw the fill unmodified.
    pub fn with_glyph_tints(mut self, glyph_tints: impl IntoIterator<Item = [f32; 4]>) -> Self {
        self.glyph_tints = glyph_tints.into_iter().collect();
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.style.opacity = opacity;
        self
    }

    /// Set the draw order. Lower values render first (further back).
    pub fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Clip the run to the mask shape with this id (registered via
    /// [`OverlayShapeItem::with_clip_mask`]). Fragments outside the mask are
    /// discarded.
    ///
    /// [`OverlayShapeItem::with_clip_mask`]: crate::overlay::OverlayShapeItem::with_clip_mask
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_builders() {
        let run = GlyphRunItem::default();
        assert!(run.text_style.font.is_none());
        assert_eq!(run.text_style.size, 14.0);
        assert_eq!(run.transform.translate, [0.0, 0.0]);
        assert!(run.glyphs.is_empty());
        assert!(run.glyph_tints.is_empty());
        assert_eq!(run.style.opacity, 1.0);
        assert_eq!(run.z_order, 0);
        assert!(run.clip.mask.is_none());

        let glyphs = vec![
            PositionedGlyph::new(4, 0.0, 0.0),
            PositionedGlyph::new(9, 8.5, 0.0),
        ];
        let run = GlyphRunItem::new(glyphs.clone())
            .with_font_size(20.0)
            .with_position([10.0, 12.0])
            .with_fill(crate::overlay::OverlayFill::Solid(
                [1.0, 0.0, 0.0, 1.0].into(),
            ))
            .with_glyph_tints(vec![[0.0, 1.0, 0.0, 1.0]])
            .with_opacity(0.5)
            .with_z_order(3)
            .with_clip(7);

        assert_eq!(run.glyphs, glyphs);
        assert_eq!(run.text_style.size, 20.0);
        assert_eq!(run.transform.translate, [10.0, 12.0]);
        assert_eq!(
            run.style.fill,
            crate::overlay::OverlayFill::Solid([1.0, 0.0, 0.0, 1.0].into())
        );
        assert_eq!(run.glyph_tints, vec![[0.0, 1.0, 0.0, 1.0]]);
        assert_eq!(run.style.opacity, 0.5);
        assert_eq!(run.z_order, 3);
        assert_eq!(run.clip.mask, Some(7));
    }

    #[test]
    fn extent_spans_the_authored_pen_positions() {
        let run = GlyphRunItem::new(vec![
            PositionedGlyph::new(4, 10.0, -4.0),
            PositionedGlyph::new(9, 30.0, 6.0),
        ]);
        let (min, size) = run.extent().unwrap();
        assert_eq!(min, [10.0, -4.0]);
        assert_eq!(size, [20.0, 10.0]);
        assert!(GlyphRunItem::default().extent().is_none());
    }

    #[test]
    fn resolve_top_left_places_the_extent_box() {
        let glyphs = vec![
            PositionedGlyph::new(4, 10.0, 0.0),
            PositionedGlyph::new(9, 30.0, 10.0),
        ];
        // Default anchor and alignment: the box stays where it was authored,
        // offset by the position, and needs no camera.
        let run = GlyphRunItem::new(glyphs.clone()).with_position([5.0, 7.0]);
        let tl = run
            .resolve_top_left([800.0, 600.0], &glam::Mat4::IDENTITY, &glam::Mat4::IDENTITY)
            .unwrap();
        assert_eq!(tl, [15.0, 7.0]);

        // Anchored and aligned bottom-right: the box's bottom-right corner sits
        // on the viewport's, so its top-left is back by the extent.
        let run = GlyphRunItem::new(glyphs)
            .with_anchor(crate::overlay::OverlayOrigin::Viewport(
                crate::overlay::Alignment::new(
                    crate::overlay::AnchorX::Right,
                    crate::overlay::AnchorY::Bottom,
                ),
            ))
            .with_align(crate::overlay::Alignment::new(
                crate::overlay::AnchorX::Right,
                crate::overlay::AnchorY::Bottom,
            ));
        let tl = run
            .resolve_top_left([800.0, 600.0], &glam::Mat4::IDENTITY, &glam::Mat4::IDENTITY)
            .unwrap();
        assert_eq!(tl, [800.0 - 20.0 + 10.0, 600.0 - 10.0]);
    }
}

impl GlyphRunItem {
    /// The extent box of the authored glyph positions, as `[min_x, min_y]` and
    /// `[width, height]` in logical pixels. This is the box alignment shifts and
    /// the box the run turns inside, measured from the pen positions rather than
    /// from the rasterised glyph bitmaps, so it matches what the renderer uses.
    /// An empty run has no extent and returns `None`.
    pub fn extent(&self) -> Option<([f32; 2], [f32; 2])> {
        let (first, rest) = self.glyphs.split_first()?;
        let (mut min_x, mut min_y) = (first.x, first.y);
        let (mut max_x, mut max_y) = (first.x, first.y);
        for g in rest {
            min_x = min_x.min(g.x);
            min_y = min_y.min(g.y);
            max_x = max_x.max(g.x);
            max_y = max_y.max(g.y);
        }
        Some(([min_x, min_y], [max_x - min_x, max_y - min_y]))
    }

    /// Resolve the top-left pixel of the run's extent box for a frame: the
    /// origin, plus `position`, shifted by `anchoring.align` for
    /// that box. Returns `None` when a `World` anchor projects behind the camera
    /// or off-screen, which is the frame the run is skipped on, and for a run
    /// with no glyphs.
    ///
    /// Glyph positions are authored relative to the run origin, which is this
    /// value minus the extent box's own `[min_x, min_y]` from [`Self::extent`].
    /// The box returned is the unrotated one, as with the other overlay items.
    pub fn resolve_top_left(
        &self,
        viewport_size: [f32; 2],
        view: &glam::Mat4,
        proj: &glam::Mat4,
    ) -> Option<[f32; 2]> {
        let origin = crate::overlay::resolve_anchor_origin(
            &self.anchoring.origin,
            viewport_size,
            view,
            proj,
        )?;
        let (min, size) = self.extent()?;
        Some([
            origin[0]
                + self.transform.translate[0]
                + self.anchoring.align.x.align_shift(size[0])
                + min[0],
            origin[1]
                + self.transform.translate[1]
                + self.anchoring.align.y.align_shift(size[1])
                + min[1],
        ])
    }

    /// Set the transform: translate, rotate, scale, and pivot at once.
    pub fn with_transform(mut self, transform: crate::overlay::OverlayTransform) -> Self {
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
    pub fn with_animations(mut self, animations: crate::overlay::OverlayAnimations) -> Self {
        self.animations = Some(Box::new(animations));
        self
    }

    /// Clip to an axis-aligned box in logical pixels `[x0, y0, x1, y1]`.
    pub fn with_clip_rect(mut self, clip_rect: [f32; 4]) -> Self {
        self.clip.rect = Some(clip_rect);
        self
    }
}
