/// One glyph placed at an explicit position within a [`GlyphRunItem`].
///
/// `glyph_id` is an index into the font's glyph table, not a Unicode codepoint.
/// It is what a shaping engine emits after applying the font's substitution and
/// positioning tables, so a ligature or a joined Arabic form is a single glyph
/// id that may correspond to no single character. viewport-lib rasterizes the
/// glyph straight from this id; it never sees the source text.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PositionedGlyph {
    /// Index into the run's font glyph table.
    pub glyph_id: u16,
    /// Pen position in logical pixels, relative to the run origin. This is the
    /// position before the glyph's own bitmap bearing, which the draw path adds,
    /// matching how [`LabelItem`] places glyphs.
    ///
    /// [`LabelItem`]: crate::renderer::types::LabelItem
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
/// only rasterizes and draws them, so the caller decides the layout. That is the
/// hook a shaping / bidi engine uses: it shapes a run of text into positioned
/// glyph ids for a font and submits them here, keeping the shaper itself out of
/// viewport-lib.
///
/// One run carries one font. A line that spans several fonts (script fallback,
/// or mixing a text font with an icon font) is submitted as several runs sharing
/// a baseline, one per font. Glyph positions are relative to the resolved
/// `anchor` origin, so moving a whole run is a change to `anchor` / `position`.
///
/// [`LabelItem`]: crate::renderer::types::LabelItem
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GlyphRunItem {
    /// Font the glyph ids index into. `None` uses the built-in default font.
    pub font: Option<crate::resources::overlay::font::FontHandle>,

    /// Font size in logical pixels. Sizes the rasterized glyph bitmaps; the glyph
    /// positions themselves come from `glyphs`.
    pub font_size: f32,

    /// Origin the run hangs from: a viewport corner (default top-left) or a
    /// projected world point. Each glyph's `(x, y)` is relative to this.
    pub anchor: crate::renderer::types::OverlayAnchor,

    /// Placement in logical pixels relative to the resolved `anchor` origin.
    /// With the default anchor and alignment this is the run's screen position
    /// from the viewport top-left. Default: `[0.0, 0.0]`.
    pub position: [f32; 2],

    /// How the run's glyph-extent box sits horizontally on `anchor` + `position`.
    /// Default `Left` leaves the glyph positions as authored.
    pub align_x: crate::renderer::types::AnchorX,

    /// How the run's glyph-extent box sits vertically on `anchor` + `position`.
    /// Default `Top` leaves the glyph positions as authored.
    pub align_y: crate::renderer::types::AnchorY,

    /// Positioned glyphs, in draw order.
    pub glyphs: Vec<PositionedGlyph>,

    /// RGBA tint in linear float, applied to every glyph in the run that does
    /// not have its own entry in `colours`.
    pub colour: [f32; 4],

    /// Optional per-glyph tint, parallel to `glyphs`. When non-empty, glyph `i`
    /// uses `colours[i]`; glyphs past the end of this list (or all glyphs when it
    /// is empty) fall back to `colour`. Use it for runs where glyphs differ in
    /// colour, such as syntax highlighting.
    pub colours: Vec<[f32; 4]>,

    /// Overall opacity multiplier applied to the run. Range 0.0 (invisible) to
    /// 1.0 (fully opaque).
    pub opacity: f32,

    /// Explicit draw order. Runs with lower values are drawn first (further
    /// back). Shares the cross-family z-order space with labels and shapes.
    pub z_order: i32,

    /// When set, the run is clipped to the mask shape whose `clip_mask_id`
    /// matches this value, the same clip model [`LabelItem`] uses. `None` (the
    /// default) draws the run unclipped.
    ///
    /// [`LabelItem`]: crate::renderer::types::LabelItem
    pub clip_id: Option<u32>,
}

impl Default for GlyphRunItem {
    fn default() -> Self {
        Self {
            font: None,
            font_size: 14.0,
            anchor: crate::renderer::types::OverlayAnchor::default(),
            position: [0.0, 0.0],
            align_x: crate::renderer::types::AnchorX::Left,
            align_y: crate::renderer::types::AnchorY::Top,
            glyphs: Vec::new(),
            colour: [1.0, 1.0, 1.0, 1.0],
            colours: Vec::new(),
            opacity: 1.0,
            z_order: 0,
            clip_id: None,
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
    pub fn with_font(mut self, font: crate::resources::overlay::font::FontHandle) -> Self {
        self.font = Some(font);
        self
    }

    /// Set the font size in logical pixels.
    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.font_size = font_size;
        self
    }

    /// Set the origin the run hangs from (a viewport corner or a world point).
    pub fn with_anchor(mut self, anchor: crate::renderer::types::OverlayAnchor) -> Self {
        self.anchor = anchor;
        self
    }

    /// Pin the run to a world-space position, reprojected each frame. Sugar for
    /// `with_anchor(OverlayAnchor::World(pos))`.
    pub fn with_world_anchor(mut self, pos: [f32; 3]) -> Self {
        self.anchor = crate::renderer::types::OverlayAnchor::World(pos);
        self
    }

    /// Set the placement in logical pixels relative to the resolved anchor
    /// origin. With the default anchor this is the run's screen position from
    /// the viewport top-left.
    pub fn with_position(mut self, position: [f32; 2]) -> Self {
        self.position = position;
        self
    }

    /// Set how the run's glyph-extent box aligns onto the resolved anchor origin.
    pub fn with_align(
        mut self,
        align_x: crate::renderer::types::AnchorX,
        align_y: crate::renderer::types::AnchorY,
    ) -> Self {
        self.align_x = align_x;
        self.align_y = align_y;
        self
    }

    /// Replace the positioned glyphs.
    pub fn with_glyphs(mut self, glyphs: impl Into<Vec<PositionedGlyph>>) -> Self {
        self.glyphs = glyphs.into();
        self
    }

    /// Set the run tint colour, used for any glyph without a per-glyph entry in
    /// `colours`.
    pub fn with_colour(mut self, colour: [f32; 4]) -> Self {
        self.colour = colour;
        self
    }

    /// Set per-glyph tint colours, parallel to the glyphs. Glyphs past the end of
    /// this list fall back to the run `colour`.
    pub fn with_colours(mut self, colours: impl Into<Vec<[f32; 4]>>) -> Self {
        self.colours = colours.into();
        self
    }

    /// Set the overall opacity multiplier (0.0 to 1.0).
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
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
    /// [`OverlayShapeItem::with_clip_mask`]: crate::renderer::types::OverlayShapeItem::with_clip_mask
    pub fn with_clip(mut self, clip_id: u32) -> Self {
        self.clip_id = Some(clip_id);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_builders() {
        let run = GlyphRunItem::default();
        assert!(run.font.is_none());
        assert_eq!(run.font_size, 14.0);
        assert_eq!(run.position, [0.0, 0.0]);
        assert!(run.glyphs.is_empty());
        assert!(run.colours.is_empty());
        assert_eq!(run.opacity, 1.0);
        assert_eq!(run.z_order, 0);
        assert!(run.clip_id.is_none());

        let glyphs = vec![
            PositionedGlyph::new(4, 0.0, 0.0),
            PositionedGlyph::new(9, 8.5, 0.0),
        ];
        let run = GlyphRunItem::new(glyphs.clone())
            .with_font_size(20.0)
            .with_position([10.0, 12.0])
            .with_colour([1.0, 0.0, 0.0, 1.0])
            .with_colours(vec![[0.0, 1.0, 0.0, 1.0]])
            .with_opacity(0.5)
            .with_z_order(3)
            .with_clip(7);

        assert_eq!(run.glyphs, glyphs);
        assert_eq!(run.font_size, 20.0);
        assert_eq!(run.position, [10.0, 12.0]);
        assert_eq!(run.colour, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(run.colours, vec![[0.0, 1.0, 0.0, 1.0]]);
        assert_eq!(run.opacity, 0.5);
        assert_eq!(run.z_order, 3);
        assert_eq!(run.clip_id, Some(7));
    }
}
