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
/// One run carries one font. A line that spans several fonts (script fallback)
/// is submitted as several runs. Positions are relative to `origin`, so moving a
/// whole run is a change to `origin` alone.
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

    /// Screen-space origin in logical pixels from the top-left. Each glyph's
    /// `(x, y)` is added to this.
    pub origin: [f32; 2],

    /// Positioned glyphs, in draw order.
    pub glyphs: Vec<PositionedGlyph>,

    /// RGBA tint in linear float, applied to every glyph in the run.
    pub colour: [f32; 4],

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
            origin: [0.0, 0.0],
            glyphs: Vec::new(),
            colour: [1.0, 1.0, 1.0, 1.0],
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

    /// Set the screen-space origin in logical pixels from the top-left.
    pub fn with_origin(mut self, origin: [f32; 2]) -> Self {
        self.origin = origin;
        self
    }

    /// Replace the positioned glyphs.
    pub fn with_glyphs(mut self, glyphs: impl Into<Vec<PositionedGlyph>>) -> Self {
        self.glyphs = glyphs.into();
        self
    }

    /// Set the run tint colour.
    pub fn with_colour(mut self, colour: [f32; 4]) -> Self {
        self.colour = colour;
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
        assert_eq!(run.origin, [0.0, 0.0]);
        assert!(run.glyphs.is_empty());
        assert_eq!(run.opacity, 1.0);
        assert_eq!(run.z_order, 0);
        assert!(run.clip_id.is_none());

        let glyphs = vec![
            PositionedGlyph::new(4, 0.0, 0.0),
            PositionedGlyph::new(9, 8.5, 0.0),
        ];
        let run = GlyphRunItem::new(glyphs.clone())
            .with_font_size(20.0)
            .with_origin([10.0, 12.0])
            .with_colour([1.0, 0.0, 0.0, 1.0])
            .with_opacity(0.5)
            .with_z_order(3)
            .with_clip(7);

        assert_eq!(run.glyphs, glyphs);
        assert_eq!(run.font_size, 20.0);
        assert_eq!(run.origin, [10.0, 12.0]);
        assert_eq!(run.colour, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(run.opacity, 0.5);
        assert_eq!(run.z_order, 3);
        assert_eq!(run.clip_id, Some(7));
    }
}
