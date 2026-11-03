//! Font handle for overlay text items.

/// Handle to an uploaded font, used by overlay label and glyph-run items.
///
/// Obtain one from `DeviceResources::upload_font`. Pass `None` (or omit the
/// field) on overlay items to use the built-in default font; pass
/// `Some(handle)` to use a user-supplied TTF font.
///
/// This is a plain index into the renderer's font store: the store itself (and
/// the font rasterisation behind it) lives in `viewport-lib`, but the handle a
/// consumer names on an overlay item is pure data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FontHandle(pub usize);

/// The typography of an overlay text item: which font, at what size.
///
/// Shared by [`LabelItem`](crate::overlay::LabelItem) and
/// [`GlyphRunItem`](crate::overlay::GlyphRunItem), which is the point: the two
/// families differ in what they are given (a string to lay out, or glyphs
/// already positioned) and not in how the glyphs are rasterised.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TextStyle {
    /// Font to rasterise with. `None` uses the built-in default font.
    pub font: Option<FontHandle>,
    /// Size in logical pixels.
    pub size: f32,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font: None,
            size: 14.0,
        }
    }
}

impl TextStyle {
    /// The default font at `size` logical pixels.
    pub fn new(size: f32) -> Self {
        Self { font: None, size }
    }

    /// Set the font.
    pub fn with_font(mut self, font: FontHandle) -> Self {
        self.font = Some(font);
        self
    }

    /// Set the size in logical pixels.
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }
}
