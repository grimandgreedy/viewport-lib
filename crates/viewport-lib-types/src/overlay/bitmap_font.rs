//! Bitmap fonts: glyph images packed into page images, with the metrics to lay
//! them out, as BMFont-style tools write them.

use std::collections::BTreeMap;

use crate::data::texture::TextureData;

/// A bitmap font to register with the renderer as a font face.
///
/// Everything is in the face's own pixels, X right and Y down: glyph
/// rectangles in their page image, offsets and the baseline from the top of a
/// line. Upload it with `DeviceResources::upload_bitmap_font`, which returns a
/// `FontHandle` that labels and glyph runs name like any other font.
///
/// A bitmap face draws at whole multiples of its size only: the requested text
/// size in device pixels rounds to the nearest multiple of [`size`](Self::size)
/// (at least one), every glyph is magnified by that factor with no filtering,
/// and pens land on whole device pixels.
///
/// Mapping a font decoded by `viewport-lib-io`'s BMFont loader, with its page
/// images read and decoded by the caller:
///
/// ```ignore
/// let data = viewport_lib_io::bitmap_font_from_path(path)?;
/// let pages = data.pages.iter().map(|p| decode_page(p)).collect();
/// let mut font = BitmapFont::new(
///     data.size.unsigned_abs(),
///     data.line_height,
///     data.base,
///     pages,
/// );
/// for (&code, g) in &data.glyphs {
///     let c = g.channels;
///     font = font.with_glyph(
///         code,
///         BitmapFontGlyph::new(
///             g.page,
///             [g.rect.x, g.rect.y, g.rect.width, g.rect.height],
///             g.offset,
///             g.advance,
///         )
///         .with_channels(BitmapGlyphChannels::new(c.red, c.green, c.blue, c.alpha)),
///     );
/// }
/// for (&(first, second), &amount) in &data.kernings {
///     font = font.with_kerning(first, second, amount);
/// }
/// let handle = resources.upload_bitmap_font(font)?;
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BitmapFont {
    /// The size the glyphs were drawn at, in pixels: the unit the face scales
    /// by. Zero takes [`line_height`](Self::line_height).
    pub size: u32,
    /// Distance from one line to the next, in pixels.
    pub line_height: u32,
    /// Distance from the top of a line to the baseline, in pixels.
    pub base: u32,
    /// The page images, by page index. 8-bit and float images are accepted;
    /// a block-compressed page is refused at upload.
    pub pages: Vec<TextureData>,
    /// Glyphs by Unicode code point (or by the font's own character id).
    pub glyphs: BTreeMap<u32, BitmapFontGlyph>,
    /// Extra advance in pixels between a pair of characters, by (first,
    /// second) code point, added after the first character's advance.
    pub kernings: BTreeMap<(u32, u32), i32>,
}

impl BitmapFont {
    /// A face drawn at `size` pixels with the given line metrics and pages,
    /// and no glyphs yet.
    pub fn new(size: u32, line_height: u32, base: u32, pages: Vec<TextureData>) -> Self {
        Self {
            size,
            line_height,
            base,
            pages,
            glyphs: BTreeMap::new(),
            kernings: BTreeMap::new(),
        }
    }

    /// Add the glyph for `code`, replacing any earlier one.
    pub fn with_glyph(mut self, code: u32, glyph: BitmapFontGlyph) -> Self {
        self.glyphs.insert(code, glyph);
        self
    }

    /// Add `amount` pixels of advance between `first` and `second`.
    pub fn with_kerning(mut self, first: u32, second: u32, amount: i32) -> Self {
        self.kernings.insert((first, second), amount);
        self
    }

    /// The size the face scales by: [`size`](Self::size), or the line height
    /// when the size is zero, and never below one pixel.
    pub fn unit_px(&self) -> u32 {
        if self.size > 0 {
            self.size
        } else {
            self.line_height.max(1)
        }
    }
}

/// One glyph of a [`BitmapFont`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct BitmapFontGlyph {
    /// Index of the page image the glyph is in.
    pub page: u32,
    /// The glyph's pixels in its page: x, y, width, height.
    pub rect: [u32; 4],
    /// Where `rect` draws relative to the pen at the top of the line.
    pub offset: [i32; 2],
    /// How far the pen moves after the glyph.
    pub advance: i32,
    /// The page channels the glyph is in.
    pub channels: BitmapGlyphChannels,
}

impl BitmapFontGlyph {
    /// A glyph at `rect` of page `page`, in every channel.
    pub fn new(page: u32, rect: [u32; 4], offset: [i32; 2], advance: i32) -> Self {
        Self {
            page,
            rect,
            offset,
            advance,
            channels: BitmapGlyphChannels::ALL,
        }
    }

    /// Set the page channels the glyph is in.
    pub fn with_channels(mut self, channels: BitmapGlyphChannels) -> Self {
        self.channels = channels;
        self
    }
}

/// Which channels of a page hold a glyph.
///
/// A glyph in every channel draws from the page's colour and alpha. A font
/// that packs glyphs into separate channels (one glyph in red, another in
/// green) names fewer: the glyph's coverage is the largest of the channels it
/// names, unpacked when the font is uploaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitmapGlyphChannels {
    /// The red channel.
    pub red: bool,
    /// The green channel.
    pub green: bool,
    /// The blue channel.
    pub blue: bool,
    /// The alpha channel.
    pub alpha: bool,
}

impl BitmapGlyphChannels {
    /// Every channel: the glyph is the page's colour and alpha.
    pub const ALL: Self = Self {
        red: true,
        green: true,
        blue: true,
        alpha: true,
    };

    /// The channels named.
    pub const fn new(red: bool, green: bool, blue: bool, alpha: bool) -> Self {
        Self {
            red,
            green,
            blue,
            alpha,
        }
    }

    /// Whether every channel is named.
    pub fn is_all(&self) -> bool {
        *self == Self::ALL
    }
}

impl Default for BitmapGlyphChannels {
    fn default() -> Self {
        Self::ALL
    }
}
