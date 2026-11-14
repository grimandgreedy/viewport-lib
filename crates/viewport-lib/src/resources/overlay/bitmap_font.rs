//! Bitmap font faces: a [`BitmapFont`]'s glyphs cut out of their pages at
//! upload, laid out with the face's advances and kerning, and drawn at whole
//! multiples of the face's size.

use std::collections::HashMap;

use viewport_lib_types::colour::ColourSpace;
use viewport_lib_types::data::texture::TexturePayload;

use viewport_lib_types::overlay::{BitmapFont, BitmapGlyphChannels};

use super::font::FontError;

/// One glyph cut out of its page: coverage for a coverage face, straight
/// sRGB-encoded RGBA for a colour face.
pub(crate) struct BitmapCell {
    pub width: u32,
    pub height: u32,
    /// Top-left of the cell from the pen at the top of the line.
    pub offset: [i32; 2],
    pub advance: i32,
    pub coverage: Vec<u8>,
    pub rgba: Vec<[u8; 4]>,
}

/// A bitmap face as the glyph atlas holds it. Glyph ids are indices into
/// `cells`, in code point order.
pub(crate) struct BitmapFace {
    pub unit: u32,
    pub line_height: u32,
    pub base: u32,
    /// Whether the glyphs carry their own colour. A face whose glyphs are
    /// white (or grey equal to their alpha) is a coverage face and takes the
    /// item's fill like vector text; any other draws its colours as a colour
    /// glyph does.
    pub colour: bool,
    pub cells: Vec<BitmapCell>,
    ids: HashMap<u32, u16>,
    kernings: HashMap<(u16, u16), i32>,
}

/// A page decoded to straight RGBA: colour sRGB-encoded, alpha and raw
/// channel values as bytes.
struct Page {
    width: u32,
    height: u32,
    /// sRGB-encoded colour plus linear alpha, for drawing.
    colour: Vec<[u8; 4]>,
    /// The stored channel values, for unpacking channel-packed glyphs.
    raw: Vec<[u8; 4]>,
}

fn to_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn decode_page(
    index: usize,
    page: &viewport_lib_types::data::texture::TextureData,
) -> Result<Page, FontError> {
    let (w, h) = (page.width(), page.height());
    let n = (w as usize) * (h as usize);
    let bad = |why: &str| FontError::InvalidBitmapFont(format!("page {index}: {why}"));
    let (raw, colour): (Vec<[u8; 4]>, Vec<[u8; 4]>) = match page.payload() {
        TexturePayload::Rgba8(bytes) => {
            if bytes.len() != n * 4 {
                return Err(bad("byte length does not match its size"));
            }
            let raw: Vec<[u8; 4]> = bytemuck::cast_slice::<u8, [u8; 4]>(bytes).to_vec();
            let colour = match page.colour_space() {
                ColourSpace::Srgb => raw.clone(),
                ColourSpace::Linear => raw
                    .iter()
                    .map(|p| {
                        let e = |c: u8| {
                            to_byte(viewport_lib_types::colour::linear_to_srgb(c as f32 / 255.0))
                        };
                        [e(p[0]), e(p[1]), e(p[2]), p[3]]
                    })
                    .collect(),
            };
            (raw, colour)
        }
        TexturePayload::Rgba32F(floats) => {
            if floats.len() != n * 4 {
                return Err(bad("float length does not match its size"));
            }
            let raw: Vec<[u8; 4]> = floats
                .chunks_exact(4)
                .map(|p| [to_byte(p[0]), to_byte(p[1]), to_byte(p[2]), to_byte(p[3])])
                .collect();
            let colour = floats
                .chunks_exact(4)
                .map(|p| {
                    let e = |c: f32| to_byte(viewport_lib_types::colour::linear_to_srgb(c));
                    [e(p[0]), e(p[1]), e(p[2]), to_byte(p[3])]
                })
                .collect();
            (raw, colour)
        }
        _ => return Err(bad("a block-compressed page cannot be cut into glyphs")),
    };
    Ok(Page {
        width: w,
        height: h,
        colour,
        raw,
    })
}

/// Whether a glyph pixel draws the same as coverage alone: white, or grey at
/// its own alpha (a glyph written into every channel).
fn is_coverage_texel(p: [u8; 4]) -> bool {
    if p[3] == 0 {
        return true;
    }
    let white = p[0] >= 254 && p[1] >= 254 && p[2] >= 254;
    let near = |c: u8| (c as i32 - p[3] as i32).abs() <= 1;
    white || (near(p[0]) && near(p[1]) && near(p[2]))
}

impl BitmapFace {
    /// Cut `font`'s glyphs out of its pages.
    pub(crate) fn new(font: &BitmapFont) -> Result<Self, FontError> {
        if font.glyphs.len() > u16::MAX as usize {
            return Err(FontError::InvalidBitmapFont(format!(
                "{} glyphs, more than a face can hold ({})",
                font.glyphs.len(),
                u16::MAX
            )));
        }
        let pages: Vec<Page> = font
            .pages
            .iter()
            .enumerate()
            .map(|(i, p)| decode_page(i, p))
            .collect::<Result<_, _>>()?;

        // Each glyph's texels, as the page stores them and as drawn.
        struct Cut {
            packed: Option<BitmapGlyphChannels>,
            raw: Vec<[u8; 4]>,
            colour: Vec<[u8; 4]>,
        }
        let mut cuts = Vec::with_capacity(font.glyphs.len());
        for (&code, g) in &font.glyphs {
            let [x, y, w, h] = g.rect;
            let page = pages.get(g.page as usize).ok_or_else(|| {
                FontError::InvalidBitmapFont(format!(
                    "glyph {code} names page {}, and the font has {}",
                    g.page,
                    pages.len()
                ))
            })?;
            if x.saturating_add(w) > page.width || y.saturating_add(h) > page.height {
                return Err(FontError::InvalidBitmapFont(format!(
                    "glyph {code} reaches past the edge of page {}",
                    g.page
                )));
            }
            let mut raw = Vec::with_capacity((w * h) as usize);
            let mut colour = Vec::with_capacity((w * h) as usize);
            for row in y..y + h {
                let start = (row * page.width + x) as usize;
                raw.extend_from_slice(&page.raw[start..start + w as usize]);
                colour.extend_from_slice(&page.colour[start..start + w as usize]);
            }
            let c = g.channels;
            let none = !(c.red || c.green || c.blue || c.alpha);
            cuts.push(Cut {
                packed: (!c.is_all() && !none).then_some(c),
                raw,
                colour,
            });
        }
        let colour = cuts
            .iter()
            .any(|c| c.packed.is_none() && !c.colour.iter().all(|&p| is_coverage_texel(p)));

        let mut cells = Vec::with_capacity(cuts.len());
        let mut ids = HashMap::with_capacity(cuts.len());
        for ((i, (&code, g)), cut) in font.glyphs.iter().enumerate().zip(cuts) {
            ids.insert(code, i as u16);
            let coverage: Vec<u8> = match cut.packed {
                Some(c) => cut
                    .raw
                    .iter()
                    .map(|p| {
                        let pick = |on: bool, v: u8| if on { v } else { 0 };
                        pick(c.red, p[0])
                            .max(pick(c.green, p[1]))
                            .max(pick(c.blue, p[2]))
                            .max(pick(c.alpha, p[3]))
                    })
                    .collect(),
                None => cut.colour.iter().map(|p| p[3]).collect(),
            };
            let rgba = if !colour {
                Vec::new()
            } else if cut.packed.is_some() {
                coverage.iter().map(|&a| [255, 255, 255, a]).collect()
            } else {
                cut.colour
            };
            cells.push(BitmapCell {
                width: g.rect[2],
                height: g.rect[3],
                offset: g.offset,
                advance: g.advance,
                coverage,
                rgba,
            });
        }
        let kernings = font
            .kernings
            .iter()
            .filter_map(|(&(a, b), &k)| Some(((*ids.get(&a)?, *ids.get(&b)?), k)))
            .collect();
        Ok(Self {
            unit: font.unit_px(),
            line_height: font.line_height,
            base: font.base,
            colour,
            cells,
            ids,
            kernings,
        })
    }

    /// The whole multiple of the face's size that text `px` device pixels
    /// tall draws at: the nearest, and at least one.
    pub(crate) fn scale_for(&self, px: f32) -> u32 {
        let k = (px / self.unit as f32).round();
        if k >= 1.0 { k as u32 } else { 1 }
    }

    /// The glyph id for `c`, if the face has it.
    pub(crate) fn id(&self, c: char) -> Option<u16> {
        self.ids.get(&(c as u32)).copied()
    }

    /// Lay out one run of `text` at scale `k`, appending `(id, pen x)` in
    /// device pixels from the run's start; returns the run's advance. A
    /// character the face lacks draws nothing and takes no room.
    pub(crate) fn run(&self, text: &str, k: u32, out: &mut Vec<(u16, f32)>) -> f32 {
        let k = k as i64;
        let mut pen: i64 = 0;
        let mut prev: Option<u16> = None;
        for c in text.chars() {
            let Some(id) = self.id(c) else {
                continue;
            };
            if let Some(p) = prev {
                pen += self.kernings.get(&(p, id)).copied().unwrap_or(0) as i64 * k;
            }
            out.push((id, pen as f32));
            pen += self.cells[id as usize].advance as i64 * k;
            prev = Some(id);
        }
        pen as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viewport_lib_types::data::texture::TextureData;
    use viewport_lib_types::overlay::BitmapFontGlyph;

    fn page(px: [u8; 4]) -> TextureData {
        TextureData::srgb(2, 2, px.repeat(4))
    }

    fn one_glyph(page: TextureData) -> BitmapFont {
        BitmapFont::new(4, 5, 4, vec![page])
            .with_glyph('a' as u32, BitmapFontGlyph::new(0, [0, 0, 2, 2], [0, 0], 3))
    }

    #[test]
    fn white_or_grey_at_alpha_glyphs_are_coverage() {
        let face = BitmapFace::new(&one_glyph(page([255, 255, 255, 128]))).unwrap();
        assert!(!face.colour);
        assert_eq!(face.cells[0].coverage, vec![128; 4]);
        let face = BitmapFace::new(&one_glyph(page([90, 90, 90, 90]))).unwrap();
        assert!(!face.colour, "a glyph written into every channel");
        let face = BitmapFace::new(&one_glyph(page([200, 40, 40, 255]))).unwrap();
        assert!(face.colour);
        assert_eq!(face.cells[0].rgba, vec![[200, 40, 40, 255]; 4]);
    }

    #[test]
    fn packed_glyphs_take_their_own_channel() {
        let font = BitmapFont::new(4, 5, 4, vec![page([10, 20, 30, 40])])
            .with_glyph(
                'r' as u32,
                BitmapFontGlyph::new(0, [0, 0, 2, 2], [0, 0], 3)
                    .with_channels(BitmapGlyphChannels::new(true, false, false, false)),
            )
            .with_glyph(
                'b' as u32,
                BitmapFontGlyph::new(0, [0, 0, 2, 2], [0, 0], 3)
                    .with_channels(BitmapGlyphChannels::new(false, false, true, false)),
            );
        let face = BitmapFace::new(&font).unwrap();
        // Code point order: 'b' then 'r'.
        assert_eq!(face.cells[0].coverage, vec![30; 4]);
        assert_eq!(face.cells[1].coverage, vec![10; 4]);
    }

    #[test]
    fn a_glyph_past_its_page_is_refused() {
        let font = BitmapFont::new(4, 5, 4, vec![page([255; 4])])
            .with_glyph('a' as u32, BitmapFontGlyph::new(0, [1, 1, 2, 2], [0, 0], 3));
        assert!(matches!(
            BitmapFace::new(&font),
            Err(FontError::InvalidBitmapFont(_))
        ));
        let font = BitmapFont::new(4, 5, 4, vec![page([255; 4])])
            .with_glyph('a' as u32, BitmapFontGlyph::new(1, [0, 0, 2, 2], [0, 0], 3));
        assert!(matches!(
            BitmapFace::new(&font),
            Err(FontError::InvalidBitmapFont(_))
        ));
    }

    #[test]
    fn sizes_round_to_whole_multiples() {
        let face = BitmapFace::new(&one_glyph(page([255; 4]))).unwrap();
        assert_eq!(face.scale_for(1.0), 1);
        assert_eq!(face.scale_for(5.9), 1);
        assert_eq!(face.scale_for(6.0), 2);
        assert_eq!(face.scale_for(12.0), 3);
        let mut run = Vec::new();
        assert_eq!(
            face.run("aza", 2, &mut run),
            12.0,
            "a missing glyph takes no room"
        );
        assert_eq!(run, vec![(0, 0.0), (0, 6.0)]);
    }
}
