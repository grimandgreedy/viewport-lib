//! Color glyph rasterization for the overlay atlas.
//!
//! fontdue rasterizes glyph outlines to alpha coverage only, so color emoji never
//! render through it. This module decodes the embedded PNG bitmap strikes that
//! Apple Color Emoji (`sbix`) and Noto Color Emoji (`CBDT`) ship, scaled to the
//! requested pixel size, for the atlas to store as straight RGBA. COLR (vector /
//! gradient) fonts are not handled here; a glyph with no readable color bitmap
//! returns `None` and the atlas falls back to fontdue coverage.

use std::io::Cursor;

use ttf_parser::{Face, GlyphId, RasterImageFormat};

/// A decoded color glyph bitmap, scaled to the target size, plus its placement.
pub(crate) struct ColorBitmap {
    /// Scaled bitmap width in pixels.
    pub width: u32,
    /// Scaled bitmap height in pixels.
    pub height: u32,
    /// Straight (non-premultiplied) RGBA, row-major, `width * height` pixels.
    pub rgba: Vec<[u8; 4]>,
    /// Offset from the pen (baseline) to the bitmap top-left, in pixels, matching
    /// the coverage-glyph convention (`offset_y` is negative for a glyph that sits
    /// above the baseline).
    pub offset_x: f32,
    pub offset_y: f32,
}

/// Whether `font_bytes` carries any bitmap-strike table this module can read.
///
/// Checked once per font at upload so text in ordinary fonts never pays for a
/// color-glyph lookup.
pub(crate) fn has_color_bitmaps(font_bytes: &[u8]) -> bool {
    match Face::parse(font_bytes, 0) {
        Ok(face) => {
            let tables = face.tables();
            tables.sbix.is_some()
                || tables.cbdt.is_some()
                || tables.ebdt.is_some()
                || tables.bdat.is_some()
        }
        Err(_) => false,
    }
}

/// Rasterize the color bitmap for `glyph_id` at `target_px`, or `None` if the font
/// has no PNG color image for it.
pub(crate) fn rasterize(font_bytes: &[u8], glyph_id: u16, target_px: f32) -> Option<ColorBitmap> {
    if !(target_px > 0.0) {
        return None;
    }
    let face = Face::parse(font_bytes, 0).ok()?;

    // Ask for a strike at least as large as the target, then scale down from it.
    let request = target_px.ceil().min(u16::MAX as f32) as u16;
    let img = face.glyph_raster_image(GlyphId(glyph_id), request)?;
    if img.format != RasterImageFormat::PNG {
        // Monochrome / greyscale bitmap strikes are not color emoji; let fontdue
        // handle them (or fall through to .notdef).
        return None;
    }

    // The image's real dimensions come from the PNG, not the table's width/height
    // (which the spec does not guarantee to match the data).
    let (src_w, src_h, src) = decode_png_rgba(img.data)?;
    let ppem = img.pixels_per_em.max(1) as f32;
    let scale = target_px / ppem;
    let dst_w = ((src_w as f32) * scale).round().max(1.0) as u32;
    let dst_h = ((src_h as f32) * scale).round().max(1.0) as u32;
    let rgba = box_downscale(&src, src_w, src_h, dst_w, dst_h);

    // `sbix` and `CBDT` both report `(x, y)` as the bitmap's lower-left offset from
    // the glyph origin, measured y-up in strike pixels. Convert to a top-left
    // offset in screen space (y-down) relative to the baseline, at the target
    // scale, matching how coverage glyphs place `offset_y`.
    let offset_x = img.x as f32 * scale;
    let offset_y = -((img.y as f32 + src_h as f32) * scale);

    Some(ColorBitmap {
        width: dst_w,
        height: dst_h,
        rgba,
        offset_x,
        offset_y,
    })
}

/// Decode a PNG into straight RGBA8. Returns `(width, height, pixels)`.
fn decode_png_rgba(data: &[u8]) -> Option<(u32, u32, Vec<[u8; 4]>)> {
    let mut decoder = png::Decoder::new(Cursor::new(data));
    // Expand palette / low-bit greyscale and tRNS to full channels, and drop 16-bit
    // down to 8, so the frame is one of the four 8-bit types handled below.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;

    let count = (info.width as usize).checked_mul(info.height as usize)?;
    let mut out = Vec::with_capacity(count);
    match info.color_type {
        png::ColorType::Rgba => {
            for i in 0..count {
                let o = i * 4;
                out.push([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
            }
        }
        png::ColorType::Rgb => {
            for i in 0..count {
                let o = i * 3;
                out.push([buf[o], buf[o + 1], buf[o + 2], 255]);
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for i in 0..count {
                let o = i * 2;
                let v = buf[o];
                out.push([v, v, v, buf[o + 1]]);
            }
        }
        png::ColorType::Grayscale => {
            for i in 0..count {
                let v = buf[i];
                out.push([v, v, v, 255]);
            }
        }
        _ => return None,
    }
    Some((info.width, info.height, out))
}

/// Box-average downscale of straight RGBA. Color is averaged weighted by alpha so
/// transparent pixels do not darken the edges. `dst >= src` degenerates to a
/// nearest-style sample, which is fine for the rare upscale case.
fn box_downscale(src: &[[u8; 4]], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<[u8; 4]> {
    if dw == sw && dh == sh {
        return src.to_vec();
    }
    let mut out = vec![[0u8; 4]; (dw * dh) as usize];
    for dy in 0..dh {
        let sy0 = (dy * sh) / dh;
        let sy1 = (((dy + 1) * sh) / dh).max(sy0 + 1).min(sh);
        for dx in 0..dw {
            let sx0 = (dx * sw) / dw;
            let sx1 = (((dx + 1) * sw) / dw).max(sx0 + 1).min(sw);
            let (mut r, mut g, mut b, mut a, mut n) = (0u32, 0u32, 0u32, 0u32, 0u32);
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let p = src[(sy * sw + sx) as usize];
                    let av = p[3] as u32;
                    r += p[0] as u32 * av;
                    g += p[1] as u32 * av;
                    b += p[2] as u32 * av;
                    a += av;
                    n += 1;
                }
            }
            out[(dy * dw + dx) as usize] = if a > 0 {
                [(r / a) as u8, (g / a) as u8, (b / a) as u8, (a / n) as u8]
            } else {
                [0, 0, 0, 0]
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The system color-emoji font, when present. The test skips otherwise so the
    /// suite stays portable.
    const APPLE_EMOJI: &str = "/System/Library/Fonts/Apple Color Emoji.ttc";

    #[test]
    fn decodes_a_color_emoji_when_present() {
        let Ok(bytes) = std::fs::read(APPLE_EMOJI) else {
            eprintln!("skipping: no Apple Color Emoji font at {APPLE_EMOJI}");
            return;
        };
        assert!(
            has_color_bitmaps(&bytes),
            "an emoji font reports a bitmap strike"
        );

        let face = Face::parse(&bytes, 0).expect("emoji font parses");
        let gid = face
            .glyph_index('\u{1F600}')
            .expect("grinning face is present"); // 😀

        let bmp = rasterize(&bytes, gid.0, 48.0).expect("a color bitmap is produced");
        assert!(bmp.width > 0 && bmp.height > 0);
        assert_eq!(bmp.rgba.len(), (bmp.width * bmp.height) as usize);
        assert!(bmp.rgba.iter().any(|p| p[3] > 0), "has visible pixels");
        assert!(
            bmp.rgba.iter().any(|p| p[0] != p[1] || p[1] != p[2]),
            "has real colour, not just greyscale coverage"
        );
        // The emoji sits above the baseline, so its top is a negative y offset.
        assert!(
            bmp.offset_y < 0.0,
            "placed above the baseline: {}",
            bmp.offset_y
        );
    }

    #[test]
    fn plain_font_reports_no_color() {
        // The bundled default (Inter) has no bitmap strikes.
        assert!(!has_color_bitmaps(DEFAULT_FONT));
    }

    const DEFAULT_FONT: &[u8] = include_bytes!("../../fonts/Inter-Regular.ttf");
}
