//! Outline coverage rasterization for glyph ids fontdue does not load.
//!
//! `fontdue::Font::from_bytes` only parses the glyphs it can reach from `cmap`
//! and `GSUB`. Every other slot in its glyph table stays empty, so a glyph id
//! that a shaper produced from some other table comes back as a `0x0` bitmap
//! with a zero advance even though the font outlines it. The OpenType `MATH`
//! table is the common case: the base bracket, radical or integral is mapped and
//! draws, while its larger size variants and assembly pieces have no codepoint
//! and would otherwise draw as nothing at all.
//!
//! This reads the outline with `ttf-parser` and fills it to alpha coverage in the
//! same layout and with the same placement convention as the fontdue path, so the
//! atlas can pack it the same way. Glyphs fontdue already rasterizes never come
//! here: the atlas only falls back when fontdue reports an empty bitmap.

use ttf_parser::{Face, GlyphId, OutlineBuilder};
use zeno::{Command, Mask, Vector};

/// A rasterized glyph outline plus its placement.
pub(crate) struct OutlineCoverage {
    /// Bitmap width in pixels.
    pub width: u32,
    /// Bitmap height in pixels.
    pub height: u32,
    /// Alpha coverage, row-major, `width * height` samples.
    pub coverage: Vec<u8>,
    /// Offset from the pen (baseline) to the bitmap top-left, in pixels, matching
    /// the fontdue path (`offset_y` is negative for a glyph above the baseline).
    pub offset_x: f32,
    pub offset_y: f32,
}

/// Collects a `ttf-parser` outline as a zeno path, scaled to pixels and flipped
/// into y-down raster space.
struct PathSink {
    commands: Vec<Command>,
    scale: f32,
}

impl PathSink {
    fn point(&self, x: f32, y: f32) -> Vector {
        Vector::new(x * self.scale, -y * self.scale)
    }
}

impl OutlineBuilder for PathSink {
    fn move_to(&mut self, x: f32, y: f32) {
        self.commands.push(Command::MoveTo(self.point(x, y)));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.commands.push(Command::LineTo(self.point(x, y)));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let c = self.point(x1, y1);
        self.commands.push(Command::QuadTo(c, self.point(x, y)));
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let c1 = self.point(x1, y1);
        let c2 = self.point(x2, y2);
        self.commands
            .push(Command::CurveTo(c1, c2, self.point(x, y)));
    }

    fn close(&mut self) {
        self.commands.push(Command::Close);
    }
}

/// Rasterize the outline of `glyph_id` at `target_px`, or `None` if the font does
/// not outline it (a whitespace glyph, or an id past the end of the font).
pub(crate) fn rasterise(
    font_bytes: &[u8],
    glyph_id: u16,
    target_px: f32,
) -> Option<OutlineCoverage> {
    if !(target_px > 0.0) {
        return None;
    }
    let face = Face::parse(font_bytes, 0).ok()?;
    let upem = face.units_per_em();
    if upem == 0 {
        return None;
    }

    let mut sink = PathSink {
        commands: Vec::new(),
        scale: target_px / upem as f32,
    };
    face.outline_glyph(GlyphId(glyph_id), &mut sink)?;
    if sink.commands.is_empty() {
        return None;
    }

    let (coverage, placement) = Mask::new(sink.commands.as_slice()).render();
    if placement.width == 0 || placement.height == 0 {
        // The font outlines this glyph and nothing was drawn, so the caller is
        // about to leave a gap the width of the advance it laid out with. That is
        // worth seeing rather than silently losing a glyph.
        tracing::warn!(
            glyph_id,
            target_px,
            "glyph has an outline but rasterised to nothing; it will draw as a gap"
        );
        return None;
    }

    Some(OutlineCoverage {
        width: placement.width,
        height: placement.height,
        coverage,
        offset_x: placement.left as f32,
        offset_y: placement.top as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTER: &[u8] = include_bytes!("../../fonts/Inter-Regular.ttf");

    /// Glyph 646 of the bundled font is outlined but unreachable from `cmap`, so
    /// fontdue leaves it empty and this is the only path that draws it.
    #[test]
    fn rasterises_a_glyph_fontdue_does_not_load() {
        let fd = fontdue::Font::from_bytes(INTER, fontdue::FontSettings::default()).unwrap();
        let m = fd.metrics_indexed(646, 48.0);
        assert_eq!(
            (m.width, m.height),
            (0, 0),
            "glyph 646 is the test case because fontdue reports it empty"
        );

        let cov = rasterise(INTER, 646, 48.0).expect("the font outlines glyph 646");
        assert!(cov.width > 0 && cov.height > 0);
        assert_eq!(cov.coverage.len(), (cov.width * cov.height) as usize);
        assert!(
            cov.coverage.iter().any(|&a| a > 128),
            "expected solid coverage, not a sliver"
        );
        // Sits above the baseline, so the top offset is negative.
        assert!(cov.offset_y < 0.0);
    }

    /// A glyph id past the end of the font has no outline and must not be packed.
    #[test]
    fn missing_glyph_returns_none() {
        assert!(rasterise(INTER, u16::MAX, 48.0).is_none());
    }

    /// A zero or negative size is rejected rather than rasterised.
    #[test]
    fn non_positive_size_returns_none() {
        assert!(rasterise(INTER, 646, 0.0).is_none());
        assert!(rasterise(INTER, 646, -8.0).is_none());
    }
}
