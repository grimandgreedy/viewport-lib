//! Shadow and outline styling for glyph coverage.
//!
//! A styled glyph is the plain one with its alpha put through a few
//! morphological steps: dilate or erode the coverage, fade it, and shape that
//! fade. Everything here works on an 8-bit alpha buffer and knows nothing about
//! fonts, shaping or the atlas, so the font code hands over coverage and gets
//! back a cell to pack.
//!
//! The work happens once per distinct style and size rather than per frame,
//! which is why [`GlyphStyle`] quantises its parameters: the number of distinct
//! atlas cells has to stay bounded.

/// How a glyph cell is post-processed after rasterization.
///
/// The plain style is the glyph itself. A shadow style grows the coverage by
/// `spread`, fades it over `blur`, and shapes that fade by `falloff`, producing
/// a cell that is drawn behind the glyph in the shadow colour. Baking it here
/// rather than at draw time means the work happens once per distinct style and
/// size, not per frame, and the result is exact rather than an approximation
/// built from offset copies.
///
/// All three are quantised for the same reason the font size is: to keep the
/// number of distinct atlas cells bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(crate) struct GlyphStyle {
    /// Dilation in tenths of a physical pixel.
    spread_tenths: u32,
    /// Fade distance in tenths of a physical pixel.
    blur_tenths: u32,
    /// Falloff exponent in tenths. `0` marks the plain (unstyled) glyph.
    falloff_tenths: u32,
    /// Erode inward from the glyph edge instead of dilating outward, for an
    /// inset layer. The cell is the glyph's own coverage with its interior
    /// eaten away, so it draws over the glyph rather than behind it.
    inner: bool,
    /// Where the glyph sits relative to this cell once the layer offset has
    /// moved it, in whole physical pixels. An outer cell is cut back to outside
    /// the letterform, and the cut has to land where the letterform actually
    /// is, so the offset is part of what makes a cell distinct.
    clip_dx: i32,
    clip_dy: i32,
}

/// Largest dilation or blur honoured on a glyph cell, in physical pixels.
/// Past this the cell dwarfs the glyph and the atlas cost stops being worth it.
const MAX_GLYPH_STYLE_PX: f32 = 32.0;

impl GlyphStyle {
    /// The glyph as rasterized, with no shadow processing.
    pub(crate) const PLAIN: Self = Self {
        spread_tenths: 0,
        blur_tenths: 0,
        falloff_tenths: 0,
        inner: false,
        clip_dx: 0,
        clip_dy: 0,
    };

    /// Build a style from a shadow layer's physical-pixel spread, blur and
    /// offset. Returns [`GlyphStyle::PLAIN`] when the layer would not change
    /// the cell.
    pub(crate) fn from_shadow(
        spread_px: f32,
        blur_px: f32,
        falloff: f32,
        offset_px: [f32; 2],
    ) -> Self {
        let mut s = Self::from_layer(spread_px, blur_px, falloff, false);
        if !s.is_plain() {
            let clamp = |v: f32| v.clamp(-MAX_GLYPH_STYLE_PX, MAX_GLYPH_STYLE_PX).round() as i32;
            s.clip_dx = clamp(offset_px[0]);
            s.clip_dy = clamp(offset_px[1]);
        }
        s
    }

    /// The inset counterpart: the cell is the glyph with a band eaten inward
    /// from its edge, drawn over the glyph in the layer colour.
    pub(crate) fn from_inner_shadow(spread_px: f32, blur_px: f32, falloff: f32) -> Self {
        Self::from_layer(spread_px, blur_px, falloff, true)
    }

    fn from_layer(spread_px: f32, blur_px: f32, falloff: f32, inner: bool) -> Self {
        let spread = spread_px.clamp(0.0, MAX_GLYPH_STYLE_PX);
        let blur = blur_px.clamp(0.0, MAX_GLYPH_STYLE_PX);
        if spread <= 0.0 && blur <= 0.0 {
            return Self::PLAIN;
        }
        Self {
            spread_tenths: (spread * 10.0).round() as u32,
            blur_tenths: (blur * 10.0).round() as u32,
            falloff_tenths: ((falloff.clamp(0.05, 16.0)) * 10.0).round().max(1.0) as u32,
            inner,
            clip_dx: 0,
            clip_dy: 0,
        }
    }

    pub(crate) fn is_plain(&self) -> bool {
        self.falloff_tenths == 0
    }

    fn spread(&self) -> f32 {
        self.spread_tenths as f32 * 0.1
    }

    fn blur(&self) -> f32 {
        self.blur_tenths as f32 * 0.1
    }

    fn falloff(&self) -> f32 {
        self.falloff_tenths as f32 * 0.1
    }

    /// Physical pixels the styled cell grows on every side.
    fn pad(&self) -> u32 {
        if self.is_plain() {
            return 0;
        }
        if self.inner {
            // An inset cell never grows past the glyph; one pixel of margin
            // keeps the blur off the cell border.
            return 1;
        }
        (self.spread() + self.blur()).ceil() as u32 + 1
    }
}

// ---------------------------------------------------------------------------
// Shadow cell baking
// ---------------------------------------------------------------------------

/// Turn a glyph coverage bitmap into a shadow cell: dilate by the style's
/// spread, fade over its blur, then shape that fade by its falloff.
///
/// Returns the new cell, its dimensions, and the padding added on each side
/// (the caller shifts the glyph's bearing by this so the cell stays registered
/// with the glyph it backs).
///
/// Dilation is a max over a disc, which keeps round letterforms round; a square
/// window would square off the contour on curves. The fade is two box passes,
/// whose triangle kernel is a closer match to the smoothstep the SDF shape path
/// uses than a single box would be.
pub(crate) fn style_coverage(
    coverage: &[u8],
    w: u32,
    h: u32,
    style: GlyphStyle,
) -> (Vec<[u8; 4]>, u32, u32, u32) {
    let pad = style.pad();
    let ow = w + pad * 2;
    let oh = h + pad * 2;

    // Place the source in the padded cell.
    let mut a = vec![0u8; (ow * oh) as usize];
    for y in 0..h {
        for x in 0..w {
            a[((y + pad) * ow + (x + pad)) as usize] = coverage[(y * w + x) as usize];
        }
    }

    // Kept for the cases that need the glyph's own coverage back: the inset
    // band is what an erosion ate, and an outer layer is cut back to outside
    // the letterform.
    let src = a.clone();

    let spread = style.spread();
    if spread > 0.0 {
        a = if style.inner {
            erode_disc(&a, ow, oh, spread)
        } else {
            dilate_disc(&a, ow, oh, spread)
        };
    }

    let blur = style.blur();
    if blur > 0.0 {
        // Two box passes of radius blur/4 give a ramp about `blur` wide.
        let r = (blur * 0.25).round().max(1.0) as u32;
        a = box_blur(&a, ow, oh, r);
        a = box_blur(&a, ow, oh, r);
    }

    if style.inner {
        // The band is what the erosion ate: the glyph's own coverage minus
        // what survived. Multiplying by the source keeps the outer edge as
        // clean as the glyph's, which is what an inset layer needs, since it
        // draws over the letterform rather than behind it.
        for (v, &s) in a.iter_mut().zip(src.iter()) {
            *v = 255 - *v;
            *v = ((*v as u32 * s as u32) / 255) as u8;
        }
    } else {
        // Cut the cell back to outside the letterform, the way an outer
        // box-shadow is clipped to outside the border box. The layer offset
        // moves the cell, so the letterform is sampled at that offset: the
        // hole then lands on the glyph once the cell is drawn. Each glyph is
        // cut against its own coverage, so a neighbour's shadow can still show
        // through a translucent letterform where the two overlap.
        for y in 0..oh as i32 {
            for x in 0..ow as i32 {
                let (sx, sy) = (x + style.clip_dx, y + style.clip_dy);
                if sx < 0 || sy < 0 || sx >= ow as i32 || sy >= oh as i32 {
                    continue;
                }
                let g = src[(sy * ow as i32 + sx) as usize] as u32;
                let i = (y * ow as i32 + x) as usize;
                a[i] = ((a[i] as u32 * (255 - g)) / 255) as u8;
            }
        }
    }

    let falloff = style.falloff();
    if (falloff - 1.0).abs() > f32::EPSILON {
        for v in a.iter_mut() {
            let t = (*v as f32 / 255.0).powf(falloff);
            *v = (t * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    let cell: Vec<[u8; 4]> = a.iter().map(|&v| [255, 255, 255, v]).collect();
    (cell, ow, oh, pad)
}

/// Grow coverage by taking the maximum over a disc of `radius` pixels.
fn dilate_disc(src: &[u8], w: u32, h: u32, radius: f32) -> Vec<u8> {
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
    // Offsets inside the disc, computed once rather than per pixel.
    let mut disc: Vec<(i32, i32)> = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as f32 <= r2 {
                disc.push((dx, dy));
            }
        }
    }

    let mut out = vec![0u8; src.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut m = 0u8;
            for &(dx, dy) in &disc {
                let (sx, sy) = (x + dx, y + dy);
                if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                    continue;
                }
                let v = src[(sy * w as i32 + sx) as usize];
                if v > m {
                    m = v;
                    if m == 255 {
                        break;
                    }
                }
            }
            out[(y * w as i32 + x) as usize] = m;
        }
    }
    out
}

/// Shrink coverage by taking the minimum over a disc of `radius` pixels.
/// Outside the cell counts as empty, so the glyph erodes in from its edge.
fn erode_disc(src: &[u8], w: u32, h: u32, radius: f32) -> Vec<u8> {
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
    let mut disc: Vec<(i32, i32)> = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as f32 <= r2 {
                disc.push((dx, dy));
            }
        }
    }

    let mut out = vec![0u8; src.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut m = 255u8;
            for &(dx, dy) in &disc {
                let (sx, sy) = (x + dx, y + dy);
                let v = if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 {
                    0
                } else {
                    src[(sy * w as i32 + sx) as usize]
                };
                if v < m {
                    m = v;
                    if m == 0 {
                        break;
                    }
                }
            }
            out[(y * w as i32 + x) as usize] = m;
        }
    }
    out
}

/// Separable box blur of the given radius, run horizontally then vertically.
fn box_blur(src: &[u8], w: u32, h: u32, radius: u32) -> Vec<u8> {
    let r = radius as i32;
    let n = (r * 2 + 1) as u32;
    let mut tmp = vec![0u8; src.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = 0u32;
            for d in -r..=r {
                let sx = (x + d).clamp(0, w as i32 - 1);
                sum += src[(y * w as i32 + sx) as usize] as u32;
            }
            tmp[(y * w as i32 + x) as usize] = (sum / n) as u8;
        }
    }
    let mut out = vec![0u8; src.len()];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = 0u32;
            for d in -r..=r {
                let sy = (y + d).clamp(0, h as i32 - 1);
                sum += tmp[(sy * w as i32 + x) as usize] as u32;
            }
            out[(y * w as i32 + x) as usize] = (sum / n) as u8;
        }
    }
    out
}
