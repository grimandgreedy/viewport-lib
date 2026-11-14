//! A 4x5 colour matrix: the colour filter an overlay item or layer applies
//! after its fill.

/// A colour filter as a 4x5 matrix over linear RGBA.
///
/// Row `i` computes output channel `i` (red, green, blue, alpha) from the input
/// `[r, g, b, a, 1]`: four weights and an offset. The input is straight
/// (unpremultiplied) colour, so a filter that scales alpha does not darken the
/// colour, and the result is clamped to `[0, 1]`. The renderer unpremultiplies
/// before the matrix and premultiplies after it wherever it holds premultiplied
/// colour, which is what makes a layer's filter and an item's filter agree.
///
/// The constructors cover the usual filters; [`then`](Self::then) chains them.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColourMatrix(pub [[f32; 5]; 4]);

/// Rec. 709 luma weights for linear colour.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

impl Default for ColourMatrix {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ColourMatrix {
    /// Leaves every colour unchanged.
    pub const IDENTITY: Self = Self([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]);

    /// Every colour to its luma, alpha unchanged.
    pub fn greyscale() -> Self {
        Self::saturation(0.0)
    }

    /// Scale saturation about the luma: `1.0` is unchanged, `0.0` greyscale,
    /// above `1.0` more saturated.
    pub fn saturation(s: f32) -> Self {
        let mut m = Self::IDENTITY;
        for (i, row) in m.0.iter_mut().take(3).enumerate() {
            for (j, w) in LUMA.iter().enumerate() {
                row[j] = (1.0 - s) * w + if i == j { s } else { 0.0 };
            }
        }
        m
    }

    /// Multiply red, green and blue by `b`: `1.0` is unchanged, `0.0` black.
    pub fn brightness(b: f32) -> Self {
        Self::scale([b, b, b, 1.0])
    }

    /// Scale red, green and blue about mid grey: `1.0` is unchanged, `0.0`
    /// flat grey.
    pub fn contrast(c: f32) -> Self {
        let mut m = Self::scale([c, c, c, 1.0]);
        for row in m.0.iter_mut().take(3) {
            row[4] = 0.5 * (1.0 - c);
        }
        m
    }

    /// Multiply each channel by `colour`'s, alpha included.
    pub fn tint(colour: impl Into<crate::colour::Colour>) -> Self {
        Self::scale(colour.into().to_linear_rgba())
    }

    /// Multiply alpha by `a`.
    pub fn opacity(a: f32) -> Self {
        Self::scale([1.0, 1.0, 1.0, a])
    }

    /// A diagonal matrix: each channel multiplied by its factor.
    pub fn scale(factors: [f32; 4]) -> Self {
        let mut m = Self([[0.0; 5]; 4]);
        for (i, f) in factors.iter().enumerate() {
            m.0[i][i] = *f;
        }
        m
    }

    /// This filter followed by `next`, as one matrix.
    pub fn then(&self, next: &ColourMatrix) -> Self {
        let mut out = [[0.0f32; 5]; 4];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..4 {
                    acc += next.0[i][k] * self.0[k][j];
                }
                if j == 4 {
                    acc += next.0[i][4];
                }
                *v = acc;
            }
        }
        Self(out)
    }

    /// Whether this is the identity, which the renderer skips.
    pub fn is_identity(&self) -> bool {
        *self == Self::IDENTITY
    }

    /// Apply the filter to a straight-alpha linear colour, clamped to `[0, 1]`.
    pub fn apply(&self, c: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (i, o) in out.iter_mut().enumerate() {
            let r = &self.0[i];
            *o = (r[0] * c[0] + r[1] * c[1] + r[2] * c[2] + r[3] * c[3] + r[4]).clamp(0.0, 1.0);
        }
        out
    }

    /// The matrix as five columns of four, the order the renderer uploads it
    /// in: the weights of red, green, blue and alpha, then the offsets.
    pub fn columns(&self) -> [[f32; 4]; 5] {
        let mut cols = [[0.0f32; 4]; 5];
        for (j, col) in cols.iter_mut().enumerate() {
            for (i, v) in col.iter_mut().enumerate() {
                *v = self.0[i][j];
            }
        }
        cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn greyscale_keeps_luma_and_alpha() {
        let c = [0.8, 0.2, 0.1, 0.5];
        let g = ColourMatrix::greyscale().apply(c);
        let y = 0.2126 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.1;
        assert!(close(g, [y, y, y, 0.5]));
        assert!(close(ColourMatrix::saturation(1.0).apply(c), c));
    }

    #[test]
    fn then_applies_in_order() {
        let c = [0.4, 0.6, 0.2, 1.0];
        let a = ColourMatrix::contrast(1.5);
        let b = ColourMatrix::brightness(0.5);
        assert!(close(a.then(&b).apply(c), b.apply(a.apply(c))));
        assert!(close(b.then(&a).apply(c), a.apply(b.apply(c))));
    }

    #[test]
    fn opacity_and_tint_scale_channels() {
        let c = [0.5, 0.5, 0.5, 0.8];
        assert!(close(
            ColourMatrix::opacity(0.5).apply(c),
            [0.5, 0.5, 0.5, 0.4]
        ));
        let t = ColourMatrix::tint(crate::colour::Colour::linear(1.0, 0.5, 0.0, 1.0));
        assert!(close(t.apply(c), [0.5, 0.25, 0.0, 0.8]));
        assert!(ColourMatrix::default().is_identity());
    }
}
