//! An RGBA colour, stored in linear space.
//!
//! The renderer works in linear light and writes to an sRGB target, so the
//! hardware applies a linear->sRGB encode on write. A colour taken from a hex
//! code or a colour picker is an sRGB value; handing those bytes straight to the
//! renderer encodes them a second time and washes them out. [`Colour`] removes
//! that trap: you build it from whatever space your value is in (`srgb`, `rgb`,
//! `hex`, `hsl`, or `linear`) and it stores the linear form, so every colour
//! reaching the GPU is already in the space the pipeline expects.
//!
//! Pick the constructor by where the value came from:
//!
//! - A hex code, a colour picker, a design token: [`Colour::hex`],
//!   [`Colour::rgb`], [`Colour::srgb`], [`Colour::hsl`]. These decode sRGB to
//!   linear for you.
//! - A light colour, a value you computed in linear space, a colourmap sample:
//!   [`Colour::linear`]. This passes through untouched.

/// Decode one 0..=1 sRGB channel to linear (the sRGB electro-optical transfer
/// function). Alpha is not an sRGB quantity and must not be passed through here.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode one linear channel back to 0..=1 sRGB (the inverse of
/// [`srgb_to_linear`]). Useful for reporting a stored linear colour as the hex
/// a consumer would recognise.
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// The colour space a set of channel values is expressed in.
///
/// Used at boundaries where the same byte layout can mean either space: an
/// 8-bit texture is sRGB when it holds a base-colour image and linear when it
/// holds data (roughness, metallic, a normal map). Direct [`Colour`] values do
/// not need this, since the constructor already fixes the space.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColourSpace {
    /// sRGB-encoded: decode to linear before use.
    Srgb,
    /// Already linear: use as-is.
    Linear,
}

/// An RGBA colour in linear space, alpha last.
///
/// Construct it from the space your value is in; it stores the linear form.
/// [`to_linear_rgba`](Self::to_linear_rgba) and
/// [`to_linear_rgb`](Self::to_linear_rgb) hand the pipeline the array it wants.
///
/// ```
/// use viewport_lib_types::colour::Colour;
/// // Navy #183054 from a hex code renders faithfully:
/// let navy = Colour::hex("#183054").unwrap();
/// // A light colour is already linear:
/// let warm = Colour::linear_rgb(1.0, 0.9, 0.7);
/// ```
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Colour([f32; 4]);

impl Colour {
    /// Opaque white.
    pub const WHITE: Colour = Colour([1.0, 1.0, 1.0, 1.0]);
    /// Opaque black.
    pub const BLACK: Colour = Colour([0.0, 0.0, 0.0, 1.0]);
    /// Fully transparent (all channels zero).
    pub const TRANSPARENT: Colour = Colour([0.0, 0.0, 0.0, 0.0]);

    /// A linear RGBA colour, stored verbatim. Use this for light colours,
    /// colourmap samples, and any value already computed in linear space.
    pub const fn linear(r: f32, g: f32, b: f32, a: f32) -> Self {
        Colour([r, g, b, a])
    }

    /// A linear RGB colour with alpha 1.0. See [`linear`](Self::linear).
    pub const fn linear_rgb(r: f32, g: f32, b: f32) -> Self {
        Colour([r, g, b, 1.0])
    }

    /// An sRGB RGBA colour with 0..=1 channels. The RGB channels are decoded to
    /// linear; alpha is a linear coverage value and is stored as given.
    pub fn srgb(r: f32, g: f32, b: f32, a: f32) -> Self {
        Colour([srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a])
    }

    /// An sRGB RGB colour with 0..=1 channels and alpha 1.0. See
    /// [`srgb`](Self::srgb).
    pub fn srgb_rgb(r: f32, g: f32, b: f32) -> Self {
        Self::srgb(r, g, b, 1.0)
    }

    /// An sRGB RGBA colour from 8-bit channels (the form a hex code or colour
    /// picker gives). RGB is decoded to linear; alpha is scaled to 0..=1.
    pub fn srgb_u8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::srgb(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        )
    }

    /// An opaque sRGB colour from 8-bit channels, `rgb(r, g, b)`. Alpha 255.
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::srgb_u8(r, g, b, 255)
    }

    /// An sRGB colour from 8-bit channels including alpha, `rgba(r, g, b, a)`.
    pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self::srgb_u8(r, g, b, a)
    }

    /// Parse an sRGB hex code: `#RGB`, `#RRGGBB`, or `#RRGGBBAA` (the leading
    /// `#` is optional). RGB is decoded to linear; alpha defaults to opaque when
    /// absent.
    pub fn hex(s: &str) -> Result<Self, ColourParseError> {
        let h = s.strip_prefix('#').unwrap_or(s);
        // Expand the shorthand #RGB to #RRGGBB.
        let expanded: String;
        let h = if h.len() == 3 {
            expanded = h.chars().flat_map(|c| [c, c]).collect();
            expanded.as_str()
        } else {
            h
        };
        let byte = |i: usize| -> Result<u8, ColourParseError> {
            u8::from_str_radix(&h[i..i + 2], 16).map_err(|_| ColourParseError::BadDigit)
        };
        match h.len() {
            6 => Ok(Self::rgb(byte(0)?, byte(2)?, byte(4)?)),
            8 => Ok(Self::rgba(byte(0)?, byte(2)?, byte(4)?, byte(6)?)),
            _ => Err(ColourParseError::BadLength),
        }
    }

    /// An sRGB colour from hue-saturation-lightness. `h` is in degrees
    /// (wrapped into 0..360), `s` and `l` are 0..=1. The result is treated as
    /// sRGB and decoded to linear. Alpha 1.0.
    pub fn hsl(h: f32, s: f32, l: f32) -> Self {
        let (r, g, b) = hsl_to_rgb(h, s, l);
        Self::srgb_rgb(r, g, b)
    }

    /// An sRGB colour from hue-saturation-value. `h` is in degrees (wrapped
    /// into 0..360), `s` and `v` are 0..=1. The result is treated as sRGB and
    /// decoded to linear. Alpha 1.0.
    pub fn hsv(h: f32, s: f32, v: f32) -> Self {
        let (r, g, b) = hsv_to_rgb(h, s, v);
        Self::srgb_rgb(r, g, b)
    }

    /// This colour with its alpha replaced (alpha is linear coverage, 0..=1).
    pub fn with_alpha(self, a: f32) -> Self {
        Colour([self.0[0], self.0[1], self.0[2], a])
    }

    /// The linear RGBA array the GPU consumes.
    pub const fn to_linear_rgba(self) -> [f32; 4] {
        self.0
    }

    /// The linear RGB array, dropping alpha. For fields that carry colour
    /// without an alpha channel (base colour, emissive, light colour).
    pub const fn to_linear_rgb(self) -> [f32; 3] {
        [self.0[0], self.0[1], self.0[2]]
    }

    /// The alpha channel (linear coverage, 0..=1).
    pub const fn alpha(self) -> f32 {
        self.0[3]
    }

    /// Re-encode to opaque-agnostic 8-bit sRGB channels `[r, g, b, a]`, the
    /// inverse of [`srgb_u8`](Self::srgb_u8). Handy for logging a stored colour
    /// as the hex a consumer would recognise.
    pub fn to_srgb_u8(self) -> [u8; 4] {
        let enc = |c: f32| (linear_to_srgb(c).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        [
            enc(self.0[0]),
            enc(self.0[1]),
            enc(self.0[2]),
            (self.0[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ]
    }
}

/// A linear `[f32; 4]` is taken as-is. Bare arrays are treated as linear, so
/// existing linear values keep working; reach for a constructor
/// ([`Colour::rgb`], [`Colour::hex`], ...) when the value is sRGB.
impl From<[f32; 4]> for Colour {
    fn from(v: [f32; 4]) -> Self {
        Colour(v)
    }
}

/// A linear `[f32; 3]` is taken as-is, with alpha 1.0.
impl From<[f32; 3]> for Colour {
    fn from(v: [f32; 3]) -> Self {
        Colour([v[0], v[1], v[2], 1.0])
    }
}

impl From<Colour> for [f32; 4] {
    fn from(c: Colour) -> Self {
        c.0
    }
}

impl From<Colour> for [f32; 3] {
    fn from(c: Colour) -> Self {
        c.to_linear_rgb()
    }
}

/// Why parsing a hex colour failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColourParseError {
    /// The string was not 3, 6, or 8 hex digits (after an optional `#`).
    BadLength,
    /// A character was not a hex digit.
    BadDigit,
}

impl core::fmt::Display for ColourParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ColourParseError::BadLength => f.write_str("hex colour must be 3, 6, or 8 hex digits"),
            ColourParseError::BadDigit => f.write_str("hex colour contains a non-hex character"),
        }
    }
}

impl std::error::Error for ColourParseError {}

/// HSL to sRGB RGB, all outputs 0..=1. `h` in degrees, `s`/`l` in 0..=1.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = hue_sector(h, c, x);
    (r + m, g + m, b + m)
}

/// HSV to sRGB RGB, all outputs 0..=1. `h` in degrees, `s`/`v` in 0..=1.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = hue_sector(h, c, x);
    (r + m, g + m, b + m)
}

/// Place the chroma/second-largest components into RGB by hue sector.
fn hue_sector(h: f32, c: f32, x: f32) -> (f32, f32, f32) {
    match h as u32 / 60 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_passthrough() {
        let c = Colour::linear(0.1, 0.2, 0.3, 0.4);
        assert_eq!(c.to_linear_rgba(), [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn srgb_decodes_and_round_trips() {
        // Navy #183054 and gold #FADA16 must re-encode to the original bytes.
        for (r, g, b) in [(24, 48, 84), (250, 218, 22)] {
            let c = Colour::rgb(r, g, b);
            assert_eq!(c.to_srgb_u8(), [r, g, b, 255]);
        }
    }

    #[test]
    fn hex_forms() {
        assert_eq!(Colour::hex("#183054").unwrap(), Colour::rgb(24, 48, 84));
        assert_eq!(Colour::hex("183054").unwrap(), Colour::rgb(24, 48, 84));
        // #RGB shorthand expands each nibble.
        assert_eq!(Colour::hex("#abc").unwrap(), Colour::rgb(0xaa, 0xbb, 0xcc));
        // #RRGGBBAA carries alpha.
        assert_eq!(
            Colour::hex("#18305480").unwrap(),
            Colour::rgba(24, 48, 84, 0x80)
        );
        assert_eq!(Colour::hex("#zz").err(), Some(ColourParseError::BadLength));
        assert_eq!(
            Colour::hex("#gggggg").err(),
            Some(ColourParseError::BadDigit)
        );
    }

    #[test]
    fn alpha_is_not_decoded() {
        // Alpha 128/255 stays a straight ratio, not an sRGB-decoded one.
        let c = Colour::rgba(0, 0, 0, 128);
        assert!((c.alpha() - 128.0 / 255.0).abs() < 1e-4);
    }

    #[test]
    fn srgb_is_darker_than_linear_for_same_bytes() {
        // The whole point: sRGB 0.5 decodes to a smaller linear value.
        let s = Colour::srgb_rgb(0.5, 0.5, 0.5).to_linear_rgb();
        assert!(s[0] < 0.5 - 0.05);
    }

    #[test]
    fn hsl_matches_known_colour() {
        // HSL(216, 56%, 21%) is navy #183054 (from the colour viewer). The HSL
        // percentages are rounded, so allow a byte of slack.
        let c = Colour::hsl(216.0, 0.56, 0.21).to_srgb_u8();
        let expected = [24u8, 48, 84, 255];
        for (got, want) in c.iter().zip(expected) {
            assert!(
                (*got as i32 - want as i32).abs() <= 1,
                "{c:?} vs {expected:?}"
            );
        }
    }
}
