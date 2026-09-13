//! Texture submission payload: the CPU-side image a consumer hands to the
//! renderer for upload, carrying the colour space its pixels are in.
//!
//! The space rides with the pixels rather than being implied by which upload
//! function you call, so a loader can build the payload on a worker thread,
//! cache it, or return it across its own boundary without the space going
//! missing on the way.

use crate::colour::ColourSpace;
use crate::error::{ViewportError, ViewportResult};

/// The pixel payload of a [`TextureData`], in the precision it was authored at.
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub enum TexturePayload {
    /// Eight bits per channel, RGBA order, four bytes per texel.
    Rgba8(Vec<u8>),
    /// Thirty-two bit float per channel, RGBA order, four floats per texel.
    /// Always linear: a float image has the range to hold linear values, so
    /// there is no reason to encode one.
    Rgba32F(Vec<f32>),
}

impl TexturePayload {
    /// Texel count held by this payload.
    pub fn texel_count(&self) -> usize {
        match self {
            TexturePayload::Rgba8(v) => v.len() / 4,
            TexturePayload::Rgba32F(v) => v.len() / 4,
        }
    }

    /// Element count, in bytes for `Rgba8` and in floats for `Rgba32F`.
    fn len(&self) -> usize {
        match self {
            TexturePayload::Rgba8(v) => v.len(),
            TexturePayload::Rgba32F(v) => v.len(),
        }
    }
}

/// An image to upload, together with the colour space of its pixels.
///
/// Build it by naming the space the bytes you hold are in, the same way
/// [`Colour`](crate::colour::Colour) is built by naming the space a colour
/// value is in:
///
/// ```
/// use viewport_lib_types::data::texture::TextureData;
///
/// # fn demo(albedo: Vec<u8>, roughness: Vec<u8>, crater: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
/// let albedo = TextureData::srgb(128, 128, albedo)?;          // a colour image
/// let roughness = TextureData::linear(128, 128, roughness)?;  // a data map
/// let crater = TextureData::normal_map(128, 128, crater)?;    // linear, named for the use
/// # Ok(())
/// # }
/// ```
///
/// # This labels the space, it does not convert
///
/// [`Colour`](crate::colour::Colour) resolves the space at construction and
/// stores linear, so afterwards there is one representation and nothing left to
/// get wrong. `TextureData` deliberately does not do that. Decoding an 8-bit
/// sRGB image to 8-bit linear would destroy precision in the darks, which is
/// what the sRGB texture formats exist to avoid: the sampler decodes on read,
/// in hardware, for free. So the pixels are passed through untouched and the
/// space is carried forward to the texture format the renderer creates.
///
/// One consequence is worth knowing. Because the space survives, the upload and
/// the slot the texture ends up in both have an opinion, and they have to agree:
/// a tangent-space normal map uploaded as sRGB decodes a neutral 128 to 0.216
/// instead of 0 and biases every normal the same way. Naming the space here is
/// what lets the renderer check that pairing instead of rendering it.
///
/// # Which space
///
/// - **sRGB** for images that hold colour a person chose or a camera captured:
///   base colour, emissive, sprite art, a decal's albedo.
/// - **Linear** for images that hold numbers: roughness, metallic, combined
///   metallic-roughness, ambient occlusion, and tangent-space normal maps.
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub struct TextureData {
    // Private, unlike the other `*Data` payloads in this module. The length of
    // the payload has to match the dimensions, and that invariant is the reason
    // the type exists: it moves the size check off the upload path. Public
    // fields would let a caller break it after construction.
    width: u32,
    height: u32,
    layers: u32,
    colour_space: ColourSpace,
    payload: TexturePayload,
}

impl TextureData {
    /// An 8-bit RGBA image whose pixels are sRGB-encoded colour.
    ///
    /// Use this for base colour, emissive, sprite art, and decal albedo: images
    /// holding a colour rather than a number. `rgba` must be exactly
    /// `width * height * 4` bytes.
    ///
    /// # Errors
    ///
    /// [`ViewportError::InvalidTextureData`] when the length does not match the
    /// dimensions.
    pub fn srgb(width: u32, height: u32, rgba: Vec<u8>) -> ViewportResult<Self> {
        Self::new(
            width,
            height,
            1,
            ColourSpace::Srgb,
            TexturePayload::Rgba8(rgba),
        )
    }

    /// An 8-bit RGBA image whose pixels are linear data.
    ///
    /// Use this for roughness, metallic, combined metallic-roughness (ORM), and
    /// ambient occlusion: images holding numbers rather than a colour. For a
    /// tangent-space normal map prefer [`normal_map`](Self::normal_map), which
    /// is the same thing named for the use. `rgba` must be exactly
    /// `width * height * 4` bytes.
    ///
    /// # Errors
    ///
    /// [`ViewportError::InvalidTextureData`] when the length does not match the
    /// dimensions.
    pub fn linear(width: u32, height: u32, rgba: Vec<u8>) -> ViewportResult<Self> {
        Self::new(
            width,
            height,
            1,
            ColourSpace::Linear,
            TexturePayload::Rgba8(rgba),
        )
    }

    /// A tangent-space normal map. Linear, identical to
    /// [`linear`](Self::linear), and named separately because this is the slot
    /// that most often gets handed an sRGB texture by mistake: the encode bends
    /// every normal the same way and renders as a lit surface with a
    /// directional artefact rather than as an error.
    ///
    /// `rgba` must be exactly `width * height * 4` bytes.
    ///
    /// # Errors
    ///
    /// [`ViewportError::InvalidTextureData`] when the length does not match the
    /// dimensions.
    pub fn normal_map(width: u32, height: u32, rgba: Vec<u8>) -> ViewportResult<Self> {
        Self::linear(width, height, rgba)
    }

    /// A 32-bit float RGBA image. Always linear.
    ///
    /// Use this for environment maps and any other image whose values exceed
    /// the display range. `rgba` must be exactly `width * height * 4` floats.
    ///
    /// # Errors
    ///
    /// [`ViewportError::InvalidTextureData`] when the length does not match the
    /// dimensions.
    pub fn hdr(width: u32, height: u32, rgba: Vec<f32>) -> ViewportResult<Self> {
        Self::new(
            width,
            height,
            1,
            ColourSpace::Linear,
            TexturePayload::Rgba32F(rgba),
        )
    }

    /// A layered 32-bit float RGBA image, layers stored back to back. Always
    /// linear.
    ///
    /// `rgba` must be exactly `width * height * 4 * layers` floats, and `layers`
    /// must be non-zero.
    ///
    /// # Errors
    ///
    /// [`ViewportError::InvalidTextureData`] when the length does not match the
    /// dimensions, or when `layers` is zero.
    pub fn hdr_layers(
        width: u32,
        height: u32,
        layers: u32,
        rgba: Vec<f32>,
    ) -> ViewportResult<Self> {
        Self::new(
            width,
            height,
            layers,
            ColourSpace::Linear,
            TexturePayload::Rgba32F(rgba),
        )
    }

    /// Width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Array layer count. `1` for every constructor except
    /// [`hdr_layers`](Self::hdr_layers).
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// The colour space the pixels are in. The renderer picks the texture
    /// format from this, and checks it against the space the slot requires.
    pub fn colour_space(&self) -> ColourSpace {
        self.colour_space
    }

    /// The pixels.
    pub fn payload(&self) -> &TexturePayload {
        &self.payload
    }

    /// Take the payload, leaving the description behind. Lets the renderer move
    /// the pixels onto an upload worker without copying them.
    pub fn into_payload(self) -> TexturePayload {
        self.payload
    }

    /// Everything at once, for an upload path that needs the dimensions and the
    /// pixels without cloning either.
    pub fn into_parts(self) -> (u32, u32, u32, ColourSpace, TexturePayload) {
        (
            self.width,
            self.height,
            self.layers,
            self.colour_space,
            self.payload,
        )
    }

    fn new(
        width: u32,
        height: u32,
        layers: u32,
        colour_space: ColourSpace,
        payload: TexturePayload,
    ) -> ViewportResult<Self> {
        let expected = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(layers as usize)
            .saturating_mul(4);
        // A zero-layer image has no expected length to compare against, so it
        // would otherwise slip through as "0 == 0".
        if layers == 0 || payload.len() != expected {
            return Err(ViewportError::InvalidTextureData {
                expected,
                actual: payload.len(),
            });
        }
        Ok(Self {
            width,
            height,
            layers,
            colour_space,
            payload,
        })
    }
}

impl std::fmt::Debug for TextureData {
    /// Prints the description and the payload's size. The pixels themselves are
    /// summarised: a texture is large enough that dumping it buries whatever
    /// else is being logged.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, len) = match &self.payload {
            TexturePayload::Rgba8(v) => ("Rgba8", v.len()),
            TexturePayload::Rgba32F(v) => ("Rgba32F", v.len()),
        };
        f.debug_struct("TextureData")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("layers", &self.layers)
            .field("colour_space", &self.colour_space)
            .field("payload", &format_args!("{kind}({len})"))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_record_the_space_they_name() {
        let px = vec![0u8; 4 * 4 * 4];
        assert_eq!(
            TextureData::srgb(4, 4, px.clone()).unwrap().colour_space(),
            ColourSpace::Srgb
        );
        assert_eq!(
            TextureData::linear(4, 4, px.clone())
                .unwrap()
                .colour_space(),
            ColourSpace::Linear
        );
        assert_eq!(
            TextureData::normal_map(4, 4, px).unwrap().colour_space(),
            ColourSpace::Linear
        );
    }

    #[test]
    fn float_payloads_are_always_linear() {
        let texels = vec![0.0f32; 2 * 2 * 4];
        assert_eq!(
            TextureData::hdr(2, 2, texels.clone())
                .unwrap()
                .colour_space(),
            ColourSpace::Linear
        );
        assert_eq!(
            TextureData::hdr_layers(1, 2, 2, texels)
                .unwrap()
                .colour_space(),
            ColourSpace::Linear
        );
    }

    #[test]
    fn pixels_are_passed_through_untouched() {
        // The whole point of labelling rather than converting: an sRGB payload
        // reaches the GPU byte-identical, and the sampler does the decode.
        let px: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let data = TextureData::srgb(4, 4, px.clone()).unwrap();
        match data.payload() {
            TexturePayload::Rgba8(v) => assert_eq!(v, &px),
            _ => panic!("expected an 8-bit payload"),
        }
    }

    #[test]
    fn length_is_checked_against_the_dimensions() {
        let err = TextureData::srgb(4, 4, vec![0u8; 10]).unwrap_err();
        match err {
            ViewportError::InvalidTextureData { expected, actual } => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 10);
            }
            other => panic!("expected InvalidTextureData, got {other:?}"),
        }
    }

    #[test]
    fn layered_length_accounts_for_the_layers() {
        let one_layer = vec![0.0f32; 2 * 2 * 4];
        assert!(TextureData::hdr_layers(2, 2, 2, one_layer).is_err());
        assert!(TextureData::hdr_layers(2, 2, 2, vec![0.0f32; 2 * 2 * 4 * 2]).is_ok());
    }

    #[test]
    fn zero_layers_is_rejected_rather_than_matching_an_empty_payload() {
        assert!(TextureData::hdr_layers(2, 2, 0, Vec::new()).is_err());
    }

    #[test]
    fn into_parts_round_trips_the_description() {
        let data = TextureData::linear(2, 3, vec![7u8; 2 * 3 * 4]).unwrap();
        let (w, h, layers, space, payload) = data.into_parts();
        assert_eq!((w, h, layers), (2, 3, 1));
        assert_eq!(space, ColourSpace::Linear);
        assert_eq!(payload.texel_count(), 6);
    }

    #[test]
    fn debug_summarises_the_payload_rather_than_printing_it() {
        let data = TextureData::srgb(4, 4, vec![0u8; 64]).unwrap();
        let shown = format!("{data:?}");
        assert!(shown.contains("Rgba8(64)"), "got {shown}");
        assert!(shown.contains("Srgb"), "got {shown}");
    }
}
