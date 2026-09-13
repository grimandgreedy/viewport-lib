//! Texture submission payload: the CPU-side image a consumer hands to the
//! renderer for upload, carrying the colour space its pixels are in.
//!
//! The space rides with the pixels rather than being implied by which upload
//! function you call, so a loader can build the payload on a worker thread,
//! cache it, or return it across its own boundary without the space going
//! missing on the way.

use crate::colour::ColourSpace;
use crate::error::{ViewportError, ViewportResult};

/// Which bind-group slot an uploaded texture occupies, and so what the renderer
/// treats it as. Distinct from the colour space: a normal map and a roughness
/// map are both linear, but only one of them binds as a normal map.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum TextureRole {
    /// A colour or data image, bound in the albedo slot.
    Image,
    /// A tangent-space normal map, bound in the normal-map slot.
    NormalMap,
}

/// The pixel payload of a [`TextureData`], in the precision it was authored at.
///
/// Deliberately exhaustive, unlike the types around it. The renderer maps each
/// variant to a GPU texture format, so a new variant needs a new path there and
/// is a breaking change whatever this attribute says; marking it
/// `non_exhaustive` would only force a dead arm at the one place that has to
/// handle all of them.
#[derive(Clone, PartialEq)]
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

    /// Element count: bytes for `Rgba8`, floats for `Rgba32F`.
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
/// # fn demo(albedo: Vec<u8>, roughness: Vec<u8>, crater: Vec<u8>) {
/// let albedo = TextureData::srgb(128, 128, albedo);          // a colour image
/// let roughness = TextureData::linear(128, 128, roughness);  // a data map
/// let crater = TextureData::normal_map(128, 128, crater);    // linear, binds as a normal map
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
///
/// # Size
///
/// The constructors do not check the payload length against the dimensions; the
/// upload does, returning
/// [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
/// exactly as it always has. Building a payload is therefore infallible, so a
/// call site carries one `Result` rather than two.
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub struct TextureData {
    // Private so the description cannot drift from the payload after
    // construction: the renderer reads both together when it picks a format.
    width: u32,
    height: u32,
    colour_space: ColourSpace,
    role: TextureRole,
    payload: TexturePayload,
}

impl TextureData {
    /// An 8-bit RGBA image whose pixels are sRGB-encoded colour.
    ///
    /// Use this for base colour, emissive, sprite art, and decal albedo: images
    /// holding a colour rather than a number. `rgba` should be
    /// `width * height * 4` bytes; the length is checked at upload.
    pub fn srgb(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        Self {
            width,
            height,
            colour_space: ColourSpace::Srgb,
            role: TextureRole::Image,
            payload: TexturePayload::Rgba8(rgba),
        }
    }

    /// An 8-bit RGBA image whose pixels are linear data.
    ///
    /// Use this for roughness, metallic, combined metallic-roughness (ORM), and
    /// ambient occlusion: images holding numbers rather than a colour. For a
    /// tangent-space normal map use [`normal_map`](Self::normal_map), which is
    /// also linear but binds into a different slot.
    pub fn linear(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        Self {
            width,
            height,
            colour_space: ColourSpace::Linear,
            role: TextureRole::Image,
            payload: TexturePayload::Rgba8(rgba),
        }
    }

    /// A tangent-space normal map: linear, and bound into the normal-map slot
    /// rather than the albedo one.
    ///
    /// Named separately from [`linear`](Self::linear) for both reasons. The slot
    /// differs, so the two are not interchangeable; and this is the case that
    /// most often gets handed an sRGB texture by mistake, where the encode bends
    /// every normal the same way and renders as a lit surface with a directional
    /// artefact rather than as an error.
    pub fn normal_map(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        Self {
            width,
            height,
            colour_space: ColourSpace::Linear,
            role: TextureRole::NormalMap,
            payload: TexturePayload::Rgba8(rgba),
        }
    }

    /// A 32-bit float RGBA image. Always linear.
    ///
    /// Use this for baked lightmaps and anything else whose values exceed the
    /// display range; the 8-bit paths clamp at upload and lose them.
    pub fn hdr(width: u32, height: u32, rgba: Vec<f32>) -> Self {
        Self {
            width,
            height,
            colour_space: ColourSpace::Linear,
            role: TextureRole::Image,
            payload: TexturePayload::Rgba32F(rgba),
        }
    }

    /// Width in texels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in texels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The colour space the pixels are in. The renderer picks the texture
    /// format from this, and checks it against the space the slot requires.
    pub fn colour_space(&self) -> ColourSpace {
        self.colour_space
    }

    /// Which slot the texture binds into.
    pub fn role(&self) -> TextureRole {
        self.role
    }

    /// The pixels.
    pub fn payload(&self) -> &TexturePayload {
        &self.payload
    }

    /// Check the payload length against the dimensions. The upload calls this
    /// before touching the GPU.
    ///
    /// # Errors
    ///
    /// [`ViewportError::InvalidTextureData`](crate::error::ViewportError::InvalidTextureData)
    /// when the length does not match `width * height * 4`.
    pub fn validate(&self) -> ViewportResult<()> {
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        if self.payload.len() != expected {
            return Err(ViewportError::InvalidTextureData {
                expected,
                actual: self.payload.len(),
            });
        }
        Ok(())
    }

    /// Take the pixels, leaving the description behind. Lets the renderer move
    /// the payload onto an upload worker without copying it.
    pub fn into_payload(self) -> TexturePayload {
        self.payload
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
            .field("colour_space", &self.colour_space)
            .field("role", &self.role)
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
            TextureData::srgb(4, 4, px.clone()).colour_space(),
            ColourSpace::Srgb
        );
        assert_eq!(
            TextureData::linear(4, 4, px.clone()).colour_space(),
            ColourSpace::Linear
        );
        assert_eq!(
            TextureData::normal_map(4, 4, px).colour_space(),
            ColourSpace::Linear
        );
    }

    #[test]
    fn a_normal_map_is_linear_but_not_a_plain_data_image() {
        // Both are linear, so the space alone cannot tell them apart. The role
        // is what routes them to different bind-group slots.
        let px = vec![0u8; 4 * 4 * 4];
        let data = TextureData::linear(4, 4, px.clone());
        let normal = TextureData::normal_map(4, 4, px);
        assert_eq!(data.colour_space(), normal.colour_space());
        assert_eq!(data.role(), TextureRole::Image);
        assert_eq!(normal.role(), TextureRole::NormalMap);
    }

    #[test]
    fn float_payloads_are_always_linear() {
        let texels = vec![0.0f32; 2 * 2 * 4];
        assert_eq!(
            TextureData::hdr(2, 2, texels).colour_space(),
            ColourSpace::Linear
        );
    }

    #[test]
    fn pixels_are_passed_through_untouched() {
        // The whole point of labelling rather than converting: an sRGB payload
        // reaches the GPU byte-identical, and the sampler does the decode.
        let px: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let data = TextureData::srgb(4, 4, px.clone());
        match data.payload() {
            TexturePayload::Rgba8(v) => assert_eq!(v, &px),
            _ => panic!("expected an 8-bit payload"),
        }
    }

    #[test]
    fn validate_checks_the_length_against_the_dimensions() {
        assert!(TextureData::srgb(4, 4, vec![0u8; 64]).validate().is_ok());
        let err = TextureData::srgb(4, 4, vec![0u8; 10])
            .validate()
            .unwrap_err();
        match err {
            ViewportError::InvalidTextureData { expected, actual } => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 10);
            }
            other => panic!("expected InvalidTextureData, got {other:?}"),
        }
    }

    #[test]
    fn float_payloads_are_measured_in_floats_not_bytes() {
        // width * height * 4 elements, where an element is an f32 here and a
        // byte above. Getting this wrong would reject every HDR upload.
        assert!(TextureData::hdr(2, 2, vec![0.0f32; 16]).validate().is_ok());
        assert!(TextureData::hdr(2, 2, vec![0.0f32; 64]).validate().is_err());
    }

    #[test]
    fn building_is_infallible_so_call_sites_carry_one_result() {
        // Regression guard for the ergonomics decision: a constructor that
        // returned Result would force `?` here and again at the upload.
        let data = TextureData::linear(2, 3, vec![7u8; 2 * 3 * 4]);
        assert_eq!(data.width(), 2);
        assert_eq!(data.height(), 3);
        assert_eq!(data.into_payload().texel_count(), 6);
    }

    #[test]
    fn debug_summarises_the_payload_rather_than_printing_it() {
        let data = TextureData::srgb(4, 4, vec![0u8; 64]);
        let shown = format!("{data:?}");
        assert!(shown.contains("Rgba8(64)"), "got {shown}");
        assert!(shown.contains("Srgb"), "got {shown}");
    }
}
