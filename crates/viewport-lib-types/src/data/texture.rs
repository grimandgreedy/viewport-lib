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
    /// Block-compressed, with its mip chain already built: one buffer per level,
    /// level 0 (full size) first, each tightly block-packed. The library does no
    /// encoding; compress offline in an asset pipeline. Whether colour formats
    /// decode as sRGB comes from the [`TextureData`]'s colour space.
    Compressed {
        /// The block encoding.
        format: CompressedFormat,
        /// Block bytes per mip level, level 0 first.
        mip_levels: Vec<Vec<u8>>,
    },
}

/// A block-compressed encoding, for [`TexturePayload::Compressed`].
///
/// Names the encoding only. Whether a colour format decodes as sRGB is the
/// [`TextureData`]'s colour space, so the space is stated the same way for every
/// payload; the data-only formats (BC4, BC5, BC6H, EAC, ASTC HDR) must be
/// linear. Each format needs a device feature (`TEXTURE_COMPRESSION_BC`, `_ETC2`,
/// `_ASTC`, `_ASTC_HDR`), which the upload checks.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompressedFormat {
    /// BC1: RGB with 1-bit alpha, 8 bytes per 4x4 block.
    Bc1Rgba,
    /// BC2: RGBA with explicit 4-bit alpha, 16 bytes per block.
    Bc2Rgba,
    /// BC3: RGBA with interpolated alpha, 16 bytes per block.
    Bc3Rgba,
    /// BC4: one unsigned channel, 8 bytes per block.
    Bc4R,
    /// BC4: one signed channel.
    Bc4RSigned,
    /// BC5: two unsigned channels, the usual normal-map format.
    Bc5Rg,
    /// BC5: two signed channels.
    Bc5RgSigned,
    /// BC6H: unsigned half-float RGB, the usual HDR format.
    Bc6hRgb,
    /// BC6H: signed half-float RGB.
    Bc6hRgbSigned,
    /// BC7: high-quality RGBA, 16 bytes per block.
    Bc7Rgba,
    /// ETC2 RGB, 8 bytes per 4x4 block.
    Etc2Rgb8,
    /// ETC2 RGB with 1-bit alpha.
    Etc2Rgb8A1,
    /// ETC2 RGBA, 16 bytes per block.
    Etc2Rgba8,
    /// EAC: one unsigned 11-bit channel.
    EacR11,
    /// EAC: one signed 11-bit channel.
    EacR11Signed,
    /// EAC: two unsigned 11-bit channels.
    EacRg11,
    /// EAC: two signed 11-bit channels.
    EacRg11Signed,
    /// ASTC, 16 bytes per block of the given size. `hdr` selects the HDR
    /// profile, which is always linear.
    Astc {
        /// Block footprint in texels.
        block: AstcBlock,
        /// The HDR profile rather than LDR.
        hdr: bool,
    },
}

/// An ASTC block footprint, width by height in texels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum AstcBlock {
    B4x4,
    B5x4,
    B5x5,
    B6x5,
    B6x6,
    B8x5,
    B8x6,
    B8x8,
    B10x5,
    B10x6,
    B10x8,
    B10x10,
    B12x10,
    B12x12,
}

impl AstcBlock {
    fn dimensions(self) -> (u32, u32) {
        match self {
            AstcBlock::B4x4 => (4, 4),
            AstcBlock::B5x4 => (5, 4),
            AstcBlock::B5x5 => (5, 5),
            AstcBlock::B6x5 => (6, 5),
            AstcBlock::B6x6 => (6, 6),
            AstcBlock::B8x5 => (8, 5),
            AstcBlock::B8x6 => (8, 6),
            AstcBlock::B8x8 => (8, 8),
            AstcBlock::B10x5 => (10, 5),
            AstcBlock::B10x6 => (10, 6),
            AstcBlock::B10x8 => (10, 8),
            AstcBlock::B10x10 => (10, 10),
            AstcBlock::B12x10 => (12, 10),
            AstcBlock::B12x12 => (12, 12),
        }
    }
}

impl CompressedFormat {
    /// Block footprint in texels, width by height.
    pub fn block_dimensions(self) -> (u32, u32) {
        match self {
            CompressedFormat::Astc { block, .. } => block.dimensions(),
            _ => (4, 4),
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            CompressedFormat::Bc1Rgba
            | CompressedFormat::Bc4R
            | CompressedFormat::Bc4RSigned
            | CompressedFormat::Etc2Rgb8
            | CompressedFormat::Etc2Rgb8A1
            | CompressedFormat::EacR11
            | CompressedFormat::EacR11Signed => 8,
            CompressedFormat::Bc2Rgba
            | CompressedFormat::Bc3Rgba
            | CompressedFormat::Bc5Rg
            | CompressedFormat::Bc5RgSigned
            | CompressedFormat::Bc6hRgb
            | CompressedFormat::Bc6hRgbSigned
            | CompressedFormat::Bc7Rgba
            | CompressedFormat::Etc2Rgba8
            | CompressedFormat::EacRg11
            | CompressedFormat::EacRg11Signed
            | CompressedFormat::Astc { .. } => 16,
        }
    }

    /// Whether the format holds colour that can be sRGB-encoded. The data-only
    /// formats (one or two channels, signed, or HDR) are always linear.
    pub fn holds_colour(self) -> bool {
        match self {
            CompressedFormat::Bc1Rgba
            | CompressedFormat::Bc2Rgba
            | CompressedFormat::Bc3Rgba
            | CompressedFormat::Bc7Rgba
            | CompressedFormat::Etc2Rgb8
            | CompressedFormat::Etc2Rgb8A1
            | CompressedFormat::Etc2Rgba8 => true,
            CompressedFormat::Astc { hdr, .. } => !hdr,
            CompressedFormat::Bc4R
            | CompressedFormat::Bc4RSigned
            | CompressedFormat::Bc5Rg
            | CompressedFormat::Bc5RgSigned
            | CompressedFormat::Bc6hRgb
            | CompressedFormat::Bc6hRgbSigned
            | CompressedFormat::EacR11
            | CompressedFormat::EacR11Signed
            | CompressedFormat::EacRg11
            | CompressedFormat::EacRg11Signed => false,
        }
    }

    /// Byte length of one tightly block-packed level of `width` x `height`
    /// texels. Partial blocks at the edges count as whole blocks.
    pub fn level_bytes(self, width: u32, height: u32) -> usize {
        let (bw, bh) = self.block_dimensions();
        let blocks_x = width.div_ceil(bw) as usize;
        let blocks_y = height.div_ceil(bh) as usize;
        blocks_x * blocks_y * self.block_bytes()
    }
}

impl TexturePayload {
    /// Texel count held by this payload. For a compressed payload this counts
    /// the base level, in whole blocks.
    pub fn texel_count(&self) -> usize {
        match self {
            TexturePayload::Rgba8(v) => v.len() / 4,
            TexturePayload::Rgba32F(v) => v.len() / 4,
            TexturePayload::Compressed { format, mip_levels } => {
                let (bw, bh) = format.block_dimensions();
                let base = mip_levels.first().map_or(0, Vec::len);
                base / format.block_bytes() * (bw * bh) as usize
            }
        }
    }
}

/// The kind of upload that rejected a [`TextureData`], in
/// [`ViewportError::UnsupportedTextureData`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum UploadSlot {
    /// A one-shot overlay texture.
    Overlay,
    /// A streaming overlay texture.
    OverlayStreaming,
    /// A matcap.
    Matcap,
    /// An environment.
    Environment,
}

/// Why an upload rejected a well-formed [`TextureData`], in
/// [`ViewportError::UnsupportedTextureData`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum TextureRejection {
    /// The slot needs the other colour space. Relabel with
    /// [`TextureData::with_colour_space`] if the pixels really are in it.
    WrongColourSpace,
    /// The slot needs a particular size.
    WrongSize,
    /// The slot cannot take this kind of payload (float, or compressed).
    UnsupportedPayload,
    /// The slot has no use for a normal map.
    NormalMap,
}

impl std::fmt::Display for UploadSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            UploadSlot::Overlay => "an overlay texture",
            UploadSlot::OverlayStreaming => "a streaming overlay texture",
            UploadSlot::Matcap => "a matcap",
            UploadSlot::Environment => "an environment",
        })
    }
}

impl std::fmt::Display for TextureRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TextureRejection::WrongColourSpace => "it needs the other colour space",
            TextureRejection::WrongSize => "it needs a different size",
            TextureRejection::UnsupportedPayload => "it cannot take this kind of payload",
            TextureRejection::NormalMap => "it cannot take a normal map",
        })
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

    /// A block-compressed image with its mip chain already built, holding colour
    /// or data according to `colour_space`. See [`TexturePayload::Compressed`].
    ///
    /// The data-only formats (BC4, BC5, BC6H, EAC, ASTC HDR) must be linear;
    /// [`validate`](Self::validate) rejects them labelled sRGB.
    pub fn compressed(
        width: u32,
        height: u32,
        format: CompressedFormat,
        colour_space: ColourSpace,
        mip_levels: Vec<Vec<u8>>,
    ) -> Self {
        Self {
            width,
            height,
            colour_space,
            role: TextureRole::Image,
            payload: TexturePayload::Compressed { format, mip_levels },
        }
    }

    /// A block-compressed tangent-space normal map: linear, bound into the
    /// normal-map slot. BC5 is the usual format.
    pub fn compressed_normal_map(
        width: u32,
        height: u32,
        format: CompressedFormat,
        mip_levels: Vec<Vec<u8>>,
    ) -> Self {
        Self {
            width,
            height,
            colour_space: ColourSpace::Linear,
            role: TextureRole::NormalMap,
            payload: TexturePayload::Compressed { format, mip_levels },
        }
    }

    /// The same pixels, labelled with another colour space. Nothing is
    /// converted.
    ///
    /// For a loader that cannot tell whether a file holds colour or data: it
    /// returns sRGB, and the caller relabels a data map as linear without
    /// rebuilding the payload. A label that cannot be right (an sRGB normal map,
    /// float pixels, or a data-only compressed format) is rejected by
    /// [`validate`](Self::validate).
    pub fn with_colour_space(mut self, colour_space: ColourSpace) -> Self {
        self.colour_space = colour_space;
        self
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

    /// Check the payload against the dimensions, and the colour space against
    /// the payload and role. The upload calls this before touching the GPU.
    ///
    /// # Errors
    ///
    /// - [`ViewportError::InvalidTextureData`] when an 8-bit or float payload's
    ///   length does not match `width * height * 4`.
    /// - [`ViewportError::CompressedTextureNotBlockAligned`] and
    ///   [`ViewportError::InvalidCompressedTextureData`] when a compressed
    ///   payload's dimensions or level sizes do not fit its block format.
    /// - [`ViewportError::InvalidTextureColourSpace`] for a label that cannot be
    ///   right: an sRGB normal map, sRGB float pixels, or a data-only compressed
    ///   format labelled sRGB.
    pub fn validate(&self) -> ViewportResult<()> {
        let expected = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(4);
        let srgb = self.colour_space == ColourSpace::Srgb;
        if srgb && self.role == TextureRole::NormalMap {
            return Err(ViewportError::InvalidTextureColourSpace {
                reason: "a normal map holds directions, not colour, so it must be linear",
            });
        }
        match &self.payload {
            TexturePayload::Rgba8(v) => {
                if v.len() != expected {
                    return Err(ViewportError::InvalidTextureData {
                        expected,
                        actual: v.len(),
                    });
                }
            }
            TexturePayload::Rgba32F(v) => {
                if v.len() != expected {
                    return Err(ViewportError::InvalidTextureData {
                        expected,
                        actual: v.len(),
                    });
                }
                if srgb {
                    return Err(ViewportError::InvalidTextureColourSpace {
                        reason: "float pixels are always linear",
                    });
                }
            }
            TexturePayload::Compressed { format, mip_levels } => {
                if srgb && !format.holds_colour() {
                    return Err(ViewportError::InvalidTextureColourSpace {
                        reason: "this compressed format holds data, not colour, so it must be linear",
                    });
                }
                let (bw, bh) = format.block_dimensions();
                if self.width % bw != 0 || self.height % bh != 0 {
                    return Err(ViewportError::CompressedTextureNotBlockAligned {
                        width: self.width,
                        height: self.height,
                        block_width: bw,
                        block_height: bh,
                    });
                }
                if mip_levels.is_empty() {
                    return Err(ViewportError::InvalidCompressedTextureData {
                        level: 0,
                        expected: format.level_bytes(self.width, self.height),
                        actual: 0,
                    });
                }
                for (level, data) in mip_levels.iter().enumerate() {
                    let w = (self.width >> level).max(1);
                    let h = (self.height >> level).max(1);
                    let expected = format.level_bytes(w, h);
                    if data.len() != expected {
                        return Err(ViewportError::InvalidCompressedTextureData {
                            level: level as u32,
                            expected,
                            actual: data.len(),
                        });
                    }
                }
            }
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
            TexturePayload::Compressed { mip_levels, .. } => {
                ("Compressed", mip_levels.iter().map(Vec::len).sum())
            }
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
    fn relabelling_changes_only_the_space() {
        let px: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let data = TextureData::srgb(4, 4, px.clone()).with_colour_space(ColourSpace::Linear);
        assert_eq!(data.colour_space(), ColourSpace::Linear);
        assert_eq!(data.role(), TextureRole::Image);
        assert!(data.validate().is_ok());
        assert!(data.payload() == &TexturePayload::Rgba8(px));
    }

    #[test]
    fn labels_that_cannot_be_right_fail_validation() {
        let srgb_normal =
            TextureData::normal_map(4, 4, vec![0u8; 64]).with_colour_space(ColourSpace::Srgb);
        let srgb_float = TextureData::hdr(2, 2, vec![0.0; 16]).with_colour_space(ColourSpace::Srgb);
        let srgb_bc5 = TextureData::compressed(
            4,
            4,
            CompressedFormat::Bc5Rg,
            ColourSpace::Srgb,
            vec![vec![0u8; 16]],
        );
        for data in [srgb_normal, srgb_float, srgb_bc5] {
            assert!(
                matches!(
                    data.validate(),
                    Err(ViewportError::InvalidTextureColourSpace { .. })
                ),
                "{data:?} should be rejected"
            );
        }
        // A colour block format may be either.
        let srgb_bc7 = TextureData::compressed(
            4,
            4,
            CompressedFormat::Bc7Rgba,
            ColourSpace::Srgb,
            vec![vec![0u8; 16]],
        );
        assert!(srgb_bc7.validate().is_ok());
    }

    #[test]
    fn compressed_levels_are_measured_in_blocks() {
        // 8x8 BC1: 2x2 blocks of 8 bytes at level 0, then one block for each of
        // the 4x4, 2x2 and 1x1 levels.
        let levels = vec![vec![0u8; 32], vec![0u8; 8], vec![0u8; 8], vec![0u8; 8]];
        let data =
            TextureData::compressed(8, 8, CompressedFormat::Bc1Rgba, ColourSpace::Srgb, levels);
        assert!(data.validate().is_ok());
        assert_eq!(data.into_payload().texel_count(), 64);

        let short = TextureData::compressed(
            8,
            8,
            CompressedFormat::Bc1Rgba,
            ColourSpace::Srgb,
            vec![vec![0u8; 32], vec![0u8; 4]],
        );
        assert!(matches!(
            short.validate(),
            Err(ViewportError::InvalidCompressedTextureData {
                level: 1,
                expected: 8,
                actual: 4
            })
        ));

        let ragged = TextureData::compressed(
            6,
            8,
            CompressedFormat::Bc7Rgba,
            ColourSpace::Srgb,
            vec![vec![0u8; 64]],
        );
        assert!(matches!(
            ragged.validate(),
            Err(ViewportError::CompressedTextureNotBlockAligned { .. })
        ));

        let empty = TextureData::compressed(
            4,
            4,
            CompressedFormat::Bc7Rgba,
            ColourSpace::Srgb,
            Vec::new(),
        );
        assert!(matches!(
            empty.validate(),
            Err(ViewportError::InvalidCompressedTextureData { actual: 0, .. })
        ));
    }

    #[test]
    fn astc_blocks_follow_their_footprint() {
        let format = CompressedFormat::Astc {
            block: AstcBlock::B6x5,
            hdr: false,
        };
        assert_eq!(format.block_dimensions(), (6, 5));
        // 12x10 is 2x2 blocks of 16 bytes.
        assert_eq!(format.level_bytes(12, 10), 64);
        assert!(format.holds_colour());
        assert!(
            !CompressedFormat::Astc {
                block: AstcBlock::B4x4,
                hdr: true
            }
            .holds_colour()
        );
    }

    #[test]
    fn debug_summarises_the_payload_rather_than_printing_it() {
        let data = TextureData::srgb(4, 4, vec![0u8; 64]);
        let shown = format!("{data:?}");
        assert!(shown.contains("Rgba8(64)"), "got {shown}");
        assert!(shown.contains("Srgb"), "got {shown}");
    }
}
