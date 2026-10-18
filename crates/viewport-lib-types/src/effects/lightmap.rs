//! Baked lightmap settings: which textures a lightmap carries and how it
//! combines with the shader's ambient term. The renderer registers one per mesh
//! with `DeviceResources::set_lightmap`.

/// Which baked textures a lightmap carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LightmapData {
    /// A single RGB radiance texture. Normal maps do not respond to the baked
    /// light in this mode.
    NonDirectional {
        /// Baked incoming radiance, sampled by UV1. Radiance, not colour, so
        /// upload it linear ([`TextureData::hdr`](crate::data::texture::TextureData::hdr)
        /// for values above 1, [`TextureData::linear`](crate::data::texture::TextureData::linear)
        /// for 8-bit).
        radiance: crate::ids::TextureId,
    },
    /// A radiance texture plus a per-texel dominant light direction, so a normal
    /// map responds to where the baked light came from. `direction` holds the
    /// unit dominant direction in `xyz` (world space) and a directionality factor
    /// in `w` (`0` = ambient, `1` = a single direction); upload it linear
    /// ([`TextureData::hdr`](crate::data::texture::TextureData::hdr)) so the signed
    /// direction survives.
    DominantDirection {
        /// Baked incoming radiance, sampled by UV1. Radiance, not colour, so
        /// upload it linear.
        radiance: crate::ids::TextureId,
        /// Per-texel dominant direction + directionality, sampled by UV1.
        /// Directions, not colour, so upload it linear.
        direction: crate::ids::TextureId,
    },
    /// A single-channel occlusion factor, read from the red channel. Pairs with
    /// [`LightmapMode::AmbientOcclusion`].
    AmbientOcclusion {
        /// Baked ambient occlusion, sampled by UV1. A factor, not colour, so
        /// upload it linear
        /// ([`TextureData::linear`](crate::data::texture::TextureData::linear)).
        occlusion: crate::ids::TextureId,
    },
    /// A radiance texture plus a per-light static-occluder shadowmask: the RGBA
    /// channels hold the baked visibility (0 = shadowed, 1 = lit) of up to four
    /// realtime lights (light index `i` -> channel `i`). The lit shaders combine
    /// it with each light's realtime shadow by `min`, so static shadows come from
    /// the bake (crisp at any distance) and dynamic ones from the shadow map,
    /// while the lights stay fully dynamic (Unity Shadowmask parity). Bake the
    /// shadowmask with the renderer's `raytrace::bake_shadowmask` and
    /// upload it linear. Pairs with [`LightmapMode::Replace`] (the radiance is the
    /// baked indirect). Mutually exclusive with a directional lightmap: both ride
    /// group-1 binding 18.
    Shadowmask {
        /// Baked incoming radiance (indirect), sampled by UV1. Radiance, not
        /// colour, so upload it linear.
        radiance: crate::ids::TextureId,
        /// Per-light static-occluder visibility (RGBA), sampled by UV1.
        /// Visibility factors, not colour, so upload them linear.
        shadowmask: crate::ids::TextureId,
    },
}

/// How a lightmap combines with the shader's own ambient / indirect term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LightmapMode {
    /// The baked radiance replaces the shader's indirect diffuse. Direct light
    /// from realtime lights still applies.
    Replace,
    /// The baked radiance adds on top of the shader's ambient term.
    Add,
    /// The baked value is a monochrome occlusion factor that multiplies the
    /// ambient term. Pairs with [`LightmapData::AmbientOcclusion`].
    AmbientOcclusion,
    /// Mixed lighting for static receivers: the baked radiance is the full baked
    /// lighting (direct + indirect), so the main directional light's realtime
    /// direct is suppressed here to avoid double counting. That light's realtime
    /// shadow still darkens the baked term, so dynamic objects cast shadows onto
    /// the baked surface (Unity Subtractive / Unreal stationary-light parity).
    /// Non-lightmapped (dynamic) meshes are lit fully in realtime as usual.
    Subtractive,
}

/// Where one draw of a mesh sits in a shared lightmap atlas: the atlas page and
/// the sub-rect its `[0, 1]` unwrap maps onto
/// (`lm_uv = uv1 * scale_bias.xy + scale_bias.zw`).
///
/// A mesh's lightmap registration carries a placement for every draw of it;
/// an item that names its own draws its copy at that placement instead, so
/// repeated props share one upload and one unwrap, each copy in its own rect.
/// A frame's placements live in one list, which items index with a
/// [`LightmapPlacementId`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LightmapPlacement {
    /// Base atlas page (array layer), added to the per-vertex page.
    pub layer: u32,
    /// `[scale_u, scale_v, bias_u, bias_v]` onto the page.
    pub scale_bias: [f32; 4],
}

impl LightmapPlacement {
    /// The whole of page 0: the mesh owns its atlas.
    pub const WHOLE: LightmapPlacement = LightmapPlacement {
        layer: 0,
        scale_bias: [1.0, 1.0, 0.0, 0.0],
    };

    /// A placement on `layer` with `scale_bias`.
    pub fn new(layer: u32, scale_bias: [f32; 4]) -> Self {
        Self { layer, scale_bias }
    }
}

/// An index into a frame's list of [`LightmapPlacement`]s. Four bytes, and
/// `Option<LightmapPlacementId>` is too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LightmapPlacementId(std::num::NonZeroU32);

impl LightmapPlacementId {
    /// The placement at `index` in the frame's list.
    ///
    /// # Panics
    ///
    /// If `index` is `u32::MAX` or larger.
    pub fn from_index(index: usize) -> Self {
        let n = u32::try_from(index)
            .ok()
            .and_then(|i| i.checked_add(1))
            .expect("lightmap placement index out of range");
        Self(std::num::NonZeroU32::new(n).expect("nonzero by construction"))
    }

    /// Position in the frame's list.
    pub fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

impl Default for LightmapPlacement {
    fn default() -> Self {
        Self::WHOLE
    }
}
