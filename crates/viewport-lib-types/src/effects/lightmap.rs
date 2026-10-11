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
