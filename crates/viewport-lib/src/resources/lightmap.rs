//! Baked lightmap consumption.
//!
//! A lightmap is a texture of pre-baked indirect light over a mesh's static
//! surface. It is sampled with a second UV set (UV1) that gives every surface
//! point its own non-overlapping texel, and folded into the lit fragment's
//! ambient term. The renderer only samples finished textures; producing them is
//! an offline step.
//!
//! Register one per mesh with
//! [`DeviceResources::set_lightmap`](crate::resources::ViewportGpuResources::set_lightmap).
//! Meshes without one pay nothing: the UV1 sidecar and the lightmap texture
//! bindings resolve to shared zero / 1x1 fallbacks and the shader branch is
//! skipped.

use crate::resources::TextureId;

/// Which baked textures a lightmap carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightmapData {
    /// A single RGB radiance texture. Normal maps do not respond to the baked
    /// light in this mode.
    NonDirectional {
        /// Baked incoming radiance, sampled by UV1.
        radiance: TextureId,
    },
    /// A radiance texture plus a per-texel dominant light direction, so a normal
    /// map responds to where the baked light came from. `direction` holds the
    /// unit dominant direction in `xyz` (world space) and a directionality factor
    /// in `w` (`0` = ambient, `1` = a single direction); upload it linear
    /// (`upload_texture_hdr`) so the signed direction survives.
    DominantDirection {
        /// Baked incoming radiance, sampled by UV1.
        radiance: TextureId,
        /// Per-texel dominant direction + directionality, sampled by UV1.
        direction: TextureId,
    },
    /// A single-channel occlusion factor, read from the red channel. Pairs with
    /// [`LightmapMode::AmbientOcclusion`].
    AmbientOcclusion {
        /// Baked ambient occlusion, sampled by UV1.
        occlusion: TextureId,
    },
    /// A radiance texture plus a per-light static-occluder shadowmask: the RGBA
    /// channels hold the baked visibility (0 = shadowed, 1 = lit) of up to four
    /// realtime lights (light index `i` -> channel `i`). The lit shaders combine
    /// it with each light's realtime shadow by `min`, so static shadows come from
    /// the bake (crisp at any distance) and dynamic ones from the shadow map,
    /// while the lights stay fully dynamic (Unity Shadowmask parity). Bake the
    /// shadowmask with [`bake_shadowmask`](crate::raytrace::bake_shadowmask) and
    /// upload it linear. Pairs with [`LightmapMode::Replace`] (the radiance is the
    /// baked indirect). Mutually exclusive with a directional lightmap: both ride
    /// group-1 binding 18.
    Shadowmask {
        /// Baked incoming radiance (indirect), sampled by UV1.
        radiance: TextureId,
        /// Per-light static-occluder visibility (RGBA), sampled by UV1.
        shadowmask: TextureId,
    },
}

impl LightmapData {
    /// The primary (radiance / occlusion) texture bound at binding 17.
    pub(crate) fn texture_id(self) -> TextureId {
        match self {
            LightmapData::NonDirectional { radiance } => radiance,
            LightmapData::DominantDirection { radiance, .. } => radiance,
            LightmapData::AmbientOcclusion { occlusion } => occlusion,
            LightmapData::Shadowmask { radiance, .. } => radiance,
        }
    }

    /// The secondary texture bound at binding 18: the dominant-direction atlas for
    /// a directional lightmap, or the shadowmask for a shadowmask lightmap (the two
    /// are mutually exclusive and share the binding).
    pub(crate) fn direction_texture_id(self) -> Option<TextureId> {
        match self {
            LightmapData::DominantDirection { direction, .. } => Some(direction),
            LightmapData::Shadowmask { shadowmask, .. } => Some(shadowmask),
            _ => None,
        }
    }

    /// True when binding 18 carries a shadowmask (rather than a direction atlas).
    pub(crate) fn is_shadowmask(self) -> bool {
        matches!(self, LightmapData::Shadowmask { .. })
    }
}

/// How a lightmap combines with the shader's own ambient / indirect term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

impl LightmapMode {
    /// Shader mode code: 1 = Replace, 2 = Add, 3 = AmbientOcclusion,
    /// 4 = Subtractive. 0 (no registration) skips the branch entirely.
    pub(crate) fn to_shader(self) -> u32 {
        match self {
            LightmapMode::Replace => 1,
            LightmapMode::Add => 2,
            LightmapMode::AmbientOcclusion => 3,
            LightmapMode::Subtractive => 4,
        }
    }
}

/// Per-mesh lightmap registration, held on the mesh once `set_lightmap` runs.
pub(crate) struct MeshLightmap {
    /// Per-vertex UV1 storage buffer (`array<vec4<f32>>`, UV in `.xy`), bound at
    /// group 1 binding 15. It reuses the plugin vertex-attribute slot: the Metal
    /// vertex buffer table is full, so the lightmap cannot take a new binding.
    pub(crate) uv1_buffer: crate::gpu::Buffer,
    /// The baked texture, bound at group 1 binding 17.
    pub(crate) texture_id: TextureId,
    /// The secondary atlas, bound at group 1 binding 18. `Some` for a directional
    /// lightmap (dominant-direction atlas) or a shadowmask lightmap (per-light
    /// visibility); `is_shadowmask` says which, and they are mutually exclusive.
    pub(crate) direction_texture_id: Option<TextureId>,
    /// When true, binding 18 carries a shadowmask (drives `has_shadowmask` in the
    /// object uniform); when false, a dominant-direction atlas.
    pub(crate) is_shadowmask: bool,
    /// Shader mode code (see [`LightmapMode::to_shader`]).
    pub(crate) mode: u32,
    /// Maps this mesh's `[0, 1]` unwrap onto its sub-rect of a shared scene atlas:
    /// `lm_uv = uv1 * scale_bias.xy + scale_bias.zw`. Identity `[1, 1, 0, 0]` when
    /// the mesh owns its whole atlas (a per-mesh lightmap).
    pub(crate) scale_bias: [f32; 4],
    /// Base atlas page (array layer) this mesh's lightmap starts on. Added to the
    /// per-vertex page in the shader. 0 for a single-page or per-mesh lightmap.
    pub(crate) layer: u32,
}
