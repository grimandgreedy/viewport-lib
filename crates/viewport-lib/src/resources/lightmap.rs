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

pub use viewport_lib_types::effects::lightmap::{
    LightmapData, LightmapMode, LightmapPlacement, LightmapPlacementId,
};

/// The primary (radiance / occlusion) texture bound at binding 17.
pub(crate) fn primary_texture(data: LightmapData) -> TextureId {
    match data {
        LightmapData::NonDirectional { radiance } => radiance,
        LightmapData::DominantDirection { radiance, .. } => radiance,
        LightmapData::AmbientOcclusion { occlusion } => occlusion,
        LightmapData::Shadowmask { radiance, .. } => radiance,
    }
}

/// The secondary texture bound at binding 18: the dominant-direction atlas for
/// a directional lightmap, or the shadowmask for a shadowmask lightmap (the two
/// are mutually exclusive and share the binding).
pub(crate) fn secondary_texture(data: LightmapData) -> Option<TextureId> {
    match data {
        LightmapData::DominantDirection { direction, .. } => Some(direction),
        LightmapData::Shadowmask { shadowmask, .. } => Some(shadowmask),
        _ => None,
    }
}

/// Shader mode code: 1 = Replace, 2 = Add, 3 = AmbientOcclusion,
/// 4 = Subtractive. 0 (no registration) skips the branch entirely.
pub(crate) fn mode_code(mode: LightmapMode) -> u32 {
    match mode {
        LightmapMode::Replace => 1,
        LightmapMode::Add => 2,
        LightmapMode::AmbientOcclusion => 3,
        LightmapMode::Subtractive => 4,
    }
}

/// Per-mesh lightmap registration, held on the mesh once `set_lightmap` runs.
/// The per-vertex UV1 (UV in `.xy`, atlas page in `.z`) is a stream of the
/// mesh's sidecar record.
pub(crate) struct MeshLightmap {
    /// The baked texture, bound at group 1 binding 17.
    pub(crate) texture_id: TextureId,
    /// The secondary atlas, bound at group 1 binding 18. `Some` for a directional
    /// lightmap (dominant-direction atlas) or a shadowmask lightmap (per-light
    /// visibility); `is_shadowmask` says which, and they are mutually exclusive.
    pub(crate) direction_texture_id: Option<TextureId>,
    /// When true, binding 18 carries a shadowmask (drives `has_shadowmask` in the
    /// object uniform); when false, a dominant-direction atlas.
    pub(crate) is_shadowmask: bool,
    /// Shader mode code (see [`mode_code`]).
    pub(crate) mode: u32,
    /// Maps this mesh's `[0, 1]` unwrap onto its sub-rect of a shared scene atlas:
    /// `lm_uv = uv1 * scale_bias.xy + scale_bias.zw`. Identity `[1, 1, 0, 0]` when
    /// the mesh owns its whole atlas (a per-mesh lightmap).
    pub(crate) scale_bias: [f32; 4],
    /// Base atlas page (array layer) this mesh's lightmap starts on. Added to the
    /// per-vertex page in the shader. 0 for a single-page or per-mesh lightmap.
    pub(crate) layer: u32,
}
