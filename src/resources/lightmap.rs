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
    /// A single-channel occlusion factor, read from the red channel. Pairs with
    /// [`LightmapMode::AmbientOcclusion`].
    AmbientOcclusion {
        /// Baked ambient occlusion, sampled by UV1.
        occlusion: TextureId,
    },
}

impl LightmapData {
    pub(crate) fn texture_id(self) -> TextureId {
        match self {
            LightmapData::NonDirectional { radiance } => radiance,
            LightmapData::AmbientOcclusion { occlusion } => occlusion,
        }
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
}

impl LightmapMode {
    /// Shader mode code: 1 = Replace, 2 = Add, 3 = AmbientOcclusion. 0 (no
    /// registration) skips the branch entirely.
    pub(crate) fn to_shader(self) -> u32 {
        match self {
            LightmapMode::Replace => 1,
            LightmapMode::Add => 2,
            LightmapMode::AmbientOcclusion => 3,
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
    /// Shader mode code (see [`LightmapMode::to_shader`]).
    pub(crate) mode: u32,
}
