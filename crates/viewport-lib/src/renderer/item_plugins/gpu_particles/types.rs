//! The public configuration surface for a GPU particle system: what a host
//! sets at creation and cannot change afterwards.
//!
//! Create a system with
//! [`ViewportRenderer::create_gpu_particle_system`](crate::renderer::ViewportRenderer::create_gpu_particle_system)
//! and submit a [`GpuParticleSystemItem`](crate::renderer::GpuParticleSystemItem)
//! per frame to simulate and draw it.

use crate::renderer::{ParticleMeshAlign, SpriteBlend, SpriteLitParams, SpriteSizeMode};

pub use viewport_lib_types::ids::GpuParticleSystemId;

/// Persistent configuration for a particle system. Set at creation; the render
/// route and capacity are stable for the system's lifetime.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GpuParticleSystemConfig {
    /// Maximum number of simultaneously live particles. Memory cost scales
    /// linearly with this value (currently 80 bytes per particle plus a small
    /// fixed overhead).
    pub capacity: u32,
    /// How the live particles are drawn each frame.
    pub render: ParticleRender,
}

impl Default for GpuParticleSystemConfig {
    fn default() -> Self {
        Self {
            capacity: 10_000,
            render: ParticleRender::default(),
        }
    }
}

/// How a particle system draws its live particles.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ParticleRender {
    /// Draw each particle as a camera-facing billboard sprite.
    Sprite {
        /// Optional texture sampled per fragment. `None` renders solid quads
        /// tinted by the particle colour. Colour, so upload it sRGB
        /// ([`TextureData::srgb`](crate::resources::TextureData::srgb)).
        texture_id: Option<crate::resources::TextureId>,
        /// GPU blend state.
        blend: SpriteBlend,
        /// Screen-space or world-space sizing for the per-particle `size`.
        size_mode: SpriteSizeMode,
        /// Whether the draw writes to depth.
        depth_write: bool,
        /// When `true`, the draw runs through the lit particle sprite pipeline
        /// and picks up the scene lighting environment. Default `false`
        /// preserves the emissive billboard look.
        lit: bool,
        /// Lighting parameters used when `lit` is `true`.
        lit_params: SpriteLitParams,
        /// Optional tangent-space normal map for the `NormalMap` mode.
        /// Directions, not colour, so upload it linear
        /// ([`TextureData::normal_map`](crate::resources::TextureData::normal_map)).
        normal_texture_id: Option<crate::resources::TextureId>,
    },
    /// Draw each particle as an instance of an uploaded mesh. The vertex
    /// shader composes the per-instance transform from the live particle's
    /// position, velocity, `size`, and (for `Random` align) the spawn seed.
    /// Unlit; the particle colour multiplies an optional albedo sample.
    Mesh {
        /// Mesh handle returned by `DeviceResources::upload_mesh_data`.
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        /// Optional albedo texture handle. `None` renders flat-tinted. Colour,
        /// so upload it sRGB
        /// ([`TextureData::srgb`](crate::resources::TextureData::srgb)).
        texture_id: Option<crate::resources::TextureId>,
        /// GPU blend state.
        blend: SpriteBlend,
        /// How per-particle rotation is derived.
        align: ParticleMeshAlign,
    },
}

impl Default for ParticleRender {
    fn default() -> Self {
        ParticleRender::Sprite {
            texture_id: None,
            blend: SpriteBlend::AlphaBlend,
            size_mode: SpriteSizeMode::ScreenSpace,
            depth_write: false,
            lit: false,
            lit_params: SpriteLitParams {
                roughness: 0.9,
                normal_mode: crate::renderer::SpriteNormalMode::Spherical,
                receive_shadows: false,
                ambient_scale: 1.0,
            },
            normal_texture_id: None,
        }
    }
}
