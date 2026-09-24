//! The public configuration surface for a GPU particle system: what a host
//! sets at creation and cannot change afterwards.
//!
//! Create a system with
//! [`GpuParticleSystems::create_gpu_particle_system`](crate::GpuParticleSystems::create_gpu_particle_system)
//! and submit a [`GpuParticleSystemItem`] per frame to simulate and draw it.

use crate::sprite::{SpriteLitParams, SpriteSizeMode};
use viewport_lib::ItemSettings;
use viewport_lib::renderer::SpriteBlend;

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
        /// ([`TextureData::srgb`](viewport_lib::resources::TextureData::srgb)).
        texture_id: Option<viewport_lib::resources::TextureId>,
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
        /// ([`TextureData::normal_map`](viewport_lib::resources::TextureData::normal_map)).
        normal_texture_id: Option<viewport_lib::resources::TextureId>,
    },
    /// Draw each particle as an instance of an uploaded mesh. The vertex
    /// shader composes the per-instance transform from the live particle's
    /// position, velocity, `size`, and (for `Random` align) the spawn seed.
    /// Unlit; the particle colour multiplies an optional albedo sample.
    Mesh {
        /// Mesh handle returned by `DeviceResources::upload_mesh_data`.
        mesh_id: viewport_lib::resources::mesh::mesh_store::MeshId,
        /// Optional albedo texture handle. `None` renders flat-tinted. Colour,
        /// so upload it sRGB
        /// ([`TextureData::srgb`](viewport_lib::resources::TextureData::srgb)).
        texture_id: Option<viewport_lib::resources::TextureId>,
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
                normal_mode: crate::sprite::SpriteNormalMode::Spherical,
                receive_shadows: false,
                ambient_scale: 1.0,
            },
            normal_texture_id: None,
        }
    }
}

/// Per-particle rotation rule used by the mesh render route.
///
/// Used by [`ParticleRender::Mesh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParticleMeshAlign {
    /// No rotation. The mesh keeps its authored orientation.
    #[default]
    Identity,
    /// Rotation that maps the mesh's +Y axis onto the per-particle velocity
    /// vector. Useful for projectiles, debris with tumble, casings.
    Velocity,
    /// Stable random rotation seeded at spawn and held until the particle dies.
    /// Useful for tumbling debris, gibs, scattered leaves.
    Random,
}

/// Distribution used to assign an initial velocity to a newly spawned particle.
#[derive(Debug, Clone, Copy)]
pub enum VelocityDist {
    /// Every particle gets the same velocity.
    Fixed([f32; 3]),
    /// Velocity is sampled uniformly inside an axis-aligned box.
    UniformBox {
        /// Minimum corner of the velocity box.
        min: [f32; 3],
        /// Maximum corner of the velocity box.
        max: [f32; 3],
    },
    /// Velocity direction is sampled uniformly inside a cone around `axis`,
    /// magnitude in `[min_speed, max_speed]`.
    UniformCone {
        /// Cone axis direction.
        axis: [f32; 3],
        /// Half-angle of the cone in radians.
        half_angle: f32,
        /// Lower bound on sampled speed.
        min_speed: f32,
        /// Upper bound on sampled speed.
        max_speed: f32,
    },
}

impl Default for VelocityDist {
    fn default() -> Self {
        VelocityDist::Fixed([0.0, 0.0, 1.0])
    }
}

/// Shape from which new particles are spawned.
#[derive(Debug, Clone, Copy)]
pub enum SpawnShape {
    /// All particles spawn at the same point.
    Point([f32; 3]),
    /// Spawn uniformly inside an axis-aligned box.
    Box {
        /// Minimum corner of the spawn box.
        min: [f32; 3],
        /// Maximum corner of the spawn box.
        max: [f32; 3],
    },
    /// Spawn uniformly inside a sphere.
    Sphere {
        /// Sphere center in world space.
        center: [f32; 3],
        /// Sphere radius.
        radius: f32,
    },
}

impl Default for SpawnShape {
    fn default() -> Self {
        SpawnShape::Point([0.0, 0.0, 0.0])
    }
}

/// Force applied to every live particle each simulation step.
#[derive(Debug, Clone, Copy)]
pub enum ForceField {
    /// Constant acceleration. World units per second squared.
    Gravity([f32; 3]),
    /// Velocity-proportional drag. Coefficient is the fraction of velocity
    /// lost per second.
    Drag(f32),
    /// Pull toward a world-space point. Acceleration scales as
    /// `strength / (distance + falloff)^2`.
    PointAttractor {
        /// World-space position of the attractor.
        position: [f32; 3],
        /// Acceleration coefficient. Negative values repel.
        strength: f32,
        /// Distance offset that softens the singularity at the center.
        falloff: f32,
    },
}

/// Emitter configuration for a GPU particle system.
///
/// All fields are independent of any simulation state on the GPU; the host can
/// change them between frames and the next emit pass picks up the new values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EmitterConfig {
    /// New particles per second. Fractional values accumulate across frames.
    pub rate: f32,
    /// Range of per-particle lifetimes in seconds. Each new particle gets a
    /// uniformly sampled value in `[lifetime.0, lifetime.1]`.
    pub lifetime: (f32, f32),
    /// Initial velocity distribution.
    pub initial_velocity: VelocityDist,
    /// Spawn shape relative to world space.
    pub spawn_shape: SpawnShape,
    /// Per-particle RGBA tint, multiplied with any texture sample at draw time.
    pub colour: viewport_lib::Colour,
    /// Per-particle starting size. Pixels (ScreenSpace) or world units
    /// (WorldSpace) per the system's render config.
    pub size: f32,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            rate: 100.0,
            lifetime: (1.0, 2.0),
            initial_velocity: VelocityDist::default(),
            spawn_shape: SpawnShape::default(),
            colour: [1.0, 1.0, 1.0, 1.0].into(),
            size: 16.0,
        }
    }
}

/// Per-frame submission that advances and draws one GPU particle system.
///
/// Submit with `frame.scene.submit::<GpuParticleSystemItem>(..)`. The renderer dispatches an
/// emit kernel that spawns new particles into the persistent buffer behind
/// `system_id`, then a sim kernel that integrates `forces` and decrements
/// lifetime, then draws the live particles through the render route chosen
/// when the system was created. No CPU per-particle work happens on the host.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GpuParticleSystemItem {
    /// Target system. The buffer behind this handle is updated in place.
    pub system_id: GpuParticleSystemId,
    /// Emitter parameters for this frame's emit pass.
    pub emitter: EmitterConfig,
    /// Forces applied to every live particle this frame.
    pub forces: Vec<ForceField>,
    /// Simulation time step in seconds. Typically the frame delta time.
    pub time_step: f32,
    /// Per-item render settings (visibility, picking, selection).
    pub settings: ItemSettings,
}

impl GpuParticleSystemItem {
    /// Visible item with default emitter and no forces.
    pub fn new(system_id: GpuParticleSystemId, time_step: f32) -> Self {
        Self {
            system_id,
            emitter: EmitterConfig::default(),
            forces: Vec::new(),
            time_step,
            settings: ItemSettings::default(),
        }
    }
}
