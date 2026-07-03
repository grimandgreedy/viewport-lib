use super::SpriteBlend;
use crate::scene::material::ItemSettings;

/// A host-built instanced draw of a single mesh at many per-instance transforms.
///
/// Use this for mesh-based particle effects (falling leaves, debris, alembic
/// snippets, projectile clouds) where the simulation lives in the host and the
/// renderer only needs to draw N copies of one mesh per frame in a single
/// draw call. The list of instances is rebuilt every frame from the supplied
/// `transforms` and `colours` vectors; nothing is retained between frames.
///
/// All instances in a batch share the same `mesh_id`, `texture_id`, and
/// `blend` mode. To mix meshes or blend modes, submit multiple
/// [`MeshInstanceItem`] entries on [`SceneFrame::mesh_instances`].
///
/// The fragment shader is the unlit instanced mesh shader (the same shader
/// backing the scene-graph instanced path). Mesh particles do not receive
/// shadows or lighting; they are tinted by `colours` and optionally sampled
/// from `texture_id`.
#[non_exhaustive]
#[derive(Clone)]
pub struct MeshInstanceItem {
    /// Mesh handle returned by `ViewportGpuResources::upload_mesh_data`.
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Optional albedo texture handle. `None` renders flat-shaded with the
    /// per-instance `colours` alone.
    pub texture_id: Option<u64>,
    /// Per-instance world-space TRS matrices. Length defines the instance count.
    pub transforms: Vec<[[f32; 4]; 4]>,
    /// Per-instance RGBA tints. If shorter than `transforms`, missing entries
    /// fall back to opaque white.
    pub colours: Vec<[f32; 4]>,
    /// GPU blend state for this batch. Reuses [`SpriteBlend`] from the sprite
    /// path so both particle systems share one enum.
    pub blend: SpriteBlend,
    /// Per-item render settings (visibility, pick identity, selection state).
    pub settings: ItemSettings,
    /// LOD group to draw these instances from. `None` means draw `mesh_id`
    /// directly for every instance.
    ///
    /// When set, the renderer measures each instance's on-screen size and groups
    /// the instances by level, drawing each level's subset with its own mesh. So
    /// near instances in the batch draw the full mesh while far ones drop to a
    /// cheaper one, all from a single submitted item. `mesh_id` is ignored.
    /// Set `pick_id` for the per-instance level to use hysteresis across frames.
    pub lod_group: Option<crate::resources::LodGroupId>,
}

impl Default for MeshInstanceItem {
    fn default() -> Self {
        Self {
            mesh_id: crate::resources::mesh::mesh_store::MeshId::INVALID,
            texture_id: None,
            transforms: Vec::new(),
            colours: Vec::new(),
            blend: SpriteBlend::AlphaBlend,
            settings: ItemSettings::default(),
            lod_group: None,
        }
    }
}

/// Per-particle rotation rule used by the mesh render route.
///
/// Used by [`ParticleRender::Mesh`](crate::resources::ParticleRender::Mesh).
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
    pub colour: [f32; 4],
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
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 16.0,
        }
    }
}

/// Per-frame submission that advances and draws one GPU particle system.
///
/// Submit on `SceneFrame::gpu_particle_systems`. The renderer dispatches an
/// emit kernel that spawns new particles into the persistent buffer behind
/// `system_id`, then a sim kernel that integrates `forces` and decrements
/// lifetime, then draws the live particles through the render route chosen
/// when the system was created. No CPU per-particle work happens on the host.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GpuParticleSystemItem {
    /// Target system. The buffer behind this handle is updated in place.
    pub system_id: crate::resources::GpuParticleSystemId,
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
    pub fn new(system_id: crate::resources::GpuParticleSystemId, time_step: f32) -> Self {
        Self {
            system_id,
            emitter: EmitterConfig::default(),
            forces: Vec::new(),
            time_step,
            settings: ItemSettings::default(),
        }
    }
}
