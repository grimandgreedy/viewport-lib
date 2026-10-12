//! The GPU particle system item type as an [`ItemTypePlugin`]: a persistent
//! particle buffer that lives on the GPU, advanced by an emit and a sim compute
//! pass each frame and drawn either as camera-facing billboards or as instances
//! of an uploaded mesh. Consumers create a system once with
//! [`GpuParticleSystems::create_gpu_particle_system`]
//! and submit a [`GpuParticleSystemItem`] each frame to advance and draw it.
//!
//! The compute work runs in `prepare` and is handed back as a command buffer,
//! so it is submitted before the frame's draws; the draw itself is an ordinary
//! `paint`, because particles depth-test against the opaque scene but never
//! write depth.
//!
//! The data types here build without the `gpu` feature; the plugin and its
//! upload calls need it.
//!
//! [`ItemTypePlugin`]: viewport_lib::plugin_api::ItemTypePlugin

mod types;

pub use types::GpuParticleSystemId;
pub use types::{
    EmitterConfig, ForceField, GpuParticleSystemConfig, GpuParticleSystemItem, ParticleMeshAlign,
    ParticleRender, SpawnShape, VelocityDist,
};

/// Stable name this item type submits and registers under.
pub const TYPE_NAME: &str = "vpl.gpu_particles";

#[cfg(feature = "gpu")]
mod pipeline;
#[cfg(feature = "gpu")]
mod plugin;
#[cfg(feature = "gpu")]
mod store;
#[cfg(feature = "gpu")]
mod uploads;

#[cfg(feature = "gpu")]
pub use plugin::GpuParticlesPlugin;
#[cfg(feature = "gpu")]
pub(crate) use plugin::shader_sources;
#[cfg(feature = "gpu")]
pub use uploads::GpuParticleSystems;
