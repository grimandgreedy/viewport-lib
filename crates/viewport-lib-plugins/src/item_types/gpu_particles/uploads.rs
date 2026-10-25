//! How a renderer uploads, writes and releases GPU particle systems.

use viewport_lib::gpu;
use viewport_lib::renderer::ViewportRenderer;

use super::*;
use crate::item_types::{host, plugin_mut};

/// The GPU particle system surface, on the renderer.
///
/// A system owns a persistent particle buffer the simulation advances in
/// place, so creating one is not an upload of content and neither call fits
/// [`Uploads`](viewport_lib::plugin_api::Uploads).
pub trait GpuParticleSystems {
    /// Create a persistent GPU particle system, returning its handle, or the
    /// error the renderer declined GPU particles with on a device that cannot
    /// run them.
    fn create_gpu_particle_system(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        config: &GpuParticleSystemConfig,
    ) -> viewport_lib::error::ViewportResult<GpuParticleSystemId>;

    /// Release a system. The handle stops resolving and its buffers are freed.
    fn drop_gpu_particle_system(&mut self, id: GpuParticleSystemId);
}

impl GpuParticleSystems for ViewportRenderer {
    fn create_gpu_particle_system(
        &mut self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        config: &GpuParticleSystemConfig,
    ) -> viewport_lib::error::ViewportResult<GpuParticleSystemId> {
        let host = host::<GpuParticlesPlugin>(self, TYPE_NAME)?;
        Ok(host
            .plugin
            .create_system(device, queue, host.resources, config))
    }

    fn drop_gpu_particle_system(&mut self, id: GpuParticleSystemId) {
        if let Ok(plugin) = plugin_mut::<GpuParticlesPlugin>(self, TYPE_NAME) {
            plugin.drop_system(id);
        }
    }
}
