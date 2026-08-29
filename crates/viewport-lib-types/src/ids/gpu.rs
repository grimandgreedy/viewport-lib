//! Append-only registry handles for persistent GPU compute resources.

crate::registry_handle! {
    /// Handle to a persistent GPU particle system.
    ///
    /// Returned by `create_gpu_particle_system`. Stable until
    /// `drop_gpu_particle_system` is called. An append-only registry handle.
    pub struct GpuParticleSystemId;
}

crate::registry_handle! {
    /// Handle to a persistent external instance set.
    ///
    /// Returned by `create_external_instance_set`. Stable until
    /// `drop_external_instance_set` is called.
    pub struct ExternalInstanceSetId;
}
