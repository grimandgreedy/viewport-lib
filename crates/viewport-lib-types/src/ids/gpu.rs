//! Handles for persistent GPU compute resources.

crate::slot_handle! {
    /// Handle to a persistent GPU particle system.
    ///
    /// Returned by `create_gpu_particle_system`. Stable until
    /// `drop_gpu_particle_system` is called.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. Dropping a system frees its slot for the next create call,
    /// so without the generation a handle to the dropped system would resolve
    /// to whatever now occupies its slot.
    pub struct GpuParticleSystemId;
}

crate::slot_handle! {
    /// Handle to a persistent external instance set.
    ///
    /// Returned by `create_external_instance_set`. Stable until
    /// `drop_external_instance_set` is called. Generational for the same reason
    /// as [`GpuParticleSystemId`]: dropped slots are reused.
    pub struct ExternalInstanceSetId;
}
