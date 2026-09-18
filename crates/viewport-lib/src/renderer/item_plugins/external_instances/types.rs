//! The public configuration surface for an external instance set: what a host
//! registers once and can re-point later.

pub use viewport_lib_types::ids::ExternalInstanceSetId;

/// Persistent configuration for an external instance set.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExternalInstanceSetConfig {
    /// Mesh drawn once per instance.
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Consumer-owned same-device buffer of tightly packed `[x, y, z]` `f32`
    /// triples, 12 bytes per instance. Must have
    /// [`wgpu::BufferUsages::STORAGE`](crate::gpu::BufferUsages::STORAGE).
    /// The renderer only reads it.
    pub positions: crate::gpu::Buffer,
}

impl ExternalInstanceSetConfig {
    /// Set drawing `mesh_id` at every position in `positions`.
    pub fn new(
        mesh_id: crate::resources::mesh::mesh_store::MeshId,
        positions: crate::gpu::Buffer,
    ) -> Self {
        Self { mesh_id, positions }
    }
}
