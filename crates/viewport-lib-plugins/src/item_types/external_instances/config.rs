//! The configuration an external instance set registers with. It holds the
//! consumer's GPU buffer, so it needs the `gpu` feature.

use viewport_lib::gpu;

/// Persistent configuration for an external instance set.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ExternalInstanceSetConfig {
    /// Mesh drawn once per instance.
    pub mesh_id: viewport_lib::resources::mesh::mesh_store::MeshId,
    /// Consumer-owned same-device buffer of tightly packed `[x, y, z]` `f32`
    /// triples, 12 bytes per instance. Must have
    /// [`wgpu::BufferUsages::STORAGE`](gpu::BufferUsages::STORAGE).
    /// The renderer only reads it.
    pub positions: gpu::Buffer,
}

impl ExternalInstanceSetConfig {
    /// Set drawing `mesh_id` at every position in `positions`.
    pub fn new(
        mesh_id: viewport_lib::resources::mesh::mesh_store::MeshId,
        positions: gpu::Buffer,
    ) -> Self {
        Self { mesh_id, positions }
    }
}
