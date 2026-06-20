//! State for the non-instanced (per-object) mesh draw path: one `ObjectUniform`
//! buffer and bind group per scene item, plus the wireframe and
//! transparent-volume-mesh wireframe draw resources.

use crate::resources::mesh_store::MeshId;

pub(crate) struct PerObjectState {
    /// Per-item uniform buffers for the per-object draw path (one per scene item).
    pub(crate) uniform_bufs: Vec<wgpu::Buffer>,
    /// Per-item bind groups pairing the per-item uniform with the item's mesh textures.
    pub(crate) bind_groups: Vec<Option<wgpu::BindGroup>>,
    /// Cache keys for per-item bind group reuse (material/texture fingerprint).
    pub(crate) cache_keys: Vec<u64>,
    /// Per-item uniform buffers used in wireframe mode.
    pub(crate) wireframe_uniform_bufs: Vec<wgpu::Buffer>,
    /// Per-item bind groups pairing wireframe uniforms with fallback textures.
    pub(crate) wireframe_bind_groups: Vec<wgpu::BindGroup>,
    /// TransparentVolumeMesh boundary wireframe mesh IDs to draw.
    pub(crate) tvm_wireframe_draws: Vec<MeshId>,
    /// Shared wireframe uniform (identity matrix, wireframe = 1) for TVM draws.
    pub(crate) tvm_wireframe_buf: Option<wgpu::Buffer>,
    /// Bind group for the TVM wireframe draws.
    pub(crate) tvm_wireframe_bg: Option<wgpu::BindGroup>,
}

impl PerObjectState {
    pub(crate) fn new() -> Self {
        Self {
            uniform_bufs: Vec::new(),
            bind_groups: Vec::new(),
            cache_keys: Vec::new(),
            wireframe_uniform_bufs: Vec::new(),
            wireframe_bind_groups: Vec::new(),
            tvm_wireframe_draws: Vec::new(),
            tvm_wireframe_buf: None,
            tvm_wireframe_bg: None,
        }
    }
}
