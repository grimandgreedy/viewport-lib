//! Tetrahedral mesh data type.

pub use viewport_lib_types::data::volume::{TetMesh, TetMeshAttributes};

/// Uniform buffer layout for the projected tetrahedra pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ProjectedTetUniform {
    pub(crate) density: f32,
    pub(crate) scalar_min: f32,
    pub(crate) scalar_max: f32,
    pub(crate) threshold_min: f32,
    pub(crate) threshold_max: f32,
    /// 1 = skip Beer-Lambert thickness modulation and emit a flat density-scaled
    /// alpha per visible fragment. Wired from `ItemSettings.unlit`.
    pub(crate) unlit: u32,
    /// Multiplied into the final alpha. Wired from `ItemSettings.opacity`.
    pub(crate) opacity: f32,
    pub(crate) _pad: f32,
}

/// One device-limit-bounded chunk of a projected-tet mesh.
pub(crate) struct ProjectedTetChunk {
    /// Storage buffer for this chunk's tetrahedra geometry (kept alive for the
    /// bind group). Vertex positions only; the per-tet scalar lives in
    /// `scalar_buffer` so a scalar-only refresh does not touch geometry.
    #[allow(dead_code)]
    pub tet_buffer: crate::gpu::Buffer,
    /// Per-tet scalar storage buffer (one `f32` per tet), parallel to
    /// `tet_buffer`. Rewritten in place by `replace_projected_tet_scalar`
    /// without rebuilding the geometry.
    #[allow(dead_code)]
    pub scalar_buffer: crate::gpu::Buffer,
    /// Number of tetrahedra in this chunk (= instanced draw count).
    pub tet_count: u32,
    /// Bind group: shared uniform + this chunk's tet + scalar buffers + colourmap.
    pub bind_group: crate::gpu::BindGroup,
}

/// Uploaded projected-tetrahedra mesh, stored persistently on the GPU.
///
/// Large meshes are split into multiple chunks so each storage buffer
/// stays within `max_storage_buffer_binding_size`.
/// Created by [`DeviceResources::upload_projected_tet`].
pub(crate) struct GpuProjectedTetMesh {
    /// One or more device-limit-bounded chunks.
    pub chunks: Vec<ProjectedTetChunk>,
    /// Uniform buffer shared across all chunks (density, scalar_min/max). Written each frame.
    pub uniform_buffer: crate::gpu::Buffer,
    /// Auto-detected scalar range from the uploaded data (min, max).
    pub scalar_range: (f32, f32),
}

impl GpuProjectedTetMesh {
    /// Resident GPU bytes: every chunk's tet geometry and per-tet scalar buffers
    /// plus the shared uniform. Charged into the store on insert/replace so the
    /// mesh shows up in `ResidentBytes::projected_tet_bytes`.
    pub(crate) fn gpu_bytes(&self) -> u64 {
        self.uniform_buffer.size()
            + self
                .chunks
                .iter()
                .map(|c| c.tet_buffer.size() + c.scalar_buffer.size())
                .sum::<u64>()
    }
}
