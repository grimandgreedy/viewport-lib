//! Surface line-integral-convolution pipelines, layouts, and uniforms.

/// Surface line-integral-convolution pipelines and their layouts.
///
/// Device-shared and lazily built by the LIC post-process setup. The
/// per-viewport vector/intensity textures and the `lic_enabled` / `lic_strength`
/// state live on `ViewportHdrState`, not here.
#[derive(Default)]
pub(crate) struct LicResources {
    /// Renders mesh with vector storage buffer -> lic_vector_texture (Rgba8Unorm).
    pub(crate) surface_pipeline: Option<wgpu::RenderPipeline>,
    /// Group 1 layout of the LIC surface pass (object uniform + vector buffer + noise).
    pub(crate) surface_bgl: Option<wgpu::BindGroupLayout>,
    /// Reads lic_vector_texture, writes LIC intensity to R8Unorm target.
    pub(crate) advect_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for the LIC advect pass.
    pub(crate) advect_bgl: Option<wgpu::BindGroupLayout>,
    /// Bilinear sampler for the LIC advect pass.
    pub(crate) noise_sampler: Option<wgpu::Sampler>,
    /// 1x1 R8Unorm white placeholder bound to tone_map binding 7 when LIC is inactive.
    pub(crate) placeholder_view: Option<wgpu::TextureView>,
}

/// Uniform for the LIC advect render pass (step counts and viewport dims).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LicAdvectUniform {
    pub(crate) steps: u32,
    pub(crate) step_size: f32,
    pub(crate) vp_width: f32,
    pub(crate) vp_height: f32,
}

/// Uniform for the LIC surface pass (model matrix per object, 64 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LicObjectUniform {
    pub(crate) model: [[f32; 4]; 4],
}

/// Per-frame GPU data for one Surface LIC item, created in `prepare()`.
pub struct LicSurfaceGpuData {
    /// Bind group (group 1): LicObjectUniform only. Flow vectors bound as vertex buffer 1.
    pub(crate) bind_group: wgpu::BindGroup,
    /// Owned uniform buffer for the model matrix. Kept alive by this struct.
    pub(crate) _object_uniform_buf: wgpu::Buffer,
    /// MeshId used to look up vertex + index buffers in the render pass.
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Name of the flow vector attribute for looking up the vertex buffer in the render pass.
    pub(crate) vector_attribute: String,
}
