//! Surface line-integral-convolution pipelines, layouts, and uniforms.

use crate::resources::pipeline_slot::LazyFamily;
use crate::resources::types::Vertex;

/// What the LIC surface pipeline build reads.
pub(crate) struct LicSurfaceRecipe {
    pub(crate) device: crate::gpu::Device,
    pub(crate) layout: crate::gpu::PipelineLayout,
    pub(crate) shader: crate::gpu::ShaderModule,
}

/// The mesh pass into the `Rgba8Unorm` vector target. Vertex buffer 0 is the
/// full `Vertex` stride with the position at location 0; buffer 1 is the
/// tightly packed `[f32; 3]` flow vectors at location 1.
pub(crate) fn build_surface(r: &LicSurfaceRecipe, _i: usize) -> crate::gpu::RenderPipeline {
    let lic_vertex_layout = crate::gpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as crate::gpu::BufferAddress,
        step_mode: crate::gpu::VertexStepMode::Vertex,
        attributes: &[crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }],
    };
    let lic_flow_layout = crate::gpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: crate::gpu::VertexStepMode::Vertex,
        attributes: &[crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: crate::gpu::VertexFormat::Float32x3,
        }],
    };
    crate::resources::builders::render_pipeline(
        &r.device,
        crate::resources::builders::RenderPipelineDesc {
            label: "lic_surface_pipeline",
            layout: &r.layout,
            vertex_module: &r.shader,
            vertex_entry: "vs_main",
            vertex_buffers: &[lic_vertex_layout, lic_flow_layout],
            fragment: Some(crate::gpu::FragmentState {
                module: &r.shader,
                entry_point: Some("fs_main"),
                targets: &[Some(crate::gpu::ColorTargetState {
                    format: crate::gpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: crate::gpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: crate::gpu::PrimitiveState {
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: crate::gpu::MultisampleState::default(),
            cache: None,
        },
    )
}

/// Surface line-integral-convolution pipelines and their layouts.
///
/// Device-shared and composed by the LIC post-process setup; each pipeline is
/// built when a frame first binds it. The per-viewport vector/intensity
/// textures and the `lic_enabled` / `lic_strength` state live on
/// `ViewportHdrState`, not here.
#[derive(Default)]
pub(crate) struct LicResources {
    /// Renders mesh with vector storage buffer -> lic_vector_texture (Rgba8Unorm).
    pub(crate) surface_pipeline: Option<LazyFamily<LicSurfaceRecipe, 1>>,
    /// Group 1 layout of the LIC surface pass (object uniform + vector buffer + noise).
    pub(crate) surface_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Reads lic_vector_texture, writes LIC intensity to R8Unorm target.
    pub(crate) advect_pipeline: Option<super::LazyFullscreen>,
    /// Bind group layout for the LIC advect pass.
    pub(crate) advect_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Bilinear sampler for the LIC advect pass.
    pub(crate) noise_sampler: Option<crate::gpu::Sampler>,
    /// 1x1 R8Unorm white placeholder bound to tone_map binding 7 when LIC is inactive.
    pub(crate) placeholder_view: Option<crate::gpu::TextureView>,
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

/// Highest per-item LIC strength the vector texture's blue channel can
/// encode: the surface pass writes `strength / LIC_STRENGTH_ENCODE_MAX` and
/// the advect pass decodes it back, so per-item strength survives the
/// Rgba8Unorm carrier. Strengths above this clamp.
pub(crate) const LIC_STRENGTH_ENCODE_MAX: f32 = 4.0;

/// Uniform for the LIC surface pass (per object, 80 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LicObjectUniform {
    pub(crate) model: [[f32; 4]; 4],
    /// The item's `SurfaceLICConfig::strength`, pre-normalised by
    /// `LIC_STRENGTH_ENCODE_MAX` for the vector texture's blue channel, so
    /// the advect output carries per-item strength per pixel.
    pub(crate) strength: f32,
    pub(crate) _pad: [f32; 3],
}

/// Per-frame GPU data for one Surface LIC item, created in `prepare()`.
pub struct LicSurfaceGpuData {
    /// Bind group (group 1): LicObjectUniform only. Flow vectors bound as vertex buffer 1.
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Owned uniform buffer for the model matrix. Kept alive by this struct.
    pub(crate) _object_uniform_buf: crate::gpu::Buffer,
    /// MeshId used to look up vertex + index buffers in the render pass.
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Name of the flow vector attribute for looking up the vertex buffer in the render pass.
    pub(crate) vector_attribute: String,
}
