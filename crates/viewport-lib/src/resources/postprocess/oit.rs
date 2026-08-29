//! Weighted-blended order-independent transparency pipelines and layout.

/// Weighted-blended OIT pipelines and composite layout.
///
/// Device-shared and lazily built: the mesh/instanced pipelines by
/// `ensure_oit_instanced_pipeline` and the shared pipeline builders, the
/// composite pipeline / BGL / sampler by the post-process setup. The
/// viewport-sized accumulation and reveal textures live on `ViewportHdrState`,
/// not here.
#[derive(Default)]
pub(crate) struct OitResources {
    /// OIT mesh pipeline (non-instanced, mesh_oit.wgsl, two colour targets).
    pub(crate) pipeline: Option<crate::gpu::RenderPipeline>,
    /// OIT instanced mesh pipeline (mesh_instanced_oit.wgsl / mesh_instanced with OIT targets).
    pub(crate) instanced_pipeline: Option<crate::gpu::RenderPipeline>,
    /// OIT composite pipeline (oit_composite.wgsl, fullscreen tri, no depth).
    pub(crate) composite_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout for the OIT composite pass (group 0: accum + reveal + sampler).
    pub(crate) composite_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Linear clamp sampler shared by the OIT composite pass.
    pub(crate) composite_sampler: Option<crate::gpu::Sampler>,
}
