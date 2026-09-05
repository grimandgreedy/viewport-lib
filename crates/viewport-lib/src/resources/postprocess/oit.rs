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
    /// OIT mesh pipeline (non-instanced, mesh_oit.wgsl, two colour targets),
    /// keyed by facedness (the only axis this family varies on -- cutout and
    /// no-discard are not real distinctions for a blended transparent pass).
    /// Without the two-sided variant a two-sided transparent surface loses
    /// its back faces on the OIT path.
    pub(crate) pipeline: Option<crate::renderer::pipeline_key::PipelineVariantSet>,
    /// OIT instanced mesh pipeline (mesh_instanced_oit.wgsl / mesh_instanced with OIT targets).
    pub(crate) instanced_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Two-sided (`cull_mode: None`) variant of `instanced_pipeline`.
    pub(crate) instanced_pipeline_two_sided: Option<crate::gpu::RenderPipeline>,
    /// OIT composite pipeline (oit_composite.wgsl, fullscreen tri, no depth).
    pub(crate) composite_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout for the OIT composite pass (group 0: accum + reveal + sampler).
    pub(crate) composite_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Linear clamp sampler shared by the OIT composite pass.
    pub(crate) composite_sampler: Option<crate::gpu::Sampler>,
}

#[cfg(test)]
mod tests {
    /// Same completeness guarantee as `scene_pipelines::hdr_opaque_resolves_every_key_once_built`,
    /// applied to the OIT family once it is built: every key in
    /// `PipelineKey::all()` must resolve through `get()` without panicking,
    /// including the `cutout` / `no_discard_eligible` combinations this
    /// family ignores (its only real axis is facedness).
    #[test]
    fn oit_pipeline_resolves_every_key_once_built() {
        let Some((device, queue, mut res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        res.ensure_hdr_shared(&device, &queue, crate::gpu::TextureFormat::Rgba8UnormSrgb);
        let oit = res
            .oit
            .pipeline
            .as_ref()
            .expect("ensure_hdr_shared must build the OIT pipeline");
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            let _ = oit.get(key);
        }
    }
}
