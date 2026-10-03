//! Weighted-blended order-independent transparency pipelines and layout.

use crate::renderer::pipeline_key::PipelineKey;
use crate::resources::pipeline_slot::LazyFamily;

/// What an OIT accumulate pipeline build reads.
pub(crate) struct OitContext {
    pub(crate) device: crate::gpu::Device,
    pub(crate) layout: crate::gpu::PipelineLayout,
    pub(crate) shader: crate::gpu::ShaderModule,
}

/// Weighted-blended OIT pipelines and composite layout.
///
/// Device-shared and lazily built: the mesh and instanced families are
/// composed by `ensure_oit_mesh_pipelines` and `ensure_oit_instanced_pipeline`
/// and their members built by the first draw that selects each; the composite
/// pipeline / BGL / sampler by the post-process setup. The viewport-sized
/// accumulation and reveal textures live on `ViewportHdrState`, not here.
#[derive(Default)]
pub(crate) struct OitResources {
    /// OIT mesh pipelines (non-instanced, mesh_oit.wgsl, two colour targets),
    /// keyed by facedness, the only axis this family varies on. Without the
    /// two-sided variant a two-sided transparent surface loses its back faces
    /// on the OIT path.
    pub(crate) pipeline: Option<LazyFamily<OitContext, 2>>,
    /// The instanced twins (mesh_instanced_oit.wgsl through `vs_main`).
    pub(crate) instanced: Option<LazyFamily<OitContext, 2>>,
    /// OIT composite pipeline (oit_composite.wgsl, fullscreen tri, no depth).
    pub(crate) composite_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Bind group layout for the OIT composite pass (group 0: accum + reveal + sampler).
    pub(crate) composite_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Linear clamp sampler shared by the OIT composite pass.
    pub(crate) composite_sampler: Option<crate::gpu::Sampler>,
}

pub(crate) fn build_per_object(ctx: &OitContext, i: usize) -> crate::gpu::RenderPipeline {
    crate::resources::mesh::mesh_pipelines::build_oit_pipeline(
        &ctx.device,
        &ctx.layout,
        &ctx.shader,
        i & 1 != 0,
    )
}

pub(crate) fn build_instanced(ctx: &OitContext, i: usize) -> crate::gpu::RenderPipeline {
    let two_sided = i & 1 != 0;
    crate::resources::mesh::mesh_pipelines::build_oit_instanced_pipeline(
        &ctx.device,
        &ctx.layout,
        &ctx.shader,
        if two_sided {
            "oit_instanced_pipeline_two_sided"
        } else {
            "oit_instanced_pipeline"
        },
        "vs_main",
        two_sided,
    )
}

impl OitResources {
    /// The per-object accumulate pipeline for `key`'s facedness, or `None`
    /// while a worker has it (or before the family is composed).
    pub(crate) fn per_object(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.pipeline.as_ref()?.get(key.two_sided as usize)
    }

    /// The instanced accumulate pipeline for `key`'s facedness.
    pub(crate) fn instanced(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.instanced.as_ref()?.get(key.two_sided as usize)
    }
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
        res.pipeline_compiler
            .set_policy(crate::resources::PipelineCompilation::Blocking);
        res.ensure_hdr_pipelines(&device, &queue, crate::gpu::TextureFormat::Rgba8UnormSrgb);
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            assert!(res.oit.per_object(key).is_some());
        }
    }
}
