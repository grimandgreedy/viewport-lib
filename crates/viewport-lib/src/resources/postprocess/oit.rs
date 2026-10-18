//! Weighted-blended order-independent transparency pipelines and layout.

use crate::renderer::pipeline_key::PipelineKey;
use crate::resources::pipeline_slot::LazyFamily;

/// What an OIT accumulate pipeline build reads.
pub(crate) struct OitContext {
    pub(crate) device: crate::gpu::Device,
    pub(crate) layout: crate::gpu::PipelineLayout,
    pub(crate) shader: crate::resources::pipeline_slot::LazyModule,
}

/// Weighted-blended OIT pipelines and composite layout.
///
/// Device-shared and lazily built: the mesh family is composed by
/// `ensure_oit_mesh_pipelines` and its members built by the first draw that
/// selects each; the composite
/// pipeline / BGL / sampler by the post-process setup. The viewport-sized
/// accumulation and reveal textures live on `ViewportHdrState`, not here.
#[derive(Default)]
pub(crate) struct OitResources {
    /// OIT mesh pipelines (non-instanced, mesh_oit.wgsl, two colour targets),
    /// keyed by facedness, the only axis this family varies on. Without the
    /// two-sided variant a two-sided transparent surface loses its back faces
    /// on the OIT path.
    pub(crate) pipeline: Option<LazyFamily<OitContext, 2>>,
    /// The same family on the bindless member of group 1, built alongside
    /// it when material textures bind bindlessly.
    pub(crate) pipeline_bindless: Option<LazyFamily<OitContext, 2>>,
    /// OIT composite pipeline (oit_composite.wgsl, fullscreen tri, no depth).
    pub(crate) composite_pipeline: Option<super::LazyFullscreen>,
    /// Bind group layout for the OIT composite pass (group 0: accum + reveal + sampler).
    pub(crate) composite_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Linear clamp sampler shared by the OIT composite pass.
    pub(crate) composite_sampler: Option<crate::gpu::Sampler>,
}

pub(crate) fn build_per_object(ctx: &OitContext, i: usize) -> crate::gpu::RenderPipeline {
    build_per_object_features(ctx, i, crate::renderer::pipeline_key::FeatureSet::UNIVERSAL)
}

/// Per-object member `i` built with `feats`.
pub(crate) fn build_per_object_features(
    ctx: &OitContext,
    i: usize,
    feats: crate::renderer::pipeline_key::FeatureSet,
) -> crate::gpu::RenderPipeline {
    let label = if feats == crate::renderer::pipeline_key::FeatureSet::UNIVERSAL {
        "oit_pipeline".to_owned()
    } else {
        feats.label("oit_pipeline")
    };
    crate::resources::mesh::mesh_pipelines::build_oit_pipeline_with(
        &ctx.device,
        &ctx.layout,
        ctx.shader.get(),
        i & 1 != 0,
        &label,
        &feats.constants(),
    )
}

impl OitResources {
    /// The per-object accumulate pipeline for `key`'s facedness, or `None`
    /// while a worker has it (or before the family is composed).
    #[cfg(test)]
    pub(crate) fn per_object(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.pipeline.as_ref()?.get(key.two_sided as usize)
    }

    /// The per-object accumulate pipeline for `key` built with `feats`, with
    /// the universal one standing in while it compiles. `bindless` selects
    /// the family on the bindless member of group 1.
    pub(crate) fn per_object_features(
        &self,
        key: PipelineKey,
        feats: crate::renderer::pipeline_key::FeatureSet,
        bindless: bool,
    ) -> crate::resources::pipeline_slot::FeatureGet<'_, crate::gpu::RenderPipeline> {
        let family = if bindless {
            self.pipeline_bindless.as_ref()
        } else {
            self.pipeline.as_ref()
        };
        match family {
            Some(f) => f.get_features(key.two_sided as usize, feats),
            None => crate::resources::pipeline_slot::FeatureGet {
                pipeline: None,
                stand_in: false,
                capped: false,
            },
        }
    }

    /// Ask for the plain feature set of every member, on both group-1
    /// members. Asks for no universal member: see
    /// [`request_members`](Self::request_members).
    pub(crate) fn request_plain(&self) {
        for f in [&self.pipeline, &self.pipeline_bindless]
            .into_iter()
            .flatten()
        {
            for i in 0..2 {
                f.request_features(i, crate::renderer::pipeline_key::FeatureSet::NONE);
            }
        }
    }

    /// Ask for every member of the OIT mesh families, the universal
    /// pipelines among them.
    pub(crate) fn request_members(&self) {
        for f in [&self.pipeline, &self.pipeline_bindless]
            .into_iter()
            .flatten()
        {
            f.request_all();
        }
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
