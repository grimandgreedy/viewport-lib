//! Core scene mesh pipelines: the base LDR set and their HDR-format variants.
//!
//! These draw plain `Material` surfaces (solid, two-sided, transparent,
//! wireframe). The HDR twins are built lazily the first time the HDR path runs,
//! so they are `Option`. Grouped off `DeviceResources` as a plain data holder;
//! construction and the lazy HDR builds stay in their existing paths.

/// Base and HDR-variant pipelines for core scene surfaces.
pub(crate) struct SceneCorePipelines {
    /// Solid-shaded render pipeline (TriangleList topology, no blending).
    pub(crate) solid: crate::gpu::RenderPipeline,
    /// Solid-shaded render pipeline with back-face culling disabled (two-sided surfaces).
    pub(crate) solid_two_sided: crate::gpu::RenderPipeline,
    /// Transparent render pipeline (TriangleList topology, alpha blending).
    pub(crate) transparent: crate::gpu::RenderPipeline,
    /// Wireframe render pipeline (LineList topology, same shader).
    pub(crate) wireframe: crate::gpu::RenderPipeline,
    /// Per-object HDR opaque pipelines, keyed by facedness and discard-free
    /// early-Z eligibility (`cutout` is not a real axis here: the opaque
    /// fragment shader branches on a per-object uniform instead of a
    /// dedicated pipeline). `None` until the HDR path first builds it. The
    /// instanced paths carry their own variant sets.
    pub(crate) hdr_opaque: Option<crate::renderer::pipeline_key::PipelineVariantSet>,
    pub(crate) hdr_transparent: Option<crate::gpu::RenderPipeline>,
    pub(crate) hdr_wireframe: Option<crate::gpu::RenderPipeline>,
    /// HDR overlay pipeline (TriangleList, Rgba16Float, alpha blending) for cap fill in HDR path.
    pub(crate) hdr_overlay: Option<crate::gpu::RenderPipeline>,
}

#[cfg(test)]
mod tests {
    /// The base LDR pipelines exist at construction; the HDR twins are lazy and
    /// start empty. Guards the init-assembly grouping.
    #[test]
    fn scene_pipelines_base_present_hdr_lazy() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        // Base pipelines are non-optional and built at new(); nothing to unwrap.
        // The HDR variants are built on first HDR-path use.
        assert!(res.scene.hdr_opaque.is_none());
        assert!(res.scene.hdr_transparent.is_none());
        assert!(res.scene.hdr_wireframe.is_none());
        assert!(res.scene.hdr_overlay.is_none());
    }
}
