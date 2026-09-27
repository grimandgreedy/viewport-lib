//! Core scene mesh pipelines: the base LDR set and their HDR-format variants.
//!
//! These draw plain `Material` surfaces (solid, two-sided, transparent,
//! wireframe). Every field is `Option`: the base set is built by
//! `ensure_ldr_mesh_pipelines` on the first frame carrying mesh-family content,
//! and the HDR twins the first time the HDR path runs. Grouped off
//! `DeviceResources` as a plain data holder; the builds stay in their own paths.

/// Base and HDR-variant pipelines for core scene surfaces.
pub(crate) struct SceneCorePipelines {
    /// Solid-shaded render pipeline (TriangleList topology, no blending).
    pub(crate) solid: Option<crate::gpu::RenderPipeline>,
    /// Solid-shaded render pipeline with back-face culling disabled (two-sided surfaces).
    pub(crate) solid_two_sided: Option<crate::gpu::RenderPipeline>,
    /// Transparent render pipeline (TriangleList topology, alpha blending).
    pub(crate) transparent: Option<crate::gpu::RenderPipeline>,
    /// Wireframe render pipeline (LineList topology, same shader).
    pub(crate) wireframe: Option<crate::gpu::RenderPipeline>,
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

/// The base pipelines are built by the first prepare that sees mesh-family
/// content, so a draw path that reaches for one is past that point. The
/// accessors below say so once rather than at every draw site.
const NOT_BUILT: &str =
    "base LDR mesh pipelines missing; prepare must run before paint on a frame with mesh content";

impl SceneCorePipelines {
    pub(crate) fn solid(&self) -> &crate::gpu::RenderPipeline {
        self.solid.as_ref().expect(NOT_BUILT)
    }

    pub(crate) fn solid_two_sided(&self) -> &crate::gpu::RenderPipeline {
        self.solid_two_sided.as_ref().expect(NOT_BUILT)
    }

    pub(crate) fn transparent(&self) -> &crate::gpu::RenderPipeline {
        self.transparent.as_ref().expect(NOT_BUILT)
    }

    pub(crate) fn wireframe(&self) -> &crate::gpu::RenderPipeline {
        self.wireframe.as_ref().expect(NOT_BUILT)
    }
}

impl crate::resources::DeviceResources {
    /// Build the base LDR mesh pipelines and the module they share. Only a frame
    /// carrying mesh-family content binds them, so the first such prepare calls
    /// this rather than paying for it at construction. No-op after that.
    ///
    /// `register_deformer` rebuilds the same four through
    /// `rebuild_mesh_pipelines`, which composes the registered deformers in; the
    /// composition here is the identity-hook one a renderer starts with.
    pub(crate) fn ensure_ldr_mesh_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.scene.solid.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let mesh_src = if self.deform.enabled {
            include_str!(concat!(env!("OUT_DIR"), "/mesh.wgsl"))
        } else {
            include_str!(concat!(env!("OUT_DIR"), "/mesh_noop.wgsl"))
        };
        let ldr = {
            let shader = crate::resources::builders::wgsl_module(
                device,
                "mesh_shader",
                crate::resources::builders::builtin_hook_env(
                    crate::resources::builders::strip_mesh_non_pbr(
                        crate::resources::builders::strip_mesh_discards(
                            crate::resources::builders::strip_debug_vis(
                                mesh_src,
                                self.debug_vis_shaders,
                            ),
                        ),
                    ),
                ),
            );
            let layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
                device,
                "mesh_pipeline_layout",
                &self.binds.camera_bgl,
                &self.binds.object_bgl,
                self.deform
                    .enabled
                    .then_some(&self.deform.bind_group_layout),
            );
            crate::resources::mesh::mesh_pipelines::build_ldr_mesh_pipelines(
                device,
                &layout,
                &shader,
                self.target_format,
                self.sample_count,
                self.pipeline_cache.as_ref(),
            )
        };
        self.scene.solid = Some(ldr.solid);
        self.scene.solid_two_sided = Some(ldr.solid_two_sided);
        self.scene.transparent = Some(ldr.transparent);
        self.scene.wireframe = Some(ldr.wireframe);
    }
}

#[cfg(test)]
mod tests {
    /// Every field starts empty: the base LDR set waits for the first frame with
    /// mesh content and the HDR twins for the first HDR frame. Guards the
    /// init-assembly grouping.
    #[test]
    fn scene_pipelines_start_empty() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        assert!(res.scene.solid.is_none());
        assert!(res.scene.solid_two_sided.is_none());
        assert!(res.scene.transparent.is_none());
        assert!(res.scene.wireframe.is_none());
        assert!(res.scene.hdr_opaque.is_none());
        assert!(res.scene.hdr_transparent.is_none());
        assert!(res.scene.hdr_wireframe.is_none());
        assert!(res.scene.hdr_overlay.is_none());
    }

    /// The completeness guarantee, applied to the one family migrated to
    /// `PipelineVariantSet` so far: once built, every key in
    /// `PipelineKey::all()` must resolve
    /// through `get()` without panicking. For this family that is guaranteed by
    /// `PipelineVariantSet::build`'s signature (it returns a concrete pipeline,
    /// never `None`), so this test is a regression guard on that contract
    /// rather than a check that could currently fail -- it exists so that if a
    /// future refactor ever reintroduces an `Option` here, CI catches the
    /// regression on every backend the tests run, not just the ones a human
    /// happens to eyeball.
    #[test]
    fn hdr_opaque_resolves_every_key_once_built() {
        let Some((device, queue, mut res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        res.ensure_hdr_pipelines(&device, &queue, crate::gpu::TextureFormat::Rgba8UnormSrgb);
        let hdr_opaque = res
            .scene
            .hdr_opaque
            .as_ref()
            .expect("ensure_hdr_pipelines must build hdr_opaque");
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            // Must not panic for any of the 8 keys, including the `cutout`
            // combinations this family ignores (see the field doc on
            // `hdr_opaque`): every key still has to resolve to *some*
            // pipeline, even one shared with its sibling key.
            let _ = hdr_opaque.get(key);
        }
    }
}
