//! Core scene mesh pipelines: the base LDR set and their HDR-format variants.
//!
//! These draw plain `Material` surfaces (solid, two-sided, transparent,
//! wireframe). Each family is `None` until the first frame that needs it
//! composes its shader and layout (`ensure_ldr_mesh_pipelines`,
//! `ensure_hdr_mesh_pipelines`); the pipelines themselves are built one at a
//! time by the first draw that selects each, under the renderer's compilation
//! policy. Every accessor returns `None` while a worker has the pipeline, and
//! the draw is skipped until it is ready.

use crate::renderer::pipeline_key::PipelineKey;
use crate::resources::pipeline_slot::LazyFamily;

/// What every LDR mesh pipeline build reads.
pub(crate) struct LdrMeshContext {
    device: crate::gpu::Device,
    layout: crate::gpu::PipelineLayout,
    shader: crate::resources::pipeline_slot::LazyModule,
    target_format: crate::gpu::TextureFormat,
    sample_count: u32,
}

/// What every HDR mesh pipeline build reads: the lit module and its
/// discard-free twin, plus the overlay pair the cap fill draws with.
pub(crate) struct HdrMeshContext {
    device: crate::gpu::Device,
    layout: crate::gpu::PipelineLayout,
    shader: crate::resources::pipeline_slot::LazyModule,
    shader_nodiscard: crate::resources::pipeline_slot::LazyModule,
    overlay_layout: crate::gpu::PipelineLayout,
    overlay_shader: crate::gpu::ShaderModule,
}

/// Base and HDR-variant pipelines for core scene surfaces.
#[derive(Default)]
pub(crate) struct SceneCorePipelines {
    /// The four LDR pipelines (solid, two-sided, transparent, wireframe),
    /// drawing into the swapchain format.
    pub(crate) ldr: Option<LazyFamily<LdrMeshContext, 4>>,
    /// The HDR family: four opaque pipelines keyed by facedness and
    /// discard-free early-Z eligibility (`cutout` is not a real axis: the
    /// opaque fragment shader branches on a per-object uniform), transparent,
    /// wireframe, and the cap-fill overlay.
    pub(crate) hdr: Option<LazyFamily<HdrMeshContext, 7>>,
}

const LDR_SOLID: usize = 0;
const LDR_SOLID_TWO_SIDED: usize = 1;
const LDR_TRANSPARENT: usize = 2;
const LDR_WIREFRAME: usize = 3;

/// Four slots: bit 0 is two-sided, bit 1 is discard-free.
const HDR_OPAQUE: usize = 0;
const HDR_TRANSPARENT: usize = 4;
const HDR_WIREFRAME: usize = 5;
const HDR_OVERLAY: usize = 6;

fn build_ldr(ctx: &LdrMeshContext, i: usize) -> crate::gpu::RenderPipeline {
    use crate::gpu::{BlendState, Face, PrimitiveTopology};
    let make = |label, cull, blend, topo, depth_write| {
        crate::resources::mesh::mesh_pipelines::ldr_mesh_pipeline(
            &ctx.device,
            &ctx.layout,
            ctx.shader.get(),
            ctx.target_format,
            ctx.sample_count,
            None,
            label,
            cull,
            blend,
            topo,
            depth_write,
        )
    };
    match i {
        LDR_SOLID => make(
            "solid_pipeline",
            Some(Face::Back),
            None,
            PrimitiveTopology::TriangleList,
            true,
        ),
        LDR_SOLID_TWO_SIDED => make(
            "solid_two_sided_pipeline",
            None,
            None,
            PrimitiveTopology::TriangleList,
            true,
        ),
        // `ALPHA_BLENDING` rather than a hand-written state: its alpha
        // component is `OVER`, so a transparent draw composes with the
        // destination's coverage instead of replacing it.
        LDR_TRANSPARENT => make(
            "transparent_pipeline",
            None,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleList,
            false,
        ),
        _ => make(
            "wireframe_pipeline",
            None,
            None,
            PrimitiveTopology::LineList,
            true,
        ),
    }
}

fn build_hdr(ctx: &HdrMeshContext, i: usize) -> crate::gpu::RenderPipeline {
    use crate::gpu::{BlendState, Face, PrimitiveTopology};
    let make = |shader, label, cull, blend, topo, depth_write| {
        crate::resources::mesh::mesh_pipelines::hdr_mesh_pipeline(
            &ctx.device,
            &ctx.layout,
            shader,
            label,
            cull,
            blend,
            topo,
            depth_write,
        )
    };
    match i {
        0..=3 => {
            let two_sided = (i - HDR_OPAQUE) & 1 != 0;
            let nodiscard = (i - HDR_OPAQUE) & 2 != 0;
            // Early-Z twin: identical shading with every `discard;` removed,
            // valid only for draws that would not have discarded (see the
            // per-object gate in hdr_path.rs).
            let shader = if nodiscard {
                ctx.shader_nodiscard.get()
            } else {
                ctx.shader.get()
            };
            let label = match (two_sided, nodiscard) {
                (false, false) => "hdr_solid_pipeline",
                (true, false) => "hdr_solid_two_sided_pipeline",
                (false, true) => "hdr_solid_nodiscard_pipeline",
                (true, true) => "hdr_solid_two_sided_nodiscard_pipeline",
            };
            make(
                shader,
                label,
                (!two_sided).then_some(Face::Back),
                None,
                PrimitiveTopology::TriangleList,
                true,
            )
        }
        HDR_TRANSPARENT => make(
            ctx.shader.get(),
            "hdr_transparent_pipeline",
            None,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleList,
            false,
        ),
        HDR_WIREFRAME => make(
            ctx.shader.get(),
            "hdr_wireframe_pipeline",
            None,
            None,
            PrimitiveTopology::LineList,
            true,
        ),
        _ => crate::resources::builders::render_pipeline(
            &ctx.device,
            crate::resources::builders::RenderPipelineDesc {
                label: "hdr_overlay_pipeline",
                layout: &ctx.overlay_layout,
                vertex_module: &ctx.overlay_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[crate::resources::OverlayVertex::buffer_layout()],
                fragment: Some(crate::gpu::FragmentState {
                    module: &ctx.overlay_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        ),
    }
}

impl SceneCorePipelines {
    pub(crate) fn solid(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.ldr.as_ref()?.get(LDR_SOLID)
    }

    pub(crate) fn solid_two_sided(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.ldr.as_ref()?.get(LDR_SOLID_TWO_SIDED)
    }

    /// The LDR solid pipeline for `key`'s facedness.
    pub(crate) fn ldr_opaque(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        if key.two_sided {
            self.solid_two_sided()
        } else {
            self.solid()
        }
    }

    pub(crate) fn transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.ldr.as_ref()?.get(LDR_TRANSPARENT)
    }

    pub(crate) fn wireframe(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.ldr.as_ref()?.get(LDR_WIREFRAME)
    }

    /// The HDR opaque pipeline for `key`: facedness and discard-free
    /// eligibility.
    pub(crate) fn hdr_opaque(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.hdr
            .as_ref()?
            .get(HDR_OPAQUE + key.two_sided as usize + 2 * key.no_discard_eligible as usize)
    }

    pub(crate) fn hdr_transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.hdr.as_ref()?.get(HDR_TRANSPARENT)
    }

    pub(crate) fn hdr_wireframe(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.hdr.as_ref()?.get(HDR_WIREFRAME)
    }

    pub(crate) fn hdr_overlay(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.hdr.as_ref()?.get(HDR_OVERLAY)
    }

    /// Whether the opaque pipeline for `key` in the `hdr` or LDR family is
    /// built, without starting it.
    pub(crate) fn opaque_ready(&self, hdr: bool, key: PipelineKey) -> bool {
        if hdr {
            self.hdr.as_ref().is_some_and(|f| {
                f.is_ready(
                    HDR_OPAQUE + key.two_sided as usize + 2 * key.no_discard_eligible as usize,
                )
            })
        } else {
            self.ldr.as_ref().is_some_and(|f| {
                f.is_ready(if key.two_sided {
                    LDR_SOLID_TWO_SIDED
                } else {
                    LDR_SOLID
                })
            })
        }
    }

    /// The LDR family seen through [`MeshColourFamily`].
    pub(crate) fn ldr_family(&self) -> LdrScene<'_> {
        LdrScene(self)
    }

    /// The HDR family seen through [`MeshColourFamily`].
    pub(crate) fn hdr_family(&self) -> HdrScene<'_> {
        HdrScene(self)
    }
}

/// The three pipelines a per-object mesh draw selects between, for either
/// colour family, so one draw helper serves the LDR and HDR paths.
pub(crate) trait MeshColourFamily {
    fn opaque(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline>;
    fn transparent(&self) -> Option<&crate::gpu::RenderPipeline>;
    fn wireframe(&self) -> Option<&crate::gpu::RenderPipeline>;
}

pub(crate) struct LdrScene<'a>(&'a SceneCorePipelines);

impl MeshColourFamily for LdrScene<'_> {
    fn opaque(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.0.ldr_opaque(key)
    }

    fn transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.0.transparent()
    }

    fn wireframe(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.0.wireframe()
    }
}

pub(crate) struct HdrScene<'a>(&'a SceneCorePipelines);

impl MeshColourFamily for HdrScene<'_> {
    fn opaque(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.0.hdr_opaque(key)
    }

    fn transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.0.hdr_transparent()
    }

    fn wireframe(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.0.hdr_wireframe()
    }
}

impl crate::resources::DeviceResources {
    /// Compose the base LDR mesh family: the module it shares with the HDR
    /// family and its layout. Only a frame carrying mesh-family content binds
    /// these, so the first such prepare calls this rather than paying for it
    /// at construction. No-op after that. The pipelines are built as draws
    /// select them.
    ///
    /// Composed with the registered deformers, so `register_deformer` rebuilds
    /// the family through here once it exists.
    pub(crate) fn ensure_ldr_mesh_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.scene.ldr.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let mesh_src = if self.deform.enabled {
            include_str!(concat!(env!("OUT_DIR"), "/mesh.wgsl"))
        } else {
            include_str!(concat!(env!("OUT_DIR"), "/mesh_noop.wgsl"))
        };
        let source = crate::resources::builders::builtin_hook_env(
            crate::resources::builders::strip_mesh_non_pbr(
                crate::resources::builders::strip_mesh_discards(
                    crate::resources::builders::strip_debug_vis(
                        crate::resources::mesh_sidecar::registry::compose_shader(
                            mesh_src,
                            &self.deform.registrations,
                        ),
                        self.debug_vis_shaders,
                    ),
                ),
            ),
        );
        // The HDR family compiles the same source, so the module is shared.
        let shader = self.shared_module(device, "mesh_shader", &source);
        let layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            "mesh_pipeline_layout",
            &self.binds.camera_bgl,
            &self.binds.object_bgl,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        self.scene.ldr = Some(LazyFamily::new(
            LdrMeshContext {
                device: device.clone(),
                layout,
                shader,
                target_format: self.target_format,
                sample_count: self.sample_count,
            },
            std::sync::Arc::clone(&self.pipeline_compiler),
            build_ldr,
        ));
    }

    /// Compose the HDR mesh family: the lit module (shared with the LDR
    /// family), its discard-free twin, the layouts, and the overlay module
    /// the cap fill uses. Composed with the registered deformers, so a
    /// registration made before the first HDR mesh frame is picked up here.
    pub(crate) fn ensure_hdr_mesh_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.scene.hdr.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let source = {
            let base = if self.deform.enabled {
                include_str!(concat!(env!("OUT_DIR"), "/mesh.wgsl"))
            } else {
                include_str!(concat!(env!("OUT_DIR"), "/mesh_noop.wgsl"))
            };
            crate::resources::mesh_sidecar::registry::compose_shader(
                base,
                &self.deform.registrations,
            )
        };
        // Materialised so the discard-free twin is stripped from the exact
        // source the discarding module compiles.
        let final_src = crate::resources::builders::builtin_hook_env(
            crate::resources::builders::strip_debug_vis(source, self.debug_vis_shaders),
        )
        .into_owned();
        let shader = self.shared_module(device, "mesh_shader_hdr", &final_src);
        let shader_nodiscard = self.shared_module(
            device,
            "mesh_shader_hdr_nodiscard",
            &crate::resources::builders::strip_discards(&final_src),
        );
        let layout = crate::resources::mesh::mesh_pipelines::mesh_pipeline_layout(
            device,
            "hdr_mesh_pipeline_layout",
            &self.binds.camera_bgl,
            &self.binds.object_bgl,
            self.deform
                .enabled
                .then_some(&self.deform.bind_group_layout),
        );
        let overlay_shader = crate::resources::builders::wgsl_module(
            device,
            "overlay_shader_hdr",
            crate::resources::builders::wgsl_source!("overlay"),
        );
        let overlay_layout = crate::resources::builders::pipeline_layout(
            device,
            "hdr_overlay_pipeline_layout",
            &[&self.binds.camera_bgl, &self.guides.overlay_bgl],
        );
        self.scene.hdr = Some(LazyFamily::new(
            HdrMeshContext {
                device: device.clone(),
                layout,
                shader,
                shader_nodiscard,
                overlay_layout,
                overlay_shader,
            },
            std::sync::Arc::clone(&self.pipeline_compiler),
            build_hdr,
        ));
    }
}

#[cfg(test)]
mod tests {
    /// Both families start empty: the LDR one waits for the first frame with
    /// mesh content and the HDR one for the first HDR frame.
    #[test]
    fn scene_pipelines_start_empty() {
        let Some((_device, _queue, res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        assert!(res.scene.ldr.is_none());
        assert!(res.scene.hdr.is_none());
    }

    /// Once the HDR family is composed, every key resolves to a pipeline
    /// under `Blocking`, including the `cutout` combinations this family
    /// ignores.
    #[test]
    fn hdr_opaque_resolves_every_key_once_built() {
        let Some((device, queue, mut res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        res.pipeline_compiler
            .set_policy(crate::resources::PipelineCompilation::Blocking);
        res.ensure_hdr_pipelines(&device, &queue, crate::gpu::TextureFormat::Rgba8UnormSrgb);
        for key in crate::renderer::pipeline_key::PipelineKey::all() {
            assert!(res.scene.hdr_opaque(key).is_some());
        }
    }

    /// Families compiled from one source share one module. The LDR and HDR
    /// mesh families use the same `mesh.wgsl`, and the LDR, HDR and culled
    /// instanced families the same `mesh_instanced.wgsl`, so composing a
    /// second family adds only the modules the first did not need. Composing
    /// compiles none of them: the first pipeline build that reads a module
    /// does.
    #[test]
    fn pipeline_families_share_their_shader_modules() {
        let Some((device, _queue, mut res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let modules = |res: &crate::DeviceResources| res.shader_modules.lock().unwrap().len();
        let compiled = |res: &crate::DeviceResources| {
            res.shader_modules
                .lock()
                .unwrap()
                .values()
                .filter(|cell| cell.get().is_some())
                .count()
        };

        res.ensure_ldr_mesh_pipelines(&device);
        assert_eq!(modules(&res), 1, "the LDR mesh family is one module");
        res.ensure_hdr_mesh_pipelines(&device);
        assert_eq!(
            modules(&res),
            2,
            "the HDR mesh family adds its discard-free twin and nothing else"
        );

        res.ensure_instanced_pipelines(&device);
        res.ensure_ldr_instanced_pipelines(&device);
        assert_eq!(
            modules(&res),
            5,
            "the instanced module, its twin and the instanced shadow module"
        );
        // Under bindless the blended HDR pipelines stay on the per-batch
        // source, which is one more module; the solids share the pair above.
        let per_batch = usize::from(res.bindless_textures());
        res.ensure_hdr_instanced_pipelines(&device);
        res.ensure_cull_instance_pipelines(&device);
        res.ensure_hdr_cull_pipelines(&device);
        assert_eq!(
            modules(&res),
            5 + per_batch,
            "the HDR and culled instanced families, and the culled shadows, reuse the modules above"
        );

        res.ensure_oit_instanced_pipeline(&device);
        res.ensure_oit_cull_pipelines(&device);
        assert_eq!(
            modules(&res),
            6 + per_batch,
            "the OIT instanced pipelines and their culled twins share one module"
        );
        assert_eq!(compiled(&res), 0, "composing a family compiled a module");

        res.pipeline_compiler
            .set_policy(crate::resources::PipelineCompilation::Blocking);
        res.scene.hdr.as_ref().unwrap().get(0);
        assert_eq!(
            compiled(&res),
            1,
            "one build compiles the one module it reads"
        );
    }
}
