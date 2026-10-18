//! Core scene mesh pipelines: the base LDR set and their HDR-format variants.
//!
//! These draw plain `Material` surfaces (solid, two-sided, transparent,
//! wireframe). Each family is `None` until the first frame that needs it
//! composes its shader and layout (`ensure_ldr_mesh_pipelines`,
//! `ensure_hdr_mesh_pipelines`); the pipelines themselves are built one at a
//! time by the first draw that selects each, under the renderer's compilation
//! policy. Every accessor returns `None` while a worker has the pipeline, and
//! the draw is skipped until it is ready.

use crate::renderer::pipeline_key::{FeatureSet, PipelineKey};
use crate::resources::pipeline_slot::FeatureGet;
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
    /// opaque fragment shader branches on a per-object uniform),
    /// transparent, wireframe, and the cap-fill overlay. The opaque and
    /// transparent members also come in feature sets, built as draws ask.
    pub(crate) hdr: Option<LazyFamily<HdrMeshContext, 9>>,
    /// The LDR and HDR families on the bindless member of group 1, built
    /// alongside the two above when material textures bind bindlessly. The
    /// batch draws whose bind groups are bindless draw through these.
    pub(crate) ldr_bindless: Option<LazyFamily<LdrMeshContext, 4>>,
    pub(crate) hdr_bindless: Option<LazyFamily<HdrMeshContext, 9>>,
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
/// The additive and premultiplied blends an explicit mesh-instance item can
/// ask for, beside the alpha-blended `HDR_TRANSPARENT`.
const HDR_ADDITIVE: usize = 7;
const HDR_PREMULTIPLIED: usize = 8;

fn hdr_opaque_slot(key: PipelineKey) -> usize {
    HDR_OPAQUE + key.two_sided as usize + 2 * key.no_discard_eligible as usize
}

/// A feature-set pipeline's label: the universal member's, suffixed with the
/// set.
fn feature_label(base: &'static str, feats: FeatureSet) -> std::borrow::Cow<'static, str> {
    if feats == FeatureSet::UNIVERSAL {
        base.into()
    } else {
        feats.label(base).into()
    }
}

fn build_ldr(ctx: &LdrMeshContext, i: usize) -> crate::gpu::RenderPipeline {
    build_ldr_features(ctx, i, FeatureSet::UNIVERSAL)
}

fn build_ldr_features(
    ctx: &LdrMeshContext,
    i: usize,
    feats: FeatureSet,
) -> crate::gpu::RenderPipeline {
    use crate::gpu::{BlendState, Face, PrimitiveTopology};
    let constants = feats.constants();
    let make = |label, cull, blend, topo, depth_write| {
        crate::resources::mesh::mesh_pipelines::ldr_mesh_pipeline_with(
            &ctx.device,
            &ctx.layout,
            ctx.shader.get(),
            ctx.target_format,
            ctx.sample_count,
            None,
            &feature_label(label, feats),
            cull,
            blend,
            topo,
            depth_write,
            &constants,
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
    build_hdr_features(ctx, i, FeatureSet::UNIVERSAL)
}

fn build_hdr_features(
    ctx: &HdrMeshContext,
    i: usize,
    feats: FeatureSet,
) -> crate::gpu::RenderPipeline {
    use crate::gpu::{BlendState, Face, PrimitiveTopology};
    let constants = feats.constants();
    let make = |shader, label, cull, blend, topo, depth_write| {
        crate::resources::mesh::mesh_pipelines::hdr_mesh_pipeline(
            &ctx.device,
            &ctx.layout,
            shader,
            &feature_label(label, feats),
            cull,
            blend,
            topo,
            depth_write,
            &constants,
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
        HDR_ADDITIVE => make(
            ctx.shader.get(),
            "hdr_additive_pipeline",
            None,
            Some(crate::resources::mesh::mesh_pipelines::ADDITIVE_BLEND),
            PrimitiveTopology::TriangleList,
            false,
        ),
        HDR_PREMULTIPLIED => make(
            ctx.shader.get(),
            "hdr_premultiplied_pipeline",
            None,
            Some(crate::resources::mesh::mesh_pipelines::PREMULTIPLIED_BLEND),
            PrimitiveTopology::TriangleList,
            false,
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

    pub(crate) fn wireframe(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.ldr.as_ref()?.get(LDR_WIREFRAME)
    }

    /// Ask the HDR families for every member and the plain feature set of
    /// every opaque and transparent member, so a plain lit scene draws no
    /// stand-in after warm-up.
    pub(crate) fn request_plain_hdr(&self) {
        for f in [&self.hdr, &self.hdr_bindless].into_iter().flatten() {
            f.request_all();
            for key in [false, true].into_iter().flat_map(|two_sided| {
                [false, true].map(|no_discard_eligible| PipelineKey {
                    two_sided,
                    no_discard_eligible,
                    ..PipelineKey::default()
                })
            }) {
                f.get_features(hdr_opaque_slot(key), FeatureSet::NONE);
            }
            f.get_features(HDR_TRANSPARENT, FeatureSet::NONE);
        }
    }

    /// [`request_plain_hdr`](Self::request_plain_hdr) for the LDR families.
    pub(crate) fn request_plain_ldr(&self) {
        for f in [&self.ldr, &self.ldr_bindless].into_iter().flatten() {
            f.request_all();
            for slot in [LDR_SOLID, LDR_SOLID_TWO_SIDED, LDR_TRANSPARENT] {
                f.get_features(slot, FeatureSet::NONE);
            }
        }
    }

    /// Feature sets asked of the LDR and HDR families.
    pub(crate) fn feature_sets(&self) -> usize {
        self.ldr.as_ref().map_or(0, |f| f.feature_sets())
            + self.hdr.as_ref().map_or(0, |f| f.feature_sets())
            + self.ldr_bindless.as_ref().map_or(0, |f| f.feature_sets())
            + self.hdr_bindless.as_ref().map_or(0, |f| f.feature_sets())
    }

    pub(crate) fn hdr_wireframe(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.hdr.as_ref()?.get(HDR_WIREFRAME)
    }

    pub(crate) fn hdr_overlay(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.hdr.as_ref()?.get(HDR_OVERLAY)
    }

    /// The HDR blended pipeline an explicit mesh-instance item draws with.
    pub(crate) fn hdr_blend(
        &self,
        blend: crate::renderer::SpriteBlend,
    ) -> Option<&crate::gpu::RenderPipeline> {
        use crate::renderer::SpriteBlend;
        self.hdr.as_ref()?.get(match blend {
            SpriteBlend::AlphaBlend => HDR_TRANSPARENT,
            SpriteBlend::Additive => HDR_ADDITIVE,
            SpriteBlend::Premultiplied => HDR_PREMULTIPLIED,
        })
    }

    /// Whether the opaque pipeline for `key` in the `hdr` or LDR family is
    /// built, without starting it.
    pub(crate) fn opaque_ready(&self, hdr: bool, key: PipelineKey) -> bool {
        if hdr {
            self.hdr
                .as_ref()
                .is_some_and(|f| f.is_ready(hdr_opaque_slot(key)))
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
        LdrScene(self.ldr.as_ref())
    }

    /// The HDR family seen through [`MeshColourFamily`].
    pub(crate) fn hdr_family(&self) -> HdrScene<'_> {
        HdrScene(self.hdr.as_ref())
    }

    /// The LDR family on the bindless member of group 1.
    pub(crate) fn ldr_bindless_family(&self) -> LdrScene<'_> {
        LdrScene(self.ldr_bindless.as_ref())
    }

    /// The HDR family on the bindless member of group 1.
    pub(crate) fn hdr_bindless_family(&self) -> HdrScene<'_> {
        HdrScene(self.hdr_bindless.as_ref())
    }
}

/// The three pipelines a per-object mesh draw selects between, for either
/// colour family, so one draw helper serves the LDR and HDR paths.
pub(crate) trait MeshColourFamily {
    fn opaque(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline>;
    fn transparent(&self) -> Option<&crate::gpu::RenderPipeline>;
    fn wireframe(&self) -> Option<&crate::gpu::RenderPipeline>;
    /// [`opaque`](Self::opaque) built with `feats`, the universal pipeline
    /// standing in while it compiles.
    fn opaque_features(
        &self,
        key: PipelineKey,
        feats: FeatureSet,
    ) -> FeatureGet<'_, crate::gpu::RenderPipeline>;
    /// [`transparent`](Self::transparent) built with `feats`.
    fn transparent_features(&self, feats: FeatureSet)
    -> FeatureGet<'_, crate::gpu::RenderPipeline>;
    /// Whether an opaque draw for `key` with `feats` has a pipeline to
    /// bind, without starting one.
    fn opaque_ready(&self, key: PipelineKey, feats: FeatureSet) -> bool;
}

fn feature_get<'a, C: Send + Sync + 'static, const N: usize>(
    family: Option<&'a LazyFamily<C, N>>,
    i: usize,
    feats: FeatureSet,
) -> FeatureGet<'a, crate::gpu::RenderPipeline> {
    match family {
        Some(f) => f.get_features(i, feats),
        None => FeatureGet {
            pipeline: None,
            stand_in: false,
            capped: false,
        },
    }
}

pub(crate) struct LdrScene<'a>(Option<&'a LazyFamily<LdrMeshContext, 4>>);

fn ldr_solid_slot(key: PipelineKey) -> usize {
    if key.two_sided {
        LDR_SOLID_TWO_SIDED
    } else {
        LDR_SOLID
    }
}

impl MeshColourFamily for LdrScene<'_> {
    fn opaque(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.0?.get(ldr_solid_slot(key))
    }

    fn transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.0?.get(LDR_TRANSPARENT)
    }

    fn wireframe(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.0?.get(LDR_WIREFRAME)
    }

    fn opaque_features(
        &self,
        key: PipelineKey,
        feats: FeatureSet,
    ) -> FeatureGet<'_, crate::gpu::RenderPipeline> {
        feature_get(self.0, ldr_solid_slot(key), feats)
    }

    fn transparent_features(
        &self,
        feats: FeatureSet,
    ) -> FeatureGet<'_, crate::gpu::RenderPipeline> {
        feature_get(self.0, LDR_TRANSPARENT, feats)
    }

    fn opaque_ready(&self, key: PipelineKey, feats: FeatureSet) -> bool {
        self.0
            .is_some_and(|f| f.features_ready(ldr_solid_slot(key), feats))
    }
}

pub(crate) struct HdrScene<'a>(Option<&'a LazyFamily<HdrMeshContext, 9>>);

impl MeshColourFamily for HdrScene<'_> {
    fn opaque(&self, key: PipelineKey) -> Option<&crate::gpu::RenderPipeline> {
        self.0?.get(hdr_opaque_slot(key))
    }

    fn transparent(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.0?.get(HDR_TRANSPARENT)
    }

    fn wireframe(&self) -> Option<&crate::gpu::RenderPipeline> {
        self.0?.get(HDR_WIREFRAME)
    }

    fn opaque_features(
        &self,
        key: PipelineKey,
        feats: FeatureSet,
    ) -> FeatureGet<'_, crate::gpu::RenderPipeline> {
        feature_get(self.0, hdr_opaque_slot(key), feats)
    }

    fn transparent_features(
        &self,
        feats: FeatureSet,
    ) -> FeatureGet<'_, crate::gpu::RenderPipeline> {
        feature_get(self.0, HDR_TRANSPARENT, feats)
    }

    fn opaque_ready(&self, key: PipelineKey, feats: FeatureSet) -> bool {
        self.0
            .is_some_and(|f| f.features_ready(hdr_opaque_slot(key), feats))
    }
}

impl crate::resources::DeviceResources {
    /// Feature-set pipelines asked of the per-object mesh families.
    pub(crate) fn feature_sets(&self) -> usize {
        self.scene.feature_sets()
            + self.cull.feature_sets()
            + self.oit.pipeline.as_ref().map_or(0, |f| f.feature_sets())
            + self
                .oit
                .pipeline_bindless
                .as_ref()
                .map_or(0, |f| f.feature_sets())
    }

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
        let bindless = if self.bindless_textures() {
            let shader = self.shared_module(
                device,
                "mesh_bindless_shader",
                &crate::resources::mesh::instanced_bindless::bindlessify_mesh(&source),
            );
            Some((
                shader,
                self.object_bindless_layout(device, "mesh_bindless_pipeline_layout"),
            ))
        } else {
            None
        };
        let family = |layout, shader| {
            LazyFamily::new(
                LdrMeshContext {
                    device: device.clone(),
                    layout,
                    shader,
                    target_format: self.target_format,
                    sample_count: self.sample_count,
                },
                std::sync::Arc::clone(&self.pipeline_compiler),
                build_ldr,
            )
            .with_features(build_ldr_features)
        };
        let ldr = family(layout, shader);
        let ldr_bindless = bindless.map(|(shader, layout)| family(layout, shader));
        self.scene.ldr = Some(ldr);
        self.scene.ldr_bindless = ldr_bindless;
    }

    /// The `mesh.wgsl` source the HDR families compile: composed with the
    /// registered deformers and the debug-vis state. Materialised so the
    /// discard-free twin is stripped from the exact source the discarding
    /// module compiles, and shared by the scene and GPU-culled families.
    pub(crate) fn hdr_mesh_source(&self) -> String {
        let base = if self.deform.enabled {
            include_str!(concat!(env!("OUT_DIR"), "/mesh.wgsl"))
        } else {
            include_str!(concat!(env!("OUT_DIR"), "/mesh_noop.wgsl"))
        };
        let source =
            crate::resources::mesh_sidecar::registry::compose_shader(base, &self.deform.registrations);
        crate::resources::builders::builtin_hook_env(
            crate::resources::builders::strip_debug_vis(source, self.debug_vis_shaders),
        )
        .into_owned()
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
        let final_src = self.hdr_mesh_source();
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
        let bindless = if self.bindless_textures() {
            let source = crate::resources::mesh::instanced_bindless::bindlessify_mesh(&final_src);
            let shader = self.shared_module(device, "mesh_bindless_shader_hdr", &source);
            let shader_nodiscard = self.shared_module(
                device,
                "mesh_bindless_shader_hdr_nodiscard",
                &crate::resources::builders::strip_discards(&source),
            );
            Some((
                shader,
                shader_nodiscard,
                self.object_bindless_layout(device, "hdr_mesh_bindless_pipeline_layout"),
            ))
        } else {
            None
        };
        let family = |layout, shader, shader_nodiscard| {
            LazyFamily::new(
                HdrMeshContext {
                    device: device.clone(),
                    layout,
                    shader,
                    shader_nodiscard,
                    overlay_layout: overlay_layout.clone(),
                    overlay_shader: overlay_shader.clone(),
                },
                std::sync::Arc::clone(&self.pipeline_compiler),
                build_hdr,
            )
            .with_features(build_hdr_features)
        };
        let hdr = family(layout, shader, shader_nodiscard);
        let hdr_bindless = bindless.map(|(s, n, l)| family(l, s, n));
        self.scene.hdr = Some(hdr);
        self.scene.hdr_bindless = hdr_bindless;
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
            use super::MeshColourFamily as _;
            assert!(res.scene.hdr_family().opaque(key).is_some());
        }
    }

    /// Families compiled from one source share one module. The HDR and
    /// culled mesh families use the same composed `mesh.wgsl`, and the OIT
    /// family and its culled twin the same `mesh_oit.wgsl`, so composing a
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

        // A bindless device composes every mesh family on both group-1
        // members, so each module has a bindless twin.
        let b = usize::from(res.bindless_textures());
        res.ensure_ldr_mesh_pipelines(&device);
        assert_eq!(modules(&res), 1 + b, "the LDR mesh family is one module");
        res.ensure_hdr_mesh_pipelines(&device);
        assert_eq!(
            modules(&res),
            2 + 2 * b,
            "the HDR mesh family adds its discard-free twin and nothing else"
        );

        res.ensure_instanced_pipelines(&device);
        assert_eq!(modules(&res), 3 + 2 * b, "the instanced shadow module");
        res.ensure_cull_instance_pipelines(&device);
        res.ensure_hdr_cull_pipelines(&device);
        assert_eq!(
            modules(&res),
            3 + 2 * b,
            "the culled HDR family and the culled shadows reuse the modules above"
        );

        res.ensure_oit_cull_pipelines(&device);
        res.ensure_oit_mesh_pipelines(&device);
        assert_eq!(
            modules(&res),
            4 + 3 * b,
            "the OIT family and its culled twin share one module"
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
