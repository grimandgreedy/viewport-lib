//! GPU state for the sprite item type: the twelve blend/depth/lit render
//! pipeline variants, the refractive pipeline and its scene-colour layout, the
//! soft-particle depth layout and its fallback, the weighted-blended OIT
//! pipelines, the pick pipeline, and the selection-outline mask pipeline.
//!
//! The group-1 and group-3 bind group layouts live in `store`, beside the
//! upload that builds bind groups against them; this module borrows them to
//! build pipelines over them.

use viewport_lib::plugin_api::builders::DualPipeline;
use viewport_lib::plugin_api::builders::DualPipelineDesc;
use viewport_lib::plugin_api::shared_wgsl;
use viewport_lib::resources::DeviceResources;

/// Sprite pipeline variant axes: depth-write, blend mode, and unlit vs
/// `apply_scene_lighting`-lit shading. Kept separate from the mesh family's
/// `PipelineKey` -- sprite's axes don't map onto `two_sided` / `cutout` /
/// `no_discard_eligible`, and `blend` is three-valued, not boolean, so
/// reusing that type would just be confusing field names for an unrelated
/// set of axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SpriteKey {
    pub depth_write: bool,
    pub blend: viewport_lib::renderer::SpriteBlend,
    pub lit: bool,
}

impl SpriteKey {
    fn blend_index(self) -> usize {
        match self.blend {
            viewport_lib::renderer::SpriteBlend::AlphaBlend => 0,
            viewport_lib::renderer::SpriteBlend::Additive => 1,
            viewport_lib::renderer::SpriteBlend::Premultiplied => 2,
        }
    }

    /// Every axis combination, for eager cross-product construction
    /// (`SpriteVariantSet::build`).
    pub fn all() -> impl Iterator<Item = SpriteKey> {
        [
            viewport_lib::renderer::SpriteBlend::AlphaBlend,
            viewport_lib::renderer::SpriteBlend::Additive,
            viewport_lib::renderer::SpriteBlend::Premultiplied,
        ]
        .into_iter()
        .flat_map(|blend| {
            [false, true].into_iter().flat_map(move |lit| {
                [false, true].into_iter().map(move |depth_write| SpriteKey {
                    depth_write,
                    blend,
                    lit,
                })
            })
        })
    }

    /// Dense index in `0..12`, stable across calls, for the hash-free array
    /// lookup `SpriteVariantSet` uses.
    fn slot(self) -> usize {
        self.depth_write as usize + 2 * self.blend_index() + 6 * (self.lit as usize)
    }
}

/// A `DualPipeline` built for every reachable [`SpriteKey`], indexed for a
/// hash-free draw-time lookup (`get`). Construction is eager: `build` runs
/// once per key when `ensure_sprite_pipelines` first runs, not per draw call.
pub(crate) struct SpriteVariantSet {
    variants: [DualPipeline; 12],
}

/// Build one value per [`SpriteKey`] and place each at its own
/// [`slot`](SpriteKey::slot), which is *not* the order `all()` yields them in.
/// Collecting in iteration order instead puts variants under the wrong keys, so
/// `get` hands back a pipeline belonging to a different blend or lit-ness: an
/// unlit draw handed a lit pipeline fails validation, because the unlit draw
/// path never binds the lit pipeline's group-3 normal map.
fn place_by_slot<T>(mut build: impl FnMut(SpriteKey) -> T) -> Vec<T> {
    let mut slots: Vec<Option<T>> = (0..12).map(|_| None).collect();
    for key in SpriteKey::all() {
        slots[key.slot()] = Some(build(key));
    }
    slots
        .into_iter()
        .map(|v| v.unwrap_or_else(|| unreachable!("every slot is covered by all()")))
        .collect()
}

impl SpriteVariantSet {
    pub fn build(build: impl FnMut(SpriteKey) -> DualPipeline) -> Self {
        Self {
            variants: place_by_slot(build)
                .try_into()
                .unwrap_or_else(|_| unreachable!("SpriteKey::all() yields exactly 12 keys")),
        }
    }

    pub fn get(&self, key: SpriteKey) -> &DualPipeline {
        &self.variants[key.slot()]
    }
}

/// Every pipeline and layout the sprite draw hooks need, built on the first
/// prepare that sees an item.
pub(super) struct SpriteGpu {
    pub(super) pipelines: SpriteVariantSet,
    pub(super) refraction_pipeline: viewport_lib::gpu::RenderPipeline,
    pub(super) refraction_bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) refraction_sampler: viewport_lib::gpu::Sampler,
    pub(super) pick_pipeline: viewport_lib::gpu::RenderPipeline,
    pub(super) pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) soft_fallback_bg: viewport_lib::gpu::BindGroup,
    /// 1x1 Depth32Float texture backing the soft fallback bind group. Held so
    /// the bind group keeps a valid texture; not read after construction.
    #[allow(dead_code)]
    pub(super) soft_fallback_tex: viewport_lib::gpu::Texture,
    pub(super) lit_fallback_bg: viewport_lib::gpu::BindGroup,
    pub(super) outline_mask_pipeline: viewport_lib::gpu::RenderPipeline,
    /// The unlit colour shader again, stamping the scene stencil. It discards
    /// where the colour pass did, so a cut-out sprite stamps only what it
    /// drew.
    pub(super) surface_mask_pipeline: viewport_lib::gpu::RenderPipeline,
    pub(super) oit_pipeline: viewport_lib::gpu::RenderPipeline,
    pub(super) oit_pipeline_premultiplied: viewport_lib::gpu::RenderPipeline,
    pub(super) oit_lit_pipeline: viewport_lib::gpu::RenderPipeline,
    pub(super) oit_lit_pipeline_premultiplied: viewport_lib::gpu::RenderPipeline,
}

impl SpriteGpu {
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::SpriteLayouts,
    ) -> Self {
        let bgl = &layouts.bgl;

        // Group 2: scene depth + sampler for soft-particle fade. The shader
        // skips sampling unless soft_particle_distance > 0, so callers may bind
        // a placeholder when no resolved depth is available.
        // Group 2: scene depth + sampler for the soft-particle fade. The
        // shader skips the sample unless soft_particle_distance > 0, so a pass
        // that cannot expose live depth binds the 1x1 fallback below instead.
        // This is the lib's shared depth-read layout rather than one of our
        // own: the entries are identical, and matching it means the ready-made
        // `DepthReadContext::scene_depth_bind_group` binds here directly.
        let soft_bgl = resources.depth_read_bind_group_layout();

        let fallback_tex = device.create_texture(&viewport_lib::gpu::TextureDescriptor {
            label: Some("sprite_soft_fallback_tex"),
            size: viewport_lib::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: viewport_lib::gpu::TextureDimension::D2,
            format: viewport_lib::gpu::TextureFormat::Depth32Float,
            usage: viewport_lib::gpu::TextureUsages::TEXTURE_BINDING
                | viewport_lib::gpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let fallback_view = fallback_tex.create_view(&viewport_lib::gpu::TextureViewDescriptor {
            aspect: viewport_lib::gpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let fallback_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("sprite_soft_fallback_bg"),
            layout: soft_bgl,
            entries: &[
                viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: viewport_lib::gpu::BindingResource::TextureView(&fallback_view),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_lib::gpu::BindingResource::Sampler(
                        resources.depth_read_sampler(),
                    ),
                },
            ],
        });

        let shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "sprite_shader",
            &crate::shader::scene_shader(&[], crate::shader::wgsl_source!("sprite")),
        );

        let layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_pipeline_layout",
            &[resources.shared_bindings().group0_layout, bgl, soft_bgl],
        );

        // Position vertex buffer: one vec3 per sprite, Instance stepping.
        // Stored in an array so both pipeline creations can borrow from it.
        let vert_attrs = [viewport_lib::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: viewport_lib::gpu::VertexFormat::Float32x3,
        }];
        let vertex_buffers = [viewport_lib::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: viewport_lib::gpu::VertexStepMode::Instance,
            attributes: &vert_attrs,
        }];

        let sample_count = resources.sample_count();
        let ldr_format = resources.target_format();
        // Sprites are billboards drawn with `Less` depth test, no culling. Each
        // variant differs only in blend mode and whether it writes depth.
        let make_sprite = |depth_write: bool, blend: viewport_lib::gpu::BlendState, label: &str| {
            viewport_lib::plugin_api::builders::build_dual_pipeline(
                device,
                &DualPipelineDesc {
                    label,
                    layout: &layout,
                    shader: &shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &vertex_buffers,
                    blend: Some(blend),
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write,
                    depth_compare: viewport_lib::gpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        let surface_mask_pipeline = viewport_lib::plugin_api::builders::build_surface_mask_pipeline(
            device,
            "sprite_surface_mask_pipeline",
            &layout,
            &shader,
            &vertex_buffers,
            None,
        );

        let lit_bgl = &layouts.lit_bgl;

        let alpha = viewport_lib::gpu::BlendState::ALPHA_BLENDING;
        let additive = viewport_lib::plugin_api::builders::ADDITIVE_BLEND;
        let premultiplied = viewport_lib::plugin_api::builders::PREMULTIPLIED_BLEND;

        // -----------------------------------------------------------------
        // Refractive sprite pipeline.
        //
        // Group 0: shared camera bindings.
        // Group 1: shared sprite BGL (uniform / texture / sampler / instance buf).
        // Group 2: scene-colour resolve texture + sampler.
        //
        // Available only on the HDR path. The LDR `paint_to` route has no
        // resolvable scene-colour texture to sample, mirroring the
        // soft-particle constraint.
        let refraction_bgl = viewport_lib::plugin_api::builders::texture_sampler_bgl(
            device,
            "sprite_refraction_bgl",
            viewport_lib::gpu::ShaderStages::FRAGMENT,
        );

        let refraction_sampler = viewport_lib::plugin_api::builders::clamp_linear_sampler(
            device,
            "sprite_refraction_sampler",
        );

        let refraction_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "sprite_refraction_shader",
            &crate::shader::scene_shader(&[], crate::shader::wgsl_source!("sprite_refraction")),
        );

        let bgl_ref = bgl;
        let refraction_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_refraction_pipeline_layout",
            &[
                resources.shared_bindings().group0_layout,
                bgl_ref,
                &refraction_bgl,
            ],
        );

        let refraction_pipeline = viewport_lib::plugin_api::builders::render_pipeline(
            device,
            viewport_lib::plugin_api::builders::RenderPipelineDesc {
                label: "sprite_refraction_pipeline",
                layout: &refraction_layout,
                vertex_module: &refraction_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &vertex_buffers,
                fragment: Some(viewport_lib::gpu::FragmentState {
                    module: &refraction_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(viewport_lib::gpu::ColorTargetState {
                        format: viewport_lib::gpu::TextureFormat::Rgba16Float,
                        blend: Some(viewport_lib::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: viewport_lib::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: viewport_lib::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(viewport_lib::plugin_api::builders::scene_depth_stencil(
                    false,
                    viewport_lib::gpu::CompareFunction::Less,
                )),
                multisample: viewport_lib::gpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        );

        // -----------------------------------------------------------------
        // Lit sprite pipelines.
        //
        // Group 0: shared camera + clip + lighting bindings (already provides
        //          the lights uniform at binding 3 and the lights storage at
        //          binding 13 via `camera_bind_group_layout`).
        // Group 1: shared sprite BGL (uniform / texture / sampler / instance buf).
        // Group 2: shared soft-particle BGL (depth + sampler). Lit sprites
        //          honour the same per-instance soft-fade distance as the
        //          emissive path.
        // Group 3: new lit BGL (optional normal map + sampler).
        let lit_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "sprite_lit_shader",
            &crate::shader::lit_shader(
                &[
                    shared_wgsl::SHARED_CLIP_VOLUME_WGSL,
                    shared_wgsl::SHARED_CSM_WGSL,
                ],
                crate::shader::wgsl_source!("sprite_lit"),
            ),
        );

        let sprite_bgl_ref = bgl;
        let soft_bgl_ref = soft_bgl;
        let lit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_lit_pipeline_layout",
            &[
                resources.shared_bindings().group0_layout,
                sprite_bgl_ref,
                soft_bgl_ref,
                lit_bgl,
            ],
        );

        let make_lit = |depth_write: bool, blend: viewport_lib::gpu::BlendState, label: &str| {
            viewport_lib::plugin_api::builders::build_dual_pipeline(
                device,
                &DualPipelineDesc {
                    label,
                    layout: &lit_layout,
                    shader: &lit_shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &vertex_buffers,
                    blend: Some(blend),
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write,
                    depth_compare: viewport_lib::gpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        // One PipelineVariantSet-style build covers all 12 (depth_write x
        // blend x lit) combinations: the same closure picks the unlit or lit
        // shader/layout pair and the blend state for every key up front.
        let pipelines = SpriteVariantSet::build(|key| {
            let blend = match key.blend {
                viewport_lib::renderer::SpriteBlend::AlphaBlend => alpha,
                viewport_lib::renderer::SpriteBlend::Additive => additive,
                viewport_lib::renderer::SpriteBlend::Premultiplied => premultiplied,
            };
            if key.lit {
                make_lit(key.depth_write, blend, "sprite_lit_pipeline_variant")
            } else {
                make_sprite(key.depth_write, blend, "sprite_pipeline_variant")
            }
        });

        // The fallback bind group reuses the crate-wide `fallback_normal_map`,
        // already populated with `(128, 128, 255, 255)` for tangent-space `(0, 0, 1)`.
        let lit_fallback_bg = device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("sprite_lit_fallback_bg"),
            layout: lit_bgl,
            entries: &[
                viewport_lib::gpu::BindGroupEntry {
                    binding: 0,
                    resource: viewport_lib::gpu::BindingResource::TextureView(
                        resources.fallback_texture_view(viewport_lib::TextureSlot::Normal),
                    ),
                },
                viewport_lib::gpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_lib::gpu::BindingResource::Sampler(
                        resources.material_sampler(),
                    ),
                },
            ],
        });

        // -----------------------------------------------------------------
        // OIT (weighted-blended, order-independent transparency) sprite
        // pipelines. HDR-only: the OIT accum/reveal targets do not exist on
        // the LDR path. Only ever selected for `AlphaBlend`/`Premultiplied`
        // sprites with `depth_write: false` and no active soft-particle
        // fade (see `SpriteGpuData::oit_eligible`); `Additive` sprites and
        // any sprite excluded by that check keep drawing through the
        // ordinary pipelines above.
        //
        // "Straight alpha" and "premultiplied" are not separate pipelines --
        // the GPU blend state for the accum/reveal targets is identical
        // either way (see `viewport_lib::plugin_api::target_desc::OIT_ACCUM_BLEND`/
        // `OIT_REVEAL_BLEND`); the only difference is whether the fragment
        // shader multiplies by alpha before weighting. Each OIT shader
        // exposes `fs_oit`/`fs_oit_premultiplied` from the same module, so
        // one pipeline layout builds both pipelines.
        let oit_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "sprite_oit_shader",
            &crate::shader::scene_shader(&[], crate::shader::wgsl_source!("sprite_oit")),
        );
        let oit_lit_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "sprite_lit_oit_shader",
            &crate::shader::lit_shader(
                &[
                    shared_wgsl::SHARED_CLIP_VOLUME_WGSL,
                    shared_wgsl::SHARED_CSM_WGSL,
                ],
                crate::shader::wgsl_source!("sprite_lit_oit"),
            ),
        );

        let sprite_bgl_for_oit = bgl;
        let oit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_oit_pipeline_layout",
            &[
                resources.shared_bindings().group0_layout,
                sprite_bgl_for_oit,
            ],
        );
        let lit_bgl_for_oit = lit_bgl;
        let oit_lit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_lit_oit_pipeline_layout",
            &[
                resources.shared_bindings().group0_layout,
                sprite_bgl_for_oit,
                lit_bgl_for_oit,
            ],
        );

        let make_oit_pipeline = |layout: &viewport_lib::gpu::PipelineLayout,
                                 shader: &viewport_lib::gpu::ShaderModule,
                                 entry: &str,
                                 label: &str| {
            viewport_lib::plugin_api::builders::render_pipeline(
                device,
                viewport_lib::plugin_api::builders::RenderPipelineDesc {
                    label,
                    layout,
                    vertex_module: shader,
                    vertex_entry: "vs_main",
                    vertex_buffers: &vertex_buffers,
                    fragment: Some(viewport_lib::gpu::FragmentState {
                        module: shader,
                        entry_point: Some(entry),
                        targets: &[
                            Some(viewport_lib::gpu::ColorTargetState {
                                format: viewport_lib::gpu::TextureFormat::Rgba16Float,
                                blend: Some(viewport_lib::plugin_api::target_desc::OIT_ACCUM_BLEND),
                                write_mask: viewport_lib::gpu::ColorWrites::ALL,
                            }),
                            Some(viewport_lib::gpu::ColorTargetState {
                                format: viewport_lib::gpu::TextureFormat::R8Unorm,
                                blend: Some(
                                    viewport_lib::plugin_api::target_desc::OIT_REVEAL_BLEND,
                                ),
                                write_mask: viewport_lib::gpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: viewport_lib::gpu::PipelineCompilationOptions::default(
                        ),
                    }),
                    primitive: viewport_lib::gpu::PrimitiveState {
                        topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(viewport_lib::plugin_api::builders::scene_depth_stencil(
                        false,
                        viewport_lib::gpu::CompareFunction::LessEqual,
                    )),
                    multisample: viewport_lib::gpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };

        let oit_pipeline =
            make_oit_pipeline(&oit_layout, &oit_shader, "fs_oit", "sprite_oit_pipeline");
        let oit_pipeline_premultiplied = make_oit_pipeline(
            &oit_layout,
            &oit_shader,
            "fs_oit_premultiplied",
            "sprite_oit_pipeline_premultiplied",
        );
        let oit_lit_pipeline = make_oit_pipeline(
            &oit_lit_layout,
            &oit_lit_shader,
            "fs_oit",
            "sprite_lit_oit_pipeline",
        );
        let oit_lit_pipeline_premultiplied = make_oit_pipeline(
            &oit_lit_layout,
            &oit_lit_shader,
            "fs_oit_premultiplied",
            "sprite_lit_oit_pipeline_premultiplied",
        );

        let mask_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "sprite_outline_mask_shader",
            &crate::shader::scene_shader(&[], crate::shader::wgsl_source!("sprite_outline_mask")),
        );

        let mask_layout = viewport_lib::plugin_api::builders::standard_scene_layout(
            device,
            "sprite_outline_mask_pipeline_layout",
            resources.shared_bindings().group0_layout,
            bgl,
        );

        let mask_vert_attrs = [viewport_lib::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: viewport_lib::gpu::VertexFormat::Float32x3,
        }];
        let mask_vertex_buffers = [viewport_lib::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: viewport_lib::gpu::VertexStepMode::Instance,
            attributes: &mask_vert_attrs,
        }];

        let outline_mask_pipeline = viewport_lib::plugin_api::builders::build_outline_mask_pipeline(
            device,
            "sprite_outline_mask_pipeline",
            &mask_layout,
            &mask_shader,
            viewport_lib::gpu::TextureFormat::R8Unorm,
            &mask_vertex_buffers,
            None,
            false,
            viewport_lib::gpu::CompareFunction::Less,
        );

        let pick_id_bgl =
            device.create_bind_group_layout(&viewport_lib::gpu::BindGroupLayoutDescriptor {
                label: Some("sprite_pick_id_bgl"),
                entries: &[viewport_lib::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: viewport_lib::gpu::ShaderStages::FRAGMENT,
                    ty: viewport_lib::gpu::BindingType::Buffer {
                        ty: viewport_lib::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pick_shader = viewport_lib::plugin_api::builders::wgsl_module(
            device,
            "sprite_pick_shader",
            &crate::shader::scene_shader(&[], crate::shader::wgsl_source!("sprite_pick")),
        );
        let pick_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_pick_pipeline_layout",
            &[resources.shared_bindings().group0_layout, bgl, &pick_id_bgl],
        );

        // Position vertex buffer: one vec3 per sprite, instance-stepped, exactly
        // as the sprite render pipeline binds it.
        let pick_vert_attrs = [viewport_lib::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: viewport_lib::gpu::VertexFormat::Float32x3,
        }];
        let pick_vertex_buffers = [viewport_lib::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: viewport_lib::gpu::VertexStepMode::Instance,
            attributes: &pick_vert_attrs,
        }];

        let pick_pipeline = viewport_lib::plugin_api::builders::render_pipeline(
            device,
            viewport_lib::plugin_api::builders::RenderPipelineDesc {
                label: "sprite_pick_pipeline",
                layout: &pick_layout,
                vertex_module: &pick_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &pick_vertex_buffers,
                fragment: Some(viewport_lib::gpu::FragmentState {
                    module: &pick_shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(viewport_lib::gpu::ColorTargetState {
                            format: viewport_lib::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: viewport_lib::gpu::ColorWrites::ALL,
                        }),
                        Some(viewport_lib::gpu::ColorTargetState {
                            format: viewport_lib::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: viewport_lib::gpu::ColorWrites::ALL,
                        }),
                        Some(viewport_lib::gpu::ColorTargetState {
                            format: viewport_lib::gpu::TextureFormat::R32Float,
                            blend: None,
                            write_mask: viewport_lib::gpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: viewport_lib::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: viewport_lib::gpu::PrimitiveState {
                    topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(viewport_lib::plugin_api::builders::scene_depth_stencil(
                    true,
                    viewport_lib::gpu::CompareFunction::Less,
                )),
                multisample: viewport_lib::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        Self {
            pipelines,
            refraction_pipeline,
            refraction_bgl,
            refraction_sampler,
            pick_pipeline,
            pick_id_bgl,
            soft_fallback_bg: fallback_bg,
            soft_fallback_tex: fallback_tex,
            lit_fallback_bg,
            outline_mask_pipeline,
            surface_mask_pipeline,
            oit_pipeline,
            oit_pipeline_premultiplied,
            oit_lit_pipeline,
            oit_lit_pipeline_premultiplied,
        }
    }
}

impl SpriteGpu {
    /// Group-2 bind group carrying one batch's object pick id, for the pick pass.
    pub(super) fn pick_bind_group(
        &self,
        device: &viewport_lib::gpu::Device,
        queue: &viewport_lib::gpu::Queue,
        pick_id: viewport_lib::renderer::PickId,
    ) -> viewport_lib::gpu::BindGroup {
        // The id plus padding to the uniform's 16-byte size, matching the
        // layout the pick shader declares.
        let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
        let buf = device.create_buffer(&viewport_lib::gpu::BufferDescriptor {
            label: Some("sprite_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: viewport_lib::gpu::BufferUsages::UNIFORM
                | viewport_lib::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&viewport_lib::gpu::BindGroupDescriptor {
            label: Some("sprite_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[viewport_lib::gpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The variant set is built by iterating `SpriteKey::all()` and read by
    /// `SpriteKey::slot()`. Those are two separate expressions of the same key
    /// and they do not enumerate in the same order, so building into a vector
    /// in iteration order files every variant under the wrong key.
    ///
    /// The visible symptom was a validation failure rather than a wrong-looking
    /// sprite: an unlit Additive batch resolved to an AlphaBlend *lit*
    /// pipeline, and the unlit draw path does not bind the group-3 normal map
    /// that pipeline's layout requires.
    #[test]
    fn every_key_resolves_to_the_variant_built_for_it() {
        // Build with the identity, so each slot holds the key it was built for.
        let placed = place_by_slot(|key| key);
        for key in SpriteKey::all() {
            assert_eq!(
                placed[key.slot()],
                key,
                "slot {} holds the variant built for a different key",
                key.slot()
            );
        }
    }

    /// `slot()` has to be a bijection onto `0..12`, or two keys collide and a
    /// third slot is never written.
    #[test]
    fn slots_are_dense_and_unique() {
        let mut seen = [false; 12];
        let mut count = 0;
        for key in SpriteKey::all() {
            let slot = key.slot();
            assert!(!seen[slot], "slot {slot} claimed twice");
            seen[slot] = true;
            count += 1;
        }
        assert_eq!(count, 12, "all() must yield exactly 12 keys");
        assert!(seen.iter().all(|s| *s), "every slot must be claimed");
    }
}
