//! GPU state for the sprite item type: the twelve blend/depth/lit render
//! pipeline variants, the refractive pipeline and its scene-colour layout, the
//! soft-particle depth layout and its fallback, the weighted-blended OIT
//! pipelines, the pick pipeline, and the selection-outline mask pipeline.
//!
//! The group-1 and group-3 bind group layouts stay in `resources`, because
//! `upload_sprite_set` is public API and builds its bind groups against them;
//! this module borrows them to build pipelines over them.

use crate::resources::{DeviceResources, DualPipeline};

/// Sprite pipeline variant axes: depth-write, blend mode, and unlit vs
/// `apply_scene_lighting`-lit shading. Kept separate from the mesh family's
/// `PipelineKey` -- sprite's axes don't map onto `two_sided` / `cutout` /
/// `no_discard_eligible`, and `blend` is three-valued, not boolean, so
/// reusing that type would just be confusing field names for an unrelated
/// set of axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SpriteKey {
    pub depth_write: bool,
    pub blend: crate::renderer::SpriteBlend,
    pub lit: bool,
}

impl SpriteKey {
    fn blend_index(self) -> usize {
        match self.blend {
            crate::renderer::SpriteBlend::AlphaBlend => 0,
            crate::renderer::SpriteBlend::Additive => 1,
            crate::renderer::SpriteBlend::Premultiplied => 2,
        }
    }

    /// Every axis combination, for eager cross-product construction
    /// (`SpriteVariantSet::build`).
    pub fn all() -> impl Iterator<Item = SpriteKey> {
        [
            crate::renderer::SpriteBlend::AlphaBlend,
            crate::renderer::SpriteBlend::Additive,
            crate::renderer::SpriteBlend::Premultiplied,
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

impl SpriteVariantSet {
    pub fn build(mut build: impl FnMut(SpriteKey) -> DualPipeline) -> Self {
        let mut variants: Vec<DualPipeline> = Vec::with_capacity(12);
        for key in SpriteKey::all() {
            variants.push(build(key));
        }
        Self {
            variants: variants
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
    pub(super) refraction_pipeline: crate::gpu::RenderPipeline,
    pub(super) refraction_bgl: crate::gpu::BindGroupLayout,
    pub(super) refraction_sampler: crate::gpu::Sampler,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_id_bgl: crate::gpu::BindGroupLayout,
    pub(super) soft_fallback_bg: crate::gpu::BindGroup,
    /// 1x1 Depth32Float texture backing the soft fallback bind group. Held so
    /// the bind group keeps a valid texture; not read after construction.
    #[allow(dead_code)]
    pub(super) soft_fallback_tex: crate::gpu::Texture,
    pub(super) lit_fallback_bg: crate::gpu::BindGroup,
    pub(super) outline_mask_pipeline: crate::gpu::RenderPipeline,
    pub(super) oit_pipeline: crate::gpu::RenderPipeline,
    pub(super) oit_pipeline_premultiplied: crate::gpu::RenderPipeline,
    pub(super) oit_lit_pipeline: crate::gpu::RenderPipeline,
    pub(super) oit_lit_pipeline_premultiplied: crate::gpu::RenderPipeline,
}

impl SpriteGpu {
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        let bgl = &resources.sprite.bgl;

        // Group 2: scene depth + sampler for soft-particle fade. The shader
        // skips sampling unless soft_particle_distance > 0, so callers may bind
        // a placeholder when no resolved depth is available.
        // Group 2: scene depth + sampler for the soft-particle fade. The
        // shader skips the sample unless soft_particle_distance > 0, so a pass
        // that cannot expose live depth binds the 1x1 fallback below instead.
        // This is the lib's shared depth-read layout rather than one of our
        // own: the entries are identical, and matching it means the ready-made
        // `DepthReadContext::scene_depth_bind_group` binds here directly.
        let soft_bgl = &resources.material.depth_read_bgl;

        let fallback_tex = device.create_texture(&crate::gpu::TextureDescriptor {
            label: Some("sprite_soft_fallback_tex"),
            size: crate::gpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: crate::gpu::TextureDimension::D2,
            format: crate::gpu::TextureFormat::Depth32Float,
            usage: crate::gpu::TextureUsages::TEXTURE_BINDING
                | crate::gpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let fallback_view = fallback_tex.create_view(&crate::gpu::TextureViewDescriptor {
            aspect: crate::gpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let fallback_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("sprite_soft_fallback_bg"),
            layout: soft_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&fallback_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(
                        &resources.material.depth_read_sampler,
                    ),
                },
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_shader",
            crate::resources::builders::wgsl_source!("sprite"),
        );

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_pipeline_layout",
            &[&resources.binds.camera_bgl, bgl, soft_bgl],
        );

        // Position vertex buffer: one vec3 per sprite, Instance stepping.
        // Stored in an array so both pipeline creations can borrow from it.
        let vert_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }];
        let vertex_buffers = [crate::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &vert_attrs,
        }];

        let sample_count = resources.sample_count;
        let ldr_format = resources.target_format;
        // Sprites are billboards drawn with `Less` depth test, no culling. Each
        // variant differs only in blend mode and whether it writes depth.
        let make_sprite = |depth_write: bool, blend: crate::gpu::BlendState, label: &str| {
            crate::resources::builders::build_dual_pipeline(
                device,
                &crate::resources::builders::DualPipelineDesc {
                    label,
                    layout: &layout,
                    shader: &shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &vertex_buffers,
                    blend: Some(blend),
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write,
                    depth_compare: crate::gpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        let lit_bgl = &resources.sprite.lit_bgl;

        let alpha = crate::gpu::BlendState::ALPHA_BLENDING;
        let additive = crate::resources::builders::ADDITIVE_BLEND;
        let premultiplied = crate::resources::builders::PREMULTIPLIED_BLEND;

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
        let refraction_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "sprite_refraction_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let refraction_sampler =
            crate::resources::builders::clamp_linear_sampler(device, "sprite_refraction_sampler");

        let refraction_shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_refraction_shader",
            crate::resources::builders::wgsl_source!("sprite_refraction"),
        );

        let bgl_ref = bgl;
        let refraction_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_refraction_pipeline_layout",
            &[&resources.binds.camera_bgl, bgl_ref, &refraction_bgl],
        );

        let refraction_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "sprite_refraction_pipeline",
                layout: &refraction_layout,
                vertex_module: &refraction_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &vertex_buffers,
                fragment: Some(crate::gpu::FragmentState {
                    module: &refraction_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::Rgba16Float,
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
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
        let lit_shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_lit_shader",
            crate::resources::builders::wgsl_source!("sprite_lit"),
        );

        let sprite_bgl_ref = bgl;
        let soft_bgl_ref = soft_bgl;
        let lit_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_lit_pipeline_layout",
            &[
                &resources.binds.camera_bgl,
                sprite_bgl_ref,
                soft_bgl_ref,
                lit_bgl,
            ],
        );

        let make_lit = |depth_write: bool, blend: crate::gpu::BlendState, label: &str| {
            crate::resources::builders::build_dual_pipeline(
                device,
                &crate::resources::builders::DualPipelineDesc {
                    label,
                    layout: &lit_layout,
                    shader: &lit_shader,
                    vertex_entry: "vs_main",
                    fragment_entry: "fs_main",
                    vertex_buffers: &vertex_buffers,
                    blend: Some(blend),
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write,
                    depth_compare: crate::gpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        // One PipelineVariantSet-style build covers all 12 (depth_write x
        // blend x lit) combinations: the same closure picks the unlit or lit
        // shader/layout pair and the blend state for every key up front.
        let pipelines = (SpriteVariantSet::build(|key| {
            let blend = match key.blend {
                crate::renderer::SpriteBlend::AlphaBlend => alpha,
                crate::renderer::SpriteBlend::Additive => additive,
                crate::renderer::SpriteBlend::Premultiplied => premultiplied,
            };
            if key.lit {
                make_lit(key.depth_write, blend, "sprite_lit_pipeline_variant")
            } else {
                make_sprite(key.depth_write, blend, "sprite_pipeline_variant")
            }
        }));

        // The fallback bind group reuses the crate-wide `fallback_normal_map`,
        // already populated with `(128, 128, 255, 255)` for tangent-space `(0, 0, 1)`.
        let lit_fallback_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("sprite_lit_fallback_bg"),
            layout: lit_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(
                        &resources.material.normal_map_view,
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(&resources.material.sampler),
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
        // either way (see `crate::plugin_api::target_desc::OIT_ACCUM_BLEND`/
        // `OIT_REVEAL_BLEND`); the only difference is whether the fragment
        // shader multiplies by alpha before weighting. Each OIT shader
        // exposes `fs_oit`/`fs_oit_premultiplied` from the same module, so
        // one pipeline layout builds both pipelines.
        let oit_shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_oit_shader",
            crate::resources::builders::wgsl_source!("sprite_oit"),
        );
        let oit_lit_shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_lit_oit_shader",
            crate::resources::builders::wgsl_source!("sprite_lit_oit"),
        );

        let sprite_bgl_for_oit = bgl;
        let oit_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_oit_pipeline_layout",
            &[&resources.binds.camera_bgl, sprite_bgl_for_oit],
        );
        let lit_bgl_for_oit = lit_bgl;
        let oit_lit_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_lit_oit_pipeline_layout",
            &[
                &resources.binds.camera_bgl,
                sprite_bgl_for_oit,
                lit_bgl_for_oit,
            ],
        );

        let make_oit_pipeline = |layout: &crate::gpu::PipelineLayout,
                                 shader: &crate::gpu::ShaderModule,
                                 entry: &str,
                                 label: &str| {
            crate::resources::builders::render_pipeline(
                device,
                crate::resources::builders::RenderPipelineDesc {
                    label,
                    layout,
                    vertex_module: shader,
                    vertex_entry: "vs_main",
                    vertex_buffers: &vertex_buffers,
                    fragment: Some(crate::gpu::FragmentState {
                        module: shader,
                        entry_point: Some(entry),
                        targets: &[
                            Some(crate::gpu::ColorTargetState {
                                format: crate::gpu::TextureFormat::Rgba16Float,
                                blend: Some(crate::plugin_api::target_desc::OIT_ACCUM_BLEND),
                                write_mask: crate::gpu::ColorWrites::ALL,
                            }),
                            Some(crate::gpu::ColorTargetState {
                                format: crate::gpu::TextureFormat::R8Unorm,
                                blend: Some(crate::plugin_api::target_desc::OIT_REVEAL_BLEND),
                                write_mask: crate::gpu::ColorWrites::RED,
                            }),
                        ],
                        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: crate::gpu::PrimitiveState {
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                        false,
                        crate::gpu::CompareFunction::LessEqual,
                    )),
                    multisample: crate::gpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };

        let oit_pipeline =
            (make_oit_pipeline(&oit_layout, &oit_shader, "fs_oit", "sprite_oit_pipeline"));
        let oit_pipeline_premultiplied = (make_oit_pipeline(
            &oit_layout,
            &oit_shader,
            "fs_oit_premultiplied",
            "sprite_oit_pipeline_premultiplied",
        ));
        let oit_lit_pipeline = (make_oit_pipeline(
            &oit_lit_layout,
            &oit_lit_shader,
            "fs_oit",
            "sprite_lit_oit_pipeline",
        ));
        let oit_lit_pipeline_premultiplied = (make_oit_pipeline(
            &oit_lit_layout,
            &oit_lit_shader,
            "fs_oit_premultiplied",
            "sprite_lit_oit_pipeline_premultiplied",
        ));

        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_outline_mask_shader",
            crate::resources::builders::wgsl_source!("sprite_outline_mask"),
        );

        let mask_layout = crate::resources::builders::standard_scene_layout(
            device,
            "sprite_outline_mask_pipeline_layout",
            &resources.binds.camera_bgl,
            bgl,
        );

        let mask_vert_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }];
        let mask_vertex_buffers = [crate::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &mask_vert_attrs,
        }];

        let outline_mask_pipeline = crate::resources::builders::build_outline_mask_pipeline(
            device,
            "sprite_outline_mask_pipeline",
            &mask_layout,
            &mask_shader,
            crate::gpu::TextureFormat::R8Unorm,
            &mask_vertex_buffers,
            None,
            false,
            crate::gpu::CompareFunction::Less,
        );

        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("sprite_pick_id_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pick_shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_pick_shader",
            crate::resources::builders::wgsl_source!("sprite_pick"),
        );
        let pick_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_pick_pipeline_layout",
            &[&resources.binds.camera_bgl, bgl, &pick_id_bgl],
        );

        // Position vertex buffer: one vec3 per sprite, instance-stepped, exactly
        // as the sprite render pipeline binds it.
        let pick_vert_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }];
        let pick_vertex_buffers = [crate::gpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &pick_vert_attrs,
        }];

        let pick_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "sprite_pick_pipeline",
                layout: &pick_layout,
                vertex_module: &pick_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &pick_vertex_buffers,
                fragment: Some(crate::gpu::FragmentState {
                    module: &pick_shader,
                    entry_point: Some("fs_main"),
                    targets: &[
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                        Some(crate::gpu::ColorTargetState {
                            format: crate::gpu::TextureFormat::R32Float,
                            blend: None,
                            write_mask: crate::gpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    true,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
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
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        pick_id: crate::renderer::PickId,
    ) -> crate::gpu::BindGroup {
        // The id plus padding to the uniform's 16-byte size, matching the
        // layout the pick shader declares.
        let id_data = [pick_id.0 as u32, 0u32, 0u32, 0u32];
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("sprite_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&id_data));
        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("sprite_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        })
    }
}
