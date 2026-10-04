//! GPU state for the sprite item type: the twelve blend/depth/lit render
//! pipeline variants, the refractive pipeline and its scene-colour layout, the
//! soft-particle depth layout and its fallback, the weighted-blended OIT
//! pipelines, the pick pipeline, and the selection-outline mask pipeline. Each
//! pipeline is built the first time a draw needs it.
//!
//! The group-1 and group-3 bind group layouts live in `store`, beside the
//! upload that builds bind groups against them; this module borrows them to
//! build pipelines over them.

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

    /// Every axis combination.
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

    /// Dense index in `0..12`, stable across calls.
    fn slot(self) -> usize {
        self.depth_write as usize + 2 * self.blend_index() + 6 * (self.lit as usize)
    }

    /// The key at `slot`; the inverse of [`slot`](Self::slot).
    fn from_slot(slot: usize) -> Self {
        Self {
            depth_write: slot & 1 != 0,
            blend: match (slot / 2) % 3 {
                0 => viewport_lib::renderer::SpriteBlend::AlphaBlend,
                1 => viewport_lib::renderer::SpriteBlend::Additive,
                _ => viewport_lib::renderer::SpriteBlend::Premultiplied,
            },
            lit: slot >= 6,
        }
    }
}

/// The per-sprite position, one vec3 per sprite with instance stepping, as
/// every sprite pipeline binds it.
const POSITION_ATTRS: [viewport_lib::gpu::VertexAttribute; 1] =
    [viewport_lib::gpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: viewport_lib::gpu::VertexFormat::Float32x3,
    }];

fn position_layout() -> viewport_lib::gpu::VertexBufferLayout<'static> {
    viewport_lib::gpu::VertexBufferLayout {
        array_stride: 12,
        step_mode: viewport_lib::gpu::VertexStepMode::Instance,
        attributes: &POSITION_ATTRS,
    }
}

/// What a sprite colour pipeline build reads: the unlit and lit shader and
/// layout pairs, and the LDR target.
pub(super) struct SpriteRecipe {
    device: viewport_lib::gpu::Device,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    lit_layout: viewport_lib::gpu::PipelineLayout,
    lit_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
    ldr_format: viewport_lib::gpu::TextureFormat,
}

/// The 24 sprite colour pipelines: each [`SpriteKey`] in the LDR and HDR
/// formats, built the first time a draw selects it.
pub(super) type SpritePipelines = viewport_lib::plugin_api::LazyPipelines<SpriteRecipe, 24>;

/// Member index of `key` in the `hdr` or LDR format.
pub(super) fn sprite_index(key: SpriteKey, hdr: bool) -> usize {
    key.slot() * 2 + hdr as usize
}

fn build_sprite(r: &SpriteRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let key = SpriteKey::from_slot(i / 2);
    let hdr = i % 2 == 1;
    let vertex_buffers = [position_layout()];
    let blend = match key.blend {
        viewport_lib::renderer::SpriteBlend::AlphaBlend => {
            viewport_lib::gpu::BlendState::ALPHA_BLENDING
        }
        viewport_lib::renderer::SpriteBlend::Additive => {
            viewport_lib::plugin_api::builders::ADDITIVE_BLEND
        }
        viewport_lib::renderer::SpriteBlend::Premultiplied => {
            viewport_lib::plugin_api::builders::PREMULTIPLIED_BLEND
        }
    };
    let (layout, shader, label) = if key.lit {
        (
            &r.lit_layout,
            r.lit_shader.get(),
            "sprite_lit_pipeline_variant",
        )
    } else {
        (&r.layout, r.shader.get(), "sprite_pipeline_variant")
    };
    viewport_lib::plugin_api::builders::build_dual_pipeline_variant(
        &r.device,
        &DualPipelineDesc {
            label,
            layout,
            shader,
            vertex_entry: "vs_main",
            fragment_entry: "fs_main",
            vertex_buffers: &vertex_buffers,
            blend: Some(blend),
            topology: viewport_lib::gpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            depth_write: key.depth_write,
            depth_compare: viewport_lib::gpu::CompareFunction::Less,
            sample_count: r.sample_count,
            ldr_format: r.ldr_format,
        },
        hdr,
    )
}

/// Members of [`SpritePassPipelines`].
pub(super) const REFRACTION: usize = 0;
pub(super) const PICK: usize = 1;
pub(super) const OUTLINE_MASK: usize = 2;
pub(super) const SURFACE_MASK: usize = 3;
pub(super) const OIT: usize = 4;
pub(super) const OIT_PREMULTIPLIED: usize = 5;
pub(super) const OIT_LIT: usize = 6;
pub(super) const OIT_LIT_PREMULTIPLIED: usize = 7;

/// The OIT member a sprite with these fields draws through.
pub(super) fn oit_index(lit: bool, blend: viewport_lib::renderer::SpriteBlend) -> usize {
    let premultiplied = blend == viewport_lib::renderer::SpriteBlend::Premultiplied;
    match (lit, premultiplied) {
        (true, true) => OIT_LIT_PREMULTIPLIED,
        (true, false) => OIT_LIT,
        (false, true) => OIT_PREMULTIPLIED,
        (false, false) => OIT,
    }
}

/// What a build of the pipelines outside the colour variants reads.
pub(super) struct SpritePassRecipe {
    device: viewport_lib::gpu::Device,
    layout: viewport_lib::gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
    refraction_layout: viewport_lib::gpu::PipelineLayout,
    refraction_shader: viewport_lib::plugin_api::LazyModule,
    oit_layout: viewport_lib::gpu::PipelineLayout,
    oit_shader: viewport_lib::plugin_api::LazyModule,
    oit_lit_layout: viewport_lib::gpu::PipelineLayout,
    oit_lit_shader: viewport_lib::plugin_api::LazyModule,
    mask_layout: viewport_lib::gpu::PipelineLayout,
    mask_shader: viewport_lib::plugin_api::LazyModule,
    pick_layout: viewport_lib::gpu::PipelineLayout,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    sample_count: u32,
}

/// The refractive, pick, outline mask, surface mask and four OIT pipelines,
/// each built the first time a draw needs it.
pub(super) type SpritePassPipelines = viewport_lib::plugin_api::LazyPipelines<SpritePassRecipe, 8>;

fn build_pass(r: &SpritePassRecipe, i: usize) -> viewport_lib::gpu::RenderPipeline {
    let vertex_buffers = [position_layout()];
    match i {
        // Group 2 is the scene-colour copy. HDR only: the LDR path has no
        // scene colour to copy.
        REFRACTION => viewport_lib::plugin_api::builders::render_pipeline(
            &r.device,
            viewport_lib::plugin_api::builders::RenderPipelineDesc {
                label: "sprite_refraction_pipeline",
                layout: &r.refraction_layout,
                vertex_module: r.refraction_shader.get(),
                vertex_entry: "vs_main",
                vertex_buffers: &vertex_buffers,
                fragment: Some(viewport_lib::gpu::FragmentState {
                    module: r.refraction_shader.get(),
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
                    count: r.sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        ),
        PICK => viewport_lib::plugin_api::builders::render_pipeline(
            &r.device,
            viewport_lib::plugin_api::builders::RenderPipelineDesc {
                label: "sprite_pick_pipeline",
                layout: &r.pick_layout,
                vertex_module: r.pick_shader.get(),
                vertex_entry: "vs_main",
                vertex_buffers: &vertex_buffers,
                fragment: Some(viewport_lib::gpu::FragmentState {
                    module: r.pick_shader.get(),
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
        ),
        OUTLINE_MASK => viewport_lib::plugin_api::builders::build_outline_mask_pipeline(
            &r.device,
            "sprite_outline_mask_pipeline",
            &r.mask_layout,
            r.mask_shader.get(),
            viewport_lib::gpu::TextureFormat::R8Unorm,
            &vertex_buffers,
            None,
            false,
            viewport_lib::gpu::CompareFunction::Less,
        ),
        // The unlit colour shader again, stamping the scene stencil. It
        // discards where the colour pass did, so a cut-out sprite stamps only
        // what it drew.
        SURFACE_MASK => viewport_lib::plugin_api::builders::build_surface_mask_pipeline(
            &r.device,
            "sprite_surface_mask_pipeline",
            &r.layout,
            r.shader.get(),
            &vertex_buffers,
            None,
        ),
        // Weighted-blended OIT, HDR only: the accum/reveal targets do not
        // exist on the LDR path. Straight and premultiplied alpha share the
        // blend state and differ only in the fragment entry point.
        _ => {
            let (layout, shader, entry, label) = match i {
                OIT => (
                    &r.oit_layout,
                    r.oit_shader.get(),
                    "fs_oit",
                    "sprite_oit_pipeline",
                ),
                OIT_PREMULTIPLIED => (
                    &r.oit_layout,
                    r.oit_shader.get(),
                    "fs_oit_premultiplied",
                    "sprite_oit_pipeline_premultiplied",
                ),
                OIT_LIT => (
                    &r.oit_lit_layout,
                    r.oit_lit_shader.get(),
                    "fs_oit",
                    "sprite_lit_oit_pipeline",
                ),
                _ => (
                    &r.oit_lit_layout,
                    r.oit_lit_shader.get(),
                    "fs_oit_premultiplied",
                    "sprite_lit_oit_pipeline_premultiplied",
                ),
            };
            viewport_lib::plugin_api::builders::render_pipeline(
                &r.device,
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
                        count: r.sample_count,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        }
    }
}

/// Every pipeline and layout the sprite draw hooks need, made on the first
/// prepare that sees an item. The pipelines themselves are built the first
/// time a draw needs them.
pub(super) struct SpriteGpu {
    pub(super) pipelines: SpritePipelines,
    pub(super) passes: SpritePassPipelines,
    pub(super) refraction_bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) refraction_sampler: viewport_lib::gpu::Sampler,
    pub(super) pick_id_bgl: viewport_lib::gpu::BindGroupLayout,
    pub(super) soft_fallback_bg: viewport_lib::gpu::BindGroup,
    /// 1x1 Depth32Float texture backing the soft fallback bind group. Held so
    /// the bind group keeps a valid texture; not read after construction.
    #[allow(dead_code)]
    pub(super) soft_fallback_tex: viewport_lib::gpu::Texture,
    pub(super) lit_fallback_bg: viewport_lib::gpu::BindGroup,
}

impl SpriteGpu {
    pub(super) fn new(
        device: &viewport_lib::gpu::Device,
        resources: &DeviceResources,
        layouts: &super::store::SpriteLayouts,
    ) -> Self {
        let bgl = &layouts.bgl;
        let lit_bgl = &layouts.lit_bgl;
        let group0 = resources.shared_bindings().group0_layout;

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

        let shader = resources.lazy_module(
            device,
            "sprite_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("sprite"),
            ),
        );
        let layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_pipeline_layout",
            &[group0, bgl, soft_bgl],
        );

        // Refractive sprites: group 2 is the scene-colour copy and its sampler.
        let refraction_bgl = viewport_lib::plugin_api::builders::texture_sampler_bgl(
            device,
            "sprite_refraction_bgl",
            viewport_lib::gpu::ShaderStages::FRAGMENT,
        );
        let refraction_sampler = viewport_lib::plugin_api::builders::clamp_linear_sampler(
            device,
            "sprite_refraction_sampler",
        );
        let refraction_shader = resources.lazy_module(
            device,
            "sprite_refraction_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("sprite_refraction"),
            ),
        );
        let refraction_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_refraction_pipeline_layout",
            &[group0, bgl, &refraction_bgl],
        );

        // Lit sprites. Group 0 already carries the lights; group 2 is the
        // soft-particle depth, as for the unlit path, and group 3 the optional
        // normal map.
        let lit_shader = resources.lazy_module(
            device,
            "sprite_lit_shader",
            &crate::item_types::shader::lit_shader(
                &[
                    shared_wgsl::SHARED_CLIP_VOLUME_WGSL,
                    shared_wgsl::SHARED_CSM_WGSL,
                ],
                crate::item_types::shader::wgsl_source!("sprite_lit"),
            ),
        );
        let lit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_lit_pipeline_layout",
            &[group0, bgl, soft_bgl, lit_bgl],
        );

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

        // OIT. Only ever selected for `AlphaBlend`/`Premultiplied` sprites
        // with `depth_write: false` and no active soft-particle fade (see
        // `SpriteGpuData::oit_eligible`); everything else keeps drawing through
        // the colour variants. Each OIT shader exposes `fs_oit` and
        // `fs_oit_premultiplied` from one module.
        let oit_shader = resources.lazy_module(
            device,
            "sprite_oit_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("sprite_oit"),
            ),
        );
        let oit_lit_shader = resources.lazy_module(
            device,
            "sprite_lit_oit_shader",
            &crate::item_types::shader::lit_shader(
                &[
                    shared_wgsl::SHARED_CLIP_VOLUME_WGSL,
                    shared_wgsl::SHARED_CSM_WGSL,
                ],
                crate::item_types::shader::wgsl_source!("sprite_lit_oit"),
            ),
        );
        let oit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_oit_pipeline_layout",
            &[group0, bgl],
        );
        let oit_lit_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_lit_oit_pipeline_layout",
            &[group0, bgl, lit_bgl],
        );

        let mask_shader = resources.lazy_module(
            device,
            "sprite_outline_mask_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("sprite_outline_mask"),
            ),
        );
        let mask_layout = viewport_lib::plugin_api::builders::standard_scene_layout(
            device,
            "sprite_outline_mask_pipeline_layout",
            group0,
            bgl,
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
        let pick_shader = resources.lazy_module(
            device,
            "sprite_pick_shader",
            &crate::item_types::shader::scene_shader(
                &[],
                crate::item_types::shader::wgsl_source!("sprite_pick"),
            ),
        );
        let pick_layout = viewport_lib::plugin_api::builders::pipeline_layout(
            device,
            "sprite_pick_pipeline_layout",
            &[group0, bgl, &pick_id_bgl],
        );

        let sample_count = resources.sample_count();

        // The 12 (depth_write x blend x lit) keys in both formats. Sprites are
        // billboards drawn with a `Less` depth test and no culling; each is
        // built the first time a draw selects it.
        let pipelines: SpritePipelines = resources.lazy_pipelines(
            SpriteRecipe {
                device: device.clone(),
                layout: layout.clone(),
                shader: shader.clone(),
                lit_layout,
                lit_shader,
                sample_count,
                ldr_format: resources.target_format(),
            },
            build_sprite,
        );
        let passes: SpritePassPipelines = resources.lazy_pipelines(
            SpritePassRecipe {
                device: device.clone(),
                layout,
                shader,
                refraction_layout,
                refraction_shader,
                oit_layout,
                oit_shader,
                oit_lit_layout,
                oit_lit_shader,
                mask_layout,
                mask_shader,
                pick_layout,
                pick_shader,
                sample_count,
            },
            build_pass,
        );

        Self {
            pipelines,
            passes,
            refraction_bgl,
            refraction_sampler,
            pick_id_bgl,
            soft_fallback_bg: fallback_bg,
            soft_fallback_tex: fallback_tex,
            lit_fallback_bg,
        }
    }

    /// Whether the pipeline `sprite` draws with can draw this frame, in
    /// either format. The outline, surface mask and pick passes wait for it,
    /// so a batch is never outlined, stamped or picked before it is drawn.
    pub(super) fn drawn(&self, sprite: &super::store::SpriteGpuData) -> bool {
        let key = SpriteKey {
            depth_write: sprite.depth_write,
            blend: sprite.blend,
            lit: sprite.lit,
        };
        let colour = self.pipelines.available(sprite_index(key, false))
            || self.pipelines.available(sprite_index(key, true));
        if sprite.refraction_strength > 0.0 {
            self.passes.available(REFRACTION)
        } else if sprite.oit_eligible {
            self.passes.available(oit_index(sprite.lit, sprite.blend)) || colour
        } else {
            colour
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

    /// A draw reads a variant by `SpriteKey::slot()` and the build decodes the
    /// key back with `SpriteKey::from_slot()`. The two have to agree, or every
    /// variant is built for the wrong key.
    ///
    /// The visible symptom was a validation failure rather than a wrong-looking
    /// sprite: an unlit Additive batch resolved to an AlphaBlend *lit*
    /// pipeline, and the unlit draw path does not bind the group-3 normal map
    /// that pipeline's layout requires.
    #[test]
    fn every_key_resolves_to_the_variant_built_for_it() {
        for key in SpriteKey::all() {
            assert_eq!(
                SpriteKey::from_slot(key.slot()),
                key,
                "slot {} builds the variant for a different key",
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
