use super::*;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins `SpriteKey::all()` and `slot()` in sync, the same regression
    /// class `pipeline_key::tests::all_keys_are_distinct_and_densely_slotted`
    /// guards for the mesh family's `PipelineKey`: if a future axis widens
    /// past what `slot()` computes, two distinct keys would collide on the
    /// same array index and `build` would silently drop one of them.
    #[test]
    fn all_keys_are_distinct_and_densely_slotted() {
        let keys: Vec<SpriteKey> = SpriteKey::all().collect();
        assert_eq!(
            keys.len(),
            12,
            "SpriteKey has depth_write x blend(3) x lit = 12 keys"
        );

        let mut seen_keys = std::collections::HashSet::new();
        let mut seen_slots = std::collections::HashSet::new();
        for key in keys {
            assert!(
                seen_keys.insert(key),
                "all() yielded {key:?} more than once"
            );
            let slot = key.slot();
            assert!(slot < 12, "{key:?} slotted out of range: {slot}");
            assert!(
                seen_slots.insert(slot),
                "{key:?} collided with another key at slot {slot}"
            );
        }
    }

    /// Same completeness guarantee as the mesh-family `PipelineVariantSet`
    /// tests: once built, every key in `SpriteKey::all()` must resolve
    /// through `get()` without panicking.
    #[test]
    fn sprite_pipelines_resolve_every_key_once_built() {
        let Some((device, _queue, mut res)) = crate::resources::test_support::try_make_resources()
        else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        res.ensure_sprite_pipelines(&device);
        let pipelines = res
            .sprite
            .pipelines
            .as_ref()
            .expect("ensure_sprite_pipelines must build the sprite variant set");
        for key in SpriteKey::all() {
            let _ = pipelines.get(key);
        }
    }
}

/// Sprite billboard pipelines (emissive + lit, keyed by depth-write / blend /
/// lit), their bind group layouts, refraction pass, and soft-particle
/// fallbacks. All lazily built; the uploaded sprite sets live in separate
/// flat stores.
#[derive(Default)]
pub(crate) struct SpriteResources {
    /// Sprite render pipelines, keyed by `SpriteKey`.
    pub(crate) pipelines: Option<SpriteVariantSet>,
    /// Refractive sprite pipeline (HDR target only).
    pub(crate) refraction_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 BGL for the refraction pipeline: scene-colour texture + sampler.
    pub(crate) refraction_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Sampler used by the refraction shader to read the scene-colour resolve.
    pub(crate) refraction_sampler: Option<crate::gpu::Sampler>,
    /// Bind group layout for sprite uniforms + texture + instance buffer (group 1).
    pub(crate) bgl: Option<crate::gpu::BindGroupLayout>,
    /// GPU object-id pick pipeline. Reuses the sprite render vertex expansion and
    /// writes the item's pick_id. None until the first pick call with sprites.
    pub(crate) pick_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Group 2 layout for the per-draw pick_id uniform used by `pick_pipeline`.
    pub(crate) pick_id_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Bind group layout for the per-pass scene-depth resolve bound at group 2.
    pub(crate) soft_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Fallback bind group for the group-2 soft-particle binding.
    pub(crate) soft_fallback_bg: Option<crate::gpu::BindGroup>,
    /// Sampler used for the group-2 scene-depth binding.
    pub(crate) soft_sampler: Option<crate::gpu::Sampler>,
    /// 1x1 Depth32Float texture backing the soft fallback bind group.
    pub(crate) soft_fallback_tex: Option<crate::gpu::Texture>,
    /// Group 3 BGL for the optional lit normal map (texture + sampler).
    pub(crate) lit_bgl: Option<crate::gpu::BindGroupLayout>,
    /// Fallback bind group for the lit normal map binding.
    pub(crate) lit_fallback_bg: Option<crate::gpu::BindGroup>,
    /// 1x1 RGBA8Unorm texture backing the lit fallback bind group. Held so the
    /// fallback bind group keeps a valid texture; not read after construction.
    #[allow(dead_code)]
    pub(crate) lit_fallback_tex: Option<crate::gpu::Texture>,
    /// Sprite outline mask pipeline (R8Unorm). None until first selected sprite.
    pub(crate) outline_mask_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Weighted-blended OIT pipeline, unlit, straight alpha. HDR-only (see
    /// `docs/plans/non-mesh-pipeline-consistency-plan.md#phase-6b`).
    pub(crate) oit_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Weighted-blended OIT pipeline, unlit, premultiplied alpha.
    pub(crate) oit_pipeline_premultiplied: Option<crate::gpu::RenderPipeline>,
    /// Weighted-blended OIT pipeline, lit, straight alpha.
    pub(crate) oit_lit_pipeline: Option<crate::gpu::RenderPipeline>,
    /// Weighted-blended OIT pipeline, lit, premultiplied alpha.
    pub(crate) oit_lit_pipeline_premultiplied: Option<crate::gpu::RenderPipeline>,
}

impl DeviceResources {
    /// Lazily create the sprite billboard pipelines (alpha-blended, instanced quad expansion).
    ///
    /// Creates two pipelines that share the same shader and bind group layout but differ
    /// in `depth_write_enabled`: one for transparent effects (`depth_write: false`) and one
    /// for opaque-style placed sprites (`depth_write: true`).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.sprite_items` is
    /// non-empty.
    pub(crate) fn ensure_sprite_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.sprite.bgl.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("sprite_bgl"),
            entries: &[
                // binding 0: SpriteUniform (model, world_space, has_texture)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: sprite texture (or fallback 1x1 when has_texture == 0)
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Float { filterable: true },
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: sampler
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(crate::gpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: per-sprite instance storage buffer
                crate::gpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Group 2: scene depth + sampler for soft-particle fade. The shader
        // skips sampling unless soft_particle_distance > 0, so callers may bind
        // a placeholder when no resolved depth is available.
        let soft_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("sprite_soft_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Texture {
                        sample_type: crate::gpu::TextureSampleType::Depth,
                        view_dimension: crate::gpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Sampler(
                        crate::gpu::SamplerBindingType::NonFiltering,
                    ),
                    count: None,
                },
            ],
        });

        let soft_sampler =
            crate::resources::builders::clamp_nearest_sampler(device, "sprite_soft_sampler");

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
            layout: &soft_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(&fallback_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(&soft_sampler),
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
            &[&self.binds.camera_bgl, &bgl, &soft_bgl],
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

        let sample_count = self.sample_count;
        let ldr_format = self.target_format;
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

        // Group 3 BGL for the lit sprite path: optional tangent-space normal
        // map + filtering sampler. Bound by every lit batch; a 1x1 default
        // backs the binding when no map is supplied.
        let lit_bgl = crate::resources::builders::texture_sampler_bgl(
            device,
            "sprite_lit_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        let alpha = crate::gpu::BlendState::ALPHA_BLENDING;
        let additive = crate::resources::builders::ADDITIVE_BLEND;
        let premultiplied = crate::resources::builders::PREMULTIPLIED_BLEND;
        self.sprite.bgl = Some(bgl);
        self.sprite.soft_bgl = Some(soft_bgl);
        self.sprite.soft_sampler = Some(soft_sampler);
        self.sprite.soft_fallback_tex = Some(fallback_tex);
        self.sprite.soft_fallback_bg = Some(fallback_bg);

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

        let bgl_ref = self.sprite.bgl.as_ref().unwrap();
        let refraction_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_refraction_pipeline_layout",
            &[&self.binds.camera_bgl, bgl_ref, &refraction_bgl],
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

        self.sprite.refraction_bgl = Some(refraction_bgl);
        self.sprite.refraction_sampler = Some(refraction_sampler);
        self.sprite.refraction_pipeline = Some(refraction_pipeline);

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

        let sprite_bgl_ref = self.sprite.bgl.as_ref().unwrap();
        let soft_bgl_ref = self.sprite.soft_bgl.as_ref().unwrap();
        let lit_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_lit_pipeline_layout",
            &[
                &self.binds.camera_bgl,
                sprite_bgl_ref,
                soft_bgl_ref,
                &lit_bgl,
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
        self.sprite.pipelines = Some(SpriteVariantSet::build(|key| {
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
            layout: &lit_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: crate::gpu::BindingResource::TextureView(
                        &self.material.normal_map_view,
                    ),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::Sampler(&self.material.sampler),
                },
            ],
        });

        self.sprite.lit_bgl = Some(lit_bgl);
        self.sprite.lit_fallback_bg = Some(lit_fallback_bg);

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

        let sprite_bgl_for_oit = self.sprite.bgl.as_ref().unwrap();
        let oit_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_oit_pipeline_layout",
            &[&self.binds.camera_bgl, sprite_bgl_for_oit],
        );
        let lit_bgl_for_oit = self.sprite.lit_bgl.as_ref().unwrap();
        let oit_lit_layout = crate::resources::builders::pipeline_layout(
            device,
            "sprite_lit_oit_pipeline_layout",
            &[&self.binds.camera_bgl, sprite_bgl_for_oit, lit_bgl_for_oit],
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

        self.sprite.oit_pipeline = Some(make_oit_pipeline(
            &oit_layout,
            &oit_shader,
            "fs_oit",
            "sprite_oit_pipeline",
        ));
        self.sprite.oit_pipeline_premultiplied = Some(make_oit_pipeline(
            &oit_layout,
            &oit_shader,
            "fs_oit_premultiplied",
            "sprite_oit_pipeline_premultiplied",
        ));
        self.sprite.oit_lit_pipeline = Some(make_oit_pipeline(
            &oit_lit_layout,
            &oit_lit_shader,
            "fs_oit",
            "sprite_lit_oit_pipeline",
        ));
        self.sprite.oit_lit_pipeline_premultiplied = Some(make_oit_pipeline(
            &oit_lit_layout,
            &oit_lit_shader,
            "fs_oit_premultiplied",
            "sprite_lit_oit_pipeline_premultiplied",
        ));
    }

    /// Upload one [`SpriteItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.scene.sprite_items`.
    pub(crate) fn upload_sprite(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::SpriteItem,
    ) -> SpriteGpuData {
        use crate::resources::TextureSlot;
        self.check_texture_slot(item.texture_id, TextureSlot::SpriteAlbedo);
        self.check_texture_slot(item.normal_texture_id, TextureSlot::SpriteNormalMap);

        let count = item.positions.len() as u32;

        // Position vertex buffer (one vec3 per sprite, instance-stepped).
        let pos_bytes: Vec<u8> = item
            .positions
            .iter()
            .flat_map(|p| bytemuck::bytes_of(p).iter().copied())
            .collect();
        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("sprite_vertex_buf"),
            size: pos_bytes.len().max(12) as u64,
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &pos_bytes);

        // Per-instance storage buffer: build by zipping item vecs with defaults.
        // Layout matches `SpriteInstance` in `sprite.wgsl`. 64 bytes per instance.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuSpriteInstance {
            colour: [f32; 4],
            size: f32,
            rotation: f32,
            soft_distance: f32,
            _pad1: f32,
            uv_rect: [f32; 4],
            velocity: [f32; 3],
            _pad2: f32,
        }

        let instances: Vec<GpuSpriteInstance> = (0..item.positions.len())
            .map(|i| GpuSpriteInstance {
                colour: if i < item.colours.len() {
                    item.colours[i].to_linear_rgba()
                } else {
                    item.default_colour.to_linear_rgba()
                },
                size: if i < item.sizes.len() {
                    item.sizes[i]
                } else {
                    item.default_size
                },
                rotation: if i < item.rotations.len() {
                    item.rotations[i]
                } else {
                    0.0
                },
                soft_distance: if i < item.soft_particle_distances.len() {
                    item.soft_particle_distances[i].max(0.0)
                } else {
                    0.0
                },
                _pad1: 0.0,
                uv_rect: if i < item.uv_rects.len() {
                    item.uv_rects[i]
                } else {
                    [0.0, 0.0, 1.0, 1.0]
                },
                velocity: if i < item.velocities.len() {
                    item.velocities[i]
                } else {
                    [0.0, 0.0, 0.0]
                },
                _pad2: 0.0,
            })
            .collect();

        let instance_bytes = bytemuck::cast_slice(&instances);
        let instance_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("sprite_instance_buf"),
            size: instance_bytes.len().max(48) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, instance_bytes);

        // Uniform buffer: model matrix + flags + soft-particle distance + orientation
        // + refraction strength + lit parameters. Layout mirrors `SpriteUniform`
        // in `sprite_lit.wgsl`; the emissive `sprite.wgsl` reads only the first
        // half and ignores the trailing lit fields.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SpriteUniformData {
            model: [[f32; 4]; 4],
            world_space: u32,
            has_texture: u32,
            soft_particle_distance: f32,
            orientation: u32,
            axis: [f32; 3],
            refraction_strength: f32,
            lit: u32,
            normal_mode: u32,
            has_normal_map: u32,
            ambient_scale: f32,
            roughness: f32,
            receive_shadows: u32,
            _pad_lit_b: u32,
            _pad_lit_c: u32,
        }

        let (texture_view, has_texture): (&crate::gpu::TextureView, u32) =
            if let Some(id) = item.texture_id {
                if let Some(tex) = self.content.textures.get(id) {
                    (&tex.view, 1)
                } else {
                    (&self.content.fallback_lut_view, 0)
                }
            } else {
                (&self.content.fallback_lut_view, 0)
            };

        let orientation = match item.orientation {
            crate::renderer::SpriteOrientation::CameraFacing => 0u32,
            crate::renderer::SpriteOrientation::VelocityStretched => 1u32,
            crate::renderer::SpriteOrientation::AxisLocked => 2u32,
        };

        let normal_mode = match item.lit_params.normal_mode {
            crate::renderer::SpriteNormalMode::Spherical => 0u32,
            crate::renderer::SpriteNormalMode::Flat => 1u32,
            crate::renderer::SpriteNormalMode::NormalMap => 2u32,
        };

        let (normal_view, has_normal_map): (&crate::gpu::TextureView, u32) =
            if let Some(id) = item.normal_texture_id {
                if let Some(tex) = self.content.textures.get(id) {
                    (&tex.view, 1)
                } else {
                    (&self.material.normal_map_view, 0)
                }
            } else {
                (&self.material.normal_map_view, 0)
            };

        let uniform_data = SpriteUniformData {
            model: item.model,
            world_space: if item.size_mode == crate::renderer::SpriteSizeMode::WorldSpace {
                1
            } else {
                0
            },
            has_texture,
            soft_particle_distance: item
                .soft_particle_distance
                .filter(|d| *d > 0.0)
                .unwrap_or(0.0),
            orientation,
            axis: item.axis,
            refraction_strength: item.refraction_strength.filter(|s| *s > 0.0).unwrap_or(0.0),
            lit: item.lit as u32,
            normal_mode,
            has_normal_map,
            ambient_scale: item.lit_params.ambient_scale,
            roughness: item.lit_params.roughness,
            receive_shadows: item.lit_params.receive_shadows as u32,
            _pad_lit_b: 0,
            _pad_lit_c: 0,
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("sprite_uniform_buf"),
            size: std::mem::size_of::<SpriteUniformData>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .sprite
            .bgl
            .as_ref()
            .expect("ensure_sprite_pipelines not called");

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("sprite_bind_group"),
            layout: bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(texture_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(&self.material.sampler),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_buf.as_entire_binding(),
                },
            ],
        });

        let lit_normal_bg = if item.lit {
            self.sprite.lit_bgl.as_ref().map(|lit_bgl| {
                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("sprite_lit_normal_bg"),
                    layout: lit_bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: crate::gpu::BindingResource::TextureView(normal_view),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: crate::gpu::BindingResource::Sampler(&self.material.sampler),
                        },
                    ],
                })
            })
        } else {
            None
        };

        let oit_eligible = matches!(
            item.blend,
            crate::renderer::SpriteBlend::AlphaBlend | crate::renderer::SpriteBlend::Premultiplied
        ) && !item.depth_write
            && item.soft_particle_distance.is_none_or(|d| d <= 0.0)
            && item.soft_particle_distances.is_empty()
            && item.refraction_strength.is_none_or(|s| s <= 0.0);

        SpriteGpuData {
            vertex_buffer,
            sprite_count: count,
            pick_id: item.settings.pick_id,
            bind_group,
            depth_write: item.depth_write,
            blend: item.blend,
            wireframe: false,
            refraction_strength: item.refraction_strength.filter(|s| *s > 0.0).unwrap_or(0.0),
            lit: item.lit,
            lit_normal_bg,
            oit_eligible,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Lazily create the sprite outline mask pipeline (R8Unorm, mask-only).
    ///
    /// Same bind group layout and vertex transform as the normal sprite pipeline but
    /// outputs a flat mask value.  Must be called after `ensure_sprite_pipelines`.
    pub(crate) fn ensure_sprite_outline_mask_pipeline(&mut self, device: &crate::gpu::Device) {
        if self.sprite.outline_mask_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));
        let bgl = self
            .sprite
            .bgl
            .as_ref()
            .expect("ensure_sprite_pipelines must be called first");

        let shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_outline_mask_shader",
            crate::resources::builders::wgsl_source!("sprite_outline_mask"),
        );

        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "sprite_outline_mask_pipeline_layout",
            &self.binds.camera_bgl,
            bgl,
        );

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

        self.sprite.outline_mask_pipeline =
            Some(crate::resources::builders::build_outline_mask_pipeline(
                device,
                "sprite_outline_mask_pipeline",
                &layout,
                &shader,
                crate::gpu::TextureFormat::R8Unorm,
                &vertex_buffers,
                None,
                false,
                crate::gpu::CompareFunction::Less,
            ));
    }

    /// Pre-upload a static sprite set and return a typed handle.
    ///
    /// Use this for sprites whose positions, sizes, and colours never
    /// change between frames: foliage, signage, light flares. Submit a
    /// [`SpriteSetRefItem`](crate::renderer::SpriteSetRefItem) on
    /// `SceneFrame::sprite_set_refs` each frame to draw the set.
    pub fn upload_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::SpriteItem,
    ) -> crate::resources::SpriteSetId {
        self.ensure_sprite_pipelines(device);
        let gpu = self.upload_sprite(device, queue, item);
        self.content.sprite_set_store.insert(gpu)
    }

    /// Remove a pre-uploaded sprite set.
    pub fn drop_sprite_set(&mut self, id: crate::resources::SpriteSetId) -> bool {
        self.content.sprite_set_store.remove(id)
    }

    /// Replace the contents of a pre-uploaded sprite set, keeping the same id.
    pub fn replace_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::SpriteSetId,
        item: &crate::renderer::SpriteItem,
    ) -> bool {
        if !self.content.sprite_set_store.contains(id) {
            return false;
        }
        self.ensure_sprite_pipelines(device);
        let gpu = self.upload_sprite(device, queue, item);
        self.content.sprite_set_store.replace(id, gpu)
    }

    /// Start an asynchronous sprite set upload.
    pub fn begin_upload_sprite_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::SpriteItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::SpriteSetId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let sid =
                            resources.upload_sprite_set(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(sid);
                    }),
                ))
            })
        };
        self.job_results
            .sprite_set
            .lock()
            .expect("sprite set result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`SpriteSetId`] produced by a completed
    /// [`begin_upload_sprite_set`](Self::begin_upload_sprite_set) job.
    pub fn upload_result_sprite_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::SpriteSetId> {
        let mut map = self
            .job_results
            .sprite_set
            .lock()
            .expect("sprite set result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(sid) => {
                map.remove(&id);
                Ok(sid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Pre-upload a sprite instance set and return a typed handle.
    ///
    /// Use this for sprites whose definition (texture, blend, size mode)
    /// is stable but whose instance transforms change every frame: NPCs,
    /// item drops, damage numbers. Submit a
    /// [`SpriteInstanceSetRefItem`](crate::renderer::SpriteInstanceSetRefItem)
    /// on `SceneFrame::sprite_instance_set_refs` each frame.
    ///
    /// The current implementation pre-bakes both the definition and the
    /// instance transforms; full per-frame instance transform override
    /// against a stable definition is a planned follow-up.
    pub fn upload_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::SpriteItem,
    ) -> crate::resources::SpriteInstanceSetId {
        self.ensure_sprite_pipelines(device);
        let gpu = self.upload_sprite(device, queue, item);
        self.content.sprite_instance_set_store.insert(gpu)
    }

    /// Remove a pre-uploaded sprite instance set.
    pub fn drop_sprite_instance_set(&mut self, id: crate::resources::SpriteInstanceSetId) -> bool {
        self.content.sprite_instance_set_store.remove(id)
    }

    /// Replace the contents of a pre-uploaded sprite instance set, keeping
    /// the same id.
    pub fn replace_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::SpriteInstanceSetId,
        item: &crate::renderer::SpriteItem,
    ) -> bool {
        if !self.content.sprite_instance_set_store.contains(id) {
            return false;
        }
        self.ensure_sprite_pipelines(device);
        let gpu = self.upload_sprite(device, queue, item);
        self.content.sprite_instance_set_store.replace(id, gpu)
    }

    /// Start an asynchronous sprite instance set upload.
    pub fn begin_upload_sprite_instance_set(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::SpriteItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::SpriteInstanceSetId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let sid = resources.upload_sprite_instance_set(
                            &device_for_apply,
                            &queue_for_apply,
                            &item,
                        );
                        slot_for_apply.set(sid);
                    }),
                ))
            })
        };
        self.job_results
            .sprite_instance_set
            .lock()
            .expect("sprite instance set result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`SpriteInstanceSetId`] produced by a completed
    /// [`begin_upload_sprite_instance_set`](Self::begin_upload_sprite_instance_set) job.
    pub fn upload_result_sprite_instance_set(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::SpriteInstanceSetId> {
        let mut map = self
            .job_results
            .sprite_instance_set
            .lock()
            .expect("sprite instance set result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(sid) => {
                map.remove(&id);
                Ok(sid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }
}

/// Per-frame GPU data for one sprite batch item, created in `prepare()`.
#[derive(Clone)]
pub struct SpriteGpuData {
    /// Object-level pick id shared by every instance in the batch (from the
    /// item's `settings.pick_id`); `PickId::NONE` when not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Position vertex buffer: one `vec3` per sprite, instance-stepped.
    pub(crate) vertex_buffer: crate::gpu::Buffer,
    /// Number of sprites (= draw instance count).
    pub(crate) sprite_count: u32,
    /// Bind group (group 1): uniform + texture + sampler + instance storage buffer.
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// Whether this batch was submitted with `depth_write: true`.
    pub(crate) depth_write: bool,
    /// Blend mode requested by the host for this batch.
    pub(crate) blend: crate::renderer::SpriteBlend,
    /// When true, skip the billboard draw; the wireframe overlay polyline is rendered instead.
    pub(crate) wireframe: bool,
    /// Refractive distortion strength in NDC pixels; `0.0` means a regular
    /// sprite. Routes the draw through the sprite refraction post-pass
    /// instead of the normal sprite pass.
    pub(crate) refraction_strength: f32,
    /// When true, this batch was submitted with `SpriteItem::lit = true` and
    /// is drawn through the lit sprite pipeline.
    pub(crate) lit: bool,
    /// Group 3 bind group for the lit normal-map binding. Always populated for
    /// lit batches: a fallback texture is bound when no normal map is supplied
    /// so the same pipeline layout is honoured.
    pub(crate) lit_normal_bg: Option<crate::gpu::BindGroup>,
    /// True when this batch qualifies for true (weighted-blended) OIT
    /// instead of ordinary alpha blending: `blend` is `AlphaBlend` or
    /// `Premultiplied` (not `Additive`, which is already order-independent
    /// at the GPU blend-state level), `depth_write` is `false` (OIT
    /// pipelines never write depth), and neither soft-particle fade nor
    /// refractive distortion is active -- both need to sample a resolved
    /// scene texture mid-fragment (depth for soft-particle, colour for
    /// refraction), which the OIT pass exposes for neither (see
    /// `docs/plans/non-mesh-pipeline-consistency-plan.md#phase-6d`). Read by
    /// the HDR path to route the batch through `oit_pass` instead of the
    /// ordinary sprite passes.
    pub(crate) oit_eligible: bool,
    // Keep buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    pub(crate) _instance_buf: crate::gpu::Buffer,
}
