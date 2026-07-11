use super::*;

/// Sprite billboard pipelines (emissive + lit, one per blend/depth-write pair),
/// their bind group layouts, refraction pass, and soft-particle fallbacks. All
/// lazily built; the uploaded sprite sets live in separate flat stores.
#[derive(Default)]
pub(crate) struct SpriteResources {
    /// Sprite pipeline, alpha-blend, depth_write_enabled: false.
    pub(crate) pipeline: Option<DualPipeline>,
    pub(crate) pipeline_depth_write: Option<DualPipeline>,
    pub(crate) pipeline_additive: Option<DualPipeline>,
    pub(crate) pipeline_additive_depth_write: Option<DualPipeline>,
    pub(crate) pipeline_premultiplied: Option<DualPipeline>,
    pub(crate) pipeline_premultiplied_depth_write: Option<DualPipeline>,
    /// Refractive sprite pipeline (HDR target only).
    pub(crate) refraction_pipeline: Option<wgpu::RenderPipeline>,
    /// Group 2 BGL for the refraction pipeline: scene-colour texture + sampler.
    pub(crate) refraction_bgl: Option<wgpu::BindGroupLayout>,
    /// Sampler used by the refraction shader to read the scene-colour resolve.
    pub(crate) refraction_sampler: Option<wgpu::Sampler>,
    /// Bind group layout for sprite uniforms + texture + instance buffer (group 1).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for the per-pass scene-depth resolve bound at group 2.
    pub(crate) soft_bgl: Option<wgpu::BindGroupLayout>,
    /// Fallback bind group for the group-2 soft-particle binding.
    pub(crate) soft_fallback_bg: Option<wgpu::BindGroup>,
    /// Sampler used for the group-2 scene-depth binding.
    pub(crate) soft_sampler: Option<wgpu::Sampler>,
    /// 1x1 Depth32Float texture backing the soft fallback bind group.
    pub(crate) soft_fallback_tex: Option<wgpu::Texture>,
    /// Lit sprite pipelines: one per (blend, depth_write) pair.
    pub(crate) lit_pipeline: Option<DualPipeline>,
    pub(crate) lit_pipeline_depth_write: Option<DualPipeline>,
    pub(crate) lit_pipeline_additive: Option<DualPipeline>,
    pub(crate) lit_pipeline_additive_depth_write: Option<DualPipeline>,
    pub(crate) lit_pipeline_premultiplied: Option<DualPipeline>,
    pub(crate) lit_pipeline_premultiplied_depth_write: Option<DualPipeline>,
    /// Group 3 BGL for the optional lit normal map (texture + sampler).
    pub(crate) lit_bgl: Option<wgpu::BindGroupLayout>,
    /// Fallback bind group for the lit normal map binding.
    pub(crate) lit_fallback_bg: Option<wgpu::BindGroup>,
    /// 1x1 RGBA8Unorm texture backing the lit fallback bind group.
    pub(crate) lit_fallback_tex: Option<wgpu::Texture>,
    /// Sprite outline mask pipeline (R8Unorm). None until first selected sprite.
    pub(crate) outline_mask_pipeline: Option<wgpu::RenderPipeline>,
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
    pub(crate) fn ensure_sprite_pipelines(&mut self, device: &wgpu::Device) {
        if self.sprite.bgl.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprite_bgl"),
            entries: &[
                // binding 0: SpriteUniform (model, world_space, has_texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: sprite texture (or fallback 1x1 when has_texture == 0)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: per-sprite instance storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let soft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprite_soft_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let soft_sampler =
            crate::resources::builders::clamp_nearest_sampler(device, "sprite_soft_sampler");

        let fallback_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sprite_soft_fallback_tex"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let fallback_view = fallback_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let fallback_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite_soft_fallback_bg"),
            layout: &soft_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fallback_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&soft_sampler),
                },
            ],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_shader",
            crate::resources::builders::wgsl_source!("sprite"),
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &bgl, &soft_bgl],
            push_constant_ranges: &[],
        });

        // Position vertex buffer: one vec3 per sprite, Instance stepping.
        // Stored in an array so both pipeline creations can borrow from it.
        let vert_attrs = [wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        }];
        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &vert_attrs,
        }];

        let sample_count = self.sample_count;
        let ldr_format = self.target_format;
        // Sprites are billboards drawn with `Less` depth test, no culling. Each
        // variant differs only in blend mode and whether it writes depth.
        let make_sprite = |depth_write: bool, blend: wgpu::BlendState, label: &str| {
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
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write,
                    depth_compare: wgpu::CompareFunction::Less,
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
            wgpu::ShaderStages::FRAGMENT,
        );

        let alpha = wgpu::BlendState::ALPHA_BLENDING;
        let additive = crate::resources::builders::ADDITIVE_BLEND;
        let premultiplied = crate::resources::builders::PREMULTIPLIED_BLEND;
        self.sprite.bgl = Some(bgl);
        self.sprite.soft_bgl = Some(soft_bgl);
        self.sprite.soft_sampler = Some(soft_sampler);
        self.sprite.soft_fallback_tex = Some(fallback_tex);
        self.sprite.soft_fallback_bg = Some(fallback_bg);
        self.sprite.pipeline = Some(make_sprite(false, alpha, "sprite_pipeline"));
        self.sprite.pipeline_depth_write =
            Some(make_sprite(true, alpha, "sprite_pipeline_depth_write"));
        self.sprite.pipeline_additive =
            Some(make_sprite(false, additive, "sprite_pipeline_additive"));
        self.sprite.pipeline_additive_depth_write = Some(make_sprite(
            true,
            additive,
            "sprite_pipeline_additive_depth_write",
        ));
        self.sprite.pipeline_premultiplied = Some(make_sprite(
            false,
            premultiplied,
            "sprite_pipeline_premultiplied",
        ));
        self.sprite.pipeline_premultiplied_depth_write = Some(make_sprite(
            true,
            premultiplied,
            "sprite_pipeline_premultiplied_depth_write",
        ));

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
            wgpu::ShaderStages::FRAGMENT,
        );

        let refraction_sampler =
            crate::resources::builders::clamp_linear_sampler(device, "sprite_refraction_sampler");

        let refraction_shader = crate::resources::builders::wgsl_module(
            device,
            "sprite_refraction_shader",
            crate::resources::builders::wgsl_source!("sprite_refraction"),
        );

        let bgl_ref = self.sprite.bgl.as_ref().unwrap();
        let refraction_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite_refraction_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, bgl_ref, &refraction_bgl],
            push_constant_ranges: &[],
        });

        let refraction_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprite_refraction_pipeline"),
            layout: Some(&refraction_layout),
            vertex: wgpu::VertexState {
                module: &refraction_shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &refraction_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

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
        let lit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite_lit_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                sprite_bgl_ref,
                soft_bgl_ref,
                &lit_bgl,
            ],
            push_constant_ranges: &[],
        });

        let make_lit = |depth_write: bool, blend: wgpu::BlendState, label: &str| {
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
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    depth_write,
                    depth_compare: wgpu::CompareFunction::Less,
                    sample_count,
                    ldr_format,
                },
            )
        };

        self.sprite.lit_pipeline = Some(make_lit(false, alpha, "sprite_lit_pipeline"));
        self.sprite.lit_pipeline_depth_write =
            Some(make_lit(true, alpha, "sprite_lit_pipeline_depth_write"));
        self.sprite.lit_pipeline_additive =
            Some(make_lit(false, additive, "sprite_lit_pipeline_additive"));
        self.sprite.lit_pipeline_additive_depth_write = Some(make_lit(
            true,
            additive,
            "sprite_lit_pipeline_additive_depth_write",
        ));
        self.sprite.lit_pipeline_premultiplied = Some(make_lit(
            false,
            premultiplied,
            "sprite_lit_pipeline_premultiplied",
        ));
        self.sprite.lit_pipeline_premultiplied_depth_write = Some(make_lit(
            true,
            premultiplied,
            "sprite_lit_pipeline_premultiplied_depth_write",
        ));

        // The fallback bind group reuses the crate-wide `fallback_normal_map`,
        // already populated with `(128, 128, 255, 255)` for tangent-space `(0, 0, 1)`.
        let lit_fallback_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite_lit_fallback_bg"),
            layout: &lit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_normal_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
            ],
        });

        self.sprite.lit_bgl = Some(lit_bgl);
        self.sprite.lit_fallback_bg = Some(lit_fallback_bg);
    }

    /// Upload one [`SpriteItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.scene.sprite_items`.
    pub(crate) fn upload_sprite(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::SpriteItem,
    ) -> SpriteGpuData {
        let count = item.positions.len() as u32;

        // Position vertex buffer (one vec3 per sprite, instance-stepped).
        let pos_bytes: Vec<u8> = item
            .positions
            .iter()
            .flat_map(|p| bytemuck::bytes_of(p).iter().copied())
            .collect();
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite_vertex_buf"),
            size: pos_bytes.len().max(12) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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
                    item.colours[i]
                } else {
                    item.default_colour
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
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite_instance_buf"),
            size: instance_bytes.len().max(48) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        let (texture_view, has_texture): (&wgpu::TextureView, u32) =
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

        let (normal_view, has_normal_map): (&wgpu::TextureView, u32) =
            if let Some(id) = item.normal_texture_id {
                if let Some(tex) = self.content.textures.get(id) {
                    (&tex.view, 1)
                } else {
                    (&self.fallback_normal_map_view, 0)
                }
            } else {
                (&self.fallback_normal_map_view, 0)
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
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite_uniform_buf"),
            size: std::mem::size_of::<SpriteUniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .sprite
            .bgl
            .as_ref()
            .expect("ensure_sprite_pipelines not called");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite_bind_group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: instance_buf.as_entire_binding(),
                },
            ],
        });

        let lit_normal_bg = if item.lit {
            self.sprite.lit_bgl.as_ref().map(|lit_bgl| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sprite_lit_normal_bg"),
                    layout: lit_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                        },
                    ],
                })
            })
        } else {
            None
        };

        SpriteGpuData {
            vertex_buffer,
            sprite_count: count,
            bind_group,
            depth_write: item.depth_write,
            blend: item.blend,
            wireframe: false,
            refraction_strength: item.refraction_strength.filter(|s| *s > 0.0).unwrap_or(0.0),
            lit: item.lit,
            lit_normal_bg,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Lazily create the sprite outline mask pipeline (R8Unorm, mask-only).
    ///
    /// Same bind group layout and vertex transform as the normal sprite pipeline but
    /// outputs a flat mask value.  Must be called after `ensure_sprite_pipelines`.
    pub(crate) fn ensure_sprite_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
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
            &self.camera_bind_group_layout,
            bgl,
        );

        let vert_attrs = [wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        }];
        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &vert_attrs,
        }];

        self.sprite.outline_mask_pipeline =
            Some(crate::resources::builders::build_outline_mask_pipeline(
                device,
                "sprite_outline_mask_pipeline",
                &layout,
                &shader,
                wgpu::TextureFormat::R8Unorm,
                &vertex_buffers,
                None,
                false,
                wgpu::CompareFunction::Less,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
    /// Position vertex buffer: one `vec3` per sprite, instance-stepped.
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// Number of sprites (= draw instance count).
    pub(crate) sprite_count: u32,
    /// Bind group (group 1): uniform + texture + sampler + instance storage buffer.
    pub(crate) bind_group: wgpu::BindGroup,
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
    pub(crate) lit_normal_bg: Option<wgpu::BindGroup>,
    // Keep buffers alive for the lifetime of this struct.
    pub(crate) _uniform_buf: wgpu::Buffer,
    pub(crate) _instance_buf: wgpu::Buffer,
}
