use super::*;

impl ViewportGpuResources {
    /// Lazily create the sprite billboard pipelines (alpha-blended, instanced quad expansion).
    ///
    /// Creates two pipelines that share the same shader and bind group layout but differ
    /// in `depth_write_enabled`: one for transparent effects (`depth_write: false`) and one
    /// for opaque-style placed sprites (`depth_write: true`).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.sprite_items` is
    /// non-empty.
    pub(crate) fn ensure_sprite_pipelines(&mut self, device: &wgpu::Device) {
        if self.sprite_bgl.is_some() {
            return;
        }

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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sprite_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/sprite.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &bgl],
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
        let make_sprite = |fmt: wgpu::TextureFormat, depth_write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if depth_write { "sprite_pipeline_depth_write" } else { "sprite_pipeline" }),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
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
                    depth_write_enabled: depth_write,
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
            })
        };

        let ldr = self.target_format;
        let hdr = wgpu::TextureFormat::Rgba16Float;
        self.sprite_bgl = Some(bgl);
        self.sprite_pipeline = Some(DualPipeline {
            ldr: make_sprite(ldr, false),
            hdr: make_sprite(hdr, false),
        });
        self.sprite_pipeline_depth_write = Some(DualPipeline {
            ldr: make_sprite(ldr, true),
            hdr: make_sprite(hdr, true),
        });
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
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuSpriteInstance {
            color:    [f32; 4],
            size:     f32,
            rotation: f32,
            _pad0:    f32,
            _pad1:    f32,
            uv_rect:  [f32; 4],
        }

        let instances: Vec<GpuSpriteInstance> = (0..item.positions.len())
            .map(|i| GpuSpriteInstance {
                color: if i < item.colors.len() { item.colors[i] } else { item.default_color },
                size: if i < item.sizes.len() { item.sizes[i] } else { item.default_size },
                rotation: if i < item.rotations.len() { item.rotations[i] } else { 0.0 },
                _pad0: 0.0,
                _pad1: 0.0,
                uv_rect: if i < item.uv_rects.len() {
                    item.uv_rects[i]
                } else {
                    [0.0, 0.0, 1.0, 1.0]
                },
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

        // Uniform buffer: model matrix + flags.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SpriteUniformData {
            model:       [[f32; 4]; 4],
            world_space: u32,
            has_texture: u32,
            _pad0:       u32,
            _pad1:       u32,
        }

        let (texture_view, has_texture): (&wgpu::TextureView, u32) =
            if let Some(id) = item.texture_id {
                if let Some(tex) = self.textures.get(id as usize) {
                    (&tex.view, 1)
                } else {
                    (&self.fallback_lut_view, 0)
                }
            } else {
                (&self.fallback_lut_view, 0)
            };

        let uniform_data = SpriteUniformData {
            model: item.model,
            world_space: if item.size_mode
                == crate::renderer::SpriteSizeMode::WorldSpace
            {
                1
            } else {
                0
            },
            has_texture,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite_uniform_buf"),
            size: std::mem::size_of::<SpriteUniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .sprite_bgl
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

        SpriteGpuData {
            vertex_buffer,
            sprite_count: count,
            bind_group,
            depth_write: item.depth_write,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }
}
