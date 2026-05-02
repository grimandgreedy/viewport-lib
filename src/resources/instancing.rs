use super::*;

impl ViewportGpuResources {
    /// Ensure the instanced pipelines and bind group layout are created.
    /// Called lazily when the instanced draw path is first needed.
    pub(crate) fn ensure_instanced_pipelines(&mut self, device: &wgpu::Device) {
        if self.instance_bind_group_layout.is_some() {
            return; // Already initialized.
        }

        // Instanced bind group layout (group 1 for instanced pipelines).
        // binding 0: instance storage buffer
        // binding 1-4: albedo texture, sampler, normal map, AO map
        // Co-located in group 1 to stay within iced's max_bind_groups = 2.
        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_bgl"),
            entries: &[
                // binding 0: instance storage buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: albedo texture
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
                // binding 3: normal map texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 4: AO map texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Instanced mesh shader.
        let instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_instanced_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh_instanced.wgsl").into()),
        });

        let instanced_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("instanced_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &instance_bgl],
            push_constant_ranges: &[],
        });

        // Solid instanced pipeline.
        let solid_instanced = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("solid_instanced_pipeline"),
            layout: Some(&instanced_layout),
            vertex: wgpu::VertexState {
                module: &instanced_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &instanced_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        // Transparent instanced pipeline.
        let transparent_instanced =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("transparent_instanced_pipeline"),
                layout: Some(&instanced_layout),
                vertex: wgpu::VertexState {
                    module: &instanced_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &instanced_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.target_format,
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
                    count: self.sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

        // Shadow instanced pipeline.
        let shadow_instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_instanced_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/shadow_instanced.wgsl").into(),
            ),
        });

        // Shadow instanced uses the shadow bind group layout (group 0) + instance_bgl (group 1).
        // Re-derive the shadow BGL from the existing shadow_bind_group.
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_bgl_for_instanced"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_instanced_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow_instanced_pipeline_layout"),
                bind_group_layouts: &[&shadow_bgl, &instance_bgl],
                push_constant_ranges: &[],
            });

        let shadow_instanced = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_instanced_pipeline"),
            layout: Some(&shadow_instanced_layout),
            vertex: wgpu::VertexState {
                module: &shadow_instanced_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Allocate 4 per-cascade uniform buffers (64 bytes each = one mat4x4) and
        // create bind groups for shadow_instanced_pipeline group 0.
        // Each cascade has its own small buffer so we can write_buffer(buf, 0, ...) without
        // dynamic offsets (shadow_instanced.wgsl group 0 binds a single uniform, not an array).
        let cascade_bufs: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("shadow_instanced_cascade_buf_{i}")),
                size: 64, // sizeof(mat4x4<f32>)
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let cascade_bgs: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shadow_instanced_cascade_bg_{i}")),
                layout: &shadow_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cascade_bufs[i].as_entire_binding(),
                }],
            })
        });
        self.shadow_instanced_cascade_bufs = cascade_bufs.map(Some);
        self.shadow_instanced_cascade_bgs = cascade_bgs.map(Some);

        self.instance_bind_group_layout = Some(instance_bgl);
        self.solid_instanced_pipeline = Some(solid_instanced);
        self.transparent_instanced_pipeline = Some(transparent_instanced);
        self.shadow_instanced_pipeline = Some(shadow_instanced);
    }

    /// Upload instance data to the storage buffer, resizing if needed.
    /// Returns the bind group for the instance storage buffer.
    pub(crate) fn upload_instance_data(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[InstanceData],
    ) {
        if data.is_empty() {
            return;
        }

        let _bgl = self
            .instance_bind_group_layout
            .as_ref()
            .expect("ensure_instanced_pipelines must be called first");

        // Clamp to the device's max_storage_buffer_binding_size so bind group
        // creation never panics regardless of scene size.
        let max_instances = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<InstanceData>();
        let data = &data[..data.len().min(max_instances)];

        let needed = data.len();
        if needed > self.instance_storage_capacity {
            // Grow with 2x strategy, capped at the device limit.
            let new_cap = (needed * 2).max(64).min(max_instances);
            let buf_size = (new_cap * std::mem::size_of::<InstanceData>()) as u64;
            self.instance_storage_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_storage_buf"),
                size: buf_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_storage_capacity = new_cap;

            // Invalidate all per-texture-key bind groups; they reference the old buffer.
            self.instance_bind_groups.clear();
        }

        queue.write_buffer(
            self.instance_storage_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(data),
        );
    }

    /// Upload per-instance AABB data and per-batch metadata to GPU buffers.
    ///
    /// Allocates or grows buffers using the same 2x strategy as `upload_instance_data`.
    /// Also allocates `visibility_index_buf`, `batch_counter_buf`, `indirect_args_buf`,
    /// and `shadow_indirect_bufs` at the same time since they share the same capacity.
    /// Call on every batch cache miss, immediately after `upload_instance_data`.
    pub(crate) fn upload_aabb_and_batch_meta(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        aabbs: &[crate::resources::types::InstanceAabb],
        metas: &[crate::resources::types::BatchMeta],
    ) {
        // --- AABB buffer (per-instance) ---
        let max_instances = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<crate::resources::types::InstanceAabb>();
        let aabbs = &aabbs[..aabbs.len().min(max_instances)];

        if aabbs.len() > self.instance_aabb_capacity {
            let new_cap = (aabbs.len() * 2).max(64).min(max_instances);
            self.instance_aabb_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_aabb_buf"),
                size: (new_cap * std::mem::size_of::<crate::resources::types::InstanceAabb>())
                    as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.instance_aabb_capacity = new_cap;
        }
        if !aabbs.is_empty() {
            queue.write_buffer(
                self.instance_aabb_buf.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(aabbs),
            );
        }

        // --- visibility_index_buf: same count as instances ---
        if aabbs.len() > self.visibility_index_capacity {
            let new_cap = (aabbs.len() * 2).max(64).min(max_instances);
            self.visibility_index_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visibility_index_buf"),
                size: (new_cap * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.visibility_index_capacity = new_cap;
        }

        // --- Batch meta + counter + indirect args buffers (per-batch) ---
        let max_batches = (device.limits().max_storage_buffer_binding_size as usize)
            / std::mem::size_of::<crate::resources::types::BatchMeta>();
        let metas = &metas[..metas.len().min(max_batches)];
        let batch_count = metas.len();

        if batch_count > self.batch_meta_capacity {
            let new_cap = (batch_count * 2).max(16).min(max_batches);
            let meta_size =
                (new_cap * std::mem::size_of::<crate::resources::types::BatchMeta>()) as u64;
            let counter_size = (new_cap * std::mem::size_of::<u32>()) as u64;
            // wgpu::util::DrawIndexedIndirect is 5 × u32 = 20 bytes.
            let indirect_size = (new_cap * 20) as u64;

            self.batch_meta_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch_meta_buf"),
                size: meta_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.batch_counter_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch_counter_buf"),
                size: counter_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.indirect_args_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("indirect_args_buf"),
                size: indirect_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            for i in 0..4 {
                self.shadow_indirect_bufs[i] =
                    Some(device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("shadow_indirect_buf_{i}")),
                        size: indirect_size,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::INDIRECT
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }
            self.batch_meta_capacity = new_cap;
        }

        if !metas.is_empty() {
            queue.write_buffer(
                self.batch_meta_buf.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(metas),
            );
        }

        // Invalidate cull bind groups when visibility_index_buf was just (re-)allocated.
        // The new buffer is only detectable by comparing capacity before and after.
        // Simplest approach: always invalidate on upload (cache miss already guards frequency).
        self.instance_cull_bind_groups.clear();
    }

    /// Ensure the GPU-driven cull variant pipelines and BGL are created.
    ///
    /// Must be called after `ensure_instanced_pipelines`.  Idempotent.
    pub(crate) fn ensure_cull_instance_pipelines(&mut self, device: &wgpu::Device) {
        if self.instance_cull_bind_group_layout.is_some() {
            return;
        }

        let Some(ref _instance_bgl) = self.instance_bind_group_layout else {
            return; // ensure_instanced_pipelines must be called first.
        };

        // Cull BGL = instance_bgl bindings 0-4 + binding 5: visibility_indices (read, VERTEX).
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instance_cull_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 5: visibility_indices (written by compute cull pass, read in vertex shader)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        // HDR solid cull pipeline: Rgba16Float target, vs_main_cull, back-face cull.
        let instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_instanced_shader_cull"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mesh_instanced.wgsl").into(),
            ),
        });
        let inst_cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdr_instanced_cull_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &cull_bgl],
            push_constant_ranges: &[],
        });
        let hdr_solid_cull = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hdr_solid_instanced_cull_pipeline"),
            layout: Some(&inst_cull_layout),
            vertex: wgpu::VertexState {
                module: &instanced_shader,
                entry_point: Some("vs_main_cull"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &instanced_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        // OIT cull pipeline: Rgba16Float + R8Unorm targets, vs_main_cull, no depth write.
        let oit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_instanced_oit_shader_cull"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mesh_instanced_oit.wgsl").into(),
            ),
        });
        let oit_cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("oit_instanced_cull_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &cull_bgl],
            push_constant_ranges: &[],
        });
        let accum_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let reveal_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let oit_cull = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("oit_instanced_cull_pipeline"),
            layout: Some(&oit_cull_layout),
            vertex: wgpu::VertexState {
                module: &oit_shader,
                entry_point: Some("vs_main_cull"),
                buffers: &[Vertex::buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &oit_shader,
                entry_point: Some("fs_oit_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(accum_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: Some(reveal_blend),
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.instance_cull_bind_group_layout = Some(cull_bgl);
        self.hdr_solid_instanced_cull_pipeline = Some(hdr_solid_cull);
        self.oit_instanced_cull_pipeline = Some(oit_cull);
    }

    /// Get or create a cull-path bind group for the instanced cull pipeline.
    ///
    /// Identical to `get_instance_bind_group` but uses `instance_cull_bind_group_layout`
    /// and includes the `visibility_index_buf` at binding 5.
    pub(crate) fn get_instance_cull_bind_group(
        &mut self,
        device: &wgpu::Device,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
    ) -> Option<&wgpu::BindGroup> {
        let key = (
            albedo_id.unwrap_or(u64::MAX),
            normal_map_id.unwrap_or(u64::MAX),
            ao_map_id.unwrap_or(u64::MAX),
        );

        if !self.instance_cull_bind_groups.contains_key(&key) {
            let bgl = self.instance_cull_bind_group_layout.as_ref()?;
            let inst_buf = self.instance_storage_buf.as_ref()?;
            let vis_buf = self.visibility_index_buf.as_ref()?;

            let albedo_view = match albedo_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("instance_cull_tex_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: vis_buf.as_entire_binding(),
                    },
                ],
            });
            self.instance_cull_bind_groups.insert(key, bg);
        }

        self.instance_cull_bind_groups.get(&key)
    }

    /// Get or create a combined instance+texture bind group for the instanced pipeline.
    ///
    /// The bind group combines the shared instance storage buffer (binding 0) with the
    /// texture views for the given material key (bindings 1-4). Results are cached by key.
    ///
    /// `u64::MAX` in any key component means "use fallback texture for that slot".
    pub(crate) fn get_instance_bind_group(
        &mut self,
        device: &wgpu::Device,
        albedo_id: Option<u64>,
        normal_map_id: Option<u64>,
        ao_map_id: Option<u64>,
    ) -> Option<&wgpu::BindGroup> {
        let key = (
            albedo_id.unwrap_or(u64::MAX),
            normal_map_id.unwrap_or(u64::MAX),
            ao_map_id.unwrap_or(u64::MAX),
        );

        if !self.instance_bind_groups.contains_key(&key) {
            let bgl = self.instance_bind_group_layout.as_ref()?;
            let buf = self.instance_storage_buf.as_ref()?;

            let albedo_view = match albedo_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_texture.view,
            };
            let normal_view = match normal_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_normal_map_view,
            };
            let ao_view = match ao_map_id {
                Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
                _ => &self.fallback_ao_map_view,
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("instance_tex_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(ao_view),
                    },
                ],
            });
            self.instance_bind_groups.insert(key, bg);
        }

        self.instance_bind_groups.get(&key)
    }
}
