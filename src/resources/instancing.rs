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
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced.wgsl")).into(),
            ),
        });

        let instanced_layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
            device,
            "instanced_pipeline_layout",
            &self.camera_bind_group_layout,
            &instance_bgl,
            &self.deform.bind_group_layout,
        );
        let ldr_inst = crate::resources::mesh_pipelines::build_ldr_instanced_mesh_pipelines(
            device,
            &instanced_layout,
            &instanced_shader,
            self.target_format,
            self.sample_count,
        );
        let solid_instanced = ldr_inst.solid;
        let transparent_instanced = ldr_inst.transparent;

        // Shadow instanced pipeline.
        let shadow_instanced_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_instanced_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/shadow_instanced.wgsl")).into(),
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

    /// Ensure the HDR instanced pipelines exist. Called after
    /// `ensure_instanced_pipelines` so that `instance_bind_group_layout` is
    /// available. Idempotent: returns immediately if the pipelines already
    /// exist or if the BGL hasn't been created yet.
    pub(crate) fn ensure_hdr_instanced_pipelines(&mut self, device: &wgpu::Device) {
        if self.hdr_solid_instanced_pipeline.is_some() {
            return;
        }
        let Some(ref instance_bgl) = self.instance_bind_group_layout else {
            return;
        };

        let inst_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_instanced_shader_hdr"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced.wgsl")).into(),
            ),
        });
        let inst_layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
            device,
            "hdr_instanced_pipeline_layout",
            &self.camera_bind_group_layout,
            instance_bgl,
            &self.deform.bind_group_layout,
        );
        let hdr_inst = crate::resources::mesh_pipelines::build_hdr_instanced_mesh_pipelines(
            device,
            &inst_layout,
            &inst_shader,
        );
        self.hdr_solid_instanced_pipeline = Some(hdr_inst.solid);
        self.hdr_transparent_instanced_pipeline = Some(hdr_inst.transparent);
        self.hdr_instanced_additive_pipeline = Some(hdr_inst.additive);
        self.hdr_instanced_premultiplied_pipeline = Some(hdr_inst.premultiplied);
    }

    /// Ensure the OIT instanced pipeline exists. Called after
    /// `ensure_instanced_pipelines` so that `instance_bind_group_layout` is
    /// available. Idempotent: returns immediately if the pipeline already
    /// exists or if the BGL hasn't been created yet.
    pub(crate) fn ensure_oit_instanced_pipeline(&mut self, device: &wgpu::Device) {
        if self.oit_instanced_pipeline.is_some() {
            return;
        }
        let Some(ref instance_bgl) = self.instance_bind_group_layout else {
            return;
        };

        let instanced_oit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_instanced_oit_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced_oit.wgsl")).into(),
            ),
        });
        let instanced_oit_layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
            device,
            "oit_instanced_pipeline_layout",
            &self.camera_bind_group_layout,
            instance_bgl,
            &self.deform.bind_group_layout,
        );
        let pipeline = crate::resources::mesh_pipelines::build_oit_instanced_pipeline(
            device,
            &instanced_oit_layout,
            &instanced_oit_shader,
            "oit_instanced_pipeline",
            "vs_main",
        );

        self.oit_instanced_pipeline = Some(pipeline);
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

        // --- visibility_index_buf + shadow_vis_bufs: same count as instances ---
        if aabbs.len() > self.visibility_index_capacity {
            let new_cap = (aabbs.len() * 2).max(64).min(max_instances);
            let vis_size = (new_cap * std::mem::size_of::<u32>()) as u64;
            self.visibility_index_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visibility_index_buf"),
                size: vis_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.visibility_index_capacity = new_cap;
            // Per-cascade shadow visibility buffers grow with the instance count.
            for i in 0..4 {
                self.shadow_vis_bufs[i] = Some(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("shadow_vis_buf_{i}")),
                    size: vis_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            // Invalidate bind groups that reference the old shadow vis buffers.
            self.shadow_cull_instance_bgs = [None, None, None, None];
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
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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
        self.shadow_cull_instance_bgs = [None, None, None, None];
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
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced.wgsl")).into(),
            ),
        });
        let inst_cull_layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
            device,
            "hdr_instanced_cull_pipeline_layout",
            &self.camera_bind_group_layout,
            &cull_bgl,
            &self.deform.bind_group_layout,
        );
        let hdr_solid_cull = crate::resources::mesh_pipelines::build_hdr_instanced_cull_pipeline(
            device,
            &inst_cull_layout,
            &instanced_shader,
        );

        // OIT cull pipeline: Rgba16Float + R8Unorm targets, vs_main_cull, no depth write.
        let oit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_instanced_oit_shader_cull"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/mesh_instanced_oit.wgsl")).into(),
            ),
        });
        let oit_cull_layout = crate::resources::mesh_pipelines::instanced_pipeline_layout(
            device,
            "oit_instanced_cull_pipeline_layout",
            &self.camera_bind_group_layout,
            &cull_bgl,
            &self.deform.bind_group_layout,
        );
        let oit_cull = crate::resources::mesh_pipelines::build_oit_instanced_pipeline(
            device,
            &oit_cull_layout,
            &oit_shader,
            "oit_instanced_cull_pipeline",
            "vs_main_cull",
        );

        self.instance_cull_bind_group_layout = Some(cull_bgl);
        self.hdr_solid_instanced_cull_pipeline = Some(hdr_solid_cull);
        self.oit_instanced_cull_pipeline = Some(oit_cull);

        // Shadow instanced cull pipeline.
        // Uses a minimal BGL for group 1: binding 0 (instances) + binding 5 (visibility_indices).
        // Group 0 reuses the existing shadow cascade BGL (single mat4x4 uniform).
        let shadow_cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_cull_instance_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
        // Recreate the shadow cascade BGL (same definition as in ensure_instanced_pipelines).
        let shadow_bgl_for_cull =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow_bgl_for_cull"),
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
        let shadow_cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_instanced_cull_pipeline_layout"),
            bind_group_layouts: &[&shadow_bgl_for_cull, &shadow_cull_bgl],
            push_constant_ranges: &[],
        });
        let shadow_cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_instanced_cull_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/shadow_instanced.wgsl")).into(),
            ),
        });
        let shadow_instanced_cull =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow_instanced_cull_pipeline"),
                layout: Some(&shadow_cull_layout),
                vertex: wgpu::VertexState {
                    module: &shadow_cull_shader,
                    entry_point: Some("vs_shadow_cull"),
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
        self.shadow_instanced_cull_pipeline = Some(shadow_instanced_cull);
        self.shadow_cull_instance_bgl = Some(shadow_cull_bgl);
    }

    /// Get or create the shadow cull instance bind group for a given cascade index.
    ///
    /// Binds `instance_storage_buf` (binding 0) and `shadow_vis_bufs[cascade_idx]` (binding 5).
    /// Returns `None` if the required buffers or BGL are not yet allocated.
    pub(crate) fn get_shadow_cull_instance_bind_group(
        &mut self,
        device: &wgpu::Device,
        cascade_idx: usize,
    ) -> Option<&wgpu::BindGroup> {
        if self.shadow_cull_instance_bgs[cascade_idx].is_none() {
            let bgl = self.shadow_cull_instance_bgl.as_ref()?;
            let inst_buf = self.instance_storage_buf.as_ref()?;
            let vis_buf = self.shadow_vis_bufs[cascade_idx].as_ref()?;
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("shadow_cull_instance_bg_{cascade_idx}")),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: vis_buf.as_entire_binding(),
                    },
                ],
            });
            self.shadow_cull_instance_bgs[cascade_idx] = Some(bg);
        }
        self.shadow_cull_instance_bgs[cascade_idx].as_ref()
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

    /// Upload one [`MeshInstanceItem`] batch and return draw data.
    ///
    /// Builds a per-batch instance storage buffer in the layout expected by
    /// `mesh_instanced.wgsl`'s `InstanceData` struct, allocates a one-shot
    /// bind group against `instance_bind_group_layout`, and packages it into
    /// a [`MeshInstanceGpuData`]. The host is expected to rebuild this once
    /// per frame for moving particle systems.
    ///
    /// Lighting flags are filled with unlit defaults: the shader receives the
    /// per-instance colour with the optional albedo sampled on top, without
    /// going through shadow or lighting math.
    pub(crate) fn upload_mesh_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::MeshInstanceItem,
    ) -> Option<crate::resources::types::MeshInstanceGpuData> {
        let instance_count = item.transforms.len() as u32;
        if instance_count == 0 {
            return None;
        }

        // Per-instance struct must match `InstanceData` in `mesh_instanced.wgsl`.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GpuInstanceData {
            model: [[f32; 4]; 4],
            colour: [f32; 4],
            selected: u32,
            wireframe: u32,
            ambient: f32,
            diffuse: f32,
            specular: f32,
            shininess: f32,
            has_texture: u32,
            use_pbr: u32,
            metallic: f32,
            roughness: f32,
            has_normal_map: u32,
            has_ao_map: u32,
            unlit: u32,
            receive_shadows: u32,
            use_flat: u32,
            _pad_inst1: u32,
        }

        let has_texture = if item
            .texture_id
            .is_some_and(|id| (id as usize) < self.textures.len())
        {
            1u32
        } else {
            0u32
        };

        let instances: Vec<GpuInstanceData> = (0..item.transforms.len())
            .map(|i| GpuInstanceData {
                model: item.transforms[i],
                colour: item.colours.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
                selected: 0,
                wireframe: 0,
                ambient: 1.0,
                diffuse: 0.0,
                specular: 0.0,
                shininess: 0.0,
                has_texture,
                use_pbr: 0,
                metallic: 0.0,
                roughness: 0.0,
                has_normal_map: 0,
                has_ao_map: 0,
                unlit: 1,
                receive_shadows: 0,
                use_flat: 1,
                _pad_inst1: 0,
            })
            .collect();

        let instance_bytes = bytemuck::cast_slice(&instances);
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh_instance_buf"),
            size: instance_bytes.len().max(144) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, instance_bytes);

        let bgl = self.instance_bind_group_layout.as_ref()?;
        let albedo_view = match item.texture_id {
            Some(id) if (id as usize) < self.textures.len() => &self.textures[id as usize].view,
            _ => &self.fallback_texture.view,
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh_instance_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instance_buf.as_entire_binding(),
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
                    resource: wgpu::BindingResource::TextureView(&self.fallback_normal_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_ao_map_view),
                },
            ],
        });

        let mesh_id = crate::resources::mesh_store::MeshId(item.mesh_id as usize);

        Some(crate::resources::types::MeshInstanceGpuData {
            mesh_id,
            instance_count,
            bind_group,
            blend: item.blend,
            _instance_buf: instance_buf,
        })
    }
}
