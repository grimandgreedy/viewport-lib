use super::*;

impl ViewportGpuResources {
    /// Lazily create the glyph render pipeline (instanced TriangleList).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.glyphs` is non-empty.
    pub(crate) fn ensure_glyph_pipeline(&mut self, device: &wgpu::Device) {
        if self.glyph_pipeline.is_some() {
            return;
        }

        let glyph_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("glyph_bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let glyph_instance_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("glyph_instance_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glyph_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/glyph.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glyph_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &glyph_bgl,
                &glyph_instance_bgl,
            ],
            push_constant_ranges: &[],
        });

        let sample_count = self.sample_count;
        let make = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("glyph_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
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
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        self.glyph_bgl = Some(glyph_bgl);
        self.glyph_instance_bgl = Some(glyph_instance_bgl);
        self.glyph_pipeline = Some(DualPipeline {
            ldr: make(self.target_format),
            hdr: make(wgpu::TextureFormat::Rgba16Float),
        });
    }

    /// Upload one [`GlyphItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.scene.glyphs`.
    /// The glyph base mesh is cached in `glyph_arrow_mesh` / `glyph_sphere_mesh` / `glyph_cube_mesh`.
    pub(crate) fn upload_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::GlyphItem,
    ) -> GlyphGpuData {
        let instance_count = item.positions.len() as u32;

        self.ensure_glyph_mesh(device, item.glyph_type);

        let (mesh_vbuf, mesh_ibuf, mesh_idx_count) = {
            let mesh = match item.glyph_type {
                crate::renderer::GlyphType::Arrow => self.glyph_arrow_mesh.as_ref(),
                crate::renderer::GlyphType::Sphere => self.glyph_sphere_mesh.as_ref(),
                crate::renderer::GlyphType::Cube => self.glyph_cube_mesh.as_ref(),
            }
            .expect("glyph mesh should have been created by ensure_glyph_mesh");

            let vbuf: &'static wgpu::Buffer = unsafe { &*(&mesh.vertex_buffer as *const _) };
            let ibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.index_buffer as *const _) };
            (vbuf, ibuf, mesh.index_count)
        };

        let mags: Vec<f32> = item
            .vectors
            .iter()
            .map(|v| glam::Vec3::from(*v).length())
            .collect();

        let (scalar_min, scalar_max) = if !item.scalars.is_empty() {
            item.scalar_range.unwrap_or_else(|| {
                let min = item.scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = item
                    .scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
        } else {
            item.scalar_range.unwrap_or_else(|| {
                let min = mags.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = mags.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            })
        };

        let (mag_clamp_min, mag_clamp_max, has_mag_clamp) = item
            .magnitude_clamp
            .map(|(mn, mx)| (mn, mx, 1u32))
            .unwrap_or((0.0, 1.0, 0u32));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GlyphInstance {
            position: [f32; 3],
            _pad0: f32,
            direction: [f32; 3],
            scalar: f32,
        }

        let instances: Vec<GlyphInstance> = (0..item.positions.len())
            .map(|i| GlyphInstance {
                position: item.positions[i],
                _pad0: 0.0,
                direction: item.vectors.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                scalar: item
                    .scalars
                    .get(i)
                    .copied()
                    .unwrap_or(mags.get(i).copied().unwrap_or(0.0)),
            })
            .collect();

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_instance_buf"),
            size: (std::mem::size_of::<GlyphInstance>() * instances.len()).max(32) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct GlyphUniform {
            global_scale: f32,
            scale_by_magnitude: u32,
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            mag_clamp_min: f32,
            mag_clamp_max: f32,
            has_mag_clamp: u32,
            default_color: [f32; 4],
            use_default_color: u32,
            unlit: u32,
            _pad: [u32; 2],
        }
        let uniform_data = GlyphUniform {
            global_scale: item.scale,
            scale_by_magnitude: if item.scale_by_magnitude { 1 } else { 0 },
            has_scalars: if !item.scalars.is_empty() { 1 } else { 0 },
            scalar_min,
            scalar_max,
            mag_clamp_min,
            mag_clamp_max,
            has_mag_clamp,
            default_color: item.default_color,
            use_default_color: if item.default_color[3] > 0.0 && item.use_default_color {
                1
            } else {
                0
            },
            unlit: if item.unlit { 1 } else { 0 },
            _pad: [0; 2],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_uniform_buf"),
            size: std::mem::size_of::<GlyphUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = self
            .builtin_colormap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colormap_id
                    .unwrap_or(ids[crate::resources::BuiltinColormap::Viridis as usize]);
                self.colormap_views.get(preset_id.0)
            })
            .unwrap_or(&self.fallback_lut_view);

        let lut_sampler = &self.material_sampler;

        let bgl1 = self
            .glyph_bgl
            .as_ref()
            .expect("ensure_glyph_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glyph_uniform_bg"),
            layout: bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        let bgl2 = self
            .glyph_instance_bgl
            .as_ref()
            .expect("ensure_glyph_pipeline not called");
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glyph_instance_bg"),
            layout: bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        GlyphGpuData {
            mesh_vertex_buffer: mesh_vbuf,
            mesh_index_buffer: mesh_ibuf,
            mesh_index_count: mesh_idx_count,
            instance_count,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Ensure a glyph base mesh is cached for the given [`GlyphType`].
    /// Creates and uploads the mesh on first call for that type.
    fn ensure_glyph_mesh(&mut self, device: &wgpu::Device, glyph_type: crate::renderer::GlyphType) {
        use crate::renderer::GlyphType;

        let already_cached = match glyph_type {
            GlyphType::Arrow => self.glyph_arrow_mesh.is_some(),
            GlyphType::Sphere => self.glyph_sphere_mesh.is_some(),
            GlyphType::Cube => self.glyph_cube_mesh.is_some(),
        };
        if already_cached {
            return;
        }

        let (verts, indices) = match glyph_type {
            GlyphType::Arrow => build_glyph_arrow(),
            GlyphType::Sphere => build_glyph_sphere(),
            GlyphType::Cube => build_unit_cube(),
        };

        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_mesh_vbuf"),
            size: (std::mem::size_of::<Vertex>() * verts.len()).max(64) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        vbuf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&verts));
        vbuf.unmap();

        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("glyph_mesh_ibuf"),
            size: (std::mem::size_of::<u32>() * indices.len()).max(12) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        ibuf.slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&indices));
        ibuf.unmap();

        let mesh = GlyphBaseMesh {
            vertex_buffer: vbuf,
            index_buffer: ibuf,
            index_count: indices.len() as u32,
        };

        match glyph_type {
            GlyphType::Arrow => self.glyph_arrow_mesh = Some(mesh),
            GlyphType::Sphere => self.glyph_sphere_mesh = Some(mesh),
            GlyphType::Cube => self.glyph_cube_mesh = Some(mesh),
        }
    }

    /// Lazily create the tensor glyph render pipeline (instanced ellipsoids, Phase 5).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.tensor_glyphs`
    /// is non-empty.
    pub(crate) fn ensure_tensor_glyph_pipeline(&mut self, device: &wgpu::Device) {
        if self.tensor_glyph_pipeline.is_some() {
            return;
        }

        let tg_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor_glyph_bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let tg_instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tensor_glyph_instance_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor_glyph_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/tensor_glyph.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tensor_glyph_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &tg_bgl, &tg_instance_bgl],
            push_constant_ranges: &[],
        });

        let sample_count = self.sample_count;
        let make = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("tensor_glyph_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
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
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        self.tensor_glyph_bgl = Some(tg_bgl);
        self.tensor_glyph_instance_bgl = Some(tg_instance_bgl);
        self.tensor_glyph_pipeline = Some(DualPipeline {
            ldr: make(self.target_format),
            hdr: make(wgpu::TextureFormat::Rgba16Float),
        });
    }

    /// Upload one [`TensorGlyphItem`] to the GPU and return draw data (Phase 5).
    ///
    /// Called from `prepare()` for each non-empty item in `frame.scene.tensor_glyphs`.
    /// Reuses the sphere base mesh cached by the glyph pipeline.
    pub(crate) fn upload_tensor_glyph_set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::TensorGlyphItem,
    ) -> TensorGlyphGpuData {
        use crate::renderer::GlyphType;

        let instance_count = item.positions.len() as u32;

        // Reuse the cached sphere mesh from the glyph pipeline.
        self.ensure_glyph_mesh(device, GlyphType::Sphere);
        let (mesh_vbuf, mesh_ibuf, mesh_idx_count) = {
            let mesh = self
                .glyph_sphere_mesh
                .as_ref()
                .expect("sphere mesh should be present after ensure_glyph_mesh");
            let vbuf: &'static wgpu::Buffer = unsafe { &*(&mesh.vertex_buffer as *const _) };
            let ibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.index_buffer as *const _) };
            (vbuf, ibuf, mesh.index_count)
        };

        // Pre-compute per-instance model and normal matrices on the CPU.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TensorInstance {
            model_col0: [f32; 4],
            model_col1: [f32; 4],
            model_col2: [f32; 4],
            model_col3: [f32; 4],
            normal_col0: [f32; 4],
            normal_col1: [f32; 4],
            normal_col2: [f32; 4],
            scalar: f32,
            _pad: [f32; 3],
        }

        let outer_model = glam::Mat4::from_cols_array_2d(&item.model);

        // Determine scalars for LUT lookup.
        let has_scalars = item.color_attribute.is_some();
        let (scalar_min, scalar_max) = if let Some(ref scalars) = item.color_attribute {
            item.scalar_range.unwrap_or_else(|| {
                let mn = scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (mn, mx)
            })
        } else {
            // Sign coloring: map [-1, 1] so LUT midpoint = neutral.
            item.scalar_range.unwrap_or((-1.0, 1.0))
        };

        let instances: Vec<TensorInstance> = (0..item.positions.len())
            .map(|i| {
                let pos = glam::Vec3::from(item.positions[i]);
                let ev = if i < item.eigenvalues.len() {
                    item.eigenvalues[i]
                } else {
                    [1.0, 1.0, 1.0]
                };
                let vecs = if i < item.eigenvectors.len() {
                    item.eigenvectors[i]
                } else {
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                };

                // Scale by |eigenvalue| * global_scale, minimum 1e-6 to avoid degenerate.
                let s0 = (ev[0].abs() * item.scale).max(1e-6_f32);
                let s1 = (ev[1].abs() * item.scale).max(1e-6_f32);
                let s2 = (ev[2].abs() * item.scale).max(1e-6_f32);

                // Rotation matrix: columns are the eigenvectors.
                let col0 = glam::Vec3::from(vecs[0]);
                let col1 = glam::Vec3::from(vecs[1]);
                let col2 = glam::Vec3::from(vecs[2]);

                // Rotation-scale block: RS = R * diag(s0, s1, s2).
                let rs = glam::Mat3::from_cols(col0 * s0, col1 * s1, col2 * s2);

                // 4x4 model matrix.
                let local_model = glam::Mat4::from_mat3(rs) * glam::Mat4::IDENTITY;
                let mut m4 = local_model;
                m4.w_axis = glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
                let world_model = outer_model * m4;

                // Normal matrix: R * diag(1/s0, 1/s1, 1/s2).
                let nm = glam::Mat3::from_cols(col0 / s0, col1 / s1, col2 / s2);

                // Scalar for LUT.
                let scalar = if has_scalars {
                    item.color_attribute
                        .as_ref()
                        .and_then(|sc| sc.get(i))
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    // Sign of dominant eigenvalue.
                    if i < item.eigenvalues.len() {
                        item.eigenvalues[i][0]
                    } else {
                        0.0
                    }
                };

                let mc = world_model.to_cols_array_2d();
                TensorInstance {
                    model_col0: mc[0],
                    model_col1: mc[1],
                    model_col2: mc[2],
                    model_col3: mc[3],
                    normal_col0: [nm.x_axis.x, nm.x_axis.y, nm.x_axis.z, 0.0],
                    normal_col1: [nm.y_axis.x, nm.y_axis.y, nm.y_axis.z, 0.0],
                    normal_col2: [nm.z_axis.x, nm.z_axis.y, nm.z_axis.z, 0.0],
                    scalar,
                    _pad: [0.0; 3],
                }
            })
            .collect();

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor_glyph_instance_buf"),
            size: (std::mem::size_of::<TensorInstance>() * instances.len()).max(128) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TensorGlyphUniform {
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            _pad0: f32,
            _pad1: [[f32; 4]; 3],
        }
        let uniform_data = TensorGlyphUniform {
            has_scalars: if has_scalars { 1 } else { 0 },
            scalar_min,
            scalar_max,
            _pad0: 0.0,
            _pad1: [[0.0; 4]; 3],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tensor_glyph_uniform_buf"),
            size: std::mem::size_of::<TensorGlyphUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = self
            .builtin_colormap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colormap_id
                    .unwrap_or(ids[crate::resources::BuiltinColormap::Viridis as usize]);
                self.colormap_views.get(preset_id.0)
            })
            .unwrap_or(&self.fallback_lut_view);

        let lut_sampler = &self.material_sampler;

        let bgl1 = self
            .tensor_glyph_bgl
            .as_ref()
            .expect("ensure_tensor_glyph_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tensor_glyph_uniform_bg"),
            layout: bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        let bgl2 = self
            .tensor_glyph_instance_bgl
            .as_ref()
            .expect("ensure_tensor_glyph_pipeline not called");
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tensor_glyph_instance_bg"),
            layout: bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        TensorGlyphGpuData {
            mesh_vertex_buffer: mesh_vbuf,
            mesh_index_buffer: mesh_ibuf,
            mesh_index_count: mesh_idx_count,
            instance_count,
            uniform_bind_group,
            instance_bind_group,
            _uniform_buf: uniform_buf,
            _instance_buf: instance_buf,
        }
    }

    /// Lazily create the glyph outline mask pipeline.
    ///
    /// Renders the instanced glyph mesh into the R8 outline mask texture so
    /// outlines follow the actual arrow/sphere shape.  Reuses the bind group
    /// layouts from the main glyph pipeline (must be called after
    /// `ensure_glyph_pipeline`).
    pub(crate) fn ensure_glyph_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.glyph_outline_mask_pipeline.is_some() {
            return;
        }
        let glyph_bgl = self
            .glyph_bgl
            .as_ref()
            .expect("ensure_glyph_pipeline must be called first");
        let glyph_instance_bgl = self
            .glyph_instance_bgl
            .as_ref()
            .expect("ensure_glyph_pipeline must be called first");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("glyph_outline_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/glyph_outline_mask.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glyph_outline_mask_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                glyph_bgl,
                glyph_instance_bgl,
            ],
            push_constant_ranges: &[],
        });

        self.glyph_outline_mask_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("glyph_outline_mask_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
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
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            },
        ));
    }

    /// Lazily create the tensor glyph outline mask pipeline.
    ///
    /// Same idea as `ensure_glyph_outline_mask_pipeline` but for tensor
    /// glyph ellipsoids.  Must be called after `ensure_tensor_glyph_pipeline`.
    pub(crate) fn ensure_tensor_glyph_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.tensor_glyph_outline_mask_pipeline.is_some() {
            return;
        }
        let tg_bgl = self
            .tensor_glyph_bgl
            .as_ref()
            .expect("ensure_tensor_glyph_pipeline must be called first");
        let tg_instance_bgl = self
            .tensor_glyph_instance_bgl
            .as_ref()
            .expect("ensure_tensor_glyph_pipeline must be called first");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tensor_glyph_outline_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/tensor_glyph_outline_mask.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tensor_glyph_outline_mask_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, tg_bgl, tg_instance_bgl],
            push_constant_ranges: &[],
        });

        self.tensor_glyph_outline_mask_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("tensor_glyph_outline_mask_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R8Unorm,
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
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            },
        ));
    }
}
