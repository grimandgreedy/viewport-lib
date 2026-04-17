use super::*;

impl ViewportGpuResources {
    /// Lazily create the point cloud render pipeline (PointList topology).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.point_clouds` is non-empty.
    pub(crate) fn ensure_point_cloud_pipeline(&mut self, device: &wgpu::Device) {
        if self.point_cloud_pipeline.is_some() {
            return;
        }

        let pc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point_cloud_bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("point_cloud_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/point_cloud.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("point_cloud_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &pc_bgl],
            push_constant_ranges: &[],
        });

        let pc_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point_cloud_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[pc_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        self.point_cloud_bgl = Some(pc_bgl);
        self.point_cloud_pipeline = Some(pipeline);
    }

    /// Upload one [`PointCloudItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.point_clouds`.
    pub(crate) fn upload_point_cloud(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::PointCloudItem,
    ) -> PointCloudGpuData {
        let point_count = item.positions.len() as u32;

        let pos_bytes: Vec<u8> = item
            .positions
            .iter()
            .flat_map(|p| bytemuck::bytes_of(p).iter().copied())
            .collect();
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_vertex_buf"),
            size: pos_bytes.len().max(12) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &pos_bytes);

        let (scalar_buf, has_scalars, scalar_min, scalar_max) = if !item.scalars.is_empty() {
            let min = item
                .scalar_range
                .map(|r| r.0)
                .unwrap_or_else(|| item.scalars.iter().cloned().fold(f32::INFINITY, f32::min));
            let max = item.scalar_range.map(|r| r.1).unwrap_or_else(|| {
                item.scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            });
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_scalar_buf"),
                size: (std::mem::size_of::<f32>() * item.scalars.len()).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.scalars));
            (buf, 1u32, min, max)
        } else {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_scalar_buf_fallback"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32, 0.0f32, 1.0f32)
        };

        let (color_buf, has_colors) = if !item.colors.is_empty() && has_scalars == 0 {
            let bytes: &[u8] = bytemuck::cast_slice(&item.colors);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_color_buf"),
                size: bytes.len().max(16) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytes);
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_color_buf_fallback"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PointCloudUniform {
            model: [[f32; 4]; 4],
            default_color: [f32; 4],
            point_size: f32,
            has_scalars: u32,
            scalar_min: f32,
            scalar_max: f32,
            has_colors: u32,
            _pad: [u32; 3],
        }
        let uniform_data = PointCloudUniform {
            model: item.model,
            default_color: item.default_color,
            point_size: item.point_size,
            has_scalars,
            scalar_min,
            scalar_max,
            has_colors,
            _pad: [0; 3],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_uniform_buf"),
            size: std::mem::size_of::<PointCloudUniform>() as u64,
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

        let bgl = self
            .point_cloud_bgl
            .as_ref()
            .expect("ensure_point_cloud_pipeline not called");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pc_bind_group"),
            layout: bgl,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scalar_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: color_buf.as_entire_binding(),
                },
            ],
        });

        PointCloudGpuData {
            vertex_buffer,
            point_count,
            bind_group,
            _uniform_buf: uniform_buf,
            _scalar_buf: scalar_buf,
            _color_buf: color_buf,
        }
    }

    /// Lazily create the polyline render pipeline (LineStrip topology).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.polylines` is non-empty.
    pub(crate) fn ensure_polyline_pipeline(&mut self, device: &wgpu::Device) {
        if self.polyline_pipeline.is_some() {
            return;
        }

        let pl_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("polyline_bgl"),
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polyline_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/polyline.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("polyline_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &pl_bgl],
            push_constant_ranges: &[],
        });

        let pl_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("polyline_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[pl_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                strip_index_format: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
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

        self.polyline_bgl = Some(pl_bgl);
        self.polyline_pipeline = Some(pipeline);
    }

    /// Upload one [`PolylineItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.polylines`.
    pub(crate) fn upload_polyline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::PolylineItem,
    ) -> PolylineGpuData {
        let vertex_count = item.positions.len() as u32;

        let mut raw_verts: Vec<f32> = Vec::with_capacity(item.positions.len() * 4);
        for (i, pos) in item.positions.iter().enumerate() {
            raw_verts.push(pos[0]);
            raw_verts.push(pos[1]);
            raw_verts.push(pos[2]);
            raw_verts.push(item.scalars.get(i).copied().unwrap_or(0.0));
        }
        let vert_bytes: &[u8] = bytemuck::cast_slice(&raw_verts);
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polyline_vertex_buf"),
            size: vert_bytes.len().max(16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, vert_bytes);

        let mut strip_ranges: Vec<std::ops::Range<u32>> = Vec::new();
        let mut offset: u32 = 0;
        for &len in &item.strip_lengths {
            if len >= 2 {
                strip_ranges.push(offset..offset + len);
            }
            offset += len;
        }
        if item.strip_lengths.is_empty() && vertex_count >= 2 {
            strip_ranges.push(0..vertex_count);
        }

        let (has_scalar, scalar_min, scalar_max) = if !item.scalars.is_empty() {
            let (min, max) = item.scalar_range.unwrap_or_else(|| {
                let mn = item.scalars.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = item
                    .scalars
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                (mn, mx)
            });
            (1u32, min, max)
        } else {
            (0u32, 0.0f32, 1.0f32)
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PolylineUniform {
            default_color: [f32; 4],
            line_width: f32,
            scalar_min: f32,
            scalar_max: f32,
            has_scalar: u32,
            _pad: [f32; 4],
        }
        let uniform_data = PolylineUniform {
            default_color: item.default_color,
            line_width: item.line_width,
            scalar_min,
            scalar_max,
            has_scalar,
            _pad: [0.0; 4],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polyline_uniform_buf"),
            size: std::mem::size_of::<PolylineUniform>() as u64,
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

        let bgl = self
            .polyline_bgl
            .as_ref()
            .expect("ensure_polyline_pipeline not called");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("polyline_bind_group"),
            layout: bgl,
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

        PolylineGpuData {
            vertex_buffer,
            vertex_count,
            strip_ranges,
            bind_group,
            _uniform_buf: uniform_buf,
        }
    }

    /// Lazily create the glyph render pipeline (instanced TriangleList).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.glyphs` is non-empty.
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/glyph.wgsl").into()),
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: self.target_format,
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
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.glyph_bgl = Some(glyph_bgl);
        self.glyph_instance_bgl = Some(glyph_instance_bgl);
        self.glyph_pipeline = Some(pipeline);
    }

    /// Upload one [`GlyphItem`] to the GPU and return draw data.
    ///
    /// Called from `prepare()` for each non-empty item in `frame.glyphs`.
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
            _pad: [[f32; 4]; 3],
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
            _pad: [[0.0; 4]; 3],
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

    /// Lazily create the streamtube render pipeline (instanced cylinder TriangleList).
    ///
    /// No-op if already created.  Called from `prepare()` when `frame.streamtube_items`
    /// is non-empty.
    pub(crate) fn ensure_streamtube_pipeline(&mut self, device: &wgpu::Device) {
        if self.streamtube_pipeline.is_some() {
            return;
        }

        let streamtube_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("streamtube_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let streamtube_instance_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("streamtube_instance_bgl"),
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
            label: Some("streamtube_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/streamtube.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("streamtube_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &streamtube_bgl,
                &streamtube_instance_bgl,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("streamtube_pipeline"),
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
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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

        self.streamtube_bgl = Some(streamtube_bgl);
        self.streamtube_instance_bgl = Some(streamtube_instance_bgl);
        self.streamtube_pipeline = Some(pipeline);
    }

    /// Upload one [`StreamtubeItem`] to the GPU and return draw data.
    ///
    /// Converts each consecutive point pair in each strip into one cylinder instance.
    /// The cylinder base mesh is cached in `streamtube_cylinder_mesh` and created on
    /// first call.  Returns a [`StreamtubeGpuData`] with `instance_count = 0` when
    /// the item has no renderable segments.
    pub(crate) fn upload_streamtube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::StreamtubeItem,
    ) -> StreamtubeGpuData {
        if self.streamtube_cylinder_mesh.is_none() {
            let (verts, indices) = build_streamtube_cylinder();

            let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("streamtube_cylinder_vbuf"),
                size: (std::mem::size_of::<Vertex>() * verts.len()).max(64) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            vbuf.slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&verts));
            vbuf.unmap();

            let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("streamtube_cylinder_ibuf"),
                size: (std::mem::size_of::<u32>() * indices.len()).max(12) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            ibuf.slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&indices));
            ibuf.unmap();

            self.streamtube_cylinder_mesh = Some(GlyphBaseMesh {
                vertex_buffer: vbuf,
                index_buffer: ibuf,
                index_count: indices.len() as u32,
            });
        }

        let (mesh_vbuf, mesh_ibuf, mesh_idx_count) = {
            let mesh = self
                .streamtube_cylinder_mesh
                .as_ref()
                .expect("streamtube cylinder mesh created above");
            let vbuf: &'static wgpu::Buffer = unsafe { &*(&mesh.vertex_buffer as *const _) };
            let ibuf: &'static wgpu::Buffer = unsafe { &*(&mesh.index_buffer as *const _) };
            (vbuf, ibuf, mesh.index_count)
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StreamtubeInstance {
            position: [f32; 3],
            half_len: f32,
            direction: [f32; 3],
            _pad: f32,
        }

        let mut instances: Vec<StreamtubeInstance> = Vec::new();
        let positions = &item.positions;
        let mut strip_start = 0usize;
        for &len in &item.strip_lengths {
            let len = len as usize;
            let strip_end = (strip_start + len).min(positions.len());
            for i in strip_start..strip_end.saturating_sub(1) {
                let a = glam::Vec3::from(positions[i]);
                let b = glam::Vec3::from(positions[i + 1]);
                let seg = b - a;
                let seg_len = seg.length();
                if seg_len < f32::EPSILON {
                    continue;
                }
                instances.push(StreamtubeInstance {
                    position: ((a + b) * 0.5).to_array(),
                    half_len: seg_len * 0.5,
                    direction: (seg / seg_len).to_array(),
                    _pad: 0.0,
                });
            }
            strip_start += len;
        }

        let instance_count = instances.len() as u32;

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streamtube_instance_buf"),
            size: (std::mem::size_of::<StreamtubeInstance>() * instances.len().max(1)) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !instances.is_empty() {
            queue.write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StreamtubeUniform {
            color: [f32; 4],
            radius: f32,
            _pad: [f32; 7],
        }
        let uniform_data = StreamtubeUniform {
            color: item.color,
            radius: item.radius.max(f32::EPSILON),
            _pad: [0.0; 7],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streamtube_uniform_buf"),
            size: std::mem::size_of::<StreamtubeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl1 = self
            .streamtube_bgl
            .as_ref()
            .expect("ensure_streamtube_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("streamtube_uniform_bg"),
            layout: bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let bgl2 = self
            .streamtube_instance_bgl
            .as_ref()
            .expect("ensure_streamtube_pipeline not called");
        let instance_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("streamtube_instance_bg"),
            layout: bgl2,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buf.as_entire_binding(),
            }],
        });

        StreamtubeGpuData {
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
}
