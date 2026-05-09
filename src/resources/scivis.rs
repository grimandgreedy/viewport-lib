use super::*;

impl ViewportGpuResources {
    /// Lazily create the point cloud render pipeline (PointList topology).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.point_clouds` is non-empty.
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
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
    /// Called from `prepare()` for each non-empty item in `frame.scene.point_clouds`.
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

        // Radius buffer: radius_scalars (mapped to radius_range) take priority over
        // explicit per-point radii.
        let (radius_buf, has_radius) = if !item.radius_scalars.is_empty() {
            let r_min = item
                .radius_scalar_range
                .map(|r| r.0)
                .unwrap_or_else(|| {
                    item.radius_scalars.iter().cloned().fold(f32::INFINITY, f32::min)
                });
            let r_max = item
                .radius_scalar_range
                .map(|r| r.1)
                .unwrap_or_else(|| {
                    item.radius_scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                });
            let range = (r_max - r_min).max(f32::EPSILON);
            let (out_min, out_max) = item.radius_range;
            let mapped: Vec<f32> = item
                .radius_scalars
                .iter()
                .map(|&s| {
                    let t = ((s - r_min) / range).clamp(0.0, 1.0);
                    out_min + t * (out_max - out_min)
                })
                .collect();
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_radius_buf"),
                size: (std::mem::size_of::<f32>() * mapped.len()).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&mapped));
            (buf, 1u32)
        } else if !item.radii.is_empty() {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_radius_buf"),
                size: (std::mem::size_of::<f32>() * item.radii.len()).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.radii));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_radius_buf_fallback"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (buf, 0u32)
        };

        let (transparency_buf, has_transparency) = if !item.transparencies.is_empty() {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_transparency_buf"),
                size: (std::mem::size_of::<f32>() * item.transparencies.len()).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&item.transparencies));
            (buf, 1u32)
        } else {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pc_transparency_buf_fallback"),
                size: 4,
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
            has_radius: u32,
            has_transparency: u32,
            gaussian: u32,
        }
        let uniform_data = PointCloudUniform {
            model: item.model,
            default_color: item.default_color,
            point_size: item.point_size,
            has_scalars,
            scalar_min,
            scalar_max,
            has_colors,
            has_radius,
            has_transparency,
            gaussian: if item.gaussian { 1 } else { 0 },
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: radius_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: transparency_buf.as_entire_binding(),
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
            _radius_buf: radius_buf,
            _transparency_buf: transparency_buf,
        }
    }

    /// Lazily create the polyline render pipeline (instanced TriangleList : screen-space thick lines).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.polylines` is non-empty.
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

        // Instance buffer layout (112 bytes per segment):
        //   offset   0: pos_a             vec3  : segment start (world space)
        //   offset  12: pos_b             vec3  : segment end   (world space)
        //   offset  24: prev_pos          vec3  : point before pos_a (for miter at A); equals pos_a if strip start
        //   offset  36: next_pos          vec3  : point after  pos_b (for miter at B); equals pos_b if strip end
        //   offset  48: scalar_a          f32
        //   offset  52: scalar_b          f32
        //   offset  56: has_prev          u32   : 1 = prev_pos is valid (interior join at A), 0 = square cap
        //   offset  60: has_next          u32   : 1 = next_pos is valid (interior join at B), 0 = square cap
        //   offset  64: color_a           vec4  : direct RGBA at segment start
        //   offset  80: color_b           vec4  : direct RGBA at segment end
        //   offset  96: radius_a          f32   : line width in px at A (= line_width when node_radii is empty)
        //   offset 100: radius_b          f32   : line width in px at B
        //   offset 104: use_direct_color  u32   : 1 = use color_a/b, 0 = use scalar LUT / default
        //   offset 108: _pad              u32
        let pl_instance_layout = wgpu::VertexBufferLayout {
            array_stride: 112,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                }, // pos_a
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }, // pos_b
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                }, // prev_pos
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                }, // next_pos
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                }, // scalar_a
                wgpu::VertexAttribute {
                    offset: 52,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                }, // scalar_b
                wgpu::VertexAttribute {
                    offset: 56,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Uint32,
                }, // has_prev
                wgpu::VertexAttribute {
                    offset: 60,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Uint32,
                }, // has_next
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                }, // color_a
                wgpu::VertexAttribute {
                    offset: 80,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                }, // color_b
                wgpu::VertexAttribute {
                    offset: 96,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32,
                }, // radius_a
                wgpu::VertexAttribute {
                    offset: 100,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32,
                }, // radius_b
                wgpu::VertexAttribute {
                    offset: 104,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Uint32,
                }, // use_direct_color
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("polyline_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[pl_instance_layout],
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
    /// Converts the strip-based point list into a flat segment-instance buffer
    /// suitable for the screen-space thick-line pipeline with miter joints.
    ///
    /// Each consecutive pair of points in a strip becomes one 112-byte instance
    /// containing miter geometry, scalar coloring, direct RGBA colors, and per-vertex
    /// radii. See the comment in `ensure_polyline_pipeline` for the full layout.
    ///
    /// `viewport_size` is `[width_px, height_px]` and is baked into the per-item
    /// uniform so the vertex shader can compute correct pixel offsets.
    pub(crate) fn upload_polyline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::PolylineItem,
        viewport_size: [f32; 2],
    ) -> PolylineGpuData {
        // Build the segment instance buffer (112 bytes per segment).
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SegInstance {
            pos_a: [f32; 3],         // offset   0
            pos_b: [f32; 3],         // offset  12
            prev_pos: [f32; 3],      // offset  24
            next_pos: [f32; 3],      // offset  36
            scalar_a: f32,           // offset  48
            scalar_b: f32,           // offset  52
            has_prev: u32,           // offset  56
            has_next: u32,           // offset  60
            color_a: [f32; 4],       // offset  64
            color_b: [f32; 4],       // offset  80
            radius_a: f32,           // offset  96
            radius_b: f32,           // offset 100
            use_direct_color: u32,   // offset 104
            _pad: u32,               // offset 108
        }

        // Determine which color/scalar/radius source to use per segment.
        let use_direct = !item.node_colors.is_empty() || !item.edge_colors.is_empty();
        let use_edge_scalars = item.scalars.is_empty() && !item.edge_scalars.is_empty();
        let use_node_radii = !item.node_radii.is_empty();

        let mut instances: Vec<SegInstance> = Vec::new();
        let positions = &item.positions;
        let npos = positions.len();

        // Collect strip ranges: (start_idx, end_idx) into `positions`.
        let strip_ranges: Vec<(usize, usize)> = if item.strip_lengths.is_empty() {
            vec![(0, npos)]
        } else {
            let mut ranges = Vec::with_capacity(item.strip_lengths.len());
            let mut off = 0usize;
            for &l in &item.strip_lengths {
                ranges.push((off, off + l as usize));
                off += l as usize;
            }
            ranges
        };

        let mut seg_idx_global: usize = 0; // monotonic segment counter across all strips

        for &(strip_start, strip_end) in &strip_ranges {
            let end = strip_end.min(npos);
            for i in strip_start..end.saturating_sub(1) {
                let j = i + 1;
                let has_prev = i > strip_start;
                let has_next = j + 1 < end;

                // Scalar: edge_scalars (flat per segment) > per-node scalars > 0
                let (scalar_a, scalar_b) = if use_edge_scalars {
                    let s = item.edge_scalars.get(seg_idx_global).copied().unwrap_or(0.0);
                    (s, s)
                } else {
                    (
                        item.scalars.get(i).copied().unwrap_or(0.0),
                        item.scalars.get(j).copied().unwrap_or(0.0),
                    )
                };

                // Direct color: node_colors (per-endpoint) > edge_colors (per-segment)
                let (color_a, color_b) = if !item.node_colors.is_empty() {
                    (
                        item.node_colors.get(i).copied().unwrap_or([1.0; 4]),
                        item.node_colors.get(j).copied().unwrap_or([1.0; 4]),
                    )
                } else if !item.edge_colors.is_empty() {
                    let c = item.edge_colors.get(seg_idx_global).copied().unwrap_or([1.0; 4]);
                    (c, c)
                } else {
                    ([1.0; 4], [1.0; 4])
                };

                // Radius: per-node > global line_width
                let (radius_a, radius_b) = if use_node_radii {
                    (
                        item.node_radii.get(i).copied().unwrap_or(item.line_width),
                        item.node_radii.get(j).copied().unwrap_or(item.line_width),
                    )
                } else {
                    (item.line_width, item.line_width)
                };

                instances.push(SegInstance {
                    pos_a: positions[i],
                    pos_b: positions[j],
                    prev_pos: if has_prev { positions[i - 1] } else { positions[i] },
                    next_pos: if has_next { positions[j + 1] } else { positions[j] },
                    scalar_a,
                    scalar_b,
                    has_prev: has_prev as u32,
                    has_next: has_next as u32,
                    color_a,
                    color_b,
                    radius_a,
                    radius_b,
                    use_direct_color: use_direct as u32,
                    _pad: 0,
                });

                seg_idx_global += 1;
            }
        }

        let seg_count = instances.len() as u32;

        // Allocate instance buffer (min 112 bytes so wgpu doesn't complain on empty).
        let seg_bytes: &[u8] = bytemuck::cast_slice(&instances);
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("polyline_vertex_buf"),
            size: seg_bytes.len().max(112) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !seg_bytes.is_empty() {
            queue.write_buffer(&vertex_buffer, 0, seg_bytes);
        }

        // Determine scalar range for the LUT uniform (node or edge scalars).
        let scalar_source: &[f32] = if !item.scalars.is_empty() {
            &item.scalars
        } else {
            &item.edge_scalars
        };
        let (has_scalar, scalar_min, scalar_max) = if !scalar_source.is_empty() {
            let (min, max) = item.scalar_range.unwrap_or_else(|| {
                let mn = scalar_source.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = scalar_source.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (mn, mx)
            });
            (1u32, min, max)
        } else {
            (0u32, 0.0f32, 1.0f32)
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PolylineUniform {
            default_color: [f32; 4], // offset  0
            line_width: f32,         // offset 16
            scalar_min: f32,         // offset 20
            scalar_max: f32,         // offset 24
            has_scalar: u32,         // offset 28
            viewport_width: f32,     // offset 32
            viewport_height: f32,    // offset 36
            _pad: [f32; 2],          // offset 40  (total 48 bytes)
        }
        let uniform_data = PolylineUniform {
            default_color: item.default_color,
            line_width: item.line_width,
            scalar_min,
            scalar_max,
            has_scalar,
            viewport_width: viewport_size[0].max(1.0),
            viewport_height: viewport_size[1].max(1.0),
            _pad: [0.0; 2],
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
            segment_count: seg_count,
            bind_group,
            _uniform_buf: uniform_buf,
        }
    }

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
            _pad: [u32; 3],
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
            use_default_color: if item.default_color[3] > 0.0 && item.use_default_color { 1 } else { 0 },
            _pad: [0; 3],
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

        let tg_instance_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                include_str!("../shaders/tensor_glyph.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tensor_glyph_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &tg_bgl,
                &tg_instance_bgl,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        self.tensor_glyph_bgl = Some(tg_bgl);
        self.tensor_glyph_instance_bgl = Some(tg_instance_bgl);
        self.tensor_glyph_pipeline = Some(pipeline);
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
            model_col0:  [f32; 4],
            model_col1:  [f32; 4],
            model_col2:  [f32; 4],
            model_col3:  [f32; 4],
            normal_col0: [f32; 4],
            normal_col1: [f32; 4],
            normal_col2: [f32; 4],
            scalar:      f32,
            _pad:        [f32; 3],
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
                let local_model = glam::Mat4::from_mat3(rs)
                    * glam::Mat4::IDENTITY;
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
                    if i < item.eigenvalues.len() { item.eigenvalues[i][0] } else { 0.0 }
                };

                let mc = world_model.to_cols_array_2d();
                TensorInstance {
                    model_col0:  mc[0],
                    model_col1:  mc[1],
                    model_col2:  mc[2],
                    model_col3:  mc[3],
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
            scalar_min:  f32,
            scalar_max:  f32,
            _pad0:       f32,
            _pad1:       [[f32; 4]; 3],
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

    /// Lazily create the streamtube render pipeline (connected tube mesh, TriangleList).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.streamtube_items`
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("streamtube_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/streamtube.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("streamtube_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &streamtube_bgl],
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
        self.streamtube_pipeline = Some(pipeline);
    }

    /// Upload one [`StreamtubeItem`] to the GPU and return draw data.
    ///
    /// Generates a connected tube mesh CPU-side using a parallel-transport frame along
    /// each polyline strip, then uploads the result as a single owned vertex+index buffer.
    /// Adjacent rings are joined by quads (2 triangles each) giving a smooth, seamless tube
    /// without the z-fighting or inter-segment gaps that plagued the old instanced approach.
    pub(crate) fn upload_streamtube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::StreamtubeItem,
    ) -> StreamtubeGpuData {
        const SIDES: usize = 12; // tube cross-section resolution

        let radius = item.radius.max(f32::EPSILON);

        let mut verts: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let positions = &item.positions;
        let mut strip_start = 0usize;

        for &strip_len in &item.strip_lengths {
            let strip_len = strip_len as usize;
            let strip_end = (strip_start + strip_len).min(positions.len());
            let pts: Vec<glam::Vec3> = positions[strip_start..strip_end]
                .iter()
                .map(|&p| glam::Vec3::from(p))
                .collect();
            strip_start += strip_len;

            if pts.len() < 2 {
                continue;
            }

            // ---- Parallel transport frame ----------------------------------------
            // Seed: find an initial tangent and an arbitrary perpendicular.
            let t0 = (pts[1] - pts[0]).normalize_or_zero();
            if t0.length_squared() < 1e-10 {
                continue;
            }
            // Choose a reference vector not parallel to t0.
            let ref_v = if t0.x.abs() < 0.9 {
                glam::Vec3::X
            } else {
                glam::Vec3::Y
            };
            let mut u = t0.cross(ref_v).normalize(); // initial "up"

            // Emit rings for each point, transporting the frame forward.
            let ring_base = verts.len() as u32;
            let n_rings = pts.len();

            for (k, &pt) in pts.iter().enumerate() {
                // Tangent at this point (forward difference, except at the last point).
                let tangent = if k + 1 < pts.len() {
                    (pts[k + 1] - pt).normalize_or_zero()
                } else {
                    (pt - pts[k - 1]).normalize_or_zero()
                };

                // Transport u: project out the component along the new tangent.
                if k > 0 {
                    let t_prev = (pts[k] - pts[k - 1]).normalize_or_zero();
                    // Rodrigues rotation: rotate u by the same angle that t_prev -> tangent.
                    let axis = t_prev.cross(tangent);
                    let sin_a = axis.length().min(1.0);
                    if sin_a > 1e-6 {
                        let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                        let ax = axis / sin_a;
                        // Rodrigues: u' = u cos(a) + (ax×u) sin(a) + ax(ax·u)(1−cos(a))
                        u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                        u = u.normalize_or_zero();
                    }
                }

                let v = tangent.cross(u).normalize_or_zero();

                // Emit SIDES vertices around the ring.
                for s in 0..SIDES {
                    let theta = 2.0 * std::f32::consts::PI * (s as f32) / (SIDES as f32);
                    let nx = theta.cos() * u.x + theta.sin() * v.x;
                    let ny = theta.cos() * u.y + theta.sin() * v.y;
                    let nz = theta.cos() * u.z + theta.sin() * v.z;
                    let normal = glam::Vec3::new(nx, ny, nz);
                    let world_pos = pt + normal * radius;
                    verts.push(Vertex {
                        position: world_pos.to_array(),
                        normal: normal.to_array(),
                        color: [1.0, 1.0, 1.0, 1.0], // overridden by uniform in shader
                        uv: [0.0, 0.0],
                        tangent: [1.0, 0.0, 0.0, 1.0],
                    });
                }

                // Emit quad strip between ring k-1 and ring k.
                // Winding: outward-facing CCW (right-hand rule gives outward normal).
                // Verified: T1=(r0+s, r0+s1, r1+s) has normal·Y > 0 for s=0 on Z-axis tube.
                if k > 0 {
                    let r0 = ring_base + ((k - 1) * SIDES) as u32;
                    let r1 = ring_base + (k * SIDES) as u32;
                    for s in 0..SIDES {
                        let s1 = (s + 1) % SIDES;
                        indices.push(r0 + s as u32);
                        indices.push(r0 + s1 as u32);
                        indices.push(r1 + s as u32);

                        indices.push(r0 + s1 as u32);
                        indices.push(r1 + s1 as u32);
                        indices.push(r1 + s as u32);
                    }
                }
            }

            // End cap (flat fan at last ring, facing forward = outward at tube end).
            // CCW from the forward direction: (center, s, s1).
            {
                let last_ring = ring_base + ((n_rings - 1) * SIDES) as u32;
                let tangent = (pts[n_rings - 1] - pts[n_rings - 2]).normalize_or_zero();
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[n_rings - 1].to_array(),
                    normal: tangent.to_array(),
                    color: [1.0, 1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                for s in 0..SIDES {
                    let s1 = (s + 1) % SIDES;
                    indices.push(cap_center_idx);
                    indices.push(last_ring + s as u32);
                    indices.push(last_ring + s1 as u32);
                }
            }

            // Start cap (flat fan at first ring, facing backward = outward at tube start).
            // CCW from the backward direction = CW from forward = (center, s1, s).
            {
                let tangent = (pts[0] - pts[1]).normalize_or_zero();
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[0].to_array(),
                    normal: tangent.to_array(),
                    color: [1.0, 1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                for s in 0..SIDES {
                    let s1 = (s + 1) % SIDES;
                    indices.push(cap_center_idx);
                    indices.push(ring_base + s1 as u32);
                    indices.push(ring_base + s as u32);
                }
            }
        }

        // Upload vertex + index buffers.
        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streamtube_vbuf"),
            size: vert_bytes.len().max(std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !vert_bytes.is_empty() {
            queue.write_buffer(&vertex_buffer, 0, vert_bytes);
        }

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streamtube_ibuf"),
            size: idx_bytes.len().max(12) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !idx_bytes.is_empty() {
            queue.write_buffer(&index_buffer, 0, idx_bytes);
        }

        let index_count = indices.len() as u32;

        // Uniform buffer: color + radius + use_vertex_color flag.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StreamtubeUniform {
            color: [f32; 4],
            radius: f32,
            use_vertex_color: u32,
            _pad: [f32; 6],
        }
        let uniform_data = StreamtubeUniform {
            color: item.color,
            radius,
            use_vertex_color: 0,
            _pad: [0.0; 6],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streamtube_uniform_buf"),
            size: std::mem::size_of::<StreamtubeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .streamtube_bgl
            .as_ref()
            .expect("ensure_streamtube_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("streamtube_uniform_bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        StreamtubeGpuData {
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_bind_group,
            _uniform_buf: uniform_buf,
        }
    }

    // -------------------------------------------------------------------------
    // Phase 3.3 : General Tube representation
    // -------------------------------------------------------------------------

    /// Upload one [`TubeItem`] to the GPU and return draw data.
    ///
    /// Generates a connected tube mesh CPU-side using a parallel-transport frame.
    /// Scalar values are baked into per-vertex colors using the CPU-side colormap copy.
    /// Uses the same streamtube pipeline; sets `use_vertex_color=1` when scalars are present.
    pub(crate) fn upload_tube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::TubeItem,
    ) -> StreamtubeGpuData {
        let sides = (item.sides.max(3)) as usize;

        // Resolve scalar-to-color mapping upfront if scalars are provided.
        let (use_vertex_color, lut_rgba): (u32, Option<[[u8; 4]; 256]>) =
            if !item.scalars.is_empty() {
                let lut = self
                    .builtin_colormap_ids
                    .and_then(|ids| {
                        let preset_id = item
                            .colormap_id
                            .unwrap_or(ids[crate::resources::BuiltinColormap::Viridis as usize]);
                        self.colormaps_cpu.get(preset_id.0).copied()
                    })
                    .unwrap_or([[128u8; 4]; 256]);
                (1, Some(lut))
            } else {
                (0, None)
            };

        let scalar_min = item
            .scalar_range
            .map(|r| r.0)
            .unwrap_or_else(|| item.scalars.iter().cloned().fold(f32::INFINITY, f32::min));
        let scalar_max = item
            .scalar_range
            .map(|r| r.1)
            .unwrap_or_else(|| {
                item.scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            });
        let scalar_range = (scalar_max - scalar_min).max(f32::EPSILON);

        // Helper: map a scalar value to an RGBA f32 color from the LUT.
        let scalar_to_color = |idx: usize| -> [f32; 4] {
            if let Some(ref lut) = lut_rgba {
                let s = *item.scalars.get(idx).unwrap_or(&0.0);
                let t = ((s - scalar_min) / scalar_range).clamp(0.0, 1.0);
                let lut_idx = ((t * 255.0).round() as usize).min(255);
                let c = lut[lut_idx];
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]
            } else {
                item.color
            }
        };

        let mut verts: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let positions = &item.positions;
        let mut strip_start = 0usize;

        for &strip_len in &item.strip_lengths {
            let strip_len = strip_len as usize;
            let strip_end = (strip_start + strip_len).min(positions.len());
            let pts: Vec<glam::Vec3> = positions[strip_start..strip_end]
                .iter()
                .map(|&p| glam::Vec3::from(p))
                .collect();
            let pts_scalar_start = strip_start;
            strip_start += strip_len;

            if pts.len() < 2 {
                continue;
            }

            // Parallel transport frame (same as upload_streamtube).
            let t0 = (pts[1] - pts[0]).normalize_or_zero();
            if t0.length_squared() < 1e-10 {
                continue;
            }
            let ref_v = if t0.x.abs() < 0.9 {
                glam::Vec3::X
            } else {
                glam::Vec3::Y
            };
            let mut u = t0.cross(ref_v).normalize();

            let ring_base = verts.len() as u32;
            let n_rings = pts.len();

            for (k, &pt) in pts.iter().enumerate() {
                let tangent = if k + 1 < pts.len() {
                    (pts[k + 1] - pt).normalize_or_zero()
                } else {
                    (pt - pts[k - 1]).normalize_or_zero()
                };

                if k > 0 {
                    let t_prev = (pts[k] - pts[k - 1]).normalize_or_zero();
                    let axis = t_prev.cross(tangent);
                    let sin_a = axis.length().min(1.0);
                    if sin_a > 1e-6 {
                        let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                        let ax = axis / sin_a;
                        u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                        u = u.normalize_or_zero();
                    }
                }

                let v = tangent.cross(u).normalize_or_zero();

                // Per-point radius: from radius_attribute if provided, else uniform radius.
                let point_radius = item
                    .radius_attribute
                    .as_ref()
                    .and_then(|ra| ra.get(pts_scalar_start + k).copied())
                    .unwrap_or(item.radius)
                    .max(f32::EPSILON);

                let vertex_color = scalar_to_color(pts_scalar_start + k);

                for s in 0..sides {
                    let theta = 2.0 * std::f32::consts::PI * (s as f32) / (sides as f32);
                    let nx = theta.cos() * u.x + theta.sin() * v.x;
                    let ny = theta.cos() * u.y + theta.sin() * v.y;
                    let nz = theta.cos() * u.z + theta.sin() * v.z;
                    let normal = glam::Vec3::new(nx, ny, nz);
                    let world_pos = pt + normal * point_radius;
                    verts.push(Vertex {
                        position: world_pos.to_array(),
                        normal: normal.to_array(),
                        color: vertex_color,
                        uv: [0.0, 0.0],
                        tangent: [1.0, 0.0, 0.0, 1.0],
                    });
                }

                if k > 0 {
                    let r0 = ring_base + ((k - 1) * sides) as u32;
                    let r1 = ring_base + (k * sides) as u32;
                    for s in 0..sides {
                        let s1 = (s + 1) % sides;
                        indices.push(r0 + s as u32);
                        indices.push(r0 + s1 as u32);
                        indices.push(r1 + s as u32);

                        indices.push(r0 + s1 as u32);
                        indices.push(r1 + s1 as u32);
                        indices.push(r1 + s as u32);
                    }
                }
            }

            // End cap.
            {
                let last_ring = ring_base + ((n_rings - 1) * sides) as u32;
                let tangent = (pts[n_rings - 1] - pts[n_rings - 2]).normalize_or_zero();
                let cap_color = scalar_to_color(pts_scalar_start + n_rings - 1);
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[n_rings - 1].to_array(),
                    normal: tangent.to_array(),
                    color: cap_color,
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                for s in 0..sides {
                    let s1 = (s + 1) % sides;
                    indices.push(cap_center_idx);
                    indices.push(last_ring + s as u32);
                    indices.push(last_ring + s1 as u32);
                }
            }

            // Start cap.
            {
                let tangent = (pts[0] - pts[1]).normalize_or_zero();
                let cap_color = scalar_to_color(pts_scalar_start);
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[0].to_array(),
                    normal: tangent.to_array(),
                    color: cap_color,
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                for s in 0..sides {
                    let s1 = (s + 1) % sides;
                    indices.push(cap_center_idx);
                    indices.push(ring_base + s1 as u32);
                    indices.push(ring_base + s as u32);
                }
            }
        }

        // Upload vertex + index buffers.
        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tube_vbuf"),
            size: vert_bytes.len().max(std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !vert_bytes.is_empty() {
            queue.write_buffer(&vertex_buffer, 0, vert_bytes);
        }

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tube_ibuf"),
            size: idx_bytes.len().max(12) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !idx_bytes.is_empty() {
            queue.write_buffer(&index_buffer, 0, idx_bytes);
        }

        let index_count = indices.len() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TubeUniform {
            color: [f32; 4],
            radius: f32,
            use_vertex_color: u32,
            _pad: [f32; 6],
        }
        let uniform_data = TubeUniform {
            color: item.color,
            radius: item.radius.max(f32::EPSILON),
            use_vertex_color,
            _pad: [0.0; 6],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tube_uniform_buf"),
            size: std::mem::size_of::<TubeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .streamtube_bgl
            .as_ref()
            .expect("ensure_streamtube_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tube_uniform_bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        StreamtubeGpuData {
            vertex_buffer,
            index_buffer,
            index_count,
            uniform_bind_group,
            _uniform_buf: uniform_buf,
        }
    }

    // -------------------------------------------------------------------------
    // Phase 3.2 : 2D Image Slice representation
    // -------------------------------------------------------------------------

    /// Lazily create the image slice render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.image_slices` is non-empty.
    pub(crate) fn ensure_image_slice_pipeline(&mut self, device: &wgpu::Device) {
        if self.image_slice_pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image_slice_bgl"),
            entries: &[
                // binding 0: ImageSliceUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: texture_3d<f32> — R32Float is not filterable on most hardware,
                // so declare as non-filterable and use a NonFiltering sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                // binding 2: vol_sampler (non-filtering nearest — matches R32Float)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 3: lut_tex (colormap texture_2d)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 4: lut_sampler (linear)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("image_slice_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/image_slice.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("image_slice_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("image_slice_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // no vertex buffer: generates quad from vertex_index
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
                cull_mode: None,
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
                count: self.sample_count,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });

        self.image_slice_bgl = Some(bgl);
        self.image_slice_pipeline = Some(pipeline);
    }

    /// Upload one [`ImageSliceItem`] to the GPU and return draw data.
    ///
    /// Creates a uniform buffer describing the slice parameters and a bind group
    /// referencing the existing uploaded volume texture.  No vertex buffer is needed:
    /// the shader generates a quad from `vertex_index`.
    pub(crate) fn upload_image_slice(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::ImageSliceItem,
    ) -> Option<crate::resources::ImageSliceGpuData> {
        // Check volume exists before allocating anything.
        if item.volume_id.0 >= self.volume_textures.len() {
            return None;
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ImageSliceUniform {
            bbox_min: [f32; 3],
            axis: u32,
            bbox_max: [f32; 3],
            offset: f32,
            scalar_min: f32,
            scalar_max: f32,
            opacity: f32,
            _pad: f32,
        }

        let axis_u32 = match item.axis {
            crate::renderer::SliceAxis::X => 0u32,
            crate::renderer::SliceAxis::Y => 1u32,
            crate::renderer::SliceAxis::Z => 2u32,
        };

        let uniform_data = ImageSliceUniform {
            bbox_min: item.bbox_min,
            axis: axis_u32,
            bbox_max: item.bbox_max,
            offset: item.offset.clamp(0.0, 1.0),
            scalar_min: item.scalar_range.0,
            scalar_max: item.scalar_range.1,
            opacity: item.opacity,
            _pad: 0.0,
        };

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image_slice_uniform_buf"),
            size: std::mem::size_of::<ImageSliceUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        // Nearest-neighbor sampler for crisp slice sampling.
        let vol_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("image_slice_vol_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Resolve LUT view index before creating any bind group references.
        let lut_view_idx: Option<usize> = self.builtin_colormap_ids.and_then(|ids| {
            let preset_id = item
                .color_lut
                .unwrap_or(ids[crate::resources::BuiltinColormap::Viridis as usize]);
            if preset_id.0 < self.colormap_views.len() {
                Some(preset_id.0)
            } else {
                None
            }
        });

        let bgl = self
            .image_slice_bgl
            .as_ref()
            .expect("ensure_image_slice_pipeline not called");

        // Borrow vol_view and lut_view after all mutable references are resolved.
        let vol_view = &self.volume_textures[item.volume_id.0].1;
        let lut_view = lut_view_idx
            .map(|i| &self.colormap_views[i])
            .unwrap_or(&self.fallback_lut_view);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("image_slice_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(vol_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&vol_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
            ],
        });

        Some(crate::resources::ImageSliceGpuData {
            bind_group,
            _uniform_buf: uniform_buf,
        })
    }

    // -------------------------------------------------------------------------
    // Phase 10B : screen-space image overlays
    // -------------------------------------------------------------------------

    /// Lazily create the screen-space image render pipeline.
    ///
    /// No-op if already created. Called from `prepare()` when
    /// `frame.scene.screen_images` is non-empty.
    pub(crate) fn ensure_screen_image_pipeline(&mut self, device: &wgpu::Device) {
        if self.screen_image_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("screen_image_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/screen_image.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("screen_image_bgl"),
            entries: &[
                // binding 0: ScreenImageUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("screen_image_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screen_image_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
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
                cull_mode: None,
                ..Default::default()
            },
            // Use Always depth compare (never test) so screen images are always on top.
            // No depth writes. Format must match the depth attachment of the render pass.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
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
        });

        self.screen_image_bgl = Some(bgl);
        self.screen_image_pipeline = Some(pipeline);
    }

    /// Lazily create the depth-composite screen-image render pipeline (Phase 12).
    ///
    /// No-op if already created. Called from `prepare()` when any submitted
    /// `ScreenImageItem` carries per-pixel depth data.
    pub(crate) fn ensure_screen_image_dc_pipeline(&mut self, device: &wgpu::Device) {
        if self.screen_image_dc_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("screen_image_dc_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/screen_image_dc.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("screen_image_dc_bgl"),
            entries: &[
                // binding 0: ScreenImageUniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: colour texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 2: sampler (filtering, for colour texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 3: R32Float depth texture (non-filterable, read via textureLoad)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("screen_image_dc_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("screen_image_dc_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
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
                cull_mode: None,
                ..Default::default()
            },
            // Depth test: discard fragments whose image depth exceeds scene depth.
            // depth_write_enabled: false so the scene depth buffer is not modified.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
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
        });

        self.screen_image_dc_bgl = Some(bgl);
        self.screen_image_dc_pipeline = Some(pipeline);
    }

    /// Upload one [`ScreenImageItem`] to the GPU and return its per-frame GPU data.
    ///
    /// Creates a new RGBA8Unorm texture each call : intended for per-frame data.
    /// The returned [`ScreenImageGpuData`] is valid only for one frame.
    pub(crate) fn upload_screen_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::ScreenImageItem,
        viewport_w: f32,
        viewport_h: f32,
    ) -> ScreenImageGpuData {
        use crate::ImageAnchor;

        // Create texture from pixel data.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screen_image_tex"),
            size: wgpu::Extent3d {
                width: item.width.max(1),
                height: item.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        if !item.pixels.is_empty() && item.width > 0 && item.height > 0 {
            let raw: Vec<u8> = item.pixels.iter().flat_map(|p| p.iter().copied()).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &raw,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(item.width * 4),
                    rows_per_image: Some(item.height),
                },
                wgpu::Extent3d {
                    width: item.width,
                    height: item.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("screen_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Compute NDC extents from anchor, image size, and scale.
        let img_w_ndc = 2.0 * item.width as f32 * item.scale / viewport_w.max(1.0);
        let img_h_ndc = 2.0 * item.height as f32 * item.scale / viewport_h.max(1.0);

        let (ndc_min_x, ndc_max_x, ndc_min_y, ndc_max_y) = match item.anchor {
            ImageAnchor::TopLeft => (-1.0, -1.0 + img_w_ndc, 1.0 - img_h_ndc, 1.0),
            ImageAnchor::TopRight => (1.0 - img_w_ndc, 1.0, 1.0 - img_h_ndc, 1.0),
            ImageAnchor::BottomLeft => (-1.0, -1.0 + img_w_ndc, -1.0, -1.0 + img_h_ndc),
            ImageAnchor::BottomRight => (1.0 - img_w_ndc, 1.0, -1.0, -1.0 + img_h_ndc),
            _ => (
                -img_w_ndc * 0.5,
                img_w_ndc * 0.5,
                -img_h_ndc * 0.5,
                img_h_ndc * 0.5,
            ),
        };

        // ScreenImageUniform: ndc_min(vec2) + ndc_max(vec2) + alpha(f32) + pad(3xf32) = 32 bytes
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct ScreenImageUniform {
            ndc_min: [f32; 2],
            ndc_max: [f32; 2],
            alpha: f32,
            _pad: [f32; 3],
        }

        let uniform_data = ScreenImageUniform {
            ndc_min: [ndc_min_x, ndc_min_y],
            ndc_max: [ndc_max_x, ndc_max_y],
            alpha: item.alpha,
            _pad: [0.0; 3],
        };

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screen_image_uniform"),
            size: std::mem::size_of::<ScreenImageUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .screen_image_bgl
            .as_ref()
            .expect("ensure_screen_image_pipeline not called");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("screen_image_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Phase 12: if the item carries per-pixel depth data, upload a R32Float depth texture
        // and create a second bind group for the depth-composite pipeline.
        let (depth_texture_opt, depth_bind_group_opt) = if let Some(depth_values) = &item.depth {
            let dc_bgl = self
                .screen_image_dc_bgl
                .as_ref()
                .expect("ensure_screen_image_dc_pipeline not called before upload_screen_image");

            let dtex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("screen_image_depth_tex"),
                size: wgpu::Extent3d {
                    width: item.width.max(1),
                    height: item.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            // Upload depth values as raw bytes (each f32 = 4 bytes).
            let pixel_count = (item.width * item.height) as usize;
            let safe_depth: Vec<f32> = if depth_values.len() >= pixel_count {
                depth_values[..pixel_count].to_vec()
            } else {
                // Pad with far-plane depth (1.0) if caller supplied too few values.
                let mut v = depth_values.clone();
                v.resize(pixel_count, 1.0);
                v
            };

            if item.width > 0 && item.height > 0 {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &dtex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&safe_depth),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(item.width * 4),
                        rows_per_image: Some(item.height),
                    },
                    wgpu::Extent3d {
                        width: item.width,
                        height: item.height,
                        depth_or_array_layers: 1,
                    },
                );
            }

            let dview = dtex.create_view(&wgpu::TextureViewDescriptor::default());

            let dc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("screen_image_dc_bg"),
                layout: dc_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&dview),
                    },
                ],
            });

            (Some(dtex), Some(dc_bg))
        } else {
            (None, None)
        };

        ScreenImageGpuData {
            _uniform_buf: uniform_buf,
            _texture: texture,
            bind_group,
            _depth_texture: depth_texture_opt,
            depth_bind_group: depth_bind_group_opt,
        }
    }

    /// Upload one [`OverlayImageItem`] to the GPU and return its render data (Phase 7).
    ///
    /// Reuses the `screen_image_pipeline` and its bind group layout; the shaders and
    /// uniform layout are identical. No depth path: `OverlayImageItem` has no depth field.
    ///
    /// Caller must have called [`ensure_screen_image_pipeline`] first.
    pub(crate) fn upload_overlay_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::OverlayImageItem,
        viewport_w: f32,
        viewport_h: f32,
    ) -> ScreenImageGpuData {
        use crate::ImageAnchor;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay_image_tex"),
            size: wgpu::Extent3d {
                width: item.width.max(1),
                height: item.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        if !item.pixels.is_empty() && item.width > 0 && item.height > 0 {
            let raw: Vec<u8> = item.pixels.iter().flat_map(|p| p.iter().copied()).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &raw,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(item.width * 4),
                    rows_per_image: Some(item.height),
                },
                wgpu::Extent3d {
                    width: item.width,
                    height: item.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let img_w_ndc = 2.0 * item.width as f32 * item.scale / viewport_w.max(1.0);
        let img_h_ndc = 2.0 * item.height as f32 * item.scale / viewport_h.max(1.0);

        let (ndc_min_x, ndc_max_x, ndc_min_y, ndc_max_y) = match item.anchor {
            ImageAnchor::TopLeft => (-1.0, -1.0 + img_w_ndc, 1.0 - img_h_ndc, 1.0),
            ImageAnchor::TopRight => (1.0 - img_w_ndc, 1.0, 1.0 - img_h_ndc, 1.0),
            ImageAnchor::BottomLeft => (-1.0, -1.0 + img_w_ndc, -1.0, -1.0 + img_h_ndc),
            ImageAnchor::BottomRight => (1.0 - img_w_ndc, 1.0, -1.0, -1.0 + img_h_ndc),
            _ => (
                -img_w_ndc * 0.5,
                img_w_ndc * 0.5,
                -img_h_ndc * 0.5,
                img_h_ndc * 0.5,
            ),
        };

        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct ScreenImageUniform {
            ndc_min: [f32; 2],
            ndc_max: [f32; 2],
            alpha: f32,
            _pad: [f32; 3],
        }

        let uniform_data = ScreenImageUniform {
            ndc_min: [ndc_min_x, ndc_min_y],
            ndc_max: [ndc_max_x, ndc_max_y],
            alpha: item.alpha,
            _pad: [0.0; 3],
        };

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_image_uniform"),
            size: std::mem::size_of::<ScreenImageUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .screen_image_bgl
            .as_ref()
            .expect("ensure_screen_image_pipeline not called before upload_overlay_image");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_image_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        ScreenImageGpuData {
            _uniform_buf: uniform_buf,
            _texture: texture,
            bind_group,
            _depth_texture: None,
            depth_bind_group: None,
        }
    }

    /// Lazily create the projected-tetrahedra render pipeline.
    ///
    /// No-op if already created. Called from `render.rs` when transparent_volume_meshes
    /// is non-empty. Also ensures the bind group layout exists.
    pub(crate) fn ensure_pt_pipeline(&mut self, device: &wgpu::Device) {
        if self.pt_pipeline.is_some() {
            return;
        }

        self.ensure_pt_bind_group_layout(device);
        let bgl = self
            .pt_bind_group_layout
            .as_ref()
            .expect("pt_bind_group_layout must exist after ensure_pt_bind_group_layout");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("projected_tet_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/projected_tet.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pt_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, bgl],
            push_constant_ranges: &[],
        });

        // Blend states match the existing OIT mesh pipeline.
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pt_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // all data comes from storage buffer via instance_index
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
                cull_mode: None, // bounding quad can have any winding
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

        self.pt_pipeline = Some(pipeline);
    }
}
