use super::*;

impl ViewportGpuResources {
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/polyline.wgsl").into()),
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
        //   offset  64: colour_a           vec4  : direct RGBA at segment start
        //   offset  80: colour_b           vec4  : direct RGBA at segment end
        //   offset  96: radius_a          f32   : line width in px at A (= line_width when node_radii is empty)
        //   offset 100: radius_b          f32   : line width in px at B
        //   offset 104: use_direct_colour  u32   : 1 = use colour_a/b, 0 = use scalar LUT / default
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
                }, // colour_a
                wgpu::VertexAttribute {
                    offset: 80,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                }, // colour_b
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
                }, // use_direct_colour
            ],
        };

        let sample_count = self.sample_count;
        let make = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("polyline_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[pl_instance_layout.clone()],
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
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        self.polyline_bgl = Some(pl_bgl);
        self.polyline_pipeline = Some(DualPipeline {
            ldr: make(self.target_format),
            hdr: make(wgpu::TextureFormat::Rgba16Float),
        });

        self.ensure_polyline_wireframe_pipeline(device);
    }

    /// Lazily create the thin wireframe polyline pipeline (LineList, 1px).
    ///
    /// Reads segment endpoints from a storage buffer so no vertex buffer is needed.
    /// Created alongside `ensure_polyline_pipeline`; no-op if already created.
    pub(crate) fn ensure_polyline_wireframe_pipeline(&mut self, device: &wgpu::Device) {
        if self.polyline_wireframe_pipeline.is_some() {
            return;
        }

        let wf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("polyline_wireframe_bgl"),
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
            label: Some("polyline_wireframe_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/polyline_wireframe.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("polyline_wireframe_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &wf_bgl],
            push_constant_ranges: &[],
        });

        let sample_count = self.sample_count;
        let make = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("polyline_wireframe_pipeline"),
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
                        format: fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    cull_mode: None,
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
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        self.polyline_wireframe_bgl = Some(wf_bgl);
        self.polyline_wireframe_pipeline = Some(DualPipeline {
            ldr: make(self.target_format),
            hdr: make(wgpu::TextureFormat::Rgba16Float),
        });
    }

    /// Upload one [`PolylineItem`] to the GPU and return draw data.
    ///
    /// Converts the strip-based point list into a flat segment-instance buffer
    /// suitable for the screen-space thick-line pipeline with miter joints.
    ///
    /// Each consecutive pair of points in a strip becomes one 112-byte instance
    /// containing miter geometry, scalar colouring, direct RGBA colours, and per-vertex
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
            pos_a: [f32; 3],       // offset   0
            pos_b: [f32; 3],       // offset  12
            prev_pos: [f32; 3],    // offset  24
            next_pos: [f32; 3],    // offset  36
            scalar_a: f32,         // offset  48
            scalar_b: f32,         // offset  52
            has_prev: u32,         // offset  56
            has_next: u32,         // offset  60
            colour_a: [f32; 4],     // offset  64
            colour_b: [f32; 4],     // offset  80
            radius_a: f32,         // offset  96
            radius_b: f32,         // offset 100
            use_direct_colour: u32, // offset 104
            _pad: u32,             // offset 108
        }

        // Determine which colour/scalar/radius source to use per segment.
        let use_direct = !item.node_colours.is_empty() || !item.edge_colours.is_empty();
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
                    let s = item
                        .edge_scalars
                        .get(seg_idx_global)
                        .copied()
                        .unwrap_or(0.0);
                    (s, s)
                } else {
                    (
                        item.scalars.get(i).copied().unwrap_or(0.0),
                        item.scalars.get(j).copied().unwrap_or(0.0),
                    )
                };

                // Direct colour: node_colours (per-endpoint) > edge_colours (per-segment)
                let (colour_a, colour_b) = if !item.node_colours.is_empty() {
                    (
                        item.node_colours.get(i).copied().unwrap_or([1.0; 4]),
                        item.node_colours.get(j).copied().unwrap_or([1.0; 4]),
                    )
                } else if !item.edge_colours.is_empty() {
                    let c = item
                        .edge_colours
                        .get(seg_idx_global)
                        .copied()
                        .unwrap_or([1.0; 4]);
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
                    prev_pos: if has_prev {
                        positions[i - 1]
                    } else {
                        positions[i]
                    },
                    next_pos: if has_next {
                        positions[j + 1]
                    } else {
                        positions[j]
                    },
                    scalar_a,
                    scalar_b,
                    has_prev: has_prev as u32,
                    has_next: has_next as u32,
                    colour_a,
                    colour_b,
                    radius_a,
                    radius_b,
                    use_direct_colour: use_direct as u32,
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
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                let mx = scalar_source
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
            default_colour: [f32; 4], // offset  0
            line_width: f32,         // offset 16
            scalar_min: f32,         // offset 20
            scalar_max: f32,         // offset 24
            has_scalar: u32,         // offset 28
            viewport_width: f32,     // offset 32
            viewport_height: f32,    // offset 36
            _pad: [f32; 2],          // offset 40  (total 48 bytes)
        }
        let uniform_data = PolylineUniform {
            default_colour: item.default_colour,
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
            .builtin_colourmap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colourmap_id
                    .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                self.colourmap_views.get(preset_id.0)
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

        let wireframe_bind_group = self.polyline_wireframe_bgl.as_ref().map(|bgl| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("polyline_wireframe_bind_group"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                }],
            })
        });

        PolylineGpuData {
            vertex_buffer,
            segment_count: seg_count,
            bind_group,
            _uniform_buf: uniform_buf,
            skip_clip: false,
            wireframe: false,
            wireframe_bind_group,
        }
    }

    /// Lazily create the clip-exempt polyline pipeline.
    ///
    /// Identical to the regular polyline pipeline but uses `fs_main_no_clip` so
    /// fragments are never discarded by clip planes or clip volumes. Used for
    /// clip object wireframe overlays which must always be fully visible.
    pub(crate) fn ensure_polyline_no_clip_pipeline(&mut self, device: &wgpu::Device) {
        if self.polyline_no_clip_pipeline.is_some() {
            return;
        }
        // The regular pipeline (and its BGL) must exist first.
        self.ensure_polyline_pipeline(device);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polyline_no_clip_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/polyline.wgsl").into()),
        });

        let pl_bgl = self
            .polyline_bgl
            .as_ref()
            .expect("polyline_bgl must exist after ensure_polyline_pipeline");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("polyline_no_clip_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, pl_bgl],
            push_constant_ranges: &[],
        });

        // Vertex buffer layout is identical to the regular polyline pipeline (112 bytes/segment).
        let pl_instance_layout = wgpu::VertexBufferLayout {
            array_stride: 112,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 52,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 56,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: 60,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: 64,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 80,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 96,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 100,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 104,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        let sample_count = self.sample_count;
        let make = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("polyline_no_clip_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[pl_instance_layout.clone()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main_no_clip"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
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
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        self.polyline_no_clip_pipeline = Some(DualPipeline {
            ldr: make(self.target_format),
            hdr: make(wgpu::TextureFormat::Rgba16Float),
        });
    }

    /// Lazily create the polyline outline mask pipeline.
    ///
    /// Renders polyline segments into the R8 mask texture using the same
    /// screen-space quad expansion as the regular pipeline, but outputs white
    /// and skips clip plane / colour logic.
    pub(crate) fn ensure_polyline_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.polyline_outline_mask_pipeline.is_some() {
            return;
        }
        self.ensure_polyline_pipeline(device);

        let pl_bgl = self
            .polyline_bgl
            .as_ref()
            .expect("polyline_bgl must exist after ensure_polyline_pipeline");

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("polyline_outline_mask_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, pl_bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polyline_outline_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/polyline_outline_mask.wgsl").into(),
            ),
        });

        let pl_instance_layout = wgpu::VertexBufferLayout {
            array_stride: 112,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,   shader_location: 0,  format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12,  shader_location: 1,  format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 24,  shader_location: 2,  format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 36,  shader_location: 3,  format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 48,  shader_location: 4,  format: wgpu::VertexFormat::Float32   },
                wgpu::VertexAttribute { offset: 52,  shader_location: 5,  format: wgpu::VertexFormat::Float32   },
                wgpu::VertexAttribute { offset: 56,  shader_location: 6,  format: wgpu::VertexFormat::Uint32    },
                wgpu::VertexAttribute { offset: 60,  shader_location: 7,  format: wgpu::VertexFormat::Uint32    },
                wgpu::VertexAttribute { offset: 64,  shader_location: 8,  format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 80,  shader_location: 9,  format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 96,  shader_location: 10, format: wgpu::VertexFormat::Float32   },
                wgpu::VertexAttribute { offset: 100, shader_location: 11, format: wgpu::VertexFormat::Float32   },
                wgpu::VertexAttribute { offset: 104, shader_location: 12, format: wgpu::VertexFormat::Uint32    },
            ],
        };

        self.polyline_outline_mask_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("polyline_outline_mask_pipeline"),
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
                        format: wgpu::TextureFormat::R8Unorm,
                        blend: None,
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
                    depth_write_enabled: true,
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
            },
        ));
    }
}
