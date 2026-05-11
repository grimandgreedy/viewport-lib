use super::*;

impl ViewportGpuResources {

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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/streamtube.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("streamtube_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &streamtube_bgl],
            push_constant_ranges: &[],
        });

        let sample_count = self.sample_count;
        let make_tube = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        format: fmt,
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
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        // Ribbon pipeline: same layout, two-sided shader, cull_mode None.
        let ribbon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ribbon_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ribbon.wgsl").into()),
        });
        let make_ribbon = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ribbon_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &ribbon_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::buffer_layout()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &ribbon_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
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

        let ldr = self.target_format;
        let hdr = wgpu::TextureFormat::Rgba16Float;
        self.streamtube_bgl = Some(streamtube_bgl);
        self.streamtube_pipeline = Some(DualPipeline { ldr: make_tube(ldr), hdr: make_tube(hdr) });
        self.ribbon_pipeline = Some(DualPipeline { ldr: make_ribbon(ldr), hdr: make_ribbon(hdr) });
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
    // Phase 8.1 : Ribbon representation
    // -------------------------------------------------------------------------

    /// Build and upload GPU data for a `RibbonItem`.
    ///
    /// Each strip is swept as a flat quad surface. Two vertices are generated per
    /// point (left and right edges), connected as a triangle strip. The normal is
    /// the cross product of the tangent and the lateral direction `u`.
    pub(crate) fn upload_ribbon(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::RibbonItem,
    ) -> StreamtubeGpuData {
        // Resolve LUT for scalar coloring.
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
            .unwrap_or_else(|| item.scalars.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        let scalar_range = (scalar_max - scalar_min).max(f32::EPSILON);

        let scalar_to_color = |idx: usize| -> [f32; 4] {
            if let Some(ref lut) = lut_rgba {
                let s = *item.scalars.get(idx).unwrap_or(&0.0);
                let t = ((s - scalar_min) / scalar_range).clamp(0.0, 1.0);
                let lut_idx = ((t * 255.0).round() as usize).min(255);
                let c = lut[lut_idx];
                [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, c[3] as f32 / 255.0]
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
            let pts_start = strip_start;
            strip_start += strip_len;

            if pts.len() < 2 {
                continue;
            }

            // Build parallel transport frame.
            let t0 = (pts[1] - pts[0]).normalize_or_zero();
            if t0.length_squared() < 1e-10 {
                continue;
            }
            let ref_v = if t0.x.abs() < 0.9 { glam::Vec3::X } else { glam::Vec3::Y };
            let mut u = t0.cross(ref_v).normalize();

            let base = verts.len() as u32;

            for (k, &pt) in pts.iter().enumerate() {
                let tangent = if k + 1 < pts.len() {
                    (pts[k + 1] - pt).normalize_or_zero()
                } else {
                    (pt - pts[k - 1]).normalize_or_zero()
                };

                // Parallel transport: rotate u to stay perpendicular to new tangent.
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

                // If twist_attribute provided, align u with projection of that vector onto
                // the plane perpendicular to the tangent.
                let mut lateral = u;
                if let Some(ref twist) = item.twist_attribute {
                    if let Some(&tv) = twist.get(pts_start + k) {
                        let tv = glam::Vec3::from(tv);
                        let proj = tv - tangent * tangent.dot(tv);
                        if proj.length_squared() > 1e-10 {
                            lateral = proj.normalize();
                        }
                    }
                }

                let normal = tangent.cross(lateral).normalize_or_zero();
                let half_w = item
                    .width_attribute
                    .as_ref()
                    .and_then(|wa| wa.get(pts_start + k).copied())
                    .unwrap_or(item.width)
                    * 0.5;
                let color = scalar_to_color(pts_start + k);

                // Left edge vertex.
                verts.push(Vertex {
                    position: (pt + lateral * half_w).to_array(),
                    normal: normal.to_array(),
                    color,
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                // Right edge vertex.
                verts.push(Vertex {
                    position: (pt - lateral * half_w).to_array(),
                    normal: normal.to_array(),
                    color,
                    uv: [1.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });

                // Connect to previous pair as two triangles.
                if k > 0 {
                    let r0 = base + ((k - 1) * 2) as u32;
                    let r1 = base + (k * 2) as u32;
                    // Triangle 1: r0+0, r0+1, r1+0
                    indices.push(r0);
                    indices.push(r0 + 1);
                    indices.push(r1);
                    // Triangle 2: r0+1, r1+1, r1+0
                    indices.push(r0 + 1);
                    indices.push(r1 + 1);
                    indices.push(r1);
                }
            }
        }

        // Upload vertex + index buffers.
        let vert_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let idx_bytes: &[u8] = bytemuck::cast_slice(&indices);

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ribbon_vbuf"),
            size: vert_bytes.len().max(std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !vert_bytes.is_empty() {
            queue.write_buffer(&vertex_buffer, 0, vert_bytes);
        }

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ribbon_ibuf"),
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
        struct RibbonUniform {
            color: [f32; 4],
            radius: f32,
            use_vertex_color: u32,
            _pad: [f32; 6],
        }
        let uniform_data = RibbonUniform {
            color: item.color,
            radius: item.width * 0.5,
            use_vertex_color,
            _pad: [0.0; 6],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ribbon_uniform_buf"),
            size: std::mem::size_of::<RibbonUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .streamtube_bgl
            .as_ref()
            .expect("ensure_streamtube_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ribbon_uniform_bg"),
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

}
