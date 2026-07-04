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
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/streamtube.wgsl")).into(),
            ),
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

        // Ribbon BGL adds an optional streak texture + sampler alongside the
        // shared uniform binding. The fragment shader keys off `has_texture`
        // and falls back to the resolved colour when no texture is bound.
        let ribbon_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ribbon_bgl"),
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
            ],
        });
        let ribbon_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ribbon_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &ribbon_bgl],
            push_constant_ranges: &[],
        });

        // Ribbon pipeline: same layout, two-sided shader, cull_mode None.
        let ribbon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ribbon_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/ribbon.wgsl")).into(),
            ),
        });
        let additive_blend = wgpu::BlendState {
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
        let premultiplied_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        // Additive and premultiplied ribbons are typically used for emissive
        // trails; depth write is disabled so successive segments accumulate
        // rather than clipping each other when they overlap.
        let make_ribbon =
            |fmt: wgpu::TextureFormat, blend: wgpu::BlendState, depth_write: bool, label: &str| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&ribbon_layout),
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
                            blend: Some(blend),
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

        // Wireframe pipeline: same shader and bind groups as the solid tube, but LineList
        // topology and no back-face culling so edges on both sides are visible.
        let make_tube_wireframe = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("streamtube_wireframe_pipeline"),
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
                    topology: wgpu::PrimitiveTopology::LineList,
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

        // Ribbon wireframe pipeline: same as tube wireframe but using the ribbon shader.
        let make_ribbon_wireframe = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ribbon_wireframe_pipeline"),
                layout: Some(&ribbon_layout),
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
                    topology: wgpu::PrimitiveTopology::LineList,
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
        self.ribbon_bgl = Some(ribbon_bgl);
        self.streamtube_pipeline = Some(DualPipeline {
            ldr: make_tube(ldr),
            hdr: make_tube(hdr),
        });
        self.streamtube_wireframe_pipeline = Some(DualPipeline {
            ldr: make_tube_wireframe(ldr),
            hdr: make_tube_wireframe(hdr),
        });
        let alpha_blend = wgpu::BlendState::ALPHA_BLENDING;
        self.ribbon_pipeline = Some(DualPipeline {
            ldr: make_ribbon(ldr, alpha_blend, true, "ribbon_pipeline"),
            hdr: make_ribbon(hdr, alpha_blend, true, "ribbon_pipeline"),
        });
        self.ribbon_pipeline_additive = Some(DualPipeline {
            ldr: make_ribbon(ldr, additive_blend, false, "ribbon_pipeline_additive"),
            hdr: make_ribbon(hdr, additive_blend, false, "ribbon_pipeline_additive"),
        });
        self.ribbon_pipeline_premultiplied = Some(DualPipeline {
            ldr: make_ribbon(
                ldr,
                premultiplied_blend,
                false,
                "ribbon_pipeline_premultiplied",
            ),
            hdr: make_ribbon(
                hdr,
                premultiplied_blend,
                false,
                "ribbon_pipeline_premultiplied",
            ),
        });
        self.ribbon_wireframe_pipeline = Some(DualPipeline {
            ldr: make_ribbon_wireframe(ldr),
            hdr: make_ribbon_wireframe(hdr),
        });
    }

    /// Upload one [`StreamtubeItem`] to the GPU and return draw data.
    ///
    /// Generates a connected tube mesh CPU-side using a parallel-transport frame along
    /// each polyline strip, then uploads the result as a single owned vertex+index buffer.
    /// Adjacent rings are joined by quads (2 triangles each) giving a smooth, seamless tube
    /// without the z-fighting or inter-segment gaps that plagued the old instanced approach.
    pub(crate) fn upload_streamtube_per_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::StreamtubeItem,
        wireframe: bool,
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
                        // Rodrigues: u' = u cos(a) + cross(ax, u) sin(a) + ax dot(ax, u) (1 - cos(a))
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
                        colour: [1.0, 1.0, 1.0, 1.0], // overridden by uniform in shader
                        uv: [0.0, 0.0],
                        tangent: [1.0, 0.0, 0.0, 1.0],
                    });
                }

                // Emit quad strip between ring k-1 and ring k.
                // Winding: outward-facing CCW (right-hand rule gives outward normal).
                // Verified: T1=(r0+s, r0+s1, r1+s) has dot(normal, Y) > 0 for s=0 on Z-axis tube.
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
                    colour: [1.0, 1.0, 1.0, 1.0],
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
                    colour: [1.0, 1.0, 1.0, 1.0],
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

        // Edge index buffer: deduplicated triangle edges as line-list pairs for wireframe.
        let edge_indices = crate::resources::extra_impls::generate_edge_indices(&indices);
        let edge_bytes: &[u8] = bytemuck::cast_slice(&edge_indices);
        let edge_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("streamtube_edge_ibuf"),
            size: edge_bytes.len().max(8) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !edge_bytes.is_empty() {
            queue.write_buffer(&edge_index_buffer, 0, edge_bytes);
        }
        let edge_index_count = edge_indices.len() as u32;

        // Uniform buffer: model + colour + radius + use_vertex_colour + unlit + opacity + wireframe.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StreamtubeUniform {
            model: [[f32; 4]; 4],
            colour: [f32; 4],
            radius: f32,
            use_vertex_colour: u32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
            _pad: [f32; 3],
        }
        let uniform_data = StreamtubeUniform {
            model: item.model,
            colour: item.colour,
            radius,
            use_vertex_colour: 0,
            unlit: item.settings.unlit as u32,
            opacity: item.settings.opacity,
            wireframe: wireframe as u32,
            _pad: [0.0; 3],
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
            edge_index_buffer,
            edge_index_count,
            wireframe,
            uniform_bind_group,
            blend: crate::renderer::SpriteBlend::AlphaBlend,
            _uniform_buf: uniform_buf,
        }
    }

    /// Pre-upload a streamtube and return a typed handle.
    ///
    /// Submit a [`StreamtubeRefItem`](crate::renderer::StreamtubeRefItem) on
    /// `SceneFrame::streamtube_refs` each frame to draw the tube at a
    /// per-frame model transform without rebuilding its mesh.
    pub fn upload_streamtube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::StreamtubeItem,
    ) -> crate::resources::StreamtubeId {
        self.ensure_streamtube_pipeline(device);
        let gpu = self.upload_streamtube_per_frame(device, queue, item, false);
        self.streamtube_store.insert(gpu)
    }

    /// Remove a pre-uploaded streamtube.
    pub fn drop_streamtube(&mut self, id: crate::resources::StreamtubeId) -> bool {
        self.streamtube_store.remove(id)
    }

    /// Replace the geometry of a pre-uploaded streamtube, keeping the same id.
    pub fn replace_streamtube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: crate::resources::StreamtubeId,
        item: &crate::renderer::StreamtubeItem,
    ) -> bool {
        if !self.streamtube_store.contains(id) {
            return false;
        }
        self.ensure_streamtube_pipeline(device);
        let gpu = self.upload_streamtube_per_frame(device, queue, item, false);
        self.streamtube_store.replace(id, gpu)
    }

    // -------------------------------------------------------------------------
    // General Tube representation
    // -------------------------------------------------------------------------

    /// Upload one [`TubeItem`] to the GPU and return draw data.
    ///
    /// Generates a connected tube mesh CPU-side using a parallel-transport frame.
    /// Scalar values are baked into per-vertex colours using the CPU-side colourmap copy.
    /// Uses the same streamtube pipeline; sets `use_vertex_colour=1` when scalars are present.
    pub(crate) fn upload_tube_per_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::TubeItem,
        wireframe: bool,
    ) -> StreamtubeGpuData {
        let sides = (item.sides.max(3)) as usize;

        // Resolve scalar-to-colour mapping upfront if scalars are provided.
        let (use_vertex_colour, lut_rgba): (u32, Option<[[u8; 4]; 256]>) =
            if !item.scalars.is_empty() {
                let lut = self
                    .builtin_colourmap_ids
                    .and_then(|ids| {
                        let preset_id = item
                            .colourmap_id
                            .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                        self.colourmaps_cpu.get(preset_id.0).copied()
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
        let scalar_max = item.scalar_range.map(|r| r.1).unwrap_or_else(|| {
            item.scalars
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        });
        let scalar_range = (scalar_max - scalar_min).max(f32::EPSILON);

        // Helper: map a scalar value to an RGBA f32 colour from the LUT.
        let scalar_to_colour = |idx: usize| -> [f32; 4] {
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
                item.colour
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

                let vertex_colour = scalar_to_colour(pts_scalar_start + k);

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
                        colour: vertex_colour,
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
                let cap_colour = scalar_to_colour(pts_scalar_start + n_rings - 1);
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[n_rings - 1].to_array(),
                    normal: tangent.to_array(),
                    colour: cap_colour,
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
                let cap_colour = scalar_to_colour(pts_scalar_start);
                let cap_center_idx = verts.len() as u32;
                verts.push(Vertex {
                    position: pts[0].to_array(),
                    normal: tangent.to_array(),
                    colour: cap_colour,
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

        let edge_indices = crate::resources::extra_impls::generate_edge_indices(&indices);
        let edge_bytes: &[u8] = bytemuck::cast_slice(&edge_indices);
        let edge_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tube_edge_ibuf"),
            size: edge_bytes.len().max(8) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !edge_bytes.is_empty() {
            queue.write_buffer(&edge_index_buffer, 0, edge_bytes);
        }
        let edge_index_count = edge_indices.len() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TubeUniform {
            model: [[f32; 4]; 4],
            colour: [f32; 4],
            radius: f32,
            use_vertex_colour: u32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
            _pad: [f32; 3],
        }
        let uniform_data = TubeUniform {
            model: item.model,
            colour: item.colour,
            radius: item.radius.max(f32::EPSILON),
            use_vertex_colour,
            unlit: item.settings.unlit as u32,
            opacity: item.settings.opacity,
            wireframe: wireframe as u32,
            _pad: [0.0; 3],
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
            edge_index_buffer,
            edge_index_count,
            wireframe,
            uniform_bind_group,
            blend: crate::renderer::SpriteBlend::AlphaBlend,
            _uniform_buf: uniform_buf,
        }
    }

    /// Pre-upload a general tube and return a typed handle.
    pub fn upload_tube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::TubeItem,
    ) -> crate::resources::TubeId {
        self.ensure_streamtube_pipeline(device);
        let gpu = self.upload_tube_per_frame(device, queue, item, false);
        self.tube_store.insert(gpu)
    }

    /// Remove a pre-uploaded tube.
    pub fn drop_tube(&mut self, id: crate::resources::TubeId) -> bool {
        self.tube_store.remove(id)
    }

    /// Replace the geometry of a pre-uploaded tube, keeping the same id.
    pub fn replace_tube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: crate::resources::TubeId,
        item: &crate::renderer::TubeItem,
    ) -> bool {
        if !self.tube_store.contains(id) {
            return false;
        }
        self.ensure_streamtube_pipeline(device);
        let gpu = self.upload_tube_per_frame(device, queue, item, false);
        self.tube_store.replace(id, gpu)
    }

    // -------------------------------------------------------------------------
    // Ribbon representation
    // -------------------------------------------------------------------------

    /// Build and upload GPU data for a `RibbonItem`.
    ///
    /// Each strip is swept as a flat quad surface. Two vertices are generated per
    /// point (left and right edges), connected as a triangle strip. The normal is
    /// the cross product of the tangent and the lateral direction `u`.
    pub(crate) fn upload_ribbon_per_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::RibbonItem,
        wireframe: bool,
    ) -> StreamtubeGpuData {
        // Per-vertex RGBA (`colour_attribute`) takes precedence over the
        // scalar+LUT path and the flat `colour` fallback. Trails typically
        // drive only the alpha channel to fade along their length.
        let has_colour_attribute = !item.colour_attribute.is_empty();

        // Resolve LUT for scalar colouring.
        let (use_vertex_colour, lut_rgba): (u32, Option<[[u8; 4]; 256]>) = if has_colour_attribute {
            (1, None)
        } else if !item.scalars.is_empty() {
            let lut = self
                .builtin_colourmap_ids
                .and_then(|ids| {
                    let preset_id = item
                        .colourmap_id
                        .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                    self.colourmaps_cpu.get(preset_id.0).copied()
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
        let scalar_max = item.scalar_range.map(|r| r.1).unwrap_or_else(|| {
            item.scalars
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        });
        let scalar_range = (scalar_max - scalar_min).max(f32::EPSILON);

        let scalar_to_colour = |idx: usize| -> [f32; 4] {
            if has_colour_attribute {
                return item
                    .colour_attribute
                    .get(idx)
                    .copied()
                    .unwrap_or(item.colour);
            }
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
                item.colour
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
            let ref_v = if t0.x.abs() < 0.9 {
                glam::Vec3::X
            } else {
                glam::Vec3::Y
            };
            let mut u = t0.cross(ref_v).normalize();

            // Per-vertex u along the strip. Defaults to cumulative-arc-length
            // normalised to [0, 1] when the host did not supply a `u_attribute`.
            let mut strip_u: Vec<f32> = Vec::with_capacity(pts.len());
            if item.u_attribute.is_empty() {
                let mut cum = 0.0_f32;
                strip_u.push(0.0);
                for k in 1..pts.len() {
                    cum += (pts[k] - pts[k - 1]).length();
                    strip_u.push(cum);
                }
                let total = strip_u.last().copied().unwrap_or(1.0).max(1e-6);
                for v in &mut strip_u {
                    *v /= total;
                }
            } else {
                for k in 0..pts.len() {
                    strip_u.push(*item.u_attribute.get(pts_start + k).unwrap_or(&0.0));
                }
            }

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
                let colour = scalar_to_colour(pts_start + k);

                let uval = strip_u[k];
                // Left edge vertex. `uv.x` runs along the strip; `uv.y` is the
                // cross-strip coordinate that picks the left or right edge.
                verts.push(Vertex {
                    position: (pt + lateral * half_w).to_array(),
                    normal: normal.to_array(),
                    colour,
                    uv: [uval, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
                // Right edge vertex.
                verts.push(Vertex {
                    position: (pt - lateral * half_w).to_array(),
                    normal: normal.to_array(),
                    colour,
                    uv: [uval, 1.0],
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

        let edge_indices = crate::resources::extra_impls::generate_edge_indices(&indices);
        let edge_bytes: &[u8] = bytemuck::cast_slice(&edge_indices);
        let edge_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ribbon_edge_ibuf"),
            size: edge_bytes.len().max(8) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !edge_bytes.is_empty() {
            queue.write_buffer(&edge_index_buffer, 0, edge_bytes);
        }
        let edge_index_count = edge_indices.len() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RibbonUniform {
            model: [[f32; 4]; 4],
            colour: [f32; 4],
            radius: f32,
            use_vertex_colour: u32,
            unlit: u32,
            opacity: f32,
            wireframe: u32,
            has_texture: u32,
            _pad: [f32; 2],
        }
        let (texture_view, has_texture): (&wgpu::TextureView, u32) =
            if let Some(id) = item.texture_id {
                if let Some(tex) = self.textures.get(id) {
                    (&tex.view, 1)
                } else {
                    (&self.fallback_lut_view, 0)
                }
            } else {
                (&self.fallback_lut_view, 0)
            };
        let uniform_data = RibbonUniform {
            model: item.model,
            colour: item.colour,
            radius: item.width * 0.5,
            use_vertex_colour,
            unlit: item.settings.unlit as u32,
            opacity: item.settings.opacity,
            wireframe: wireframe as u32,
            has_texture,
            _pad: [0.0; 2],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ribbon_uniform_buf"),
            size: std::mem::size_of::<RibbonUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let bgl = self
            .ribbon_bgl
            .as_ref()
            .expect("ensure_streamtube_pipeline not called");
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ribbon_uniform_bg"),
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
            ],
        });

        StreamtubeGpuData {
            vertex_buffer,
            index_buffer,
            index_count,
            edge_index_buffer,
            edge_index_count,
            wireframe,
            uniform_bind_group,
            blend: item.blend,
            _uniform_buf: uniform_buf,
        }
    }

    /// Pre-upload a ribbon and return a typed handle.
    pub fn upload_ribbon(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::RibbonItem,
    ) -> crate::resources::RibbonId {
        self.ensure_streamtube_pipeline(device);
        let gpu = self.upload_ribbon_per_frame(device, queue, item, false);
        self.ribbon_store.insert(gpu)
    }

    /// Remove a pre-uploaded ribbon.
    pub fn drop_ribbon(&mut self, id: crate::resources::RibbonId) -> bool {
        self.ribbon_store.remove(id)
    }

    /// Replace the geometry of a pre-uploaded ribbon, keeping the same id.
    pub fn replace_ribbon(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: crate::resources::RibbonId,
        item: &crate::renderer::RibbonItem,
    ) -> bool {
        if !self.ribbon_store.contains(id) {
            return false;
        }
        self.ensure_streamtube_pipeline(device);
        let gpu = self.upload_ribbon_per_frame(device, queue, item, false);
        self.ribbon_store.replace(id, gpu)
    }

    /// Start an asynchronous streamtube upload.
    pub fn begin_upload_streamtube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: crate::renderer::StreamtubeItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::StreamtubeId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut ViewportGpuResources| {
                        let sid =
                            resources.upload_streamtube(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(sid);
                    }),
                ))
            })
        };
        self.job_results
            .streamtube
            .lock()
            .expect("streamtube result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`StreamtubeId`](crate::resources::StreamtubeId) produced by a
    /// completed [`begin_upload_streamtube`](Self::begin_upload_streamtube) job.
    pub fn upload_result_streamtube(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::StreamtubeId> {
        let mut map = self
            .job_results
            .streamtube
            .lock()
            .expect("streamtube result map poisoned");
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

    /// Start an asynchronous tube upload.
    pub fn begin_upload_tube(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: crate::renderer::TubeItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::TubeId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut ViewportGpuResources| {
                        let tid = resources.upload_tube(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(tid);
                    }),
                ))
            })
        };
        self.job_results
            .tube
            .lock()
            .expect("tube result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`TubeId`](crate::resources::TubeId) produced by a completed
    /// [`begin_upload_tube`](Self::begin_upload_tube) job.
    pub fn upload_result_tube(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::TubeId> {
        let mut map = self
            .job_results
            .tube
            .lock()
            .expect("tube result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(tid) => {
                map.remove(&id);
                Ok(tid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Start an asynchronous ribbon upload.
    pub fn begin_upload_ribbon(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: crate::renderer::RibbonItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::RibbonId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();
        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut ViewportGpuResources| {
                        let rid =
                            resources.upload_ribbon(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(rid);
                    }),
                ))
            })
        };
        self.job_results
            .ribbon
            .lock()
            .expect("ribbon result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`RibbonId`](crate::resources::RibbonId) produced by a completed
    /// [`begin_upload_ribbon`](Self::begin_upload_ribbon) job.
    pub fn upload_result_ribbon(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::RibbonId> {
        let mut map = self
            .job_results
            .ribbon
            .lock()
            .expect("ribbon result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(rid) => {
                map.remove(&id);
                Ok(rid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ViewportGpuResources;
    use crate::renderer::{RibbonItem, StreamtubeItem, TubeItem};
    use crate::resources::UploadStatus;

    fn try_make_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn sample_streamtube() -> StreamtubeItem {
        StreamtubeItem {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            strip_lengths: vec![3],
            radius: 0.1,
            ..Default::default()
        }
    }

    fn sample_tube() -> TubeItem {
        TubeItem {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            strip_lengths: vec![3],
            radius: 0.1,
            ..Default::default()
        }
    }

    fn sample_ribbon() -> RibbonItem {
        RibbonItem {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            strip_lengths: vec![3],
            width: 0.2,
            ..Default::default()
        }
    }

    fn drive_until_ready(
        resources: &mut ViewportGpuResources,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: crate::resources::JobId,
        label: &'static str,
    ) {
        for _ in 0..200 {
            resources.process_uploads(device, queue);
            match resources.upload_status(id) {
                UploadStatus::Ready => return,
                UploadStatus::Failed(e) => panic!("{label} upload failed: {e:?}"),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                UploadStatus::Unknown => panic!("{label} job id disappeared"),
            }
        }
        panic!("{label} upload did not complete in time");
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn streamtube_default_model_is_identity() {
        assert_eq!(StreamtubeItem::default().model, IDENTITY);
    }

    #[test]
    fn tube_default_model_is_identity() {
        assert_eq!(TubeItem::default().model, IDENTITY);
    }

    #[test]
    fn ribbon_default_model_is_identity() {
        assert_eq!(RibbonItem::default().model, IDENTITY);
    }

    #[test]
    fn streamtube_carries_non_identity_model() {
        let mut m = IDENTITY;
        m[3] = [1.0, 2.0, 3.0, 1.0];
        let item = StreamtubeItem {
            model: m,
            ..StreamtubeItem::default()
        };
        assert_eq!(item.model[3], [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn tube_carries_non_identity_model() {
        let mut m = IDENTITY;
        m[3] = [1.0, 2.0, 3.0, 1.0];
        let item = TubeItem {
            model: m,
            ..TubeItem::default()
        };
        assert_eq!(item.model[3], [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn ribbon_carries_non_identity_model() {
        let mut m = IDENTITY;
        m[3] = [1.0, 2.0, 3.0, 1.0];
        let item = RibbonItem {
            model: m,
            ..RibbonItem::default()
        };
        assert_eq!(item.model[3], [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn upload_streamtube_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_streamtube(&device, &queue, &sample_streamtube());
        assert!(resources.streamtube_store.contains(id));
        assert!(resources.drop_streamtube(id));
        assert!(!resources.streamtube_store.contains(id));
    }

    #[test]
    fn upload_tube_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let start = resources.resident_bytes().scivis_bytes;
        let id = resources.upload_tube(&device, &queue, &sample_tube());
        assert!(resources.tube_store.contains(id));
        let after_upload = resources.resident_bytes().scivis_bytes;
        assert!(
            after_upload > start,
            "uploading a tube must increase resident scivis bytes"
        );
        assert!(resources.drop_tube(id));
        assert_eq!(
            resources.resident_bytes().scivis_bytes,
            start,
            "dropping the tube must return resident scivis bytes to the start"
        );
    }

    #[test]
    fn upload_ribbon_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_ribbon(&device, &queue, &sample_ribbon());
        assert!(resources.ribbon_store.contains(id));
        assert!(resources.drop_ribbon(id));
    }

    #[test]
    fn begin_upload_streamtube_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_streamtube(&device, &queue, sample_streamtube());
        drive_until_ready(&mut resources, &device, &queue, job, "streamtube");
        let id = resources.upload_result_streamtube(job).expect("ready");
        assert!(resources.streamtube_store.contains(id));
        let err = resources.upload_result_streamtube(job).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::JobResultMissing { .. }
        ));
    }

    #[test]
    fn begin_upload_tube_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_tube(&device, &queue, sample_tube());
        drive_until_ready(&mut resources, &device, &queue, job, "tube");
        let id = resources.upload_result_tube(job).expect("ready");
        assert!(resources.tube_store.contains(id));
    }

    #[test]
    fn begin_upload_ribbon_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            ViewportGpuResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_ribbon(&device, &queue, sample_ribbon());
        drive_until_ready(&mut resources, &device, &queue, job, "ribbon");
        let id = resources.upload_result_ribbon(job).expect("ready");
        assert!(resources.ribbon_store.contains(id));
    }
}
