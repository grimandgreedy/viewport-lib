use super::types::{SubHighlightGpuData, SubHighlightUniform, ViewportGpuResources};
use crate::interaction::sub_object::{SubObjectRef, SubSelectionRef};

impl ViewportGpuResources {
    /// Lazily create sub-object highlight pipelines for both the HDR path
    /// (`Rgba16Float` color target) and the LDR path (swapchain `target_format`).
    /// Idempotent: returns immediately if already created.
    pub(crate) fn ensure_sub_highlight_pipelines(&mut self, device: &wgpu::Device) {
        if self.sub_highlight_fill_pipeline.is_some() {
            return;
        }

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sub_highlight_bgl"),
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sub_highlight_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &bgl],
            push_constant_ranges: &[],
        });

        let fill_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sub_highlight_fill_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sub_highlight_fill.wgsl").into(),
            ),
        });
        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sub_highlight_edge_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sub_highlight_edge.wgsl").into(),
            ),
        });
        let sprite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sub_highlight_sprite_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sub_highlight_sprite.wgsl").into(),
            ),
        });

        // Inline helper: build one fill pipeline for the given color format.
        let make_fill = |label: &'static str, fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &fill_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fill_shader,
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
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState { constant: -2, slope_scale: -1.0, clamp: 0.0 },
                }),
                multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
                multiview: None,
                cache: None,
            })
        };
        let make_edge = |label: &'static str, fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &edge_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 24,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &edge_shader,
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
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
                multiview: None,
                cache: None,
            })
        };
        let make_sprite = |label: &'static str, fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &sprite_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &sprite_shader,
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
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState { count: 1, ..Default::default() },
                multiview: None,
                cache: None,
            })
        };

        let ldr_fmt = self.target_format;
        self.sub_highlight_fill_pipeline =
            Some(make_fill("sub_highlight_fill_hdr", wgpu::TextureFormat::Rgba16Float));
        self.sub_highlight_edge_pipeline =
            Some(make_edge("sub_highlight_edge_hdr", wgpu::TextureFormat::Rgba16Float));
        self.sub_highlight_sprite_pipeline =
            Some(make_sprite("sub_highlight_sprite_hdr", wgpu::TextureFormat::Rgba16Float));
        self.sub_highlight_fill_ldr_pipeline =
            Some(make_fill("sub_highlight_fill_ldr", ldr_fmt));
        self.sub_highlight_edge_ldr_pipeline =
            Some(make_edge("sub_highlight_edge_ldr", ldr_fmt));
        self.sub_highlight_sprite_ldr_pipeline =
            Some(make_sprite("sub_highlight_sprite_ldr", ldr_fmt));
        self.sub_highlight_bgl = Some(bgl);
    }

    /// Build or rebuild `SubHighlightGpuData` from a `SubSelectionRef` snapshot.
    pub(crate) fn build_sub_highlight(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sel: &SubSelectionRef,
        fill_color: [f32; 4],
        edge_color: [f32; 4],
        edge_width: f32,
        vertex_size: f32,
        viewport_width: f32,
        viewport_height: f32,
    ) -> SubHighlightGpuData {
        let mut fill_verts: Vec<[f32; 3]> = Vec::new();
        // Each segment is (pos_a, pos_b) stored flat: [pos_a.x, pos_a.y, pos_a.z, pos_b.x, ...]
        let mut edge_data: Vec<f32> = Vec::new();
        let mut sprite_pos: Vec<[f32; 3]> = Vec::new();

        for (node_id, sub_ref) in &sel.items {
            let model = sel
                .model_matrices
                .get(node_id)
                .copied()
                .unwrap_or(glam::Mat4::IDENTITY);

            let xform = |lp: [f32; 3]| -> [f32; 3] {
                (model * glam::Vec4::new(lp[0], lp[1], lp[2], 1.0))
                    .truncate()
                    .to_array()
            };

            match sub_ref {
                SubObjectRef::Face(i) => {
                    if let Some((positions, indices)) = sel.mesh_lookup.get(node_id) {
                        // parry3d encodes backface hits as face_idx + n_triangles.
                        // Wrap to the canonical front-face index so both sides highlight.
                        let n_tri = indices.len() / 3;
                        let face_raw = *i as usize;
                        let face = if face_raw >= n_tri { face_raw - n_tri } else { face_raw };
                        let base = face * 3;
                        if base + 2 < indices.len() {
                            let ia = indices[base] as usize;
                            let ib = indices[base + 1] as usize;
                            let ic = indices[base + 2] as usize;
                            if ia < positions.len() && ib < positions.len() && ic < positions.len()
                            {
                                let a = xform(positions[ia]);
                                let b = xform(positions[ib]);
                                let c = xform(positions[ic]);
                                // Face fill: one triangle.
                                fill_verts.extend_from_slice(&[a, b, c]);
                                // Edge outline: three edges of the triangle.
                                for (p0, p1) in [(a, b), (b, c), (c, a)] {
                                    edge_data.extend_from_slice(&p0);
                                    edge_data.extend_from_slice(&p1);
                                }
                            }
                        }
                    }
                }
                SubObjectRef::Vertex(v) => {
                    if let Some((positions, _)) = sel.mesh_lookup.get(node_id) {
                        if let Some(lp) = positions.get(*v as usize) {
                            sprite_pos.push(xform(*lp));
                        }
                    }
                }
                SubObjectRef::Point(i) => {
                    if let Some(pts) = sel.point_positions.get(node_id) {
                        if let Some(p) = pts.get(*i as usize) {
                            sprite_pos.push(*p);
                        }
                    }
                }
                _ => {}
            }
        }

        // Helper: create a VERTEX | COPY_DST buffer from a byte slice, or a 1-byte
        // placeholder when the slice is empty (wgpu requires non-zero size).
        let make_buf = |label: &str, data: &[u8]| -> wgpu::Buffer {
            let size = data.len().max(1) as u64;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !data.is_empty() {
                queue.write_buffer(&buf, 0, data);
            }
            buf
        };

        let fill_vertex_buf = make_buf(
            "sub_hl_fill_vb",
            bytemuck::cast_slice::<[f32; 3], u8>(&fill_verts),
        );
        let fill_vertex_count = fill_verts.len() as u32;

        let edge_vertex_buf = make_buf(
            "sub_hl_edge_vb",
            bytemuck::cast_slice::<f32, u8>(&edge_data),
        );
        // Each edge segment is 6 f32 values (24 bytes); segment count = total floats / 6.
        let edge_segment_count = (edge_data.len() / 6) as u32;

        let sprite_vertex_buf = make_buf(
            "sub_hl_sprite_vb",
            bytemuck::cast_slice::<[f32; 3], u8>(&sprite_pos),
        );
        let sprite_point_count = sprite_pos.len() as u32;

        // Shared highlight uniform buffer.
        let uniform = SubHighlightUniform {
            fill_color,
            edge_color,
            edge_width,
            vertex_size,
            viewport_width,
            viewport_height,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sub_hl_uniform"),
            size: std::mem::size_of::<SubHighlightUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::cast_slice(&[uniform]));

        let (fill_bind_group, edge_bind_group, sprite_bind_group) = {
            let bgl = self.sub_highlight_bgl.as_ref().unwrap();
            let binding = uniform_buf.as_entire_binding();
            let fill_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sub_hl_fill_bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: binding.clone() }],
            });
            let edge_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sub_hl_edge_bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: binding.clone() }],
            });
            let sprite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sub_hl_sprite_bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: binding }],
            });
            (fill_bg, edge_bg, sprite_bg)
        }; // bgl borrow dropped here

        SubHighlightGpuData {
            fill_vertex_buf,
            fill_vertex_count,
            edge_vertex_buf,
            edge_segment_count,
            sprite_vertex_buf,
            sprite_point_count,
            _uniform_buf: uniform_buf,
            fill_bind_group,
            edge_bind_group,
            sprite_bind_group,
        }
    }
}
