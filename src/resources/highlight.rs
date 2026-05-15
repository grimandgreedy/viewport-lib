use super::types::{SubHighlightGpuData, SubHighlightUniform, ViewportGpuResources};
use crate::interaction::sub_object::{SubObjectRef, SubSelectionRef};

/// Recover the two world-space endpoint positions for a global polyline segment index.
///
/// Returns `None` if the index is out of range or positions are missing.
fn segment_endpoints(
    seg_idx: u32,
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
) -> Option<([f32; 3], [f32; 3])> {
    let mut seg_off = 0u32;
    let mut node_off = 0usize;
    if strip_lengths.is_empty() {
        let node = seg_idx as usize;
        return Some((*positions.get(node)?, *positions.get(node + 1)?));
    }
    for &slen in strip_lengths {
        let segs = slen.saturating_sub(1);
        if seg_idx < seg_off + segs {
            let local = (seg_idx - seg_off) as usize;
            let a = node_off + local;
            return Some((*positions.get(a)?, *positions.get(a + 1)?));
        }
        seg_off += segs;
        node_off += slen as usize;
    }
    None
}

impl ViewportGpuResources {
    /// Lazily create sub-object highlight pipelines for both the HDR path
    /// (`Rgba16Float` colour target) and the LDR path (swapchain `target_format`).
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

        // Inline helper: build one fill pipeline for the given colour format.
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
                    bias: wgpu::DepthBiasState {
                        constant: -2,
                        slope_scale: -1.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
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
                // Always pass depth so edge lines are visible even when the
                // geometry they belong to is a 3D solid (e.g. tube/streamtube
                // where the control curve sits inside the rendered mesh).
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
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
                // Always pass depth so point sprites are visible even when
                // the control point is inside a 3D solid (e.g. tube/streamtube).
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };

        let ldr_fmt = self.target_format;
        self.sub_highlight_fill_pipeline = Some(make_fill(
            "sub_highlight_fill_hdr",
            wgpu::TextureFormat::Rgba16Float,
        ));
        self.sub_highlight_edge_pipeline = Some(make_edge(
            "sub_highlight_edge_hdr",
            wgpu::TextureFormat::Rgba16Float,
        ));
        self.sub_highlight_sprite_pipeline = Some(make_sprite(
            "sub_highlight_sprite_hdr",
            wgpu::TextureFormat::Rgba16Float,
        ));
        self.sub_highlight_fill_ldr_pipeline = Some(make_fill("sub_highlight_fill_ldr", ldr_fmt));
        self.sub_highlight_edge_ldr_pipeline = Some(make_edge("sub_highlight_edge_ldr", ldr_fmt));
        self.sub_highlight_sprite_ldr_pipeline =
            Some(make_sprite("sub_highlight_sprite_ldr", ldr_fmt));
        self.sub_highlight_bgl = Some(bgl);
    }

    /// Build or rebuild `SubHighlightGpuData` from an optional `SubSelectionRef` snapshot.
    ///
    // ---------------------------------------------------------------------------
    // highlight build helpers
    // ---------------------------------------------------------------------------

    /// `sel` may be `None` when only `extra_edge_data` edges need to be rendered
    /// (e.g. volume AABB outlines with no active sub-element selection).
    ///
    /// `extra_edge_data` is a flat list of f32 pairs (start xyz, end xyz per segment)
    /// appended to the edge geometry after all sub-selection edges are emitted.
    /// Pass an empty slice when there are no extra edges.
    pub(crate) fn build_sub_highlight(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sel: Option<&SubSelectionRef>,
        _splat_positions: &std::collections::HashMap<u64, Vec<[f32; 3]>>,
        extra_edge_data: &[f32],
        fill_colour: [f32; 4],
        edge_colour: [f32; 4],
        edge_width: f32,
        vertex_size: f32,
        viewport_width: f32,
        viewport_height: f32,
    ) -> SubHighlightGpuData {
        let mut fill_verts: Vec<[f32; 3]> = Vec::new();
        // Each segment is (pos_a, pos_b) stored flat: [pos_a.x, pos_a.y, pos_a.z, pos_b.x, ...]
        let mut edge_data: Vec<f32> = Vec::new();
        let mut sprite_pos: Vec<[f32; 3]> = Vec::new();

        if let Some(sel) = sel {
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
                            let face = if face_raw >= n_tri {
                                face_raw - n_tri
                            } else {
                                face_raw
                            };
                            let base = face * 3;
                            if base + 2 < indices.len() {
                                let ia = indices[base] as usize;
                                let ib = indices[base + 1] as usize;
                                let ic = indices[base + 2] as usize;
                                if ia < positions.len()
                                    && ib < positions.len()
                                    && ic < positions.len()
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
                    SubObjectRef::Voxel(flat) => {
                        if let Some(info) = sel.voxel_lookup.get(node_id) {
                            let [nx, ny, nz] = info.dims;
                            if nx == 0 || ny == 0 || nz == 0 {
                                continue;
                            }
                            let flat = *flat;
                            let ix = flat % nx;
                            let iy = (flat / nx) % ny;
                            let iz = flat / (nx * ny);
                            let bbox_min = glam::Vec3::from(info.bbox_min);
                            let bbox_max = glam::Vec3::from(info.bbox_max);
                            let cell = (bbox_max - bbox_min)
                                / glam::Vec3::new(nx as f32, ny as f32, nz as f32);
                            let lo =
                                bbox_min + cell * glam::Vec3::new(ix as f32, iy as f32, iz as f32);
                            let hi = lo + cell;
                            let m = glam::Mat4::from_cols_array_2d(&info.model);
                            let xv =
                                |lp: glam::Vec3| -> [f32; 3] { m.transform_point3(lp).to_array() };
                            // 8 corners of the voxel AABB.
                            let c = [
                                xv(glam::Vec3::new(lo.x, lo.y, lo.z)),
                                xv(glam::Vec3::new(hi.x, lo.y, lo.z)),
                                xv(glam::Vec3::new(hi.x, hi.y, lo.z)),
                                xv(glam::Vec3::new(lo.x, hi.y, lo.z)),
                                xv(glam::Vec3::new(lo.x, lo.y, hi.z)),
                                xv(glam::Vec3::new(hi.x, lo.y, hi.z)),
                                xv(glam::Vec3::new(hi.x, hi.y, hi.z)),
                                xv(glam::Vec3::new(lo.x, hi.y, hi.z)),
                            ];
                            // 12 edges of the cube.
                            for (a, b) in [
                                (0, 1),
                                (1, 2),
                                (2, 3),
                                (3, 0), // bottom face
                                (4, 5),
                                (5, 6),
                                (6, 7),
                                (7, 4), // top face
                                (0, 4),
                                (1, 5),
                                (2, 6),
                                (3, 7), // verticals
                            ] {
                                edge_data.extend_from_slice(&c[a]);
                                edge_data.extend_from_slice(&c[b]);
                            }
                        }
                    }
                    SubObjectRef::Cell(i) => {
                        if let Some(info) = sel.cell_lookup.get(node_id) {
                            if let Some(cell) = info.cells.get(*i as usize) {
                                const S: u32 = u32::MAX; // CELL_SENTINEL
                                let nv: usize = if cell[4] == S {
                                    4
                                } else if cell[5] == S {
                                    5
                                } else if cell[6] == S {
                                    6
                                } else {
                                    8
                                };
                                let edges: &[(usize, usize)] = match nv {
                                    4 => &[(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)],
                                    5 => &[
                                        (0, 1),
                                        (1, 2),
                                        (2, 3),
                                        (3, 0),
                                        (0, 4),
                                        (1, 4),
                                        (2, 4),
                                        (3, 4),
                                    ],
                                    6 => &[
                                        (0, 1),
                                        (1, 2),
                                        (0, 2),
                                        (3, 4),
                                        (4, 5),
                                        (3, 5),
                                        (0, 3),
                                        (1, 4),
                                        (2, 5),
                                    ],
                                    _ => &[
                                        (0, 1),
                                        (1, 2),
                                        (2, 3),
                                        (3, 0),
                                        (4, 5),
                                        (5, 6),
                                        (6, 7),
                                        (7, 4),
                                        (0, 4),
                                        (1, 5),
                                        (2, 6),
                                        (3, 7),
                                    ],
                                };
                                for &(a, b) in edges {
                                    if let (Some(&pa), Some(&pb)) = (
                                        info.positions.get(cell[a] as usize),
                                        info.positions.get(cell[b] as usize),
                                    ) {
                                        edge_data.extend_from_slice(&xform(pa));
                                        edge_data.extend_from_slice(&xform(pb));
                                    }
                                }
                            }
                        }
                    }
                    SubObjectRef::Point(i) => {
                        // Polyline node sprite. Falls back to curve_family_lookup for
                        // streamtube/tube/ribbon picks, then point_positions for
                        // point-cloud picks that share the same SubObjectRef variant.
                        if let Some(info) = sel
                            .polyline_lookup
                            .get(node_id)
                            .or_else(|| sel.curve_family_lookup.get(node_id))
                        {
                            if let Some(&pos) = info.positions.get(*i as usize) {
                                sprite_pos.push(xform(pos));
                            }
                        } else if let Some(positions) = sel.point_positions.get(node_id) {
                            if let Some(&pos) = positions.get(*i as usize) {
                                sprite_pos.push(xform(pos));
                            }
                        }
                    }
                    SubObjectRef::Segment(idx) => {
                        // Polyline or curve-family segment edge. Recover the two endpoint
                        // positions for the global segment index by walking strip_lengths.
                        let info = sel
                            .polyline_lookup
                            .get(node_id)
                            .or_else(|| sel.curve_family_lookup.get(node_id));
                        if let Some(info) = info {
                            if let Some((pa, pb)) =
                                segment_endpoints(*idx, &info.positions, &info.strip_lengths)
                            {
                                edge_data.extend_from_slice(&xform(pa));
                                edge_data.extend_from_slice(&xform(pb));
                            }
                        }
                    }
                    SubObjectRef::Strip(s) => {
                        // All segments in the strip rendered as edge lines.
                        let info = sel
                            .polyline_lookup
                            .get(node_id)
                            .or_else(|| sel.curve_family_lookup.get(node_id));
                        if let Some(info) = info {
                            let node_start: usize = info
                                .strip_lengths
                                .iter()
                                .take(*s as usize)
                                .map(|&l| l as usize)
                                .sum();
                            let strip_len = info
                                .strip_lengths
                                .get(*s as usize)
                                .copied()
                                .unwrap_or(info.positions.len() as u32)
                                as usize;
                            for j in node_start..node_start + strip_len.saturating_sub(1) {
                                if let (Some(&pa), Some(&pb)) =
                                    (info.positions.get(j), info.positions.get(j + 1))
                                {
                                    edge_data.extend_from_slice(&xform(pa));
                                    edge_data.extend_from_slice(&xform(pb));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        } // end if let Some(sel)

        // Append any extra edge segments (e.g. volume AABB outlines from object-level selection).
        edge_data.extend_from_slice(extra_edge_data);

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
            fill_colour,
            edge_colour,
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
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding.clone(),
                }],
            });
            let edge_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sub_hl_edge_bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding.clone(),
                }],
            });
            let sprite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sub_hl_sprite_bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: binding,
                }],
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
