use crate::renderer::{SubObjectRef, SubSelectionRef};
use crate::resources::types::DeviceResources;

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

impl DeviceResources {
    /// Lazily create sub-object highlight pipelines for both the HDR path
    /// (`Rgba16Float` colour target) and the LDR path (swapchain `target_format`).
    /// Idempotent: returns immediately if already created.
    pub(crate) fn ensure_sub_highlight_pipelines(&mut self, device: &crate::gpu::Device) {
        if self.sub_highlight.fill_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let bgl = crate::resources::builders::uniform_bgl(
            device,
            "sub_highlight_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
        );

        let layout = crate::resources::builders::pipeline_layout(
            device,
            "sub_highlight_layout",
            &[&self.camera_bind_group_layout, &bgl],
        );

        let fill_shader = crate::resources::builders::wgsl_module(
            device,
            "sub_highlight_fill_shader",
            crate::resources::builders::wgsl_source!("sub_highlight_fill"),
        );
        let edge_shader = crate::resources::builders::wgsl_module(
            device,
            "sub_highlight_edge_shader",
            crate::resources::builders::wgsl_source!("sub_highlight_edge"),
        );
        let sprite_shader = crate::resources::builders::wgsl_module(
            device,
            "sub_highlight_sprite_shader",
            crate::resources::builders::wgsl_source!("sub_highlight_sprite"),
        );

        // Inline helper: build one fill pipeline for the given colour format.
        let make_fill = |label: &'static str, fmt: crate::gpu::TextureFormat| {
            crate::resources::builders::render_pipeline(
                device,
                crate::resources::builders::RenderPipelineDesc {
                    label,
                    layout: &layout,
                    vertex: crate::gpu::VertexState {
                        module: &fill_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        buffers: &[crate::gpu::VertexBufferLayout {
                            array_stride: 12,
                            step_mode: crate::gpu::VertexStepMode::Vertex,
                            attributes: &crate::gpu::vertex_attr_array![0 => Float32x3],
                        }],
                    },
                    fragment: Some(crate::gpu::FragmentState {
                        module: &fill_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        targets: &[Some(crate::gpu::ColorTargetState {
                            format: fmt,
                            blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                            write_mask: crate::gpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: crate::gpu::PrimitiveState {
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(crate::gpu::DepthStencilState {
                        format: crate::gpu::TextureFormat::Depth24PlusStencil8,
                        depth_write_enabled: crate::resources::builders::dwrite(false),
                        depth_compare: crate::resources::builders::dcompare(
                            crate::gpu::CompareFunction::LessEqual,
                        ),
                        stencil: crate::gpu::StencilState::default(),
                        bias: crate::gpu::DepthBiasState {
                            constant: -2,
                            slope_scale: -1.0,
                            clamp: 0.0,
                        },
                    }),
                    multisample: crate::gpu::MultisampleState {
                        count: 1,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };
        let make_edge = |label: &'static str, fmt: crate::gpu::TextureFormat| {
            crate::resources::builders::render_pipeline(
                device,
                crate::resources::builders::RenderPipelineDesc {
                    label,
                    layout: &layout,
                    vertex: crate::gpu::VertexState {
                        module: &edge_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        buffers: &[crate::gpu::VertexBufferLayout {
                            array_stride: 24,
                            step_mode: crate::gpu::VertexStepMode::Instance,
                            attributes: &crate::gpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                        }],
                    },
                    fragment: Some(crate::gpu::FragmentState {
                        module: &edge_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        targets: &[Some(crate::gpu::ColorTargetState {
                            format: fmt,
                            blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                            write_mask: crate::gpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: crate::gpu::PrimitiveState {
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    // Always pass depth so edge lines are visible even when the
                    // geometry they belong to is a 3D solid (e.g. tube/streamtube
                    // where the control curve sits inside the rendered mesh).
                    depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                        false,
                        crate::gpu::CompareFunction::Always,
                    )),
                    multisample: crate::gpu::MultisampleState {
                        count: 1,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };
        let make_sprite = |label: &'static str, fmt: crate::gpu::TextureFormat| {
            crate::resources::builders::render_pipeline(
                device,
                crate::resources::builders::RenderPipelineDesc {
                    label,
                    layout: &layout,
                    vertex: crate::gpu::VertexState {
                        module: &sprite_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        buffers: &[crate::gpu::VertexBufferLayout {
                            array_stride: 12,
                            step_mode: crate::gpu::VertexStepMode::Instance,
                            attributes: &crate::gpu::vertex_attr_array![0 => Float32x3],
                        }],
                    },
                    fragment: Some(crate::gpu::FragmentState {
                        module: &sprite_shader,
                        entry_point: Some("fs_main"),
                        compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                        targets: &[Some(crate::gpu::ColorTargetState {
                            format: fmt,
                            blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                            write_mask: crate::gpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: crate::gpu::PrimitiveState {
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    // Always pass depth so point sprites are visible even when
                    // the control point is inside a 3D solid (e.g. tube/streamtube).
                    depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                        false,
                        crate::gpu::CompareFunction::Always,
                    )),
                    multisample: crate::gpu::MultisampleState {
                        count: 1,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };

        let ldr_fmt = self.target_format;
        self.sub_highlight.fill_pipeline = Some(make_fill(
            "sub_highlight_fill_hdr",
            crate::gpu::TextureFormat::Rgba16Float,
        ));
        self.sub_highlight.edge_pipeline = Some(make_edge(
            "sub_highlight_edge_hdr",
            crate::gpu::TextureFormat::Rgba16Float,
        ));
        self.sub_highlight.sprite_pipeline = Some(make_sprite(
            "sub_highlight_sprite_hdr",
            crate::gpu::TextureFormat::Rgba16Float,
        ));
        self.sub_highlight.fill_ldr_pipeline = Some(make_fill("sub_highlight_fill_ldr", ldr_fmt));
        self.sub_highlight.edge_ldr_pipeline = Some(make_edge("sub_highlight_edge_ldr", ldr_fmt));
        self.sub_highlight.sprite_ldr_pipeline =
            Some(make_sprite("sub_highlight_sprite_ldr", ldr_fmt));
        self.sub_highlight.bgl = Some(bgl);
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
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        sel: Option<&SubSelectionRef>,
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
                    SubObjectRef::Instance(i) | SubObjectRef::Splat(i) => {
                        // Instanced items (glyphs, tensor glyphs, sprites) and
                        // Gaussian splats highlight as a sprite marker at the
                        // instance position, transformed by the node model.
                        if let Some(positions) = sel.instance_lookup.get(node_id) {
                            if let Some(&pos) = positions.get(*i as usize) {
                                sprite_pos.push(xform(pos));
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
        let make_buf = |label: &str, data: &[u8]| -> crate::gpu::Buffer {
            let size = data.len().max(1) as u64;
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
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
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("sub_hl_uniform"),
            size: std::mem::size_of::<SubHighlightUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::cast_slice(&[uniform]));

        let (fill_bind_group, edge_bind_group, sprite_bind_group) = {
            let bgl = self.sub_highlight.bgl.as_ref().unwrap();
            let binding = uniform_buf.as_entire_binding();
            let fill_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("sub_hl_fill_bg"),
                layout: bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: binding.clone(),
                }],
            });
            let edge_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("sub_hl_edge_bg"),
                layout: bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: binding.clone(),
                }],
            });
            let sprite_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("sub_hl_sprite_bg"),
                layout: bgl,
                entries: &[crate::gpu::BindGroupEntry {
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

/// Per-object outline uniform for the two-pass stencil outline effect.
///
/// Layout (112 bytes):
/// - model:        [[f32;4];4] = 64 bytes
/// - colour:         [f32;4]   = 16 bytes  (outline RGBA)
/// - pixel_offset:  f32       =  4 bytes  (outline ring width in pixels)
/// - _pad:          [f32;3]   = 12 bytes
/// - deform_flags:  u32       =  4 bytes  (bit i set when deformer slot i is active for this draw)
/// - _deform_pad:   [u32;3]   = 12 bytes
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OutlineUniform {
    pub(crate) model: [[f32; 4]; 4],  //  64 bytes
    pub(crate) colour: [f32; 4],      //  16 bytes
    pub(crate) pixel_offset: f32,     //   4 bytes
    pub(crate) _pad: [f32; 3],        //  12 bytes
    pub(crate) deform_flags: u32,     //   4 bytes
    pub(crate) _deform_pad: [u32; 3], //  12 bytes
}

pub(crate) struct OutlineObjectBuffers {
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
    pub two_sided: bool,
    /// Per-instance deformer id for the picked node, or `None` when the node
    /// has no per-instance deformer data. When `Some` and the renderer has
    /// per-instance data for `(mesh_id, instance_id)`, the outline mask is
    /// drawn with that bind group so the selection halo tracks the deformed
    /// silhouette.
    pub deform_instance: Option<u32>,
    pub _mask_uniform_buf: crate::gpu::Buffer,
    pub mask_bind_group: crate::gpu::BindGroup,
}

/// Per-item uniform for the Gaussian splat outline mask pass (112 bytes).
///
/// Padded to 112 bytes to match `OutlineUniform`. Both structs share the same
/// bind group layout (`outline_bgl`) and wgpu enforces the maximum required
/// size across all pipelines using that layout.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SplatOutlineMaskUniform {
    pub(crate) model: [[f32; 4]; 4], // 64 bytes
    pub(crate) viewport_w: f32,      //  4 bytes
    pub(crate) viewport_h: f32,      //  4 bytes
    pub(crate) pixel_radius: f32,    //  4 bytes
    pub(crate) _pad: [f32; 9],       // 36 bytes  (total: 112)
}

/// Per-frame GPU buffers for one selected Gaussian splat set's outline mask draw.
pub(crate) struct SplatOutlineBuffers {
    /// Object-space positions as `[f32; 3]` per splat, instance-stepped.
    pub(crate) position_buf: crate::gpu::Buffer,
    /// Per-instance pixel radius as `f32`, instance-stepped.
    pub(crate) size_buf: crate::gpu::Buffer,
    /// Number of splats (= instance count).
    pub(crate) instance_count: u32,
    /// Uniform buffer kept alive for the duration of the frame.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    /// Bind group for group 1 (SplatOutlineMaskUniform).
    pub(crate) bind_group: crate::gpu::BindGroup,
}

/// Inline geometry outline buffers for flat world-space quads (image slices).
///
/// Unlike `OutlineObjectBuffers`, the vertex/index data is owned here rather than
/// looked up via a `MeshId`.
pub(crate) struct RawGeomOutlineBuffers {
    pub vertex_buf: crate::gpu::Buffer,
    pub index_buf: crate::gpu::Buffer,
    pub index_count: u32,
    pub two_sided: bool,
    pub _uniform_buf: crate::gpu::Buffer,
    pub mask_bind_group: crate::gpu::BindGroup,
}

/// Per-frame outline item for a tube/streamtube/ribbon mesh.
///
/// Holds an index into the per-frame gpu_data array and a mask bind group
/// that supplies an identity model matrix to the outline_mask shader.
pub(crate) struct CurveMeshOutlineItem {
    pub index: usize,
    pub two_sided: bool,
    pub _mask_uniform_buf: crate::gpu::Buffer,
    pub mask_bind_group: crate::gpu::BindGroup,
}

/// NDC-space rect outline for screen image overlays.
pub(crate) struct ScreenRectOutlineBuffers {
    pub _uniform_buf: crate::gpu::Buffer,
    pub bind_group: crate::gpu::BindGroup,
}

/// Uniform for the fullscreen outline edge-detection pass (32 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct OutlineEdgeUniform {
    pub(crate) colour: [f32; 4], // 16 bytes
    pub(crate) radius: f32,      //  4 bytes
    pub(crate) viewport_w: f32,  //  4 bytes
    pub(crate) viewport_h: f32,  //  4 bytes
    pub(crate) _pad: f32,        //  4 bytes
}

/// Per-frame uniform for the sub-object highlight pass (48 bytes).
///
/// Shared by the fill, edge, and sprite draw calls.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SubHighlightUniform {
    pub(crate) fill_colour: [f32; 4], // 16 bytes
    pub(crate) edge_colour: [f32; 4], // 16 bytes
    pub(crate) edge_width: f32,       //  4 bytes (pixels)
    pub(crate) vertex_size: f32,      //  4 bytes (pixels)
    pub(crate) viewport_width: f32,   //  4 bytes
    pub(crate) viewport_height: f32,  //  4 bytes
                                      // total 48 bytes
}

/// GPU buffers for one frame of sub-object highlight rendering.
///
/// Rebuilt whenever [`InteractionFrame::sub_selection`] version changes.
/// All three passes (fill, edges, sprites) share a single
/// [`SubHighlightUniform`] buffer bound at group 1.
pub(crate) struct SubHighlightGpuData {
    // Face fill : flat triangle vertex list (xyz f32, 12 bytes each, non-indexed).
    pub(crate) fill_vertex_buf: crate::gpu::Buffer,
    pub(crate) fill_vertex_count: u32,
    // Edge lines : segment instances (pos_a xyz + pos_b xyz, 24 bytes each).
    pub(crate) edge_vertex_buf: crate::gpu::Buffer,
    pub(crate) edge_segment_count: u32,
    // Vertex / point sprites : positions (xyz padded to 16 bytes).
    pub(crate) sprite_vertex_buf: crate::gpu::Buffer,
    pub(crate) sprite_point_count: u32,
    // Shared uniform buffer.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    // Per-pass bind groups (group 1: SubHighlightUniform).
    pub(crate) fill_bind_group: crate::gpu::BindGroup,
    pub(crate) edge_bind_group: crate::gpu::BindGroup,
    pub(crate) sprite_bind_group: crate::gpu::BindGroup,
}
