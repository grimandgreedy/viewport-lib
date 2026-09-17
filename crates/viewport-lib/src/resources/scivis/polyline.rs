use super::*;

/// Polyline pipeline variant axes: clip-exemption and thin-wireframe vs
/// thick-billboard geometry. The two axes select genuinely different shader
/// modules and bind group layouts (wireframe reads segment data from a
/// storage buffer with no vertex buffer; the thick path uses an instanced
/// vertex buffer plus a texture+sampler bind group), unlike the mesh
/// family's `PipelineKey` axes, which all reuse one shader/layout. `build`
/// picks the shader/layout pair on `wireframe` and the fragment entry point
/// on `skip_clip`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PolylineKey {
    pub skip_clip: bool,
    pub wireframe: bool,
}

impl PolylineKey {
    /// Every axis combination, for eager cross-product construction
    /// (`PolylineVariantSet::build`).
    pub fn all() -> impl Iterator<Item = PolylineKey> {
        (0u8..4).map(|bits| PolylineKey {
            skip_clip: bits & 1 != 0,
            wireframe: bits & 2 != 0,
        })
    }

    fn slot(self) -> usize {
        (self.skip_clip as usize) | (self.wireframe as usize) << 1
    }
}

/// A `DualPipeline` built for every reachable [`PolylineKey`], indexed for a
/// hash-free draw-time lookup (`get`).
#[derive(Clone)]
pub(crate) struct PolylineVariantSet {
    variants: [DualPipeline; 4],
}

impl PolylineVariantSet {
    pub fn build(mut build: impl FnMut(PolylineKey) -> DualPipeline) -> Self {
        let mut variants: Vec<DualPipeline> = Vec::with_capacity(4);
        for key in PolylineKey::all() {
            variants.push(build(key));
        }
        Self {
            variants: variants
                .try_into()
                .unwrap_or_else(|_| unreachable!("PolylineKey::all() yields exactly 4 keys")),
        }
    }

    pub fn get(&self, key: PolylineKey) -> &DualPipeline {
        &self.variants[key.slot()]
    }
}

#[cfg(test)]
mod polyline_key_tests {
    use super::*;

    #[test]
    fn all_keys_are_distinct_and_densely_slotted() {
        let keys: Vec<PolylineKey> = PolylineKey::all().collect();
        assert_eq!(keys.len(), 4, "PolylineKey has 2 bool axes: 2^2 = 4 keys");

        let mut seen_keys = std::collections::HashSet::new();
        let mut seen_slots = std::collections::HashSet::new();
        for key in keys {
            assert!(
                seen_keys.insert(key),
                "all() yielded {key:?} more than once"
            );
            let slot = key.slot();
            assert!(slot < 4, "{key:?} slotted out of range: {slot}");
            assert!(
                seen_slots.insert(slot),
                "{key:?} collided with another key at slot {slot}"
            );
        }
    }
}

/// Polyline (screen-space thick line) pipelines and their layouts.
///
/// This is the shared line substrate: besides the polyline item type, isolines,
/// scatter-volume bounds, volume bounding boxes, clip-object outlines and the
/// splat and sprite wireframe overlays all render through it. The layouts are
/// created up front because uploads bind against them; the pipelines are still
/// built on first use, but through a shared reference, so the polyline item
/// type's plugin can reach them from `prepare`, which holds
/// `&DeviceResources`.
pub(crate) struct PolylineResources {
    /// Polyline render pipelines, keyed by `PolylineKey`.
    pub(crate) pipelines: std::sync::OnceLock<PolylineVariantSet>,
    /// Bind group layout for polyline uniforms (group 1).
    pub(crate) bgl: crate::gpu::BindGroupLayout,
    /// Bind group layout for the wireframe polyline pipeline (group 1).
    pub(crate) wireframe_bgl: crate::gpu::BindGroupLayout,
}

impl PolylineResources {
    pub(crate) fn new(device: &crate::gpu::Device) -> Self {
        let bgl = crate::resources::builders::uniform_texture_sampler_bgl(
            device,
            "polyline_bgl",
            crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            crate::gpu::ShaderStages::VERTEX,
        );
        let wireframe_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("polyline_wireframe_bgl"),
                entries: &[crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        Self {
            pipelines: std::sync::OnceLock::new(),
            bgl,
            wireframe_bgl,
        }
    }
}

impl DeviceResources {
    /// Build (on first call) and return the polyline render pipelines.
    ///
    /// Takes a shared reference so the polyline item type's plugin can reach
    /// them from `prepare`, alongside the core producers that render through
    /// the same substrate.
    pub(crate) fn ensure_polyline_pipeline(
        &self,
        device: &crate::gpu::Device,
    ) -> &PolylineVariantSet {
        if let Some(set) = self.polyline.pipelines.get() {
            return set;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        let pl_bgl = &self.polyline.bgl;

        let shader = crate::resources::builders::wgsl_module(
            device,
            "polyline_shader",
            crate::resources::builders::wgsl_source!("polyline"),
        );

        let layout = crate::resources::builders::standard_scene_layout(
            device,
            "polyline_pipeline_layout",
            &self.binds.camera_bgl,
            pl_bgl,
        );

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
        //   offset 108: dist_a            f32   : cumulative arc length at segment start (for dashing)
        let pl_instance_layout = crate::gpu::VertexBufferLayout {
            array_stride: 112,
            step_mode: crate::gpu::VertexStepMode::Instance,
            attributes: &[
                crate::gpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: crate::gpu::VertexFormat::Float32x3,
                }, // pos_a
                crate::gpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: crate::gpu::VertexFormat::Float32x3,
                }, // pos_b
                crate::gpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: crate::gpu::VertexFormat::Float32x3,
                }, // prev_pos
                crate::gpu::VertexAttribute {
                    offset: 36,
                    shader_location: 3,
                    format: crate::gpu::VertexFormat::Float32x3,
                }, // next_pos
                crate::gpu::VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: crate::gpu::VertexFormat::Float32,
                }, // scalar_a
                crate::gpu::VertexAttribute {
                    offset: 52,
                    shader_location: 5,
                    format: crate::gpu::VertexFormat::Float32,
                }, // scalar_b
                crate::gpu::VertexAttribute {
                    offset: 56,
                    shader_location: 6,
                    format: crate::gpu::VertexFormat::Uint32,
                }, // has_prev
                crate::gpu::VertexAttribute {
                    offset: 60,
                    shader_location: 7,
                    format: crate::gpu::VertexFormat::Uint32,
                }, // has_next
                crate::gpu::VertexAttribute {
                    offset: 64,
                    shader_location: 8,
                    format: crate::gpu::VertexFormat::Float32x4,
                }, // colour_a
                crate::gpu::VertexAttribute {
                    offset: 80,
                    shader_location: 9,
                    format: crate::gpu::VertexFormat::Float32x4,
                }, // colour_b
                crate::gpu::VertexAttribute {
                    offset: 96,
                    shader_location: 10,
                    format: crate::gpu::VertexFormat::Float32,
                }, // radius_a
                crate::gpu::VertexAttribute {
                    offset: 100,
                    shader_location: 11,
                    format: crate::gpu::VertexFormat::Float32,
                }, // radius_b
                crate::gpu::VertexAttribute {
                    offset: 104,
                    shader_location: 12,
                    format: crate::gpu::VertexFormat::Uint32,
                }, // use_direct_colour
                crate::gpu::VertexAttribute {
                    offset: 108,
                    shader_location: 13,
                    format: crate::gpu::VertexFormat::Float32,
                }, // dist_a
            ],
        };

        let wf_bgl = &self.polyline.wireframe_bgl;

        let wf_shader = crate::resources::builders::wgsl_module(
            device,
            "polyline_wireframe_shader",
            crate::resources::builders::wgsl_source!("polyline_wireframe"),
        );

        let wf_layout = crate::resources::builders::standard_scene_layout(
            device,
            "polyline_wireframe_pipeline_layout",
            &self.binds.camera_bgl,
            wf_bgl,
        );

        let sample_count = self.sample_count;
        let ldr_format = self.target_format;

        // One `PolylineVariantSet::build` covers all 4 (skip_clip x wireframe)
        // combinations: `wireframe` picks the shader/layout/vertex-buffer/
        // topology triple, `skip_clip` picks the fragment entry point within
        // whichever shader that is (both `polyline.wgsl` and
        // `polyline_wireframe.wgsl` export `fs_main` / `fs_main_no_clip`).
        let built = crate::resources::scivis::polyline::PolylineVariantSet::build(|key| {
            let fragment_entry = if key.skip_clip {
                "fs_main_no_clip"
            } else {
                "fs_main"
            };
            if key.wireframe {
                crate::resources::builders::build_dual_pipeline(
                    device,
                    &crate::resources::builders::DualPipelineDesc {
                        label: "polyline_wireframe_pipeline_variant",
                        layout: &wf_layout,
                        shader: &wf_shader,
                        vertex_entry: "vs_main",
                        fragment_entry,
                        vertex_buffers: &[],
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        topology: crate::gpu::PrimitiveTopology::LineList,
                        cull_mode: None,
                        depth_write: true,
                        depth_compare: crate::gpu::CompareFunction::LessEqual,
                        sample_count,
                        ldr_format,
                    },
                )
            } else {
                crate::resources::builders::build_dual_pipeline(
                    device,
                    &crate::resources::builders::DualPipelineDesc {
                        label: "polyline_pipeline_variant",
                        layout: &layout,
                        shader: &shader,
                        vertex_entry: "vs_main",
                        fragment_entry,
                        vertex_buffers: &[pl_instance_layout.clone()],
                        blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                        topology: crate::gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        depth_write: true,
                        depth_compare: crate::gpu::CompareFunction::LessEqual,
                        sample_count,
                        ldr_format,
                    },
                )
            }
        });
        self.polyline.pipelines.get_or_init(|| built)
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
    pub(crate) fn upload_polyline_per_frame(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::PolylineItem,
        viewport_size: [f32; 2],
    ) -> PolylineGpuData {
        // Build the segment instance buffer (112 bytes per segment).
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SegInstance {
            pos_a: [f32; 3],        // offset   0
            pos_b: [f32; 3],        // offset  12
            prev_pos: [f32; 3],     // offset  24
            next_pos: [f32; 3],     // offset  36
            scalar_a: f32,          // offset  48
            scalar_b: f32,          // offset  52
            has_prev: u32,          // offset  56
            has_next: u32,          // offset  60
            colour_a: [f32; 4],     // offset  64
            colour_b: [f32; 4],     // offset  80
            radius_a: f32,          // offset  96
            radius_b: f32,          // offset 100
            use_direct_colour: u32, // offset 104
            dist_a: f32,            // offset 108 : cumulative arc length at segment start
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
            // Cumulative arc length along this strip, in the units of `positions`
            // (world space, ignoring the per-item model matrix). Used by the
            // fragment shader for dash/dot placement. Reset per strip so the
            // pattern restarts at each strip's first node.
            let mut cum_dist = 0.0f32;
            for i in strip_start..end.saturating_sub(1) {
                let j = i + 1;
                let has_prev = i > strip_start;
                let has_next = j + 1 < end;
                let dist_a = cum_dist;
                let pa = positions[i];
                let pb = positions[j];
                cum_dist +=
                    ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2) + (pb[2] - pa[2]).powi(2))
                        .sqrt();

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
                        item.node_colours
                            .get(i)
                            .map(|c| c.to_linear_rgba())
                            .unwrap_or([1.0; 4]),
                        item.node_colours
                            .get(j)
                            .map(|c| c.to_linear_rgba())
                            .unwrap_or([1.0; 4]),
                    )
                } else if !item.edge_colours.is_empty() {
                    let c = item
                        .edge_colours
                        .get(seg_idx_global)
                        .map(|c| c.to_linear_rgba())
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
                    dist_a,
                });

                seg_idx_global += 1;
            }
        }

        let seg_count = instances.len() as u32;

        // Allocate instance buffer (min 112 bytes so wgpu doesn't complain on empty).
        let seg_bytes: &[u8] = bytemuck::cast_slice(&instances);
        let vertex_buffer = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("polyline_vertex_buf"),
            size: seg_bytes.len().max(112) as u64,
            usage: crate::gpu::BufferUsages::VERTEX
                | crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::COPY_DST,
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

        // Map the stroke pattern to shader parameters. Cadence is in the units
        // of `positions` (world-space arc length). Dotted is expressed as a
        // short dash so the same on/period machinery covers all three cases.
        use crate::StrokePattern;
        let (dash_mode, dash_on, dash_period, dash_offset) = match item.stroke_pattern {
            StrokePattern::Solid => (0u32, 0.0f32, 0.0f32, 0.0f32),
            StrokePattern::Dashed {
                dash_length,
                gap_length,
                offset,
            } => (
                1,
                dash_length.max(0.0),
                (dash_length + gap_length).max(0.0),
                offset,
            ),
            StrokePattern::Dotted { spacing, offset } => {
                let period = spacing.max(0.0);
                (1, (period * 0.25).max(0.0), period, offset)
            }
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PolylineUniform {
            model: [[f32; 4]; 4],     // offset   0
            default_colour: [f32; 4], // offset  64
            line_width: f32,          // offset  80
            scalar_min: f32,          // offset  84
            scalar_max: f32,          // offset  88
            has_scalar: u32,          // offset  92
            viewport_width: f32,      // offset  96
            viewport_height: f32,     // offset 100
            dash_mode: u32,           // offset 104 : 0 = solid, 1 = dashed/dotted
            dash_on: f32,             // offset 108 : visible run length (arc units)
            dash_period: f32,         // offset 112 : on + off run length
            dash_offset: f32,         // offset 116 : phase shift along the line
            _pad: [f32; 2],           // offset 120 (total 128 bytes)
        }
        let uniform_data = PolylineUniform {
            model: item.model,
            default_colour: item.default_colour.to_linear_rgba(),
            line_width: item.line_width,
            scalar_min,
            scalar_max,
            has_scalar,
            viewport_width: viewport_size[0].max(1.0),
            viewport_height: viewport_size[1].max(1.0),
            dash_mode,
            dash_on,
            dash_period,
            dash_offset,
            _pad: [0.0; 2],
        };
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("polyline_uniform_buf"),
            size: std::mem::size_of::<PolylineUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniform_data));

        let lut_view = self
            .content
            .builtin_colourmap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colourmap_id
                    .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                self.content.colourmap_views.get(preset_id.0)
            })
            .unwrap_or(&self.content.fallback_lut_view);

        let lut_sampler = &self.material.lut_sampler;

        let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("polyline_bind_group"),
            layout: &self.polyline.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: crate::gpu::BindingResource::TextureView(lut_view),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: crate::gpu::BindingResource::Sampler(lut_sampler),
                },
            ],
        });

        let wireframe_bind_group =
            Some(device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("polyline_wireframe_bind_group"),
                layout: &self.polyline.wireframe_bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                }],
            }));

        PolylineGpuData {
            pick_id: item.settings.pick_id,
            strip_lengths: item.strip_lengths.clone(),
            vertex_buffer,
            segment_count: seg_count,
            bind_group,
            _uniform_buf: uniform_buf,
            skip_clip: false,
            wireframe: false,
            wireframe_bind_group,
        }
    }

    /// Pre-upload a polyline and return a typed handle.
    ///
    /// The returned [`PolylineId`](crate::resources::PolylineId) refers to GPU
    /// buffers retained by the renderer until [`drop_polyline`] is called.
    /// Submit a [`PolylineRefItem`](crate::renderer::PolylineRefItem) on
    /// `SceneFrame::polyline_refs` each frame to draw the polyline at a
    /// custom model transform without rebuilding its segment buffer.
    ///
    /// The viewport size used for screen-space miter calculations is set
    /// from the most recent ref-item draw of this polyline. Stationary
    /// callers can rely on it being correct after the first frame.
    pub fn upload_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: &crate::renderer::PolylineItem,
    ) -> crate::resources::PolylineId {
        self.ensure_polyline_pipeline(device);
        let gpu = self.upload_polyline_per_frame(device, queue, item, [1.0, 1.0]);
        self.content.polyline_store.insert_sized(gpu)
    }

    /// Remove a pre-uploaded polyline. Returns `true` if a polyline was
    /// actually removed, `false` if the id was already invalid.
    pub fn drop_polyline(&mut self, id: crate::resources::PolylineId) -> bool {
        self.content.polyline_store.remove(id).is_some()
    }

    /// Replace the geometry of a pre-uploaded polyline, keeping the same
    /// [`PolylineId`](crate::resources::PolylineId).
    ///
    /// Returns `true` if the id was valid and the polyline was replaced,
    /// `false` if the slot was empty (call [`upload_polyline`](Self::upload_polyline) instead).
    pub fn replace_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::resources::PolylineId,
        item: &crate::renderer::PolylineItem,
    ) -> bool {
        if !self.content.polyline_store.contains(id) {
            return false;
        }
        self.ensure_polyline_pipeline(device);
        let gpu = self.upload_polyline_per_frame(device, queue, item, [1.0, 1.0]);
        self.content.polyline_store.replace_sized(id, gpu).is_some()
    }

    /// Start an asynchronous polyline upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. The upload
    /// runs during the next `process_uploads` call (driven by `prepare_scene`);
    /// once the status is `Ready`, call
    /// [`upload_result_polyline`](Self::upload_result_polyline) to take the
    /// resulting handle.
    ///
    /// Ownership of `item` transfers into the worker.
    pub fn begin_upload_polyline(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        item: crate::renderer::PolylineItem,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<crate::resources::PolylineId>::new();
        let slot_for_apply = slot.clone();
        let device_for_apply = device.clone();
        let queue_for_apply = queue.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.9);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let pid =
                            resources.upload_polyline(&device_for_apply, &queue_for_apply, &item);
                        slot_for_apply.set(pid);
                    }),
                ))
            })
        };

        self.job_results
            .polyline
            .lock()
            .expect("polyline result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`PolylineId`](crate::resources::PolylineId) produced by a
    /// completed [`begin_upload_polyline`](Self::begin_upload_polyline) job.
    pub fn upload_result_polyline(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<crate::resources::PolylineId> {
        let mut map = self
            .job_results
            .polyline
            .lock()
            .expect("polyline result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(pid) => {
                map.remove(&id);
                Ok(pid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PolylineKey;
    use crate::DeviceResources;
    use crate::renderer::PolylineItem;
    use crate::resources::UploadStatus;

    fn try_make_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
        let instance = crate::gpu::default_instance();
        let adapter = pollster::block_on(instance.request_adapter(
            &crate::gpu::RequestAdapterOptions {
                power_preference: crate::gpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
                #[cfg(wgpu30)]
                apply_limit_buckets: false,
            },
        ))
        .ok()?;
        pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor::default())).ok()
    }

    fn sample_polyline() -> PolylineItem {
        PolylineItem {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            strip_lengths: vec![3],
            ..Default::default()
        }
    }

    /// Same completeness guarantee as the mesh-family `PipelineVariantSet`
    /// tests: once built, every key in `PolylineKey::all()` must resolve
    /// through `get()` without panicking. Covers the `skip_clip x wireframe`
    /// cross product that the two pipelines used to build separately and
    /// combine with an "wireframe wins" shortcut.
    #[test]
    fn polyline_pipelines_resolve_every_key_once_built() {
        let Some((device, _queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let resources = DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let pipelines = resources.ensure_polyline_pipeline(&device);
        for key in PolylineKey::all() {
            let _ = pipelines.get(key);
        }
    }

    #[test]
    fn default_model_is_identity() {
        let item = PolylineItem::default();
        let expected = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(item.model, expected);
    }

    #[test]
    fn upload_polyline_returns_valid_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_polyline(&device, &queue, &sample_polyline());
        assert!(resources.content.polyline_store.contains(id));
        // drop + reupload cycles the slot.
        assert!(resources.drop_polyline(id));
        assert!(!resources.content.polyline_store.contains(id));
    }

    #[test]
    fn stale_polyline_handle_does_not_alias_after_slot_reuse() {
        // The curve stores share one macro, so this covers the whole family
        // (tube, ribbon, glyph set, sprite set, and the rest).
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);

        let id1 = resources.upload_polyline(&device, &queue, &sample_polyline());
        assert!(resources.content.polyline_store.get(id1).is_some());
        assert!(resources.drop_polyline(id1));
        assert!(
            resources.content.polyline_store.get(id1).is_none(),
            "a dropped handle must not resolve"
        );

        let id2 = resources.upload_polyline(&device, &queue, &sample_polyline());
        assert_eq!(id1.index(), id2.index(), "the freed slot should be reused");
        assert_ne!(id1, id2, "the reused slot must carry a new generation");
        assert!(resources.content.polyline_store.get(id2).is_some());
        assert!(
            resources.content.polyline_store.get(id1).is_none(),
            "the stale handle must not alias the polyline now in its slot"
        );
    }

    #[test]
    fn replace_polyline_keeps_handle_stable() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let id = resources.upload_polyline(&device, &queue, &sample_polyline());
        let mut updated = sample_polyline();
        updated.line_width = 5.0;
        assert!(resources.replace_polyline(&device, &queue, id, &updated));
        assert!(resources.content.polyline_store.contains(id));
    }

    #[test]
    fn begin_upload_polyline_drains_to_handle() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources =
            DeviceResources::new(&device, crate::gpu::TextureFormat::Rgba8UnormSrgb, 1);
        let job = resources.begin_upload_polyline(&device, &queue, sample_polyline());

        // Not ready yet.
        let err = resources.upload_result_polyline(job).unwrap_err();
        assert!(matches!(err, crate::error::ViewportError::JobNotReady));

        for _ in 0..200 {
            resources.process_uploads(&device, &queue);
            match resources.upload_status(job) {
                UploadStatus::Ready => break,
                UploadStatus::Failed(e) => panic!("polyline upload failed: {e:?}"),
                UploadStatus::Pending { .. } => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                UploadStatus::Unknown => panic!("polyline job id disappeared"),
            }
        }

        let id = resources.upload_result_polyline(job).expect("ready result");
        assert!(resources.content.polyline_store.contains(id));

        // Second take of the same id should now report missing.
        let err = resources.upload_result_polyline(job).unwrap_err();
        assert!(matches!(
            err,
            crate::error::ViewportError::JobResultMissing { .. }
        ));
    }

    #[test]
    fn non_identity_model_is_carried_on_item() {
        // A translation of (3, 4, 5) in the last column. Round-trips through
        // the public field so consumers can set it before upload.
        let m = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [3.0, 4.0, 5.0, 1.0],
        ];
        let item = PolylineItem {
            model: m,
            ..PolylineItem::default()
        };
        assert_eq!(item.model[3], [3.0, 4.0, 5.0, 1.0]);
    }
}

/// Per-frame GPU data for one polyline item, created in `prepare()`.
#[derive(Clone)]
pub struct PolylineGpuData {
    /// Object-level pick id for this polyline (from the item's `settings.pick_id`);
    /// `PickId::NONE` when not pickable.
    pub(crate) pick_id: crate::PickId,
    /// Per-strip vertex counts, copied from the source item at build time. A GPU
    /// pick reads the hit segment from the primitive channel; `strip_for_segment`
    /// maps that segment to its strip using these counts, so `SubObjectRef::Strip`
    /// resolves without touching the per-frame `pick_polyline_items` cache. Empty
    /// for a single-strip polyline (all segments belong to strip 0).
    pub(crate) strip_lengths: Vec<u32>,
    /// Instance buffer: `[xa, ya, za, xb, yb, zb, scalar_a, scalar_b]` per segment (32 bytes).
    pub(crate) vertex_buffer: crate::gpu::Buffer,
    /// Number of line segments (instances).  Draw call: `draw(0..6, 0..segment_count)`.
    pub(crate) segment_count: u32,
    /// Bind group (group 1): uniform + LUT texture + sampler.
    pub(crate) bind_group: crate::gpu::BindGroup,
    // Keep the uniform buffer alive for the lifetime of this struct.
    pub(crate) _uniform_buf: crate::gpu::Buffer,
    /// When true, renders with the clip-exempt pipeline (no clip plane or clip volume test).
    /// Used for clip object wireframe overlays that must always be fully visible.
    pub(crate) skip_clip: bool,
    /// When true, render as thin 1px lines using the wireframe pipeline instead of thick billboards.
    pub(crate) wireframe: bool,
    /// Bind group for the wireframe pipeline (group 1: segment storage buffer).
    /// None when the wireframe pipeline has not been created yet.
    pub(crate) wireframe_bind_group: Option<crate::gpu::BindGroup>,
}
