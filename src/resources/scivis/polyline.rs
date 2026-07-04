use super::*;

/// Polyline (screen-space thick line) pipelines and their layouts. All lazily
/// built; the uploaded polyline data lives in a separate flat store.
#[derive(Default)]
pub(crate) struct PolylineResources {
    /// Polyline render pipeline. None until first polyline set is submitted.
    pub(crate) pipeline: Option<DualPipeline>,
    /// Clip-exempt polyline pipeline (uses fs_main_no_clip).
    pub(crate) no_clip_pipeline: Option<DualPipeline>,
    /// Bind group layout for polyline uniforms (group 1).
    pub(crate) bgl: Option<wgpu::BindGroupLayout>,
    /// Thin 1px LineList wireframe polyline pipeline.
    pub(crate) wireframe_pipeline: Option<DualPipeline>,
    /// Bind group layout for the wireframe polyline pipeline (group 1).
    pub(crate) wireframe_bgl: Option<wgpu::BindGroupLayout>,
    /// Polyline outline mask pipeline (R8Unorm). None until first selected polyline.
    pub(crate) outline_mask_pipeline: Option<wgpu::RenderPipeline>,
}

impl DeviceResources {
    /// Lazily create the polyline render pipeline (instanced TriangleList : screen-space thick lines).
    ///
    /// No-op if already created. Called from `prepare()` when `frame.scene.polylines` is non-empty.
    pub(crate) fn ensure_polyline_pipeline(&mut self, device: &wgpu::Device) {
        if self.polyline.pipeline.is_some() {
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
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/polyline.wgsl")).into(),
            ),
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

        self.polyline.bgl = Some(pl_bgl);
        self.polyline.pipeline = Some(DualPipeline {
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
        if self.polyline.wireframe_pipeline.is_some() {
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
                include_str!(concat!(env!("OUT_DIR"), "/polyline_wireframe.wgsl")).into(),
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

        self.polyline.wireframe_bgl = Some(wf_bgl);
        self.polyline.wireframe_pipeline = Some(DualPipeline {
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
    pub(crate) fn upload_polyline_per_frame(
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
            _pad: u32,              // offset 108
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
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
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
            model: [[f32; 4]; 4],     // offset  0
            default_colour: [f32; 4], // offset 64
            line_width: f32,          // offset 80
            scalar_min: f32,          // offset 84
            scalar_max: f32,          // offset 88
            has_scalar: u32,          // offset 92
            viewport_width: f32,      // offset 96
            viewport_height: f32,     // offset 100
            _pad: [f32; 2],           // offset 104 (total 112 bytes)
        }
        let uniform_data = PolylineUniform {
            model: item.model,
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
            .content
            .builtin_colourmap_ids
            .and_then(|ids| {
                let preset_id = item
                    .colourmap_id
                    .unwrap_or(ids[crate::resources::BuiltinColourmap::Viridis as usize]);
                self.content.colourmap_views.get(preset_id.0)
            })
            .unwrap_or(&self.content.fallback_lut_view);

        let lut_sampler = &self.lut_sampler;

        let bgl = self
            .polyline
            .bgl
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

        let wireframe_bind_group = self.polyline.wireframe_bgl.as_ref().map(|bgl| {
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item: &crate::renderer::PolylineItem,
    ) -> crate::resources::PolylineId {
        self.ensure_polyline_pipeline(device);
        let gpu = self.upload_polyline_per_frame(device, queue, item, [1.0, 1.0]);
        self.content.polyline_store.insert(gpu)
    }

    /// Remove a pre-uploaded polyline. Returns `true` if a polyline was
    /// actually removed, `false` if the id was already invalid.
    pub fn drop_polyline(&mut self, id: crate::resources::PolylineId) -> bool {
        self.content.polyline_store.remove(id)
    }

    /// Replace the geometry of a pre-uploaded polyline, keeping the same
    /// [`PolylineId`](crate::resources::PolylineId).
    ///
    /// Returns `true` if the id was valid and the polyline was replaced,
    /// `false` if the slot was empty (call [`upload_polyline`](Self::upload_polyline) instead).
    pub fn replace_polyline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: crate::resources::PolylineId,
        item: &crate::renderer::PolylineItem,
    ) -> bool {
        if !self.content.polyline_store.contains(id) {
            return false;
        }
        self.ensure_polyline_pipeline(device);
        let gpu = self.upload_polyline_per_frame(device, queue, item, [1.0, 1.0]);
        self.content.polyline_store.replace(id, gpu)
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
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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

    /// Lazily create the clip-exempt polyline pipeline.
    ///
    /// Identical to the regular polyline pipeline but uses `fs_main_no_clip` so
    /// fragments are never discarded by clip planes or clip volumes. Used for
    /// clip object wireframe overlays which must always be fully visible.
    pub(crate) fn ensure_polyline_no_clip_pipeline(&mut self, device: &wgpu::Device) {
        if self.polyline.no_clip_pipeline.is_some() {
            return;
        }
        // The regular pipeline (and its BGL) must exist first.
        self.ensure_polyline_pipeline(device);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("polyline_no_clip_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/polyline.wgsl")).into(),
            ),
        });

        let pl_bgl = self
            .polyline
            .bgl
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

        self.polyline.no_clip_pipeline = Some(DualPipeline {
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
        if self.polyline.outline_mask_pipeline.is_some() {
            return;
        }
        self.ensure_polyline_pipeline(device);

        let pl_bgl = self
            .polyline
            .bgl
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
                include_str!(concat!(env!("OUT_DIR"), "/polyline_outline_mask.wgsl")).into(),
            ),
        });

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

        self.polyline.outline_mask_pipeline = Some(device.create_render_pipeline(
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

#[cfg(test)]
mod tests {
    use crate::DeviceResources;
    use crate::renderer::PolylineItem;
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

    fn sample_polyline() -> PolylineItem {
        PolylineItem {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            strip_lengths: vec![3],
            ..Default::default()
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
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
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
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

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
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
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
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);
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
