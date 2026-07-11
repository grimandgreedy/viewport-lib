//! GPU compute-filter pipeline for Clip/Threshold index compaction.

/// Output from a single GPU compute filter dispatch.
///
/// Contains a compacted index buffer (triangles that passed the filter)
/// and the count of valid indices. The renderer swaps this in during draw.
pub struct ComputeFilterResult {
    /// Output index buffer containing only passing triangles.
    pub index_buffer: wgpu::Buffer,
    /// Number of valid indices in `index_buffer` (may be 0 if all filtered).
    pub index_count: u32,
    /// `MeshId` this result corresponds to.
    pub mesh_id: crate::resources::mesh::mesh_store::MeshId,
}

impl crate::resources::DeviceResources {
    fn ensure_compute_filter_pipeline(&mut self, device: &wgpu::Device) {
        if self.compute_filter_pipeline.is_some() {
            return;
        }
        self.note_pipeline_built(concat!(file!(), ":", line!()));

        // Build bind group layout.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_filter_bgl"),
            entries: &[
                // binding 0: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: vertices (f32 storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: source indices (u32 storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: scalars (f32 storage, read) : dummy for Clip
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: output compacted indices (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: atomic counter (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_filter_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = crate::resources::builders::wgsl_module(
            device,
            "compute_filter_shader",
            crate::resources::builders::wgsl_source!("compute_filter"),
        );

        let pipeline = crate::resources::builders::compute_pipeline(
            device,
            "compute_filter_pipeline",
            &pipeline_layout,
            &shader,
            "main",
        );

        self.compute_filter_bgl = Some(bgl);
        self.compute_filter_pipeline = Some(pipeline);
    }

    /// Dispatch GPU compute filters for all items in the list.
    ///
    /// Returns one [`ComputeFilterResult`] per item. The renderer uses these
    /// during `paint()` to override the mesh's default index buffer.
    ///
    /// This is a synchronous v1 implementation: it submits each dispatch
    /// individually and polls the device to read back the counter. This is
    /// acceptable for v1; async readback can be added later.
    pub fn run_compute_filters(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        items: &[crate::renderer::ComputeFilterItem],
    ) -> Vec<ComputeFilterResult> {
        if items.is_empty() {
            return Vec::new();
        }

        self.ensure_compute_filter_pipeline(device);

        // Dummy 4-byte buffer used as the scalar binding when doing a Clip filter.
        let dummy_scalar_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compute_filter_dummy_scalar"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut results = Vec::with_capacity(items.len());

        for item in items {
            // Resolve the mesh.
            let gpu_mesh = match self.mesh_store.get(item.mesh_id) {
                Some(m) => m,
                None => continue,
            };

            let triangle_count = gpu_mesh.index_count / 3;
            if triangle_count == 0 {
                continue;
            }

            // Vertex stride: the Vertex struct is 64 bytes = 16 f32s.
            const VERTEX_STRIDE_F32: u32 = 16;

            // Build params uniform matching compute_filter.wgsl Params struct layout.
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct FilterParams {
                mode: u32,
                clip_type: u32,
                threshold_min: f32,
                threshold_max: f32,
                triangle_count: u32,
                vertex_stride_f32: u32,
                _pad: [u32; 2],
                // Plane params
                plane_nx: f32,
                plane_ny: f32,
                plane_nz: f32,
                plane_dist: f32,
                // Box params
                box_cx: f32,
                box_cy: f32,
                box_cz: f32,
                _padb0: f32,
                box_hex: f32,
                box_hey: f32,
                box_hez: f32,
                _padb1: f32,
                box_col0x: f32,
                box_col0y: f32,
                box_col0z: f32,
                _padb2: f32,
                box_col1x: f32,
                box_col1y: f32,
                box_col1z: f32,
                _padb3: f32,
                box_col2x: f32,
                box_col2y: f32,
                box_col2z: f32,
                _padb4: f32,
                // Sphere params
                sphere_cx: f32,
                sphere_cy: f32,
                sphere_cz: f32,
                sphere_radius: f32,
            }

            let mut params: FilterParams = bytemuck::Zeroable::zeroed();
            params.triangle_count = triangle_count;
            params.vertex_stride_f32 = VERTEX_STRIDE_F32;

            match item.kind {
                crate::renderer::ComputeFilterKind::Clip {
                    plane_normal,
                    plane_dist,
                } => {
                    params.mode = 0;
                    params.clip_type = 1;
                    params.plane_nx = plane_normal[0];
                    params.plane_ny = plane_normal[1];
                    params.plane_nz = plane_normal[2];
                    params.plane_dist = plane_dist;
                }
                crate::renderer::ComputeFilterKind::ClipBox {
                    center,
                    half_extents,
                    orientation,
                } => {
                    params.mode = 0;
                    params.clip_type = 2;
                    params.box_cx = center[0];
                    params.box_cy = center[1];
                    params.box_cz = center[2];
                    params.box_hex = half_extents[0];
                    params.box_hey = half_extents[1];
                    params.box_hez = half_extents[2];
                    params.box_col0x = orientation[0][0];
                    params.box_col0y = orientation[0][1];
                    params.box_col0z = orientation[0][2];
                    params.box_col1x = orientation[1][0];
                    params.box_col1y = orientation[1][1];
                    params.box_col1z = orientation[1][2];
                    params.box_col2x = orientation[2][0];
                    params.box_col2y = orientation[2][1];
                    params.box_col2z = orientation[2][2];
                }
                crate::renderer::ComputeFilterKind::ClipSphere { center, radius } => {
                    params.mode = 0;
                    params.clip_type = 3;
                    params.sphere_cx = center[0];
                    params.sphere_cy = center[1];
                    params.sphere_cz = center[2];
                    params.sphere_radius = radius;
                }
                crate::renderer::ComputeFilterKind::Threshold { min, max } => {
                    params.mode = 1;
                    params.threshold_min = min;
                    params.threshold_max = max;
                }
            }

            let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute_filter_params"),
                size: std::mem::size_of::<FilterParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

            // Output index buffer (worst-case: all triangles pass).
            let out_index_size = (gpu_mesh.index_count as u64) * 4;
            let out_index_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute_filter_out_indices"),
                size: out_index_size.max(4),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDEX
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // 4-byte atomic counter buffer (cleared to 0).
            let counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute_filter_counter"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            {
                let mut view = counter_buf.slice(..).get_mapped_range_mut();
                view[0..4].copy_from_slice(&0u32.to_le_bytes());
            }
            counter_buf.unmap();

            // Staging buffer to read back the counter.
            let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compute_filter_counter_staging"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Pick the scalar buffer: named attribute or dummy.
            let scalar_buf_ref: &wgpu::Buffer = match &item.kind {
                crate::renderer::ComputeFilterKind::Threshold { .. } => {
                    if let Some(attr_name) = &item.attribute_name {
                        gpu_mesh
                            .attribute_buffers
                            .get(attr_name.as_str())
                            .unwrap_or(&dummy_scalar_buf)
                    } else {
                        &dummy_scalar_buf
                    }
                }
                // Clip variants don't use the scalar buffer.
                _ => &dummy_scalar_buf,
            };

            // Build bind group.
            let bgl = self.compute_filter_bgl.as_ref().unwrap();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute_filter_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gpu_mesh.vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: gpu_mesh.index_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scalar_buf_ref.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_index_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: counter_buf.as_entire_binding(),
                    },
                ],
            });

            // Encode and submit compute + counter copy.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compute_filter_encoder"),
            });

            {
                let pipeline = self.compute_filter_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compute_filter_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                let workgroups = triangle_count.div_ceil(64);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            encoder.copy_buffer_to_buffer(&counter_buf, 0, &staging_buf, 0, 4);
            queue.submit(std::iter::once(encoder.finish()));

            // Synchronous readback (v1 : acceptable; async readback can follow later).
            let slice = staging_buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(5)),
            });

            let index_count = {
                let data = slice.get_mapped_range();
                u32::from_le_bytes([data[0], data[1], data[2], data[3]])
            };
            staging_buf.unmap();

            results.push(ComputeFilterResult {
                index_buffer: out_index_buf,
                index_count,
                mesh_id: item.mesh_id,
            });
        }

        results
    }
}
