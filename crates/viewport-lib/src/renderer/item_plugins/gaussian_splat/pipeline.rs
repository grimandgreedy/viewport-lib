//! GPU state for the Gaussian splat item type: the render pipeline, the
//! depth + radix-sort compute passes and their per-viewport scratch, the
//! pick and outline-mask pipelines, and the per-frame outline buffers.

use super::store::{GaussianSplatGpuSet, ShDegree};
use crate::resources::DeviceResources;

// Per-viewport SplatUniform layout (must match gaussian_splat.wgsl).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SplatUniform {
    pub model: [[f32; 4]; 4],
    pub viewport_w: f32,
    pub viewport_h: f32,
    pub sh_degree: u32,
    pub count: u32,
}

// Depth compute uniform (must match gaussian_splat_sort.wgsl DepthUniform).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthUniform {
    model: [[f32; 4]; 4],
    eye: [f32; 3],
    count: u32,
}

// Sort pass uniform (must match gaussian_splat_sort.wgsl SortUniform).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SortUniform {
    shift: u32,
    count: u32,
    pass_num: u32,
    _pad: u32,
}

/// Pipelines and layouts, built lazily on the first prepare with items.
pub(super) struct SplatGpu {
    pub(super) bgl: crate::gpu::BindGroupLayout,
    pub(super) pipeline: crate::resources::DualPipeline,
    depth_pipeline: crate::gpu::ComputePipeline,
    sort_init_pipeline: crate::gpu::ComputePipeline,
    sort_clear_pipeline: crate::gpu::ComputePipeline,
    sort_histogram_pipeline: crate::gpu::ComputePipeline,
    sort_prefix_pipeline: crate::gpu::ComputePipeline,
    sort_scatter_pipeline: crate::gpu::ComputePipeline,
    depth_bgl: crate::gpu::BindGroupLayout,
    sort_bgl: crate::gpu::BindGroupLayout,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_id_bgl: crate::gpu::BindGroupLayout,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
    /// Group 1 of the outline mask pipeline: the single uniform
    /// `splat_outline_mask.wgsl` reads.
    pub(super) mask_bgl: crate::gpu::BindGroupLayout,
}

/// Per-(set, viewport) sort scratch and the render bind group built over it.
pub(super) struct SortState {
    depth_buf: crate::gpu::Buffer,
    keys_ping: crate::gpu::Buffer,
    keys_pong: crate::gpu::Buffer,
    vals_ping: crate::gpu::Buffer,
    vals_pong: crate::gpu::Buffer,
    histogram_buf: crate::gpu::Buffer,
    uniform_buf: crate::gpu::Buffer,
    /// Render bind group (group 1): SplatUniform, sorted indices
    /// (`vals_ping` holds the result after the even number of sort passes),
    /// and the set's five data buffers.
    pub(super) render_bg: crate::gpu::BindGroup,
}

/// One selected set's outline coverage: instance-stepped disc positions and
/// pixel sizes for the point-sprite mask pipeline.
pub(super) struct SplatOutlineEntry {
    pub(super) position_buf: crate::gpu::Buffer,
    pub(super) size_buf: crate::gpu::Buffer,
    pub(super) instance_count: u32,
    pub(super) _uniform_buf: crate::gpu::Buffer,
    pub(super) bind_group: crate::gpu::BindGroup,
}

impl SplatGpu {
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        // Group 1 BGL: SplatUniform, sorted_indices, positions,
        //              scales, rotations, opacities, sh_coefficients.
        let storage_entry = |binding: u32| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: crate::gpu::ShaderStages::VERTEX,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::VERTEX
                        | crate::gpu::ShaderStages::FRAGMENT,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
                storage_entry(5),
                storage_entry(6),
            ],
        });

        let render_shader = crate::resources::builders::wgsl_module(
            device,
            "gaussian_splat_shader",
            crate::resources::builders::wgsl_source!("gaussian_splat"),
        );
        let render_layout = crate::resources::builders::standard_scene_layout(
            device,
            "gaussian_splat_pipeline_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        // No MSAA for Gaussian splats (alpha blending requires single-sample).
        let pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "gaussian_splat_pipeline",
                layout: &render_layout,
                shader: &render_shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[],
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: crate::gpu::CompareFunction::Less,
                sample_count: 1,
                ldr_format: resources.target_format,
            },
        );

        // Sort compute pipelines.
        let sort_shader = crate::resources::builders::wgsl_module(
            device,
            "gaussian_splat_sort_shader",
            crate::resources::builders::wgsl_source!("gaussian_splat_sort"),
        );
        // Depth compute BGL: DepthUniform (b0), positions (b1), keys out (b2).
        let depth_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_depth_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                crate::gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let depth_layout = crate::resources::builders::pipeline_layout(
            device,
            "gaussian_splat_depth_layout",
            &[&depth_bgl],
        );
        let depth_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "gaussian_splat_depth_pipeline",
            &depth_layout,
            &sort_shader,
            "compute_depths",
        );

        // Sort BGL: SortUniform (b0), keys ping/pong (b1/b2), vals ping/pong
        // (b3/b4), histogram (b5).
        let rw_entry = |binding: u32| crate::gpu::BindGroupLayoutEntry {
            binding,
            visibility: crate::gpu::ShaderStages::COMPUTE,
            ty: crate::gpu::BindingType::Buffer {
                ty: crate::gpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sort_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_sort_bgl"),
            entries: &[
                crate::gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: crate::gpu::ShaderStages::COMPUTE,
                    ty: crate::gpu::BindingType::Buffer {
                        ty: crate::gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                rw_entry(1),
                rw_entry(2),
                rw_entry(3),
                rw_entry(4),
                rw_entry(5),
            ],
        });
        let sort_layout = crate::resources::builders::pipeline_layout(
            device,
            "gaussian_splat_sort_layout",
            &[&sort_bgl],
        );
        let compute = |label: &str, entry: &str| {
            crate::resources::builders::compute_pipeline(
                device,
                label,
                &sort_layout,
                &sort_shader,
                entry,
            )
        };
        let sort_init_pipeline = compute("gaussian_splat_sort_init_pipeline", "init_indices");
        let sort_clear_pipeline = compute("gaussian_splat_sort_clear_pipeline", "clear_histogram");
        let sort_histogram_pipeline =
            compute("gaussian_splat_sort_histogram_pipeline", "histogram_pass");
        let sort_prefix_pipeline =
            compute("gaussian_splat_sort_prefix_pipeline", "prefix_sum_pass");
        let sort_scatter_pipeline = compute("gaussian_splat_sort_scatter_pipeline", "scatter_pass");

        // Pick: the same covariance-projected billboard expansion, object id
        // at group 2, splat index in the primitive channel. Laid out against
        // the shared group-0 camera, which the pick pass binds before plugin
        // dispatch.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_pick_id_bgl"),
            entries: &[crate::gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: crate::gpu::ShaderStages::FRAGMENT,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pick_shader = crate::resources::builders::wgsl_module(
            device,
            "gaussian_splat_pick_shader",
            crate::resources::builders::wgsl_source!("gaussian_splat_pick"),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&bgl, &pick_id_bgl],
                ..crate::resources::PluginPipelineOpts::new(
                    Some("gaussian_splat_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[],
                )
            },
        );

        // Outline mask: point-sprite discs, instance-stepped position and
        // pixel-size vertex buffers, over the shared outline bind group
        // layout (the same shape the point-cloud outline pipeline uses).
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "splat_outline_mask_shader",
            crate::resources::builders::wgsl_source!("splat_outline_mask"),
        );
        let mask_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_mask_bgl"),
            entries: &[crate::resources::builders::uniform_entry(
                0,
                crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            )],
        });
        let mask_layout = crate::resources::builders::pipeline_layout(
            device,
            "gaussian_splat_mask_layout",
            &[resources.shared_bindings().group0_layout, &mask_bgl],
        );
        let mask_pos_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: crate::gpu::VertexFormat::Float32x3,
        }];
        let mask_size_attrs = [crate::gpu::VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: crate::gpu::VertexFormat::Float32,
        }];
        let mask_pipeline = crate::resources::builders::render_pipeline(
            device,
            crate::resources::builders::RenderPipelineDesc {
                label: "splat_outline_mask_pipeline",
                layout: &mask_layout,
                vertex_module: &mask_shader,
                vertex_entry: "vs_main",
                vertex_buffers: &[
                    crate::gpu::VertexBufferLayout {
                        array_stride: 12, // vec3<f32>
                        step_mode: crate::gpu::VertexStepMode::Instance,
                        attributes: &mask_pos_attrs,
                    },
                    crate::gpu::VertexBufferLayout {
                        array_stride: 4, // f32
                        step_mode: crate::gpu::VertexStepMode::Instance,
                        attributes: &mask_size_attrs,
                    },
                ],
                fragment: Some(crate::gpu::FragmentState {
                    module: &mask_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(crate::gpu::ColorTargetState {
                        format: crate::gpu::TextureFormat::R8Unorm,
                        blend: None,
                        write_mask: crate::gpu::ColorWrites::ALL,
                    })],
                    compilation_options: crate::gpu::PipelineCompilationOptions::default(),
                }),
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::resources::builders::scene_depth_stencil(
                    false,
                    crate::gpu::CompareFunction::Less,
                )),
                multisample: crate::gpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                cache: None,
            },
        );

        Self {
            bgl,
            pipeline,
            depth_pipeline,
            sort_init_pipeline,
            sort_clear_pipeline,
            sort_histogram_pipeline,
            sort_prefix_pipeline,
            sort_scatter_pipeline,
            depth_bgl,
            sort_bgl,
            pick_pipeline,
            pick_id_bgl,
            mask_pipeline,
            mask_bgl,
        }
    }

    /// Build the sort scratch and render bind group for one (set, viewport).
    pub(super) fn make_sort_state(
        &self,
        device: &crate::gpu::Device,
        set: &GaussianSplatGpuSet,
    ) -> SortState {
        let buf_size = (set.count as usize * 4).max(4) as u64;
        let depth_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("splat_depth_buf"),
            size: buf_size,
            usage: crate::gpu::BufferUsages::STORAGE
                | crate::gpu::BufferUsages::COPY_DST
                | crate::gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sort_buf_usage = crate::gpu::BufferUsages::STORAGE
            | crate::gpu::BufferUsages::COPY_SRC
            | crate::gpu::BufferUsages::COPY_DST;
        let make_sort_buf = |label: &str| {
            device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some(label),
                size: buf_size,
                usage: sort_buf_usage,
                mapped_at_creation: false,
            })
        };
        let keys_ping = make_sort_buf("splat_keys_ping");
        let keys_pong = make_sort_buf("splat_keys_pong");
        let vals_ping = make_sort_buf("splat_vals_ping");
        let vals_pong = make_sort_buf("splat_vals_pong");
        // Histogram: 256 x u32 atomic.
        let histogram_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("splat_histogram"),
            size: 256 * 4,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("splat_uniform_buf"),
            size: std::mem::size_of::<SplatUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // vals_ping holds sorted indices after 4 sort passes (an even number
        // of passes means the result ends in ping).
        let render_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("splat_render_bg"),
            layout: &self.bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: vals_ping.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: set.position_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 3,
                    resource: set.scale_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 4,
                    resource: set.rotation_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 5,
                    resource: set.opacity_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 6,
                    resource: set.sh_buf.as_entire_binding(),
                },
            ],
        });
        SortState {
            depth_buf,
            keys_ping,
            keys_pong,
            vals_ping,
            vals_pong,
            histogram_buf,
            uniform_buf,
            render_bg,
        }
    }

    /// Encode the depth compute + 4-pass radix sort for one set / viewport,
    /// and write the viewport's `SplatUniform`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_sort(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        encoder: &mut crate::gpu::CommandEncoder,
        set: &GaussianSplatGpuSet,
        sort: &SortState,
        eye: [f32; 3],
        model: [[f32; 4]; 4],
        vp_w: f32,
        vp_h: f32,
    ) {
        let count = set.count;
        if count == 0 {
            return;
        }

        let splat_uni = SplatUniform {
            model,
            viewport_w: vp_w,
            viewport_h: vp_h,
            sh_degree: match set.sh_degree {
                ShDegree::Zero => 0,
                ShDegree::One => 1,
                ShDegree::Three => 3,
            },
            count,
        };
        queue.write_buffer(&sort.uniform_buf, 0, bytemuck::bytes_of(&splat_uni));

        let depth_uni = DepthUniform { model, eye, count };
        let depth_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("splat_depth_uniform_tmp"),
            size: std::mem::size_of::<DepthUniform>() as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&depth_uniform_buf, 0, bytemuck::bytes_of(&depth_uni));

        let depth_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("splat_depth_bg"),
            layout: &self.depth_bgl,
            entries: &[
                crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: depth_uniform_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 1,
                    resource: set.position_buf.as_entire_binding(),
                },
                crate::gpu::BindGroupEntry {
                    binding: 2,
                    resource: sort.depth_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = count.div_ceil(256);

        {
            let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                label: Some("splat_depth_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.depth_pipeline);
            cpass.set_bind_group(0, &depth_bg, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy depth keys into keys_ping.
        encoder.copy_buffer_to_buffer(&sort.depth_buf, 0, &sort.keys_ping, 0, (count as u64) * 4);

        // 4-pass radix sort (shift = 0, 8, 16, 24).
        for pass in 0u32..4u32 {
            let sort_uni = SortUniform {
                shift: pass * 8,
                count,
                pass_num: pass,
                _pad: 0,
            };
            let sort_uniform_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("splat_sort_uniform_tmp"),
                size: std::mem::size_of::<SortUniform>() as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&sort_uniform_buf, 0, bytemuck::bytes_of(&sort_uni));

            let sort_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("splat_sort_bg"),
                layout: &self.sort_bgl,
                entries: &[
                    crate::gpu::BindGroupEntry {
                        binding: 0,
                        resource: sort_uniform_buf.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 1,
                        resource: sort.keys_ping.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 2,
                        resource: sort.keys_pong.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 3,
                        resource: sort.vals_ping.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 4,
                        resource: sort.vals_pong.as_entire_binding(),
                    },
                    crate::gpu::BindGroupEntry {
                        binding: 5,
                        resource: sort.histogram_buf.as_entire_binding(),
                    },
                ],
            });

            let mut dispatch = |label: &str, pipeline: &crate::gpu::ComputePipeline, wg: u32| {
                let mut cpass = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(wg, 1, 1);
            };
            if pass == 0 {
                dispatch("splat_init_pass", &self.sort_init_pipeline, workgroups);
            }
            dispatch("splat_clear_hist", &self.sort_clear_pipeline, 1);
            dispatch("splat_hist_pass", &self.sort_histogram_pipeline, workgroups);
            dispatch("splat_prefix_pass", &self.sort_prefix_pipeline, 1);
            dispatch(
                "splat_scatter_pass",
                &self.sort_scatter_pipeline,
                workgroups,
            );
        }
    }
}
