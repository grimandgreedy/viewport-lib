//! GPU state for the Gaussian splat item type: the render pipeline, the
//! depth + radix-sort compute passes and their per-viewport scratch, the
//! pick and outline-mask pipelines, and the per-frame outline buffers. The
//! render, pick and mask pipelines are built the first time a draw needs them.

use super::store::{GaussianSplatGpuSet, ShDegree};
use crate::item_types::shader::{scene_shader, wgsl_source};
use viewport_lib::gpu;
use viewport_lib::plugin_api::builders;
use viewport_lib::resources::DeviceResources;

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

/// Members of [`SplatPipelines`].
pub(super) const COLOUR_LDR: usize = 0;
pub(super) const COLOUR_HDR: usize = 1;
pub(super) const PICK: usize = 2;
pub(super) const MASK: usize = 3;

/// What a splat render pipeline build reads.
pub(super) struct SplatRecipe {
    device: gpu::Device,
    builder: viewport_lib::plugin_api::PipelineBuilder,
    render_layout: gpu::PipelineLayout,
    render_shader: viewport_lib::plugin_api::LazyModule,
    bgl: gpu::BindGroupLayout,
    pick_shader: viewport_lib::plugin_api::LazyModule,
    pick_id_bgl: gpu::BindGroupLayout,
    mask_layout: gpu::PipelineLayout,
    mask_shader: viewport_lib::plugin_api::LazyModule,
    ldr_format: gpu::TextureFormat,
}

/// The render pipeline in both formats, the pick pipeline and the outline
/// mask, each built the first time a draw needs it.
pub(super) type SplatPipelines = viewport_lib::plugin_api::LazyPipelines<SplatRecipe, 4>;

fn build(r: &SplatRecipe, i: usize) -> gpu::RenderPipeline {
    match i {
        // No MSAA for Gaussian splats (alpha blending requires single-sample).
        COLOUR_LDR | COLOUR_HDR => builders::build_dual_pipeline_variant(
            &r.device,
            &builders::DualPipelineDesc {
                label: "gaussian_splat_pipeline",
                layout: &r.render_layout,
                shader: r.render_shader.get(),
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[],
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                topology: gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: false,
                depth_compare: gpu::CompareFunction::Less,
                sample_count: 1,
                ldr_format: r.ldr_format,
            },
            i == COLOUR_HDR,
        ),
        // Pick: the same covariance-projected billboard expansion, object id
        // at group 2, splat index in the primitive channel. Laid out against
        // the shared group-0 camera, which the pick pass binds before plugin
        // dispatch.
        PICK => r.builder.build_pick_pipeline(
            &r.device,
            &viewport_lib::resources::PluginPipelineOpts {
                primitive: gpu::PrimitiveState {
                    topology: gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&r.bgl, &r.pick_id_bgl],
                ..viewport_lib::resources::PluginPipelineOpts::new(
                    Some("gaussian_splat_pick_pipeline"),
                    r.pick_shader.get(),
                    "vs_main",
                    "fs_main",
                    &[],
                )
            },
        ),
        // Outline mask: point-sprite discs, instance-stepped position and
        // pixel-size vertex buffers, over the shared outline bind group
        // layout (the same shape the point-cloud outline pipeline uses).
        _ => {
            let mask_pos_attrs = [gpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: gpu::VertexFormat::Float32x3,
            }];
            let mask_size_attrs = [gpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: gpu::VertexFormat::Float32,
            }];
            builders::render_pipeline(
                &r.device,
                builders::RenderPipelineDesc {
                    label: "point_disc_mask_pipeline",
                    layout: &r.mask_layout,
                    vertex_module: r.mask_shader.get(),
                    vertex_entry: "vs_main",
                    vertex_buffers: &[
                        gpu::VertexBufferLayout {
                            array_stride: 12, // vec3<f32>
                            step_mode: gpu::VertexStepMode::Instance,
                            attributes: &mask_pos_attrs,
                        },
                        gpu::VertexBufferLayout {
                            array_stride: 4, // f32
                            step_mode: gpu::VertexStepMode::Instance,
                            attributes: &mask_size_attrs,
                        },
                    ],
                    fragment: Some(gpu::FragmentState {
                        module: r.mask_shader.get(),
                        entry_point: Some("fs_main"),
                        targets: &[Some(gpu::ColorTargetState {
                            format: gpu::TextureFormat::R8Unorm,
                            blend: None,
                            write_mask: gpu::ColorWrites::ALL,
                        })],
                        compilation_options: gpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: gpu::PrimitiveState {
                        topology: gpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(builders::scene_depth_stencil(
                        false,
                        gpu::CompareFunction::Less,
                    )),
                    multisample: gpu::MultisampleState {
                        count: 1,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        }
    }
}

/// What the depth and sort compute builds read: both come from one module.
pub(super) struct SplatComputeRecipe {
    device: gpu::Device,
    depth_layout: gpu::PipelineLayout,
    sort_layout: gpu::PipelineLayout,
    shader: viewport_lib::plugin_api::LazyModule,
}

const DEPTH: usize = 0;
const SORT_INIT: usize = 1;
const SORT_CLEAR: usize = 2;
const SORT_HISTOGRAM: usize = 3;
const SORT_PREFIX: usize = 4;
const SORT_SCATTER: usize = 5;
const COMPUTE_COUNT: usize = 6;

fn build_compute(r: &SplatComputeRecipe, i: usize) -> gpu::ComputePipeline {
    let (label, entry) = match i {
        DEPTH => ("gaussian_splat_depth_pipeline", "compute_depths"),
        SORT_INIT => ("gaussian_splat_sort_init_pipeline", "init_indices"),
        SORT_CLEAR => ("gaussian_splat_sort_clear_pipeline", "clear_histogram"),
        SORT_HISTOGRAM => ("gaussian_splat_sort_histogram_pipeline", "histogram_pass"),
        SORT_PREFIX => ("gaussian_splat_sort_prefix_pipeline", "prefix_sum_pass"),
        _ => ("gaussian_splat_sort_scatter_pipeline", "scatter_pass"),
    };
    let layout = if i == DEPTH {
        &r.depth_layout
    } else {
        &r.sort_layout
    };
    builders::compute_pipeline(&r.device, label, layout, r.shader.get(), entry)
}

/// Pipelines and layouts, made on the first prepare with items. Every
/// pipeline is built the first time a frame needs it.
pub(super) struct SplatGpu {
    pub(super) bgl: gpu::BindGroupLayout,
    pub(super) pipelines: SplatPipelines,
    /// The depth pass and the radix sort. Until all are built a frame sorts
    /// and draws no splats, since the draw reads the sorted indices.
    compute: viewport_lib::plugin_api::LazyPipelines<
        SplatComputeRecipe,
        COMPUTE_COUNT,
        gpu::ComputePipeline,
    >,
    depth_bgl: gpu::BindGroupLayout,
    sort_bgl: gpu::BindGroupLayout,
    pub(super) pick_id_bgl: gpu::BindGroupLayout,
    /// Group 1 of the outline mask pipeline: the single uniform
    /// `point_disc_mask.wgsl` reads.
    pub(super) mask_bgl: gpu::BindGroupLayout,
}

/// Per-(set, viewport) sort scratch and the render bind group built over it.
pub(super) struct SortState {
    depth_buf: gpu::Buffer,
    keys_ping: gpu::Buffer,
    keys_pong: gpu::Buffer,
    vals_ping: gpu::Buffer,
    vals_pong: gpu::Buffer,
    histogram_buf: gpu::Buffer,
    uniform_buf: gpu::Buffer,
    /// Render bind group (group 1): SplatUniform, sorted indices
    /// (`vals_ping` holds the result after the even number of sort passes),
    /// and the set's five data buffers.
    pub(super) render_bg: gpu::BindGroup,
}

/// One selected set's outline coverage: instance-stepped disc positions and
/// pixel sizes for the point-sprite mask pipeline.
pub(super) struct SplatOutlineEntry {
    pub(super) position_buf: gpu::Buffer,
    pub(super) size_buf: gpu::Buffer,
    pub(super) instance_count: u32,
    pub(super) _uniform_buf: gpu::Buffer,
    pub(super) bind_group: gpu::BindGroup,
}

impl SplatGpu {
    pub(super) fn new(device: &gpu::Device, resources: &DeviceResources) -> Self {
        // Group 1 BGL: SplatUniform, sorted_indices, positions,
        //              scales, rotations, opacities, sh_coefficients.
        let storage_entry = |binding: u32| gpu::BindGroupLayoutEntry {
            binding,
            visibility: gpu::ShaderStages::VERTEX,
            ty: gpu::BindingType::Buffer {
                ty: gpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_bgl"),
            entries: &[
                gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Uniform,
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

        let render_shader = resources.lazy_module(
            device,
            "gaussian_splat_shader",
            &scene_shader(&[], wgsl_source!("gaussian_splat")),
        );
        let render_layout = builders::standard_scene_layout(
            device,
            "gaussian_splat_pipeline_layout",
            resources.shared_bindings().group0_layout,
            &bgl,
        );
        // Sort compute pipelines.
        let sort_shader = resources.lazy_module(
            device,
            "gaussian_splat_sort_shader",
            wgsl_source!("gaussian_splat_sort"),
        );
        // Depth compute BGL: DepthUniform (b0), positions (b1), keys out (b2).
        let depth_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_depth_bgl"),
            entries: &[
                gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: gpu::ShaderStages::COMPUTE,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: gpu::ShaderStages::COMPUTE,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                gpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: gpu::ShaderStages::COMPUTE,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let depth_layout =
            builders::pipeline_layout(device, "gaussian_splat_depth_layout", &[&depth_bgl]);

        // Sort BGL: SortUniform (b0), keys ping/pong (b1/b2), vals ping/pong
        // (b3/b4), histogram (b5).
        let rw_entry = |binding: u32| gpu::BindGroupLayoutEntry {
            binding,
            visibility: gpu::ShaderStages::COMPUTE,
            ty: gpu::BindingType::Buffer {
                ty: gpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sort_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_sort_bgl"),
            entries: &[
                gpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: gpu::ShaderStages::COMPUTE,
                    ty: gpu::BindingType::Buffer {
                        ty: gpu::BufferBindingType::Uniform,
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
        let sort_layout =
            builders::pipeline_layout(device, "gaussian_splat_sort_layout", &[&sort_bgl]);
        let compute = resources.lazy_compute_pipelines(
            SplatComputeRecipe {
                device: device.clone(),
                depth_layout,
                sort_layout,
                shader: sort_shader,
            },
            build_compute,
        );

        // Group 2 of the pick pipeline: the set's object id.
        let pick_id_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_pick_id_bgl"),
            entries: &[gpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: gpu::ShaderStages::FRAGMENT,
                ty: gpu::BindingType::Buffer {
                    ty: gpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pick_shader = resources.lazy_module(
            device,
            "gaussian_splat_pick_shader",
            &scene_shader(&[], wgsl_source!("gaussian_splat_pick")),
        );
        // Outline mask: a group-1 layout holding the single uniform
        // `point_disc_mask.wgsl` reads.
        let mask_shader = resources.lazy_module(
            device,
            "point_disc_mask_shader",
            &scene_shader(&[], wgsl_source!("point_disc_mask")),
        );
        let mask_bgl = device.create_bind_group_layout(&gpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_mask_bgl"),
            entries: &[builders::uniform_entry(
                0,
                gpu::ShaderStages::VERTEX | gpu::ShaderStages::FRAGMENT,
            )],
        });
        let mask_layout = builders::pipeline_layout(
            device,
            "gaussian_splat_mask_layout",
            &[resources.shared_bindings().group0_layout, &mask_bgl],
        );
        let pipelines = resources.lazy_pipelines(
            SplatRecipe {
                device: device.clone(),
                builder: resources.pipeline_builder(),
                render_layout,
                render_shader,
                bgl: bgl.clone(),
                pick_shader,
                pick_id_bgl: pick_id_bgl.clone(),
                mask_layout,
                mask_shader,
                ldr_format: resources.target_format(),
            },
            build,
        );

        Self {
            bgl,
            pipelines,
            compute,
            depth_bgl,
            sort_bgl,
            pick_id_bgl,
            mask_bgl,
        }
    }

    /// Whether the render pipeline can draw this frame in either format. The
    /// mask and pick passes wait for it, so a set is never outlined or picked
    /// before it is drawn.
    pub(super) fn drawn(&self) -> bool {
        self.pipelines.available(COLOUR_LDR) || self.pipelines.available(COLOUR_HDR)
    }

    /// Build the sort scratch and render bind group for one (set, viewport).
    pub(super) fn make_sort_state(
        &self,
        device: &gpu::Device,
        set: &GaussianSplatGpuSet,
    ) -> SortState {
        let buf_size = (set.count() as usize * 4).max(4) as u64;
        let depth_buf = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("splat_depth_buf"),
            size: buf_size,
            usage: gpu::BufferUsages::STORAGE
                | gpu::BufferUsages::COPY_DST
                | gpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sort_buf_usage =
            gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_SRC | gpu::BufferUsages::COPY_DST;
        let make_sort_buf = |label: &str| {
            device.create_buffer(&gpu::BufferDescriptor {
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
        let histogram_buf = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("splat_histogram"),
            size: 256 * 4,
            usage: gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("splat_uniform_buf"),
            size: std::mem::size_of::<SplatUniform>() as u64,
            usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // vals_ping holds sorted indices after 4 sort passes (an even number
        // of passes means the result ends in ping).
        let render_bg = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("splat_render_bg"),
            layout: &self.bgl,
            entries: &[
                gpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 1,
                    resource: vals_ping.as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 2,
                    resource: set.positions.buffer().as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 3,
                    resource: set.scales.buffer().as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 4,
                    resource: set.rotations.buffer().as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 5,
                    resource: set.opacities.buffer().as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 6,
                    resource: set.sh.buffer().as_entire_binding(),
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
    /// Sort pipeline `i`, which the caller has checked with
    /// [`sort_ready`](Self::sort_ready).
    fn built(&self, i: usize) -> &gpu::ComputePipeline {
        self.compute
            .get(i)
            .expect("encode_sort runs only once sort_ready")
    }

    /// Whether every sort pipeline is built, asking for any that is not.
    /// [`encode_sort`](Self::encode_sort) needs all of them.
    pub(super) fn sort_ready(&self) -> bool {
        (0..COMPUTE_COUNT).fold(true, |ready, i| self.compute.get(i).is_some() && ready)
    }

    /// Ask for every sort pipeline, for a warm-up.
    pub(super) fn request_sort(&self) {
        self.compute.request_all();
    }

    pub(super) fn encode_sort(
        &self,
        device: &gpu::Device,
        queue: &gpu::Queue,
        encoder: &mut gpu::CommandEncoder,
        set: &GaussianSplatGpuSet,
        sort: &SortState,
        eye: [f32; 3],
        model: [[f32; 4]; 4],
        vp_w: f32,
        vp_h: f32,
    ) {
        let count = set.count();
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
        let depth_uniform_buf = device.create_buffer(&gpu::BufferDescriptor {
            label: Some("splat_depth_uniform_tmp"),
            size: std::mem::size_of::<DepthUniform>() as u64,
            usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&depth_uniform_buf, 0, bytemuck::bytes_of(&depth_uni));

        let depth_bg = device.create_bind_group(&gpu::BindGroupDescriptor {
            label: Some("splat_depth_bg"),
            layout: &self.depth_bgl,
            entries: &[
                gpu::BindGroupEntry {
                    binding: 0,
                    resource: depth_uniform_buf.as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 1,
                    resource: set.positions.buffer().as_entire_binding(),
                },
                gpu::BindGroupEntry {
                    binding: 2,
                    resource: sort.depth_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = count.div_ceil(256);

        {
            let mut cpass = encoder.begin_compute_pass(&gpu::ComputePassDescriptor {
                label: Some("splat_depth_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.built(DEPTH));
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
            let sort_uniform_buf = device.create_buffer(&gpu::BufferDescriptor {
                label: Some("splat_sort_uniform_tmp"),
                size: std::mem::size_of::<SortUniform>() as u64,
                usage: gpu::BufferUsages::UNIFORM | gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&sort_uniform_buf, 0, bytemuck::bytes_of(&sort_uni));

            let sort_bg = device.create_bind_group(&gpu::BindGroupDescriptor {
                label: Some("splat_sort_bg"),
                layout: &self.sort_bgl,
                entries: &[
                    gpu::BindGroupEntry {
                        binding: 0,
                        resource: sort_uniform_buf.as_entire_binding(),
                    },
                    gpu::BindGroupEntry {
                        binding: 1,
                        resource: sort.keys_ping.as_entire_binding(),
                    },
                    gpu::BindGroupEntry {
                        binding: 2,
                        resource: sort.keys_pong.as_entire_binding(),
                    },
                    gpu::BindGroupEntry {
                        binding: 3,
                        resource: sort.vals_ping.as_entire_binding(),
                    },
                    gpu::BindGroupEntry {
                        binding: 4,
                        resource: sort.vals_pong.as_entire_binding(),
                    },
                    gpu::BindGroupEntry {
                        binding: 5,
                        resource: sort.histogram_buf.as_entire_binding(),
                    },
                ],
            });

            let mut dispatch = |label: &str, pipeline: &gpu::ComputePipeline, wg: u32| {
                let mut cpass = encoder.begin_compute_pass(&gpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(wg, 1, 1);
            };
            if pass == 0 {
                dispatch("splat_init_pass", self.built(SORT_INIT), workgroups);
            }
            dispatch("splat_clear_hist", self.built(SORT_CLEAR), 1);
            dispatch("splat_hist_pass", self.built(SORT_HISTOGRAM), workgroups);
            dispatch("splat_prefix_pass", self.built(SORT_PREFIX), 1);
            dispatch("splat_scatter_pass", self.built(SORT_SCATTER), workgroups);
        }
    }
}
