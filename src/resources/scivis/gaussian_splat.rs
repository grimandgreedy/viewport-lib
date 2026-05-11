use super::*;
use crate::renderer::ShDegree;
use crate::resources::types::{GaussianSplatGpuSet, GaussianSplatViewportSort};

// Per-viewport SplatUniform layout (must match gaussian_splat.wgsl).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SplatUniform {
    model:      [[f32; 4]; 4],
    viewport_w: f32,
    viewport_h: f32,
    sh_degree:  u32,
    count:      u32,
}

// Depth compute uniform (must match gaussian_splat_sort.wgsl DepthUniform).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthUniform {
    model:  [[f32; 4]; 4],
    eye:    [f32; 3],
    count:  u32,
}

// Sort pass uniform (must match gaussian_splat_sort.wgsl SortUniform).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SortUniform {
    shift:    u32,
    count:    u32,
    pass_num: u32,
    _pad:     u32,
}

impl ViewportGpuResources {
    /// Lazily create all Gaussian splat render and compute pipelines.
    ///
    /// No-op after first call. Called from prepare when gaussian_splats is non-empty.
    pub(crate) fn ensure_gaussian_splat_pipelines(&mut self, device: &wgpu::Device) {
        if self.gaussian_splat_bgl.is_some() {
            return;
        }

        // ---------------------------------------------------------------
        // Render pipeline
        // ---------------------------------------------------------------

        // Group 1 BGL: SplatUniform (b0), sorted_indices (b1), positions (b2),
        //              scales (b3), rotations (b4), opacities (b5), sh_coefficients (b6).
        let splat_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_bgl"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gaussian_splat_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/gaussian_splat.wgsl").into(),
            ),
        });

        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gaussian_splat_pipeline_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &splat_bgl],
            push_constant_ranges: &[],
        });

        let make_splat_pipeline = |fmt: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gaussian_splat_pipeline"),
                layout: Some(&render_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
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
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                // No MSAA for Gaussian splats (alpha blending requires single-sample).
                multisample: wgpu::MultisampleState {
                    count: 1,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            })
        };
        let gaussian_splat_pipeline = DualPipeline {
            ldr: make_splat_pipeline(self.target_format),
            hdr: make_splat_pipeline(wgpu::TextureFormat::Rgba16Float),
        };

        // ---------------------------------------------------------------
        // Sort compute pipelines
        // ---------------------------------------------------------------

        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gaussian_splat_sort_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/gaussian_splat_sort.wgsl").into(),
            ),
        });

        // Depth compute BGL: DepthUniform (b0), positions (b1), keys_ping_out (b2).
        let depth_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_depth_bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let depth_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gaussian_splat_depth_layout"),
            bind_group_layouts: &[&depth_bgl],
            push_constant_ranges: &[],
        });

        let gaussian_splat_depth_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gaussian_splat_depth_pipeline"),
                layout: Some(&depth_layout),
                module: &sort_shader,
                entry_point: Some("compute_depths"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Sort BGL: SortUniform (b0), keys_ping (b1), keys_pong (b2),
        //           vals_ping (b3), vals_pong (b4), histogram (b5).
        let sort_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian_splat_sort_bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

        let sort_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gaussian_splat_sort_layout"),
            bind_group_layouts: &[&sort_bgl],
            push_constant_ranges: &[],
        });

        let gaussian_splat_sort_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gaussian_splat_sort_init_pipeline"),
                layout: Some(&sort_layout),
                module: &sort_shader,
                entry_point: Some("init_indices"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let gaussian_splat_sort_clear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gaussian_splat_sort_clear_pipeline"),
                layout: Some(&sort_layout),
                module: &sort_shader,
                entry_point: Some("clear_histogram"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let gaussian_splat_sort_histogram_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gaussian_splat_sort_histogram_pipeline"),
                layout: Some(&sort_layout),
                module: &sort_shader,
                entry_point: Some("histogram_pass"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let gaussian_splat_sort_prefix_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gaussian_splat_sort_prefix_pipeline"),
                layout: Some(&sort_layout),
                module: &sort_shader,
                entry_point: Some("prefix_sum_pass"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let gaussian_splat_sort_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gaussian_splat_sort_scatter_pipeline"),
                layout: Some(&sort_layout),
                module: &sort_shader,
                entry_point: Some("scatter_pass"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        self.gaussian_splat_bgl = Some(splat_bgl);
        self.gaussian_splat_pipeline = Some(gaussian_splat_pipeline);
        self.gaussian_splat_depth_bgl = Some(depth_bgl);
        self.gaussian_splat_depth_pipeline = Some(gaussian_splat_depth_pipeline);
        self.gaussian_splat_sort_bgl = Some(sort_bgl);
        self.gaussian_splat_sort_init_pipeline = Some(gaussian_splat_sort_init_pipeline);
        self.gaussian_splat_sort_clear_pipeline = Some(gaussian_splat_sort_clear_pipeline);
        self.gaussian_splat_sort_histogram_pipeline =
            Some(gaussian_splat_sort_histogram_pipeline);
        self.gaussian_splat_sort_prefix_pipeline = Some(gaussian_splat_sort_prefix_pipeline);
        self.gaussian_splat_sort_scatter_pipeline = Some(gaussian_splat_sort_scatter_pipeline);
    }

    /// Upload one Gaussian splat set to the GPU and return its handle.
    ///
    /// Call once per splat set at startup (or when the set changes). The returned
    /// [`GaussianSplatId`] is stable until [`remove_gaussian_splats`] is called.
    pub fn upload_gaussian_splats(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &crate::renderer::GaussianSplatData,
    ) -> crate::renderer::GaussianSplatId {
        let count = data.positions.len() as u32;

        // Pad positions/scales/rotations to vec4 (w=1 / w=0 / raw).
        let pos_data: Vec<[f32; 4]> = data
            .positions
            .iter()
            .map(|p| [p[0], p[1], p[2], 1.0])
            .collect();
        let scale_data: Vec<[f32; 4]> = data
            .scales
            .iter()
            .map(|s| [s[0], s[1], s[2], 0.0])
            .collect();
        let rotation_data: Vec<[f32; 4]> = data
            .rotations
            .iter()
            .map(|r| [r[0], r[1], r[2], r[3]])
            .collect();

        let buf_size_pos = (pos_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
        let buf_size_scale = (scale_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
        let buf_size_rot = (rotation_data.len() * std::mem::size_of::<[f32; 4]>()).max(16) as u64;
        let buf_size_opa = (data.opacities.len() * 4).max(4) as u64;
        let sh_bytes = data.sh_coefficients.len() * 4;
        let buf_size_sh = sh_bytes.max(4) as u64;

        let position_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_position_buf"),
            size: buf_size_pos,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&position_buf, 0, bytemuck::cast_slice(&pos_data));

        let scale_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_scale_buf"),
            size: buf_size_scale,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&scale_buf, 0, bytemuck::cast_slice(&scale_data));

        let rotation_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_rotation_buf"),
            size: buf_size_rot,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&rotation_buf, 0, bytemuck::cast_slice(&rotation_data));

        let opacity_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_opacity_buf"),
            size: buf_size_opa,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &opacity_buf,
            0,
            bytemuck::cast_slice(&data.opacities),
        );

        let sh_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_sh_buf"),
            size: buf_size_sh,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !data.sh_coefficients.is_empty() {
            queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&data.sh_coefficients));
        }

        let cpu_positions = data.positions.clone();
        let cpu_scales = data.scales.clone();

        let gpu_set = GaussianSplatGpuSet {
            position_buf,
            scale_buf,
            rotation_buf,
            opacity_buf,
            sh_buf,
            sh_degree: data.sh_degree,
            count,
            viewport_sort: Vec::new(),
            cpu_positions,
            cpu_scales,
        };

        let store_index = self.gaussian_splat_store.insert(gpu_set);
        crate::renderer::GaussianSplatId(store_index)
    }

    /// Remove an uploaded Gaussian splat set by handle.
    pub fn remove_gaussian_splats(
        &mut self,
        id: crate::renderer::GaussianSplatId,
    ) {
        self.gaussian_splat_store.remove(id.0);
    }

    /// Ensure per-viewport sort buffers exist for (store_index, viewport_index).
    ///
    /// Also creates the render bind group for the render pipeline (group 1).
    /// Called from run_gaussian_splat_sort before dispatching.
    pub(crate) fn ensure_gaussian_splat_sort_buffers(
        &mut self,
        device: &wgpu::Device,
        store_index: usize,
        viewport_index: usize,
    ) {
        let set = match self.gaussian_splat_store.get(store_index) {
            Some(s) => s,
            None => return,
        };
        let count = set.count as usize;

        // Grow the per-viewport vec if needed.
        if viewport_index >= set.viewport_sort.len()
            || set.viewport_sort[viewport_index].is_none()
        {
            // We need mutable access - re-borrow.
            let set_mut = self.gaussian_splat_store.get_mut(store_index).unwrap();
            while set_mut.viewport_sort.len() <= viewport_index {
                set_mut.viewport_sort.push(None);
            }

            if set_mut.viewport_sort[viewport_index].is_none() {
                let buf_size = (count * 4).max(4) as u64;

                let depth_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("splat_depth_buf"),
                    size: buf_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let sort_buf_usage = wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST;
                let keys_ping = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("splat_keys_ping"),
                    size: buf_size,
                    usage: sort_buf_usage,
                    mapped_at_creation: false,
                });
                let keys_pong = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("splat_keys_pong"),
                    size: buf_size,
                    usage: sort_buf_usage,
                    mapped_at_creation: false,
                });
                let vals_ping = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("splat_vals_ping"),
                    size: buf_size,
                    usage: sort_buf_usage,
                    mapped_at_creation: false,
                });
                let vals_pong = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("splat_vals_pong"),
                    size: buf_size,
                    usage: sort_buf_usage,
                    mapped_at_creation: false,
                });
                // Histogram: 256 x u32 atomic.
                let histogram_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("splat_histogram"),
                    size: 256 * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // Per-viewport SplatUniform buffer.
                let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("splat_uniform_buf"),
                    size: std::mem::size_of::<SplatUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // Build the render bind group (group 1). vals_ping holds sorted indices
                // after 4 sort passes (even number of passes means result ends in ping).
                let render_bg = {
                    let bgl = self.gaussian_splat_bgl.as_ref().unwrap();
                    let set_ref = self.gaussian_splat_store.get(store_index).unwrap();
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("splat_render_bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: uniform_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: vals_ping.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: set_ref.position_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: set_ref.scale_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: set_ref.rotation_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: set_ref.opacity_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: set_ref.sh_buf.as_entire_binding(),
                            },
                        ],
                    })
                };

                let vp_sort = GaussianSplatViewportSort {
                    depth_buf,
                    keys_ping,
                    keys_pong,
                    vals_ping,
                    vals_pong,
                    histogram_buf,
                    render_bg,
                    last_eye: [f32::NAN; 3],
                    uniform_buf,
                };

                let set_mut2 = self.gaussian_splat_store.get_mut(store_index).unwrap();
                set_mut2.viewport_sort[viewport_index] = Some(vp_sort);
            }
        }
    }

    /// Run the GPU depth compute + 4-pass radix sort for one splat set / viewport.
    ///
    /// Uploads updated SplatUniform (viewport dims, model, sh_degree), dispatches
    /// the depth compute shader, then runs init_indices + 4 sort passes.
    pub(crate) fn run_gaussian_splat_sort(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        store_index: usize,
        viewport_index: usize,
        eye: [f32; 3],
        model: [[f32; 4]; 4],
        vp_w: f32,
        vp_h: f32,
        sh_degree: ShDegree,
    ) {
        // Ensure sort buffers and render BG exist.
        self.ensure_gaussian_splat_sort_buffers(device, store_index, viewport_index);

        let set = match self.gaussian_splat_store.get(store_index) {
            Some(s) => s,
            None => return,
        };
        let count = set.count;
        if count == 0 {
            return;
        }

        let vp_sort = match set.viewport_sort.get(viewport_index) {
            Some(Some(s)) => s,
            _ => return,
        };

        // Update the SplatUniform for this viewport.
        let splat_uni = SplatUniform {
            model,
            viewport_w: vp_w,
            viewport_h: vp_h,
            sh_degree: match sh_degree {
                ShDegree::Zero => 0,
                ShDegree::One => 1,
                ShDegree::Three => 3,
            },
            count,
        };
        queue.write_buffer(&vp_sort.uniform_buf, 0, bytemuck::bytes_of(&splat_uni));

        // Upload depth uniform.
        let depth_uni = DepthUniform {
            model,
            eye,
            count,
        };
        let depth_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat_depth_uniform_tmp"),
            size: std::mem::size_of::<DepthUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&depth_uniform_buf, 0, bytemuck::bytes_of(&depth_uni));

        // Build depth BG.
        let depth_bg = {
            let bgl = self.gaussian_splat_depth_bgl.as_ref().unwrap();
            let set_ref = self.gaussian_splat_store.get(store_index).unwrap();
            let vp_sort_ref = set_ref.viewport_sort[viewport_index].as_ref().unwrap();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splat_depth_bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: depth_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: set_ref.position_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vp_sort_ref.depth_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let workgroups = (count + 255) / 256;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("splat_sort_encoder"),
            });

        // --- Depth compute pass ---
        {
            let depth_pipeline = self.gaussian_splat_depth_pipeline.as_ref().unwrap();
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("splat_depth_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(depth_pipeline);
            cpass.set_bind_group(0, &depth_bg, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy depth keys into keys_ping (depth_buf -> keys_ping).
        {
            let set_ref = self.gaussian_splat_store.get(store_index).unwrap();
            let vp_sort_ref = set_ref.viewport_sort[viewport_index].as_ref().unwrap();
            encoder.copy_buffer_to_buffer(
                &vp_sort_ref.depth_buf,
                0,
                &vp_sort_ref.keys_ping,
                0,
                (count as u64) * 4,
            );
        }

        // --- 4-pass radix sort ---
        // Build per-pass sort uniforms (shift = 0, 8, 16, 24; pass_num = 0..3).
        // We need sort_bgl and references to sort buffers for each pass.
        let sort_bgl = self.gaussian_splat_sort_bgl.as_ref().unwrap();

        for pass in 0u32..4u32 {
            let sort_uni = SortUniform {
                shift: pass * 8,
                count,
                pass_num: pass,
                _pad: 0,
            };
            let sort_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("splat_sort_uniform_tmp"),
                size: std::mem::size_of::<SortUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&sort_uniform_buf, 0, bytemuck::bytes_of(&sort_uni));

            let set_ref = self.gaussian_splat_store.get(store_index).unwrap();
            let vp_sort_ref = set_ref.viewport_sort[viewport_index].as_ref().unwrap();

            let sort_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("splat_sort_bg"),
                layout: sort_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sort_uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vp_sort_ref.keys_ping.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vp_sort_ref.keys_pong.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: vp_sort_ref.vals_ping.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: vp_sort_ref.vals_pong.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: vp_sort_ref.histogram_buf.as_entire_binding(),
                    },
                ],
            });

            // If pass 0: run init_indices first.
            if pass == 0 {
                let init_pipeline = self.gaussian_splat_sort_init_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_init_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(init_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Clear histogram.
            {
                let clear_pipeline =
                    self.gaussian_splat_sort_clear_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_clear_hist"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(clear_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Histogram pass.
            {
                let hist_pipeline =
                    self.gaussian_splat_sort_histogram_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_hist_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(hist_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }

            // Prefix sum.
            {
                let prefix_pipeline =
                    self.gaussian_splat_sort_prefix_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_prefix_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(prefix_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Scatter.
            {
                let scatter_pipeline =
                    self.gaussian_splat_sort_scatter_pipeline.as_ref().unwrap();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("splat_scatter_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(scatter_pipeline);
                cpass.set_bind_group(0, &sort_bg, &[]);
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Record eye so callers can skip re-sort if unchanged (optional optimisation).
        if let Some(set_mut) = self.gaussian_splat_store.get_mut(store_index) {
            if let Some(Some(vp_sort_mut)) = set_mut.viewport_sort.get_mut(viewport_index) {
                vp_sort_mut.last_eye = eye;
            }
        }
    }
}
