//! GPU state for the GPU marching cubes item type: the three compute
//! pipelines that extract the isosurface, the surface and wireframe render
//! pipelines, the shadow, outline-mask and pick pipelines, and the shared case
//! tables.
//!
//! The uploaded volumes live in `resources`, because `upload_volume_for_mc` and
//! `McVolumeId` are consumer API and the store is wired into the resident-byte
//! accounting and the async upload runner. Nothing outside this type reads them. This module
//! reads the store each prepare and clones the per-slab buffer handles it needs
//! into [`McFrame`], so the draw hooks never need a borrow of it.

use crate::geometry::marching_cubes::TRI_TABLE;
use crate::gpu::util::DeviceExt as _;
use crate::renderer::{GpuMarchingCubesItem, PickId};
use crate::resources::DeviceResources;

/// The generated geometry of one volume slab, as the draw hooks see it.
pub(super) struct McSlabDraw {
    pub(super) vertex_buf: crate::gpu::Buffer,
    /// Indirect args for the solid surface draw.
    pub(super) indirect_buf: crate::gpu::Buffer,
    /// Indirect args for the line-list wireframe draw.
    pub(super) wire_indirect_buf: crate::gpu::Buffer,
}

/// Per-item draw data rebuilt each prepare.
pub(super) struct McFrame {
    pub(super) slabs: Vec<McSlabDraw>,
    pub(super) render_bg: crate::gpu::BindGroup,
    /// True when the item was submitted with `settings.wireframe`.
    pub(super) wireframe: bool,
    /// Per-slab bind groups for the wireframe pipeline (binding 0 = vertex storage buffer).
    pub(super) wire_slab_bgs: Vec<crate::gpu::BindGroup>,
    pub(super) pick_id: PickId,
    /// Shadows reflect the actual surface, not its display mode, so a
    /// wireframe item still casts through the solid slab data.
    pub(super) cast_shadows: bool,
    pub(super) selected: bool,
    pub(super) hidden: bool,
}

/// Pipelines, layouts, and case tables, built lazily on the first prepare with
/// items.
pub(super) struct McGpu {
    classify_pipeline: crate::gpu::ComputePipeline,
    prefix_sum_pipeline: crate::gpu::ComputePipeline,
    generate_pipeline: crate::gpu::ComputePipeline,
    pub(super) surface_pipeline: crate::resources::DualPipeline,
    pub(super) wireframe_pipeline: crate::resources::DualPipeline,
    /// Depth-only shadow-cast pipeline. MC vertices are already world-space
    /// (no per-item model matrix anywhere in the MC path), so there is no
    /// group-1 bind group at all: just the shadow pass's camera layout at
    /// group 0.
    pub(super) shadow_pipeline: crate::gpu::RenderPipeline,
    pub(super) mask_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_pipeline: crate::gpu::RenderPipeline,
    pub(super) pick_id_bgl: crate::gpu::BindGroupLayout,
    wireframe_render_bgl: crate::gpu::BindGroupLayout,
    classify_bgl: crate::gpu::BindGroupLayout,
    prefix_sum_bgl: crate::gpu::BindGroupLayout,
    generate_bgl: crate::gpu::BindGroupLayout,
    render_bgl: crate::gpu::BindGroupLayout,
    case_count_buf: crate::gpu::Buffer,
    case_table_buf: crate::gpu::Buffer,
}

impl McGpu {
    pub(super) fn new(device: &crate::gpu::Device, resources: &DeviceResources) -> Self {
        // ----------------------------------------------------------------
        // Shared lookup buffers (uploaded once).
        // ----------------------------------------------------------------
        let count_table = case_triangle_count_table();
        let mc_case_count_buf =
            device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("mc_case_count_buf"),
                contents: bytemuck::cast_slice(&count_table),
                usage: crate::gpu::BufferUsages::STORAGE,
            });

        let flat_table = case_table_flat();
        let mc_case_table_buf =
            device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("mc_case_table_buf"),
                contents: bytemuck::cast_slice(&flat_table),
                usage: crate::gpu::BufferUsages::STORAGE,
            });

        // ----------------------------------------------------------------
        // Bind group layouts.
        // ----------------------------------------------------------------

        // Classify: 5 bindings (uniform + 2 read storage + 2 rw storage).
        let classify_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("mc_classify_bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_ro(2),
                    bgl_storage_rw(3),
                    bgl_storage_rw(4),
                ],
            });

        // Prefix sum: 6 bindings (uniform + ro + 3 rw + wire_indirect_buf rw).
        let prefix_sum_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("mc_prefix_sum_bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_rw(2),
                    bgl_storage_rw(3),
                    bgl_storage_rw(4),
                    bgl_storage_rw(5), // wire_indirect_buf
                ],
            });

        // Generate: 6 bindings (uniform + 3 ro + 2 rw [case_indices ro, vertex_buf rw]).
        let generate_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("mc_generate_bgl"),
                entries: &[
                    bgl_uniform(0),
                    bgl_storage_ro(1),
                    bgl_storage_ro(2),
                    bgl_storage_ro(3),
                    bgl_storage_ro(4),
                    bgl_storage_rw(5),
                ],
            });

        // Surface render: one per-draw material uniform.
        let render_bgl = crate::resources::builders::uniform_bgl(
            device,
            "mc_render_bgl",
            crate::gpu::ShaderStages::FRAGMENT,
        );

        // ----------------------------------------------------------------
        // Compute pipelines.
        // ----------------------------------------------------------------
        let classify_shader = crate::resources::builders::wgsl_module(
            device,
            "mc_classify_shader",
            crate::resources::builders::wgsl_source!("mc_classify"),
        );
        let classify_layout = crate::resources::builders::pipeline_layout(
            device,
            "mc_classify_layout",
            &[&classify_bgl],
        );
        let classify_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "mc_classify_pipeline",
            &classify_layout,
            &classify_shader,
            "main",
        );

        let prefix_sum_shader = crate::resources::builders::wgsl_module(
            device,
            "mc_prefix_sum_shader",
            crate::resources::builders::wgsl_source!("mc_prefix_sum"),
        );
        let prefix_sum_layout = crate::resources::builders::pipeline_layout(
            device,
            "mc_prefix_sum_layout",
            &[&prefix_sum_bgl],
        );
        let prefix_sum_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "mc_prefix_sum_pipeline",
            &prefix_sum_layout,
            &prefix_sum_shader,
            "main",
        );

        let generate_shader = crate::resources::builders::wgsl_module(
            device,
            "mc_generate_shader",
            crate::resources::builders::wgsl_source!("mc_generate"),
        );
        let generate_layout = crate::resources::builders::pipeline_layout(
            device,
            "mc_generate_layout",
            &[&generate_bgl],
        );
        let generate_pipeline = crate::resources::builders::compute_pipeline(
            device,
            "mc_generate_pipeline",
            &generate_layout,
            &generate_shader,
            "main",
        );

        // ----------------------------------------------------------------
        // Surface render pipeline.
        // ----------------------------------------------------------------
        let surface_shader = crate::resources::builders::wgsl_module(
            device,
            "mc_surface_shader",
            crate::resources::builders::wgsl_source!("mc_surface"),
        );
        let surface_layout = crate::resources::builders::standard_scene_layout(
            device,
            "mc_surface_layout",
            resources.shared_bindings().group0_layout,
            &render_bgl,
        );

        let vertex_attrs = [
            crate::gpu::VertexAttribute {
                format: crate::gpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            crate::gpu::VertexAttribute {
                format: crate::gpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ];
        let vertex_layout = crate::gpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: crate::gpu::VertexStepMode::Vertex,
            attributes: &vertex_attrs,
        };

        // ----------------------------------------------------------------
        // Shadow-cast pipeline. `mc_surface_pipeline` draws with
        // `cull_mode: None` (open isosurfaces are expected), so the caster
        // matches with `cull_mode: None` and the two-sided caster bias --
        // the same convention Ribbon's shadow caster uses.
        // ----------------------------------------------------------------
        let mc_shadow_shader = crate::resources::builders::wgsl_module(
            device,
            "mc_shadow_shader",
            crate::resources::builders::wgsl_source!("mc_shadow"),
        );
        let mut mc_shadow_opts = crate::resources::PluginPipelineOpts::new(
            Some("mc_shadow_pipeline"),
            &mc_shadow_shader,
            "vs_main",
            "",
            std::slice::from_ref(&vertex_layout),
        );
        mc_shadow_opts.primitive.cull_mode = None;
        mc_shadow_opts.depth_compare = crate::gpu::CompareFunction::Less;
        // The isosurface is an open, thin shell, so it self-shadows badly under
        // the mild default. Same bias the lib uses where the shadow pass does
        // not cull.
        mc_shadow_opts.depth_bias =
            Some(crate::resources::mesh::mesh_pipelines::CSM_SHADOW_BIAS_TWO_SIDED);
        let mc_shadow_pipeline = resources.build_shadow_pipeline(device, &mc_shadow_opts);

        // ----------------------------------------------------------------
        // Wireframe render pipeline.
        // ----------------------------------------------------------------
        let wireframe_render_bgl =
            device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
                label: Some("mc_wireframe_render_bgl"),
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

        let wireframe_shader = crate::resources::builders::wgsl_module(
            device,
            "mc_wireframe_shader",
            crate::resources::builders::wgsl_source!("mc_wireframe"),
        );
        let wireframe_layout = crate::resources::builders::standard_scene_layout(
            device,
            "mc_wireframe_layout",
            resources.shared_bindings().group0_layout,
            &wireframe_render_bgl,
        );
        // ----------------------------------------------------------------
        // Commit all resources.
        // ----------------------------------------------------------------
        let case_count_buf = mc_case_count_buf;
        let case_table_buf = mc_case_table_buf;
        let shadow_pipeline = mc_shadow_pipeline;
        let surface_pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "mc_surface_pipeline",
                layout: &surface_layout,
                shader: &surface_shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[vertex_layout.clone()],
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::LessEqual,
                sample_count: 1,
                ldr_format: resources.target_format,
            },
        );
        let wireframe_pipeline = crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "mc_wireframe_pipeline",
                layout: &wireframe_layout,
                shader: &wireframe_shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[], // positions read from storage buffer
                blend: Some(crate::gpu::BlendState::ALPHA_BLENDING),
                topology: crate::gpu::PrimitiveTopology::LineList,
                cull_mode: None,
                depth_write: true,
                depth_compare: crate::gpu::CompareFunction::LessEqual,
                sample_count: 1,
                ldr_format: resources.target_format,
            },
        );
        // Outline mask: the generated vertex buffer rasterised into the R8
        // mask. MC vertices are world-space, so there is no model transform
        // and no group-1 data at all. LessEqual matches the surface pipeline
        // so the mask marks the surface's own front pixels instead of
        // rejecting them at equal depth.
        let mask_shader = crate::resources::builders::wgsl_module(
            device,
            "mc_outline_mask_shader",
            crate::resources::builders::wgsl_source!("mc_outline_mask"),
        );
        let mask_pipeline = resources.build_mask_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[],
                ..crate::resources::PluginPipelineOpts::new(
                    Some("mc_outline_mask_pipeline"),
                    &mask_shader,
                    "vs_main",
                    "fs_main",
                    &[MC_VERTEX_LAYOUT],
                )
            },
        );

        // Pick: the same generated vertex buffer, writing the item's object id.
        // Group 1 is the per-item object-id uniform.
        let pick_id_bgl = device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
            label: Some("mc_pick_id_bgl"),
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
            "mc_pick_shader",
            crate::resources::builders::wgsl_source!("mc_pick"),
        );
        let pick_pipeline = resources.build_pick_pipeline(
            device,
            &crate::resources::PluginPipelineOpts {
                primitive: crate::gpu::PrimitiveState {
                    topology: crate::gpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                extra_bind_group_layouts: &[&pick_id_bgl],
                ..crate::resources::PluginPipelineOpts::new(
                    Some("mc_pick_pipeline"),
                    &pick_shader,
                    "vs_main",
                    "fs_main",
                    &[MC_VERTEX_LAYOUT],
                )
            },
        );

        Self {
            classify_pipeline,
            prefix_sum_pipeline,
            generate_pipeline,
            surface_pipeline,
            wireframe_pipeline,
            shadow_pipeline,
            mask_pipeline,
            pick_pipeline,
            pick_id_bgl,
            wireframe_render_bgl,
            classify_bgl,
            prefix_sum_bgl,
            generate_bgl,
            render_bgl,
            case_count_buf,
            case_table_buf,
        }
    }

    /// Object-id uniform + bind group for the GPU pick pass, or `None` when
    /// the item is not pickable.
    pub(super) fn pick_bind_group(
        &self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        pick_id: PickId,
    ) -> Option<(crate::gpu::Buffer, crate::gpu::BindGroup)> {
        if pick_id == PickId::NONE {
            return None;
        }
        let id_data: [u32; 4] = [pick_id.0 as u32, 0, 0, 0];
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("mc_pick_id_buf"),
            size: std::mem::size_of_val(&id_data) as u64,
            usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&id_data));
        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
            label: Some("mc_pick_id_bg"),
            layout: &self.pick_id_bgl,
            entries: &[crate::gpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        Some((buf, bg))
    }

    /// Dispatch the three compute passes for every submitted item and build
    /// the per-item draw data. Returns the encoded work for the lib's
    /// deferred-submit sink alongside it.
    pub(super) fn run_jobs(
        &self,
        device: &crate::gpu::Device,
        resources: &DeviceResources,
        jobs: &[GpuMarchingCubesItem],
    ) -> (Vec<McFrame>, Option<crate::gpu::CommandBuffer>) {
        if jobs.is_empty() {
            return (Vec::new(), None);
        }

        let classify_pipeline = &self.classify_pipeline;
        let prefix_sum_pipeline = &self.prefix_sum_pipeline;
        let generate_pipeline = &self.generate_pipeline;
        let classify_bgl = &self.classify_bgl;
        let prefix_sum_bgl = &self.prefix_sum_bgl;
        let generate_bgl = &self.generate_bgl;
        let render_bgl = &self.render_bgl;
        let case_count_buf = &self.case_count_buf;
        let case_table_buf = &self.case_table_buf;

        let mut frame_data = Vec::with_capacity(jobs.len());
        let mut encoder = device.create_command_encoder(&crate::gpu::CommandEncoderDescriptor {
            label: Some("mc_compute_encoder"),
        });

        // Refresh slab scalars from external sources before any compute.
        // Once per unique volume, even when several items reference it. The
        // copies sit in the same encoder ahead of the compute passes, so
        // queue-submission order is the only synchronisation needed against
        // the consumer's earlier compute submissions.
        let mut scalar_copied: Vec<u32> = Vec::new();
        for job in jobs {
            if scalar_copied.contains(&job.volume_id.index) {
                continue;
            }
            let Some(vol) = resources.mc_volume(job.volume_id) else {
                continue;
            };
            if let Some(src) = &vol.external_scalar {
                for slab in &vol.slabs {
                    encoder.copy_buffer_to_buffer(
                        &src.buffer,
                        src.offset_bytes + slab.scalar_byte_offset,
                        &slab.scalar_buf,
                        0,
                        slab.scalar_buf.size(),
                    );
                }
                scalar_copied.push(job.volume_id.index);
            }
        }

        for job in jobs {
            let Some(vol) = resources.mc_volume(job.volume_id) else {
                continue;
            };

            // ----------------------------------------------------------
            // Per-item surface material (one bind group shared by all slabs).
            // ----------------------------------------------------------
            let mat_raw = McSurfaceRaw {
                base_colour: job.material.base_colour.to_linear_rgb(),
                roughness: job.material.roughness,
                unlit: job.settings.unlit as u32,
                opacity: job.settings.opacity,
                ambient: job.material.ambient,
                receive_shadows: job.settings.receive_shadows as u32,
            };
            let mat_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("mc_surface_mat"),
                contents: bytemuck::bytes_of(&mat_raw),
                usage: crate::gpu::BufferUsages::UNIFORM,
            });
            let render_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                label: Some("mc_render_bg"),
                layout: render_bgl,
                entries: &[crate::gpu::BindGroupEntry {
                    binding: 0,
                    resource: mat_buf.as_entire_binding(),
                }],
            });

            // Run all three compute passes for each slab independently.
            for slab in &vol.slabs {
                let cc = slab.cell_count;
                let bc = slab.block_count;

                // ----------------------------------------------------------
                // Per-slab classify uniform.
                // ----------------------------------------------------------
                let classify_params = ClassifyParams {
                    nx: slab.dims[0],
                    ny: slab.dims[1],
                    nz: slab.dims[2],
                    isovalue: job.isovalue,
                };
                let classify_uniform =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("mc_classify_uniform"),
                        contents: bytemuck::bytes_of(&classify_params),
                        usage: crate::gpu::BufferUsages::UNIFORM,
                    });

                let classify_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("mc_classify_bg"),
                    layout: classify_bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: classify_uniform.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: slab.scalar_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 2,
                            resource: case_count_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 3,
                            resource: slab.counts_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 4,
                            resource: slab.case_idx_buf.as_entire_binding(),
                        },
                    ],
                });

                // ----------------------------------------------------------
                // Per-slab prefix-sum uniforms (one per level).
                // ----------------------------------------------------------
                let ps_uniforms: [crate::gpu::Buffer; 3] = std::array::from_fn(|level| {
                    let params = PrefixSumParams {
                        cell_count: cc,
                        block_count: bc,
                        level: level as u32,
                        _pad: 0,
                    };
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("mc_ps_uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: crate::gpu::BufferUsages::UNIFORM,
                    })
                });

                let ps_bgs: [crate::gpu::BindGroup; 3] = std::array::from_fn(|level| {
                    device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("mc_ps_bg"),
                        layout: prefix_sum_bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: ps_uniforms[level].as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.counts_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 2,
                                resource: slab.offsets_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 3,
                                resource: slab.block_sums_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 4,
                                resource: slab.indirect_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 5,
                                resource: slab.wire_indirect_buf.as_entire_binding(),
                            },
                        ],
                    })
                });

                // ----------------------------------------------------------
                // Per-slab generate uniform (origin_z shifted by slab offset).
                // ----------------------------------------------------------
                let generate_params = GenerateParams {
                    nx: slab.dims[0],
                    ny: slab.dims[1],
                    nz: slab.dims[2],
                    isovalue: job.isovalue,
                    origin_x: slab.origin[0],
                    origin_y: slab.origin[1],
                    origin_z: slab.origin[2],
                    _pad0: 0.0,
                    spacing_x: slab.spacing[0],
                    spacing_y: slab.spacing[1],
                    spacing_z: slab.spacing[2],
                    _pad1: 0.0,
                };
                let generate_uniform =
                    device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                        label: Some("mc_generate_uniform"),
                        contents: bytemuck::bytes_of(&generate_params),
                        usage: crate::gpu::BufferUsages::UNIFORM,
                    });

                let generate_bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                    label: Some("mc_generate_bg"),
                    layout: generate_bgl,
                    entries: &[
                        crate::gpu::BindGroupEntry {
                            binding: 0,
                            resource: generate_uniform.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 1,
                            resource: slab.scalar_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 2,
                            resource: case_table_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 3,
                            resource: slab.offsets_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 4,
                            resource: slab.case_idx_buf.as_entire_binding(),
                        },
                        crate::gpu::BindGroupEntry {
                            binding: 5,
                            resource: slab.vertex_buf.as_entire_binding(),
                        },
                    ],
                });

                // ----------------------------------------------------------
                // Pass 1: classify.
                // ----------------------------------------------------------
                {
                    let mut cp = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                        label: Some("mc_classify_pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(classify_pipeline);
                    cp.set_bind_group(0, &classify_bg, &[]);
                    cp.dispatch_workgroups(cc.div_ceil(256), 1, 1);
                }

                // ----------------------------------------------------------
                // Pass 2a: prefix sum level 0.
                // ----------------------------------------------------------
                {
                    let mut cp = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                        label: Some("mc_ps_level0_pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(prefix_sum_pipeline);
                    cp.set_bind_group(0, &ps_bgs[0], &[]);
                    cp.dispatch_workgroups(bc, 1, 1);
                }

                // ----------------------------------------------------------
                // Pass 2b: prefix sum level 1 (single workgroup, sequential).
                // ----------------------------------------------------------
                {
                    let mut cp = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                        label: Some("mc_ps_level1_pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(prefix_sum_pipeline);
                    cp.set_bind_group(0, &ps_bgs[1], &[]);
                    cp.dispatch_workgroups(1, 1, 1);
                }

                // ----------------------------------------------------------
                // Pass 2c: prefix sum level 2 (propagate block offsets).
                // ----------------------------------------------------------
                {
                    let mut cp = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                        label: Some("mc_ps_level2_pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(prefix_sum_pipeline);
                    cp.set_bind_group(0, &ps_bgs[2], &[]);
                    cp.dispatch_workgroups(bc, 1, 1);
                }

                // ----------------------------------------------------------
                // Pass 3: generate vertices.
                // ----------------------------------------------------------
                {
                    let mut cp = encoder.begin_compute_pass(&crate::gpu::ComputePassDescriptor {
                        label: Some("mc_generate_pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(generate_pipeline);
                    cp.set_bind_group(0, &generate_bg, &[]);
                    cp.dispatch_workgroups(cc.div_ceil(256), 1, 1);
                }
            }

            let wire_slab_bgs: Vec<crate::gpu::BindGroup> = {
                let wire_bgl = &self.wireframe_render_bgl;
                vol.slabs
                    .iter()
                    .map(|slab| {
                        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                            label: Some("mc_wire_slab_bg"),
                            layout: wire_bgl,
                            entries: &[crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: slab.vertex_buf.as_entire_binding(),
                            }],
                        })
                    })
                    .collect()
            };

            // Buffer handles are reference-counted, so cloning them here is
            // what lets the draw hooks reach the generated geometry without a
            // borrow of the volume store.
            let slabs: Vec<McSlabDraw> = vol
                .slabs
                .iter()
                .map(|slab| McSlabDraw {
                    vertex_buf: slab.vertex_buf.clone(),
                    indirect_buf: slab.indirect_buf.clone(),
                    wire_indirect_buf: slab.wire_indirect_buf.clone(),
                })
                .collect();

            frame_data.push(McFrame {
                slabs,
                render_bg,
                wireframe: job.settings.wireframe,
                wire_slab_bgs,
                pick_id: job.settings.pick_id,
                cast_shadows: job.settings.cast_shadows,
                selected: job.settings.selected,
                hidden: job.settings.hidden,
            });
        }

        (frame_data, Some(encoder.finish()))
    }
}

/// The MC compute output vertex layout: position at offset 0, normal at 12.
const MC_VERTEX_LAYOUT: crate::gpu::VertexBufferLayout<'static> = crate::gpu::VertexBufferLayout {
    array_stride: 24,
    step_mode: crate::gpu::VertexStepMode::Vertex,
    attributes: &[crate::gpu::VertexAttribute {
        offset: 0,
        shader_location: 0,
        format: crate::gpu::VertexFormat::Float32x3,
    }],
};

/// Triangle count per case: derived from TRI_TABLE by counting non-sentinel entries.
fn case_triangle_count_table() -> [u32; 256] {
    let mut out = [0u32; 256];
    for (i, row) in TRI_TABLE.iter().enumerate() {
        let mut count = 0u32;
        let mut j = 0;
        while j < 15 && row[j] >= 0 {
            count += 1;
            j += 3;
        }
        out[i] = count;
    }
    out
}

/// Flat TRI_TABLE for the GPU: 256 x 16 i32 values.
fn case_table_flat() -> [i32; 256 * 16] {
    let mut out = [-1i32; 256 * 16];
    for (i, row) in TRI_TABLE.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[i * 16 + j] = v as i32;
        }
    }
    out
}

fn bgl_uniform(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::COMPUTE,
        ty: crate::gpu::BindingType::Buffer {
            ty: crate::gpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::COMPUTE,
        ty: crate::gpu::BindingType::Buffer {
            ty: crate::gpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> crate::gpu::BindGroupLayoutEntry {
    crate::gpu::BindGroupLayoutEntry {
        binding,
        visibility: crate::gpu::ShaderStages::COMPUTE,
        ty: crate::gpu::BindingType::Buffer {
            ty: crate::gpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ---------------------------------------------------------------------------
// Raw uniform buffer layouts (bytemuck-safe)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ClassifyParams {
    nx: u32,
    ny: u32,
    nz: u32,
    isovalue: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrefixSumParams {
    cell_count: u32,
    block_count: u32,
    level: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GenerateParams {
    nx: u32,
    ny: u32,
    nz: u32,
    isovalue: f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    _pad0: f32,
    spacing_x: f32,
    spacing_y: f32,
    spacing_z: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct McSurfaceRaw {
    base_colour: [f32; 3],
    roughness: f32,
    unlit: u32,
    opacity: f32,
    /// Per-material ambient scalar from `Material::ambient`. Added to the
    /// hemisphere ambient term so the MC shaded result matches the regular
    /// mesh shader's Blinn-Phong path on materials that use the default
    /// `ambient = 0.15`. Without this field the MC surface reads notably
    /// darker on its shadowed side than an equivalent regular mesh.
    ambient: f32,
    receive_shadows: u32,
}
