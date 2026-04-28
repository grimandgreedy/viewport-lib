//! GPU marching cubes — Phase 17.
//!
//! Three-pass GPU compute pipeline for isosurface extraction:
//!   1. Classify — computes case index and triangle count per cell.
//!   2. Prefix sum — hierarchical exclusive scan to build triangle offsets.
//!   3. Generate — interpolates vertex positions and normals into a vertex buffer.
//!
//! The output is drawn with a lightweight Phong render pipeline via `draw_indirect`.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use crate::{
    geometry::marching_cubes::{TRI_TABLE, VolumeData},
    resources::ViewportGpuResources,
    scene::material::Material,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Handle to a volume scalar field uploaded for GPU marching cubes.
///
/// Returned by [`ViewportGpuResources::upload_volume_for_mc`]. Pass to
/// [`GpuMarchingCubesJob`] to select which volume to triangulate each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VolumeGpuId(pub(crate) usize);

/// One GPU marching cubes draw job submitted per frame.
///
/// The volume referenced by `volume_id` is triangulated on the GPU at `isovalue`
/// and drawn with `material`. No CPU readback occurs; the vertex count is
/// determined by an indirect draw call.
pub struct GpuMarchingCubesJob {
    /// Volume to triangulate (must remain alive).
    pub volume_id: VolumeGpuId,
    /// Isovalue at which to extract the surface.
    pub isovalue: f32,
    /// Surface material (colour + roughness).
    pub material: Material,
}

// ---------------------------------------------------------------------------
// GPU-internal types
// ---------------------------------------------------------------------------

/// GPU buffers for one Z-axis slab of an uploaded volume.
///
/// A slab covers `dims[2]` scalar Z-layers (`dims[2] - 1` cell layers).
/// Adjacent slabs share exactly one scalar Z-layer at their boundary so MC
/// edge interpolation produces no seams.
pub(crate) struct McSlabGpuData {
    pub scalar_buf:     wgpu::Buffer,  // f32 per slab node; STORAGE | COPY_DST
    pub counts_buf:     wgpu::Buffer,  // u32 per slab cell; STORAGE
    pub case_idx_buf:   wgpu::Buffer,  // u32 per slab cell; STORAGE
    pub offsets_buf:    wgpu::Buffer,  // u32 per slab cell; STORAGE
    pub block_sums_buf: wgpu::Buffer,  // u32 per slab block; STORAGE
    pub vertex_buf:     wgpu::Buffer,  // f32 * 6 per vertex; STORAGE | VERTEX
    pub indirect_buf:   wgpu::Buffer,  // 4 u32; STORAGE | INDIRECT
    pub dims:           [u32; 3],      // [nx, ny, slab_nz] (scalar layers)
    pub origin:         [f32; 3],      // world origin; z is offset per slab
    pub spacing:        [f32; 3],
    pub cell_count:     u32,
    pub block_count:    u32,
}

/// Persistent GPU resources for one uploaded volume, split into Z-axis slabs.
///
/// Z-axis chunking keeps every allocation within `device.limits().max_buffer_size`
/// regardless of volume size. The single-slab path is equivalent to the old layout.
pub(crate) struct McVolumeGpuData {
    pub slabs: Vec<McSlabGpuData>,
    /// False after `remove_mc_volume` is called; the slot is reused lazily.
    pub alive: bool,
}

/// Per-frame data for one MC job, consumed by the render phase.
pub(crate) struct McFrameData {
    pub volume_idx: usize,
    pub render_bg:  wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Raw uniform buffer layouts (bytemuck-safe)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClassifyParams {
    nx:       u32,
    ny:       u32,
    nz:       u32,
    isovalue: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefixSumParams {
    cell_count:  u32,
    block_count: u32,
    level:       u32,
    _pad:        u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GenerateParams {
    nx:       u32,
    ny:       u32,
    nz:       u32,
    isovalue: f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    _pad0:    f32,
    spacing_x: f32,
    spacing_y: f32,
    spacing_z: f32,
    _pad1:    f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct McSurfaceRaw {
    base_color: [f32; 3],
    roughness:  f32,
}

// ---------------------------------------------------------------------------
// Lookup table helpers
// ---------------------------------------------------------------------------

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

/// Flat TRI_TABLE for the GPU: 256 × 16 i32 values.
fn case_table_flat() -> [i32; 256 * 16] {
    let mut out = [-1i32; 256 * 16];
    for (i, row) in TRI_TABLE.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[i * 16 + j] = v as i32;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Pipeline init and volume upload (impl ViewportGpuResources)
// ---------------------------------------------------------------------------

impl ViewportGpuResources {
    /// Lazily create all GPU MC pipelines and shared lookup buffers.
    ///
    /// No-op if already initialised.
    pub(crate) fn ensure_mc_pipelines(&mut self, device: &wgpu::Device) {
        if self.mc_classify_pipeline.is_some() {
            return;
        }

        // ----------------------------------------------------------------
        // Shared lookup buffers (uploaded once).
        // ----------------------------------------------------------------
        let count_table = case_triangle_count_table();
        let mc_case_count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc_case_count_buf"),
            contents: bytemuck::cast_slice(&count_table),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let flat_table = case_table_flat();
        let mc_case_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc_case_table_buf"),
            contents: bytemuck::cast_slice(&flat_table),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ----------------------------------------------------------------
        // Bind group layouts.
        // ----------------------------------------------------------------

        // Classify: 5 bindings (uniform + 2 read storage + 2 rw storage).
        let classify_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mc_classify_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_rw(3),
                bgl_storage_rw(4),
            ],
        });

        // Prefix sum: 5 bindings (uniform + ro + 3 rw).
        let prefix_sum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mc_prefix_sum_bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_storage_rw(3),
                bgl_storage_rw(4),
            ],
        });

        // Generate: 6 bindings (uniform + 3 ro + 2 rw [case_indices ro, vertex_buf rw]).
        let generate_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mc_render_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // ----------------------------------------------------------------
        // Compute pipelines.
        // ----------------------------------------------------------------
        let classify_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc_classify_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mc_classify.wgsl").into(),
            ),
        });
        let classify_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc_classify_layout"),
            bind_group_layouts: &[&classify_bgl],
            push_constant_ranges: &[],
        });
        let classify_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mc_classify_pipeline"),
                layout: Some(&classify_layout),
                module: &classify_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc_prefix_sum_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mc_prefix_sum.wgsl").into(),
            ),
        });
        let prefix_sum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc_prefix_sum_layout"),
            bind_group_layouts: &[&prefix_sum_bgl],
            push_constant_ranges: &[],
        });
        let prefix_sum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mc_prefix_sum_pipeline"),
                layout: Some(&prefix_sum_layout),
                module: &prefix_sum_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let generate_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc_generate_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mc_generate.wgsl").into(),
            ),
        });
        let generate_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc_generate_layout"),
            bind_group_layouts: &[&generate_bgl],
            push_constant_ranges: &[],
        });
        let generate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mc_generate_pipeline"),
                layout: Some(&generate_layout),
                module: &generate_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // ----------------------------------------------------------------
        // Surface render pipeline.
        // ----------------------------------------------------------------
        let surface_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc_surface_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mc_surface.wgsl").into(),
            ),
        });
        let surface_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc_surface_layout"),
            bind_group_layouts: &[&self.camera_bind_group_layout, &render_bgl],
            push_constant_ranges: &[],
        });

        let vertex_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
        ];
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attrs,
        };

        let surface_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mc_surface_pipeline"),
                layout: Some(&surface_layout),
                vertex: wgpu::VertexState {
                    module: &surface_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &surface_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.target_format,
                        blend: None, // opaque
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
            });

        // ----------------------------------------------------------------
        // Commit all resources.
        // ----------------------------------------------------------------
        self.mc_case_count_buf    = Some(mc_case_count_buf);
        self.mc_case_table_buf    = Some(mc_case_table_buf);
        self.mc_classify_bgl      = Some(classify_bgl);
        self.mc_prefix_sum_bgl    = Some(prefix_sum_bgl);
        self.mc_generate_bgl      = Some(generate_bgl);
        self.mc_render_bgl        = Some(render_bgl);
        self.mc_classify_pipeline = Some(classify_pipeline);
        self.mc_prefix_sum_pipeline = Some(prefix_sum_pipeline);
        self.mc_generate_pipeline = Some(generate_pipeline);
        self.mc_surface_pipeline  = Some(surface_pipeline);
    }

    /// Upload a [`VolumeData`] to GPU, pre-allocating all intermediate and output
    /// buffers for GPU marching cubes.
    ///
    /// The returned [`VolumeGpuId`] is stable until [`remove_mc_volume`] is called.
    ///
    /// Returns `Err(ViewportError::McBufferTooLarge)` if any required buffer exceeds
    /// the device's `max_buffer_size`; the caller should fall back to CPU isosurface
    /// extraction.
    pub fn upload_volume_for_mc(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vol: &VolumeData,
    ) -> crate::ViewportResult<VolumeGpuId> {
        let [nx, ny, nz] = vol.dims;
        let max_buf  = device.limits().max_buffer_size;

        // Worst-case vertex buffer bytes per Z-cell-layer:
        // (nx-1)*(ny-1) cells × 5 triangles × 3 vertices × 24 bytes = cells_xy × 360.
        // Compute how many Z-cell layers fit within max_buffer_size.
        let cells_xy = (nx - 1) as u64 * (ny - 1) as u64;
        let max_cells_per_slab = max_buf / (15 * 24);
        let z_cells_per_slab = if cells_xy > 0 {
            (max_cells_per_slab / cells_xy).min((nz - 1) as u64) as u32
        } else {
            nz - 1
        };
        if z_cells_per_slab == 0 {
            // Even a single Z-layer of cells exceeds the device vertex buffer limit.
            return Err(crate::ViewportError::McBufferTooLarge {
                buffer: "vertex_buf",
                needed: cells_xy * 15 * 24,
                limit:  max_buf,
            });
        }

        let nz_cells_total = nz - 1;
        let slab_count     = nz_cells_total.div_ceil(z_cells_per_slab);
        let nodes_per_z    = (nx * ny) as usize;

        let mut slabs = Vec::with_capacity(slab_count as usize);

        for s in 0..slab_count {
            let z_cell_start = s * z_cells_per_slab;
            let z_cell_end   = (z_cell_start + z_cells_per_slab).min(nz_cells_total);
            let slab_z_cells = z_cell_end - z_cell_start;  // cell layers in this slab
            let slab_nz      = slab_z_cells + 1;           // scalar layers in this slab

            // slab_cell_count is bounded by max_cells_per_slab, which fits in u32
            // at any realistic max_buffer_size value.
            let slab_cell_count  = (cells_xy * slab_z_cells as u64) as u32;
            let slab_block_count = slab_cell_count.div_ceil(256);
            let slab_cell_bytes  = (slab_cell_count as u64) * 4;
            let slab_block_bytes = (slab_block_count as u64) * 4;
            // At most 15 vertices per cell (5 triangles × 3 vertices) × 24 bytes each.
            let slab_vertex_bytes = (slab_cell_count as u64) * 15 * 24;

            // Scalar data is x-fastest: index = x + y*nx + z*nx*ny.
            // A Z-slab covering scalar layers z_cell_start..z_cell_start+slab_nz is
            // a contiguous slice — no copying required.
            let scalar_start  = z_cell_start as usize * nodes_per_z;
            let scalar_end    = (z_cell_start + slab_nz) as usize * nodes_per_z;
            let slab_origin_z = vol.origin[2] + z_cell_start as f32 * vol.spacing[2];

            let scalar_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc_scalar_buf"),
                contents: bytemuck::cast_slice(&vol.data[scalar_start..scalar_end]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc_counts_buf"),
                size: slab_cell_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let case_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc_case_idx_buf"),
                size: slab_cell_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc_offsets_buf"),
                size: slab_cell_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let block_sums_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc_block_sums_buf"),
                size: slab_block_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc_vertex_buf"),
                size: slab_vertex_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
            let indirect_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc_indirect_buf"),
                // Initial: 0 vertices, 1 instance, 0 first_vertex, 0 first_instance.
                contents: bytemuck::cast_slice(&[0u32, 1u32, 0u32, 0u32]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
            });

            slabs.push(McSlabGpuData {
                scalar_buf,
                counts_buf,
                case_idx_buf,
                offsets_buf,
                block_sums_buf,
                vertex_buf,
                indirect_buf,
                dims:    [nx, ny, slab_nz],
                origin:  [vol.origin[0], vol.origin[1], slab_origin_z],
                spacing: vol.spacing,
                cell_count:  slab_cell_count,
                block_count: slab_block_count,
            });
        }

        let _ = queue; // retained for potential future use (e.g. scalar updates)

        let gpu_data = McVolumeGpuData { slabs, alive: true };

        // Find a free slot (from a previous remove_mc_volume call) or push a new one.
        let idx = if let Some(free_idx) = self.mc_volumes.iter().position(|v| !v.alive) {
            self.mc_volumes[free_idx] = gpu_data;
            free_idx
        } else {
            self.mc_volumes.push(gpu_data);
            self.mc_volumes.len() - 1
        };

        Ok(VolumeGpuId(idx))
    }

    /// Mark a MC volume slot as free. The GPU buffers are dropped immediately.
    pub fn remove_mc_volume(&mut self, id: VolumeGpuId) {
        if let Some(v) = self.mc_volumes.get_mut(id.0) {
            v.alive = false;
        }
    }

    /// Dispatch all three compute passes for every pending MC job.
    ///
    /// Returns the per-frame render data to be stored in `ViewportRenderer.mc_gpu_data`.
    pub(crate) fn run_mc_jobs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        jobs: &[GpuMarchingCubesJob],
    ) -> Vec<McFrameData> {
        if jobs.is_empty() {
            return Vec::new();
        }

        let classify_pipeline = self.mc_classify_pipeline.as_ref().expect("mc pipelines");
        let prefix_sum_pipeline = self.mc_prefix_sum_pipeline.as_ref().unwrap();
        let generate_pipeline = self.mc_generate_pipeline.as_ref().unwrap();
        let classify_bgl = self.mc_classify_bgl.as_ref().unwrap();
        let prefix_sum_bgl = self.mc_prefix_sum_bgl.as_ref().unwrap();
        let generate_bgl = self.mc_generate_bgl.as_ref().unwrap();
        let render_bgl = self.mc_render_bgl.as_ref().unwrap();
        let case_count_buf = self.mc_case_count_buf.as_ref().unwrap();
        let case_table_buf = self.mc_case_table_buf.as_ref().unwrap();

        let mut frame_data = Vec::with_capacity(jobs.len());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mc_compute_encoder"),
        });

        for job in jobs {
            let vol = &self.mc_volumes[job.volume_id.0];
            if !vol.alive {
                continue;
            }

            // ----------------------------------------------------------
            // Per-job surface material (one bind group shared by all slabs).
            // ----------------------------------------------------------
            let mat_raw = McSurfaceRaw {
                base_color: job.material.base_color,
                roughness:  job.material.roughness,
            };
            let mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc_surface_mat"),
                contents: bytemuck::bytes_of(&mat_raw),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mc_render_bg"),
                layout: render_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: mat_buf.as_entire_binding() }],
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
                let classify_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mc_classify_uniform"),
                    contents: bytemuck::bytes_of(&classify_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let classify_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mc_classify_bg"),
                    layout: classify_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: classify_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: slab.scalar_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: case_count_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: slab.counts_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: slab.case_idx_buf.as_entire_binding() },
                    ],
                });

                // ----------------------------------------------------------
                // Per-slab prefix-sum uniforms (one per level).
                // ----------------------------------------------------------
                let ps_uniforms: [wgpu::Buffer; 3] = std::array::from_fn(|level| {
                    let params = PrefixSumParams { cell_count: cc, block_count: bc, level: level as u32, _pad: 0 };
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mc_ps_uniform"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
                });

                let ps_bgs: [wgpu::BindGroup; 3] = std::array::from_fn(|level| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("mc_ps_bg"),
                        layout: prefix_sum_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: ps_uniforms[level].as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: slab.counts_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 2, resource: slab.offsets_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 3, resource: slab.block_sums_buf.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 4, resource: slab.indirect_buf.as_entire_binding() },
                        ],
                    })
                });

                // ----------------------------------------------------------
                // Per-slab generate uniform (origin_z shifted by slab offset).
                // ----------------------------------------------------------
                let generate_params = GenerateParams {
                    nx:        slab.dims[0],
                    ny:        slab.dims[1],
                    nz:        slab.dims[2],
                    isovalue:  job.isovalue,
                    origin_x:  slab.origin[0],
                    origin_y:  slab.origin[1],
                    origin_z:  slab.origin[2],
                    _pad0:     0.0,
                    spacing_x: slab.spacing[0],
                    spacing_y: slab.spacing[1],
                    spacing_z: slab.spacing[2],
                    _pad1:     0.0,
                };
                let generate_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mc_generate_uniform"),
                    contents: bytemuck::bytes_of(&generate_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let generate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mc_generate_bg"),
                    layout: generate_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: generate_uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: slab.scalar_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: case_table_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: slab.offsets_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: slab.case_idx_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: slab.vertex_buf.as_entire_binding() },
                    ],
                });

                // ----------------------------------------------------------
                // Pass 1: classify.
                // ----------------------------------------------------------
                {
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("mc_generate_pass"),
                        timestamp_writes: None,
                    });
                    cp.set_pipeline(generate_pipeline);
                    cp.set_bind_group(0, &generate_bg, &[]);
                    cp.dispatch_workgroups(cc.div_ceil(256), 1, 1);
                }
            }

            frame_data.push(McFrameData { volume_idx: job.volume_id.0, render_bg });
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame_data
    }
}

// ---------------------------------------------------------------------------
// Bind group layout entry helpers
// ---------------------------------------------------------------------------

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
