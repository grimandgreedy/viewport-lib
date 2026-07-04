//! GPU marching cubes.
//!
//! Three-pass GPU compute pipeline for isosurface extraction:
//!   1. Classify       computes case index and triangle count per cell.
//!   2. Prefix sum     hierarchical exclusive scan to build triangle offsets.
//!   3. Generate       interpolates vertex positions and normals into a vertex buffer.
//!
//! The output is drawn with a lightweight Phong render pipeline via `draw_indirect`.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use crate::{
    geometry::marching_cubes::{TRI_TABLE, VolumeData},
    renderer::GpuMarchingCubesJob,
    resources::{DeviceResources, DualPipeline},
};

/// GPU marching-cubes pipelines, layouts, and static case tables.
///
/// The three compute passes (classify / prefix-sum / generate), the surface and
/// wireframe render pipelines, and the shared case-count / case-table buffers.
/// All device-shared and lazily built; `volumes` holds the per-item extraction
/// state keyed by submission order.
#[derive(Default)]
pub(crate) struct McResources {
    pub(crate) classify_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) prefix_sum_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) generate_pipeline: Option<wgpu::ComputePipeline>,
    pub(crate) surface_pipeline: Option<DualPipeline>,
    pub(crate) wireframe_pipeline: Option<DualPipeline>,
    pub(crate) wireframe_render_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) classify_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) prefix_sum_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) generate_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) render_bgl: Option<wgpu::BindGroupLayout>,
    pub(crate) case_count_buf: Option<wgpu::Buffer>,
    pub(crate) case_table_buf: Option<wgpu::Buffer>,
    pub(crate) volumes: Vec<McVolumeGpuData>,
    /// Outline mask pipeline for MC surfaces (stride-24 vertex buffer, draw_indirect).
    pub(crate) outline_mask_pipeline: Option<wgpu::RenderPipeline>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

crate::resources::handle::slot_handle! {
    /// Handle to a volume scalar field uploaded for GPU marching cubes.
    ///
    /// Returned by [`DeviceResources::upload_volume_for_mc`]. Pass to
    /// [`GpuMarchingCubesJob`] to select which volume to triangulate each frame.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. A handle whose volume was removed (its slot freed and reused
    /// by a later upload) resolves to nothing on lookup, so it cannot alias the
    /// volume now in its slot.
    pub struct McVolumeId;
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
    pub scalar_buf: wgpu::Buffer,   // f32 per slab node; STORAGE | COPY_DST
    pub counts_buf: wgpu::Buffer,   // u32 per slab cell; STORAGE
    pub case_idx_buf: wgpu::Buffer, // u32 per slab cell; STORAGE
    pub offsets_buf: wgpu::Buffer,  // u32 per slab cell; STORAGE
    pub block_sums_buf: wgpu::Buffer, // u32 per slab block; STORAGE
    pub vertex_buf: wgpu::Buffer,   // f32 * 6 per vertex; STORAGE | VERTEX
    pub indirect_buf: wgpu::Buffer, // 4 u32; STORAGE | INDIRECT (surface draw)
    pub wire_indirect_buf: wgpu::Buffer, // 4 u32; STORAGE | INDIRECT (wireframe draw)
    pub dims: [u32; 3],             // [nx, ny, slab_nz] (scalar layers)
    pub origin: [f32; 3],           // world origin; z is offset per slab
    pub spacing: [f32; 3],
    pub cell_count: u32,
    pub block_count: u32,
}

/// Persistent GPU resources for one uploaded volume, split into Z-axis slabs.
///
/// Z-axis chunking keeps every allocation within `device.limits().max_buffer_size`
/// regardless of volume size. The single-slab path is equivalent to the old layout.
pub(crate) struct McVolumeGpuData {
    pub slabs: Vec<McSlabGpuData>,
    /// False after `free_mc_volume` is called; the emptied slot is reused lazily.
    pub alive: bool,
    /// Bumped each time the slot is freed, so a handle issued for an earlier
    /// occupant no longer resolves once the slot is reused.
    pub generation: u32,
}

impl McVolumeGpuData {
    /// Resident GPU bytes across every slab buffer of this volume. Zero for a
    /// freed slot (its slabs are dropped on free).
    pub fn gpu_bytes(&self) -> u64 {
        self.slabs
            .iter()
            .map(|s| {
                s.scalar_buf.size()
                    + s.counts_buf.size()
                    + s.case_idx_buf.size()
                    + s.offsets_buf.size()
                    + s.block_sums_buf.size()
                    + s.vertex_buf.size()
                    + s.indirect_buf.size()
                    + s.wire_indirect_buf.size()
            })
            .sum()
    }
}

/// Per-frame data for one MC job, consumed by the render phase.
pub(crate) struct McFrameData {
    pub volume_idx: usize,
    pub render_bg: wgpu::BindGroup,
    /// True if this job was submitted with `appearance.wireframe = true`.
    pub wireframe: bool,
    /// Per-slab bind groups for the wireframe pipeline (binding 0 = vertex storage buffer).
    pub wire_slab_bgs: Vec<wgpu::BindGroup>,
}

/// Per-selected MC job data for the outline mask pass.
pub(crate) struct McOutlineItem {
    /// Index into `mc_gpu_data` (frame-level array of processed MC jobs).
    pub mc_gpu_idx: usize,
    pub _uniform_buf: wgpu::Buffer,
    pub mask_bind_group: wgpu::BindGroup,
}

// ---------------------------------------------------------------------------
// Raw uniform buffer layouts (bytemuck-safe)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClassifyParams {
    nx: u32,
    ny: u32,
    nz: u32,
    isovalue: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefixSumParams {
    cell_count: u32,
    block_count: u32,
    level: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
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
#[derive(Clone, Copy, Pod, Zeroable)]
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
    _pad: u32,
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

// ---------------------------------------------------------------------------
// Pipeline init and volume upload (impl DeviceResources)
// ---------------------------------------------------------------------------

impl DeviceResources {
    /// Lazily create all GPU MC pipelines and shared lookup buffers.
    ///
    /// No-op if already initialised.
    pub(crate) fn ensure_mc_pipelines(&mut self, device: &wgpu::Device) {
        if self.mc.classify_pipeline.is_some() {
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

        // Prefix sum: 6 bindings (uniform + ro + 3 rw + wire_indirect_buf rw).
        let prefix_sum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                include_str!(concat!(env!("OUT_DIR"), "/mc_classify.wgsl")).into(),
            ),
        });
        let classify_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc_classify_layout"),
            bind_group_layouts: &[&classify_bgl],
            push_constant_ranges: &[],
        });
        let classify_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
                include_str!(concat!(env!("OUT_DIR"), "/mc_prefix_sum.wgsl")).into(),
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
                include_str!(concat!(env!("OUT_DIR"), "/mc_generate.wgsl")).into(),
            ),
        });
        let generate_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc_generate_layout"),
            bind_group_layouts: &[&generate_bgl],
            push_constant_ranges: &[],
        });
        let generate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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
                include_str!(concat!(env!("OUT_DIR"), "/mc_surface.wgsl")).into(),
            ),
        });
        let surface_layout = crate::resources::builders::standard_scene_layout(
            device,
            "mc_surface_layout",
            &self.camera_bind_group_layout,
            &render_bgl,
        );

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

        // ----------------------------------------------------------------
        // Wireframe render pipeline.
        // ----------------------------------------------------------------
        let wireframe_render_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mc_wireframe_render_bgl"),
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

        let wireframe_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc_wireframe_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/mc_wireframe.wgsl")).into(),
            ),
        });
        let wireframe_layout = crate::resources::builders::standard_scene_layout(
            device,
            "mc_wireframe_layout",
            &self.camera_bind_group_layout,
            &wireframe_render_bgl,
        );
        // ----------------------------------------------------------------
        // Commit all resources.
        // ----------------------------------------------------------------
        self.mc.case_count_buf = Some(mc_case_count_buf);
        self.mc.case_table_buf = Some(mc_case_table_buf);
        self.mc.classify_bgl = Some(classify_bgl);
        self.mc.prefix_sum_bgl = Some(prefix_sum_bgl);
        self.mc.generate_bgl = Some(generate_bgl);
        self.mc.render_bgl = Some(render_bgl);
        self.mc.classify_pipeline = Some(classify_pipeline);
        self.mc.prefix_sum_pipeline = Some(prefix_sum_pipeline);
        self.mc.generate_pipeline = Some(generate_pipeline);
        self.mc.surface_pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "mc_surface_pipeline",
                layout: &surface_layout,
                shader: &surface_shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[vertex_layout.clone()],
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                depth_write: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                sample_count: 1,
                ldr_format: self.target_format,
            },
        ));
        self.mc.wireframe_render_bgl = Some(wireframe_render_bgl);
        self.mc.wireframe_pipeline = Some(crate::resources::builders::build_dual_pipeline(
            device,
            &crate::resources::builders::DualPipelineDesc {
                label: "mc_wireframe_pipeline",
                layout: &wireframe_layout,
                shader: &wireframe_shader,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
                vertex_buffers: &[], // positions read from storage buffer
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                depth_write: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                sample_count: 1,
                ldr_format: self.target_format,
            },
        ));
    }

    /// Upload a [`VolumeData`] to GPU, pre-allocating all intermediate and output
    /// buffers for GPU marching cubes.
    ///
    /// The returned [`McVolumeId`] is stable until [`free_mc_volume`] is called.
    ///
    /// Returns `Err(ViewportError::McBufferTooLarge)` if any required buffer exceeds
    /// the device's `max_buffer_size`; the caller should fall back to CPU isosurface
    /// extraction.
    pub fn upload_volume_for_mc(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vol: &VolumeData,
    ) -> crate::ViewportResult<McVolumeId> {
        let gpu_data = build_mc_volume_gpu_data(device, queue, vol)?;
        Ok(self.insert_mc_volume_gpu_data(gpu_data))
    }

    /// Main-thread half of an async marching-cubes volume upload: insert
    /// pre-built GPU data into the store and return its handle. Reuses a freed
    /// slot when one is available, carrying that slot's current generation so a
    /// stale handle to the previous occupant no longer resolves.
    pub(crate) fn insert_mc_volume_gpu_data(
        &mut self,
        mut gpu_data: McVolumeGpuData,
    ) -> McVolumeId {
        if let Some(free_idx) = self.mc.volumes.iter().position(|v| !v.alive) {
            gpu_data.generation = self.mc.volumes[free_idx].generation;
            self.mc.volumes[free_idx] = gpu_data;
            McVolumeId::new(free_idx as u32, self.mc.volumes[free_idx].generation)
        } else {
            let idx = self.mc.volumes.len();
            self.mc.volumes.push(gpu_data);
            McVolumeId::new(idx as u32, 0)
        }
    }

    /// Look up a live volume by handle, validating the generation. Returns
    /// `None` for a stale handle, a freed slot, or an out-of-range index.
    pub(crate) fn mc_volume(&self, id: McVolumeId) -> Option<&McVolumeGpuData> {
        let vol = self.mc.volumes.get(id.index as usize)?;
        if vol.generation != id.generation || !vol.alive {
            return None;
        }
        Some(vol)
    }
}

/// CPU + GPU-buffer work for an MC volume upload, factored out so the same
/// code can run on a worker thread for the async path.
pub(crate) fn build_mc_volume_gpu_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vol: &VolumeData,
) -> crate::ViewportResult<McVolumeGpuData> {
    {
        let [nx, ny, nz] = vol.dims;
        // The vertex buffer is bound as both STORAGE (compute) and VERTEX (render).
        // The binding limit for compute shaders is max_storage_buffer_binding_size, which
        // is often half of max_buffer_size (e.g. 128 MiB vs 256 MiB). Use the smaller of
        // the two so slab sizing respects both constraints.
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        let max_buf = device.limits().max_buffer_size;
        let max_limit = max_binding.min(max_buf);

        // Worst-case vertex buffer bytes per Z-cell-layer:
        // (nx-1)*(ny-1) cells x 5 triangles x 3 vertices x 24 bytes = cells_xy x 360.
        // Compute how many Z-cell layers fit within the effective limit.
        let cells_xy = (nx - 1) as u64 * (ny - 1) as u64;
        let max_cells_per_slab = max_limit / (15 * 24);
        let z_cells_per_slab = if cells_xy > 0 {
            (max_cells_per_slab / cells_xy).min((nz - 1) as u64) as u32
        } else {
            nz - 1
        };
        if z_cells_per_slab == 0 {
            // Even a single Z-layer of cells exceeds the effective binding limit.
            return Err(crate::ViewportError::McBufferTooLarge {
                buffer: "vertex_buf",
                needed: cells_xy * 15 * 24,
                limit: max_limit,
            });
        }

        let nz_cells_total = nz - 1;
        let slab_count = nz_cells_total.div_ceil(z_cells_per_slab);
        let nodes_per_z = (nx * ny) as usize;

        let mut slabs = Vec::with_capacity(slab_count as usize);

        for s in 0..slab_count {
            let z_cell_start = s * z_cells_per_slab;
            let z_cell_end = (z_cell_start + z_cells_per_slab).min(nz_cells_total);
            let slab_z_cells = z_cell_end - z_cell_start; // cell layers in this slab
            let slab_nz = slab_z_cells + 1; // scalar layers in this slab

            // slab_cell_count is bounded by max_cells_per_slab, which fits in u32
            // at any realistic max_buffer_size value.
            let slab_cell_count = (cells_xy * slab_z_cells as u64) as u32;
            let slab_block_count = slab_cell_count.div_ceil(256);
            let slab_cell_bytes = (slab_cell_count as u64) * 4;
            let slab_block_bytes = (slab_block_count as u64) * 4;
            // At most 15 vertices per cell (5 triangles x 3 vertices) x 24 bytes each.
            let slab_vertex_bytes = (slab_cell_count as u64) * 15 * 24;

            // Scalar data is x-fastest: index = x + y*nx + z*nx*ny.
            // A Z-slab covering scalar layers z_cell_start..z_cell_start+slab_nz is
            // a contiguous slice, no copying required.
            let scalar_start = z_cell_start as usize * nodes_per_z;
            let scalar_end = (z_cell_start + slab_nz) as usize * nodes_per_z;
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
            let initial_indirect = bytemuck::cast_slice(&[0u32, 1u32, 0u32, 0u32]);
            let indirect_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc_indirect_buf"),
                // Initial: 0 vertices, 1 instance, 0 first_vertex, 0 first_instance.
                contents: initial_indirect,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
            });
            let wire_indirect_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc_wire_indirect_buf"),
                contents: initial_indirect,
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
                wire_indirect_buf,
                dims: [nx, ny, slab_nz],
                origin: [vol.origin[0], vol.origin[1], slab_origin_z],
                spacing: vol.spacing,
                cell_count: slab_cell_count,
                block_count: slab_block_count,
            });
        }

        let _ = queue; // retained for potential future use (e.g. scalar updates)

        Ok(McVolumeGpuData {
            slabs,
            alive: true,
            generation: 0,
        })
    }
}

impl DeviceResources {
    /// Start an asynchronous marching-cubes-ready volume upload.
    ///
    /// Returns a [`JobId`](crate::resources::JobId) immediately. Slab
    /// sizing, scalar buffer allocation, and intermediate / output buffer
    /// allocation run on a worker thread on cloned `Device` and `Queue`
    /// handles. The apply step inserts the resulting GPU buffers into the
    /// MC volume store; once `UploadStatus::Ready`, call
    /// [`upload_result_volume_mc`](Self::upload_result_volume_mc) to take
    /// the [`McVolumeId`].
    ///
    /// Ownership of `vol` transfers into the worker.
    ///
    /// # Errors
    ///
    /// The worker surfaces
    /// [`ViewportError::McBufferTooLarge`](crate::error::ViewportError::McBufferTooLarge)
    /// through [`UploadStatus::Failed`] when the device's
    /// `max_storage_buffer_binding_size` cannot fit a single Z-cell layer.
    pub fn begin_upload_volume_for_mc(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vol: VolumeData,
    ) -> crate::resources::JobId {
        let slot = crate::resources::ResultSlot::<McVolumeId>::new();
        let slot_for_apply = slot.clone();
        let device_for_worker = device.clone();
        let queue_for_worker = queue.clone();

        let id = {
            let mut runner = self.jobs.lock().expect("upload job runner poisoned");
            runner.submit_cpu(move |progress| {
                progress.set(0.1);
                let gpu_data =
                    build_mc_volume_gpu_data(&device_for_worker, &queue_for_worker, &vol)?;
                progress.set(0.95);
                Ok(crate::resources::upload_jobs::JobProduct::with_apply(
                    Box::new(move |resources: &mut DeviceResources| {
                        let id = resources.insert_mc_volume_gpu_data(gpu_data);
                        slot_for_apply.set(id);
                    }),
                ))
            })
        };

        self.job_results
            .volume_mc
            .lock()
            .expect("volume mc result map poisoned")
            .insert(id, slot);
        id
    }

    /// Take the [`McVolumeId`] produced by a completed
    /// [`begin_upload_volume_for_mc`](Self::begin_upload_volume_for_mc) job.
    pub fn upload_result_volume_mc(
        &mut self,
        id: crate::resources::JobId,
    ) -> crate::error::ViewportResult<McVolumeId> {
        let mut map = self
            .job_results
            .volume_mc
            .lock()
            .expect("volume mc result map poisoned");
        let slot = match map.get(&id) {
            Some(s) => s.clone(),
            None => {
                return Err(crate::error::ViewportError::JobResultMissing {
                    reason: "unknown id or wrong upload type",
                });
            }
        };
        match slot.take() {
            Some(vid) => {
                map.remove(&id);
                Ok(vid)
            }
            None => Err(crate::error::ViewportError::JobNotReady),
        }
    }

    /// Free a MC volume: drop its slab GPU buffers to reclaim the memory now,
    /// mark the slot free, and bump its generation so a stale handle no longer
    /// resolves. The emptied slot is reused by a later upload. Dropping the
    /// buffers drops this volume out of [`resident_bytes`](Self::resident_bytes)
    /// immediately (wgpu defers the real GPU free until in-flight commands that
    /// reference the buffers complete).
    pub fn free_mc_volume(&mut self, id: McVolumeId) {
        if let Some(v) = self.mc.volumes.get_mut(id.index as usize) {
            if v.generation == id.generation && v.alive {
                v.slabs.clear();
                v.alive = false;
                v.generation = v.generation.wrapping_add(1);
            }
        }
    }

    /// Total resident GPU bytes across every live MC volume.
    pub(crate) fn mc_volume_resident_bytes(&self) -> u64 {
        self.mc
            .volumes
            .iter()
            .filter(|v| v.alive)
            .map(|v| v.gpu_bytes())
            .sum()
    }

    /// Lazily create the MC surface outline mask pipeline.
    ///
    /// Layout is `[camera_bind_group_layout, outline_bind_group_layout]`. The vertex
    /// buffer matches the MC output format: stride 24 (position f32x3 at offset 0,
    /// normal f32x3 at offset 12). Uses the existing `outline_mask.wgsl` shader since
    /// only position is needed. No-op if already created.
    pub(crate) fn ensure_mc_outline_mask_pipeline(&mut self, device: &wgpu::Device) {
        if self.mc.outline_mask_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mc_outline_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/outline_mask.wgsl")).into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mc_outline_mask_pipeline_layout"),
            bind_group_layouts: &[
                &self.camera_bind_group_layout,
                &self.outline.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let vert_attrs = [wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x3,
        }];
        let vert_layout = wgpu::VertexBufferLayout {
            array_stride: 24, // position (12 bytes) + normal (12 bytes)
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vert_attrs,
        };

        self.mc.outline_mask_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("mc_outline_mask_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vert_layout],
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
                    depth_compare: wgpu::CompareFunction::Less,
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

        let classify_pipeline = self.mc.classify_pipeline.as_ref().expect("mc pipelines");
        let prefix_sum_pipeline = self.mc.prefix_sum_pipeline.as_ref().unwrap();
        let generate_pipeline = self.mc.generate_pipeline.as_ref().unwrap();
        let classify_bgl = self.mc.classify_bgl.as_ref().unwrap();
        let prefix_sum_bgl = self.mc.prefix_sum_bgl.as_ref().unwrap();
        let generate_bgl = self.mc.generate_bgl.as_ref().unwrap();
        let render_bgl = self.mc.render_bgl.as_ref().unwrap();
        let case_count_buf = self.mc.case_count_buf.as_ref().unwrap();
        let case_table_buf = self.mc.case_table_buf.as_ref().unwrap();

        let mut frame_data = Vec::with_capacity(jobs.len());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mc_compute_encoder"),
        });

        for job in jobs {
            let Some(vol) = self.mc_volume(job.volume_id) else {
                continue;
            };

            // ----------------------------------------------------------
            // Per-job surface material (one bind group shared by all slabs).
            // ----------------------------------------------------------
            let mat_raw = McSurfaceRaw {
                base_colour: job.material.base_colour,
                roughness: job.material.roughness,
                unlit: job.settings.unlit as u32,
                opacity: job.settings.opacity,
                ambient: job.material.ambient,
                _pad: 0,
            };
            let mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mc_surface_mat"),
                contents: bytemuck::bytes_of(&mat_raw),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mc_render_bg"),
                layout: render_bgl,
                entries: &[wgpu::BindGroupEntry {
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
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mc_classify_uniform"),
                        contents: bytemuck::bytes_of(&classify_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                let classify_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mc_classify_bg"),
                    layout: classify_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: classify_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: slab.scalar_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: case_count_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: slab.counts_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: slab.case_idx_buf.as_entire_binding(),
                        },
                    ],
                });

                // ----------------------------------------------------------
                // Per-slab prefix-sum uniforms (one per level).
                // ----------------------------------------------------------
                let ps_uniforms: [wgpu::Buffer; 3] = std::array::from_fn(|level| {
                    let params = PrefixSumParams {
                        cell_count: cc,
                        block_count: bc,
                        level: level as u32,
                        _pad: 0,
                    };
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
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: ps_uniforms[level].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: slab.counts_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: slab.offsets_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: slab.block_sums_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: slab.indirect_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
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
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mc_generate_uniform"),
                        contents: bytemuck::bytes_of(&generate_params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                let generate_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("mc_generate_bg"),
                    layout: generate_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: generate_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: slab.scalar_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: case_table_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: slab.offsets_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: slab.case_idx_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: slab.vertex_buf.as_entire_binding(),
                        },
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

            let wire_slab_bgs: Vec<wgpu::BindGroup> =
                if let Some(ref wire_bgl) = self.mc.wireframe_render_bgl {
                    vol.slabs
                        .iter()
                        .map(|slab| {
                            device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("mc_wire_slab_bg"),
                                layout: wire_bgl,
                                entries: &[wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: slab.vertex_buf.as_entire_binding(),
                                }],
                            })
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            frame_data.push(McFrameData {
                volume_idx: job.volume_id.index(),
                render_bg,
                wireframe: job.settings.wireframe,
                wire_slab_bgs,
            });
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

#[cfg(test)]
mod residency_tests {
    use crate::DeviceResources;
    use crate::geometry::marching_cubes::VolumeData;

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

    fn sample_volume() -> VolumeData {
        let dims = [4u32, 4, 4];
        let data = (0..(dims[0] * dims[1] * dims[2]))
            .map(|i| (i % 2) as f32)
            .collect();
        VolumeData {
            data,
            dims,
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn stale_mc_volume_handle_does_not_alias_after_slot_reuse() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

        let id1 = resources
            .upload_volume_for_mc(&device, &queue, &sample_volume())
            .unwrap();
        assert!(resources.mc_volume(id1).is_some());
        resources.free_mc_volume(id1);
        assert!(
            resources.mc_volume(id1).is_none(),
            "a removed handle must not resolve"
        );

        // The next upload reuses the freed slot at a new generation.
        let id2 = resources
            .upload_volume_for_mc(&device, &queue, &sample_volume())
            .unwrap();
        assert_eq!(id1.index(), id2.index(), "the freed slot should be reused");
        assert_ne!(id1, id2, "the reused slot must carry a new generation");
        assert!(resources.mc_volume(id2).is_some());
        assert!(
            resources.mc_volume(id1).is_none(),
            "the stale handle must not alias the volume now occupying its slot"
        );
    }

    #[test]
    fn free_mc_volume_reclaims_resident_bytes() {
        let Some((device, queue)) = try_make_device() else {
            eprintln!("skipping: no wgpu adapter available");
            return;
        };
        let mut resources = DeviceResources::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb, 1);

        let start = resources.resident_bytes().mc_volume_bytes;
        let id = resources
            .upload_volume_for_mc(&device, &queue, &sample_volume())
            .unwrap();
        let after_upload = resources.resident_bytes().mc_volume_bytes;
        assert!(
            after_upload > start,
            "uploading a volume must increase resident volume bytes"
        );

        resources.free_mc_volume(id);
        assert_eq!(
            resources.resident_bytes().mc_volume_bytes,
            start,
            "freeing a volume must drop its slab buffers out of the resident total"
        );
    }
}
