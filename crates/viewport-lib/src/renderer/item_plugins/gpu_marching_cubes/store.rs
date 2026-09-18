//! The scalar volumes this item type holds on the consumer's behalf.
//!
//! A `GpuMarchingCubesItem` names a volume by [`McVolumeId`] rather than
//! carrying its scalar field, so the field is uploaded once and triangulated
//! every frame from here. Each volume is split into Z-axis slabs, which is what
//! keeps every allocation inside `device.limits().max_buffer_size` no matter
//! how large the field is. The upload entry points are the `*_mc_*` and
//! `*_volume_for_mc` methods on
//! [`ViewportRenderer`](crate::renderer::ViewportRenderer), which find this
//! plugin by name and call through to the methods below.

use super::types::McVolumeId;
use crate::geometry::marching_cubes::VolumeData;
use crate::gpu::util::DeviceExt as _;

/// GPU buffers for one Z-axis slab of an uploaded volume.
///
/// A slab covers `dims[2]` scalar Z-layers (`dims[2] - 1` cell layers).
/// Adjacent slabs share exactly one scalar Z-layer at their boundary so MC
/// edge interpolation produces no seams.
pub(super) struct McSlabGpuData {
    pub scalar_buf: crate::gpu::Buffer, // f32 per slab node; STORAGE | COPY_DST
    /// Byte offset of this slab's first scalar in the full linear volume
    /// (x-fastest node order). Used to source the slab's range out of an
    /// external scalar buffer with one `copy_buffer_to_buffer` per slab.
    pub scalar_byte_offset: u64,
    pub counts_buf: crate::gpu::Buffer, // u32 per slab cell; STORAGE
    pub case_idx_buf: crate::gpu::Buffer, // u32 per slab cell; STORAGE
    pub offsets_buf: crate::gpu::Buffer, // u32 per slab cell; STORAGE
    pub block_sums_buf: crate::gpu::Buffer, // u32 per slab block; STORAGE
    pub vertex_buf: crate::gpu::Buffer, // f32 * 6 per vertex; STORAGE | VERTEX
    pub indirect_buf: crate::gpu::Buffer, // 4 u32; STORAGE | INDIRECT (surface draw)
    pub wire_indirect_buf: crate::gpu::Buffer, // 4 u32; STORAGE | INDIRECT (wireframe draw)
    pub dims: [u32; 3],                 // [nx, ny, slab_nz] (scalar layers)
    pub origin: [f32; 3],               // world origin; z is offset per slab
    pub spacing: [f32; 3],
    pub cell_count: u32,
    pub block_count: u32,
}

/// Persistent GPU resources for one uploaded volume, split into Z-axis slabs.
///
/// Z-axis chunking keeps every allocation within `device.limits().max_buffer_size`
/// regardless of volume size. The single-slab path is equivalent to the old layout.
pub(super) struct McVolumeGpuData {
    pub slabs: Vec<McSlabGpuData>,
    /// Full-volume scalar dims `[nx, ny, nz]`, kept for validating an
    /// external scalar source against the volume's node count.
    pub dims: [u32; 3],
    /// When `Some`, the slab scalar buffers are refreshed from this
    /// caller-supplied buffer before every MC dispatch, so the isosurface
    /// tracks the buffer's contents with no CPU upload.
    pub external_scalar: Option<McExternalScalarSource>,
}

/// A caller-supplied buffer feeding a volume's scalar field.
pub(super) struct McExternalScalarSource {
    pub buffer: crate::gpu::Buffer,
    /// Byte offset of the volume's first scalar inside `buffer`.
    pub offset_bytes: u64,
}

impl crate::resources::handle::GpuByteSize for McVolumeGpuData {
    /// Resident GPU bytes across every slab buffer of this volume.
    fn gpu_bytes(&self) -> u64 {
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

/// CPU + GPU-buffer work for an MC volume upload, factored out so the same
/// code can run on a worker thread for the async path.
pub(super) fn build_mc_volume_gpu_data(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
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

            let scalar_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("mc_scalar_buf"),
                contents: bytemuck::cast_slice(&vol.data[scalar_start..scalar_end]),
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            });
            let counts_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_counts_buf"),
                size: slab_cell_bytes,
                usage: crate::gpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let case_idx_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_case_idx_buf"),
                size: slab_cell_bytes,
                usage: crate::gpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let offsets_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_offsets_buf"),
                size: slab_cell_bytes,
                usage: crate::gpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let block_sums_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_block_sums_buf"),
                size: slab_block_bytes,
                usage: crate::gpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let vertex_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("mc_vertex_buf"),
                size: slab_vertex_bytes,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
            let initial_indirect = bytemuck::cast_slice(&[0u32, 1u32, 0u32, 0u32]);
            let indirect_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                label: Some("mc_indirect_buf"),
                // Initial: 0 vertices, 1 instance, 0 first_vertex, 0 first_instance.
                contents: initial_indirect,
                usage: crate::gpu::BufferUsages::STORAGE
                    | crate::gpu::BufferUsages::INDIRECT
                    | crate::gpu::BufferUsages::COPY_DST,
            });
            let wire_indirect_buf =
                device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
                    label: Some("mc_wire_indirect_buf"),
                    contents: initial_indirect,
                    usage: crate::gpu::BufferUsages::STORAGE
                        | crate::gpu::BufferUsages::INDIRECT
                        | crate::gpu::BufferUsages::COPY_DST,
                });

            slabs.push(McSlabGpuData {
                scalar_buf,
                scalar_byte_offset: scalar_start as u64 * 4,
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

        let _ = queue;

        Ok(McVolumeGpuData {
            slabs,
            dims: vol.dims,
            external_scalar: None,
        })
    }
}

/// Slotted store for uploaded marching-cubes volumes.
///
/// A removed volume leaves an empty slot that a later upload reuses. Each slot
/// carries a generation bumped on removal, and a [`McVolumeId`] captures the
/// generation it was issued against, so a stale handle resolves to `None`
/// rather than aliasing the volume now in its slot.
pub(super) type McVolumeStore = crate::resources::handle::SlotStore<McVolumeGpuData, McVolumeId>;
