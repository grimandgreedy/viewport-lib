//! The splat sets this item type holds on the consumer's behalf.
//!
//! A `GaussianSplatItem` names a set by [`GaussianSplatId`] rather than
//! carrying its splats, so the data is uploaded once and drawn every frame
//! from here. The upload entry points are the `*_gaussian_splat` methods on
//! [`ViewportRenderer`](crate::renderer::ViewportRenderer), which find this
//! plugin by name and call through to the methods below.

use crate::resources::GaussianSplatId;

pub use viewport_lib_types::data::point::{GaussianSplatData, ShDegree};

/// Check that a splat set is non-empty and its per-attribute vectors agree in
/// length. Shared by the sync, async, and replace upload paths.
pub(super) fn validate_gaussian_splat_data(
    data: &GaussianSplatData,
) -> crate::error::ViewportResult<()> {
    if data.positions.is_empty() {
        return Err(crate::error::ViewportError::InvalidGaussianSplatData {
            reason: "empty splat list",
        });
    }
    let n = data.positions.len();
    if data.scales.len() != n || data.rotations.len() != n || data.opacities.len() != n {
        return Err(crate::error::ViewportError::InvalidGaussianSplatData {
            reason: "mismatched buffer lengths",
        });
    }
    Ok(())
}

/// Build the persistent GPU buffers for a splat set and assemble the
/// `GaussianSplatGpuSet`. Assumes `data` already passed
/// [`validate_gaussian_splat_data`]. Shared by the sync upload, the async
/// worker, and the replace path so all three produce identical resources.
pub(super) fn build_gaussian_splat_set(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    data: &GaussianSplatData,
) -> GaussianSplatGpuSet {
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
    let buf_size_sh = (data.sh_coefficients.len() * 4).max(4) as u64;

    let position_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_position_buf"),
        size: buf_size_pos,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&position_buf, 0, bytemuck::cast_slice(&pos_data));

    let scale_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_scale_buf"),
        size: buf_size_scale,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&scale_buf, 0, bytemuck::cast_slice(&scale_data));

    let rotation_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_rotation_buf"),
        size: buf_size_rot,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&rotation_buf, 0, bytemuck::cast_slice(&rotation_data));

    let opacity_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_opacity_buf"),
        size: buf_size_opa,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&opacity_buf, 0, bytemuck::cast_slice(&data.opacities));

    let sh_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("splat_sh_buf"),
        size: buf_size_sh,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.sh_coefficients.is_empty() {
        queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&data.sh_coefficients));
    }

    GaussianSplatGpuSet {
        position_buf,
        scale_buf,
        rotation_buf,
        opacity_buf,
        sh_buf,
        sh_degree: data.sh_degree,
        count,
        cpu_positions: std::sync::Arc::new(data.positions.clone()),
        cpu_scales: std::sync::Arc::new(data.scales.clone()),
    }
}

/// Persistent GPU state for one uploaded Gaussian splat set.
pub(crate) struct GaussianSplatGpuSet {
    /// Positions as vec4<f32> (w=1), one per splat.
    pub position_buf: crate::gpu::Buffer,
    /// Scales as vec4<f32> (w=0), one per splat.
    pub scale_buf: crate::gpu::Buffer,
    /// Rotations as vec4<f32> [x,y,z,w], one per splat.
    pub rotation_buf: crate::gpu::Buffer,
    /// Opacities as f32, one per splat.
    pub opacity_buf: crate::gpu::Buffer,
    /// SH coefficients as f32, count = splat_count * sh_degree.coeff_count().
    pub sh_buf: crate::gpu::Buffer,
    /// SH degree for this set.
    pub sh_degree: ShDegree,
    /// Number of splats.
    pub count: u32,
    /// CPU positions kept for picking and the wireframe overlay
    /// (object-space). Shared so per-frame consumers snapshot without
    /// copying the set.
    pub cpu_positions: std::sync::Arc<Vec<[f32; 3]>>,
    /// CPU scales kept for picking and the wireframe overlay.
    pub cpu_scales: std::sync::Arc<Vec<[f32; 3]>>,
}

impl crate::resources::handle::GpuByteSize for GaussianSplatGpuSet {
    /// Resident GPU bytes for the persistent source buffers (position, scale,
    /// rotation, opacity, SH). Per-viewport sort scratch is derived and grows
    /// lazily, so it is not counted here.
    fn gpu_bytes(&self) -> u64 {
        self.position_buf.size()
            + self.scale_buf.size()
            + self.rotation_buf.size()
            + self.opacity_buf.size()
            + self.sh_buf.size()
    }
}

/// Slotted store for Gaussian splat sets.
///
/// A removed set leaves an empty slot that a later insert reuses. Each slot
/// carries a generation bumped on removal, and a [`GaussianSplatId`] captures
/// the generation it was issued against, so a stale handle resolves to `None`
/// rather than aliasing the set now in its slot. An entry's byte charge is its
/// [`GpuByteSize::gpu_bytes`](crate::resources::handle::GpuByteSize::gpu_bytes),
/// and its revision is what the item type keys its per-viewport sort scratch on.
pub(crate) type GaussianSplatStore =
    crate::resources::handle::SlotStore<GaussianSplatGpuSet, GaussianSplatId>;
