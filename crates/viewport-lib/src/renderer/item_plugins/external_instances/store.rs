//! The instance sets this item type holds on the consumer's behalf, and the
//! per-frame draw data every external-instances draw is built from.
//!
//! A set is a consumer-owned `wgpu::Buffer` of tightly packed `[x, y, z]` `f32`
//! triples, registered once and read in place every frame it is submitted. The
//! renderer never writes it and never copies it, so what the consumer's compute
//! passes last wrote is what renders.
//!
//! The group-1 bind group layout lives here rather than with the pipeline,
//! because a set's draw data is built against it during `prepare`, before the
//! pass that binds the pipeline has begun.

use crate::gpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

pub(crate) use super::types::ExternalInstanceSetId;

/// Build the external-instances group-1 bind group layout: the per-item
/// uniform plus the consumer's positions buffer.
pub(super) fn build_bgl(device: &crate::gpu::Device) -> crate::gpu::BindGroupLayout {
    device.create_bind_group_layout(&crate::gpu::BindGroupLayoutDescriptor {
        label: Some("external_instances_bgl"),
        entries: &[
            crate::resources::builders::uniform_entry(
                0,
                crate::gpu::ShaderStages::VERTEX | crate::gpu::ShaderStages::FRAGMENT,
            ),
            crate::gpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: crate::gpu::ShaderStages::VERTEX,
                ty: crate::gpu::BindingType::Buffer {
                    ty: crate::gpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Persistent renderer-side state for one external instance set.
pub(crate) struct ExternalInstanceSet {
    pub(crate) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// The consumer's buffer. The clone shares the underlying allocation, so
    /// the renderer keeps it alive even if the consumer drops its handle.
    pub(crate) positions: crate::gpu::Buffer,
}

/// Per-item uniform for the external-instances draw (96 bytes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct ExternalInstancesUniform {
    pub(super) model: [[f32; 4]; 4], // 64 bytes
    pub(super) colour: [f32; 4],     // 16 bytes
    pub(super) scale: f32,           //  4 bytes
    pub(super) _pad: [f32; 3],       // 12 bytes
}

/// Per-frame draw data for one submitted `ExternalInstancesItem`.
pub(super) struct ExternalInstancesGpuData {
    pub(super) mesh_id: crate::resources::mesh::mesh_store::MeshId,
    /// Kept alive for the frame; referenced by `bind_group`.
    pub(super) _uniform_buf: crate::gpu::Buffer,
    pub(super) bind_group: crate::gpu::BindGroup,
    /// Draw instance range into the positions buffer, already clamped to the
    /// buffer's element count.
    pub(super) first_instance: u32,
    pub(super) instance_count: u32,
}

/// Build one submitted item's uniform and bind group against a registered set.
///
/// Returns `None` when the requested window is empty after clamping, which is
/// how a stale instance count after a pool shrink is kept from reading past the
/// end of the consumer's buffer.
pub(super) fn build_draw_data(
    device: &crate::gpu::Device,
    bgl: &crate::gpu::BindGroupLayout,
    set: &ExternalInstanceSet,
    item: &crate::renderer::ExternalInstancesItem,
) -> Option<ExternalInstancesGpuData> {
    // Clamp the requested window to the buffer's whole elements so a stale
    // count after a pool shrink cannot read past the end.
    let buffer_elements = (set.positions.size() / 12) as u32;
    let first = item.first_instance.min(buffer_elements);
    let count = item.instance_count.min(buffer_elements - first);
    if count == 0 {
        return None;
    }
    let uniform = ExternalInstancesUniform {
        model: item.model,
        colour: item.colour.to_linear_rgba(),
        scale: item.scale,
        _pad: [0.0; 3],
    };
    let uniform_buf = device.create_buffer_init(&crate::gpu::util::BufferInitDescriptor {
        label: Some("external_instances_uniform_buf"),
        contents: bytemuck::bytes_of(&uniform),
        usage: crate::gpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
        label: Some("external_instances_bg"),
        layout: bgl,
        entries: &[
            crate::gpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            crate::gpu::BindGroupEntry {
                binding: 1,
                resource: set.positions.as_entire_binding(),
            },
        ],
    });
    Some(ExternalInstancesGpuData {
        mesh_id: set.mesh_id,
        _uniform_buf: uniform_buf,
        bind_group,
        first_instance: first,
        instance_count: count,
    })
}

/// Slotted store of registered instance sets.
///
/// A set removed by `drop_external_instance_set` leaves an empty slot that a
/// later create reuses. Each slot carries a generation bumped on removal, so a
/// stale [`ExternalInstanceSetId`] resolves to nothing rather than aliasing the
/// set now in its slot.
pub(super) type ExternalInstanceSetStore =
    crate::resources::handle::SlotStore<ExternalInstanceSet, ExternalInstanceSetId>;

impl crate::resources::handle::GpuByteSize for ExternalInstanceSet {
    /// Zero: the positions buffer belongs to the consumer, who allocated it and
    /// is already accounting for it. The renderer holds a clone of the handle
    /// to keep it alive, not a copy of the data, so counting it here would
    /// report the same allocation twice.
    fn gpu_bytes(&self) -> u64 {
        0
    }
}
