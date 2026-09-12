//! A deformer whose body moves every vertex by a constant.

use viewport_lib::resources::DeviceResources;
use viewport_lib::wgpu;
use viewport_lib::{DEFORM_PARAMS_PER_SLOT, DeformStage, DeformerDesc, MeshId};

/// The descriptor for the constant-offset deformer: a world-space body that
/// adds `slot_params[slot * 4].xyz` to every vertex position.
///
/// The body skips itself when its slot has no data attached
/// (`deform_slot_stride(ctx.slot) == 0u`), which is the contract's opt-out, so
/// a mesh only moves once [`ConstantOffsetDeformer::attach`] has run for it.
pub fn constant_offset_deformer() -> DeformerDesc {
    DeformerDesc {
        name: ConstantOffsetDeformer::NAME,
        stage: DeformStage::WorldSpace,
        priority: 0,
        wgsl_body: "\
fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    var out = v;
    if (deform_slot_stride(ctx.slot) == 0u) {
        return out;
    }
    let params = deform_header.slot_params[ctx.slot * 4u];
    out.position = v.position + params.xyz;
    return out;
}
"
        .to_string(),
        per_vertex_stride: 4,
    }
}

/// The slot-data side of [`constant_offset_deformer`]: what a host has to do
/// after registering it for a mesh to actually move.
///
/// The body reads nothing per vertex, but a slot with no data attached is
/// gated off, so the attached bytes are one zeroed `u32` per vertex: enough to
/// give the slot a non-zero stride.
pub struct ConstantOffsetDeformer;

impl ConstantOffsetDeformer {
    /// The registered name, and the prefix the shader composer applies to the
    /// body's declarations.
    pub const NAME: &'static str = "constant_offset";

    /// Attach one zeroed `u32` per vertex at `slot` for `mesh_id`, switching
    /// the body on for that mesh.
    pub fn attach(
        resources: &mut DeviceResources,
        device: &wgpu::Device,
        mesh_id: MeshId,
        slot: usize,
        vertex_count: usize,
    ) {
        let data = vec![0u8; vertex_count * 4];
        resources.attach_deform_slot(device, mesh_id, slot, 4, &data);
    }

    /// Write the offset the body adds. Zero leaves the mesh where it was.
    pub fn set_offset(
        resources: &mut DeviceResources,
        queue: &wgpu::Queue,
        slot: usize,
        offset: glam::Vec3,
    ) {
        let mut params = [[0.0f32; 4]; DEFORM_PARAMS_PER_SLOT];
        params[0] = [offset.x, offset.y, offset.z, 0.0];
        resources.set_deform_slot_params(queue, slot, params);
    }
}
