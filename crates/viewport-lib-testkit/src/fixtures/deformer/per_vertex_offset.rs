//! A deformer whose body reads its per-vertex slot data.

use viewport_lib::resources::DeviceResources;
use viewport_lib::wgpu;
use viewport_lib::{DeformStage, DeformerDesc, MeshId};

/// The descriptor for the per-vertex-offset deformer: a world-space body that
/// moves each vertex along `+X` by a float read from its own slot data.
///
/// Where [`constant_offset_deformer`](super::constant_offset_deformer) proves
/// registration and the params window, this one proves the data path: the
/// body calls `deform_read_f32(ctx.slot, v.vertex_index, 0u)`, so the
/// `(offset, stride)` prefix layout of `deform_data` and the helper that
/// indexes it are both exercised. A test attaches a different value per
/// vertex, so a regression in the addressing shows up as the wrong vertices
/// moving, not just as nothing moving.
pub fn per_vertex_offset_deformer() -> DeformerDesc {
    DeformerDesc {
        name: PerVertexOffsetDeformer::NAME,
        stage: DeformStage::WorldSpace,
        priority: 0,
        wgsl_body: "\
fn deform(v: DeformVertex, ctx: DeformContext) -> DeformVertex {
    var out = v;
    if (deform_slot_stride(ctx.slot) == 0u) {
        return out;
    }
    let dx = deform_read_f32(ctx.slot, v.vertex_index, 0u);
    out.position = v.position + vec3<f32>(dx, 0.0, 0.0);
    return out;
}
"
        .to_string(),
        per_vertex_stride: 4,
    }
}

/// The slot-data side of [`per_vertex_offset_deformer`]: one `f32` per vertex.
pub struct PerVertexOffsetDeformer;

impl PerVertexOffsetDeformer {
    /// The registered name, and the prefix the shader composer applies to the
    /// body's declarations.
    pub const NAME: &'static str = "per_vertex_offset";

    /// Attach one `f32` per vertex at `slot` for `mesh_id`: `offsets[i]` is
    /// how far vertex `i` moves along `+X`.
    pub fn attach(
        resources: &mut DeviceResources,
        device: &wgpu::Device,
        mesh_id: MeshId,
        slot: usize,
        offsets: &[f32],
    ) {
        let mut data = Vec::with_capacity(offsets.len() * 4);
        for offset in offsets {
            data.extend_from_slice(&offset.to_ne_bytes());
        }
        resources.attach_deform_slot(device, mesh_id, slot, 4, &data);
    }
}
