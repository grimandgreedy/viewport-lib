//! GPU-driven culling service published to plugins.
//!
//! A plugin that submits its own instanced geometry can run the lib's
//! frustum-cull compute against its instance AABB buffer to get a compacted
//! visibility list and a `DrawIndexedIndirect` entry suitable for
//! `draw_indexed_indirect`.
//!
//! [`CullSubmission`] carries the buffers for a multi-batch cull: one big
//! AABB list, one [`BatchMeta`] entry per batch, one atomic counter slot per
//! batch, and one indirect-draw entry per batch. Plugins with many meshes
//! pack them all into one submission so the renderer dispatches the compute
//! pass once for the whole group.
//!
//! For the common one-mesh-N-instances case, see
//! [`crate::renderer::ViewportRenderer::submit_cull_single_mesh`]. It takes
//! the per-mesh draw parameters directly and fills the lib's scratch meta +
//! counter buffers for you.
//!
//! Submit via [`crate::renderer::ViewportRenderer::submit_cull`] or
//! [`crate::renderer::ViewportRenderer::submit_cull_shadow`].

pub use crate::resources::mesh::instancing::BatchMeta;

/// Per-instance world-space bounding box for GPU culling.
///
/// Plugin code builds an `array<InstanceAabb>` storage buffer, one entry per
/// drawable instance, and passes it via [`CullSubmission::instance_aabbs`].
///
/// Layout matches the cull shader's contract (32 bytes per entry).
/// `batch_index` selects which [`BatchMeta`] entry the instance belongs to;
/// for a single-batch submission set it to 0. `cast_shadows` is honoured
/// when [`CullSubmission::shadow_pass`] is `true` or the submission goes
/// through `submit_cull_shadow`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct InstanceAabb {
    /// World-space minimum corner.
    pub min: [f32; 3],
    /// Index into the batch_meta buffer that this instance belongs to.
    pub batch_index: u32,
    /// World-space maximum corner.
    pub max: [f32; 3],
    /// `1` if the instance participates in shadow casting, `0` to skip it
    /// during shadow cull dispatches. Ignored on non-shadow submissions.
    pub cast_shadows: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceAabb>() == 32);

/// Inputs to one cull dispatch.
///
/// Buffer requirements:
///
/// - `instance_aabbs`: `STORAGE`. `instance_count` x [`InstanceAabb`].
/// - `batch_meta`: `STORAGE`. `batch_count` x [`BatchMeta`].
/// - `counter`: `STORAGE`. `batch_count` x `u32`, used as atomic counters.
///   The renderer zeroes this slot before the dispatch.
/// - `visible_out`: `STORAGE`. Holds up to `instance_count` x `u32`, the
///   compacted list of visible instance indices written by the cull pass.
/// - `indirect_out`: `STORAGE | INDIRECT`. `batch_count` x
///   `DrawIndexedIndirect` (20 bytes per entry). The static fields
///   (`index_count`, `first_index`, `base_vertex`, `first_instance`) come
///   from the matching [`BatchMeta`] entry; `instance_count` is overwritten
///   by the cull pass.
///
/// After the encoder runs, draw each batch with
/// `pass.draw_indexed_indirect(indirect_out, batch_idx * 20)` using
/// `visible_out` as the per-instance lookup buffer.
pub struct CullSubmission<'a> {
    /// Storage buffer of [`InstanceAabb`] entries.
    pub instance_aabbs: &'a crate::gpu::Buffer,
    /// Number of valid entries in `instance_aabbs` (not the buffer capacity).
    pub instance_count: u32,
    /// Storage buffer of [`BatchMeta`] entries, one per batch.
    pub batch_meta: &'a crate::gpu::Buffer,
    /// Number of valid entries in `batch_meta`.
    pub batch_count: u32,
    /// Storage buffer used for per-batch atomic counters. Must hold at
    /// least `batch_count` u32 slots. The cull pass zeroes the counters at
    /// the end of its dispatch so the same buffer can be reused across
    /// cascades or frames.
    pub counter: &'a crate::gpu::Buffer,
    /// Storage buffer that receives the compacted list of visible instance
    /// indices. Sized for at least `instance_count` u32 slots.
    pub visible_out: &'a crate::gpu::Buffer,
    /// Storage / indirect buffer that receives one `DrawIndexedIndirect`
    /// entry per batch (20 bytes each).
    pub indirect_out: &'a crate::gpu::Buffer,
    /// `true` to run the shadow-cull variant on the main camera pass: the
    /// cull shader honours `InstanceAabb::cast_shadows` and skips entries
    /// with the flag clear. `submit_cull_shadow` forces this on regardless
    /// of the value here.
    pub shadow_pass: bool,
}

/// Draw parameters for a one-mesh-N-instances submission.
///
/// Passed to [`crate::renderer::ViewportRenderer::submit_cull_single_mesh`]
/// when a plugin only has one mesh to cull. The renderer fills its scratch
/// `BatchMeta` and counter slots from these fields, so the plugin does not
/// have to allocate either buffer.
#[derive(Copy, Clone, Debug)]
pub struct SingleMeshDraw {
    /// Mesh index count for one instance.
    pub index_count: u32,
    /// First index offset (typically 0).
    pub first_index: u32,
    /// Vertex offset (typically 0).
    pub base_vertex: i32,
    /// First-instance value written into the indirect entry (typically 0).
    pub first_instance: u32,
}
