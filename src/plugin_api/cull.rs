//! GPU-driven culling service published to plugins.
//!
//! A plugin that submits its own instanced geometry can run the lib's
//! frustum-cull compute against its instance AABB buffer to get a
//! compacted visibility list and a [`wgpu::DrawIndexedIndirect`] entry
//! suitable for `draw_indexed_indirect`.
//!
//! Single-batch shape: the plugin describes one mesh worth of instances per
//! submission. Multiple meshes => multiple submissions. The compute pass
//! reuses the lib's existing cull pipelines; a small set of internal
//! scratch buffers (1-entry batch meta + 1-entry counter) lives on the
//! cull resources and is overwritten each call.
//!
//! Submit via
//! [`crate::renderer::ViewportRenderer::submit_cull`].

/// Per-instance world-space bounding box for GPU culling.
///
/// Plugin code builds an `array<InstanceAabb>` storage buffer, one entry
/// per drawable instance, and passes it via
/// [`CullSubmission::instance_aabbs`].
///
/// Layout matches the lib's internal cull shader contract (32 bytes per
/// entry). `batch_index` is unused in the single-batch service path: set
/// it to 0. `cast_shadows` is honoured when `shadow_pass = true`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct InstanceAabb {
    /// World-space minimum corner.
    pub min: [f32; 3],
    /// Unused in the single-batch service path. Set to 0.
    pub batch_index: u32,
    /// World-space maximum corner.
    pub max: [f32; 3],
    /// `1` = instance participates in shadow casting; `0` = instance is
    /// skipped during shadow cull dispatches. Ignored when
    /// [`CullSubmission::shadow_pass`] is false.
    pub cast_shadows: u32,
}

const _: () = assert!(std::mem::size_of::<InstanceAabb>() == 32);

/// Inputs to one cull dispatch.
///
/// Buffer requirements:
///
/// - `instance_aabbs`: `STORAGE`. Holds `instance_count` × [`InstanceAabb`].
/// - `visible_out`: `STORAGE`. Holds up to `instance_count` × `u32` indices
///   (compacted; the high entries are left untouched).
/// - `indirect_out`: `STORAGE | INDIRECT`. Receives one
///   `DrawIndexedIndirect` entry (20 bytes).
///
/// The plugin draws with `pass.draw_indexed_indirect(indirect_out, 0)`
/// after binding `visible_out` as the per-instance index buffer.
pub struct CullSubmission<'a> {
    /// Storage buffer of [`InstanceAabb`] entries.
    pub instance_aabbs: &'a wgpu::Buffer,
    /// Number of valid entries in `instance_aabbs`.
    pub instance_count: u32,
    /// Storage buffer that receives the compacted list of visible instance
    /// indices. Sized to at least `instance_count * 4` bytes.
    pub visible_out: &'a wgpu::Buffer,
    /// Storage / indirect buffer that receives the
    /// `DrawIndexedIndirect` entry (`index_count, instance_count,
    /// first_index, base_vertex, first_instance`).
    pub indirect_out: &'a wgpu::Buffer,
    /// Index count for one instance.
    pub index_count: u32,
    /// First index offset (typically 0).
    pub first_index: u32,
    /// Vertex offset (typically 0).
    pub base_vertex: i32,
    /// First-instance value written into the indirect entry (typically 0).
    pub first_instance: u32,
    /// `false` for the main camera frustum cull; `true` for a shadow
    /// cascade cull. In shadow mode, instances with `cast_shadows = 0`
    /// are skipped.
    pub shadow_pass: bool,
}
