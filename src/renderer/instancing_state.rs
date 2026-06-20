//! State for the instanced (GPU-driven) mesh draw path: the per-frame batch
//! list, the cache used to skip unchanged uploads, the GPU culling pipelines,
//! and the indirect-draw readback buffers.

use super::indirect;
use super::types::InstancedBatch;

pub(crate) struct InstancingState {
    /// Instanced batches prepared for the current frame. Empty when using the
    /// per-object path.
    pub(crate) batches: Vec<InstancedBatch>,
    /// Whether the current frame uses the instanced draw path.
    pub(crate) use_instancing: bool,
    /// True when the device supports `INDIRECT_FIRST_INSTANCE`.
    pub(crate) gpu_culling_supported: bool,
    /// True when GPU-driven culling is active (supported and not disabled by the caller).
    pub(crate) gpu_culling_enabled: bool,
    /// GPU culling compute pipelines and frustum buffer. Created lazily on the first
    /// frame where `gpu_culling_enabled` is true and instance buffers are present.
    pub(crate) cull_resources: Option<indirect::CullResources>,
    /// Last scene generation seen during prepare(). u64::MAX forces rebuild on first frame.
    pub(crate) last_scene_generation: u64,
    /// Last selection generation seen during prepare(). u64::MAX forces rebuild on first frame.
    pub(crate) last_selection_generation: u64,
    /// Last scene_items count seen during prepare(). usize::MAX forces rebuild on first frame.
    /// Included in the cache key so frustum-culling changes (different visible set, different
    /// count) invalidate the instance buffer even when scene_generation is stable.
    pub(crate) last_scene_items_count: usize,
    /// Count of items that passed the instanced-path filter on the last rebuild.
    /// Used in place of has_per_frame_mutations so scenes that mix instanced and
    /// non-instanced items still hit the instanced batch cache on frames where the
    /// filtered set is unchanged.
    pub(crate) last_instancable_count: usize,
    /// Total instance count from the last rebuild. Fast length check in
    /// `structure_preserved` and `instance_count` for GPU cull dispatches.
    pub(crate) cached_instance_count: usize,
    /// Per-batch content hash from the last rebuild, indexed by batch position.
    /// A hash mismatch triggers a `write_buffer` for that batch; a match skips it.
    pub(crate) cached_instance_hashes: Vec<u64>,
    /// Cached instanced batch descriptors from the last rebuild.
    pub(crate) cached_batches: Vec<InstancedBatch>,
    /// When true, the next cache-miss forces a full buffer upload instead of the
    /// per-batch partial-upload path. Set by `force_dirty()` and consumed once.
    pub(crate) force_full_upload: bool,
    /// CPU-readable copy of the indirect draw args, one frame behind.
    pub(crate) indirect_readback_buf: Option<wgpu::Buffer>,
    /// Number of batches whose data was copied into `indirect_readback_buf` last frame.
    pub(crate) indirect_readback_batch_count: u32,
    /// True when `indirect_readback_buf` holds unread data from the previous cull pass.
    pub(crate) indirect_readback_pending: bool,
}

impl InstancingState {
    pub(crate) fn new(gpu_culling_supported: bool) -> Self {
        Self {
            batches: Vec::new(),
            use_instancing: false,
            gpu_culling_supported,
            gpu_culling_enabled: gpu_culling_supported,
            cull_resources: None,
            last_scene_generation: u64::MAX,
            last_selection_generation: u64::MAX,
            last_scene_items_count: usize::MAX,
            last_instancable_count: usize::MAX,
            cached_instance_count: 0,
            cached_instance_hashes: Vec::new(),
            cached_batches: Vec::new(),
            force_full_upload: false,
            indirect_readback_buf: None,
            indirect_readback_batch_count: 0,
            indirect_readback_pending: false,
        }
    }
}
