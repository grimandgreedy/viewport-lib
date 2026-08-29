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
    /// True when `multi_draw_indexed_indirect` runs natively (signalled by
    /// `MULTI_DRAW_INDIRECT_COUNT`, present on Vulkan/DX12). On these backends
    /// the per-batch indirect draw loop collapses a run of batches that share
    /// pipeline, bind group, and slab chunk into a single multi-draw. Metal
    /// lacks the feature and keeps the per-batch loop over the same contiguous
    /// args buffer.
    pub(crate) multi_draw_supported: bool,
    /// Diagnostic override that forces the multi-draw collapse on even where the
    /// backend emulates `multi_draw_indexed_indirect` as a per-entry loop (Metal).
    /// The emulated result is identical, so this lets the collapse path (run
    /// forming, the actual multi-draw call) be exercised and pixel-compared on
    /// the correctness box. Off by default; set via `set_force_multi_draw`.
    pub(crate) multi_draw_forced: bool,
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
    /// World-space per-instance AABBs from the last rebuild, parallel to the
    /// instance buffer. Used by the CPU per-cascade shadow cull on devices
    /// without GPU-driven culling.
    pub(crate) cached_aabbs: Vec<crate::resources::InstanceAabb>,
    /// When true, the next cache-miss forces a full buffer upload instead of the
    /// per-batch partial-upload path. Set by `force_dirty()` and consumed once.
    pub(crate) force_full_upload: bool,
    /// CPU-readable copy of the indirect draw args, one frame behind.
    pub(crate) indirect_readback_buf: Option<crate::gpu::Buffer>,
    /// Number of batches whose data was copied into `indirect_readback_buf` last frame.
    pub(crate) indirect_readback_batch_count: u32,
    /// True when `indirect_readback_buf` holds resolved data that has not yet
    /// been mapped for readback.
    pub(crate) indirect_readback_pending: bool,
    /// True when a map of `indirect_readback_buf` is in flight. The cull pass
    /// skips the copy while this is set so the buffer is not overwritten before
    /// `prepare()` has read it.
    pub(crate) indirect_map_inflight: bool,
    /// In-flight indirect map status: 0 = pending, 1 = mapped, 2 = failed.
    pub(crate) indirect_map_status: std::sync::Arc<std::sync::atomic::AtomicU8>,
    /// Bumped whenever the shared instance storage buffer is rebuilt (a full
    /// batch upload). Per-viewport cull bind groups compare against this to
    /// detect when their binding-0 reference is stale.
    pub(crate) instance_gen: u64,
    /// Bumped whenever a batch rebuild produces a batch list that differs from
    /// the previous one in any field (mesh, textures, offsets, flags). The
    /// cached shadow render bundles key on this: a stable batch list means the
    /// recorded draw sequence is still valid.
    pub(crate) batches_gen: u64,
    /// GPU cull outputs for the directional shadow cascades. Shadows are fit to
    /// the primary camera and rendered once, so this is scene-scoped rather than
    /// per-viewport.
    pub(crate) shadow_cull: crate::resources::ShadowCullState,
}

impl InstancingState {
    /// Whether the indirect draw paths should collapse batch runs into
    /// `multi_draw_indexed_indirect`: true when the backend runs it natively or
    /// the diagnostic override is on (emulated, for the correctness box).
    pub(crate) fn multi_draw_active(&self) -> bool {
        self.multi_draw_supported || self.multi_draw_forced
    }

    pub(crate) fn new(gpu_culling_supported: bool, multi_draw_supported: bool) -> Self {
        Self {
            batches: Vec::new(),
            use_instancing: false,
            gpu_culling_supported,
            gpu_culling_enabled: gpu_culling_supported,
            multi_draw_supported,
            multi_draw_forced: false,
            cull_resources: None,
            last_scene_generation: u64::MAX,
            last_selection_generation: u64::MAX,
            last_scene_items_count: usize::MAX,
            last_instancable_count: usize::MAX,
            cached_instance_count: 0,
            cached_instance_hashes: Vec::new(),
            cached_batches: Vec::new(),
            cached_aabbs: Vec::new(),
            force_full_upload: false,
            indirect_readback_buf: None,
            indirect_readback_batch_count: 0,
            indirect_readback_pending: false,
            indirect_map_inflight: false,
            indirect_map_status: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
            instance_gen: 0,
            batches_gen: 0,
            shadow_cull: crate::resources::ShadowCullState::new(),
        }
    }
}
