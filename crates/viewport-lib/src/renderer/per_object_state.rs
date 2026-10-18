//! State for the mesh batches: their group-1
//! bind groups, the opaque-draw render bundle, the per-viewport foreground
//! items and the transparent-volume-mesh wireframe draw resources.

use crate::resources::mesh::mesh_store::MeshId;

/// One cached group-1 bind group, shared by every batch whose mesh +
/// material resources hash to the same [`crate::resources::DeviceResources::per_item_object_bg_key`].
/// Binding 0 is the shared instance buffer; a draw selects its records by
/// instance range, so batches with the same material reuse this one handle.
pub(crate) struct MaterialBindGroup {
    /// The bind group. Binding 0 references the shared object-data buffer.
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// `InstancingState::instance_gen` this bind group was built against: the
    /// instance buffer it binds. A reallocated buffer leaves it stale.
    pub(crate) data_gen: u64,
    /// Frame index this entry was last used, for capacity-based pruning.
    pub(crate) last_frame: u64,
    /// The mesh and texture ids this bind group samples, kept so the entry can
    /// be validated when something is freed. The map's key is a hash, so the ids
    /// cannot be recovered from it; without them the only safe response to a
    /// free is to clear the whole map.
    pub(crate) deps: crate::resources::resource_deps::ResourceDeps,
}

impl MaterialBindGroup {
    /// Whether every resource this bind group samples is still resident.
    ///
    /// A texture whose pixels were replaced under a live id is NOT detected
    /// here and never can be, which is why the view epoch clears the map
    /// unconditionally.
    pub(crate) fn resources_resolve(&self, resources: &crate::resources::DeviceResources) -> bool {
        self.deps.resolves(resources)
    }
}

/// A pre-recorded render bundle covering the opaque batch draws for the
/// current batch list, replayed by the paint path instead of re-encoding one
/// draw per batch every frame. Built in `prepare()` when the batch list is
/// stable and eligible (no deformers, no plugin batches, enough opaque
/// draws); per-instance transforms and colours still update every frame
/// through the instance buffer the bundle's bind groups reference.
pub(crate) struct PerObjectBundle {
    /// The recorded opaque draws.
    pub(crate) bundle: crate::gpu::RenderBundle,
    /// Hash of the facts the recording depends on (the batch list generation,
    /// the instance buffer generation, pipeline selection). A mismatch
    /// re-records.
    pub(crate) key: u64,
    /// Whether the bundle was recorded for the HDR scene pass (Rgba16Float,
    /// HDR pipelines) or the LDR path (target format, LDR pipelines). A
    /// bundle only replays in the pass family it was recorded for; `key`
    /// also hashes this, so switching paths re-records.
    pub(crate) hdr: bool,
    /// Whether the bundle holds only the batches the GPU cull does not draw.
    /// Recorded that way when GPU-driven submission draws the rest from
    /// indirect args; the pass then replays it after those draws. `key` hashes this too.
    pub(crate) direct_only: bool,
    /// The camera bind group recorded into the bundle. Paint only replays the
    /// bundle when its pass camera bind group is this exact resource, so a
    /// different viewport (or a rebuilt slot) falls back to immediate draws.
    pub(crate) camera_bg: crate::gpu::BindGroup,
}

/// Hysteresis for the batch bundle under batch-list churn.
///
/// Re-recording the bundle every frame is a net loss: record plus replay
/// encodes every draw twice, and dropping a just-recorded bundle each frame
/// leaks it in wgpu 27 (gfx-rs/wgpu#8656), which compounds into unbounded
/// memory growth when the churn also creates fresh bind groups. A single
/// isolated change still re-records immediately (measured hitchless); the
/// gate only engages when the set changes twice in short succession, and
/// re-arms after a stretch of stable frames.
pub(crate) struct BundleChurnGate {
    /// Plan key from the previous prepare.
    pub(crate) last_key: Option<u64>,
    /// Whether the previous plan covered the unculled batches only. A switch
    /// (GPU-driven submission coming up on the second HDR frame) changes
    /// the key without the batch list moving, so it is not counted as churn.
    pub(crate) last_direct_only: Option<bool>,
    /// Frames since the plan key or a per-item bind group last changed.
    pub(crate) frames_since_change: u32,
    /// While set, the bundle is not recorded and draws stay immediate.
    pub(crate) suppressed: bool,
}

impl Default for BundleChurnGate {
    fn default() -> Self {
        Self {
            last_key: None,
            last_direct_only: None,
            // Large so the very first plan key does not read as a second
            // change in quick succession.
            frames_since_change: u32::MAX,
            suppressed: false,
        }
    }
}

/// Per-item draw resources for one viewport's foreground items, indexed by
/// the item's position in `SceneFrame::foreground_items`. Foreground items
/// are keyed positionally rather than through the [`PerObjectKey`] cache:
/// the list is small, per-viewport, and rebuilt from scratch when it shrinks.
pub(crate) struct ForegroundObjectEntry {
    /// This item's own uniform buffer.
    pub(crate) uniform_buf: crate::gpu::Buffer,
    /// Bind group pairing the uniform with the item's mesh textures.
    /// `None` until first built (or while the item's mesh is missing).
    pub(crate) bind_group: Option<crate::gpu::BindGroup>,
    /// Material/texture fingerprint the bind group was built from.
    pub(crate) cache_key: u64,
    /// Last `InstanceData` written, used to skip unchanged writes.
    pub(crate) last_uniform: Option<crate::resources::InstanceData>,
}

pub(crate) struct PerObjectState {
    /// Group-1 bind groups of the batches, deduped by mesh+material
    /// fingerprint (`InstancedBatch::bind_key`). Binding 0 is the shared instance
    /// buffer. Persists across frames; pruned by capacity and on a
    /// resource-free epoch bump.
    pub(crate) material_bind_groups: crate::resources::fast_hash::FastMap<u64, MaterialBindGroup>,
    /// Resource epochs as of the last prepare. A free keeps only the entries
    /// whose deps still resolve; a replace clears the map, since a swapped
    /// view behind a live id cannot be validated per entry.
    pub(crate) deps_gate: crate::resources::resource_deps::ResourceGate,
    /// TransparentVolumeMesh boundary wireframe mesh IDs to draw.
    pub(crate) tvm_wireframe_draws: Vec<MeshId>,
    /// Object records for the TVM draws, one per entry of
    /// `tvm_wireframe_draws`, each carrying its item's model matrix.
    pub(crate) tvm_wireframe_buf: Option<crate::gpu::Buffer>,
    /// How many records `tvm_wireframe_buf` holds.
    pub(crate) tvm_wireframe_capacity: usize,
    /// Bind group for the TVM wireframe draws.
    pub(crate) tvm_wireframe_bg: Option<crate::gpu::BindGroup>,
}

impl PerObjectState {
    pub(crate) fn new() -> Self {
        Self {
            material_bind_groups: Default::default(),
            deps_gate: crate::resources::resource_deps::ResourceGate::default(),
            tvm_wireframe_draws: Vec::new(),
            tvm_wireframe_buf: None,
            tvm_wireframe_capacity: 0,
            tvm_wireframe_bg: None,
        }
    }
}
