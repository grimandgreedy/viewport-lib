//! State for the non-instanced (per-object) mesh draw path: one `ObjectUniform`
//! buffer and bind group per scene item, plus the wireframe and
//! transparent-volume-mesh wireframe draw resources.

use crate::resources::mesh::mesh_store::MeshId;
use std::collections::HashMap;

/// Stable key for a cached per-object draw resource.
///
/// The key is the item's `pick_id` plus an `occurrence` counter that
/// distinguishes items sharing the same `pick_id` within one frame. A single
/// pickable object keys on `(pick_id, 0)`, so the cache survives the host
/// reordering or rebuilding the item list each frame: a static object keeps its
/// uniform buffer and bind group instead of rebuilding every frame.
///
/// `occurrence` is what makes the key correct when several render items share a
/// `pick_id`: every non-pickable item defaults to `PickId::NONE`, and one
/// logical object can be drawn by more than one item (for example a volume mesh
/// submitted both as a coloured boundary surface and as a pickable cell mesh).
/// Without `occurrence` those items would collide on one cache entry, share a
/// uniform buffer, and all draw with the last item's transform and material.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PerObjectKey {
    /// The item's `pick_id` (0 == `PickId::NONE`).
    pub(crate) pick_id: u64,
    /// How many earlier items this frame shared the same `pick_id`.
    pub(crate) occurrence: u32,
}

/// One cached per-object draw resource.
pub(crate) struct PerObjectCacheEntry {
    /// This object's own uniform buffer (rewritten each frame; its transform changes).
    pub(crate) uniform_buf: crate::gpu::Buffer,
    /// Bind group pairing the uniform with the object's mesh textures.
    /// `None` until first built.
    pub(crate) bind_group: Option<crate::gpu::BindGroup>,
    /// Material/texture fingerprint the bind group was built from.
    pub(crate) cache_key: u64,
    /// Last `ObjectUniform` written to `uniform_buf`, used to skip the uniform
    /// write when the object is unchanged. `None` until first written.
    pub(crate) last_uniform: Option<crate::resources::ObjectUniform>,
    /// Frame index this entry was last used, for pruning evicted objects.
    pub(crate) last_frame: u64,
}

/// A pre-recorded render bundle covering the opaque per-object draws for the
/// current item set, replayed by the paint path instead of re-recording one
/// draw per item every frame. Built in `prepare()` when the per-object item
/// set is stable and eligible (plain solid meshes: no wireframe, attribute,
/// warp, or deform features); per-item transforms and colours still update
/// every frame through the uniform buffers the bundle's bind groups reference.
pub(crate) struct PerObjectBundle {
    /// The recorded opaque draws.
    pub(crate) bundle: crate::gpu::RenderBundle,
    /// Hash of the item facts the recording depends on (order, mesh ids,
    /// pipeline selection, pick ids, opacity split). A mismatch re-records.
    pub(crate) key: u64,
    /// Whether the bundle was recorded for the HDR scene pass (Rgba16Float,
    /// HDR pipelines) or the LDR path (target format, LDR pipelines). A
    /// bundle only replays in the pass family it was recorded for; `key`
    /// also hashes this, so switching paths re-records.
    pub(crate) hdr: bool,
    /// The camera bind group recorded into the bundle. Paint only replays the
    /// bundle when its pass camera bind group is this exact resource, so a
    /// different viewport (or a rebuilt slot) falls back to immediate draws.
    pub(crate) camera_bg: crate::gpu::BindGroup,
    /// Indices (into the prepared item list) of blended items, which are
    /// excluded from the bundle and drawn immediately, depth-sorted.
    pub(crate) transparent: Vec<usize>,
}

/// Hysteresis for the per-object bundle under item-set churn.
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
    /// Frames since the plan key or a per-item bind group last changed.
    pub(crate) frames_since_change: u32,
    /// While set, the bundle is not recorded and draws stay immediate.
    pub(crate) suppressed: bool,
}

impl Default for BundleChurnGate {
    fn default() -> Self {
        Self {
            last_key: None,
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
    /// Last `ObjectUniform` written, used to skip unchanged writes.
    pub(crate) last_uniform: Option<crate::resources::ObjectUniform>,
}

pub(crate) struct PerObjectState {
    /// Per-object draw resources keyed by [`PerObjectKey`].
    pub(crate) cache: HashMap<PerObjectKey, PerObjectCacheEntry>,
    /// `DeviceResources::resource_free_epoch` as of the last prepare. When the
    /// epoch moves (a texture or mesh was freed), stale cache entries are
    /// purged so their bind groups stop pinning the freed resource's memory.
    pub(crate) free_epoch: u64,
    /// Per-item bind groups for the current frame, indexed by the item's position
    /// in the frame's item list. Populated from `cache` each frame (cheap
    /// reference-counted clones) so the render path can index by item slot.
    pub(crate) bind_groups: Vec<Option<crate::gpu::BindGroup>>,
    /// Per-item uniform buffers used in wireframe mode.
    pub(crate) wireframe_uniform_bufs: Vec<crate::gpu::Buffer>,
    /// Per-item bind groups pairing wireframe uniforms with fallback textures.
    pub(crate) wireframe_bind_groups: Vec<crate::gpu::BindGroup>,
    /// TransparentVolumeMesh boundary wireframe mesh IDs to draw.
    pub(crate) tvm_wireframe_draws: Vec<MeshId>,
    /// Shared wireframe uniform (identity matrix, wireframe = 1) for TVM draws.
    pub(crate) tvm_wireframe_buf: Option<crate::gpu::Buffer>,
    /// Bind group for the TVM wireframe draws.
    pub(crate) tvm_wireframe_bg: Option<crate::gpu::BindGroup>,
}

impl PerObjectState {
    pub(crate) fn new() -> Self {
        Self {
            cache: HashMap::new(),
            free_epoch: 0,
            bind_groups: Vec::new(),
            wireframe_uniform_bufs: Vec::new(),
            wireframe_bind_groups: Vec::new(),
            tvm_wireframe_draws: Vec::new(),
            tvm_wireframe_buf: None,
            tvm_wireframe_bg: None,
        }
    }
}
