//! State for the non-instanced (per-object) mesh draw path: one `ObjectUniform`
//! buffer and bind group per scene item, plus the wireframe and
//! transparent-volume-mesh wireframe draw resources.

use crate::resources::mesh::mesh_store::MeshId;
use std::collections::HashMap;

/// One cached group-1 bind group, shared by every per-object item whose mesh +
/// material resources hash to the same [`crate::resources::DeviceResources::per_item_object_bg_key`].
/// Binding 0 is the shared object-data storage buffer; each item selects its
/// element with `@builtin(instance_index)`, so items with the same material
/// reuse this one handle and the draw loop stops rebinding group 1 per draw.
pub(crate) struct MaterialBindGroup {
    /// The bind group. Binding 0 references the shared object-data buffer.
    pub(crate) bind_group: crate::gpu::BindGroup,
    /// `object_data_gen` this bind group was built against. When the shared
    /// object-data buffer is reallocated the whole map is cleared, so this is a
    /// belt-and-braces check rather than a per-entry gate.
    pub(crate) data_gen: u64,
    /// Frame index this entry was last used, for capacity-based pruning.
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
    /// Group-1 bind groups deduped by mesh+material fingerprint. Every per-object
    /// draw binds one of these (binding 0 is the shared `object_data_buf`) and
    /// selects its element with an object-data index, so items sharing a material
    /// share one bind group. Persists across frames; pruned by capacity and on a
    /// resource-free epoch bump.
    pub(crate) material_bind_groups: HashMap<u64, MaterialBindGroup>,
    /// Shared storage buffer holding this frame's `array<ObjectUniform>`. Binding
    /// 0 of every per-object group-1 bind group. Grown (reallocated) when the
    /// per-frame object count exceeds its capacity.
    pub(crate) object_data_buf: Option<crate::gpu::Buffer>,
    /// Capacity of `object_data_buf` in `ObjectUniform` elements.
    pub(crate) object_data_capacity: usize,
    /// Bumped whenever `object_data_buf` is reallocated. A change clears
    /// `material_bind_groups` (their binding-0 reference went stale).
    pub(crate) object_data_gen: u64,
    /// This frame's object-data element index for each item's whole-mesh draw,
    /// parallel to `bind_groups`. Meaningful only where `bind_groups[i]` is
    /// `Some`; a `None` slot falls back to the mesh's single-element bind group
    /// and draws at instance 0.
    pub(crate) object_indices: Vec<u32>,
    /// `DeviceResources::resource_free_epoch` as of the last prepare. When the
    /// epoch moves (a texture or mesh was freed), the material bind-group map is
    /// purged so its bind groups stop pinning the freed resource's memory.
    pub(crate) free_epoch: u64,
    /// Per-item bind groups for the current frame, indexed by the item's position
    /// in the frame's item list. Populated from `material_bind_groups` each frame
    /// (cheap reference-counted clones) so the render path can index by item slot.
    pub(crate) bind_groups: Vec<Option<crate::gpu::BindGroup>>,
    /// Per-range bind groups for items drawn with per-submesh materials,
    /// keyed by the item's position in the frame's item list. Only items
    /// whose `submesh_materials` matches their mesh's range count get an
    /// entry; everything else draws through `bind_groups`. Rebuilt each
    /// frame like `bind_groups`.
    pub(crate) submesh_bind_groups: HashMap<usize, Vec<Option<crate::gpu::BindGroup>>>,
    /// Object-data element indices for each submesh range, parallel to
    /// `submesh_bind_groups`. `submesh_indices[&item][r]` is the index range `r`
    /// draws at. Rebuilt each frame alongside `submesh_bind_groups`.
    pub(crate) submesh_indices: HashMap<usize, Vec<u32>>,
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
            material_bind_groups: HashMap::new(),
            object_data_buf: None,
            object_data_capacity: 0,
            object_data_gen: 0,
            object_indices: Vec::new(),
            free_epoch: 0,
            bind_groups: Vec::new(),
            submesh_bind_groups: HashMap::new(),
            submesh_indices: HashMap::new(),
            wireframe_uniform_bufs: Vec::new(),
            wireframe_bind_groups: Vec::new(),
            tvm_wireframe_draws: Vec::new(),
            tvm_wireframe_buf: None,
            tvm_wireframe_bg: None,
        }
    }
}
