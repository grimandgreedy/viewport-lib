//! State for the non-instanced (per-object) mesh draw path: one `ObjectUniform`
//! buffer and bind group per scene item, plus the wireframe and
//! transparent-volume-mesh wireframe draw resources.

use crate::resources::mesh_store::MeshId;
use std::collections::HashMap;

/// One cached per-object draw resource, keyed by the item's stable `pick_id`.
///
/// Keying on `pick_id` rather than the item's index in the frame's item list
/// means the cache survives the host reordering or rebuilding that list each
/// frame: a static object keeps its uniform buffer and bind group, so the bind
/// group is not rebuilt every frame.
pub(crate) struct PerObjectCacheEntry {
    /// This object's own uniform buffer (rewritten each frame; its transform changes).
    pub(crate) uniform_buf: wgpu::Buffer,
    /// Bind group pairing the uniform with the object's mesh textures.
    /// `None` until first built.
    pub(crate) bind_group: Option<wgpu::BindGroup>,
    /// Material/texture fingerprint the bind group was built from.
    pub(crate) cache_key: u64,
    /// Last `ObjectUniform` written to `uniform_buf`, used to skip the uniform
    /// write when the object is unchanged. `None` until first written.
    pub(crate) last_uniform: Option<crate::resources::ObjectUniform>,
    /// Frame index this entry was last used, for pruning evicted objects.
    pub(crate) last_frame: u64,
}

pub(crate) struct PerObjectState {
    /// Per-object draw resources keyed by stable `pick_id`.
    pub(crate) cache: HashMap<u64, PerObjectCacheEntry>,
    /// Per-item bind groups for the current frame, indexed by the item's position
    /// in the frame's item list. Populated from `cache` each frame (cheap
    /// reference-counted clones) so the render path can index by item slot.
    pub(crate) bind_groups: Vec<Option<wgpu::BindGroup>>,
    /// Per-item uniform buffers used in wireframe mode.
    pub(crate) wireframe_uniform_bufs: Vec<wgpu::Buffer>,
    /// Per-item bind groups pairing wireframe uniforms with fallback textures.
    pub(crate) wireframe_bind_groups: Vec<wgpu::BindGroup>,
    /// TransparentVolumeMesh boundary wireframe mesh IDs to draw.
    pub(crate) tvm_wireframe_draws: Vec<MeshId>,
    /// Shared wireframe uniform (identity matrix, wireframe = 1) for TVM draws.
    pub(crate) tvm_wireframe_buf: Option<wgpu::Buffer>,
    /// Bind group for the TVM wireframe draws.
    pub(crate) tvm_wireframe_bg: Option<wgpu::BindGroup>,
}

impl PerObjectState {
    pub(crate) fn new() -> Self {
        Self {
            cache: HashMap::new(),
            bind_groups: Vec::new(),
            wireframe_uniform_bufs: Vec::new(),
            wireframe_bind_groups: Vec::new(),
            tvm_wireframe_draws: Vec::new(),
            tvm_wireframe_buf: None,
            tvm_wireframe_bg: None,
        }
    }
}
