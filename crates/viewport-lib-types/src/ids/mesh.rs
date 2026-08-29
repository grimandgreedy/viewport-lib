//! Handles for meshes and LOD groups.

crate::slot_handle! {
    /// Handle to a mesh in the store.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. A handle whose generation is stale (its slot was freed and
    /// reused) resolves to `None` on lookup rather than aliasing a different
    /// mesh.
    pub struct MeshId;
}

crate::slot_handle! {
    /// Handle to a LOD group in the renderer's group registry.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. A group freed with `free_lod_group` bumps its slot
    /// generation, so a stale handle resolves to `None` rather than aliasing a
    /// group registered later into the reused slot.
    pub struct LodGroupId;
}
