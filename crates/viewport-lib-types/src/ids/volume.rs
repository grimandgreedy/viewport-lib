//! Handles for uploaded 3D volumes and projected-tetrahedra meshes.

crate::slot_handle! {
    /// Identifies a 3D volume texture uploaded to the GPU.
    ///
    /// A slotted, generational handle: a volume can be replaced in place
    /// (`replace_volume`) or freed (`free_volume`) so a time-series playback
    /// reuses one slot instead of leaking a 3D texture per timestep. A handle
    /// held across a free-then-reupload resolves to nothing rather than
    /// aliasing the new occupant.
    pub struct VolumeId;
}

crate::slot_handle! {
    /// Identifies a projected-tetrahedra mesh uploaded to the GPU for transparent
    /// volume rendering.
    ///
    /// A slotted, generational handle: the mesh can be refreshed in place
    /// (`replace_projected_tet`, `replace_projected_tet_scalar`) or freed
    /// (`free_projected_tet`) so a transparent time-series reuses one slot
    /// instead of leaking a tet buffer per fresh upload.
    pub struct ProjectedTetId;
}
