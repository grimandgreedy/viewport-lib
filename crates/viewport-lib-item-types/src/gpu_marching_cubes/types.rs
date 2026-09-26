use viewport_lib::{ItemSettings, Material};

viewport_lib::resources::handle::slot_handle! {
    /// Handle to a volume scalar field uploaded for GPU marching cubes.
    ///
    /// Returned by [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload). Pass to
    /// [`GpuMarchingCubesItem`](viewport_lib::renderer::GpuMarchingCubesItem) to select which volume to triangulate each frame.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. A handle whose volume was removed (its slot freed and reused
    /// by a later upload) resolves to nothing on lookup, so it cannot alias the
    /// volume now in its slot.
    pub struct McVolumeId;
}

/// One GPU marching cubes draw item submitted per frame.
///
/// The volume referenced by `volume_id` is triangulated on the GPU at `isovalue`
/// and drawn with `material`. No CPU readback occurs; the vertex count is
/// determined by an indirect draw call.
#[derive(Clone)]
pub struct GpuMarchingCubesItem {
    /// Volume to triangulate (must remain alive).
    pub volume_id: McVolumeId,
    /// Isovalue at which to extract the surface.
    pub isovalue: f32,
    /// Surface material (colour + roughness).
    pub material: Material,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
    /// CPU-side volume data for `pick()` and `pick_rect()`.
    ///
    /// When set, the CPU picker ray-marches the actual scalar field and detects
    /// isovalue crossings rather than falling back to the volume AABB. `None`
    /// means the item is not reachable by the CPU picking path.
    pub cpu_data: Option<std::sync::Arc<viewport_lib_geometry::marching_cubes::VolumeData>>,
}
