use crate::resources::VolumeGpuId;
use crate::scene::material::{ItemSettings, Material};

/// One GPU marching cubes draw job submitted per frame.
///
/// The volume referenced by `volume_id` is triangulated on the GPU at `isovalue`
/// and drawn with `material`. No CPU readback occurs; the vertex count is
/// determined by an indirect draw call.
pub struct GpuMarchingCubesJob {
    /// Volume to triangulate (must remain alive).
    pub volume_id: VolumeGpuId,
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
    pub cpu_data: Option<std::sync::Arc<crate::geometry::marching_cubes::VolumeData>>,
}
