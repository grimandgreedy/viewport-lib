//! The surface LIC submission types: the item and its advection settings.

use viewport_lib_types::ids::MeshId;
use viewport_lib_types::material::ItemSettings;

/// Advection settings for surface LIC.
///
/// The noise is one independent value per screen pixel, and the kernel reaches
/// `steps * step_size` pixels each way along the flow. Longer kernels give
/// smoother streaks; shorter ones give more contrast for less GPU time.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceLicConfig {
    /// Advection steps taken in each direction from every pixel. Default: 20.
    pub steps: u32,
    /// Distance advanced per step, in screen pixels. Default: 1.5.
    pub step_size: f32,
    /// How strongly the streaks modulate the surface colour. At 0 there is no
    /// effect; at 1 the colour ranges from black to twice as bright. Values up
    /// to 4 raise the contrast further. Default: 1.0.
    pub strength: f32,
}

impl Default for SurfaceLicConfig {
    fn default() -> Self {
        Self {
            steps: 20,
            step_size: 1.5,
            strength: 1.0,
        }
    }
}

/// Flow streaks on one mesh.
///
/// The item draws no surface of its own: it modulates whatever colour is on
/// screen where the mesh is the visible surface. Submit it beside the
/// `SceneRenderItem` that draws the same mesh, with the same `model`.
///
/// `strength` is per item. `steps` and `step_size` are read from the first
/// visible item of the frame, because every flow surface is advected in one
/// pass.
///
/// Only the HDR render path draws it.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SurfaceLicItem {
    /// The mesh the flow lies on.
    pub mesh_id: MeshId,
    /// Local to world transform. Applied to the positions and the vectors.
    pub model: [[f32; 4]; 4],
    /// Name of the `AttributeData::VertexVector` attribute the mesh was
    /// uploaded with. An item naming an attribute the mesh lacks draws nothing.
    pub vector_attribute: String,
    /// Advection settings.
    pub config: SurfaceLicConfig,
    /// Shared per-item settings. Only `hidden` is read.
    pub settings: ItemSettings,
}

impl SurfaceLicItem {
    /// Streaks along `vector_attribute` on `mesh_id`, with the default config.
    pub fn new(mesh_id: MeshId, model: [[f32; 4]; 4], vector_attribute: impl Into<String>) -> Self {
        Self {
            mesh_id,
            model,
            vector_attribute: vector_attribute.into(),
            config: SurfaceLicConfig::default(),
            settings: ItemSettings::default(),
        }
    }
}
