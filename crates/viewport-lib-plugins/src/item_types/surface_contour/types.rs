//! The surface contour submission types: the item and where its lines fall.

use viewport_lib_types::colour::Colour;
use viewport_lib_types::ids::MeshId;
use viewport_lib_types::material::ItemSettings;

/// Most levels a [`ContourLevels::Values`] list carries. The rest of a longer
/// list are dropped without a warning; [`ContourLevels::Spaced`] has no limit.
pub const MAX_CONTOUR_LEVELS: usize = 32;

/// Where the contour lines of a [`SurfaceContourItem`] fall.
#[derive(Debug, Clone, PartialEq)]
pub enum ContourLevels {
    /// A line at each of these values. Only the first [`MAX_CONTOUR_LEVELS`]
    /// are drawn; split a longer list over several items.
    Values(Vec<f32>),
    /// A line at `origin + k * interval` for every integer `k`.
    Spaced { origin: f32, interval: f32 },
}

/// Contour lines of a scalar field on one mesh.
///
/// The item draws only the lines. Submit it beside the `SceneRenderItem` that
/// draws the same mesh, with the same `model`: the lines are depth-tested
/// against that surface, so they show where it is the visible one. The field
/// that is contoured need not be the one that colours the surface.
///
/// Lines are drawn where the field is changing. Where it is flat, a plateau
/// sitting on a level draws no line, as an extracted contour would not.
///
/// Limits: a mesh deformed or displaced on the GPU is contoured in its
/// undeformed shape; a transparent surface writes no depth, so the lines on
/// the far side of a closed transparent mesh show through.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SurfaceContourItem {
    /// The mesh the field lies on.
    pub mesh_id: MeshId,
    /// Local to world transform.
    pub model: [[f32; 4]; 4],
    /// Name of a per-vertex scalar attribute the mesh was uploaded with:
    /// `AttributeData::Vertex`, or `Cell` or `Edge`, which are averaged to the
    /// vertices. An item naming an attribute the mesh lacks draws nothing.
    pub scalar_attribute: String,
    /// Where the lines fall.
    pub levels: ContourLevels,
    /// Line colour. Its alpha scales the line's coverage. Default: black.
    pub colour: Colour,
    /// Line width in logical pixels. Default: 1.5.
    pub width: f32,
    /// Shared per-item settings. Only `hidden` is read.
    pub settings: ItemSettings,
}

impl SurfaceContourItem {
    /// Black lines along `scalar_attribute` on `mesh_id`, at `levels`.
    pub fn new(
        mesh_id: MeshId,
        model: [[f32; 4]; 4],
        scalar_attribute: impl Into<String>,
        levels: ContourLevels,
    ) -> Self {
        Self {
            mesh_id,
            model,
            scalar_attribute: scalar_attribute.into(),
            levels,
            colour: Colour::linear(0.0, 0.0, 0.0, 1.0),
            width: 1.5,
            settings: ItemSettings::default(),
        }
    }
}
