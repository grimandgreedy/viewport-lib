//! Ground-plane configuration.

/// Ground plane rendering mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GroundPlaneMode {
    /// No ground plane rendered (default, zero overhead).
    #[default]
    None,
    /// Invisible plane that receives and displays shadows only.
    ShadowOnly,
    /// Procedural checkerboard tile pattern.
    Tile,
    /// Flat solid colour.
    SolidColour,
}

/// Ground plane configuration for the viewport.
///
/// Renders a large horizontal plane at a configurable world-space Z height.
/// Provides spatial grounding without explicit scene geometry.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GroundPlane {
    /// Rendering mode. Default: `None` (plane not drawn).
    pub mode: GroundPlaneMode,
    /// World-space Z coordinate of the ground plane. Default: `0.0`.
    pub height: f32,
    /// Primary colour for `Tile` and `SolidColour` modes. Default: `[1.0, 1.0, 1.0, 1.0]`.
    pub colour: [f32; 4],
    /// Secondary tile colour for `Tile` mode. Default: `[0.0, 0.0, 0.0, 1.0]`.
    pub tile_colour2: [f32; 4],
    /// Checker tile size in world units (`Tile` mode). Default: `1.0`.
    pub tile_size: f32,
    /// Shadow tint colour (`ShadowOnly` mode). Default: `[0.0, 0.0, 0.0, 1.0]`.
    pub shadow_colour: [f32; 4],
    /// Maximum shadow opacity (`ShadowOnly` mode). `0.0` = transparent, `1.0` = fully opaque. Default: `0.5`.
    pub shadow_opacity: f32,
}

impl Default for GroundPlane {
    fn default() -> Self {
        Self {
            mode: GroundPlaneMode::None,
            height: 0.0,
            colour: [1.0, 1.0, 1.0, 1.0],
            tile_colour2: [0.0, 0.0, 0.0, 1.0],
            tile_size: 1.0,
            shadow_colour: [0.0, 0.0, 0.0, 1.0],
            shadow_opacity: 0.5,
        }
    }
}
