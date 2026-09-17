//! The axis-aligned volume slice item type and the axis it slices on.

use crate::scene::material::ItemSettings;

/// Axis for an axis-aligned image slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceAxis {
    /// Slice perpendicular to the X axis (YZ plane).
    X,
    /// Slice perpendicular to the Y axis (XZ plane).
    #[default]
    Y,
    /// Slice perpendicular to the Z axis (XY plane).
    Z,
}

/// A 2D image slice item: renders one axis-aligned cross-section of an uploaded volume
/// as a flat coloured quad.
///
/// Faster and simpler than full volume ray-marching. Use it to inspect individual
/// slices of a structured grid without the depth ambiguity of ray-marching.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ImageSliceItem {
    /// Reference to a previously uploaded 3D volume texture.
    pub volume_id: crate::resources::VolumeId,
    /// Axis perpendicular to the slice plane. Default: `SliceAxis::Z`.
    pub axis: SliceAxis,
    /// Normalized position along the axis in `[0, 1]`. Default: `0.5`.
    pub offset: f32,
    /// World-space bounding box minimum corner of the volume.
    pub bbox_min: [f32; 3],
    /// World-space bounding box maximum corner of the volume.
    pub bbox_max: [f32; 3],
    /// Scalar range for colourmap mapping `[min, max]`. Default: `(0.0, 1.0)`.
    pub scalar_range: (f32, f32),
    /// Colour LUT. `None` = default builtin (viridis).
    pub colour_lut: Option<crate::resources::ColourmapId>,
    /// Overall opacity of the slice quad. Default: `1.0`.
    pub opacity: f32,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for ImageSliceItem {
    fn default() -> Self {
        Self {
            volume_id: crate::resources::VolumeId::INVALID,
            axis: SliceAxis::Z,
            offset: 0.5,
            bbox_min: [0.0, 0.0, 0.0],
            bbox_max: [1.0, 1.0, 1.0],
            scalar_range: (0.0, 1.0),
            colour_lut: None,
            opacity: 1.0,
            settings: ItemSettings::default(),
        }
    }
}
