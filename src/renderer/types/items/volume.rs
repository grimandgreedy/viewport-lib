use crate::resources::ColourmapId;
use crate::scene::material::ItemSettings;

/// A volume item to render via GPU ray-marching.
///
/// The caller uploads a 3D scalar field via [`ViewportGpuResources::upload_volume`](crate::resources::ViewportGpuResources::upload_volume) and
/// receives a [`VolumeId`](crate::resources::VolumeId). Each frame, submit a `VolumeItem` referencing that id plus
/// transfer function and display parameters.
///
/// # Picking
///
/// Set `pick_id` to a non-zero value and provide `volume_data` to enable voxel-level
/// picking via `renderer.pick()` and `renderer.pick_rect()`. `pick_id` is the ID
/// returned in the `PickHit` when a voxel is hit. A `pick_id` of zero means the volume
/// is not pickable. `volume_data` must match the data that was passed to
/// `upload_volume` for this `volume_id`.
#[derive(Clone)]
#[non_exhaustive]
pub struct VolumeItem {
    /// Reference to a previously uploaded 3D texture.
    pub volume_id: crate::resources::VolumeId,
    /// CPU scalar data for voxel picking.
    ///
    /// Must match the data passed to `upload_volume` for `volume_id`.
    /// `None` disables voxel-level picking regardless of `settings.pick_id`.
    pub volume_data: Option<std::sync::Arc<crate::geometry::marching_cubes::VolumeData>>,
    /// Colour transfer function LUT. `None` = use default builtin (viridis).
    pub colour_lut: Option<ColourmapId>,
    /// Opacity transfer function LUT. `None` = linear ramp (0 at min, 1 at max).
    pub opacity_lut: Option<ColourmapId>,
    /// Scalar range for normalization [min, max].
    pub scalar_range: (f32, f32),
    /// World-space bounding box minimum corner.
    pub bbox_min: [f32; 3],
    /// World-space bounding box maximum corner.
    pub bbox_max: [f32; 3],
    /// Ray step multiplier. Lower = higher quality, slower. Default: 1.0.
    pub step_scale: f32,
    /// World-space transform. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Whether to apply gradient-based Phong shading. Default: false.
    pub enable_shading: bool,
    /// Global opacity multiplier. Default: 1.0.
    pub opacity_scale: f32,
    /// Scalar threshold range [min, max]. Samples outside this range are discarded (opacity = 0).
    /// Default: same as scalar_range (no clipping).
    pub threshold_min: f32,
    /// Upper scalar threshold. Samples above this value are discarded.
    /// Default: same as scalar_range.1 (no clipping).
    pub threshold_max: f32,
    /// Colour and opacity to use for NaN scalar samples. `None` = skip NaN samples entirely
    /// (same as current behaviour: discard). `Some([r, g, b, a])` = render NaN voxels with
    /// this fixed RGBA colour instead of sampling the transfer function.
    pub nan_colour: Option<[f32; 4]>,
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for VolumeItem {
    fn default() -> Self {
        Self {
            volume_id: crate::resources::VolumeId(0),
            volume_data: None,
            colour_lut: None,
            opacity_lut: None,
            scalar_range: (0.0, 1.0),
            bbox_min: [0.0, 0.0, 0.0],
            bbox_max: [1.0, 1.0, 1.0],
            step_scale: 1.0,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            enable_shading: false,
            opacity_scale: 1.0,
            threshold_min: 0.0,
            threshold_max: 1.0,
            nan_colour: None,
            settings: ItemSettings::default(),
        }
    }
}
