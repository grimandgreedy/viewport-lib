use crate::renderer::types::items::IDENTITY_MAT4;
use crate::resources::ColourmapId;
use crate::scene::material::ItemSettings;

crate::resources::handle::slot_handle! {
    /// Handle to a point cloud uploaded once through
    /// [`ViewportRenderer::upload_point_cloud`](crate::renderer::ViewportRenderer::upload_point_cloud).
    ///
    /// Name it from a [`PointCloudRefItem`] to draw the stored cloud without
    /// resubmitting its points. Carries the slot index plus the generation the
    /// slot had when the handle was issued, so a handle whose cloud was
    /// dropped resolves to nothing rather than aliasing its successor.
    pub struct PointCloudId;
}

/// Render mode for point cloud items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PointRenderMode {
    /// Flat disc: billboard quad clipped to a circle. Fast, no shading.
    #[default]
    ScreenSpaceCircle,
    /// Shaded sphere: billboard quad with hemisphere normal shading (ambient + diffuse + specular).
    /// Points look like small lit spheres without actual geometry cost.
    Sphere,
}

/// A point cloud item to render in the viewport.
#[derive(Clone)]
#[non_exhaustive]
pub struct PointCloudItem {
    /// World-space positions (one vec3 per point).
    pub positions: Vec<[f32; 3]>,
    /// Optional per-point RGBA colours in linear `[0,1]`. If empty, uses `default_colour`.
    pub colours: Vec<crate::Colour>,
    /// Optional per-point scalar values for LUT colouring. If non-empty, overrides `colours`.
    pub scalars: Vec<f32>,
    /// Scalar range for LUT mapping. None = auto from min/max of `scalars`.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. None = use default builtin (viridis).
    pub colourmap_id: Option<ColourmapId>,
    /// Screen-space point size in pixels. Default: 4.0.
    pub point_size: f32,
    /// Fallback colour when neither `colours` nor `scalars` are provided.
    pub default_colour: crate::Colour,
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Render mode. Default: ScreenSpaceCircle.
    pub render_mode: PointRenderMode,
    /// Optional per-point radii in pixels. If non-empty, overrides `point_size` for each point.
    pub radii: Vec<f32>,
    /// Optional per-point opacity values in `[0, 1]`. If non-empty, scales each point's alpha.
    pub transparencies: Vec<f32>,
    /// When true, each point is rendered as a soft Gaussian splat instead of a flat circle.
    /// The alpha falls off as `exp(-3 * d^2)` where `d` is the normalised distance from the
    /// point centre. Default: false.
    pub gaussian: bool,
    /// Optional per-point scalars that drive the splat radius.  If non-empty, these values
    /// are mapped from `radius_scalar_range` (or data min/max when `None`) to `radius_range`
    /// (pixels) and used as per-point radii, overriding `radii` and `point_size`.
    pub radius_scalars: Vec<f32>,
    /// Normalization range for `radius_scalars`.  `None` = auto from data min/max.
    pub radius_scalar_range: Option<(f32, f32)>,
    /// Output pixel-radius range `[min_px, max_px]` for the radius scalar mapping.
    /// Default: `(2.0, 12.0)`.
    pub radius_range: (f32, f32),
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for PointCloudItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            colours: Vec::new(),
            scalars: Vec::new(),
            scalar_range: None,
            colourmap_id: None,
            point_size: 4.0,
            default_colour: [1.0, 1.0, 1.0, 1.0].into(),
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            render_mode: PointRenderMode::ScreenSpaceCircle,
            radii: Vec::new(),
            transparencies: Vec::new(),
            gaussian: false,
            radius_scalars: Vec::new(),
            radius_scalar_range: None,
            radius_range: (2.0, 12.0),
            settings: ItemSettings::default(),
        }
    }
}

/// Per-frame reference to a pre-uploaded point cloud. See [`PolylineRefItem`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PointCloudRefItem {
    /// Handle to GPU buffers produced by
    /// [`ViewportRenderer::upload_point_cloud`](crate::renderer::ViewportRenderer::upload_point_cloud)
    /// or `begin_upload_point_cloud`.
    pub source: crate::resources::PointCloudId,
    /// Per-frame model matrix. Composes on top of the model baked into the
    /// upload, so identity here renders the points at their original
    /// transform.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl PointCloudRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: crate::resources::PointCloudId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}
