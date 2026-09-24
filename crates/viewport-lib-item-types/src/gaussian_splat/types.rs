use viewport_lib::ItemSettings;
use viewport_lib::resources::GaussianSplatId;

/// Per-frame reference to an uploaded Gaussian splat set.
#[derive(Clone)]
#[non_exhaustive]
pub struct GaussianSplatItem {
    /// Handle to the uploaded splat set this item draws.
    pub source: GaussianSplatId,
    /// World-space model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for GaussianSplatItem {
    fn default() -> Self {
        Self {
            source: GaussianSplatId::INVALID,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
        }
    }
}
