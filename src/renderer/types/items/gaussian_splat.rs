use crate::scene::material::ItemSettings;

/// SH degree stored with a Gaussian splat set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShDegree {
    /// 3 floats per splat (base RGB only).
    #[default]
    Zero,
    /// 12 floats per splat.
    One,
    /// 48 floats per splat.
    Three,
}

impl ShDegree {
    /// Number of SH coefficients per splat for this degree.
    pub fn coeff_count(self) -> usize {
        match self {
            ShDegree::Zero => 3,
            ShDegree::One => 12,
            ShDegree::Three => 48,
        }
    }
}

crate::resources::handle::slot_handle! {
    /// Handle to an uploaded Gaussian splat set.
    ///
    /// Carries the slot index plus the generation the slot had when the handle
    /// was issued. A handle whose splat set was removed (its slot freed and
    /// reused by a later upload) resolves to no set on lookup, so it cannot
    /// alias whatever now occupies the slot.
    pub struct GaussianSplatId;
}

/// Upload data for a Gaussian splat set. Submitted once via
/// `resources_mut().upload_gaussian_splats(data)`.
pub struct GaussianSplatData {
    /// Object-space center positions, one [f32;3] per splat.
    pub positions: Vec<[f32; 3]>,
    /// Scale (positive floats, world-space metres) per splat, one [f32;3].
    pub scales: Vec<[f32; 3]>,
    /// Unit quaternion rotation per splat [x, y, z, w].
    pub rotations: Vec<[f32; 4]>,
    /// Opacity per splat in [0, 1].
    pub opacities: Vec<f32>,
    /// SH coefficients. Length must equal `positions.len() * sh_degree.coeff_count()`.
    /// For ShDegree::Zero these are [r, g, b] base colours per splat.
    pub sh_coefficients: Vec<f32>,
    /// SH degree for this splat set.
    pub sh_degree: ShDegree,
}

impl Default for GaussianSplatData {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            scales: Vec::new(),
            rotations: Vec::new(),
            opacities: Vec::new(),
            sh_coefficients: Vec::new(),
            sh_degree: ShDegree::Zero,
        }
    }
}

/// Per-frame reference to an uploaded Gaussian splat set.
#[derive(Clone)]
#[non_exhaustive]
pub struct GaussianSplatItem {
    /// Handle to the uploaded splat set.
    pub id: GaussianSplatId,
    /// World-space model matrix.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for GaussianSplatItem {
    fn default() -> Self {
        Self {
            id: GaussianSplatId::INVALID,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
        }
    }
}
