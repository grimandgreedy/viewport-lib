//! Point-based submission payloads: Gaussian splat sets.

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

/// Upload data for a Gaussian splat set. Submitted once via
/// `resources_mut().upload_gaussian_splat(data)`.
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
