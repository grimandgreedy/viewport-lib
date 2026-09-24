//! The tensor glyph item type: instanced ellipsoids scaled by an
//! eigen-decomposition, plus the handle and reference form for a set uploaded
//! once.

use viewport_lib::ItemSettings;
use viewport_lib::resources::ColourmapId;

viewport_lib::resources::handle::slot_handle! {
    /// Handle to a tensor glyph set uploaded once through
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload).
    ///
    /// Name it from a [`TensorGlyphSetRefItem`] to draw the stored set without
    /// rebuilding its instance buffer.
    pub struct TensorGlyphSetId;
}

const IDENTITY_MAT4: [[f32; 4]; 4] = glam::Mat4::IDENTITY.to_cols_array_2d();

/// Per-frame reference to a pre-uploaded tensor glyph set. See `PolylineRefItem`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TensorGlyphSetRefItem {
    /// Handle to GPU buffers produced by
    /// [`Uploads::upload`](viewport_lib::plugin_api::Uploads::upload)
    /// or `Uploads::begin_upload`.
    pub source: TensorGlyphSetId,
    /// Per-frame model matrix. Composes on top of the per-instance
    /// transforms baked at upload time.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings.
    pub settings: ItemSettings,
}

impl TensorGlyphSetRefItem {
    /// Visible reference at the identity transform.
    pub fn new(id: TensorGlyphSetId) -> Self {
        Self {
            source: id,
            model: IDENTITY_MAT4,
            settings: ItemSettings::default(),
        }
    }
}

/// A set of instanced tensor glyphs for stress/strain visualisation.
///
/// Each instance is an ellipsoid at `positions[i]`, scaled anisotropically by the
/// absolute eigenvalues along the eigenvector axes. Colour comes from `colour_attribute`
/// if provided, otherwise from the sign of the first (dominant) eigenvalue.
#[derive(Clone)]
#[non_exhaustive]
pub struct TensorGlyphItem {
    /// World-space positions, one per instance.
    pub positions: Vec<[f32; 3]>,
    /// Per-instance eigenvalues `[lambda0, lambda1, lambda2]`.
    /// The ellipsoid is scaled by `|lambda_i| * scale` along each eigenvector axis.
    pub eigenvalues: Vec<[f32; 3]>,
    /// Per-instance eigenvectors as column vectors `[[e0x,e0y,e0z], [e1x,...], [e2x,...]]`.
    /// Must form an orthonormal basis. Length must match `positions`.
    pub eigenvectors: Vec<[[f32; 3]; 3]>,
    /// Global scale factor applied to all instances. Default: 1.0.
    pub scale: f32,
    /// Optional per-instance scalar values for LUT colouring.
    /// When `None`, colours by sign of `eigenvalues[i][0]`: positive -> upper LUT, negative -> lower LUT.
    pub colour_attribute: Option<Vec<f32>>,
    /// Scalar range for LUT mapping. `None` = auto from data.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. `None` = viridis. For sign colouring, a diverging map works best.
    pub colourmap_id: Option<ColourmapId>,
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for TensorGlyphItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            scale: 1.0,
            colour_attribute: None,
            scalar_range: None,
            colourmap_id: None,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
        }
    }
}
