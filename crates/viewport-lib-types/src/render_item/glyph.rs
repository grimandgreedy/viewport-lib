//! Frame-facing glyph render-item descriptors (glyph type and placement).

use crate::colourmap::ColourmapId;
use crate::material::ItemSettings;

/// Glyph shape type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlyphType {
    /// Cone tip + cylinder shaft.
    #[default]
    Arrow,
    /// Icosphere.
    Sphere,
    /// Unit cube.
    Cube,
}

/// A set of instanced glyphs to render (e.g. velocity arrows).
#[derive(Clone)]
#[non_exhaustive]
pub struct GlyphItem {
    /// World-space base positions (one per glyph instance).
    pub positions: Vec<[f32; 3]>,
    /// Per-instance direction vectors. Length = magnitude (used for orientation + optional scale).
    pub vectors: Vec<[f32; 3]>,
    /// Global scale factor applied to all glyph instances. Default: 1.0.
    pub scale: f32,
    /// Whether glyph size scales with vector magnitude. Default: true.
    pub scale_by_magnitude: bool,
    /// Clamp magnitude range for scaling. None = no clamping.
    pub magnitude_clamp: Option<(f32, f32)>,
    /// Optional per-instance scalar values for LUT colouring. Empty = colour by magnitude.
    pub scalars: Vec<f32>,
    /// Scalar range for LUT mapping. None = auto from data.
    pub scalar_range: Option<(f32, f32)>,
    /// Colourmap for scalar colouring. None = use default builtin (viridis).
    pub colourmap_id: Option<ColourmapId>,
    /// Fallback RGBA colour used when `use_default_colour` is true. Default: transparent (unused).
    pub default_colour: crate::colour::Colour,
    /// When true, glyphs are coloured by `default_colour` (with per-instance scalar as brightness)
    /// instead of the LUT. Default: false.
    pub use_default_colour: bool,
    /// Glyph shape. Default: Arrow.
    pub glyph_type: GlyphType,
    /// World-space model matrix. Default: identity.
    pub model: [[f32; 4]; 4],
    /// Per-item render settings (visibility, appearance, pick identity, selection state).
    pub settings: ItemSettings,
}

impl Default for GlyphItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            vectors: Vec::new(),
            scale: 1.0,
            scale_by_magnitude: true,
            magnitude_clamp: None,
            scalars: Vec::new(),
            scalar_range: None,
            colourmap_id: None,
            default_colour: [0.0; 4].into(),
            use_default_colour: false,
            glyph_type: GlyphType::Arrow,
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            settings: ItemSettings::default(),
        }
    }
}
