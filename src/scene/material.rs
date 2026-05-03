/// Procedural UV visualization mode for parameterization inspection.
///
/// When set on a [`Material`], the mesh fragment shader ignores the albedo texture and
/// renders a procedural pattern driven by the mesh UV coordinates instead. Useful for
/// inspecting UV distortion, seams, and parameterization quality without needing a texture.
///
/// Requires the mesh to have UV coordinates. Has no effect if the mesh lacks UVs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamVisMode {
    /// Alternating black/white squares tiled in UV space.
    Checker = 1,
    /// Thin grid lines at UV integer boundaries.
    Grid = 2,
    /// Polar checkerboard centred at UV (0.5, 0.5) : reveals rotational consistency.
    LocalChecker = 3,
    /// Concentric rings centred at UV (0.5, 0.5) : reveals radial distortion.
    LocalRadial = 4,
}

/// UV parameterization visualization settings.
///
/// Attach to [`Material::param_vis`] to enable procedural UV pattern rendering.
/// The `scale` controls tile frequency : higher values produce more, smaller tiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamVis {
    /// Which procedural pattern to render.
    pub mode: ParamVisMode,
    /// Tile frequency multiplier. Default 8.0 : produces 8 checker squares per UV unit.
    pub scale: f32,
}

impl Default for ParamVis {
    fn default() -> Self {
        Self {
            mode: ParamVisMode::Checker,
            scale: 8.0,
        }
    }
}

/// Procedural pattern for back-face rendering.
///
/// Used with [`PatternConfig`] to select which procedural pattern is drawn on back faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackfacePattern {
    /// Alternating squares in world XY.
    Checker = 0,
    /// Diagonal lines (45 degrees).
    Hatching = 1,
    /// Two sets of diagonal lines (45 and 135 degrees).
    Crosshatch = 2,
    /// Horizontal stripes.
    Stripes = 3,
}

/// Configuration for procedural back-face patterns.
///
/// Used with [`BackfacePolicy::Pattern`]. The `scale` controls how many pattern cells
/// fit across the object's longest bounding-box dimension, so the pattern always looks
/// proportional regardless of the mesh's physical size.
///
/// Prefer using `..Default::default()` when constructing so that future fields
/// added to this struct do not require changes at every call site:
///
/// ```rust
/// # use viewport_lib::{PatternConfig, BackfacePattern};
/// let cfg = PatternConfig {
///     pattern: BackfacePattern::Hatching,
///     color: [1.0, 0.5, 0.0],
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatternConfig {
    /// Which procedural pattern to draw on back faces.
    pub pattern: BackfacePattern,
    /// RGB foreground color for the pattern (linear 0..1).
    pub color: [f32; 3],
    /// Number of pattern cells across the object's longest bounding-box dimension.
    ///
    /// Default 20.0. Increase for finer detail, decrease for coarser.
    pub scale: f32,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            pattern: BackfacePattern::Checker,
            color: [1.0, 0.5, 0.0],
            scale: 20.0,
        }
    }
}

/// Controls how back faces of a mesh are rendered.
///
/// Use [`BackfacePolicy::Cull`] (the default) to hide back faces, [`BackfacePolicy::Identical`]
/// to show them with the same shading as front faces, or [`BackfacePolicy::DifferentColor`]
/// to shade back faces in a distinct color : useful for spotting mesh orientation errors or
/// highlighting the interior of open surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackfacePolicy {
    /// Back faces are culled (invisible). Default.
    Cull,
    /// Back faces are visible and shaded identically to front faces.
    Identical,
    /// Back faces are visible and shaded in the given RGB color (linear 0..1).
    ///
    /// Front faces receive normal shading; back faces receive Blinn-Phong shading
    /// with the supplied color and the same ambient/diffuse/specular coefficients.
    /// The normal is flipped so lighting is computed from the back-face perspective.
    DifferentColor([f32; 3]),
    /// Back faces are visible and tinted darker by the given factor (0.0..1.0).
    ///
    /// The base color is multiplied by `(1.0 - factor)`, so `Tint(0.3)` means
    /// back faces are 30% darker. The normal is flipped for correct lighting.
    Tint(f32),
    /// Back faces are rendered with a procedural pattern scaled to the object's size.
    ///
    /// The pattern density is relative to the object's world-space bounding box, so
    /// the appearance stays consistent across meshes of different physical sizes.
    /// The normal is flipped for correct lighting.
    Pattern(PatternConfig),
}

impl Default for BackfacePolicy {
    fn default() -> Self {
        BackfacePolicy::Cull
    }
}

/// Per-object material properties for Blinn-Phong and PBR shading.
///
/// Materials carry all shading parameters that were previously global in `LightingSettings`.
/// Each `SceneRenderItem` now has its own `Material`, enabling per-object visual distinction.
///
/// This struct is `#[non_exhaustive]`: construct via [`Material::default`],
/// [`Material::from_color`], or spread syntax (`..Default::default()`). This allows new
/// fields to be added in future phases without breaking downstream code.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Base diffuse color [r, g, b] in linear 0..1 range. Default [0.7, 0.7, 0.7].
    pub base_color: [f32; 3],
    /// Ambient light coefficient. Default 0.15.
    pub ambient: f32,
    /// Diffuse light coefficient. Default 0.75.
    pub diffuse: f32,
    /// Specular highlight coefficient. Default 0.4.
    pub specular: f32,
    /// Specular shininess exponent. Default 32.0.
    pub shininess: f32,
    /// Metallic factor for PBR Cook-Torrance shading. 0=dielectric, 1=metal. Default 0.0.
    pub metallic: f32,
    /// Roughness factor for PBR microfacet distribution. 0=mirror, 1=fully rough. Default 0.5.
    pub roughness: f32,
    /// Opacity (1.0 = fully opaque, 0.0 = fully transparent). Default 1.0.
    pub opacity: f32,
    /// Optional albedo texture identifier. None = no texture applied. Default None.
    pub texture_id: Option<u64>,
    /// Optional normal map texture identifier. None = no normal mapping. Default None.
    ///
    /// The normal map must be in tangent-space with XY encoded as RG (0..1 -> -1..+1).
    /// Requires UVs and tangents on the mesh for correct TBN construction.
    pub normal_map_id: Option<u64>,
    /// Optional ambient occlusion map texture identifier. None = no AO map. Default None.
    ///
    /// The AO map R channel encodes cavity factor (0=fully occluded, 1=fully lit).
    /// Applied multiplicatively to ambient and diffuse terms.
    pub ao_map_id: Option<u64>,
    /// Use Cook-Torrance PBR shading instead of Blinn-Phong. Default false.
    ///
    /// When true, `metallic` and `roughness` drive the GGX BRDF.
    /// PBR outputs linear HDR values; enable `post_process.enabled` for correct tone mapping.
    pub use_pbr: bool,
    /// Optional matcap texture identifier. When set, matcap shading replaces
    /// Blinn-Phong/PBR. Default None.
    ///
    /// Obtain a `MatcapId` from [`ViewportGpuResources::builtin_matcap_id`] or
    /// [`ViewportGpuResources::upload_matcap`].  Blendable matcaps (alpha-channel)
    /// tint the result with `base_color`; static matcaps override color entirely.
    pub matcap_id: Option<crate::resources::MatcapId>,
    /// UV parameterization visualization. When set, replaces albedo/lighting with a
    /// procedural pattern in UV space : useful for inspecting parameterization quality.
    ///
    /// Requires UV coordinates on the mesh. Default None (standard shading).
    pub param_vis: Option<ParamVis>,
    /// Back-face rendering policy. Default [`BackfacePolicy::Cull`] (back faces hidden).
    ///
    /// Use [`BackfacePolicy::Identical`] for single-sided geometry like planes and open
    /// surfaces. Use [`BackfacePolicy::DifferentColor`] to highlight back faces in a
    /// distinct color : helpful for diagnosing mesh orientation errors.
    pub backface_policy: BackfacePolicy,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [0.7, 0.7, 0.7],
            ambient: 0.15,
            diffuse: 0.75,
            specular: 0.4,
            shininess: 32.0,
            metallic: 0.0,
            roughness: 0.5,
            opacity: 1.0,
            texture_id: None,
            normal_map_id: None,
            ao_map_id: None,
            use_pbr: false,
            matcap_id: None,
            param_vis: None,
            backface_policy: BackfacePolicy::Cull,
        }
    }
}

impl Material {
    /// Returns `true` if the backface policy makes back faces visible.
    pub fn is_two_sided(&self) -> bool {
        !matches!(self.backface_policy, BackfacePolicy::Cull)
    }

    /// Construct from a plain color, all other parameters at their defaults.
    pub fn from_color(color: [f32; 3]) -> Self {
        Self {
            base_color: color,
            ..Default::default()
        }
    }

    /// Construct a Cook-Torrance PBR material.
    ///
    /// - `metallic`: 0.0 = dielectric, 1.0 = full metal
    /// - `roughness`: 0.0 = mirror, 1.0 = fully rough
    ///
    /// All other parameters take their defaults. Enable post-processing
    /// (`PostProcessSettings::enabled = true`) for correct HDR tone mapping.
    pub fn pbr(base_color: [f32; 3], metallic: f32, roughness: f32) -> Self {
        Self {
            base_color,
            use_pbr: true,
            metallic,
            roughness,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values() {
        let m = Material::default();
        assert!((m.base_color[0] - 0.7).abs() < 1e-6);
        assert!((m.ambient - 0.15).abs() < 1e-6);
        assert!((m.diffuse - 0.75).abs() < 1e-6);
        assert!((m.opacity - 1.0).abs() < 1e-6);
        assert!(!m.use_pbr);
        assert!(m.texture_id.is_none());
        assert!(m.normal_map_id.is_none());
        assert!(m.ao_map_id.is_none());
        assert!(m.matcap_id.is_none());
        assert!(m.param_vis.is_none());
    }

    #[test]
    fn from_color_sets_base_color() {
        let m = Material::from_color([1.0, 0.0, 0.5]);
        assert!((m.base_color[0] - 1.0).abs() < 1e-6);
        assert!((m.base_color[1]).abs() < 1e-6);
        assert!((m.base_color[2] - 0.5).abs() < 1e-6);
        // Other fields should be defaults
        assert!((m.ambient - 0.15).abs() < 1e-6);
    }

    #[test]
    fn pbr_constructor() {
        let m = Material::pbr([0.8, 0.2, 0.1], 0.9, 0.3);
        assert!(m.use_pbr);
        assert!((m.metallic - 0.9).abs() < 1e-6);
        assert!((m.roughness - 0.3).abs() < 1e-6);
        assert!((m.base_color[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn is_two_sided_cull() {
        let m = Material::default();
        assert!(!m.is_two_sided());
    }

    #[test]
    fn is_two_sided_identical() {
        let m = Material {
            backface_policy: BackfacePolicy::Identical,
            ..Default::default()
        };
        assert!(m.is_two_sided());
    }

    #[test]
    fn is_two_sided_different_color() {
        let m = Material {
            backface_policy: BackfacePolicy::DifferentColor([1.0, 0.0, 0.0]),
            ..Default::default()
        };
        assert!(m.is_two_sided());
    }

    #[test]
    fn is_two_sided_tint() {
        let m = Material {
            backface_policy: BackfacePolicy::Tint(0.3),
            ..Default::default()
        };
        assert!(m.is_two_sided());
    }

    #[test]
    fn is_two_sided_pattern() {
        let m = Material {
            backface_policy: BackfacePolicy::Pattern(PatternConfig {
                pattern: BackfacePattern::Hatching,
                color: [0.5, 0.5, 0.5],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(m.is_two_sided());
    }

    #[test]
    fn param_vis_default() {
        let pv = ParamVis::default();
        assert_eq!(pv.mode, ParamVisMode::Checker);
        assert!((pv.scale - 8.0).abs() < 1e-6);
    }
}
