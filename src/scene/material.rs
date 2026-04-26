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
/// Used with [`BackfacePolicy::Pattern`] to render a procedural pattern on back faces.
/// The pattern is evaluated in world space so it remains stable as the camera moves.
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
    /// Back faces are visible and rendered with a procedural pattern in the given RGB color.
    ///
    /// The pattern alternates between the specified color and the object's base color.
    /// The normal is flipped for correct lighting.
    Pattern {
        /// Which procedural pattern to use.
        pattern: BackfacePattern,
        /// RGB color for the pattern foreground (linear 0..1).
        color: [f32; 3],
    },
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
