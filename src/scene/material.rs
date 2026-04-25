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
    /// Polar checkerboard centred at UV (0.5, 0.5) — reveals rotational consistency.
    LocalChecker = 3,
    /// Concentric rings centred at UV (0.5, 0.5) — reveals radial distortion.
    LocalRadial = 4,
}

/// UV parameterization visualization settings.
///
/// Attach to [`Material::param_vis`] to enable procedural UV pattern rendering.
/// The `scale` controls tile frequency — higher values produce more, smaller tiles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamVis {
    /// Which procedural pattern to render.
    pub mode: ParamVisMode,
    /// Tile frequency multiplier. Default 8.0 — produces 8 checker squares per UV unit.
    pub scale: f32,
}

impl Default for ParamVis {
    fn default() -> Self {
        Self { mode: ParamVisMode::Checker, scale: 8.0 }
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
    /// procedural pattern in UV space — useful for inspecting parameterization quality.
    ///
    /// Requires UV coordinates on the mesh. Default None (standard shading).
    pub param_vis: Option<ParamVis>,
    /// Render with back-face culling disabled so both sides are shaded.
    ///
    /// Useful for single-sided geometry like planes and open surfaces. Default false.
    pub two_sided: bool,
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
            two_sided: false,
        }
    }
}

impl Material {
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
