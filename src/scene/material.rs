/// Per-object material properties for Blinn-Phong and PBR shading.
///
/// Materials carry all shading parameters that were previously global in `LightingSettings`.
/// Each `SceneRenderItem` now has its own `Material`, enabling per-object visual distinction.
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
        }
    }
}

impl Material {
    /// Reproduce pre-Phase-2 appearance from a plain color.
    ///
    /// All other material parameters take their defaults, preserving the previous
    /// Blinn-Phong shading for objects that only had a color override.
    pub fn from_color(color: [f32; 3]) -> Self {
        Self {
            base_color: color,
            ..Default::default()
        }
    }
}
