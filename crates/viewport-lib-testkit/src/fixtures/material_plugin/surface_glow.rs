//! A material plugin whose surface hook sets emissive and alpha.

use viewport_lib::{MaterialPlugin, ParamDesc};

/// Authors a black surface through `shade_surface` with a uniform emissive
/// of `glow` and an alpha of `alpha`, both read from declared parameters.
///
/// The lighting hooks return zero, so a draw adds nothing but the emissive
/// term: over a known background a pixel reads `alpha * glow` plus
/// `(1 - alpha)` of the background, which makes the surface hook's two
/// carriers (emissive and alpha) checkable to a tight tolerance on both
/// colour paths. The material has to be `AlphaMode::Blend` for the alpha to
/// take effect.
pub struct SurfaceGlowMaterialPlugin {
    name: &'static str,
    glow: f32,
    alpha: f32,
}

impl SurfaceGlowMaterialPlugin {
    /// A plugin registered under `name` whose `glow` (linear emissive, the
    /// same on all three channels) and `alpha` parameters default to the
    /// given values.
    pub fn new(name: &'static str, glow: f32, alpha: f32) -> Self {
        Self { name, glow, alpha }
    }
}

impl MaterialPlugin for SurfaceGlowMaterialPlugin {
    fn name(&self) -> &str {
        self.name
    }

    fn wgsl_body(&self) -> String {
        "\
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {
    var ov: SurfaceOverride;
    ov.base_colour = vec3<f32>(0.0);
    ov.normal = surf.normal;
    ov.metallic = 0.0;
    ov.roughness = 1.0;
    ov.emissive = vec3<f32>(params.glow);
    ov.alpha = params.alpha;
    return ov;
}
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return vec3<f32>(0.0);
}
"
        .to_string()
    }

    fn param_layout(&self) -> Vec<ParamDesc> {
        vec![
            ParamDesc::f32("glow", self.glow),
            ParamDesc::f32("alpha", self.alpha),
        ]
    }
}
