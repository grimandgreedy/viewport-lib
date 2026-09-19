//! A material plugin that authors the PBR surface from a texture.

use viewport_lib::{MATERIAL_PLUGIN_PARAM_VEC4S, MaterialPlugin};

/// Authors the surface through `shade_surface`, taking base colour from the
/// plugin's first texture slot multiplied by a tint in `material_params[0]`,
/// and leaving stock lighting to run on the result.
///
/// Where [`FlatColourMaterialPlugin`](super::FlatColourMaterialPlugin) covers
/// the two lighting hooks, this one covers the parts a real material plugin
/// leans on and that fixture does not touch: the `shade_surface` hook and its
/// `SurfaceOverride` return, the group-3 texture bindings
/// (`material_sampler`, `material_texture_0`), and a params window a test
/// rewrites between frames through
/// [`material_plugin_params_handle`](viewport_lib::resources::DeviceResources::material_plugin_params_handle).
///
/// `shade_surface` runs before the light loop in uniform control flow, so the
/// body samples with plain `textureSample`, which the lighting hooks are not
/// allowed to do.
pub struct TexturedMaterialPlugin {
    name: &'static str,
    tint: [f32; 3],
}

impl TexturedMaterialPlugin {
    /// A plugin registered under `name` whose default params hold `tint`
    /// (linear), the multiplier applied to the sampled texture.
    pub fn new(name: &'static str, tint: [f32; 3]) -> Self {
        Self { name, tint }
    }
}

impl MaterialPlugin for TexturedMaterialPlugin {
    fn name(&self) -> &'static str {
        self.name
    }

    fn wgsl_body(&self) -> String {
        "\
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {
    var ov: SurfaceOverride;
    let sampled = textureSample(material_texture_0, material_sampler, surf.uv);
    ov.base_colour = sampled.rgb * material_params[0].rgb;
    ov.normal = surf.normal;
    ov.metallic = surf.metallic;
    ov.roughness = surf.roughness;
    ov.emissive = vec3<f32>(0.0);
    ov.alpha = surf.alpha;
    return ov;
}
"
        .to_string()
    }

    fn texture_count(&self) -> u32 {
        1
    }

    fn initial_params(&self) -> [[f32; 4]; MATERIAL_PLUGIN_PARAM_VEC4S] {
        let mut params = [[0.0; 4]; MATERIAL_PLUGIN_PARAM_VEC4S];
        params[0] = [self.tint[0], self.tint[1], self.tint[2], 1.0];
        params
    }
}
