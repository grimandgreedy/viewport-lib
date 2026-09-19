//! A material plugin that shades to a flat colour from its params window.

use viewport_lib::{MATERIAL_PLUGIN_PARAM_VEC4S, MaterialPlugin};

/// Shades every lit draw to the colour in `material_params[0]`, ignoring
/// lights, and contributes nothing from ambient.
///
/// A flat colour is the easiest thing for a test to assert: the hooked draw
/// reads as one known value, so any difference from the stock shading path is
/// unambiguous, and writing the params window changes the rendered colour
/// without re-registering.
pub struct FlatColourMaterialPlugin {
    name: &'static str,
    colour: [f32; 3],
}

impl FlatColourMaterialPlugin {
    /// A plugin registered under `name` whose default params hold `colour`
    /// (linear, not sRGB: the params window feeds the shader directly).
    pub fn new(name: &'static str, colour: [f32; 3]) -> Self {
        Self { name, colour }
    }
}

impl MaterialPlugin for FlatColourMaterialPlugin {
    fn name(&self) -> &'static str {
        self.name
    }

    fn wgsl_body(&self) -> String {
        "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return material_params[0].rgb;
}
"
        .to_string()
    }

    fn initial_params(&self) -> [[f32; 4]; MATERIAL_PLUGIN_PARAM_VEC4S] {
        let mut params = [[0.0; 4]; MATERIAL_PLUGIN_PARAM_VEC4S];
        params[0] = [self.colour[0], self.colour[1], self.colour[2], 1.0];
        params
    }
}
