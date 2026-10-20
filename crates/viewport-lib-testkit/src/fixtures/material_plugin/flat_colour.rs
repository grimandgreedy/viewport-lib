//! A material plugin that shades to a flat colour from a declared parameter.

use viewport_lib::{MaterialPlugin, ParamDesc};

/// Shades every lit draw to its material's `colour` parameter, ignoring
/// lights: the colour comes in through the ambient hook, and lights add
/// nothing.
///
/// A flat colour is the easiest thing for a test to assert: the hooked draw
/// reads as one known value, so any difference from the stock shading path is
/// unambiguous, and setting the parameter on a material changes the rendered
/// colour without re-registering.
pub struct FlatColourMaterialPlugin {
    name: &'static str,
    colour: [f32; 3],
}

impl FlatColourMaterialPlugin {
    /// A plugin registered under `name` whose `colour` parameter defaults to
    /// `colour` (linear, not sRGB: the parameter feeds the shader directly).
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
    return params.colour;
}
"
        .to_string()
    }

    fn param_layout(&self) -> Vec<ParamDesc> {
        vec![ParamDesc::vec3("colour", self.colour)]
    }
}
