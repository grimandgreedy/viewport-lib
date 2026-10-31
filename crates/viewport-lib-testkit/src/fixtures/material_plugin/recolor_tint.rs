//! A material plugin that replaces the final lit colour through `recolor`.

use viewport_lib::{MaterialPlugin, ParamDesc};

/// Returns `(direct + ambient) * tint` from the `recolor` hook, with `tint`
/// read from a declared parameter.
///
/// Lighting runs as usual and only the last step changes, so the hooked draw
/// keeps its shading and takes the tint's hue: a white tint draws the same
/// as the stock path.
pub struct RecolorTintMaterialPlugin {
    name: &'static str,
    tint: [f32; 3],
}

impl RecolorTintMaterialPlugin {
    /// A plugin registered under `name` whose `tint` parameter defaults to
    /// `tint` (linear).
    pub fn new(name: &'static str, tint: [f32; 3]) -> Self {
        Self { name, tint }
    }
}

impl MaterialPlugin for RecolorTintMaterialPlugin {
    fn name(&self) -> &str {
        self.name
    }

    fn wgsl_body(&self) -> String {
        "\
fn recolor(surf: ShadingSurface, direct: vec3<f32>, ambient: vec3<f32>) -> vec3<f32> {
    return (direct + ambient) * params.tint;
}
"
        .to_string()
    }

    fn param_layout(&self) -> Vec<ParamDesc> {
        vec![ParamDesc::vec3("tint", self.tint)]
    }
}
