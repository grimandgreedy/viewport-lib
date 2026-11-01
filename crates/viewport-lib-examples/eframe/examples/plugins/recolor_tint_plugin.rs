//! A material plugin that tints the final lit colour through `recolor`.
//!
//! Lighting runs as usual and only the last step changes: the hook returns
//! `(direct + ambient) * tint`. The hook sees exposed radiance and returns
//! exposed radiance, so a recoloured surface darkens and brightens with the
//! camera's exposure like any lit one.

use viewport_lib as vpl;

/// Returns `(direct + ambient) * tint` from the `recolor` hook, with `tint`
/// (linear) read from a declared parameter.
pub struct RecolorTintPlugin;

impl vpl::MaterialPlugin for RecolorTintPlugin {
    fn name(&self) -> &str {
        "example_recolor_tint"
    }

    fn wgsl_body(&self) -> String {
        "\
fn recolor(surf: ShadingSurface, direct: vec3<f32>, ambient: vec3<f32>) -> vec3<f32> {
    return (direct + ambient) * params.tint;
}
"
        .to_string()
    }

    fn param_layout(&self) -> Vec<vpl::ParamDesc> {
        vec![vpl::ParamDesc::vec3("tint", [1.0, 1.0, 1.0])]
    }
}
