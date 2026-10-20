//! Toon and rim material plugins built on `viewport_lib::MaterialPlugin`.
//!
//! Two fragment-shading plugins registered through
//! `register_material_plugin` and selected per material via
//! `Material::with_plugin`:
//!
//! - [`ToonPlugin`]: banded diffuse (`shade_light`) plus flat ambient
//!   (`shade_ambient`), tinted and banded through its declared parameters,
//!   with one texture slot modulating the albedo (1x1 white fallback when a
//!   material names none).
//! - [`RimPlugin`]: a `recolor` hook that adds a view-dependent rim on top of
//!   the untouched built-in lighting.
//!
//! This is example code shared across showcases. Treat it as a reference
//! `MaterialPlugin` implementation rather than a shipping toon shader; the
//! built-in toon shading model is a separate, library-side concern.

use viewport_lib as vpl;

/// A [`ToonPlugin`] material: `bands` diffuse steps, an `ambient` factor, the
/// texture's UV `tiling` and a linear `tint`.
#[allow(dead_code)] // the plugin tests include this file and build their own materials
pub fn toon_material(
    plugin: vpl::MaterialPluginId,
    bands: f32,
    ambient: f32,
    tiling: f32,
    tint: [f32; 3],
) -> vpl::Material {
    vpl::Material::default()
        .with_plugin(plugin)
        .with_param("bands", bands)
        .with_param("ambient", ambient)
        .with_param("tiling", tiling)
        .with_param("tint", tint)
}

/// Banded-diffuse toon shading with a tint and an albedo texture slot.
pub struct ToonPlugin;

impl vpl::MaterialPlugin for ToonPlugin {
    fn name(&self) -> &'static str {
        "example_toon"
    }
    fn texture_count(&self) -> u32 {
        1
    }
    fn wgsl_body(&self) -> String {
        // material_texture_0 is the 1x1 white fallback unless the material
        // names one, so untextured materials pay a neutral multiply.
        r#"
fn toon_tex(surf: ShadingSurface) -> vec3<f32> {
    let tile = max(params.tiling, 1.0);
    return textureSampleGrad(
        material_texture_0, material_sampler,
        surf.uv * tile, surf.uv_ddx * tile, surf.uv_ddy * tile,
    ).rgb;
}
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    let bands = max(params.bands, 1.0);
    let ndl = max(dot(surf.normal, light.l), 0.0);
    let stepped = ceil(ndl * bands) / bands;
    return surf.base_colour * params.tint * toon_tex(surf) * stepped * light.radiance * light.shadow;
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return surf.base_colour * params.tint * toon_tex(surf) * params.ambient * surf.ao;
}
"#
        .to_string()
    }
    fn param_layout(&self) -> Vec<vpl::ParamDesc> {
        vec![
            vpl::ParamDesc::f32("bands", 3.0),
            vpl::ParamDesc::f32("ambient", 0.25),
            vpl::ParamDesc::f32("tiling", 1.0),
            vpl::ParamDesc::vec3("tint", [1.0, 1.0, 1.0]),
        ]
    }
}

/// View-dependent rim light added over the built-in lighting: a `colour` and
/// a falloff `power`.
pub struct RimPlugin;

impl vpl::MaterialPlugin for RimPlugin {
    fn name(&self) -> &'static str {
        "example_rim"
    }
    fn wgsl_body(&self) -> String {
        // Recolor-only: the built-in PBR direct + ambient terms arrive
        // untouched and the rim adds on top.
        r#"
fn recolor(surf: ShadingSurface, direct: vec3<f32>, ambient: vec3<f32>) -> vec3<f32> {
    let facing = max(dot(surf.normal, surf.view_dir), 0.0);
    let rim = pow(1.0 - facing, max(params.power, 0.5));
    return direct + ambient + params.colour * rim;
}
"#
        .to_string()
    }
    fn param_layout(&self) -> Vec<vpl::ParamDesc> {
        vec![
            vpl::ParamDesc::vec3("colour", [0.2, 0.5, 1.0]),
            vpl::ParamDesc::f32("power", 3.0),
        ]
    }
}
