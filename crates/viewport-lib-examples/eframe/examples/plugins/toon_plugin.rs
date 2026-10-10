//! Toon and rim material plugins built on `viewport_lib::MaterialPlugin`.
//!
//! Two fragment-shading plugins registered through
//! `register_material_plugin` and selected per material via
//! `Material::shading_plugin`:
//!
//! - [`ToonPlugin`]: banded diffuse (`shade_light`) plus flat ambient
//!   (`shade_ambient`), tinted and banded through the group-3 params window,
//!   with one texture slot modulating the albedo (1x1 white fallback when a
//!   variant binds none).
//! - [`RimPlugin`]: a `recolor` hook that adds a view-dependent rim on top of
//!   the untouched built-in lighting.
//!
//! This is example code shared across showcases. Treat it as a reference
//! `MaterialPlugin` implementation rather than a shipping toon shader; the
//! built-in toon shading model is a separate, library-side concern.

/// Params window layout for [`ToonPlugin`]:
/// `[0] = (bands, ambient, uv_tiling, 0)`, `[1] = (tint rgb, 0)`.
use viewport_lib as vpl;

pub fn toon_params(
    bands: f32,
    ambient: f32,
    tiling: f32,
    tint: [f32; 3],
) -> [[f32; 4]; vpl::MATERIAL_PLUGIN_PARAM_VEC4S] {
    let mut p = [[0.0; 4]; vpl::MATERIAL_PLUGIN_PARAM_VEC4S];
    p[0] = [bands, ambient, tiling, 0.0];
    p[1] = [tint[0], tint[1], tint[2], 0.0];
    p
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
        // See `toon_params` for the params window layout. material_texture_0
        // is the 1x1 white fallback unless a variant binds one, so untextured
        // variants pay a neutral multiply.
        r#"
fn toon_tex(surf: ShadingSurface) -> vec3<f32> {
    let tile = max(material_params[0].z, 1.0);
    return textureSampleGrad(
        material_texture_0, material_sampler,
        surf.uv * tile, surf.uv_ddx * tile, surf.uv_ddy * tile,
    ).rgb;
}
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    let bands = max(material_params[0].x, 1.0);
    let ndl = max(dot(surf.normal, light.l), 0.0);
    let stepped = ceil(ndl * bands) / bands;
    let tint = material_params[1].xyz;
    return surf.base_colour * tint * toon_tex(surf) * stepped * light.radiance * light.shadow;
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    let tint = material_params[1].xyz;
    return surf.base_colour * tint * toon_tex(surf) * material_params[0].y * surf.ao;
}
"#
        .to_string()
    }
    fn initial_params(&self) -> [[f32; 4]; vpl::MATERIAL_PLUGIN_PARAM_VEC4S] {
        toon_params(3.0, 0.25, 1.0, [1.0, 1.0, 1.0])
    }
}

/// View-dependent rim light added over the built-in lighting.
///
/// Params window: `[0] = (rim rgb, rim power)`.
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
    let rim = pow(1.0 - facing, max(material_params[0].w, 0.5));
    return direct + ambient + material_params[0].xyz * rim;
}
"#
        .to_string()
    }
    fn initial_params(&self) -> [[f32; 4]; vpl::MATERIAL_PLUGIN_PARAM_VEC4S] {
        let mut p = [[0.0; 4]; vpl::MATERIAL_PLUGIN_PARAM_VEC4S];
        p[0] = [0.2, 0.5, 1.0, 3.0];
        p
    }
}
