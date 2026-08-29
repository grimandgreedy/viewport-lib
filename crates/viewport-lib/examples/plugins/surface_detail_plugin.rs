//! Detail-layer and parallax material plugins built on
//! `viewport_lib::MaterialPlugin`.
//!
//! Three fragment-shading plugins that add surface detail without any core
//! `Material` fields:
//!
//! - [`DetailLayerPlugin`]: a pure `shade_surface` body that blends a tiled
//!   detail albedo into the base colour; stock lighting, shadows, and IBL
//!   run on the authored surface. The blend can be gated by the mesh's
//!   per-vertex extension attribute (`surf.attr.x`), which is the custom
//!   vertex-attribute demo: paint a mask into
//!   `MeshData::extension_attributes` and the detail follows it.
//! - [`ParallaxPlugin`]: carries its own height and albedo textures and
//!   runs a small parallax march in tangent space, so surface relief is
//!   faked entirely inside the plugin; the library has no height binding.
//!   It keeps the lighting-hook shape (`shade_light` + `shade_ambient`
//!   calling `pbr_light_contrib`) as the reference for that style.
//! - [`DissolvePlugin`]: a `shade_surface` body driving the gated alpha
//!   output: a procedural dissolve cuts fragments on a Mask material, with
//!   an emissive glow at the burning edge.

/// Params window layout for [`DetailLayerPlugin`]:
/// `[0] = (tiling, strength, attr_mask, 0)`. `attr_mask` 0..1 fades between
/// applying the detail everywhere and gating it by `surf.attr.x`.
pub fn detail_params(
    tiling: f32,
    strength: f32,
    attr_mask: f32,
) -> [[f32; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S] {
    let mut p = [[0.0; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S];
    p[0] = [tiling, strength, attr_mask, 0.0];
    p
}

/// Tiled detail albedo blended over the base colour, optionally masked by
/// the per-vertex extension attribute.
pub struct DetailLayerPlugin;

impl viewport_lib::MaterialPlugin for DetailLayerPlugin {
    fn name(&self) -> &'static str {
        "example_detail"
    }
    fn texture_count(&self) -> u32 {
        1
    }
    fn reads_vertex_attribute(&self) -> bool {
        true
    }
    fn wgsl_body(&self) -> String {
        // See `detail_params` for the params window layout. material_texture_0
        // is the detail albedo; surf.attr.x is the per-vertex mask. A pure
        // `shade_surface` body: the detail blends into the base colour once,
        // and stock direct lighting, ambient, and IBL all run on the result.
        r#"
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {
    let tile = max(material_params[0].x, 1.0);
    let detail = textureSampleGrad(
        material_texture_0, material_sampler,
        surf.uv * tile, surf.uv_ddx * tile, surf.uv_ddy * tile,
    ).rgb;
    let mask = mix(1.0, surf.attr.x, clamp(material_params[0].z, 0.0, 1.0));
    let strength = clamp(material_params[0].y, 0.0, 1.0) * mask;
    var ov: SurfaceOverride;
    ov.base_colour = surf.base_colour * mix(vec3<f32>(1.0), detail, strength);
    ov.normal = surf.normal;
    ov.metallic = surf.metallic;
    ov.roughness = surf.roughness;
    ov.emissive = vec3<f32>(0.0);
    ov.alpha = surf.alpha;
    return ov;
}
"#
        .to_string()
    }
    fn initial_params(&self) -> [[f32; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S] {
        detail_params(6.0, 0.8, 0.0)
    }
}

/// Params window layout for [`ParallaxPlugin`]:
/// `[0] = (height_scale, tiling, 0, 0)`.
pub fn parallax_params(
    height_scale: f32,
    tiling: f32,
) -> [[f32; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S] {
    let mut p = [[0.0; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S];
    p[0] = [height_scale, tiling, 0.0, 0.0];
    p
}

/// Parallax relief from a plugin-owned height texture.
///
/// Texture slots: 0 = height (R channel, 1 = raised), 1 = albedo.
pub struct ParallaxPlugin;

impl viewport_lib::MaterialPlugin for ParallaxPlugin {
    fn name(&self) -> &'static str {
        "example_parallax"
    }
    fn texture_count(&self) -> u32 {
        2
    }
    fn wgsl_body(&self) -> String {
        // Fixed-step layer march toward the surface in tangent space, then
        // one refinement mix. Gradients are captured in uniform control flow
        // (surf.uv_ddx/uv_ddy), as the hook contract requires.
        r#"
fn parallax_height(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> f32 {
    return textureSampleGrad(material_texture_0, material_sampler, uv, ddx, ddy).r;
}
fn parallax_uv(surf: ShadingSurface) -> vec2<f32> {
    let scale = material_params[0].x;
    let tile = max(material_params[0].y, 1.0);
    let ddx = surf.uv_ddx * tile;
    let ddy = surf.uv_ddy * tile;
    // View direction in tangent space; clamp z so grazing angles do not
    // smear the march across the whole texture.
    let v_ts = vec3<f32>(
        dot(surf.view_dir, surf.tangent),
        dot(surf.view_dir, surf.bitangent),
        max(dot(surf.view_dir, surf.normal), 0.25),
    );
    let steps = 12.0;
    let layer = 1.0 / steps;
    let delta = (v_ts.xy / v_ts.z) * scale / steps;
    var uv = surf.uv * tile;
    var depth = 0.0;
    var h = 1.0 - parallax_height(uv, ddx, ddy);
    for (var i = 0; i < 12; i = i + 1) {
        if depth >= h { break; }
        uv -= delta;
        depth += layer;
        h = 1.0 - parallax_height(uv, ddx, ddy);
    }
    // One secant-style refinement between the last two layers.
    let uv_prev = uv + delta;
    let h_prev = 1.0 - parallax_height(uv_prev, ddx, ddy) - (depth - layer);
    let h_curr = h - depth;
    let w = h_curr / max(h_curr - h_prev, 1e-4);
    return mix(uv, uv_prev, clamp(w, 0.0, 1.0));
}
fn parallax_albedo(surf: ShadingSurface) -> vec3<f32> {
    let tile = max(material_params[0].y, 1.0);
    let uv = parallax_uv(surf);
    let alb = textureSampleGrad(material_texture_1, material_sampler,
                                uv, surf.uv_ddx * tile, surf.uv_ddy * tile).rgb;
    // Darken crevices slightly so the relief reads without a normal map.
    let h = parallax_height(uv, surf.uv_ddx * tile, surf.uv_ddy * tile);
    return alb * mix(0.55, 1.0, h);
}
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {
    return pbr_light_contrib(surf.normal, surf.view_dir, light.l,
                             light.radiance * light.shadow,
                             parallax_albedo(surf), surf.metallic, surf.roughness, surf.f0);
}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {
    return parallax_albedo(surf) * 0.22 * surf.ao;
}
"#
        .to_string()
    }
    fn initial_params(&self) -> [[f32; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S] {
        parallax_params(0.06, 3.0)
    }
}

/// Params window layout for [`DissolvePlugin`]:
/// `[0] = (threshold, edge_width, tiling, 0)`, `[1] = (edge glow rgb, 0)`.
pub fn dissolve_params(
    threshold: f32,
    edge_width: f32,
    tiling: f32,
    edge_colour: [f32; 3],
) -> [[f32; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S] {
    let mut p = [[0.0; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S];
    p[0] = [threshold, edge_width, tiling, 0.0];
    p[1] = [edge_colour[0], edge_colour[1], edge_colour[2], 0.0];
    p
}

/// Procedural dissolve through the surface hook's gated alpha output.
///
/// A hashed cell noise over the UVs is compared against the threshold: cells
/// below it output alpha 0 (discarded on a Mask material, faded on Blend),
/// and cells just above it glow through the emissive output. Assign to a
/// material with `alpha_mode = Mask(0.5)` (or `Blend`); opaque materials
/// ignore the alpha and only show the glow.
pub struct DissolvePlugin;

impl viewport_lib::MaterialPlugin for DissolvePlugin {
    fn name(&self) -> &'static str {
        "example_dissolve"
    }
    fn wgsl_body(&self) -> String {
        r#"
fn dissolve_noise(uv: vec2<f32>) -> f32 {
    return fract(sin(dot(floor(uv), vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {
    let threshold = clamp(material_params[0].x, 0.0, 1.0);
    let edge = max(material_params[0].y, 1e-3);
    let tile = max(material_params[0].z, 1.0);
    let n = dissolve_noise(surf.uv * tile * 8.0);
    var ov: SurfaceOverride;
    ov.base_colour = surf.base_colour;
    ov.normal = surf.normal;
    ov.metallic = surf.metallic;
    ov.roughness = surf.roughness;
    // Fragments below the threshold dissolve away; the band just above it
    // glows. threshold 0 leaves the surface intact and unlit by the glow.
    let over = n - threshold;
    ov.alpha = select(surf.alpha, 0.0, over < 0.0);
    let glow = (1.0 - clamp(over / edge, 0.0, 1.0)) * select(0.0, 1.0, threshold > 0.0);
    ov.emissive = material_params[1].xyz * glow;
    return ov;
}
"#
        .to_string()
    }
    fn initial_params(&self) -> [[f32; 4]; viewport_lib::MATERIAL_PLUGIN_PARAM_VEC4S] {
        dissolve_params(0.0, 0.15, 1.0, [2.5, 1.2, 0.3])
    }
}
