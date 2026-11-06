//! A shading hook that shades to a fixed flat colour.

use viewport_lib::ShadingHookDesc;

/// A hook registered under `name` that zeroes the lights and returns `colour`
/// (linear) as the ambient term, so a draw through it reads as one known
/// value. Registering it composes and validates every lit shader with it.
pub fn flat_colour_shading_hook(name: &'static str, colour: [f32; 3]) -> ShadingHookDesc {
    let [r, g, b] = colour;
    ShadingHookDesc {
        name: name.into(),
        wgsl_body: format!(
            "\
fn shade_light(surf: ShadingSurface, light: LightSample) -> vec3<f32> {{
    return vec3<f32>(0.0);
}}
fn shade_ambient(surf: ShadingSurface) -> vec3<f32> {{
    return vec3<f32>({r:?}, {g:?}, {b:?});
}}
"
        ),
        needs_back_hemisphere: false,
        texture_count: 0,
        reads_vertex_attribute: false,
        params: Vec::new(),
    }
}
