//! A material plugin whose surface emits the frame inputs it reads.

use viewport_lib::MaterialPlugin;

/// Emits the camera uniform's frame inputs as colour, with lighting zeroed:
/// red is `camera.time.x` (seconds), green `camera.viewport.x / 256` (the
/// viewport width in physical pixels) and blue half the raw vertex colour's
/// green (`surf.vertex_colour`, white on a mesh without colours).
///
/// The base colour is black and the lighting hooks return zero, so a pixel
/// reads exactly that emissive: a test sets `FrameData::time` and the
/// viewport size and checks each channel.
pub struct FrameInputsMaterialPlugin {
    name: &'static str,
}

impl FrameInputsMaterialPlugin {
    /// A plugin registered under `name`.
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl MaterialPlugin for FrameInputsMaterialPlugin {
    fn name(&self) -> &str {
        self.name
    }

    fn wgsl_body(&self) -> String {
        "\
fn shade_surface(surf: ShadingSurface) -> SurfaceOverride {
    var ov = surface_override_from(surf);
    ov.base_colour = vec3<f32>(0.0);
    ov.emissive = vec3<f32>(camera.time.x, camera.viewport.x / 256.0, surf.vertex_colour.g * 0.5);
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
}
