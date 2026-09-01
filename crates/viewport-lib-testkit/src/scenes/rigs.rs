//! Lighting rigs for the catalogue.
//!
//! The existing examples light everything from above with one directional light.
//! These rigs add the angles that expose real problems: grazing light (shadow
//! acne, peter-panning), light from below, multi-light interaction, and
//! backlighting.
//!
//! `LightKind::Directional.direction` is the surface-to-light direction (it
//! points toward the light), so a large +Z component means an overhead sun.

use viewport_lib::{LightKind, LightSource, LightingSettings};

fn directional(direction: [f32; 3], intensity: f32) -> LightSource {
    let mut s = LightSource::default();
    s.kind = LightKind::Directional { direction };
    s.intensity = intensity;
    s
}

fn point(position: [f32; 3], range: f32, colour: [f32; 3], intensity: f32) -> LightSource {
    let mut s = LightSource::default();
    s.kind = LightKind::Point {
        position,
        range,
        radius: 0.0,
    };
    s.colour = colour;
    s.intensity = intensity;
    s.cast_shadows = false;
    s
}

fn with_lights(lights: Vec<LightSource>, hemisphere: f32) -> LightingSettings {
    let mut l = LightingSettings::default();
    l.lights = lights;
    l.hemisphere_intensity = hemisphere;
    l
}

/// The conventional overhead key light (matches the existing examples).
pub fn from_above() -> LightingSettings {
    with_lights(vec![directional([0.4, 0.3, 1.5], 1.0)], 0.5)
}

/// A low-angle sun. Long shadows; the angle that surfaces shadow acne and
/// peter-panning on flat faces.
pub fn grazing() -> LightingSettings {
    with_lights(vec![directional([1.0, 0.35, 0.30], 1.2)], 0.3)
}

/// Light coming from underneath the scene. Inverts the usual ambient occlusion
/// read and catches one-sided normal assumptions.
pub fn from_below() -> LightingSettings {
    let mut l = with_lights(vec![directional([0.2, 0.2, -1.2], 1.0)], 0.25);
    // Warm the under-light, cool the sky so the inversion is obvious.
    l.sky_colour = [0.5, 0.55, 0.65];
    l.ground_colour = [0.7, 0.6, 0.45];
    l
}

/// Classic key / fill / back three-point setup.
pub fn three_point() -> LightingSettings {
    with_lights(
        vec![
            directional([0.6, 0.4, 1.0], 1.0),  // key
            directional([-0.7, 0.3, 0.4], 0.4), // fill
            directional([0.0, -0.8, 0.5], 0.5), // back
        ],
        0.25,
    )
}

/// Eight coloured point lights in a ring above the scene. Exercises the
/// clustered-lighting path and multi-light shading.
pub fn eight_point_lights() -> LightingSettings {
    let colours = [
        [1.0, 0.4, 0.4],
        [1.0, 0.7, 0.3],
        [1.0, 1.0, 0.4],
        [0.4, 1.0, 0.4],
        [0.4, 1.0, 0.9],
        [0.4, 0.6, 1.0],
        [0.7, 0.4, 1.0],
        [1.0, 0.4, 0.9],
    ];
    let mut lights = Vec::with_capacity(8);
    for (i, colour) in colours.into_iter().enumerate() {
        let a = i as f32 / 8.0 * std::f32::consts::TAU;
        let pos = [5.0 * a.cos(), 5.0 * a.sin(), 4.0];
        lights.push(point(pos, 20.0, colour, 6.0));
    }
    with_lights(lights, 0.15)
}

/// A single light behind the subject (on the -Y side), so a +Y camera sees a
/// rim-lit silhouette. Stresses two-sided shading and alpha edges.
pub fn backlit() -> LightingSettings {
    with_lights(vec![directional([0.0, -1.0, 0.25], 1.4)], 0.2)
}
