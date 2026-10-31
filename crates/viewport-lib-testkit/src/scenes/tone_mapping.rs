//! Tone-map operator scenes: an emissive HDR ramp and colour chart under each
//! operator, and a one-pixel-per-value strip for checking an operator against
//! its CPU version.
//!
//! Every card is a black surface with an emissive term and the scene has no
//! lights and no fill, so a card's exposed radiance is exactly its emissive
//! colour times its strength.

use glam::{Mat4, Vec3};
use viewport_lib::{AgxLook, Colour, LightingSettings, Material, PostProcessSettings, ToneMapping};
use viewport_lib::{Projection, primitives};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, item_model, orbit_camera, upload};

/// Pixel width of the ramp strip, one value per pixel.
pub const RAMP_WIDTH: u32 = 256;
/// Pixel height of the ramp strip.
pub const RAMP_HEIGHT: u32 = 4;

/// Tints the ramp strip cycles through, in linear Rec.709.
const RAMP_TINTS: [[f32; 3]; 6] = [
    [1.0, 1.0, 1.0],
    [1.0, 0.45, 0.1],
    [0.1, 0.6, 0.85],
    [0.9, 0.08, 0.06],
    [0.15, 0.85, 0.2],
    [0.55, 0.2, 0.95],
];

/// The linear radiance of ramp pixel `i`: a tint scaled from 2^-8 to 2^8.
pub fn ramp_value(i: u32) -> [f32; 3] {
    let t = i as f32 / (RAMP_WIDTH - 1) as f32;
    let scale = (-8.0 + 16.0 * t).exp2();
    let tint = RAMP_TINTS[i as usize % RAMP_TINTS.len()];
    [tint[0] * scale, tint[1] * scale, tint[2] * scale]
}

/// A black surface emitting `rgb * strength`.
fn emitter(rgb: [f32; 3], strength: f32) -> Material {
    let mut m = Material::from_colour(Colour::linear_rgb(0.0, 0.0, 0.0));
    m.specular = 0.0;
    m.emissive = Colour::linear_rgb(rgb[0], rgb[1], rgb[2]);
    m.emissive_strength = strength;
    m
}

/// No lights and no hemisphere fill: only emissive reaches the target.
fn dark() -> LightingSettings {
    let mut lighting = LightingSettings::default();
    lighting.lights.clear();
    lighting.hemisphere_intensity = 0.0;
    lighting
}

/// The ramp strip: one emissive card per pixel column, filling the frame.
fn build_ramp(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let card = upload(ctx, &primitives::plane(1.0, 1.0));
    let half = RAMP_WIDTH as f32 / 2.0;
    let items = (0..RAMP_WIDTH)
        .map(|i| {
            let value = ramp_value(i);
            let at = Mat4::from_translation(Vec3::new(i as f32 + 0.5 - half, 0.0, 0.0))
                * Mat4::from_scale(Vec3::new(1.0, 2.0 * RAMP_HEIGHT as f32, 1.0));
            item_model(ctx, card, at, emitter(value, 1.0))
        })
        .collect();
    BuiltScene {
        items,
        lighting: dark(),
        ..Default::default()
    }
}

/// The ramp strip under the default operator. Not in the catalogue: a test
/// renders it at [`RAMP_WIDTH`] x [`RAMP_HEIGHT`] under each operator.
pub fn ramp_scene() -> NamedScene {
    let mut camera = orbit_camera(Vec3::ZERO, 10.0, 0.0, 0.0);
    camera.projection = Projection::orthographic(RAMP_HEIGHT as f32);
    camera.aspect = RAMP_WIDTH as f32 / RAMP_HEIGHT as f32;
    NamedScene {
        name: "tone_ramp",
        cameras: vec![NamedCamera {
            name: "strip",
            camera,
        }],
        build: build_ramp,
    }
}

const GREY_STEPS: usize = 16;
const HUES: [[f32; 3]; 6] = [
    [0.8, 0.1, 0.08],
    [0.85, 0.45, 0.08],
    [0.75, 0.75, 0.1],
    [0.12, 0.6, 0.15],
    [0.08, 0.5, 0.75],
    [0.55, 0.15, 0.7],
];

/// A grey ramp from 2^-9 to 2^6 nits, one stop a card, over two rows of hue
/// cards at 1 and 4 nits.
fn chart(ctx: &mut BuildCtx<'_>, operator: ToneMapping, post: PostProcessSettings) -> BuiltScene {
    let card = upload(ctx, &primitives::plane(1.0, 1.0));
    let mut items = Vec::new();
    let step = 0.46;
    for i in 0..GREY_STEPS {
        let x = (i as f32 - (GREY_STEPS as f32 - 1.0) / 2.0) * step;
        let at = Mat4::from_translation(Vec3::new(x, 1.6, 0.0))
            * Mat4::from_scale(Vec3::new(0.42, 1.2, 1.0));
        let nits = (i as f32 - 9.0).exp2();
        items.push(item_model(ctx, card, at, emitter([1.0; 3], nits)));
    }
    for (y, nits) in [(0.0, 1.0), (-1.6, 4.0)] {
        for (i, &hue) in HUES.iter().enumerate() {
            let x = (i as f32 - 2.5) * 1.15;
            let at = Mat4::from_translation(Vec3::new(x, y, 0.0))
                * Mat4::from_scale(Vec3::new(1.0, 1.2, 1.0));
            items.push(item_model(ctx, card, at, emitter(hue, nits)));
        }
    }
    BuiltScene {
        items,
        lighting: dark(),
        operator: Some(operator),
        post_process: Some(post),
        ..Default::default()
    }
}

fn build_agx(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    chart(ctx, ToneMapping::Agx, PostProcessSettings::default())
}

fn build_agx_punchy(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut post = PostProcessSettings::default();
    post.agx_look = AgxLook::Punchy;
    chart(ctx, ToneMapping::Agx, post)
}

fn build_aces_rrt(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    chart(ctx, ToneMapping::AcesRrt, PostProcessSettings::default())
}

fn build_generic(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    chart(ctx, ToneMapping::Generic, PostProcessSettings::default())
}

/// Straight down, orthographic, framing the chart.
fn top_camera() -> Vec<NamedCamera> {
    let mut camera = orbit_camera(Vec3::ZERO, 8.0, 0.0, 0.0);
    camera.projection = Projection::orthographic(6.0);
    camera.aspect = 4.0 / 3.0;
    vec![NamedCamera {
        name: "top",
        camera,
    }]
}

/// The operator chart scenes.
pub fn scenes() -> Vec<NamedScene> {
    let scene = |name, build| NamedScene {
        name,
        cameras: top_camera(),
        build,
    };
    vec![
        scene("tone_agx", build_agx as fn(&mut BuildCtx<'_>) -> BuiltScene),
        scene("tone_agx_punchy", build_agx_punchy),
        scene("tone_aces_rrt", build_aces_rrt),
        scene("tone_generic", build_generic),
    ]
}
