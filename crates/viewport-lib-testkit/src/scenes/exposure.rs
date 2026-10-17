//! Exposure scenes: one chart rendered at each exposure mode other than the
//! default passthrough (which every other scene covers).
//!
//! The chart sits on a ground plane that fills the frame, seen from straight
//! above, under full daylight unless a scene says otherwise. The top row is
//! Lambert cards from 3% to 90% reflectance; the bottom row is emitters one
//! stop apart, centred on the luminance an 18% card reflects under the key
//! light. Where each row lands on the tone curve shows the exposure
//! multiplier, for lit surfaces and emitters alike.

use glam::{Mat4, Vec3};
use viewport_lib::{
    AutoExposure, Colour, ExposureMode, ExposureSettings, LightingSettings, Material, Projection,
    primitives,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, item_model, orbit_camera, upload};

const REFLECTANCES: [f32; 6] = [0.03, 0.09, 0.18, 0.36, 0.6, 0.9];
const CARD: f32 = 0.9;
const COL_STEP: f32 = 1.2;

/// A diffuse card: no specular, so it reflects `albedo * E / pi`.
fn lambert(albedo: f32) -> Material {
    let mut m = Material::from_colour(Colour::linear_rgb(albedo, albedo, albedo));
    m.specular = 0.0;
    m
}

fn chart(
    ctx: &mut BuildCtx<'_>,
    mut lighting: LightingSettings,
    exposure: ExposureSettings,
) -> BuiltScene {
    let ground = upload(ctx, &primitives::plane(40.0, 40.0));
    let card = upload(ctx, &primitives::plane(CARD, CARD));

    for light in &mut lighting.lights {
        light.cast_shadows = false;
    }
    // What an 18% card reflects under the key light at normal incidence, in
    // nits.
    let sun = lighting.lights[0].intensity;
    let grey_nits = 0.18 * sun / std::f32::consts::PI;

    // A light ground: auto-exposure meters mostly ground, and a light one keeps
    // the brighter cards below white once it is exposed to its target.
    let mut items = vec![item_model(ctx, ground, Mat4::IDENTITY, lambert(0.5))];
    let x = |i: usize| (i as f32 - 2.5) * COL_STEP;
    for (i, &albedo) in REFLECTANCES.iter().enumerate() {
        let at = Mat4::from_translation(Vec3::new(x(i), 0.75, 0.01));
        items.push(item_model(ctx, card, at, lambert(albedo)));
    }
    for i in 0..6 {
        let mut m = lambert(0.0);
        m.emissive = Colour::linear_rgb(1.0, 1.0, 1.0);
        m.emissive_strength = grey_nits * 2f32.powi(i as i32 - 3);
        let at = Mat4::from_translation(Vec3::new(x(i), -0.75, 0.01));
        items.push(item_model(ctx, card, at, m));
    }

    BuiltScene {
        items,
        lighting,
        exposure: Some(exposure),
        ..Default::default()
    }
}

fn build_manual_ev15(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    chart(
        ctx,
        LightingSettings::daylight(),
        ExposureSettings::manual(15.0),
    )
}

fn build_physical_sunny16(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    chart(
        ctx,
        LightingSettings::daylight(),
        ExposureSettings::physical(16.0, 1.0 / 125.0, 100.0),
    )
}

fn build_auto(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    chart(
        ctx,
        LightingSettings::daylight(),
        ExposureSettings::automatic(),
    )
}

/// Half adaptation: the exposure moves halfway from the passthrough anchor to
/// the metered target, so this is the scene that shows where the anchor sits.
/// Under the default lighting, the near-neutral case partial adaptation is
/// meant for (in daylight it leaves the frame several stops overexposed).
fn build_auto_partial(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut auto = AutoExposure::default();
    auto.adaptation = 0.5;
    chart(
        ctx,
        LightingSettings::default(),
        ExposureSettings::from_mode(ExposureMode::Automatic(auto)),
    )
}

/// Straight down, orthographic, so the chart fills the frame evenly.
fn top_camera() -> Vec<NamedCamera> {
    let mut camera = orbit_camera(Vec3::ZERO, 7.5, 0.0, 0.0);
    camera.projection = Projection::Orthographic;
    camera.aspect = 4.0 / 3.0;
    vec![NamedCamera {
        name: "top",
        camera,
    }]
}

/// The exposure scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "exposure_manual_ev15",
            cameras: top_camera(),
            build: build_manual_ev15,
        },
        NamedScene {
            name: "exposure_physical_sunny16",
            cameras: top_camera(),
            build: build_physical_sunny16,
        },
        NamedScene {
            name: "exposure_auto",
            cameras: top_camera(),
            build: build_auto,
        },
        NamedScene {
            name: "exposure_auto_partial",
            cameras: top_camera(),
            build: build_auto_partial,
        },
    ]
}
