//! Colour-grading scenes: a chart of grey and hue cards graded by a strong
//! teal-orange LUT in scene space, at unit scale and in daylight under a
//! sunny-16 camera.
//!
//! The chart sits on a plate smaller than the frame, so the background shows
//! around it: a scene-referred grade leaves the background colour alone.
//! The LUT builders are public so tests can grade with the same tables.

use glam::{Mat4, Vec3};
use viewport_lib::{
    Colour, ExposureSettings, LightingSettings, Material, PostProcessSettings, TextureData,
    TextureId, primitives,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, item_model, orbit_camera, upload};

/// Slices (and slice size) of the grading LUTs.
pub const LUT_SIZE: u32 = 16;

const CARD: f32 = 0.9;
const COL_STEP: f32 = 1.1;
const GREYS: [f32; 6] = [0.03, 0.09, 0.18, 0.36, 0.6, 0.9];
const HUES: [[f32; 3]; 6] = [
    [0.8, 0.1, 0.08],
    [0.85, 0.45, 0.08],
    [0.75, 0.75, 0.1],
    [0.12, 0.6, 0.15],
    [0.08, 0.5, 0.75],
    [0.55, 0.15, 0.7],
];

/// A strip LUT of `n` slices of `n` x `n` (width `n * n`, height `n`), as
/// float RGBA. `f` maps a lattice coordinate in `[0, 1]` to the value stored
/// there.
pub fn strip_lut(n: u32, f: impl Fn([f32; 3]) -> [f32; 3]) -> Vec<f32> {
    let unit = |v: u32| v as f32 / (n - 1) as f32;
    let mut out = Vec::with_capacity((n * n * n * 4) as usize);
    for g in 0..n {
        for b in 0..n {
            for r in 0..n {
                let c = f([unit(r), unit(g), unit(b)]);
                out.extend_from_slice(&[c[0], c[1], c[2], 1.0]);
            }
        }
    }
    out
}

/// A LUT whose texels hold their own coordinates: an identity in any space.
pub fn neutral_lut(n: u32) -> Vec<f32> {
    strip_lut(n, |c| c)
}

/// A teal-orange grade on shaped values: values a stop or more below
/// mid-grey lean teal, values above it lean orange, and saturation is raised
/// around the shaped luma.
pub fn teal_orange(s: [f32; 3]) -> [f32; 3] {
    let luma = 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2];
    // Mid-grey (0.18) shapes to about 0.48; 0.1 is 2.4 stops.
    let tone = ((luma - 0.48) / 0.1).clamp(-1.0, 1.0);
    let sat = |c: f32| luma + (c - luma) * 1.3;
    [
        sat(s[0]) + 0.04 * tone,
        sat(s[1]) + 0.01 * tone,
        sat(s[2]) - 0.05 * tone,
    ]
}

/// [`teal_orange`] as a strip LUT for the scene-referred shaper.
pub fn teal_orange_lut(n: u32) -> Vec<f32> {
    strip_lut(n, teal_orange)
}

/// Upload a float strip LUT of `n` slices.
pub fn upload_lut(ctx: &mut BuildCtx<'_>, n: u32, texels: Vec<f32>) -> TextureId {
    ctx.renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::hdr(n * n, n, texels))
        .expect("grade LUT upload")
}

/// A diffuse card: no specular, so it reflects `albedo * E / pi`.
fn lambert(rgb: [f32; 3]) -> Material {
    let mut m = Material::from_colour(Colour::linear_rgb(rgb[0], rgb[1], rgb[2]));
    m.specular = 0.0;
    m
}

/// Grey cards over hue cards on a mid-grey plate, graded by the teal-orange
/// LUT in scene space.
fn chart(
    ctx: &mut BuildCtx<'_>,
    mut lighting: LightingSettings,
    exposure: ExposureSettings,
) -> BuiltScene {
    let plate = upload(ctx, &primitives::plane(7.2, 2.8));
    let card = upload(ctx, &primitives::plane(CARD, CARD));
    for light in &mut lighting.lights {
        light.cast_shadows = false;
    }

    let mut items = vec![item_model(ctx, plate, Mat4::IDENTITY, lambert([0.3; 3]))];
    let x = |i: usize| (i as f32 - 2.5) * COL_STEP;
    for (i, &grey) in GREYS.iter().enumerate() {
        let at = Mat4::from_translation(Vec3::new(x(i), 0.6, 0.01));
        items.push(item_model(ctx, card, at, lambert([grey; 3])));
    }
    for (i, &hue) in HUES.iter().enumerate() {
        let at = Mat4::from_translation(Vec3::new(x(i), -0.6, 0.01));
        items.push(item_model(ctx, card, at, lambert(hue)));
    }

    let lut = upload_lut(ctx, LUT_SIZE, teal_orange_lut(LUT_SIZE));
    let mut post = PostProcessSettings::default();
    post.grade_lut = Some(lut);
    BuiltScene {
        items,
        lighting,
        exposure: Some(exposure),
        post_process: Some(post),
        background: Some([0.25, 0.3, 0.45, 1.0]),
        ..Default::default()
    }
}

/// The default lighting at unit scale.
fn build_unit(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    chart(
        ctx,
        LightingSettings::default(),
        ExposureSettings::default(),
    )
}

/// Full daylight under a sunny-16 camera.
fn build_sunny16(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    chart(
        ctx,
        LightingSettings::daylight(),
        ExposureSettings::physical(16.0, 1.0 / 125.0, 100.0),
    )
}

/// Straight down, orthographic, with a margin of background round the plate.
fn top_camera() -> Vec<NamedCamera> {
    let mut camera = orbit_camera(Vec3::ZERO, 8.0, 0.0, 0.0);
    camera.set_orthographic(true);
    camera.aspect = 4.0 / 3.0;
    vec![NamedCamera {
        name: "top",
        camera,
    }]
}

/// The grading scenes.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "grade_scene_teal_orange",
            cameras: top_camera(),
            build: build_unit,
        },
        NamedScene {
            name: "grade_scene_teal_orange_sunny16",
            cameras: top_camera(),
            build: build_sunny16,
        },
    ]
}
