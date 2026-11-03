//! Where the colour-grading LUT applies.
//!
//! A scene-referred grade reads exposed radiance through a log shaper before
//! the tone-map operator, so a neutral LUT changes nothing beyond the LUT's
//! own precision and the background colour is never graded. A
//! display-referred grade applies after the operator and the background
//! composite, so it grades the background too.
//!
//! The tests read back a half-float target, which holds the composite's
//! display values before any 8-bit encoding.

use viewport_lib::{FrameData, GradingSpace, TextureData, TextureId, ToneMapping};
use viewport_lib_testkit::scenes::grading::{LUT_SIZE, neutral_lut, teal_orange_lut};
use viewport_lib_testkit::{BuiltScene, Harness, frame_for, scene_by_name};

const W: u32 = 400;
const H: u32 = 300;

/// sRGB-encode a display value into 8-bit steps, unrounded.
fn srgb_steps(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    e * 255.0
}

/// The largest per-channel difference between two renders, in 8-bit sRGB
/// steps.
fn max_step_delta(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(p, q)| (0..3).map(move |c| (srgb_steps(p[c]) - srgb_steps(q[c])).abs()))
        .fold(0.0, f32::max)
}

/// The mean per-channel difference between two renders, in 8-bit sRGB steps.
fn mean_step_delta(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    let sum: f32 = a
        .iter()
        .zip(b)
        .flat_map(|(p, q)| (0..3).map(move |c| (srgb_steps(p[c]) - srgb_steps(q[c])).abs()))
        .sum();
    sum / (a.len() * 3) as f32
}

fn upload_lut(h: &mut Harness, texels: Vec<f32>) -> TextureId {
    h.renderer
        .resources_mut()
        .upload_texture(
            &h.device,
            &h.queue,
            TextureData::hdr(LUT_SIZE * LUT_SIZE, LUT_SIZE, texels),
        )
        .expect("LUT upload")
}

struct Chart {
    h: Harness,
    built: BuiltScene,
    camera: viewport_lib::Camera,
    operator: ToneMapping,
}

impl Chart {
    fn new(name: &str) -> Option<Self> {
        let mut h = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT)?;
        let scene = scene_by_name(name).expect("grading chart in the catalogue");
        let built = h.build_scene(&scene);
        Some(Self {
            h,
            built,
            camera: scene.cameras[0].camera.clone(),
            operator: ToneMapping::default(),
        })
    }

    fn frame(&self, lut: Option<TextureId>, space: GradingSpace) -> FrameData {
        let mut frame = frame_for(&self.built, &self.camera, [W as f32, H as f32]);
        frame.effects.display.operator = self.operator;
        frame.effects.post_process.grade_lut = lut;
        frame.effects.post_process.grading_space = space;
        frame
    }

    /// The settled second frame.
    fn render(&mut self, lut: Option<TextureId>, space: GradingSpace) -> Vec<[f32; 4]> {
        let frame = self.frame(lut, space);
        let _ = self.h.render_float(&frame, W, H);
        self.h.render_float(&frame, W, H)
    }
}

fn neutral_scene_lut_is_identity(name: &str, operator: ToneMapping) {
    let Some(mut chart) = Chart::new(name) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    chart.operator = operator;
    let neutral = upload_lut(&mut chart.h, neutral_lut(LUT_SIZE));
    let ungraded = chart.render(None, GradingSpace::Scene);
    let graded = chart.render(Some(neutral), GradingSpace::Scene);
    let delta = max_step_delta(&ungraded, &graded);
    eprintln!("{name} {operator:?}: neutral scene-referred LUT, max delta {delta} 8-bit steps");
    assert!(
        delta <= 1.0,
        "{name} {operator:?}: a neutral scene-referred LUT moved a channel by {delta} 8-bit steps"
    );
}

#[test]
fn neutral_scene_lut_is_identity_at_unit_scale() {
    neutral_scene_lut_is_identity("grade_scene_teal_orange", ToneMapping::KhronosNeutral);
}

/// Reinhard keeps the most of the shaper's floor near black.
#[test]
fn neutral_scene_lut_is_identity_under_reinhard() {
    neutral_scene_lut_is_identity("grade_scene_teal_orange", ToneMapping::Reinhard);
}

#[test]
fn neutral_scene_lut_is_identity_in_daylight() {
    neutral_scene_lut_is_identity(
        "grade_scene_teal_orange_sunny16",
        ToneMapping::KhronosNeutral,
    );
}

#[test]
fn scene_and_display_grades_differ() {
    let Some(mut chart) = Chart::new("grade_scene_teal_orange") else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let lut = upload_lut(&mut chart.h, teal_orange_lut(LUT_SIZE));
    let ungraded = chart.render(None, GradingSpace::Scene);
    let scene = chart.render(Some(lut), GradingSpace::Scene);
    let display = chart.render(Some(lut), GradingSpace::Display);

    let scene_vs_ungraded = mean_step_delta(&scene, &ungraded);
    let display_vs_ungraded = mean_step_delta(&display, &ungraded);
    let scene_vs_display = mean_step_delta(&scene, &display);
    assert!(
        scene_vs_ungraded > 2.0,
        "the scene-referred grade barely changed the image: {scene_vs_ungraded}"
    );
    assert!(
        display_vs_ungraded > 2.0,
        "the display-referred grade barely changed the image: {display_vs_ungraded}"
    );
    assert!(
        scene_vs_display > 2.0,
        "the two placements gave nearly the same image: {scene_vs_display}"
    );

    // A corner pixel is pure background: graded only in display space.
    let corner = (2 * W + 2) as usize;
    assert_eq!(
        scene[corner], ungraded[corner],
        "a scene-referred grade changed the background"
    );
    let moved = (0..3)
        .map(|c| (srgb_steps(display[corner][c]) - srgb_steps(ungraded[corner][c])).abs())
        .fold(0.0, f32::max);
    assert!(
        moved > 2.0,
        "a display-referred grade left the background alone: {:?} -> {:?}",
        ungraded[corner],
        display[corner]
    );
}
