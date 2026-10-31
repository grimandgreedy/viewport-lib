//! A built-in operator baked into a view-transform LUT and applied through
//! `ToneMapping::Custom` matches the operator itself.
//!
//! The ramp strip (one pixel per value from 2^-8 to 2^8 across six tints)
//! renders into a half-float target under the built-in operator and under
//! its bake (32 slices unless noted), and every pixel is compared after sRGB
//! encoding.

use viewport_lib::{
    OperatorCurve, PostProcessSettings, ToneMapping, WorkingSpace, bake_view_transform_lut,
};
use viewport_lib_testkit::scenes::tone_mapping::{RAMP_HEIGHT, RAMP_WIDTH, ramp_scene};
use viewport_lib_testkit::{Harness, frame_for};

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

#[test]
fn baked_operators_match_the_built_in_ones() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = ramp_scene();
    let built = h.build_scene(&scene);
    let camera = &scene.cameras[0].camera;

    let default = PostProcessSettings::default;
    let mut wide = default();
    wide.working_space = WorkingSpace::Rec2020;
    // (name, operator, settings, LUT slices, tolerance in 8-bit sRGB steps).
    // A per-channel curve bakes to within a step or three. Khronos Neutral
    // scales by its peak channel, so trilinear interpolation across the
    // grey axis pulls near-white greys down at its compression knee; the
    // error shrinks only linearly with the slice count.
    let cases: Vec<(&str, ToneMapping, PostProcessSettings, u32, f32)> = vec![
        ("reinhard", ToneMapping::Reinhard, default(), 32, 1.5),
        (
            "khronos_neutral",
            ToneMapping::KhronosNeutral,
            default(),
            32,
            17.0,
        ),
        (
            "khronos_neutral_64",
            ToneMapping::KhronosNeutral,
            default(),
            64,
            10.0,
        ),
        ("reinhard_rec2020", ToneMapping::Reinhard, wide, 32, 3.0),
        ("aces", ToneMapping::Aces, default(), 32, 4.0),
        ("agx", ToneMapping::Agx, default(), 32, 4.0),
        ("generic", ToneMapping::Generic, default(), 32, 4.0),
    ];

    let render = |h: &mut Harness, operator: ToneMapping, post: &PostProcessSettings| {
        let mut frame = frame_for(&built, camera, [RAMP_WIDTH as f32, RAMP_HEIGHT as f32]);
        frame.effects.display.operator = operator;
        frame.effects.post_process = post.clone();
        let _ = h.render_float(&frame, RAMP_WIDTH, RAMP_HEIGHT);
        let pixels = h.render_float(&frame, RAMP_WIDTH, RAMP_HEIGHT);
        let row = (RAMP_HEIGHT / 2) as usize * RAMP_WIDTH as usize;
        pixels[row..row + RAMP_WIDTH as usize].to_vec()
    };

    let mut failures = Vec::new();
    for (name, operator, post, size, tolerance) in cases {
        let lut = bake_view_transform_lut(&OperatorCurve::new(operator, &post), None, size);
        let id = h
            .renderer
            .resources_mut()
            .upload_texture(&h.device, &h.queue, lut)
            .expect("view LUT upload");
        let reference = render(&mut h, operator, &post);
        let custom = render(&mut h, ToneMapping::Custom { lut: id }, &post);

        let mut worst = (0.0f32, 0usize);
        let mut worst_linear = 0.0f32;
        for (i, (a, b)) in reference.iter().zip(&custom).enumerate() {
            for c in 0..3 {
                worst_linear = worst_linear.max((a[c] - b[c]).abs());
                let d = (srgb_steps(a[c]) - srgb_steps(b[c])).abs();
                if d > worst.0 {
                    worst = (d, i);
                }
            }
        }
        let i = worst.1;
        eprintln!(
            "{name}: max {:.3} 8-bit steps at pixel {i} (built-in {:?}, baked {:?}); max {:.5} linear",
            worst.0,
            &reference[i][..3],
            &custom[i][..3],
            worst_linear,
        );
        if worst.0 > tolerance {
            failures.push(format!("{name}: {:.3} steps at pixel {i}", worst.0));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A custom operator whose LUT id names no texture applies no operator.
#[test]
fn missing_view_lut_passes_through() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = ramp_scene();
    let built = h.build_scene(&scene);
    let camera = &scene.cameras[0].camera;
    let mut frame = frame_for(&built, camera, [RAMP_WIDTH as f32, RAMP_HEIGHT as f32]);
    frame.effects.display.operator = ToneMapping::Custom {
        lut: viewport_lib::TextureId::INVALID,
    };
    let _ = h.render_float(&frame, RAMP_WIDTH, RAMP_HEIGHT);
    let pixels = h.render_float(&frame, RAMP_WIDTH, RAMP_HEIGHT);
    let row = (RAMP_HEIGHT / 2) as usize * RAMP_WIDTH as usize;
    let i = RAMP_WIDTH as usize - 1;
    let want = viewport_lib_testkit::scenes::tone_mapping::ramp_value(i as u32);
    let got = pixels[row + i];
    for c in 0..3 {
        assert!(
            (got[c] - want[c]).abs() <= want[c] * 2e-3,
            "{got:?} {want:?}"
        );
    }
}
