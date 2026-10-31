//! Each tone-map operator on the GPU matches its CPU version.
//!
//! A strip of emissive cards, one pixel per known linear value from 2^-8 to
//! 2^8 across six tints, renders into a half-float target under each operator,
//! look and working space. Every pixel is compared with the CPU operator on
//! the same value, after sRGB encoding, within one 8-bit step.

use viewport_lib::vplt::effects::tone_operators::tone_map;
use viewport_lib::{AgxLook, GenericToneCurve, PostProcessSettings, ToneMapping, WorkingSpace};
use viewport_lib_testkit::scenes::tone_mapping::{RAMP_HEIGHT, RAMP_WIDTH, ramp_scene, ramp_value};
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
fn every_operator_matches_its_cpu_version() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = ramp_scene();
    let built = h.build_scene(&scene);
    let camera = &scene.cameras[0].camera;

    let looks = |look: AgxLook| {
        let mut post = PostProcessSettings::default();
        post.agx_look = look;
        post
    };
    let mut steep = PostProcessSettings::default();
    steep.generic_tone_curve = GenericToneCurve::new(2.0, 0.18, 0.25, 32.0);
    let mut wide = PostProcessSettings::default();
    wide.working_space = WorkingSpace::Rec2020;

    let cases: Vec<(&str, ToneMapping, PostProcessSettings)> = vec![
        (
            "reinhard",
            ToneMapping::Reinhard,
            PostProcessSettings::default(),
        ),
        ("aces", ToneMapping::Aces, PostProcessSettings::default()),
        (
            "khronos_neutral",
            ToneMapping::KhronosNeutral,
            PostProcessSettings::default(),
        ),
        ("agx", ToneMapping::Agx, looks(AgxLook::None)),
        ("agx_punchy", ToneMapping::Agx, looks(AgxLook::Punchy)),
        ("agx_golden", ToneMapping::Agx, looks(AgxLook::Golden)),
        (
            "aces_rrt",
            ToneMapping::AcesRrt,
            PostProcessSettings::default(),
        ),
        (
            "generic",
            ToneMapping::Generic,
            PostProcessSettings::default(),
        ),
        ("generic_steep", ToneMapping::Generic, steep),
        ("agx_rec2020", ToneMapping::Agx, wide.clone()),
        ("aces_rrt_rec2020", ToneMapping::AcesRrt, wide.clone()),
        ("reinhard_rec2020", ToneMapping::Reinhard, wide.clone()),
        ("generic_rec2020", ToneMapping::Generic, wide),
    ];

    let mut failures = Vec::new();
    for (name, operator, post) in cases {
        let mut frame = frame_for(&built, camera, [RAMP_WIDTH as f32, RAMP_HEIGHT as f32]);
        frame.effects.display.operator = operator;
        frame.effects.post_process = post.clone();
        let _ = h.render_float(&frame, RAMP_WIDTH, RAMP_HEIGHT);
        let pixels = h.render_float(&frame, RAMP_WIDTH, RAMP_HEIGHT);

        let row = (RAMP_HEIGHT / 2) as usize * RAMP_WIDTH as usize;
        let mut worst = (0.0f32, 0u32);
        for i in 0..RAMP_WIDTH {
            let gpu = pixels[row + i as usize];
            let cpu = tone_map(ramp_value(i), operator, &post);
            for c in 0..3 {
                let d = (srgb_steps(gpu[c]) - srgb_steps(cpu[c])).abs();
                if d > worst.0 {
                    worst = (d, i);
                }
            }
        }
        let i = worst.1;
        eprintln!(
            "{name}: max {:.3} 8-bit steps at pixel {i} (value {:?}: gpu {:?}, cpu {:?})",
            worst.0,
            ramp_value(i),
            &pixels[row + i as usize][..3],
            tone_map(ramp_value(i), operator, &post),
        );
        if worst.0 > 1.0 {
            failures.push(format!("{name}: {:.3} steps at pixel {i}", worst.0));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
