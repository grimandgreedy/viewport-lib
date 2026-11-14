//! Radiance above the half-float range does not reach the `Rgba16Float` scene
//! target as an infinity.
//!
//! A 1e6-nit emitter and a stack of bright transparent sheets, drawn under
//! passthrough exposure with bloom on, read back finite, and the bloom stays
//! a glow around the emitter rather than filling the frame.

use viewport_lib::{
    Camera, Colour, ExposureSettings, Material, MeshItem, PipelineMode, primitives,
};
use viewport_lib_testkit::{BuiltScene, Harness, frame_for};

const W: u32 = 160;
const H: u32 = 120;

/// Transparent sheets stacked towards the camera.
const LAYERS: usize = 12;

#[test]
fn bright_emissive_and_oit_stack_read_back_finite() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut scene = BuiltScene::default();
    let res = h.renderer.resources_mut();

    let mut sun = Material::pbr(Colour::BLACK, 0.0, 1.0);
    sun.emissive = Colour::WHITE;
    sun.emissive_strength = 1.0e6;
    let sun = res.upload_material(sun).expect("material upload");
    let sphere = res
        .upload_mesh_data(&h.device, &primitives::sphere(0.25, 24, 12))
        .expect("mesh upload");
    let mut emitter = MeshItem::new(sphere, sun);
    emitter.model = viewport_lib::Transform::from_translation([-0.9, 0.0, 0.0]);
    scene.items.push(emitter);

    let mut glow = Material::pbr(Colour::WHITE, 0.0, 0.5);
    glow.emissive = Colour::WHITE;
    glow.emissive_strength = 2.0e4;
    let glow = res.upload_material(glow).expect("material upload");
    let sheet = res
        .upload_mesh_data(&h.device, &primitives::plane(0.8, 0.8))
        .expect("mesh upload");
    for i in 0..LAYERS {
        let mut item = MeshItem::new(sheet, glow);
        item.settings.opacity = 0.5;
        item.model = viewport_lib::Transform::from_translation([0.7, 0.0, 0.05 * i as f64]);
        scene.items.push(item);
    }

    let camera = Camera::orbit([0.0, 0.0, 0.0], 3.0, glam::Quat::IDENTITY);
    let mut frame = frame_for(&scene, &camera, [W as f32, H as f32]);
    frame.effects.display.mode = PipelineMode::Hdr;
    frame.effects.display.exposure = ExposureSettings::passthrough();
    frame.effects.post_process.bloom.enabled = true;
    let _ = h.render_float(&frame, W, H);
    let px = h.render_float(&frame, W, H);

    let non_finite = px
        .iter()
        .filter(|p| p.iter().any(|c| !c.is_finite()))
        .count();
    assert_eq!(non_finite, 0, "{non_finite} pixels are NaN or infinite");

    // Both the emitter and the sheets drew: something reads bright.
    let at = |x: u32, y: u32| px[(y * W + x) as usize];
    let lum = |p: [f32; 4]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let emitter_px = at(W / 2 - (W as f32 * 0.9 / 3.3) as u32, H / 2);
    assert!(lum(emitter_px) > 0.9, "the emitter reads {emitter_px:?}");

    // The bloom is a glow, not a flood: the corners stay dark and most of the
    // frame is not white.
    for (x, y) in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)] {
        let p = at(x, y);
        assert!(lum(p) < 0.5, "corner ({x}, {y}) reads {p:?}");
    }
    let white = px.iter().filter(|p| lum(**p) > 0.95).count();
    assert!(
        white < px.len() / 4,
        "{white} of {} pixels are white",
        px.len()
    );
}
