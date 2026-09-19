//! Scatter-volume paths a single golden image cannot cover.
//!
//! Most of the scatter passes are goldened in the catalogue, including the two
//! that are functions of the animation clock: `ScatterSettings::time_seconds`
//! is a consumer input, so pinning it makes scrolling noise and the refraction
//! shimmer reproducible (see the `scatter_animated` scene).
//!
//! What is left here is what one recorded frame cannot express. Temporal
//! accumulation blends against a history slot built by previous frames, so it
//! is a property of a sequence rather than of a frame. Resizing and toggling
//! the downsample mode reallocate the per-viewport targets, and what matters
//! there is that the bind groups over the scene's own attachments get rebuilt,
//! which shows up as a validation error rather than as pixels. The refraction
//! test stays alongside its golden because it asserts something different and
//! backend-independent: that turning the pass on changes the image at all.
//!
//! Each test compares against the same scene rendered with the feature off,
//! rather than against a recorded image.

use glam::Vec3;
use viewport_lib::{
    Aabb, Material, RefractionParams, ScatterQuality, ScatterSettings, ScatterVolume,
    ScatterVolumeItem, primitives,
};
use viewport_lib_testkit::{Harness, scenes::BuiltScene};

const W: u32 = 200;
const H: u32 = 150;

/// Number of pixels whose value differs between two float frames.
fn pixels_moved(a: &[[f32; 4]], b: &[[f32; 4]]) -> u64 {
    a.iter()
        .zip(b.iter())
        .filter(|(pa, pb)| (0..4).any(|c| pa[c].to_bits() != pb[c].to_bits()))
        .count() as u64
}

/// A backdrop slab with one dense volume in front of it, so a volume pass has
/// both geometry to absorb and screen area to cover.
fn scene(harness: &mut Harness, refraction: bool) -> BuiltScene {
    let slab = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &primitives::cuboid(8.0, 0.4, 6.0))
        .expect("slab upload");
    let mut wall = viewport_lib::SceneRenderItem::default();
    wall.mesh_id = slab;
    wall.model = glam::Mat4::from_translation(Vec3::new(0.0, 2.5, 0.0)).to_cols_array_2d();
    wall.material = Material::pbr([0.7, 0.35, 0.3], 0.0, 0.8);

    let mut volume = ScatterVolume::box_uniform(
        Aabb {
            min: Vec3::new(-1.6, -1.2, -1.6),
            max: Vec3::new(1.6, 1.2, 1.6),
        },
        0.9,
        [0.8, 0.85, 1.0],
    );
    if refraction {
        let mut params = RefractionParams::default();
        params.strength = 0.03;
        volume.refraction = Some(params);
    }

    BuiltScene {
        items: vec![wall],
        scatter_volumes: vec![ScatterVolumeItem::new(volume)],
        ..Default::default()
    }
}

fn settings(temporal: bool, downsample: bool) -> ScatterSettings {
    let mut s = ScatterSettings::default();
    s.temporal = temporal;
    s.temporal_blend = 0.85;
    s.downsample = downsample;
    s.blue_noise_jitter = false;
    s.quality = ScatterQuality::High;
    s
}

/// The refraction pass distorts the scene colour behind the volume, so turning
/// it on has to move pixels that the plain scatter pass leaves alone.
#[test]
fn refraction_changes_the_image() {
    let Some(mut harness) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let camera = viewport_lib_testkit::scenes::orbit_camera(Vec3::ZERO, 7.0, 0.6, 0.9);

    let mut plain = scene(&mut harness, false);
    plain.scatter_settings = Some(settings(false, false));
    let plain_frame = viewport_lib_testkit::frame_for(&plain, &camera, [W as f32, H as f32]);
    let without = harness.render_float(&plain_frame, W, H);

    let mut refracted = scene(&mut harness, true);
    refracted.scatter_settings = Some(settings(false, false));
    let refracted_frame =
        viewport_lib_testkit::frame_for(&refracted, &camera, [W as f32, H as f32]);
    let with = harness.render_float(&refracted_frame, W, H);

    let moved = pixels_moved(&without, &with);
    assert!(
        moved > 200,
        "refraction moved only {moved} of {} pixels, so the pass did not run",
        (W * H) as u64
    );
}

/// Temporal accumulation blends each frame against a history slot the
/// ping-pong swaps every frame, so a frame rendered with a populated history
/// has to differ from the same frame rendered without one.
///
/// The history is populated by rendering a different camera first. Blending a
/// still scene against itself would return the same image whatever the resolve
/// pass did, and the jitter that normally varies a still frame is driven by the
/// frame counter, which only advances for presented frames: an offscreen render
/// deliberately leaves the temporal phase alone. Moving the camera is what
/// gives the blend two different images to combine.
#[test]
fn temporal_blend_mixes_in_the_history() {
    let Some(mut harness) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let near = viewport_lib_testkit::scenes::orbit_camera(Vec3::ZERO, 7.0, 0.6, 0.9);
    let away = viewport_lib_testkit::scenes::orbit_camera(Vec3::ZERO, 7.0, 0.6, 2.4);

    let mut render = |temporal: bool| -> Vec<[f32; 4]> {
        let mut built = scene(&mut harness, false);
        built.scatter_settings = Some(settings(temporal, false));
        let first = viewport_lib_testkit::frame_for(&built, &away, [W as f32, H as f32]);
        let second = viewport_lib_testkit::frame_for(&built, &near, [W as f32, H as f32]);
        // The first camera fills the history slot; the second is the frame
        // under test, which the blend mixes that history into.
        let _ = harness.render_float(&first, W, H);
        harness.render_float(&second, W, H)
    };

    let plain = render(false);
    let blended = render(true);

    let moved = pixels_moved(&plain, &blended);
    assert!(
        moved > 200,
        "the temporal blend moved only {moved} of {} pixels, so the resolve \
         pass did not mix in the previous frame",
        (W * H) as u64
    );
}

/// The per-viewport targets are keyed by size and downsample mode. Toggling
/// either reallocates them, and the bind groups over the scene's own
/// attachments have to be rebuilt rather than reused, which is what a
/// validation error would catch here.
#[test]
fn resize_and_downsample_toggle_survive() {
    let Some(mut harness) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let camera = viewport_lib_testkit::scenes::orbit_camera(Vec3::ZERO, 7.0, 0.6, 0.9);

    for (temporal, downsample, w, h) in [
        (false, false, W, H),
        (false, true, W, H),
        (true, true, W, H),
        (true, true, W + 64, H + 48),
        (true, false, W + 64, H + 48),
        (false, false, W, H),
    ] {
        let mut built = scene(&mut harness, true);
        built.scatter_settings = Some(settings(temporal, downsample));
        let frame = viewport_lib_testkit::frame_for(&built, &camera, [w as f32, h as f32]);
        let pixels = harness.render_float(&frame, w, h);
        assert_eq!(
            pixels.len(),
            (w * h) as usize,
            "readback size mismatch at {w}x{h}"
        );
    }
}
