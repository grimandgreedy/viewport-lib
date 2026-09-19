//! Smoke tests for the family-C post-effect fixtures.
//!
//! `LoggingPostEffectProducer` covers the producer lifecycle (deferred
//! `init_gpu`, the per-viewport resize signal, the per-frame `prepare` /
//! `encode` pair, the self-gate, removal) and slot contribution.
//! `PassthroughPostEffectStage` covers the chain: the composite is routed into
//! the stage's own input texture and the stage's output reaches the final
//! target, with the order key deciding chain position.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use viewport_lib::plugin_api::post_effect::stage_order;
use viewport_lib::wgpu;
use viewport_lib::{Material, PostEffectSlot, SceneRenderItem, SurfaceSubmission};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::fixtures::{
    CallLog, LoggingPostEffectProducer, PassthroughPostEffectStage, probe_frame, probe_quad,
};

const SIZE: u32 = 64;

/// Sum of the RGB bytes at a pixel, as a rough brightness.
fn luma(pixels: &[u8], x: u32, y: u32) -> i32 {
    let i = ((y * SIZE + x) * 4) as usize;
    pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
}

/// A frame with one bright unlit quad filling the middle of the view.
fn quad_frame(harness: &mut Harness) -> viewport_lib::FrameData {
    let mesh = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");
    let mut frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(0.8)).to_cols_array_2d();
    item.material = Material::from_colour([0.8, 0.8, 0.8]);
    item.settings.unlit = true;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());
    frame
}

/// The producer lifecycle, frame by frame: `init_gpu` and the resize signal
/// arrive on the first frame that runs with a live viewport, later frames are
/// just `prepare` + `encode`, the self-gate skips both, and removal silences
/// the fixture for good.
#[test]
fn producer_fixture_lifecycle() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    let enabled = Arc::new(AtomicBool::new(true));
    let id = harness
        .renderer
        .add_post_effect_producer(Box::new(LoggingPostEffectProducer::new(
            log.clone(),
            PostEffectSlot::Bloom,
            enabled.clone(),
        )));

    let frame = probe_frame(SIZE, [0.3, 0.3, 0.3, 1.0]);
    harness.render(&frame, SIZE, SIZE);
    log.assert_take(&[
        "init_gpu",
        &format!("resize:0:{SIZE}x{SIZE}"),
        "prepare:0",
        &format!("encode:0:{SIZE}x{SIZE}"),
    ]);

    harness.render(&frame, SIZE, SIZE);
    log.assert_take(&["prepare:0", &format!("encode:0:{SIZE}x{SIZE}")]);

    enabled.store(false, Ordering::Relaxed);
    harness.render(&frame, SIZE, SIZE);
    log.assert_empty();

    enabled.store(true, Ordering::Relaxed);
    harness.renderer.remove_post_effect_producer(id);
    harness.render(&frame, SIZE, SIZE);
    log.assert_empty();
}

/// A producer that returns a view fills its composite slot: the
/// ambient-occlusion slot filled with a dark value darkens the geometry even
/// with the built-in SSAO off.
#[test]
fn producer_fixture_fills_its_slot() {
    let Some((device, queue)) = viewport_lib_testkit::headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let render = |with_producer: bool| -> Vec<u8> {
        let mut harness = Harness::from_device(
            device.clone(),
            queue.clone(),
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        if with_producer {
            harness.renderer.add_post_effect_producer(Box::new(
                LoggingPostEffectProducer::new(
                    CallLog::new(),
                    PostEffectSlot::AmbientOcclusion,
                    Arc::new(AtomicBool::new(true)),
                )
                .filling(0.25),
            ));
        }
        let frame = quad_frame(&mut harness);
        harness.render(&frame, SIZE, SIZE)
    };

    let base = render(false);
    let occluded = render(true);
    let base_centre = luma(&base, SIZE / 2, SIZE / 2);
    let ao_centre = luma(&occluded, SIZE / 2, SIZE / 2);
    assert!(
        ao_centre < base_centre - 60,
        "the filled AO slot must darken the geometry: base {base_centre}, with producer {ao_centre}"
    );
}

/// A stage in the chain: its lifecycle mirrors the producer's, and its output
/// is in the image path, so a stage that scales its input down darkens the
/// whole frame.
#[test]
fn stage_fixture_is_in_the_image_path() {
    let Some((device, queue)) = viewport_lib_testkit::headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    let render = |log: &CallLog, with_stage: bool| -> Vec<u8> {
        let mut harness = Harness::from_device(
            device.clone(),
            queue.clone(),
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        if with_stage {
            harness.renderer.add_post_effect_stage(
                Box::new(PassthroughPostEffectStage::new(
                    &device,
                    log.clone(),
                    "half",
                    0.5,
                    Arc::new(AtomicBool::new(true)),
                )),
                stage_order::EXTERNAL_DEFAULT,
            );
        }
        let frame = quad_frame(&mut harness);
        harness.render(&frame, SIZE, SIZE)
    };

    let base = render(&log, false);
    log.assert_empty();
    let staged = render(&log, true);

    let entries = log.entries();
    assert!(
        entries.starts_with(&[
            "half:init_gpu".to_string(),
            format!("half:resize:0:{SIZE}x{SIZE}"),
            "half:prepare:0".to_string(),
            "half:encode:0".to_string(),
        ]),
        "stage lifecycle must run init, resize, then the per-frame pair; log holds {entries:?}"
    );

    let base_centre = luma(&base, SIZE / 2, SIZE / 2);
    let staged_centre = luma(&staged, SIZE / 2, SIZE / 2);
    assert!(
        staged_centre < base_centre - 100,
        "the stage's output must reach the final target: base {base_centre}, staged {staged_centre}"
    );
}

/// Two stages chain in order-key order, not registration order: the second
/// stage's input is the first stage's output, so both scales multiply. The
/// disabled stage drops out of the chain entirely.
#[test]
fn stage_fixtures_chain_in_order_key_order() {
    let Some((device, queue)) = viewport_lib_testkit::headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    let second_enabled = Arc::new(AtomicBool::new(true));
    let mut harness = Harness::from_device(
        device.clone(),
        queue.clone(),
        wgpu::TextureFormat::Rgba8UnormSrgb,
    );

    // Registered in reverse of chain order so the ordering cannot come from
    // registration order.
    harness.renderer.add_post_effect_stage(
        Box::new(PassthroughPostEffectStage::new(
            &device,
            log.clone(),
            "second",
            0.5,
            second_enabled.clone(),
        )),
        stage_order::EXTERNAL_DEFAULT + 10,
    );
    harness.renderer.add_post_effect_stage(
        Box::new(PassthroughPostEffectStage::new(
            &device,
            log.clone(),
            "first",
            1.0,
            Arc::new(AtomicBool::new(true)),
        )),
        stage_order::EXTERNAL_DEFAULT,
    );

    let frame = quad_frame(&mut harness);
    let both = harness.render(&frame, SIZE, SIZE);
    log.assert_before("first:encode:0", "second:encode:0");

    log.clear();
    second_enabled.store(false, Ordering::Relaxed);
    let one = harness.render(&frame, SIZE, SIZE);
    assert_eq!(
        log.count("second:encode:0"),
        0,
        "a disabled stage must drop out of the chain; log holds {:?}",
        log.entries()
    );
    assert_eq!(log.count("first:encode:0"), 1);

    let both_centre = luma(&both, SIZE / 2, SIZE / 2);
    let one_centre = luma(&one, SIZE / 2, SIZE / 2);
    assert!(
        both_centre < one_centre - 100,
        "the halving stage must apply on top of the pass-through: both {both_centre}, one {one_centre}"
    );
}
