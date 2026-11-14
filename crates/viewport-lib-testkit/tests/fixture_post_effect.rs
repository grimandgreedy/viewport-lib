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
use viewport_lib::Colour;
use viewport_lib::plugin_api::post_effect::stage_order;
use viewport_lib::wgpu;
use viewport_lib::{Material, MeshItem, PostEffectSlot, SurfaceSubmission};
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

/// A frame with one bright quad filling the middle of the view: unlit, or
/// with `fill_lit` lit by the hemisphere fill alone.
fn quad_frame(harness: &mut Harness, fill_lit: bool) -> viewport_lib::FrameData {
    let mesh = harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &probe_quad())
        .expect("upload probe quad");
    let mut frame = probe_frame(SIZE, [0.1, 0.1, 0.1, 1.0]);
    let mut item = MeshItem::default();
    item.mesh_id = mesh;
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(0.8)).into();
    item.material = harness
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8)))
        .expect("material upload");
    if fill_lit {
        // Ambient occlusion darkens the indirect share of the light, so a
        // quad lit by the hemisphere fill alone takes all of it.
        frame.effects.lighting.lights.clear();
    } else {
        item.settings.unlit = true;
    }
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
        )))
        .unwrap();

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
            harness
                .renderer
                .add_post_effect_producer(Box::new(
                    LoggingPostEffectProducer::new(
                        CallLog::new(),
                        PostEffectSlot::AmbientOcclusion,
                        Arc::new(AtomicBool::new(true)),
                    )
                    .filling(0.25),
                ))
                .unwrap();
        }
        let frame = quad_frame(&mut harness, true);
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
            harness
                .renderer
                .add_post_effect_stage(
                    Box::new(PassthroughPostEffectStage::new(
                        &device,
                        log.clone(),
                        "half",
                        0.5,
                        Arc::new(AtomicBool::new(true)),
                    )),
                    stage_order::EXTERNAL_DEFAULT,
                )
                .unwrap();
        }
        let frame = quad_frame(&mut harness, false);
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
    harness
        .renderer
        .add_post_effect_stage(
            Box::new(PassthroughPostEffectStage::new(
                &device,
                log.clone(),
                "second",
                0.5,
                second_enabled.clone(),
            )),
            stage_order::EXTERNAL_DEFAULT + 10,
        )
        .unwrap();
    harness
        .renderer
        .add_post_effect_stage(
            Box::new(PassthroughPostEffectStage::new(
                &device,
                log.clone(),
                "first",
                1.0,
                Arc::new(AtomicBool::new(true)),
            )),
            stage_order::EXTERNAL_DEFAULT,
        )
        .unwrap();

    let frame = quad_frame(&mut harness, false);
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

/// Releasing an idle target group and promoting it again keeps the scene
/// colour and depth in place, so a producer runs every frame through both
/// without a resize signal, and still gets one when the viewport is resized.
#[test]
fn producer_fixture_survives_a_target_release() {
    let Some((device, queue)) = viewport_lib_testkit::headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut harness = Harness::from_device(
        device.clone(),
        queue.clone(),
        wgpu::TextureFormat::Bgra8UnormSrgb,
    );
    harness.renderer = viewport_lib::ViewportRenderer::with_config(
        &device,
        &viewport_lib::RendererConfig::new(wgpu::TextureFormat::Bgra8UnormSrgb)
            .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking)
            .with_target_release_frames(2),
    );
    let log = CallLog::new();
    harness
        .renderer
        .add_post_effect_producer(Box::new(LoggingPostEffectProducer::new(
            log.clone(),
            PostEffectSlot::AmbientOcclusion,
            Arc::new(AtomicBool::new(true)),
        )))
        .unwrap();
    let mut bloom = probe_frame(SIZE, [0.3, 0.3, 0.3, 1.0]);
    bloom.effects.post_process.bloom.enabled = true;
    let plain = probe_frame(SIZE, [0.3, 0.3, 0.3, 1.0]);
    let per_frame = ["prepare:0".to_string(), format!("encode:0:{SIZE}x{SIZE}")];
    let per_frame: Vec<&str> = per_frame.iter().map(String::as_str).collect();
    let bloom_held = |harness: &Harness| {
        harness
            .renderer
            .debug_viewport_live_targets(0)
            .contains(&"bloom")
    };

    harness.render(&bloom, SIZE, SIZE);
    log.assert_take(&[
        "init_gpu",
        &format!("resize:0:{SIZE}x{SIZE}"),
        per_frame[0],
        per_frame[1],
    ]);
    for _ in 0..2 {
        harness.render(&plain, SIZE, SIZE);
        log.assert_take(&per_frame);
    }
    assert!(!bloom_held(&harness), "bloom must be released by now");
    harness.render(&bloom, SIZE, SIZE);
    log.assert_take(&per_frame);
    assert!(bloom_held(&harness), "bloom must be promoted again");

    let resized = SIZE * 2;
    let mut bigger = probe_frame(resized, [0.3, 0.3, 0.3, 1.0]);
    bigger.effects.post_process.bloom.enabled = true;
    harness.render(&bigger, resized, resized);
    log.assert_take(&[
        &format!("resize:0:{resized}x{resized}"),
        "prepare:0",
        &format!("encode:0:{resized}x{resized}"),
    ]);
}

/// An external producer is untouched by a target-group demotion: with
/// bloom's group released after it goes idle, the producer still prepares and
/// encodes every frame, and a later resize still reaches it.
#[test]
fn producer_fixture_survives_a_group_demotion() {
    let Some((device, queue)) = viewport_lib_testkit::headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let config = viewport_lib::RendererConfig::new(wgpu::TextureFormat::Rgba8UnormSrgb)
        .with_pipeline_compilation(viewport_lib::PipelineCompilation::Blocking)
        .with_target_release_frames(2);
    let mut renderer = viewport_lib::ViewportRenderer::with_config(&device, &config);
    let log = CallLog::new();
    renderer
        .add_post_effect_producer(Box::new(LoggingPostEffectProducer::new(
            log.clone(),
            PostEffectSlot::Bloom,
            Arc::new(AtomicBool::new(true)),
        )))
        .unwrap();
    let mut frame = probe_frame(SIZE, [0.3, 0.3, 0.3, 1.0]);
    frame.effects.post_process.bloom.enabled = true;
    renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert!(renderer.debug_viewport_live_targets(0).contains(&"bloom"));
    log.assert_take(&[
        "init_gpu",
        &format!("resize:0:{SIZE}x{SIZE}"),
        "prepare:0",
        &format!("encode:0:{SIZE}x{SIZE}"),
    ]);

    frame.effects.post_process.bloom.enabled = false;
    for _ in 0..2 {
        renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
        log.assert_take(&["prepare:0", &format!("encode:0:{SIZE}x{SIZE}")]);
    }
    assert!(
        !renderer.debug_viewport_live_targets(0).contains(&"bloom"),
        "test premise: the bloom group must be released"
    );
    renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    log.assert_take(&["prepare:0", &format!("encode:0:{SIZE}x{SIZE}")]);

    let big = SIZE * 2;
    let mut resized = probe_frame(big, [0.3, 0.3, 0.3, 1.0]);
    resized.effects.post_process.bloom.enabled = false;
    renderer.render_offscreen(&device, &queue, &resized, big, big);
    log.assert_take(&[
        &format!("resize:0:{big}x{big}"),
        "prepare:0",
        &format!("encode:0:{big}x{big}"),
    ]);
}
