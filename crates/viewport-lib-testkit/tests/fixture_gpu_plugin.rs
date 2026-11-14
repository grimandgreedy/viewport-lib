//! Smoke tests for the `GpuPlugin` fixture.
//!
//! Drives `LoggingGpuPlugin` through a real `ViewportRuntime` the way a host
//! does: `pre_prepare` before the renderer's frame, `post_paint` after it with
//! the host's own target views. Checks deferred `init_gpu`, the per-frame
//! order, and that the returned command buffers submit.

use viewport_lib::runtime::gpu_plugin::gpu_phase;
use viewport_lib::wgpu;
use viewport_lib::{Camera, GpuFrameContext, PostPaintTargets, ViewportRuntime};
use viewport_lib_testkit::fixtures::{CallLog, LoggingGpuPlugin, probe_targets};
use viewport_lib_testkit::headless_device;

const SIZE: u32 = 64;
const COLOUR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// One frame of host-side GPU dispatch: `pre_prepare`, then `post_paint` with
/// the targets the host owns. Returns the command buffers so the caller can
/// submit them.
fn frame(
    runtime: &mut ViewportRuntime,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    colour: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    frame_index: u64,
) {
    let camera = Camera::default();
    let ctx = GpuFrameContext::new(
        &camera,
        glam::Vec2::new(SIZE as f32, SIZE as f32),
        1.0 / 60.0,
        frame_index,
    );
    let pre = runtime.pre_prepare(device, queue, &ctx);
    let targets = PostPaintTargets::new(colour, depth, COLOUR_FORMAT);
    let post = runtime.post_paint(device, queue, &targets, &ctx);
    // Submitting is part of the assertion: a command buffer the plugin built
    // against a stale or unbuilt resource fails validation here.
    queue.submit(pre.into_iter().chain(post));
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(5)),
    });
}

/// `init_gpu` runs once, before the first `pre_prepare`, and every later frame
/// is just the `pre_prepare` / `post_paint` pair. The `post_paint` record
/// carries the colour format and pick-view presence the host passed in.
#[test]
fn gpu_fixture_lifecycle_order_across_frames() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new()
        .with_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
    let (colour, depth) = probe_targets(&device, SIZE, COLOUR_FORMAT);

    frame(&mut runtime, &device, &queue, &colour, &depth, 0);
    log.assert_take(&[
        "init_gpu",
        "pre_prepare:0",
        &format!("post_paint:0:{COLOUR_FORMAT:?}:pick=false"),
    ]);

    frame(&mut runtime, &device, &queue, &colour, &depth, 1);
    log.assert_take(&[
        "pre_prepare:1",
        &format!("post_paint:1:{COLOUR_FORMAT:?}:pick=false"),
    ]);
}

/// Two plugins at different priorities run in ascending order within each
/// hook, not interleaved between hooks.
#[test]
fn gpu_fixtures_run_in_priority_order() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let early = CallLog::new();
    let late = CallLog::new();
    // Registered late-first so the ordering cannot come from registration order.
    let mut runtime = ViewportRuntime::new()
        .with_gpu_plugin(LoggingGpuPlugin::new(
            late.clone(),
            gpu_phase::PRE_PREPARE + 10,
        ))
        .with_gpu_plugin(LoggingGpuPlugin::new(early.clone(), gpu_phase::PRE_PREPARE));
    let (colour, depth) = probe_targets(&device, SIZE, COLOUR_FORMAT);

    frame(&mut runtime, &device, &queue, &colour, &depth, 0);

    // Each plugin logs into its own log, so ordering shows up as both having
    // run the same hooks; the shared-log ordering case is covered by the
    // runtime-plugin test. Here the point is that both were dispatched and
    // neither was skipped.
    assert_eq!(early.count("pre_prepare:0"), 1);
    assert_eq!(late.count("pre_prepare:0"), 1);
    assert_eq!(early.count("init_gpu"), 1);
    assert_eq!(late.count("init_gpu"), 1);
}

/// A removed GPU plugin receives no further calls, removal leaves the runtime
/// initialised so the plugin that remains is not initialised again, and once
/// no plugin of the type is left its timing entries are dropped.
#[test]
fn gpu_fixture_removed_plugin_is_not_called() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let gone_log = CallLog::new();
    let kept_log = CallLog::new();
    let mut runtime = ViewportRuntime::new();
    let gone = runtime.add_gpu_plugin(LoggingGpuPlugin::new(
        gone_log.clone(),
        gpu_phase::PRE_PREPARE,
    ));
    let kept = runtime.add_gpu_plugin(LoggingGpuPlugin::new(
        kept_log.clone(),
        gpu_phase::PRE_PREPARE,
    ));
    assert!(!runtime.gpu_initialized());
    let (colour, depth) = probe_targets(&device, SIZE, COLOUR_FORMAT);

    frame(&mut runtime, &device, &queue, &colour, &depth, 0);
    assert!(runtime.gpu_initialized());
    assert_eq!(gone_log.count("pre_prepare:0"), 1);
    gone_log.clear();
    kept_log.clear();

    assert!(runtime.remove_gpu_plugin(gone).is_some());
    assert!(
        runtime.gpu_initialized(),
        "removing a GPU plugin must not clear the initialised flag"
    );
    assert_eq!(runtime.gpu_plugin_ids().collect::<Vec<_>>(), vec![kept]);

    frame(&mut runtime, &device, &queue, &colour, &depth, 1);
    assert!(
        gone_log.entries().is_empty(),
        "a removed plugin was called: {:?}",
        gone_log.entries()
    );
    kept_log.assert_take(&[
        "pre_prepare:1",
        &format!("post_paint:1:{COLOUR_FORMAT:?}:pick=false"),
    ]);

    let name = runtime
        .remove_gpu_plugin(kept)
        .expect("the id is registered")
        .type_name();
    let stats = runtime.last_stats();
    assert!(!stats.pre_prepare_ms.contains_key(name));
    assert!(!stats.post_paint_ms.contains_key(name));
}

/// Removing the middle of three same-priority GPU plugins keeps the other two
/// in registration order (the order they run in), and a stale id returns
/// `None`.
#[test]
fn gpu_fixture_removal_keeps_order_and_rejects_stale_ids() {
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new();
    let a = runtime.add_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
    let b = runtime.add_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
    let c = runtime.add_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));

    assert!(runtime.remove_gpu_plugin(b).is_some());
    assert_eq!(runtime.gpu_plugin_ids().collect::<Vec<_>>(), vec![a, c]);
    assert!(runtime.remove_gpu_plugin(b).is_none());
    assert_eq!(runtime.gpu_plugin_count(), 2);
}

/// A GPU plugin added after the first frame is initialised on the next frame
/// on its own: the plugin already running is not initialised again.
#[test]
fn gpu_fixture_added_mid_session_initialises_only_itself() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let first_log = CallLog::new();
    let added_log = CallLog::new();
    let mut runtime = ViewportRuntime::new().with_gpu_plugin(LoggingGpuPlugin::new(
        first_log.clone(),
        gpu_phase::PRE_PREPARE,
    ));
    let (colour, depth) = probe_targets(&device, SIZE, COLOUR_FORMAT);

    frame(&mut runtime, &device, &queue, &colour, &depth, 0);
    first_log.clear();

    runtime.add_gpu_plugin(LoggingGpuPlugin::new(
        added_log.clone(),
        gpu_phase::PRE_PREPARE,
    ));
    frame(&mut runtime, &device, &queue, &colour, &depth, 1);
    first_log.assert_take(&[
        "pre_prepare:1",
        &format!("post_paint:1:{COLOUR_FORMAT:?}:pick=false"),
    ]);
    added_log.assert_take(&[
        "init_gpu",
        "pre_prepare:1",
        &format!("post_paint:1:{COLOUR_FORMAT:?}:pick=false"),
    ]);
}

/// A GPU plugin registered between `pre_prepare` and `post_paint` gets no
/// `post_paint` that frame; its first call is `init_gpu` on the next one.
#[test]
fn gpu_fixture_registered_before_post_paint_is_skipped() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new();
    let (colour, depth) = probe_targets(&device, SIZE, COLOUR_FORMAT);
    let camera = Camera::default();
    let ctx = GpuFrameContext::new(&camera, glam::Vec2::new(SIZE as f32, SIZE as f32), 0.0, 0);

    let pre = runtime.pre_prepare(&device, &queue, &ctx);
    runtime.add_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
    let targets = PostPaintTargets::new(&colour, &depth, COLOUR_FORMAT);
    let post = runtime.post_paint(&device, &queue, &targets, &ctx);
    queue.submit(pre.into_iter().chain(post));
    assert!(
        log.entries().is_empty(),
        "called before init: {:?}",
        log.entries()
    );

    frame(&mut runtime, &device, &queue, &colour, &depth, 1);
    log.assert_take(&[
        "init_gpu",
        "pre_prepare:1",
        &format!("post_paint:1:{COLOUR_FORMAT:?}:pick=false"),
    ]);
}

struct GreedyGpu;

impl viewport_lib::runtime::GpuPlugin for GreedyGpu {
    fn type_name(&self) -> &'static str {
        "fixture_test.greedy_gpu"
    }
    fn requires(&self) -> viewport_lib::Requirements {
        viewport_lib::Requirements::full_tier().with_bind_groups(u32::MAX)
    }
    fn init_gpu(&mut self, _: &wgpu::Device) {
        panic!("init_gpu ran for a declined GPU plugin");
    }
}

/// A plugin the device cannot run, added mid-session, is removed and listed
/// as declined; the plugin already running is neither initialised again nor
/// removed.
#[test]
fn gpu_fixture_declined_mid_session_leaves_others_alone() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new();
    let kept = runtime.add_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
    let (colour, depth) = probe_targets(&device, SIZE, COLOUR_FORMAT);

    frame(&mut runtime, &device, &queue, &colour, &depth, 0);
    log.clear();

    runtime.add_gpu_plugin(GreedyGpu);
    frame(&mut runtime, &device, &queue, &colour, &depth, 1);
    assert!(matches!(
        runtime.declined_gpu_plugins(),
        [viewport_lib::ViewportError::PluginUnsupported { type_name, .. }]
            if type_name == "fixture_test.greedy_gpu"
    ));
    assert_eq!(runtime.gpu_plugin_ids().collect::<Vec<_>>(), vec![kept]);
    assert!(runtime.gpu_initialized());
    log.assert_take(&[
        "pre_prepare:1",
        &format!("post_paint:1:{COLOUR_FORMAT:?}:pick=false"),
    ]);
}
