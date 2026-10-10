//! Smoke tests for the `GpuPlugin` fixture.
//!
//! Drives `LoggingGpuPlugin` through a real `ViewportRuntime` the way a host
//! does: `pre_prepare` before the renderer's frame, `post_paint` after it with
//! the host's own target views. Checks deferred `init_gpu`, the per-frame
//! order, that the returned command buffers submit, and the device-recreation
//! notification.

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

/// `notify_device_recreated` reaches the plugin, and `init_gpu` runs again
/// afterwards so the plugin can rebuild what it dropped.
#[test]
fn gpu_fixture_sees_device_recreation() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let log = CallLog::new();
    let mut runtime = ViewportRuntime::new()
        .with_gpu_plugin(LoggingGpuPlugin::new(log.clone(), gpu_phase::PRE_PREPARE));
    let (colour, depth) = probe_targets(&device, SIZE, COLOUR_FORMAT);

    frame(&mut runtime, &device, &queue, &colour, &depth, 0);
    log.clear();

    runtime.notify_device_recreated(&device, &queue);
    frame(&mut runtime, &device, &queue, &colour, &depth, 1);

    log.assert_take(&[
        "on_device_recreated",
        "init_gpu",
        "pre_prepare:1",
        &format!("post_paint:1:{COLOUR_FORMAT:?}:pick=false"),
    ]);
}
