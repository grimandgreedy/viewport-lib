//! The pipeline cache, on a backend that has one.
//!
//! The lookup that finds a renderer's cache is process-wide, so everything
//! runs from one test and in sequence. On a backend with no pipeline cache
//! (Metal) only the two-device part does any work.

use viewport_lib::wgpu;

mod common;
use common::*;

fn frame() -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [64.0, 64.0];
    frame
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;

#[test]
fn lazy_pipelines_reach_the_cache_and_two_devices_do_not_share_one() {
    let Some((device, queue)) = headless_device_recommended_limits() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // The pipelines a frame builds lazily name no cache at their call sites,
    // and have to land in the renderer's cache all the same.
    let mut renderer = ViewportRenderer::new_with_pipeline_cache(&device, FORMAT, None);
    let before = renderer.pipeline_cache_data();
    let _ = renderer.render_offscreen(&device, &queue, &frame(), 64, 64);
    let saved = renderer.pipeline_cache_data();
    if device.features().contains(wgpu::Features::PIPELINE_CACHE) {
        let (before, saved) = (
            before.map_or(0, |d| d.len()),
            saved.as_ref().map_or(0, Vec::len),
        );
        assert!(
            saved > before,
            "the first frame's pipelines were not added to the cache ({before} -> {saved} bytes)"
        );
    } else {
        assert!(saved.is_none());
        eprintln!("this backend has no pipeline cache; checking the two-device case only");
    }
    drop(renderer);

    // A renderer seeded with that data draws the same frame.
    let mut seeded = ViewportRenderer::new_with_pipeline_cache(&device, FORMAT, saved.as_deref());
    let _ = seeded.render_offscreen(&device, &queue, &frame(), 64, 64);

    // A second device from a second wgpu instance, alive at the same time.
    // wgpu can report the two devices as equal, and one device's cache must
    // never be used to build the other's pipelines.
    let Some((other_device, other_queue)) = headless_device_recommended_limits() else {
        return;
    };
    let mut other = ViewportRenderer::new_with_pipeline_cache(&other_device, FORMAT, None);
    let _ = other.render_offscreen(&other_device, &other_queue, &frame(), 64, 64);
    let _ = seeded.render_offscreen(&device, &queue, &frame(), 64, 64);

    // Two renderers on one device.
    let mut sibling = ViewportRenderer::new_with_pipeline_cache(&device, FORMAT, None);
    let _ = sibling.render_offscreen(&device, &queue, &frame(), 64, 64);
}
