//! Overlay cost is reported on its own in `FrameStats`, and stays that way.
//!
//! `PrepareBreakdown::overlay_ms` exists because a consumer that draws its whole
//! interface through the overlay system (a UI toolkit, an editor chrome layer)
//! needs overlay build cost separable from the rest of the per-viewport phase:
//! the two scale with completely different things, and lumped together neither
//! can be acted on. These tests hold the contract that makes the field worth
//! having - it tracks the overlay load, it is taken out of `viewport_ms` rather
//! than double-counted inside it, and the breakdown still partitions
//! `cpu_prepare_ms`.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{LabelItem, OverlayFill, OverlayShape, OverlayShapeItem};

/// A 64x64 frame looking at nothing, chrome off, so the only work in the
/// viewport phase is whatever overlays the test puts in.
fn overlay_frame(size: u32) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&Camera::default());
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame
}

/// `count` small shapes and `count` labels, enough overlay work that its build
/// cost is well clear of timer resolution.
fn overlay_load(frame: &mut FrameData, count: usize) {
    frame.overlays.shapes = (0..count)
        .map(|i| {
            let x = (i % 8) as f32 * 8.0;
            let y = (i / 8 % 8) as f32 * 8.0;
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 2.0 },
                [x, y],
                [6.0, 6.0],
            )
            .with_fill(OverlayFill::Solid([0.8, 0.2, 0.2, 1.0].into()))
        })
        .collect();
    frame.overlays.labels = (0..count)
        .map(|i| {
            LabelItem::new(format!("label {i}"))
                .with_position([(i % 8) as f32 * 8.0, (i / 8 % 8) as f32 * 8.0])
        })
        .collect();
}

/// The sum of every `PrepareBreakdown` phase, which should account for
/// `cpu_prepare_ms`.
fn phase_sum(b: &viewport_lib::PrepareBreakdown) -> f32 {
    b.plugin_ms
        + b.lighting_ms
        + b.uniforms_ms
        + b.instancing_ms
        + b.geometry_ms
        + b.shadow_ms
        + b.viewport_ms
        + b.overlay_ms
        + b.other_ms
}

/// A frame carrying a real overlay load reports time in `overlay_ms`; an
/// otherwise identical frame with no overlays reports approximately none. Without
/// the split, both frames look the same in the breakdown and the cost is
/// invisible inside `viewport_ms`.
#[test]
fn overlay_prepare_cost_is_reported_separately() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    let empty = overlay_frame(size);
    let _ = renderer.render_offscreen(&device, &queue, &empty, size, size);
    let empty_stats = renderer.last_frame_stats();

    let mut loaded = overlay_frame(size);
    overlay_load(&mut loaded, 512);
    let _ = renderer.render_offscreen(&device, &queue, &loaded, size, size);
    let loaded_stats = renderer.last_frame_stats();

    let loaded_overlay = loaded_stats.prepare_breakdown.overlay_ms;
    let empty_overlay = empty_stats.prepare_breakdown.overlay_ms;
    assert!(
        loaded_overlay > 0.0,
        "a frame with 512 shapes and 512 labels reported no overlay prepare time"
    );
    assert!(
        loaded_overlay > empty_overlay,
        "overlay prepare time did not track the overlay load: \
         {loaded_overlay} ms with a load vs {empty_overlay} ms empty"
    );
}

/// Overlay time is taken out of `viewport_ms`, not counted in both places. The
/// overlay load must not show up as a matching rise in `viewport_ms`, or a
/// consumer reading both fields double-counts it.
#[test]
fn overlay_time_is_not_also_inside_viewport_time() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    let mut loaded = overlay_frame(size);
    overlay_load(&mut loaded, 512);
    // Two frames: the first warms caches (glyph rasterisation, buffer growth) so
    // the measured one is steady-state.
    let _ = renderer.render_offscreen(&device, &queue, &loaded, size, size);
    let _ = renderer.render_offscreen(&device, &queue, &loaded, size, size);
    let b = renderer.last_frame_stats().prepare_breakdown;

    assert!(
        b.overlay_ms > b.viewport_ms,
        "with 1024 overlay items the overlay phase should dominate the rest of \
         the viewport phase, but overlay_ms {} <= viewport_ms {} - which is what \
         it looks like when overlay time is still being counted inside \
         viewport_ms",
        b.overlay_ms,
        b.viewport_ms
    );
}

/// The breakdown still partitions `cpu_prepare_ms` after the split: every phase
/// plus `other_ms` accounts for the total. A new field that forgets to adjust
/// `other_ms` breaks this silently.
#[test]
fn breakdown_phases_still_account_for_cpu_prepare() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    let mut loaded = overlay_frame(size);
    overlay_load(&mut loaded, 256);
    let _ = renderer.render_offscreen(&device, &queue, &loaded, size, size);
    let stats = renderer.last_frame_stats();
    let total = stats.cpu_prepare_ms;
    let sum = phase_sum(&stats.prepare_breakdown);

    // `other_ms` is derived as the remainder, so the sum matches the total up to
    // the cost of reading the phase timers themselves.
    let slack = (total * 0.05).max(0.05);
    assert!(
        (sum - total).abs() <= slack,
        "phases sum to {sum} ms but cpu_prepare_ms is {total} ms (slack {slack} ms); \
         a phase is being double-counted or dropped"
    );
}
