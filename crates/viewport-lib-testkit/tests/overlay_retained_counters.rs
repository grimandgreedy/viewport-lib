//! Retained-overlay counters in `FrameStats`.
//!
//! A consumer drawing its interface through the overlay system submits handles
//! and cannot see what the renderer did with them: a group is skipped silently
//! when its handle was freed, and a glyph-bearing group is rebuilt silently when
//! the atlas grows or `pixels_per_point` changes. These assertions lock the
//! counters that report both, plus the live byte total.

use viewport_lib::{
    CameraFrame, FrameData, LabelItem, OverlayPolylineItem, RetainedOverlay, SceneFrame,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 200;
const H: u32 = 150;

/// An empty scene at `ppp` carrying `retained`. Overlays are the only content,
/// so nothing else can move the counters. The viewport is given in logical
/// pixels, so it shrinks as `ppp` grows and the render target stays `W x H`.
fn frame(ppp: f32, retained: Vec<RetainedOverlay>) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 5.0, 0.7, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32 / ppp, H as f32 / ppp])
            .with_pixels_per_point(ppp),
        SceneFrame::from_surface_items(Vec::new()),
    );
    fd.overlays.retained = retained;
    fd
}

#[test]
fn retained_counters_report_skips_and_reemits() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    // A polyline-only group: viewport-independent geometry that never re-emits.
    let polyline = h.renderer.compile_overlay_geometry(
        &h.device,
        &h.queue,
        &[OverlayPolylineItem::new(vec![
            [10.0, 10.0],
            [60.0, 40.0],
            [110.0, 20.0],
        ])],
        &[],
        &[],
        &[],
        1.0,
    );
    // A label: glyphs bake atlas UVs at a physical size, so this one re-emits
    // when `pixels_per_point` changes.
    let label = h.renderer.compile_overlay_label(
        &h.device,
        &h.queue,
        &LabelItem::new("retained").with_screen_anchor([20.0, 80.0]),
        1.0,
    );
    // A third group, freed before it is ever submitted.
    let freed = h.renderer.compile_overlay_geometry(
        &h.device,
        &h.queue,
        &[OverlayPolylineItem::new(vec![[0.0, 0.0], [20.0, 20.0]])],
        &[],
        &[],
        &[],
        1.0,
    );

    let live_bytes = h.renderer.last_frame_stats().overlay_retained_bytes;
    assert_eq!(
        live_bytes, 0,
        "test premise: bytes are published by prepare"
    );

    let submissions = || {
        vec![
            RetainedOverlay::new(polyline),
            RetainedOverlay::new(label),
            RetainedOverlay::new(freed),
        ]
    };

    // All three live: everything submitted draws.
    let stats = h.render_two_frames(&frame(1.0, submissions()), W, H);
    assert_eq!(stats.overlay_retained_submitted, 3);
    assert_eq!(stats.overlay_retained_drawn, 3);
    assert_eq!(stats.overlay_retained_reemitted, 0, "settled at this ppp");
    let all_live_bytes = stats.overlay_retained_bytes;
    assert!(all_live_bytes > 0, "three compiled groups hold geometry");

    // Free one and keep submitting it: the submission is counted, the draw is
    // not. This is the difference a consumer cannot see from its own side.
    assert!(h.renderer.free_overlay_geometry(freed));
    let stats = h.render_two_frames(&frame(1.0, submissions()), W, H);
    assert_eq!(stats.overlay_retained_submitted, 3);
    assert_eq!(stats.overlay_retained_drawn, 2);
    assert!(
        stats.overlay_retained_bytes < all_live_bytes,
        "freeing a group drops its charge"
    );

    // A DPI change re-emits the glyph-bearing group and only that one.
    let _ = h.render(&frame(2.0, submissions()), W, H);
    let stats = h.stats();
    assert_eq!(stats.overlay_retained_drawn, 2);
    assert_eq!(
        stats.overlay_retained_reemitted, 1,
        "the label re-emits, the polyline group does not"
    );

    // And settles: a second frame at the same ppp draws from the rebuilt
    // buffers rather than rebuilding again.
    let _ = h.render(&frame(2.0, submissions()), W, H);
    assert_eq!(h.stats().overlay_retained_reemitted, 0);

    // Every charge added at compile is removed at free, including for a group
    // that was rebuilt in between.
    assert!(h.renderer.free_overlay_geometry(polyline));
    assert!(h.renderer.free_overlay_geometry(label));
    let stats = h.render_two_frames(&frame(2.0, submissions()), W, H);
    assert_eq!(stats.overlay_retained_submitted, 3);
    assert_eq!(stats.overlay_retained_drawn, 0);
    assert_eq!(
        stats.overlay_retained_bytes, 0,
        "no compiled geometry left resident"
    );
}
