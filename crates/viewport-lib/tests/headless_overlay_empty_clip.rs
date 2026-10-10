//! An `OverlayClip::rect` without area clips everything.
//!
//! A consumer clipping an item to the intersection of its ancestors' boxes
//! gets an empty box once the item scrolls out of its container. That box has
//! to hide the item, on every path that takes a rect: an immediate polyline
//! (the vertex stream), an immediate shape, a compiled item, and a whole
//! retained group.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    Colour, OverlayFill, OverlayPolylineItem, OverlayShape, OverlayShapeItem, RetainedOverlay,
};

const SIZE: u32 = 64;

/// Empty in both directions, the shape of an intersection with nothing left.
const EMPTY: [f32; 4] = [40.0, 40.0, 20.0, 20.0];

fn overlay_frame() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.3, 0.3, 0.3, 1.0));
    frame
}

fn red_square() -> OverlayPolylineItem {
    let mut p =
        OverlayPolylineItem::new(vec![[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]]);
    p.closed = true;
    p.stroke = None;
    p.style.fill = OverlayFill::Solid(Colour::linear(1.0, 0.0, 0.0, 1.0));
    p
}

fn red_shape() -> OverlayShapeItem {
    OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [16.0, 16.0],
        [32.0, 32.0],
    )
    .with_fill(OverlayFill::Solid(Colour::linear(1.0, 0.0, 0.0, 1.0)))
}

fn red_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[0] > 150 && p[1] < 100 && p[2] < 100)
        .count()
}

#[test]
fn an_empty_clip_rect_hides_the_item_on_every_path() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // The unclipped polyline draws, so a zero below is the clip at work.
    let mut frame = overlay_frame();
    frame.overlays.polylines = vec![red_square()];
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert!(red_pixels(&px) > 500, "the unclipped square should draw");

    let mut poly = red_square();
    poly.clip.rect = Some(EMPTY);
    let mut frame = overlay_frame();
    frame.overlays.polylines = vec![poly.clone()];
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(
        red_pixels(&px),
        0,
        "immediate polyline with an empty clip drew"
    );

    let mut shape = red_shape();
    shape.clip.rect = Some(EMPTY);
    let mut frame = overlay_frame();
    frame.overlays.shapes = vec![shape];
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(
        red_pixels(&px),
        0,
        "immediate shape with an empty clip drew"
    );

    let id = renderer.compile_overlay_geometry(&device, &queue, &[poly], &[], &[], &[], 1.0);
    let mut frame = overlay_frame();
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(red_pixels(&px), 0, "compiled item with an empty clip drew");

    let id = renderer.compile_overlay_geometry(
        &device,
        &queue,
        &[red_square()],
        &[red_shape()],
        &[],
        &[],
        1.0,
    );
    let mut frame = overlay_frame();
    frame.overlays.retained = vec![RetainedOverlay::new(id).with_clip_rect(EMPTY)];
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(red_pixels(&px), 0, "retained group with an empty clip drew");
}

/// Two real boxes that do not overlap clip everything too. The shape shader
/// intersects a compiled item's rect with its group's rect, and the off-screen
/// box it returns for no overlap once sat at 1e9, where `x + 1` rounds back to
/// `x` in f32: the box lost its area and read as no clip.
#[test]
fn non_overlapping_item_and_group_clips_hide_the_item() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut shape = red_shape();
    shape.clip.rect = Some([0.0, 0.0, 24.0, 64.0]);
    let id = renderer.compile_overlay_geometry(&device, &queue, &[], &[shape], &[], &[], 1.0);

    let mut frame = overlay_frame();
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert!(red_pixels(&px) > 0, "the item's own clip keeps a strip");

    let mut frame = overlay_frame();
    frame.overlays.retained =
        vec![RetainedOverlay::new(id).with_clip_rect([40.0, 0.0, 64.0, 64.0])];
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(
        red_pixels(&px),
        0,
        "non-overlapping item and group clips drew"
    );
}
