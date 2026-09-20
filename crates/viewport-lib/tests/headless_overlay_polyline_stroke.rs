//! A polyline's stroke as an optional bucket.
//!
//! `OverlayPolylineItem::stroke` holds the line itself: width, colour, pattern,
//! joins and caps. `None` means no line at all, which is what a closed polyline
//! with an interior fill and no outline wants. These tests pin that an absent
//! stroke draws the fill and nothing else, and that a stroke of zero width is
//! the same picture.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{OverlayFill, OverlayPolylineItem, OverlayStroke};

const SIZE: u32 = 64;

/// A 64x64 frame looking at nothing, flat grey background, chrome off.
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
    frame.viewport.background_colour = Some([0.3, 0.3, 0.3, 1.0].into());
    frame
}

/// A closed square over pixels 16..48, filled green, with whatever stroke the
/// caller asks for.
fn filled_square(stroke: Option<OverlayStroke>) -> OverlayPolylineItem {
    let mut p =
        OverlayPolylineItem::new(vec![[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]]);
    p.closed = true;
    p.stroke = stroke;
    p.style.fill = OverlayFill::Solid([0.0, 1.0, 0.0, 1.0].into());
    p
}

fn rgb(px: &[u8], x: u32, y: u32) -> (u8, u8, u8) {
    let i = ((y * SIZE + x) * 4) as usize;
    (px[i], px[i + 1], px[i + 2])
}

/// No stroke draws the interior fill and leaves the edge alone, and a stroke of
/// zero width is the same frame: `stroke: None` is how a fill-only polyline is
/// spelled, and it has to match what a zero-width line already did.
#[test]
fn an_absent_stroke_draws_the_fill_and_no_outline() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut none_frame = overlay_frame();
    none_frame.overlays.polylines = vec![filled_square(None)];
    let none_px = renderer.render_offscreen(&device, &queue, &none_frame, SIZE, SIZE);

    // The fill is there.
    let (r, g, b) = rgb(&none_px, 32, 32);
    assert!(
        g > 150 && r < 100 && b < 100,
        "the interior should be filled green, got rgb ({r}, {g}, {b})"
    );
    // The edge is not: a red outline would show on the boundary row.
    let (er, eg, _) = rgb(&none_px, 32, 16);
    assert!(
        er < 100 || eg > er,
        "no outline should be drawn on the edge, got r {er} g {eg}"
    );

    let mut zero_frame = overlay_frame();
    zero_frame.overlays.polylines = vec![filled_square(Some(OverlayStroke::new(
        0.0,
        [1.0, 0.0, 0.0, 1.0],
    )))];
    let zero_px = renderer.render_offscreen(&device, &queue, &zero_frame, SIZE, SIZE);
    assert_eq!(
        none_px, zero_px,
        "an absent stroke and a zero-width one should render the same frame"
    );

    // A real stroke does reach the edge, so the checks above are not passing on
    // an empty frame.
    let mut stroked_frame = overlay_frame();
    stroked_frame.overlays.polylines = vec![filled_square(Some(OverlayStroke::new(
        6.0,
        [1.0, 0.0, 0.0, 1.0],
    )))];
    let stroked_px = renderer.render_offscreen(&device, &queue, &stroked_frame, SIZE, SIZE);
    let (sr, sg, _) = rgb(&stroked_px, 32, 16);
    assert!(
        sr > 150 && sg < 100,
        "a stroked square should draw its outline, got r {sr} g {sg}"
    );
}
