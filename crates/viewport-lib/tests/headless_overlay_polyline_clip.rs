//! An overlay polyline clipped to a mask shape, end to end.
//!
//! A polyline crosses a 64x64 frame from corner to corner while a mask shape
//! covers only the left half. With `with_clip` naming that mask, the stroke must
//! reach the screen inside the mask and nowhere else: this renders the real
//! overlay pass offscreen and reads a pixel on each side of the boundary.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{OverlayPolylineItem, OverlayShape, OverlayShapeItem, PolylineCap};

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

/// A mask over the left half of the frame, registered under id 7.
fn left_half_mask() -> OverlayShapeItem {
    OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [0.0, 0.0],
        [32.0, 64.0],
    )
    .with_clip_mask(7)
}

/// A thick blue stroke across the full width at y = 32, which without a clip
/// covers the centre row of both halves.
fn full_width_stroke() -> OverlayPolylineItem {
    OverlayPolylineItem::new(vec![[0.0, 32.0], [64.0, 32.0]])
        .with_thickness(16.0)
        .with_cap(PolylineCap::Square)
        .with_colour([0.0, 0.0, 1.0, 1.0])
}

/// Read a pixel as (r, g, b).
fn rgb(px: &[u8], x: u32, y: u32) -> (u8, u8, u8) {
    let i = ((y * SIZE + x) * 4) as usize;
    (px[i], px[i + 1], px[i + 2])
}

/// The clipped stroke draws inside the mask and is discarded outside it.
#[test]
fn a_clipped_polyline_draws_only_inside_its_mask() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut frame = overlay_frame();
    frame.overlays.shapes = vec![left_half_mask()];
    frame.overlays.polylines = vec![full_width_stroke().with_clip(7)];

    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    let (ir, _, ib) = rgb(&px, 16, 32);
    assert!(
        ib > 150 && ir < 100,
        "inside the mask the stroke should draw, got rgb ({ir}, _, {ib})"
    );
    // Outside the mask the pixel must still be the flat grey background: the
    // stroke would tint it blue, so a blue channel well above the red one is the
    // signal that it leaked.
    let (or_, _, ob) = rgb(&px, 48, 32);
    assert!(
        ob.saturating_sub(or_) < 10,
        "outside the mask the stroke should be discarded, got rgb ({or_}, _, {ob})"
    );
}

/// The same stroke without a clip id covers both halves, so the test above is
/// measuring the clip and not the geometry.
#[test]
fn an_unclipped_polyline_covers_both_halves() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut frame = overlay_frame();
    frame.overlays.shapes = vec![left_half_mask()];
    frame.overlays.polylines = vec![full_width_stroke()];

    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    for x in [16u32, 48] {
        let (r, _, b) = rgb(&px, x, 32);
        assert!(
            b > 150 && r < 100,
            "an unclipped stroke should cover x = {x}, got rgb ({r}, _, {b})"
        );
    }
}
