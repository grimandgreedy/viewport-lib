//! Retained overlay geometry (`OverlayGeometryId` / `RetainedOverlay`), end to end.
//!
//! Compiles a filled polyline once with `compile_overlay_geometry`, then submits
//! it through `OverlayFrame::retained` and reads back pixels to confirm: it draws
//! from the cached buffer, the per-frame `translate` moves it without
//! re-compiling, `opacity` fades it, and a freed handle draws nothing.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{OverlayFill, OverlayPolylineItem, RetainedOverlay};

/// A 64x64 frame looking at nothing, flat grey background, chrome off.
fn overlay_frame(size: u32) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.3, 0.3, 0.3, 1.0]);
    frame
}

/// A closed, red-filled square polyline over screen pixels 16..48.
fn red_square() -> OverlayPolylineItem {
    let mut p = OverlayPolylineItem::default();
    p.points = vec![[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]];
    p.closed = true;
    p.thickness = 0.0;
    p.fill = Some(OverlayFill::Solid([1.0, 0.0, 0.0, 1.0]));
    p.opacity = 1.0;
    p
}

fn rgb_at(px: &[u8], size: u32, x: u32, y: u32) -> (u8, u8, u8) {
    let i = ((y * size + x) * 4) as usize;
    (px[i], px[i + 1], px[i + 2])
}

fn is_red(c: (u8, u8, u8)) -> bool {
    c.0 > 200 && c.1 < 100 && c.2 < 100
}

fn is_background(c: (u8, u8, u8)) -> bool {
    // 0.3 linear grey encodes to ~149 in sRGB; the key point is it is not red.
    c.0 < 200 && (c.0 as i32 - c.1 as i32).abs() < 30 && (c.1 as i32 - c.2 as i32).abs() < 30
}

/// A compiled group draws from its cached buffer, and the per-frame `translate`
/// moves it without re-compiling.
#[test]
fn retained_draws_and_translates() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    let id = renderer.compile_overlay_geometry(&device, &queue, &[red_square()], &[]);

    // No translate: the square covers pixels 16..48, so the centre is red and a
    // point to its left is background.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(is_red(rgb_at(&px, size, 32, 32)), "centre should be red");
    assert!(
        is_background(rgb_at(&px, size, 8, 32)),
        "left of the square should be background"
    );

    // Translate +16 in x with the SAME compiled id: the square now covers 32..64,
    // so the old centre reads background and a point at x=56 reads red.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id).with_translate([16.0, 0.0])];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_background(rgb_at(&px, size, 24, 32)),
        "translated square should have vacated x=24"
    );
    assert!(
        is_red(rgb_at(&px, size, 56, 32)),
        "translated square should now cover x=56"
    );
}

/// `opacity` 0 fades the group out; a freed handle draws nothing.
#[test]
fn retained_opacity_and_free() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    let id = renderer.compile_overlay_geometry(&device, &queue, &[red_square()], &[]);

    // Fully transparent: the centre reads the background.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id).with_opacity(0.0)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_background(rgb_at(&px, size, 32, 32)),
        "opacity 0 should show the background"
    );

    // Free the group; a submission referencing the freed id is skipped.
    assert!(renderer.free_overlay_geometry(id), "free should report success");
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_background(rgb_at(&px, size, 32, 32)),
        "a freed group should draw nothing"
    );
}
