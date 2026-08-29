//! Cross-family overlay z-ordering, end to end.
//!
//! A filled overlay shape and an overlay polyline are drawn over the same pixels.
//! The shape and the polyline live in different overlay families (the shape
//! pipeline vs the merged text batch), so their relative order used to be fixed:
//! shapes always under the merged batch. These tests render the real overlay pass
//! offscreen and read the centre pixel to confirm `z_order` now composes across
//! families, while a scene that leaves `z_order` at its default still gets the
//! old order.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{OverlayFill, OverlayPolylineItem, OverlayShape, OverlayShapeItem, PolylineCap};

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

/// Red shape and blue polyline covering the centre, with the given z orders.
/// The polyline is a thick horizontal stroke with square caps, so it fills the
/// same 32x32 centre box the shape covers.
fn overlapping_items(shape_z: i32, poly_z: i32) -> (OverlayShapeItem, OverlayPolylineItem) {
    let pos = [16.0, 16.0];
    let sz = [32.0, 32.0];
    let shape = OverlayShapeItem::new(OverlayShape::Rect { corner_radius: 0.0 }, pos, sz)
        .with_fill(OverlayFill::Solid([1.0, 0.0, 0.0, 1.0]))
        .with_z_order(shape_z);
    let poly = OverlayPolylineItem::new(vec![[16.0, 32.0], [48.0, 32.0]])
        .with_thickness(32.0)
        .with_cap(PolylineCap::Square)
        .with_colour([0.0, 0.0, 1.0, 1.0])
        .with_z_order(poly_z);
    (shape, poly)
}

/// Read the centre pixel as (r, g, b).
fn centre_rgb(px: &[u8], size: u32) -> (u8, u8, u8) {
    let i = (((size / 2) * size + (size / 2)) * 4) as usize;
    (px[i], px[i + 1], px[i + 2])
}

/// With a high `z_order` on the shape, it draws above the polyline even though
/// the shape family is fixed below the merged text batch. Centre must read red.
#[test]
fn shape_zorder_lifts_it_above_a_polyline() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 64u32;
    let mut frame = overlay_frame(size);
    let (shape, poly) = overlapping_items(1000, 0);
    frame.overlays.shapes = vec![shape];
    frame.overlays.polylines = vec![poly];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (r, g, b) = centre_rgb(&px, size);
    assert!(
        r > 150 && b < 100,
        "expected the red shape on top, got rgb ({r}, {g}, {b})"
    );
}

/// With every `z_order` at the default 0, the fixed family order stands: the
/// polyline (merged text batch) draws over the shape. Centre must read blue. This
/// is the fast path, and it also proves `z_order` is what changed the result
/// above.
#[test]
fn default_zorder_keeps_polyline_above_shape() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 64u32;
    let mut frame = overlay_frame(size);
    let (shape, poly) = overlapping_items(0, 0);
    frame.overlays.shapes = vec![shape];
    frame.overlays.polylines = vec![poly];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (r, g, b) = centre_rgb(&px, size);
    assert!(
        b > 150 && r < 100,
        "expected the blue polyline on top, got rgb ({r}, {g}, {b})"
    );
}
