//! Streaming overlay texture lifecycle, end to end.
//!
//! A streaming overlay texture is allocated once, updated in place, and freed. A
//! positioned `OverlayShapeItem` points at it, so these tests render the real
//! overlay pass offscreen and read the centre pixel to confirm: the shape shows
//! the texture's contents, `update_overlay_texture` changes those contents while
//! the same id stays valid, and `free_overlay_texture` releases the slot so the
//! shape is skipped for the frame.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{OverlayFill, OverlayShape, OverlayShapeItem};

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

/// Solid `rgba` fill of `w * h` pixels, row-major.
fn solid_rgba(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
    let mut data = Vec::with_capacity((w * h * 4) as usize);
    for _ in 0..(w * h) {
        data.extend_from_slice(&rgba);
    }
    data
}

/// A centred shape covering the middle of the viewport, textured with `tex` and
/// a white tint (so the sampled colour shows through unchanged).
fn textured_shape(tex: viewport_lib::OverlayTextureId) -> OverlayShapeItem {
    OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [16.0, 16.0],
        [32.0, 32.0],
    )
    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0]))
    .with_texture(tex)
}

/// Read the centre pixel as (r, g, b).
fn centre_rgb(px: &[u8], size: u32) -> (u8, u8, u8) {
    let i = (((size / 2) * size + (size / 2)) * 4) as usize;
    (px[i], px[i + 1], px[i + 2])
}

/// Create a streaming texture, draw it, then update it in place to a new colour
/// and draw again with the same id. The centre pixel must follow the update,
/// proving the reused GPU texture takes new contents without a new id.
#[test]
fn streaming_texture_updates_in_place() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let (tw, th) = (8u32, 8u32);
    let tex = renderer
        .resources_mut()
        .create_streaming_overlay_texture(&device, tw, th);

    // First frame: fill red.
    renderer.resources_mut().update_overlay_texture(
        &device,
        &queue,
        tex,
        tw,
        th,
        &solid_rgba(tw, th, [255, 0, 0, 255]),
    );

    let size = 64u32;
    let mut frame = overlay_frame(size);
    frame.overlays.shapes = vec![textured_shape(tex)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (r, _g, b) = centre_rgb(&px, size);
    assert!(
        r > 150 && b < 100,
        "expected red from the texture, got ({r}, {b})"
    );

    // Second frame: update the same id to blue. No new id, no new allocation.
    renderer.resources_mut().update_overlay_texture(
        &device,
        &queue,
        tex,
        tw,
        th,
        &solid_rgba(tw, th, [0, 0, 255, 255]),
    );

    let mut frame = overlay_frame(size);
    frame.overlays.shapes = vec![textured_shape(tex)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (r, _g, b) = centre_rgb(&px, size);
    assert!(
        b > 150 && r < 100,
        "expected blue after in-place update, got ({r}, {b})"
    );
}

/// A size change reallocates behind the same id, which stays valid and renders
/// the new contents.
#[test]
fn streaming_texture_resizes_under_same_id() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let tex = renderer
        .resources_mut()
        .create_streaming_overlay_texture(&device, 4, 4);
    renderer.resources_mut().update_overlay_texture(
        &device,
        &queue,
        tex,
        4,
        4,
        &solid_rgba(4, 4, [255, 0, 0, 255]),
    );

    // Grow the source; the id must remain usable.
    let ok = renderer.resources_mut().update_overlay_texture(
        &device,
        &queue,
        tex,
        16,
        16,
        &solid_rgba(16, 16, [0, 0, 255, 255]),
    );
    assert!(ok, "update after resize should succeed on a live id");

    let size = 64u32;
    let mut frame = overlay_frame(size);
    frame.overlays.shapes = vec![textured_shape(tex)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (r, _g, b) = centre_rgb(&px, size);
    assert!(
        b > 150 && r < 100,
        "expected blue after resize, got ({r}, {b})"
    );
}

/// Freeing releases the slot: the first free reports success, a second reports
/// nothing to free, and updating a freed id is a no-op that returns false.
#[test]
fn freeing_overlay_texture_invalidates_the_id() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let tex = renderer
        .resources_mut()
        .create_streaming_overlay_texture(&device, 4, 4);
    renderer.resources_mut().update_overlay_texture(
        &device,
        &queue,
        tex,
        4,
        4,
        &solid_rgba(4, 4, [255, 0, 0, 255]),
    );

    assert!(
        renderer.resources_mut().free_overlay_texture(tex),
        "first free frees it"
    );
    assert!(
        !renderer.resources_mut().free_overlay_texture(tex),
        "second free finds nothing"
    );
    assert!(
        !renderer.resources_mut().update_overlay_texture(
            &device,
            &queue,
            tex,
            4,
            4,
            &solid_rgba(4, 4, [0, 0, 255, 255])
        ),
        "updating a freed id is a no-op"
    );

    // A textured shape whose id was freed is skipped for the frame (not drawn
    // as a tint fallback), so the centre reads the flat grey background: equal
    // channels, and in particular not the freed texture's blue.
    let size = 64u32;
    let mut frame = overlay_frame(size);
    frame.overlays.shapes = vec![textured_shape(tex)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (r, g, b) = centre_rgb(&px, size);
    assert!(
        r.abs_diff(g) < 30 && g.abs_diff(b) < 30,
        "expected flat grey background after free, got ({r}, {g}, {b})"
    );
}
