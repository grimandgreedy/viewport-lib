//! Small and retained vector shapes draw their fills from cached coverage
//! cells, and a static shape does not rasterise again frame after frame.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::Colour;
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{FillRule, OverlayFill, OverlayShapeItem, RetainedOverlay, SubPath};

fn overlay_frame(size: u32) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame
}

fn star() -> OverlayShapeItem {
    let pts: Vec<[f32; 2]> = (0..10)
        .map(|i| {
            let a = i as f32 * std::f32::consts::PI / 5.0 - std::f32::consts::FRAC_PI_2;
            let r = if i % 2 == 0 { 20.0 } else { 8.0 };
            [20.0 + a.cos() * r, 20.0 + a.sin() * r]
        })
        .collect();
    OverlayShapeItem::vector(
        vec![SubPath::polygon(&pts)],
        FillRule::NonZero,
        [10.0, 10.0],
        [40.0, 40.0],
    )
    .with_fill(OverlayFill::Solid(Colour::linear(1.0, 1.0, 1.0, 1.0)))
}

fn lit_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4).filter(|p| p[0] > 128).count()
}

#[test]
fn an_immediate_vector_shape_rasterises_once() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64;
    let mut frame = overlay_frame(size);
    frame.overlays.shapes = vec![star()];

    let first = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert_eq!(renderer.last_frame_stats().overlay_vector_rasters, 1);
    assert!(lit_pixels(&first) > 50, "the star drew");

    // The same shape again, and moved by whole pixels: both reuse the cell.
    let second = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert_eq!(renderer.last_frame_stats().overlay_vector_rasters, 0);
    assert_eq!(first, second);
    frame.overlays.shapes = vec![star().with_position([13.0, 12.0])];
    let _ = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert_eq!(renderer.last_frame_stats().overlay_vector_rasters, 0);
}

#[test]
fn a_retained_vector_shape_draws_from_its_compiled_cell() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64;
    let id = renderer.compile_overlay_geometry(&device, &queue, &[], &[star()], &[], &[], 1.0);
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];

    for _ in 0..2 {
        let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
        let stats = renderer.last_frame_stats();
        assert_eq!(
            stats.overlay_vector_rasters, 0,
            "rasterised at compile only"
        );
        assert_eq!(stats.overlay_retained_reemitted, 0);
        assert!(lit_pixels(&px) > 50, "the star drew");
    }
}
