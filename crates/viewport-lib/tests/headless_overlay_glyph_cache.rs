//! The glyph atlas caches what it rasterises: static text rasterises nothing
//! after its first frame, a retained label nothing after it was compiled, and
//! a scaled label rasterises at its drawn size rather than magnifying the
//! unscaled glyphs.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::Colour;
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{AnchorX, AnchorY, LabelItem, RetainedOverlay};

const SIZE: u32 = 96;

fn overlay_frame() -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame
}

fn label(position: [f32; 2]) -> LabelItem {
    LabelItem::new("Cached")
        .with_position(position)
        .with_font_size(13.0)
        .with_colour(Colour::linear(1.0, 1.0, 1.0, 1.0))
}

fn lit_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4).filter(|p| p[0] > 128).count()
}

#[test]
fn a_static_label_rasterises_only_on_its_first_frame() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = overlay_frame();
    frame.overlays.labels = vec![label([10.0, 10.0])];

    let first = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert!(renderer.last_frame_stats().overlay_glyph_rasters > 0);
    assert!(lit_pixels(&first) > 20, "the label drew");

    let second = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(renderer.last_frame_stats().overlay_glyph_rasters, 0);
    assert_eq!(first, second);

    // A whole-pixel move lands every pen on the same quarter, so it reuses
    // the cells and draws the same pixels one column over.
    frame.overlays.labels = vec![label([11.0, 10.0])];
    let moved = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(renderer.last_frame_stats().overlay_glyph_rasters, 0);
    let row = SIZE as usize * 4;
    for y in 0..SIZE as usize {
        assert_eq!(
            &first[y * row..y * row + row - 4],
            &moved[y * row + 4..y * row + row],
            "row {y}"
        );
    }
}

#[test]
fn a_retained_label_draws_from_its_compiled_cells() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let id = renderer.compile_overlay_geometry(
        &device,
        &queue,
        &[],
        &[],
        &[],
        &[label([10.0, 10.0])],
        1.0,
    );
    let mut frame = overlay_frame();
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    for _ in 0..2 {
        let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
        let stats = renderer.last_frame_stats();
        assert_eq!(stats.overlay_glyph_rasters, 0, "rasterised at compile only");
        assert_eq!(stats.overlay_retained_reemitted, 0);
        assert!(lit_pixels(&px) > 20, "the label drew");
    }
}

/// A label at 13px scaled 2x about its top-left draws what a 26px label at
/// the same place draws: the scale picked the raster size.
#[test]
fn a_scaled_label_rasterises_at_its_drawn_size() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Placed by its top-left, so both boxes start at the same point.
    let base = label([10.0, 10.0])
        .with_align_x(AnchorX::Left)
        .with_align_y(AnchorY::Top);
    let m = renderer
        .resources()
        .measure_overlay_text(&base.text, 13.0, None);
    let scaled = base
        .clone()
        .with_scale(2.0)
        .with_rotation_pivot([-m.width * 0.5, -m.height * 0.5]);
    let large = base.with_font_size(26.0);

    let mut frame = overlay_frame();
    frame.overlays.labels = vec![large];
    let want = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    frame.overlays.labels = vec![scaled];
    let got = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
    assert_eq!(
        renderer.last_frame_stats().overlay_glyph_rasters,
        0,
        "the scaled label reuses the 26px cells"
    );
    let differ = want
        .chunks_exact(4)
        .zip(got.chunks_exact(4))
        .filter(|(a, b)| a[0].abs_diff(b[0]) > 8)
        .count();
    assert!(lit_pixels(&want) > 100);
    assert_eq!(differ, 0, "the scaled label differs from the 26px one");
}
