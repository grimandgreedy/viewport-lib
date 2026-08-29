//! Pre-positioned glyph runs, end to end.
//!
//! `GlyphRunItem` hands the renderer glyph ids and positions directly instead of
//! a string, so these tests render the real overlay pass offscreen and confirm a
//! run rasterizes and draws its glyphs, and that an empty run draws nothing.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{GlyphRunItem, PositionedGlyph};

/// A frame looking at nothing, flat grey background, chrome off.
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

/// A grid of glyph ids from the built-in font, laid out by the caller (which is
/// the whole point of a glyph run). Low ids in Inter cover letters and digits, so
/// a spread of them is guaranteed to include visible glyphs without needing a
/// char-to-glyph lookup.
fn glyph_grid() -> Vec<PositionedGlyph> {
    let mut glyphs = Vec::new();
    let mut id: u16 = 4;
    for row in 0..4 {
        for col in 0..8 {
            let x = 4.0 + col as f32 * 15.0;
            let y = 22.0 + row as f32 * 30.0; // baseline of each row
            glyphs.push(PositionedGlyph::new(id, x, y));
            id += 1;
        }
    }
    glyphs
}

/// Count near-white pixels (drawn glyph coverage) against the grey background.
fn bright_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[0] > 230 && p[1] > 230 && p[2] > 230)
        .count()
}

/// Count strongly red and strongly blue pixels.
fn red_blue_pixels(px: &[u8]) -> (usize, usize) {
    let red = px
        .chunks_exact(4)
        .filter(|p| p[0] > 180 && p[1] < 90 && p[2] < 90)
        .count();
    let blue = px
        .chunks_exact(4)
        .filter(|p| p[2] > 180 && p[0] < 90 && p[1] < 90)
        .count();
    (red, blue)
}

/// A run of positioned glyphs rasterizes and draws: some near-white pixels land
/// on the grey background.
#[test]
fn glyph_run_draws_its_glyphs() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 128u32;
    let mut frame = overlay_frame(size);
    frame.overlays.glyph_runs = vec![
        GlyphRunItem::new(glyph_grid())
            .with_font_size(24.0)
            .with_colour([1.0, 1.0, 1.0, 1.0]),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let bright = bright_pixels(&px);
    assert!(
        bright > 20,
        "expected the glyph run to draw, got {bright} bright pixels"
    );
}

/// A run with no glyphs draws nothing: the frame stays the grey background.
#[test]
fn empty_glyph_run_draws_nothing() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 128u32;
    let mut frame = overlay_frame(size);
    frame.overlays.glyph_runs = vec![
        GlyphRunItem::new(Vec::new())
            .with_font_size(24.0)
            .with_colour([1.0, 1.0, 1.0, 1.0]),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert_eq!(bright_pixels(&px), 0, "an empty run must not draw");
}

/// Per-glyph `colours` route through layout: a run whose first half is red and
/// second half blue draws both colours.
#[test]
fn per_glyph_colours_apply() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 128u32;
    let glyphs = glyph_grid();
    let half = glyphs.len() / 2;
    let colours: Vec<[f32; 4]> = (0..glyphs.len())
        .map(|i| {
            if i < half {
                [1.0, 0.0, 0.0, 1.0]
            } else {
                [0.0, 0.0, 1.0, 1.0]
            }
        })
        .collect();

    let mut frame = overlay_frame(size);
    frame.overlays.glyph_runs = vec![
        GlyphRunItem::new(glyphs)
            .with_font_size(24.0)
            .with_colours(colours),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (red, blue) = red_blue_pixels(&px);
    assert!(
        red > 5 && blue > 5,
        "expected both red and blue glyphs, got red {red} blue {blue}"
    );
}
