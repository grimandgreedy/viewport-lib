//! Color glyph (emoji) rendering, end to end.
//!
//! Registers a system color-emoji font, submits one emoji glyph through a
//! `GlyphRunItem`, renders the overlay pass offscreen, and confirms the drawn
//! pixels carry real colour (not the monochrome coverage the tint path produces).
//! Skips when no GPU adapter or emoji font is available.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{GlyphRunItem, PositionedGlyph};

/// The system color-emoji font (sbix, PNG strikes).
const APPLE_EMOJI: &str = "/System/Library/Fonts/Apple Color Emoji.ttc";

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

/// Count pixels with a strong channel spread: the grey background is spread 0,
/// while a colour emoji (a yellow face) is strongly non-grey.
fn colourful_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| {
            let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
            let spread = (r - g).abs().max((g - b).abs()).max((r - b).abs());
            spread > 60
        })
        .count()
}

#[test]
fn color_emoji_draws_in_colour() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let Ok(bytes) = std::fs::read(APPLE_EMOJI) else {
        eprintln!("skipping: no color-emoji font at {APPLE_EMOJI}");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let handle = match renderer.resources_mut().upload_font(&bytes) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("skipping: emoji font did not upload: {e}");
            return;
        }
    };

    // Glyph id for a grinning face, from the same bytes the atlas rasterizes.
    let face = ttf_parser::Face::parse(&bytes, 0).expect("emoji font parses");
    let gid = face
        .glyph_index('\u{1F600}')
        .expect("grinning face present")
        .0;

    let size = 128u32;
    let mut frame = overlay_frame(size);
    frame.overlays.glyph_runs = vec![
        GlyphRunItem::new(vec![PositionedGlyph::new(gid, 20.0, 96.0)])
            .with_font(handle)
            .with_font_size(64.0)
            // White tint: for a colour glyph the tint RGB is ignored, so a coloured
            // result proves the atlas RGBA is drawn, not a tinted coverage mask.
            .with_colour([1.0, 1.0, 1.0, 1.0]),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let colourful = colourful_pixels(&px);
    assert!(
        colourful > 30,
        "expected the colour emoji to draw with real colour, got {colourful} colourful pixels"
    );
}
