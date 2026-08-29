//! Vector overlay shapes (`OverlayShape::Vector`), end to end.
//!
//! Renders the real overlay pass offscreen and reads back pixels to confirm an
//! arbitrary filled path draws: a square with a square hole reads empty in the
//! hole under the even-odd rule and filled under non-zero, gradients vary across
//! the shape, and an outline border draws.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{FillRule, OverlayFill, OverlayShape, OverlayShapeItem, SubPath};

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

/// Outer square (16..48) with an inner square hole (26..38). With the item at
/// position [0,0] and size [64,64], path-local coordinates equal screen pixels,
/// so the hole sits over the centre.
fn square_with_hole(fill_rule: FillRule) -> OverlayShapeItem {
    let outer = SubPath::polygon(&[[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]]);
    let hole = SubPath::polygon(&[[26.0, 26.0], [38.0, 26.0], [38.0, 38.0], [26.0, 38.0]]);
    OverlayShapeItem::vector(vec![outer, hole], fill_rule, [0.0, 0.0], [64.0, 64.0])
        .with_fill(OverlayFill::Solid([1.0, 0.0, 0.0, 1.0]))
}

/// Read a pixel as (r, g, b).
fn rgb_at(px: &[u8], size: u32, x: u32, y: u32) -> (u8, u8, u8) {
    let i = ((y * size + x) * 4) as usize;
    (px[i], px[i + 1], px[i + 2])
}

/// Even-odd: the inner square is a hole. Centre reads the grey background, a
/// point in the ring reads the red fill.
#[test]
fn even_odd_hole_reads_background() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 64u32;
    let mut frame = overlay_frame(size);
    frame.overlays.shapes = vec![square_with_hole(FillRule::EvenOdd)];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (hr, _, _) = rgb_at(&px, size, 32, 32); // hole centre
    let (rr, rg, rb) = rgb_at(&px, size, 20, 32); // ring
    assert!(
        hr < 200,
        "hole centre should be background grey, got r={hr}"
    );
    assert!(
        rr > 200 && rg < 100 && rb < 100,
        "ring should be red, got rgb ({rr}, {rg}, {rb})"
    );
}

/// Non-zero with both contours wound the same way: the inner region has winding
/// 2, still non-zero, so the hole fills. Centre reads red.
#[test]
fn nonzero_fills_the_hole() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 64u32;
    let mut frame = overlay_frame(size);
    frame.overlays.shapes = vec![square_with_hole(FillRule::NonZero)];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (r, g, b) = rgb_at(&px, size, 32, 32);
    assert!(
        r > 200 && g < 100 && b < 100,
        "non-zero should fill the hole red, got rgb ({r}, {g}, {b})"
    );
}

/// A left-to-right gradient across a filled square differs between its left and
/// right edges.
#[test]
fn gradient_varies_across_shape() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 64u32;
    let mut frame = overlay_frame(size);
    let square = SubPath::polygon(&[[8.0, 8.0], [56.0, 8.0], [56.0, 56.0], [8.0, 56.0]]);
    frame.overlays.shapes = vec![
        OverlayShapeItem::vector(vec![square], FillRule::NonZero, [0.0, 0.0], [64.0, 64.0])
            .with_fill(OverlayFill::LinearGradient {
                start_colour: [1.0, 0.0, 0.0, 1.0],
                end_colour: [0.0, 0.0, 1.0, 1.0],
                angle: 0.0,
            }),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (lr, _, lb) = rgb_at(&px, size, 12, 32);
    let (rr, _, rb) = rgb_at(&px, size, 52, 32);
    assert!(
        lr > rr && rb > lb,
        "left should be redder, right bluer: left ({lr},_,{lb}) right ({rr},_,{rb})"
    );
}

/// A vector shape with a transparent fill and a green outline border: the edge
/// reads green (the stroked contour), the interior reads the background.
#[test]
fn border_outline_draws() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let size = 64u32;
    let mut frame = overlay_frame(size);
    let square = SubPath::polygon(&[[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]]);
    frame.overlays.shapes = vec![
        OverlayShapeItem::vector(vec![square], FillRule::NonZero, [0.0, 0.0], [64.0, 64.0])
            .with_fill(OverlayFill::Solid([0.0, 0.0, 0.0, 0.0]))
            .with_border([0.0, 1.0, 0.0, 1.0], 4.0),
    ];

    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let (er, eg, eb) = rgb_at(&px, size, 16, 32); // left edge, on the stroke
    let (cr, cg, cb) = rgb_at(&px, size, 32, 32); // interior, transparent fill
    assert!(
        eg > 150 && er < 120 && eb < 120,
        "border edge should be green, got rgb ({er}, {eg}, {eb})"
    );
    assert!(
        (cr as i32 - cg as i32).abs() < 30 && (cg as i32 - cb as i32).abs() < 30,
        "interior should be neutral background, got rgb ({cr}, {cg}, {cb})"
    );
}
