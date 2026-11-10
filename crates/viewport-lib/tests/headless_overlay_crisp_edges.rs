//! Touching analytic rects drawn with `OverlayEdge::Crisp` leave no seam.
//!
//! An anti-aliased rect ramps its coverage over a one-pixel band inside its
//! outline, so two rects that share an edge each half-cover the pixels along it
//! and the background shows between them. A crisp rect covers a pixel exactly
//! when its centre is inside, so a grid of them covers every pixel in its area
//! once. Checked on the immediate and the compiled path, at a fractional
//! position and with every edge exactly on a pixel centre.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

use viewport_lib::Colour;
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{OverlayEdge, OverlayShape, OverlayShapeItem, OverlayStyle, RetainedOverlay};

const SIZE: u32 = 64;
const N: usize = 4;

/// A frame with a black background and the chrome off, at one pixel per point.
fn overlay_frame() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.0, 0.0, 0.0, 1.0));
    frame
}

/// A 4x4 grid of white rects, `cell` wide, from `origin` in both axes.
fn grid(origin: f32, cell: f32, edge: OverlayEdge) -> Vec<OverlayShapeItem> {
    let mut out = Vec::new();
    for j in 0..N {
        for i in 0..N {
            out.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [origin + i as f32 * cell, origin + j as f32 * cell],
                    [cell, cell],
                )
                .with_style(
                    OverlayStyle::solid(Colour::linear(1.0, 1.0, 1.0, 1.0)).with_edge(edge),
                ),
            );
        }
    }
    out
}

/// The pixel rows and columns whose centres the grid's area holds, half-open:
/// a centre on the far edge is outside.
fn covered_range(origin: f32, cell: f32) -> std::ops::Range<u32> {
    let end = origin + N as f32 * cell;
    let first = (origin - 0.5).ceil() as u32;
    let last = (end - 0.5).ceil() as u32;
    first..last
}

/// Linear coverage of white over black at each pixel of `range` squared.
fn coverage(px: &[u8], range: &std::ops::Range<u32>) -> Vec<f32> {
    let mut out = Vec::new();
    for y in range.clone() {
        for x in range.clone() {
            let v = px[((y * SIZE + x) * 4) as usize] as f32 / 255.0;
            out.push(Colour::srgb(v, v, v, 1.0).to_linear_rgba()[0]);
        }
    }
    out
}

fn full_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4).filter(|p| p[0] == 255).count()
}

fn render_immediate(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shapes: Vec<OverlayShapeItem>,
) -> Vec<u8> {
    let mut frame = overlay_frame();
    frame.overlays.shapes = shapes;
    renderer.render_offscreen(device, queue, &frame, SIZE, SIZE)
}

fn render_retained(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shapes: Vec<OverlayShapeItem>,
) -> Vec<u8> {
    let id = renderer.compile_overlay_geometry(device, queue, &[], &shapes, &[], &[], 1.0);
    let mut frame = overlay_frame();
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    renderer.render_offscreen(device, queue, &frame, SIZE, SIZE)
}

fn min(v: &[f32]) -> f32 {
    v.iter().copied().fold(f32::INFINITY, f32::min)
}

/// A crisp grid covers every pixel in its area fully and nothing outside it,
/// on both paths and for both placements.
#[test]
fn a_crisp_grid_covers_every_pixel_once() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // A fractional origin and cell, then every edge on a pixel centre (the
    // tie the half-open rule resolves).
    for (origin, cell) in [(10.37f32, 7.3f32), (10.5, 7.0)] {
        let range = covered_range(origin, cell);
        let expected = range.len() * range.len();
        for (path, px) in [
            (
                "immediate",
                render_immediate(
                    &mut renderer,
                    &device,
                    &queue,
                    grid(origin, cell, OverlayEdge::Crisp),
                ),
            ),
            (
                "retained",
                render_retained(
                    &mut renderer,
                    &device,
                    &queue,
                    grid(origin, cell, OverlayEdge::Crisp),
                ),
            ),
        ] {
            let c = coverage(&px, &range);
            assert_eq!(
                min(&c),
                1.0,
                "{path}, origin {origin}, cell {cell}: background shows through the grid"
            );
            assert_eq!(
                full_pixels(&px),
                expected,
                "{path}, origin {origin}, cell {cell}: the grid covers pixels outside its area"
            );
            assert!(
                px.chunks_exact(4).all(|p| p[0] == 255 || p[0] == 0),
                "{path}, origin {origin}, cell {cell}: a crisp grid has no partial pixels"
            );
        }
    }
}

/// The same grid anti-aliased shows a seam where cells meet: the problem the
/// crisp edge exists for.
#[test]
fn an_anti_aliased_grid_shows_seams() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let (origin, cell) = (10.37f32, 7.3f32);
    // Inside the grid's outer ring, so only shared edges are measured.
    let range = covered_range(origin, cell);
    let inner = (range.start + 1)..(range.end - 1);
    for (path, px) in [
        (
            "immediate",
            render_immediate(
                &mut renderer,
                &device,
                &queue,
                grid(origin, cell, OverlayEdge::AntiAliased),
            ),
        ),
        (
            "retained",
            render_retained(
                &mut renderer,
                &device,
                &queue,
                grid(origin, cell, OverlayEdge::AntiAliased),
            ),
        ),
    ] {
        let seam = min(&coverage(&px, &inner));
        eprintln!("{path}: lowest coverage on a shared edge {seam:.3}");
        assert!(
            seam < 0.5,
            "{path}: an anti-aliased grid should show background at its shared edges, \
             lowest coverage {seam}"
        );
    }
}
