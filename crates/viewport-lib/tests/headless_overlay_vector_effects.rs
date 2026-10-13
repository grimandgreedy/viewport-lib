//! An `OverlayShape::Vector` draws an image fill and a backdrop blur, both cut
//! to the path rather than its bounding box.

use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    BackdropEffects, Colour, OverlayFill, OverlayFrame, OverlayShape, OverlayShapeItem,
    OverlayStyle, SubPath,
};

const SIZE: u32 = 96;

/// Inside the triangle.
const INSIDE: (u32, u32) = (28, 28);
/// On the left edge, which lies on the bounding box.
const BOX_EDGE: (u32, u32) = (16, 40);
/// Inside the bounding box, outside the triangle.
const OUTSIDE: (u32, u32) = (70, 70);

fn frame(overlays: OverlayFrame) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some(Colour::linear(0.5, 0.5, 0.5, 1.0));
    frame.overlays = overlays;
    frame
}

fn triangle(style: OverlayStyle) -> OverlayFrame {
    let mut item = OverlayShapeItem::new(
        OverlayShape::Vector {
            subpaths: vec![SubPath::polygon(&[
                [16.0, 16.0],
                [80.0, 16.0],
                [16.0, 80.0],
            ])],
            fill_rule: viewport_lib::FillRule::NonZero,
        },
        [0.0, 0.0],
        [96.0, 96.0],
    );
    item.style = style;
    let mut ovl = OverlayFrame::default();
    ovl.shapes = vec![item];
    ovl
}

fn pixel(px: &[u8], (x, y): (u32, u32)) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn same(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= 2)
}

#[test]
fn a_vector_shape_samples_its_image_fill() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let red = renderer
        .resources_mut()
        .upload_overlay_texture(
            &device,
            &queue,
            viewport_lib::TextureData::srgb(2, 2, [255u8, 0, 0, 255].repeat(4)),
        )
        .unwrap();

    let empty =
        renderer.render_offscreen(&device, &queue, &frame(OverlayFrame::default()), SIZE, SIZE);
    let style = OverlayStyle::default().with_fill(OverlayFill::texture(red));
    let px = renderer.render_offscreen(&device, &queue, &frame(triangle(style)), SIZE, SIZE);

    for at in [INSIDE, BOX_EDGE] {
        let p = pixel(&px, at);
        assert!(
            p[0] > 240 && p[1] < 16 && p[2] < 16,
            "the image fill should draw solid red at {at:?}, got {p:?}"
        );
    }
    assert!(
        same(pixel(&px, OUTSIDE), pixel(&empty, OUTSIDE)),
        "the image fill should stop at the path, not its bounding box"
    );
}

#[test]
fn a_vector_shape_blurs_its_backdrop() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let empty =
        renderer.render_offscreen(&device, &queue, &frame(OverlayFrame::default()), SIZE, SIZE);
    // The background is flat, so a blur alone leaves it as it is: the
    // brightness filter is what shows the backdrop was drawn.
    let style = OverlayStyle::default().with_backdrop(
        BackdropEffects::default()
            .with_blur(4.0)
            .with_filters(1.0, 0.25, 0.0),
    );
    let px = renderer.render_offscreen(&device, &queue, &frame(triangle(style)), SIZE, SIZE);

    let (inside, bg) = (pixel(&px, INSIDE), pixel(&empty, INSIDE));
    assert!(
        inside[0] + 40 < bg[0],
        "the filtered backdrop should darken the path's interior: {inside:?} against {bg:?}"
    );
    assert!(
        same(pixel(&px, OUTSIDE), pixel(&empty, OUTSIDE)),
        "the backdrop should stop at the path, not its bounding box"
    );
}
