//! Measuring a label's box and resolving where it lands.
//!
//! A consumer sizing a backing shape, placing a leader line, or hit-testing a
//! label needs two things: how big the laid-out text is, and where the anchor
//! and alignment put it. `measure_overlay_text_wrapped` answers the first and
//! `LabelItem::resolve_top_left` the second. This renders the label and checks
//! the glyphs actually land in the box those two describe.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::LabelItem;

const SIZE: u32 = 128;
const FONT_SIZE: f32 = 16.0;
const TEXT: &str = "wrap this label over a few lines";
const MAX_WIDTH: f32 = 70.0;

/// A 128x128 frame on a black background, chrome off, so any lit pixel is ink.
fn overlay_frame() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0].into());
    frame
}

/// Bounding box of the lit pixels as `[x0, y0, x1, y1]`, or `None` for an empty
/// frame.
fn ink_bounds(px: &[u8]) -> Option<[f32; 4]> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    let mut any = false;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let i = ((y * SIZE + x) * 4) as usize;
            if px[i] > 40 || px[i + 1] > 40 || px[i + 2] > 40 {
                any = true;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    any.then(|| [x0 as f32, y0 as f32, x1 as f32, y1 as f32])
}

/// The measured box of a wrapped label, placed by `resolve_top_left`, contains
/// the glyphs the renderer draws, for several alignments.
#[test]
fn a_measured_wrapped_label_lands_in_its_resolved_box() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let metrics = renderer
        .resources()
        .measure_overlay_text_wrapped(TEXT, FONT_SIZE, None, MAX_WIDTH);
    assert!(
        metrics.width <= MAX_WIDTH && metrics.height > FONT_SIZE * 1.5,
        "the text should have wrapped to more than one line, got {metrics:?}"
    );

    let cases = [
        (AnchorX::Left, AnchorY::Top),
        (AnchorX::Middle, AnchorY::Middle),
        (AnchorX::Right, AnchorY::Bottom),
    ];
    for (ax, ay) in cases {
        let label = LabelItem::new(TEXT)
            .with_anchor(viewport_lib::OverlayAnchor::Viewport {
                x: AnchorX::Middle,
                y: AnchorY::Middle,
            })
            .with_font_size(FONT_SIZE)
            .with_max_width(MAX_WIDTH)
            .with_align_x(ax)
            .with_align_y(ay)
            .with_anchor_padding(0.0);

        let tl = label
            .resolve_top_left(
                [metrics.width, metrics.height],
                [SIZE as f32, SIZE as f32],
                &Camera::default().view_matrix(),
                &Camera::default().proj_matrix(),
            )
            .expect("a viewport-anchored label always resolves");

        let mut frame = overlay_frame();
        frame.overlays.labels = vec![label];
        let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);
        let ink = ink_bounds(&px).expect("the label should have drawn something");

        // One pixel of slack for anti-aliased edges: the glyphs sit inside the
        // laid-out box, which is what a backing shape would cover.
        let box_ = [tl[0], tl[1], tl[0] + metrics.width, tl[1] + metrics.height];
        assert!(
            ink[0] >= box_[0] - 1.0
                && ink[1] >= box_[1] - 1.0
                && ink[2] <= box_[2] + 1.0
                && ink[3] <= box_[3] + 1.0,
            "ink {ink:?} escaped the resolved box {box_:?} for align ({ax:?}, {ay:?})"
        );
        // And it fills most of it, so the box is not merely large enough.
        assert!(
            ink[2] - ink[0] > metrics.width * 0.5 && ink[3] - ink[1] > metrics.height * 0.4,
            "ink {ink:?} is too small for the resolved box {box_:?} for align ({ax:?}, {ay:?})"
        );
    }
}
