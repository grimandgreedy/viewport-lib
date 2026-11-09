//! Overlay draw-call counters in `FrameStats`.
//!
//! A widget toolkit lowering a panel gives every command its own `z_order` to
//! keep painter's order, so a panel of alternating shapes and labels asks for a
//! draw per alternation under the ordering rule the overlay pass uses today:
//! segments batch only at an equal `z_order`. These assertions pin that count,
//! and the count the same panel takes when its items share a `z_order`, so a
//! change to the ordering rule shows up here as a visible diff rather than as a
//! silent regression or an unnoticed improvement.

use viewport_lib::{
    CameraFrame, Colour, FrameData, LabelItem, OverlayFill, OverlayShape, OverlayShapeItem,
    SceneFrame,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 320;
const H: u32 = 240;
const ROWS: i32 = 12;

/// `ROWS` rows of a backing rect and a label. With `distinct_z` every item
/// takes the next `z_order`, the way a toolkit lowers a draw list; otherwise
/// every item sits at zero and the families draw in their fixed order.
fn frame(distinct_z: bool) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 5.0, 0.7, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]).with_pixels_per_point(1.0),
        SceneFrame::from_surface_items(Vec::new()),
    );
    // The axes indicator is drawn as overlay items of its own and would add
    // its segments to the count.
    fd.viewport.show_axes_indicator = false;
    let mut z = 0;
    for row in 0..ROWS {
        let y = 8.0 + row as f32 * 18.0;
        let rect = OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 3.0 },
            [8.0, y],
            [200.0, 16.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.3, 0.5, 1.0)));
        let label = LabelItem::new(format!("row {row}")).with_position([12.0, y + 2.0]);
        if distinct_z {
            fd.overlays.shapes.push(rect.with_z_order(z));
            fd.overlays.labels.push(label.with_z_order(z + 1));
            z += 2;
        } else {
            fd.overlays.shapes.push(rect);
            fd.overlays.labels.push(label);
        }
    }
    fd
}

#[test]
fn alternating_panel_draw_counts() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };

    // Shared `z_order`: the shapes draw as one batch and the text as another.
    let stats = h.render_two_frames(&frame(false), W, H);
    assert_eq!(
        stats.overlay_draw_calls, 2,
        "two family batches at one z_order"
    );

    // A distinct `z_order` per item: one draw per alternation of family. The
    // items do not overlap, so an ordering that batched by overlap rather than
    // by equal `z_order` would bring this back to two; update the literal when
    // it does.
    let stats = h.render_two_frames(&frame(true), W, H);
    assert_eq!(
        stats.overlay_draw_calls,
        (2 * ROWS) as u32,
        "one draw per alternation under equal-z batching"
    );
}

/// A texture fill binds its image per draw on both the shape and the text
/// pipeline, so items filled from one image batch together and the families
/// split only where the bound image changes. The rows submit, at one
/// `z_order`, an untextured shape and two shapes sharing an image, then a
/// plain label, two labels sharing that image, and another plain label.
#[test]
fn texture_fills_batch_per_image() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let image = h
        .renderer
        .resources_mut()
        .upload_overlay_texture(
            &h.device,
            &h.queue,
            viewport_lib::TextureData::srgb(2, 2, vec![255u8; 16]),
        )
        .expect("a well-formed sRGB image");
    let mut fd = frame(false);
    fd.overlays.shapes.clear();
    fd.overlays.labels.clear();
    let textured = OverlayFill::texture(image);
    for (i, fill) in [
        OverlayFill::Solid(Colour::srgb(0.2, 0.3, 0.5, 1.0)),
        textured.clone(),
        textured.clone(),
    ]
    .into_iter()
    .enumerate()
    {
        fd.overlays.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 3.0 },
                [8.0, 8.0 + i as f32 * 18.0],
                [120.0, 16.0],
            )
            .with_fill(fill),
        );
    }
    let white = OverlayFill::Solid(Colour::srgb(1.0, 1.0, 1.0, 1.0));
    for (i, fill) in [white.clone(), textured.clone(), textured, white]
        .into_iter()
        .enumerate()
    {
        fd.overlays.labels.push(
            LabelItem::new(format!("label {i}"))
                .with_position([150.0, 8.0 + i as f32 * 18.0])
                .with_fill(fill),
        );
    }
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(
        stats.overlay_draw_calls,
        2 + 3,
        "shapes: untextured, then the shared image; text: plain, image, plain"
    );
}
