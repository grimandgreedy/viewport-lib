//! Overlay draw-call counters in `FrameStats`.
//!
//! A widget toolkit lowering a panel gives every command its own `z_order` to
//! keep painter's order. The overlay pass only keeps that order between items
//! that overlap, so a panel whose rows do not overlap batches to one draw per
//! family whatever its `z_order` values, and only overlapping items that
//! alternate between families, images or blend modes split a batch. These
//! assertions pin those counts, so a change to the ordering rule shows up here
//! as a visible diff rather than as a silent regression or an unnoticed
//! improvement.

use viewport_lib::{
    CameraFrame, Colour, FrameData, LabelItem, OverlayBlend, OverlayFill, OverlayMeshItem,
    OverlayPolylineItem, OverlayShape, OverlayShapeItem, SceneFrame,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 320;
const H: u32 = 240;
const ROWS: i32 = 12;

/// `ROWS` rows of a backing rect and a label, `pitch` logical pixels apart.
/// With `distinct_z` every item takes the next `z_order`, the way a toolkit
/// lowers a draw list; otherwise every item sits at zero. At the default pitch
/// the rows do not overlap each other; a pitch under the rect height stacks
/// each row over the one before.
fn frame_with_pitch(distinct_z: bool, pitch: f32) -> FrameData {
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
        let y = 8.0 + row as f32 * pitch;
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

fn frame(distinct_z: bool) -> FrameData {
    frame_with_pitch(distinct_z, 18.0)
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

    // A distinct `z_order` per item. Each label overlaps its own rect, but no
    // row overlaps another, so every rect batches with the other rects and
    // every label with the other labels.
    let stats = h.render_two_frames(&frame(true), W, H);
    assert_eq!(
        stats.overlay_draw_calls, 2,
        "rows that do not overlap batch across z_order"
    );

    // The same alternation with each row stacked over the one before: every
    // item overlaps the item before it, so painter's order needs a draw per
    // change of family.
    let stats = h.render_two_frames(&frame_with_pitch(true, 4.0), W, H);
    assert_eq!(
        stats.overlay_draw_calls,
        (2 * ROWS) as u32,
        "overlapping items that alternate family split per alternation"
    );
}

/// A blend mode is pipeline state, so a rect that blends differently from the
/// others draws on its own even though nothing overlaps it.
#[test]
fn blend_mode_splits_a_batch() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = frame(true);
    let rect = &mut fd.overlays.shapes[ROWS as usize / 2];
    *rect = rect.clone().with_blend(OverlayBlend::Additive);
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(
        stats.overlay_draw_calls, 3,
        "source-over rects, the additive rect, the labels"
    );
}

/// A texture fill binds its image per draw on both the shape and the text
/// pipeline, so items filled from one image batch together and the families
/// split only where the bound image changes. The rows submit, at one
/// `z_order`, an untextured shape and two shapes sharing an image, then a
/// plain label, two labels sharing that image, and another plain label. No
/// item overlaps another, so the two plain labels batch although the
/// textured ones were submitted between them.
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
        2 + 2,
        "shapes: untextured, then the shared image; text: plain, then the image"
    );
}

/// A mesh draws through the text pipeline, so it batches with labels and
/// polylines at the same family rank: rows that stack a mesh, a polyline and
/// a label over each other, each at its own `z_order`, still draw as one.
#[test]
fn meshes_batch_with_text_and_polylines() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = frame(false);
    fd.overlays.shapes.clear();
    fd.overlays.labels.clear();
    let mut z = 0;
    for row in 0..ROWS {
        let y = 8.0 + row as f32 * 6.0;
        fd.overlays.meshes.push(
            OverlayMeshItem::new(
                vec![[0.0, 0.0], [120.0, 0.0], [120.0, 14.0], [0.0, 14.0]],
                vec![0, 1, 2, 0, 2, 3],
            )
            .with_colour(Colour::srgb(0.2, 0.3, 0.5, 1.0))
            .with_position([8.0, y])
            .with_z_order(z),
        );
        fd.overlays.polylines.push(
            OverlayPolylineItem::new(vec![[8.0, y + 7.0], [128.0, y + 7.0]])
                .with_thickness(2.0)
                .with_z_order(z + 1),
        );
        fd.overlays.labels.push(
            LabelItem::new(format!("row {row}"))
                .with_position([12.0, y + 1.0])
                .with_z_order(z + 2),
        );
        z += 3;
    }
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(
        stats.overlay_draw_calls, 1,
        "meshes, polylines and labels share the text pipeline"
    );
}
