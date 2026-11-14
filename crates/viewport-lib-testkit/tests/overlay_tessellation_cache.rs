//! The tessellation cache for immediate vector shapes and polylines.
//!
//! An item whose content does not change keeps its tessellated geometry, so it
//! tessellates on its first frame only. Translation is not part of the cache
//! key, so an item that only moves keeps hitting the cache too.

use viewport_lib::{
    AnimTrack, CameraFrame, Colour, FillRule, FrameData, OverlayAnimations, OverlayFill,
    OverlayPolylineItem, OverlayShape, OverlayShapeItem, SceneFrame, SubPath,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 320;
const H: u32 = 240;

fn star() -> OverlayShape {
    let pts: Vec<[f32; 2]> = (0..10)
        .map(|i| {
            let a = i as f32 * std::f32::consts::PI / 5.0;
            let r = if i % 2 == 0 { 40.0 } else { 16.0 };
            [40.0 + a.cos() * r, 40.0 + a.sin() * r]
        })
        .collect();
    OverlayShape::Vector {
        subpaths: vec![SubPath::polygon(&pts)],
        fill_rule: FillRule::NonZero,
        stroke: None,
        inverse: false,
    }
}

fn frame(time: f64, shapes: Vec<OverlayShapeItem>) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 5.0, 0.7, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]).with_pixels_per_point(1.0),
        SceneFrame::from_surface_items(Vec::new()),
    );
    fd.viewport.show_axes_indicator = false;
    fd.overlays.time = time;
    fd.overlays.shapes = shapes;
    fd.overlays.polylines = vec![
        OverlayPolylineItem::new(vec![[10.0, 200.0], [90.0, 180.0], [170.0, 220.0]])
            .with_thickness(4.0),
    ];
    fd
}

#[test]
fn a_static_shape_tessellates_once() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let shape = OverlayShapeItem::new(star(), [20.0, 20.0], [80.0, 80.0])
        .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.4, 0.2, 1.0)));
    h.render(&frame(0.0, vec![shape.clone()]), W, H);
    assert_eq!(
        h.stats().overlay_tessellations,
        2,
        "the shape and the polyline tessellate on their first frame"
    );
    for _ in 0..3 {
        h.render(&frame(0.0, vec![shape.clone()]), W, H);
        assert_eq!(h.stats().overlay_tessellations, 0);
    }
    // A changed fill is new content.
    let recoloured = shape.with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.4, 0.9, 1.0)));
    h.render(&frame(0.0, vec![recoloured]), W, H);
    assert_eq!(h.stats().overlay_tessellations, 1);
}

#[test]
fn a_moving_shape_reuses_its_tessellation() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // A translate track moves the shape every frame; the shape itself and its
    // rotation and scale stay the same.
    let anims = OverlayAnimations::default().with_translate(AnimTrack::new(
        0.0,
        1.0,
        [20.0, 20.0],
        [220.0, 120.0],
    ));
    let shape = OverlayShapeItem::new(star(), [20.0, 20.0], [80.0, 80.0])
        .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.4, 0.2, 1.0)))
        .with_animations(anims);
    h.render(&frame(0.0, vec![shape.clone()]), W, H);
    assert_eq!(h.stats().overlay_tessellations, 2);
    let mut moved = Vec::new();
    for i in 1..5 {
        h.render(&frame(i as f64 * 0.13, vec![shape.clone()]), W, H);
        assert_eq!(
            h.stats().overlay_tessellations,
            0,
            "frame {i}: a pure move is not new content"
        );
        moved = h.render(&frame(i as f64 * 0.13, vec![shape.clone()]), W, H);
    }
    // The reused geometry lands where a fresh tessellation would.
    let Some(mut fresh) = Harness::new() else {
        return;
    };
    let direct = fresh.render(&frame(4.0 * 0.13, vec![shape]), W, H);
    assert_eq!(fresh.stats().overlay_tessellations, 2);
    let worst = moved
        .iter()
        .zip(&direct)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    assert!(worst <= 2, "moved geometry differs by {worst}");
}
