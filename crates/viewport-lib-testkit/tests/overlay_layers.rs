//! Overlay layer counters in `FrameStats`.
//!
//! A layer costs a scratch target only when it has to: these pin the fold
//! rules (a one-item layer and a layer that changes nothing draw their members
//! straight into the frame), the pooling (a frame whose layers fit last
//! frame's targets allocates nothing), what a composited layer adds to the
//! overlay draw count, and how nested layers use the pool: one level at a
//! time, so a chain N deep holds N targets.

use viewport_lib::{
    CameraFrame, Colour, FrameData, NoticeDetail, OverlayBlend, OverlayFill, OverlayLayer,
    OverlayLayerMembers, OverlayLayerStyle, OverlayShape, OverlayShapeItem, SceneFrame,
    ShadowLayer,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 320;
const H: u32 = 240;

fn frame() -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 5.0, 0.7, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]).with_pixels_per_point(1.0),
        SceneFrame::from_surface_items(Vec::new()),
    );
    // The axes indicator draws overlay items of its own.
    fd.viewport.show_axes_indicator = false;
    fd
}

fn rect(x: f32, y: f32) -> OverlayShapeItem {
    OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 4.0 },
        [x, y],
        [40.0, 30.0],
    )
    .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.5, 0.8, 1.0)))
}

/// Two overlapping rects at `x`, `y` as a layer drawn with `style`.
fn push_pair(fd: &mut FrameData, x: f32, y: f32, style: OverlayLayerStyle) {
    let mark = fd.overlays.layer_mark();
    fd.overlays.shapes.push(rect(x, y));
    fd.overlays.shapes.push(rect(x + 20.0, y + 10.0));
    fd.overlays.push_layer(mark, style);
}

#[test]
fn layers_that_fold_allocate_no_target() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = frame();
    // One item, faded and filtered: its opacity, blend and colour matrix
    // ride the item.
    let mark = fd.overlays.layer_mark();
    fd.overlays.shapes.push(rect(10.0, 10.0));
    fd.overlays.push_layer(
        mark,
        OverlayLayerStyle::default()
            .with_opacity(0.5)
            .with_blend(OverlayBlend::Multiply)
            .with_colour_matrix(viewport_lib::ColourMatrix::greyscale()),
    );
    // Overlapping items at full opacity with nothing else: drawn as they are.
    push_pair(&mut fd, 100.0, 10.0, OverlayLayerStyle::default());

    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(stats.overlay_layers_composited, 0);
    assert_eq!(stats.overlay_layer_targets, 0);
    assert_eq!(stats.overlay_layer_target_allocations, 0);
}

#[test]
fn pooled_targets_are_reused() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = frame();
    let faded = OverlayLayerStyle::default().with_opacity(0.5);
    // Two faded layers apart, which share a target; a third over the first,
    // which needs one of its own; and a shadowed layer, which always has its
    // own.
    push_pair(&mut fd, 10.0, 10.0, faded.clone());
    push_pair(&mut fd, 200.0, 10.0, faded.clone());
    push_pair(&mut fd, 20.0, 20.0, faded);
    push_pair(
        &mut fd,
        100.0,
        150.0,
        OverlayLayerStyle::default().with_shadow(ShadowLayer::new(
            Colour::srgb(0.0, 0.0, 0.0, 0.6),
            4.0,
            [4.0, 4.0],
        )),
    );

    let _ = h.render(&fd, W, H);
    let first = h.stats();
    assert_eq!(first.overlay_layers_composited, 4);
    assert_eq!(first.overlay_layer_targets, 3);
    assert!(first.overlay_layer_target_allocations > 0);

    let _ = h.render(&fd, W, H);
    let second = h.stats();
    assert_eq!(second.overlay_layer_targets, 3);
    assert_eq!(
        second.overlay_layer_target_allocations, 0,
        "the second frame reuses the pooled targets"
    );
}

#[test]
fn a_composited_layer_costs_one_draw() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // The pair on its own: one shape draw.
    let mut fd = frame();
    fd.overlays.shapes.push(rect(10.0, 10.0));
    fd.overlays.shapes.push(rect(30.0, 20.0));
    let plain = h.render_two_frames(&fd, W, H).overlay_draw_calls;
    assert_eq!(plain, 1);

    // As a faded layer: the members' draw into the target, then the composite.
    let mut fd = frame();
    push_pair(
        &mut fd,
        10.0,
        10.0,
        OverlayLayerStyle::default().with_opacity(0.5),
    );
    assert_eq!(h.render_two_frames(&fd, W, H).overlay_draw_calls, plain + 1);

    // With a backdrop, the backdrop draw as well.
    let mut fd = frame();
    push_pair(
        &mut fd,
        10.0,
        10.0,
        OverlayLayerStyle::default()
            .with_backdrop(viewport_lib::BackdropEffects::default().with_blur(4.0)),
    );
    assert_eq!(h.render_two_frames(&fd, W, H).overlay_draw_calls, plain + 2);
}

/// Two overlapping rects at `x`, `y`, then whatever `inner` pushes, as a
/// layer drawn with `style`.
fn push_around(
    fd: &mut FrameData,
    x: f32,
    y: f32,
    style: OverlayLayerStyle,
    inner: impl FnOnce(&mut FrameData),
) {
    let mark = fd.overlays.layer_mark();
    fd.overlays.shapes.push(rect(x, y));
    fd.overlays.shapes.push(rect(x + 20.0, y + 10.0));
    inner(fd);
    fd.overlays.push_layer(mark, style);
}

#[test]
fn a_nested_pass_through_layer_allocates_nothing() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // A pass-through layer inside a pass-through layer: nothing composited.
    let mut fd = frame();
    push_around(&mut fd, 10.0, 10.0, OverlayLayerStyle::default(), |fd| {
        push_pair(fd, 30.0, 20.0, OverlayLayerStyle::default());
    });
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(stats.overlay_layers_composited, 0);
    assert_eq!(stats.overlay_layer_targets, 0);

    // Inside a faded layer: its members draw into the faded layer's target,
    // which is the only one.
    let mut fd = frame();
    push_around(
        &mut fd,
        10.0,
        10.0,
        OverlayLayerStyle::default().with_opacity(0.5),
        |fd| push_pair(fd, 30.0, 20.0, OverlayLayerStyle::default()),
    );
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(stats.overlay_layers_composited, 1);
    assert_eq!(stats.overlay_layer_targets, 1);
}

#[test]
fn nested_layers_hold_one_target_per_level() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let faded = || OverlayLayerStyle::default().with_opacity(0.8);
    // A chain three deep: three targets.
    let mut fd = frame();
    push_around(&mut fd, 10.0, 10.0, faded(), |fd| {
        push_around(fd, 20.0, 20.0, faded(), |fd| {
            push_pair(fd, 30.0, 30.0, faded());
        });
    });
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(stats.overlay_layers_composited, 3);
    assert_eq!(stats.overlay_layer_targets, 3);

    // Two levels of five layers: a faded group holding four disjoint faded
    // layers, which share the second level's target.
    let mut fd = frame();
    push_around(&mut fd, 10.0, 10.0, faded(), |fd| {
        for k in 0..4 {
            push_pair(fd, 10.0 + k as f32 * 75.0, 120.0, faded());
        }
    });
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(stats.overlay_layers_composited, 5);
    assert_eq!(stats.overlay_layer_targets, 2);
    assert_eq!(stats.overlay_layer_target_allocations, 0);
}

#[test]
fn a_nested_layer_costs_its_composite_and_its_run() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = frame();
    for (x, y) in [(10.0, 10.0), (30.0, 20.0), (20.0, 20.0), (40.0, 30.0)] {
        fd.overlays.shapes.push(rect(x, y));
    }
    let plain = h.render_two_frames(&fd, W, H).overlay_draw_calls;
    assert_eq!(plain, 1);
    // Each faded layer: its composite, and its members as a run of their own.
    let mut fd = frame();
    push_around(
        &mut fd,
        10.0,
        10.0,
        OverlayLayerStyle::default().with_opacity(0.5),
        |fd| {
            push_pair(
                fd,
                20.0,
                20.0,
                OverlayLayerStyle::default().with_opacity(0.5),
            )
        },
    );
    assert_eq!(h.render_two_frames(&fd, W, H).overlay_draw_calls, plain + 3);
}

#[test]
fn partly_overlapping_layers_are_reported() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = frame();
    for k in 0..4 {
        fd.overlays.shapes.push(rect(10.0 + k as f32 * 50.0, 10.0));
    }
    let faded = OverlayLayerStyle::default().with_opacity(0.5);
    fd.overlays.layers.push(OverlayLayer::new(
        OverlayLayerMembers::default().with_shapes(0..3),
        faded.clone(),
    ));
    fd.overlays.layers.push(OverlayLayer::new(
        OverlayLayerMembers::default().with_shapes(2..4),
        faded,
    ));
    let _ = h.renderer.take_notices();
    let stats = h.render_two_frames(&fd, W, H);
    // The earlier layer composites; the later is ignored.
    assert_eq!(stats.overlay_layers_composited, 1);
    let overlap = h
        .renderer
        .take_notices()
        .into_iter()
        .find_map(|n| match n.detail {
            NoticeDetail::OverlayLayersOverlap { ignored, kept } => Some((ignored, kept)),
            _ => None,
        });
    assert_eq!(overlap, Some((1, 0)));
}
