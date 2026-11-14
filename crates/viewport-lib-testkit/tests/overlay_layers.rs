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

/// The pixel at `[x, y]` of a `W` by `H` RGBA readback.
fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let at = ((y * W + x) * 4) as usize;
    [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
}

/// A red rect, then a layer over it whose first member is a destination-out
/// mask over part of the red and whose second is a green rect beside it.
fn masked_layer_frame(isolated: bool) -> FrameData {
    let mut fd = frame();
    let solid = |x: f32, y: f32, w: f32, h: f32, c: Colour| {
        OverlayShapeItem::new(OverlayShape::Rect { corner_radius: 0.0 }, [x, y], [w, h])
            .with_fill(OverlayFill::Solid(c))
    };
    fd.overlays.shapes.push(solid(
        40.0,
        40.0,
        80.0,
        60.0,
        Colour::srgb(1.0, 0.0, 0.0, 1.0),
    ));
    let mark = fd.overlays.layer_mark();
    fd.overlays.shapes.push(
        solid(50.0, 50.0, 30.0, 30.0, Colour::srgb(1.0, 1.0, 1.0, 1.0))
            .with_blend(OverlayBlend::DestinationOut)
            .with_z_order(1),
    );
    fd.overlays
        .shapes
        .push(solid(140.0, 50.0, 30.0, 30.0, Colour::srgb(0.0, 1.0, 0.0, 1.0)).with_z_order(1));
    fd.overlays
        .push_layer(mark, OverlayLayerStyle::default().with_isolated(isolated));
    fd
}

/// An isolated layer at opacity 1 keeps a destination-out member inside its
/// own target: the red under the layer stays red. Without isolation the layer
/// folds away as before and the mask cuts through the red.
#[test]
fn an_isolated_layer_keeps_a_destination_mask_inside_it() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let fd = masked_layer_frame(true);
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(stats.overlay_layers_composited, 1);
    assert_eq!(stats.overlay_layer_targets, 1);
    let isolated = h.render(&fd, W, H);
    let under = pixel(&isolated, 65, 65);
    assert!(
        under[0] > 240 && under[1] < 15 && under[2] < 15 && under[3] == 255,
        "the red under an isolated mask is kept: {under:?}"
    );
    let green = pixel(&isolated, 155, 65);
    assert!(
        green[1] > 240 && green[0] < 15,
        "the layer's content draws: {green:?}"
    );

    let fd = masked_layer_frame(false);
    let stats = h.render_two_frames(&fd, W, H);
    assert_eq!(stats.overlay_layers_composited, 0);
    assert_eq!(stats.overlay_layer_targets, 0);
    let folded = h.render(&fd, W, H);
    let cut = pixel(&folded, 65, 65);
    assert!(
        cut[0] < 15 || cut[3] < 15,
        "without isolation the mask reaches the red under the layer: {cut:?}"
    );
    assert_eq!(pixel(&folded, 155, 65), green, "the content draws the same");
    assert_eq!(pixel(&folded, 110, 90), pixel(&isolated, 110, 90));
}

/// An isolated layer that would otherwise pass through costs a target and a
/// composite.
#[test]
fn an_isolated_pass_through_layer_costs_its_composite() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut fd = frame();
    push_pair(&mut fd, 10.0, 10.0, OverlayLayerStyle::default());
    let plain = h.render_two_frames(&fd, W, H);
    assert_eq!(plain.overlay_layers_composited, 0);
    let mut fd = frame();
    push_pair(
        &mut fd,
        10.0,
        10.0,
        OverlayLayerStyle::default().with_isolated(true),
    );
    let isolated = h.render_two_frames(&fd, W, H);
    assert_eq!(isolated.overlay_layers_composited, 1);
    assert_eq!(isolated.overlay_layer_targets, 1);
    assert_eq!(isolated.overlay_draw_calls, plain.overlay_draw_calls + 1);
}
