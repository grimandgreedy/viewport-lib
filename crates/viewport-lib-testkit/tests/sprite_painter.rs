//! Sprites in painter's order: where they draw and how they pick.
//!
//! A painter batch draws after the opaque scene in its own sorted order and
//! never through order-independent transparency, so a frame holding only
//! painter sprites runs no OIT pass. Picking follows the drawn order: of two
//! overlapping painter sprites the one drawn on top is hit, and the hit names
//! it by the caller's index, not the sorted one.

use glam::Vec2;
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, PickBackend, PickId, PickMask, Plane2d, SceneFrame,
    SubObjectRef,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteOrder, SpriteSizeMode};
use viewport_lib_testkit::Harness;

const W: u32 = 200;
const H: u32 = 150;

/// One world unit per pixel, origin in the middle of the view.
fn frame(sprites: Vec<SpriteItem>) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = sprites;
    FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    )
}

/// Two overlapping blended squares; sprite 0 sits lower on screen.
fn pair(order: SpriteOrder) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[0.0, -10.0, 0.0], [10.0, 10.0, 0.0]];
    item.colours = vec![
        Colour::linear(0.9, 0.2, 0.2, 0.9),
        Colour::linear(0.2, 0.2, 0.9, 0.9),
    ];
    item.default_size = 40.0;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.order = order;
    // `-y`: the lower sprite (index 0) sorts last and draws on top, although
    // it was submitted first.
    item.sort_keys = item.positions.iter().map(|p| -p[1]).collect();
    item.settings.pick_id = PickId(42);
    item
}

const PAINTER: SpriteOrder = SpriteOrder::Painter {
    layer: 0,
    depth_test: true,
};

/// The passes that encoded for `frame`.
fn passes(harness: &mut Harness, frame: &FrameData) -> Vec<&'static str> {
    harness.render(frame, W, H);
    harness.renderer.debug_viewport_passes_ran(0)
}

#[test]
fn painter_sprites_issue_no_oit_pass() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let painter = passes(&mut harness, &frame(vec![pair(PAINTER)]));
    assert!(
        !painter.contains(&"oit"),
        "painter sprites must not open the OIT pass: {painter:?}"
    );
    assert!(
        painter.contains(&"depth_read"),
        "painter sprites draw in the read-only-depth pass: {painter:?}"
    );

    // The same sprites in the default order composite through OIT.
    let blended = passes(&mut harness, &frame(vec![pair(SpriteOrder::Depth)]));
    assert!(
        blended.contains(&"oit"),
        "blended sprites in depth order go through OIT: {blended:?}"
    );
}

#[test]
fn painter_pick_hits_the_sprite_drawn_on_top() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let frame = frame(vec![pair(PAINTER)]);
    let _ = harness.render(&frame, W, H);

    // World (5, 0) is inside both squares; it is pixel (105, 75).
    let at = Vec2::new(W as f32 * 0.5 + 5.0, H as f32 * 0.5);
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let hit = harness
        .renderer
        .pick_object(
            PickBackend::Gpu,
            at,
            &frame,
            &device,
            &queue,
            PickMask::OBJECT | PickMask::INSTANCE,
        )
        .expect("the overlap is covered by both sprites");
    assert_eq!(hit.id, 42);
    assert_eq!(
        hit.sub_object,
        Some(SubObjectRef::Instance(0)),
        "the hit is the sprite drawn on top, named by the caller's index"
    );
}
