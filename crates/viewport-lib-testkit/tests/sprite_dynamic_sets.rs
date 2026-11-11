//! Uploaded sprite sets that change after upload: ranged writes into a set
//! drawn in painter's order, the re-sort a key write asks for, and per-frame
//! overrides on an instance set.
//!
//! A painter set draws its sprites sorted by key, so a sprite's slot in the
//! buffers is not its index. Writes address sprites by index all the same; a
//! write to the keys re-sorts the set at the next frame and a write without
//! them never does. Overrides change a frame's draw and leave the set alone.

use glam::Vec2;
use viewport_lib::error::ViewportError;
use viewport_lib::plugin_api::{Uploads, Writes};
use viewport_lib::{Camera2d, CameraFrame, Colour, FrameData, Plane2d, SceneFrame};
use viewport_lib_plugins::item_types::sprite::{
    self, SpriteInstanceSetRefItem, SpriteInstanceUploads, SpriteItem, SpriteOrder,
    SpriteOverrides, SpriteSetRefItem, SpriteSizeMode, channels,
};
use viewport_lib_testkit::Harness;

const W: u32 = 200;
const H: u32 = 150;

/// One world unit per pixel, origin in the middle of the view.
fn frame(scene: SceneFrame) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    )
}

fn set_frame(id: sprite::SpriteSetId) -> FrameData {
    let mut scene = SceneFrame::default();
    scene
        .items_mut::<SpriteSetRefItem>()
        .push(SpriteSetRefItem::new(id));
    frame(scene)
}

fn instance_frame(item: SpriteInstanceSetRefItem) -> FrameData {
    let mut scene = SceneFrame::default();
    scene.items_mut::<SpriteInstanceSetRefItem>().push(item);
    frame(scene)
}

const RED: Colour = Colour::linear(1.0, 0.0, 0.0, 1.0);
const BLUE: Colour = Colour::linear(0.0, 0.0, 1.0, 1.0);

/// Two opaque squares overlapping at the origin: red (sprite 0) at key 0 and
/// blue (sprite 1) at key 1, so blue draws on top.
fn pair() -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
    item.colours = vec![RED, BLUE];
    item.default_size = 40.0;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.order = SpriteOrder::Painter {
        layer: 0,
        depth_test: false,
    };
    item.sort_keys = vec![0.0, 1.0];
    item
}

/// Which of the two colours covers world point `(x, y)`.
fn colour_at(pixels: &[u8], x: f32, y: f32) -> &'static str {
    let px = (W as f32 * 0.5 + x) as usize;
    let py = (H as f32 * 0.5 - y) as usize;
    let i = (py * W as usize + px) * 4;
    let (r, b) = (pixels[i], pixels[i + 2]);
    match (r > 200 && b < 60, b > 200 && r < 60) {
        (true, false) => "red",
        (false, true) => "blue",
        _ => "neither",
    }
}

/// The sprite type's draw calls for the last frame.
fn draw_calls(harness: &Harness) -> u32 {
    harness
        .renderer
        .plugin_frame_counters()
        .find(|(name, _)| *name == sprite::TYPE_NAME)
        .map(|(_, c)| c.draw_calls)
        .expect("sprites registered")
}

/// The sprite type's instance bytes for the last frame.
fn upload_bytes(harness: &Harness) -> u64 {
    harness
        .renderer
        .plugin_frame_counters()
        .find(|(name, _)| *name == sprite::TYPE_NAME)
        .map(|(_, c)| c.upload_bytes)
        .expect("sprites registered")
}

#[test]
fn a_key_write_re_sorts_a_painter_set_and_a_positions_write_does_not() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (device, queue) = (harness.device.clone(), harness.queue.clone());
    let id = harness.renderer.upload(&device, &queue, &pair()).unwrap();
    let f = set_frame(id);
    let pixels = harness.render(&f, W, H);
    assert_eq!(colour_at(&pixels, 0.0, 0.0), "blue", "key 1 draws on top");

    // Red's key above blue's: red moves on top at the next frame.
    harness
        .renderer
        .write_range(channels::SortKeys, &queue, id, 0, &[2.0])
        .unwrap();
    let pixels = harness.render(&f, W, H);
    assert_eq!(colour_at(&pixels, 0.0, 0.0), "red", "a key write re-sorts");
    // The sequence would place each sprite by its key even unsorted, at a
    // draw per sprite; the re-sort is what keeps the set one draw.
    assert_eq!(
        draw_calls(&harness),
        1,
        "the re-sorted set draws as one run"
    );

    // Move blue (index 1, now drawn first) up and to the left, over where red
    // is. No keys written, so red stays on top wherever blue goes, and the
    // write reaches blue although it no longer sits at slot 1.
    harness
        .renderer
        .write_range(channels::Positions, &queue, id, 1, &[[-5.0, 10.0, 0.0]])
        .unwrap();
    let pixels = harness.render(&f, W, H);
    assert_eq!(
        colour_at(&pixels, -5.0, 0.0),
        "red",
        "no re-sort without keys"
    );
    assert_eq!(
        colour_at(&pixels, -5.0, 25.0),
        "blue",
        "the write moved the sprite it named"
    );
    assert_eq!(
        colour_at(&pixels, 25.0, 0.0),
        "neither",
        "blue left its spot"
    );
    assert_eq!(upload_bytes(&harness), 12, "one position, no re-sort");

    // A record write by index lands on the named sprite too.
    harness
        .renderer
        .write_range(
            channels::Sprites,
            &queue,
            id,
            0,
            &[channels::Sprite {
                colour: BLUE,
                size: [40.0; 2],
                ..Default::default()
            }],
        )
        .unwrap();
    let pixels = harness.render(&f, W, H);
    assert_eq!(colour_at(&pixels, -25.0, 0.0), "blue");
}

#[test]
fn a_positions_write_moves_only_the_positions_bytes() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (device, queue) = (harness.device.clone(), harness.queue.clone());

    // A painter set whose keys shuffle it, and the same sprites in depth order.
    let mut item = SpriteItem::default();
    item.positions = (0..100).map(|i| [i as f32 - 50.0, 0.0, 0.0]).collect();
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_size = 4.0;
    item.order = SpriteOrder::Painter {
        layer: 0,
        depth_test: false,
    };
    item.sort_keys = (0..100).map(|i| ((i * 37) % 100) as f32).collect();
    let painter = harness.renderer.upload(&device, &queue, &item).unwrap();
    item.order = SpriteOrder::Depth;
    let depth = harness.renderer.upload(&device, &queue, &item).unwrap();

    for id in [painter, depth] {
        let f = set_frame(id);
        harness.render(&f, W, H);
        harness.render(&f, W, H);
        assert_eq!(
            upload_bytes(&harness),
            0,
            "an unchanged set uploads nothing"
        );

        let moved: Vec<[f32; 3]> = (20..30).map(|i| [i as f32 - 50.0, 5.0, 0.0]).collect();
        harness
            .renderer
            .write_range(channels::Positions, &queue, id, 20, &moved)
            .unwrap();
        harness.render(&f, W, H);
        assert_eq!(
            upload_bytes(&harness),
            10 * 12,
            "ten positions of twelve bytes, no records or keys"
        );
    }

    // Keys with the positions re-sort: the sprites whose slots changed are
    // written whole, and the rest move only their positions.
    let f = set_frame(painter);
    let moved: Vec<[f32; 3]> = (0..100).map(|i| [i as f32 - 50.0, 8.0, 0.0]).collect();
    let keys: Vec<f32> = (0..100).map(|i| -(i as f32)).collect();
    harness
        .renderer
        .write_range(channels::Positions, &queue, painter, 0, &moved)
        .unwrap();
    harness
        .renderer
        .write_range(channels::SortKeys, &queue, painter, 0, &keys)
        .unwrap();
    harness.render(&f, W, H);
    let resort = upload_bytes(&harness);
    assert!(
        resort > 100 * 12 && resort <= 100 * (12 + 64),
        "a re-sort writes positions and moved records once: {resort}"
    );
    harness.render(&f, W, H);
    assert_eq!(upload_bytes(&harness), 0, "and nothing the frame after");
}

#[test]
fn sort_keys_are_a_painter_set_channel() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (device, queue) = (harness.device.clone(), harness.queue.clone());
    let mut item = pair();
    item.order = SpriteOrder::Depth;
    let id = harness.renderer.upload(&device, &queue, &item).unwrap();
    let err = harness
        .renderer
        .write_range(channels::SortKeys, &queue, id, 0, &[1.0])
        .unwrap_err();
    assert!(matches!(
        err,
        ViewportError::ChannelNotPresent {
            channel: "sort_keys",
            ..
        }
    ));
    assert!(harness.renderer.extent(channels::SortKeys, id).is_none());
}

#[test]
fn a_painter_set_keeps_one_live_count_by_index() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (device, queue) = (harness.device.clone(), harness.queue.clone());
    // Red sorts after blue, so red sits in slot 1.
    let mut item = pair();
    item.sort_keys = vec![2.0, 1.0];
    let id = harness.renderer.upload(&device, &queue, &item).unwrap();
    let f = set_frame(id);

    // Dropping the live count to one keeps sprite 0 (red), not whatever is
    // in slot 0.
    harness
        .renderer
        .set_len(channels::Positions, id, 1)
        .unwrap();
    assert_eq!(
        harness.renderer.extent(channels::Sprites, id).unwrap().len,
        1
    );
    let pixels = harness.render(&f, W, H);
    assert_eq!(colour_at(&pixels, -25.0, 0.0), "red");
    assert_eq!(colour_at(&pixels, 25.0, 0.0), "neither");

    // Raising it again brings blue back, still drawn by its key.
    harness
        .renderer
        .set_len(channels::Positions, id, 2)
        .unwrap();
    let pixels = harness.render(&f, W, H);
    assert_eq!(colour_at(&pixels, 25.0, 0.0), "blue");
    assert_eq!(colour_at(&pixels, 0.0, 0.0), "red");
}

#[test]
fn overrides_move_and_re_sort_a_frame_and_leave_the_set_alone() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (device, queue) = (harness.device.clone(), harness.queue.clone());
    let id = harness
        .renderer
        .upload_sprite_instance_set(&device, &queue, &pair())
        .unwrap();

    let plain = instance_frame(SpriteInstanceSetRefItem::new(id));
    let pixels = harness.render(&plain, W, H);
    assert_eq!(colour_at(&pixels, 0.0, 0.0), "blue");

    // Red takes a key above blue's for this frame: it draws on top.
    let mut over = SpriteOverrides::new(vec![0]);
    over.sort_keys = vec![5.0];
    let f = instance_frame(SpriteInstanceSetRefItem::new(id).with_overrides(over));
    let pixels = harness.render(&f, W, H);
    assert_eq!(colour_at(&pixels, 0.0, 0.0), "red", "the override re-sorts");

    // Blue moves up and turns red for this frame; sprite 0 is untouched.
    let mut over = SpriteOverrides::new(vec![1]);
    over.positions = vec![[10.0, 30.0, 0.0]];
    over.colours = vec![RED];
    let f = instance_frame(SpriteInstanceSetRefItem::new(id).with_overrides(over));
    let pixels = harness.render(&f, W, H);
    assert_eq!(colour_at(&pixels, 10.0, 45.0), "red");
    assert_eq!(colour_at(&pixels, 25.0, 0.0), "neither");
    assert_eq!(colour_at(&pixels, -25.0, 0.0), "red");
    assert!(upload_bytes(&harness) > 0);

    // The next frame without overrides draws the set as uploaded.
    let pixels = harness.render(&plain, W, H);
    assert_eq!(colour_at(&pixels, 0.0, 0.0), "blue");
    assert_eq!(colour_at(&pixels, 25.0, 0.0), "blue");
    assert_eq!(colour_at(&pixels, 10.0, 45.0), "neither");
    assert_eq!(upload_bytes(&harness), 0);
}
