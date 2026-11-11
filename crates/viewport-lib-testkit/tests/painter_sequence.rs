//! The painter sequence across plugins: blended sprites and blended tile layers
//! draw in one order sorted by layer and then by sort key, whichever plugin
//! drew them, with world overlay layers at their `order`. Sprite batches on one
//! layer interleave by their sprites' keys.

use glam::Vec2;
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, OverlayFrame, OverlayShape, OverlayShapeItem,
    Plane2d, SceneFrame, WorldOverlayLayer, WorldPlane,
};
use viewport_lib_plugins::item_types::sprite::{
    self, SpriteItem, SpriteOrder, SpritePlugin, SpriteSizeMode,
};
use viewport_lib_plugins::item_types::tilemap::{
    self, Tile, Tilemap, TilemapDraw, TilemapItem, Tileset,
};
use viewport_lib_testkit::Harness;

const W: u32 = 120;
const H: u32 = 90;

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

/// One blended square per entry of `keys`, all at the origin, in `colour`.
fn squares(layer: u32, keys: &[f32], colour: Colour) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = keys.iter().map(|_| [0.0, 0.0, 0.0]).collect();
    item.default_colour = colour;
    item.default_size = 40.0;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.order = SpriteOrder::Painter {
        layer,
        depth_test: false,
    };
    item.sort_keys = keys.to_vec();
    item
}

fn sprite_plugin(harness: &Harness) -> &SpritePlugin {
    harness
        .renderer
        .item_type_plugin::<SpritePlugin>(sprite::TYPE_NAME)
        .expect("registered")
}

/// The sprite batch and drawn range behind each handle of one call.
fn pieces(harness: &Harness, batches: &[u32]) -> Vec<(usize, std::ops::Range<u32>)> {
    batches
        .iter()
        .map(|&b| {
            sprite_plugin(harness)
                .debug_painter_piece(b)
                .expect("a piece")
        })
        .collect()
}

#[test]
fn sprites_and_tile_layers_draw_in_one_order_by_layer() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut map = Tilemap::with_chunk_size(Tileset::new(None, [8, 8], 1), [4, 4], 4);
    map.set(0, 0, Tile::new(0));
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    let mut tiles = TilemapItem::new(id);
    tiles.draw = TilemapDraw::Blended {
        layer: 1,
        depth_test: false,
    };

    let mut scene = SceneFrame::default();
    // Submitted high layer first: the sequence sorts them.
    *scene.items_mut::<SpriteItem>() = vec![
        squares(2, &[0.0], Colour::linear(0.2, 0.2, 0.9, 0.8)),
        squares(0, &[0.0], Colour::linear(0.9, 0.2, 0.2, 0.8)),
    ];
    *scene.items_mut::<TilemapItem>() = vec![tiles];
    let mut layer_frame = OverlayFrame::default();
    layer_frame.shapes.push(OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [0.0, 0.0],
        [4.0, 4.0],
    ));
    scene.world_overlays.push(
        WorldOverlayLayer::new(WorldPlane::Xy)
            .with_order(1)
            .with_frame(layer_frame),
    );
    let frame = frame(scene);
    harness.render(&frame, W, H);

    let seq = harness.renderer.debug_painter_sequence(&frame, true);
    let names: Vec<(&str, std::ops::RangeInclusive<u32>)> =
        seq.iter().map(|(n, l, _)| (*n, l.clone())).collect();
    assert_eq!(
        names,
        vec![
            (sprite::TYPE_NAME, 0..=0),
            (tilemap::TYPE_NAME, 1..=1),
            ("world_overlay", 1..=1),
            (sprite::TYPE_NAME, 2..=2),
        ]
    );
    // The layer-0 call draws the second batch submitted, the layer-2 call the
    // first.
    assert_eq!(pieces(&harness, &seq[0].2), vec![(1, 0..1)]);
    assert_eq!(pieces(&harness, &seq[3].2), vec![(0, 0..1)]);
    assert_eq!(seq[1].2, vec![0]);

    // The LDR path follows the same sequence.
    let ldr: Vec<&str> = harness
        .renderer
        .debug_painter_sequence(&frame, false)
        .iter()
        .map(|(n, _, _)| *n)
        .collect();
    assert_eq!(
        ldr,
        vec![
            sprite::TYPE_NAME,
            tilemap::TYPE_NAME,
            "world_overlay",
            sprite::TYPE_NAME
        ]
    );
}

#[test]
fn sprite_batches_on_one_layer_interleave_by_key() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![
        squares(
            0,
            &[3.0, -3.0, 1.0, -1.0],
            Colour::linear(0.9, 0.2, 0.2, 1.0),
        ),
        squares(0, &[0.0], Colour::linear(0.2, 0.2, 0.9, 1.0)),
    ];
    let frame = frame(scene);
    harness.render(&frame, W, H);

    let seq = harness.renderer.debug_painter_sequence(&frame, true);
    assert_eq!(seq.len(), 1, "one run of the sprite plugin: {seq:?}");
    // The crowd's two lower keys, then the single sprite, then the rest: one
    // piece per key, so each sprite is placed by its own key.
    assert_eq!(
        pieces(&harness, &seq[0].2),
        vec![(0, 0..1), (0, 1..2), (1, 0..1), (0, 2..3), (0, 3..4)]
    );
}

#[test]
fn a_later_batch_with_a_lower_key_draws_underneath() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Two opaque squares over one another: the first batch's key is higher,
    // so it draws on top although it was submitted first.
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![
        squares(0, &[1.0], Colour::linear(0.9, 0.1, 0.1, 1.0)),
        squares(0, &[0.0], Colour::linear(0.1, 0.1, 0.9, 1.0)),
    ];
    let pixels = harness.render(&frame(scene), W, H);
    let i = ((H / 2 * W + W / 2) * 4) as usize;
    let centre = &pixels[i..i + 4];
    assert!(
        centre[0] > centre[2] + 100,
        "the higher key draws on top: {centre:?}"
    );
}
