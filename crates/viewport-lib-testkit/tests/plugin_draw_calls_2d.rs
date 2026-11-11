//! What the 2D item types report as draw calls: one per painter sprite batch,
//! one per visible tilemap chunk, and for the light layer each light's shadow
//! and quad. `FrameStats::plugin_draw_calls` is their sum.

use glam::{DVec3, Vec2};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{Camera2d, CameraFrame, Candela, FrameData, LightSource, Plane2d, SceneFrame};
use viewport_lib_plugins::item_types::lighting_2d::{Lighting2dItem, Occluder2d};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteOrder, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{Tile, Tilemap, TilemapItem, Tileset};
use viewport_lib_testkit::Harness;

const SIZE: u32 = 40;

/// A 16 x 16 map of unit cells in chunks of 8: 2 x 2 chunks, all in view.
const CHUNKS: u32 = 4;

fn draw_calls(harness: &Harness, name: &str) -> u32 {
    harness
        .renderer
        .plugin_frame_counters()
        .find(|(n, _)| *n == name)
        .map(|(_, c)| c.draw_calls)
        .unwrap()
}

/// `batches` painter sprite batches over the map, lit by `lights` lights
/// with one occluder between them.
fn frame(map: TilemapItem, batches: u32, lights: u32) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_centre(Vec2::splat(8.0))
        .with_units_per_pixel(0.5)
        .with_viewport_size(Vec2::splat(SIZE as f32))
        .camera();
    let mut scene = SceneFrame::default();
    *scene.items_mut::<TilemapItem>() = vec![map];
    *scene.items_mut::<SpriteItem>() = (0..batches)
        .map(|i| {
            let mut s = SpriteItem::default();
            s.positions = vec![[2.0 + 4.0 * i as f32, 8.0, 1.0]];
            s.default_size = 3.0;
            s.size_mode = SpriteSizeMode::WorldSpace;
            s.order = SpriteOrder::Painter {
                layer: i,
                depth_test: false,
            };
            s
        })
        .collect();
    let mut layer = Lighting2dItem::default();
    layer.occluders = vec![Occluder2d::Rect {
        min: [7.0, 2.0],
        max: [9.0, 4.0],
    }];
    *scene.items_mut::<Lighting2dItem>() = vec![layer];
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera, [SIZE as f32, SIZE as f32]),
        scene,
    );
    frame.effects.lighting.lights = (0..lights)
        .map(|i| {
            LightSource::point_candela(
                DVec3::new(2.0 + 12.0 * i as f64, 3.0, 5.0),
                Candela(1.0),
                30.0,
                0.0,
            )
        })
        .collect();
    frame.viewport.show_axes_indicator = false;
    frame
}

#[test]
fn a_2d_frame_reports_every_plugin_draw() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut map = Tilemap::with_chunk_size(Tileset::new(None, [8, 8], 1), [16, 16], 8);
    for row in 0..16 {
        for col in 0..16 {
            map.set(col, row, Tile::new(0));
        }
    }
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    let map = TilemapItem::new(id);

    let (batches, lights) = (3, 2);
    let f = frame(map.clone(), batches, lights);
    harness.render(&f, SIZE, SIZE);
    harness.render(&f, SIZE, SIZE);
    let sprites = draw_calls(
        &harness,
        viewport_lib_plugins::item_types::sprite::TYPE_NAME,
    );
    let tiles = draw_calls(
        &harness,
        viewport_lib_plugins::item_types::tilemap::TYPE_NAME,
    );
    let light = draw_calls(
        &harness,
        viewport_lib_plugins::item_types::lighting_2d::TYPE_NAME,
    );
    assert_eq!(sprites, batches, "one draw per painter batch");
    assert_eq!(tiles, CHUNKS, "one draw per visible chunk");
    assert_eq!(light, 2 * lights, "a shadow and a quad per light");
    assert_eq!(
        harness.stats().plugin_draw_calls,
        batches + CHUNKS + 2 * lights
    );

    // A frame without the light layer's items reports nothing for it, not
    // what it drew last.
    let mut unlit = frame(map, batches, lights);
    unlit
        .scene
        .plugin_items
        .remove(viewport_lib_plugins::item_types::lighting_2d::TYPE_NAME);
    harness.render(&unlit, SIZE, SIZE);
    assert_eq!(
        draw_calls(
            &harness,
            viewport_lib_plugins::item_types::lighting_2d::TYPE_NAME
        ),
        0
    );
    assert_eq!(harness.stats().plugin_draw_calls, batches + CHUNKS);
}
