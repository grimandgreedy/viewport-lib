//! Tilemaps: chunk culling, chunk re-upload on an edit, picking, and drawing
//! on the compatibility tier.

use glam::Vec2;
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, CameraFrame, FrameData, PickBackend, PickId, PickMask, Plane2d, SceneFrame,
    SubObjectRef, Tier,
};
use viewport_lib_plugins::item_types::tilemap::{
    TYPE_NAME, Tile, Tilemap, TilemapEdits, TilemapId, TilemapItem, TilemapPlugin, Tileset,
};
use viewport_lib_testkit::{DeviceProfile, Harness, catalogue, frame_for};

/// A 128 x 128 map of solid tiles in 16-tile chunks: 8 x 8 chunks.
fn full_map() -> Tilemap {
    let mut map = Tilemap::with_chunk_size(Tileset::new(None, [8, 8], 1), [128, 128], 16);
    for row in 0..128 {
        for col in 0..128 {
            map.set(col, row, Tile::new(0));
        }
    }
    map
}

fn upload(harness: &mut Harness, map: &Tilemap) -> TilemapId {
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    harness
        .renderer
        .upload(&device, &queue, map)
        .expect("upload")
}

/// A frame looking at plane `centre` with `size` units in view, one unit
/// per pixel.
fn frame(item: TilemapItem, centre: Vec2, size: u32) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_centre(centre)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::splat(size as f32))
        .camera();
    let mut scene = SceneFrame::default();
    *scene.items_mut::<TilemapItem>() = vec![item];
    FrameData::new(
        CameraFrame::from_camera(&camera, [size as f32, size as f32]),
        scene,
    )
}

fn plugin(harness: &Harness) -> &TilemapPlugin {
    harness
        .renderer
        .item_type_plugin::<TilemapPlugin>(TYPE_NAME)
        .expect("registered")
}

fn draw_calls(harness: &Harness) -> u32 {
    harness
        .renderer
        .plugin_frame_counters()
        .find(|(name, _)| *name == TYPE_NAME)
        .map(|(_, c)| c.draw_calls)
        .unwrap()
}

#[test]
fn a_view_of_a_quarter_of_the_map_draws_a_quarter_of_its_chunks() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let id = upload(&mut harness, &full_map());
    let item = TilemapItem::new(id);

    // The whole map.
    harness.render(&frame(item.clone(), Vec2::splat(64.0), 130), 130, 130);
    assert_eq!(plugin(&harness).drawn_chunks(), 64);

    // Units 1..63 on each axis: the bottom-left 4 x 4 chunks, with no edge
    // on the view's boundary.
    harness.render(&frame(item.clone(), Vec2::splat(32.0), 62), 62, 62);
    assert_eq!(plugin(&harness).drawn_chunks(), 16);
    assert_eq!(draw_calls(&harness), 16, "one draw per drawn chunk");

    // Parallax moves the bounds with the layer: at half parallax the same
    // camera sees the layer shifted by (16, 16), so plane 1..63 shows layer
    // cells -15..47, chunks 0..2 on each axis.
    let mut half = item;
    half.parallax = [0.5, 0.5];
    harness.render(&frame(half, Vec2::splat(32.0), 62), 62, 62);
    assert_eq!(plugin(&harness).drawn_chunks(), 9);
}

#[test]
fn set_tile_re_uploads_only_its_chunk() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let id = upload(&mut harness, &full_map());
    assert_eq!(plugin(&harness).chunk_uploads(id), Some(64));

    let upload_bytes = |h: &Harness| {
        h.renderer
            .plugin_frame_counters()
            .find(|(name, _)| *name == TYPE_NAME)
            .map(|(_, c)| c.upload_bytes)
            .unwrap()
    };
    let f = frame(TilemapItem::new(id), Vec2::splat(64.0), 130);
    harness.render(&f, 130, 130);
    harness.render(&f, 130, 130);
    let steady = upload_bytes(&harness);

    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let on_map = harness
        .renderer
        .set_tile(
            &device,
            &queue,
            id,
            40,
            70,
            Tile::new(0).with_flip(Default::default()),
        )
        .unwrap();
    assert!(on_map);
    assert_eq!(plugin(&harness).chunk_uploads(id), Some(65));
    harness.render(&f, 130, 130);
    // One chunk of 16 x 16 occupied cells at 16 bytes each.
    assert_eq!(upload_bytes(&harness) - steady, 16 * 16 * 16);

    // Off the map: nothing written.
    let off = harness
        .renderer
        .set_tile(&device, &queue, id, 200, 0, Tile::new(0))
        .unwrap();
    assert!(!off);
    assert_eq!(plugin(&harness).chunk_uploads(id), Some(65));

    // Emptying a cell takes it out of the CPU copy and the chunk.
    harness
        .renderer
        .set_tile(&device, &queue, id, 0, 0, Tile::EMPTY)
        .unwrap();
    assert!(
        harness
            .renderer
            .tilemap(id)
            .unwrap()
            .get(0, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn picks_name_the_tile_under_the_cursor() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness.renderer.set_cpu_pick_cache(true);
    let id = upload(&mut harness, &full_map());
    let mut item = TilemapItem::new(id);
    item.settings.pick_id = PickId(77);
    item.cell_size = [4.0, 4.0];
    // Plane 0..62 in view. Pixel (42, 20) is plane (42, 42): cell (10, 10),
    // in chunk 0, cell 10 * 16 + 10 of it.
    let f = frame(item, Vec2::splat(31.0), 62);
    harness.render(&f, 62, 62);
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let expected = Some(SubObjectRef::Tile {
        chunk: 0,
        tile: 10 * 16 + 10,
    });
    for backend in [PickBackend::Gpu, PickBackend::Cpu] {
        let hit = harness
            .renderer
            .pick_object(
                backend,
                Vec2::new(42.0, 20.0),
                &f,
                &device,
                &queue,
                PickMask::OBJECT | PickMask::INSTANCE,
            )
            .unwrap_or_else(|| panic!("{backend:?} misses the map"));
        assert_eq!(hit.id, 77, "{backend:?}");
        assert_eq!(hit.sub_object, expected, "{backend:?}");
    }
}

#[test]
fn the_catalogue_scene_draws_on_the_compatibility_tier() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("tilemap-compatibility"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), Tier::Compatibility);
    assert!(harness.renderer.has_item_type_plugin(TYPE_NAME));
    let scenes = catalogue();
    let scene = scenes.iter().find(|s| s.name == "tilemap").unwrap();
    let built = harness.build_scene(scene);
    let camera = &scene.cameras[0].camera;
    let with = frame_for(&built, camera, [200.0, 150.0]);
    let mut without = frame_for(&built, camera, [200.0, 150.0]);
    without.scene.plugin_items.remove(TYPE_NAME);
    let _ = harness.render(&with, 200, 150);
    let drawn = harness.render(&with, 200, 150);
    let bare = harness.render(&without, 200, 150);
    let changed = drawn
        .chunks_exact(4)
        .zip(bare.chunks_exact(4))
        .filter(|(a, b)| a[..3] != b[..3])
        .count();
    assert!(
        changed > 200 * 150 * 9 / 10,
        "the layers cover the view: {changed} pixels changed"
    );
    assert_eq!(harness.stats().items_skipped_by_tier, 0);
}

/// An item on the frame clock animates with `FrameData::time` exactly as one
/// handed the same time by hand.
#[test]
fn an_item_on_the_frame_clock_animates_with_the_frame_time() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scenes = catalogue();
    let scene = scenes.iter().find(|s| s.name == "tilemap").unwrap();
    let built = harness.build_scene(scene);
    let camera = &scene.cameras[0].camera;
    // The water frames last a quarter second each.
    let at = |clock: bool, item_time: f32, frame_time: f64| {
        let mut f = frame_for(&built, camera, [200.0, 150.0]);
        for item in f.scene.items_mut::<TilemapItem>().iter_mut() {
            item.use_frame_clock = clock;
            item.time = item_time;
        }
        f.time = Some(viewport_lib::FrameTime::fixed(frame_time));
        f
    };
    let by_hand = harness.render(&at(false, 0.3, 0.0), 200, 150);
    let held = harness.render(&at(false, 0.0, 0.3), 200, 150);
    let on_clock = harness.render(&at(true, 0.0, 0.3), 200, 150);
    assert!(by_hand != held, "the water shows another frame at 0.3 s");
    assert!(on_clock == by_hand, "the frame clock drives the animation");
}
