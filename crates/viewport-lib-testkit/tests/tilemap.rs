//! Tilemaps: chunk culling, chunk re-upload on an edit, picking, and drawing
//! on the compatibility tier.

use glam::Vec2;
use viewport_lib::plugin_api::{ItemTypePlugin, Uploads};
use viewport_lib::{
    Camera2d, CameraFrame, FrameData, PickBackend, PickId, PickMask, Plane2d, SceneFrame,
    SubObjectRef, Tier,
};
use viewport_lib_plugins::item_types::tilemap::{
    MAX_TILE_CELL, StaggerAxis, StaggerIndex, TYPE_NAME, Tile, TileOrientation, Tilemap,
    TilemapDraw, TilemapEdits, TilemapId, TilemapItem, TilemapPainter, TilemapPlugin, Tileset,
};
use viewport_lib_testkit::scenes::tilemap_orientations::ORIENTATION_PICK_IDS;
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

    // Outside the cell range: nothing written.
    let off = harness
        .renderer
        .set_tile(&device, &queue, id, MAX_TILE_CELL + 1, 0, Tile::new(0))
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
    // Plane 0..62 in view. Pixel (42, 20) is plane (42, 42): cell (10, 10).
    let f = frame(item, Vec2::splat(31.0), 62);
    harness.render(&f, 62, 62);
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let expected = Some(SubObjectRef::Tile { col: 10, row: 10 });
    for backend in [PickBackend::Gpu, PickBackend::Cpu] {
        let hit = harness
            .renderer
            .pick_object(
                backend,
                Vec2::new(42.0, 20.0),
                &f,
                &device,
                &queue,
                PickMask::OBJECT | PickMask::TILE,
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

/// A sparse map in `chunk`-tile chunks with tile 0 at each of `cells`.
fn sparse_map(chunk: u32, cells: impl IntoIterator<Item = (i32, i32)>) -> Tilemap {
    let mut map = Tilemap::sparse(Tileset::new(None, [8, 8], 1), chunk);
    for (col, row) in cells {
        map.set(col, row, Tile::new(0));
    }
    map
}

fn painter_batches(harness: &Harness) -> u32 {
    harness
        .renderer
        .plugin_frame_counters()
        .find(|(name, _)| *name == TYPE_NAME)
        .map(|(_, c)| c.painter_batches)
        .unwrap()
}

/// A layer sorted by row reports one batch, and draws once, per occupied row
/// of each drawn chunk; a whole-layer one reports one batch.
#[test]
fn a_layer_sorted_by_row_reports_a_batch_per_drawn_row() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // 8 x 6 cells in 4-cell chunks with row 2 left empty: the bottom chunks
    // hold rows 0, 1 and 3, the top ones rows 4 and 5.
    let cells = (0..6)
        .filter(|&r| r != 2)
        .flat_map(|r| (0..8).map(move |c| (c, r)));
    let id = upload(&mut harness, &sparse_map(4, cells));
    let mut item = TilemapItem::new(id);
    item.cell_size = [4.0, 4.0];
    item.draw = TilemapDraw::Blended {
        layer: 0,
        depth_test: false,
        painter: TilemapPainter::PerRow { offset: 0.0 },
    };

    harness.render(&frame(item.clone(), Vec2::new(16.0, 12.0), 40), 40, 40);
    assert_eq!(painter_batches(&harness), 2 * 3 + 2 * 2);
    assert_eq!(draw_calls(&harness), 10, "one draw per batch");

    // Only the left chunks in view (plane 1..15 across): half the rows.
    harness.render(&frame(item.clone(), Vec2::new(8.0, 12.0), 14), 14, 14);
    assert_eq!(painter_batches(&harness), 3 + 2);

    item.draw = TilemapDraw::Blended {
        layer: 0,
        depth_test: false,
        painter: TilemapPainter::Layer { key: 0.0 },
    };
    harness.render(&frame(item, Vec2::new(16.0, 12.0), 40), 40, 40);
    assert_eq!(painter_batches(&harness), 1);
    assert_eq!(draw_calls(&harness), 4, "one draw per chunk");
}

/// A map grown at negative coordinates draws, culls and picks those chunks
/// like any other.
#[test]
fn a_chunk_at_negative_coordinates_draws_and_culls() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness.renderer.set_cpu_pick_cache(true);
    let id = upload(&mut harness, &sparse_map(4, [(-3, -3), (5, 5)]));
    let mut item = TilemapItem::new(id);
    item.cell_size = [4.0, 4.0];
    item.settings.pick_id = PickId(9);

    // Plane -20..28: both chunks.
    let f = frame(item.clone(), Vec2::new(4.0, 4.0), 48);
    harness.render(&f, 48, 48);
    assert_eq!(plugin(&harness).drawn_chunks(), 2);
    // Plane -15..-1: the negative chunk alone.
    harness.render(&frame(item.clone(), Vec2::new(-8.0, -8.0), 14), 14, 14);
    assert_eq!(plugin(&harness).drawn_chunks(), 1);
    // Plane 1..15: neither.
    harness.render(&frame(item, Vec2::new(8.0, 8.0), 14), 14, 14);
    assert_eq!(plugin(&harness).drawn_chunks(), 0);

    // Cell (-3, -3) covers plane -12..-8: its centre (-10, -10) is pixel
    // (10, 38) in the first view.
    harness.render(&f, 48, 48);
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    for backend in [PickBackend::Gpu, PickBackend::Cpu] {
        let hit = harness
            .renderer
            .pick_object(
                backend,
                Vec2::new(10.0, 38.0),
                &f,
                &device,
                &queue,
                PickMask::OBJECT | PickMask::TILE,
            )
            .unwrap_or_else(|| panic!("{backend:?} misses the cell"));
        assert_eq!(
            hit.sub_object,
            Some(SubObjectRef::Tile { col: -3, row: -3 }),
            "{backend:?}"
        );
    }
}

/// A chunk's buffer holds an instance per occupied cell, and removing the
/// chunk frees it.
#[test]
fn chunk_buffers_fit_their_tiles_and_go_with_their_chunk() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // One tile in a 32 x 32 chunk: one 16-byte instance.
    let one = upload(&mut harness, &sparse_map(32, [(3, 4)]));
    assert_eq!(plugin(&harness).chunk_buffer_bytes(one), Some(16));

    let id = upload(&mut harness, &sparse_map(4, [(0, 0), (1, 0), (-1, 9)]));
    assert_eq!(plugin(&harness).chunk_buffer_bytes(id), Some(3 * 16));
    let before = plugin(&harness).resident_bytes();
    let removed = harness.renderer.remove_chunk(id, [-1, 2]).unwrap();
    assert!(removed);
    assert_eq!(plugin(&harness).chunk_buffer_bytes(id), Some(2 * 16));
    assert_eq!(before - plugin(&harness).resident_bytes(), 16);
    assert!(
        harness
            .renderer
            .tilemap(id)
            .unwrap()
            .chunk([-1, 2])
            .is_none()
    );
    assert!(!harness.renderer.remove_chunk(id, [-1, 2]).unwrap());

    let mut item = TilemapItem::new(id);
    item.cell_size = [4.0, 4.0];
    harness.render(&frame(item, Vec2::new(0.0, 16.0), 64), 64, 64);
    assert_eq!(plugin(&harness).drawn_chunks(), 1);
}

/// A map uploaded without its CPU copy takes whole-chunk edits only.
#[test]
fn a_map_without_its_cpu_copy_is_edited_by_chunk() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut map = sparse_map(4, [(0, 0), (5, 1)]);
    map.keep_cpu_copy = false;
    let id = upload(&mut harness, &map);
    assert_eq!(harness.renderer.tilemap(id).unwrap().chunk_count(), 0);
    assert_eq!(plugin(&harness).chunk_buffer_bytes(id), Some(2 * 16));

    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let written = harness
        .renderer
        .set_tile(&device, &queue, id, 1, 0, Tile::new(0))
        .unwrap();
    assert!(!written, "no copy to edit a cell in");
    assert_eq!(plugin(&harness).chunk_buffer_bytes(id), Some(2 * 16));

    let mut chunk = viewport_lib_plugins::item_types::tilemap::TileChunk::new(4);
    chunk.tiles[0] = Tile::new(0);
    chunk.tiles[1] = Tile::new(0);
    assert!(
        harness
            .renderer
            .set_chunk(&device, &queue, id, [0, 0], chunk)
            .unwrap()
    );
    assert_eq!(plugin(&harness).chunk_buffer_bytes(id), Some(3 * 16));
    assert_eq!(harness.renderer.tilemap(id).unwrap().chunk_count(), 0);
}

/// A 62 x 62 view of 4-unit cells (one unit per pixel) with pick id 77.
fn picking_frame(harness: &mut Harness) -> FrameData {
    harness.renderer.set_cpu_pick_cache(true);
    let id = upload(harness, &full_map());
    let mut item = TilemapItem::new(id);
    item.settings.pick_id = PickId(77);
    item.cell_size = [4.0, 4.0];
    let f = frame(item, Vec2::splat(31.0), 62);
    harness.render(&f, 62, 62);
    f
}

/// A rectangle over the centres of 4 x 3 cells returns those twelve, on
/// either backend.
#[test]
fn a_rectangle_over_four_by_three_cells_returns_twelve() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let f = picking_frame(&mut harness);
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    // Columns 2..=5 are plane x 8..24 and rows 3..=5 plane y 12..24, pixel
    // y 38..50. The rectangle takes in their centres and no other cell's
    // pixels.
    let (min, max) = (Vec2::new(9.0, 39.0), Vec2::new(23.0, 49.0));
    let mut expected: Vec<_> = (2..=5)
        .flat_map(|col| (3..=5).map(move |row| SubObjectRef::Tile { col, row }))
        .collect();
    expected.sort_by_key(|s| s.index());
    for backend in [PickBackend::Cpu, PickBackend::Gpu] {
        let result = harness.renderer.pick_rect_objects(
            backend,
            min,
            max,
            &f,
            &device,
            &queue,
            PickMask::OBJECT | PickMask::TILE,
        );
        assert_eq!(result.objects, vec![77], "{backend:?}");
        let mut got: Vec<_> = result
            .elements
            .iter()
            .filter(|(id, _)| *id == 77)
            .map(|(_, s)| *s)
            .collect();
        got.sort_by_key(|s| s.index());
        got.dedup();
        assert_eq!(got, expected, "{backend:?}");
    }
}

/// Tiles answer `PickMask::TILE`, not `INSTANCE`.
#[test]
fn a_pick_for_instances_only_returns_no_tiles() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let f = picking_frame(&mut harness);
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    for backend in [PickBackend::Gpu, PickBackend::Cpu] {
        let hit = harness.renderer.pick_object(
            backend,
            Vec2::new(42.0, 20.0),
            &f,
            &device,
            &queue,
            PickMask::INSTANCE,
        );
        assert!(
            hit.is_none_or(|h| !matches!(h.sub_object, Some(SubObjectRef::Tile { .. }))),
            "{backend:?}: {hit:?}"
        );
    }
    let rect = harness.renderer.pick_rect_objects(
        PickBackend::Cpu,
        Vec2::ZERO,
        Vec2::splat(62.0),
        &f,
        &device,
        &queue,
        PickMask::INSTANCE,
    );
    assert!(rect.elements.is_empty() && rect.objects.is_empty());
}

/// Each cell picks as itself at its centre, and a cursor on an edge two
/// cells share picks the same one of them on either backend: the one whose
/// pixel the cursor is in, right of a vertical edge and below a horizontal
/// one on screen.
#[test]
fn cells_pick_at_their_centres_and_their_shared_edges() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let f = picking_frame(&mut harness);
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let mut pick = |at: Vec2, backend| {
        harness
            .renderer
            .pick_object(backend, at, &f, &device, &queue, PickMask::TILE)
            .and_then(|h| h.sub_object)
    };
    // Plane (x, y) is pixel (x, 62 - y).
    for col in 2..6 {
        for row in 2..6 {
            let (x0, y0) = (col as f32 * 4.0, row as f32 * 4.0);
            let cell = Some(SubObjectRef::Tile { col, row });
            for backend in [PickBackend::Gpu, PickBackend::Cpu] {
                let centre = Vec2::new(x0 + 2.0, 62.0 - (y0 + 2.0));
                assert_eq!(
                    pick(centre, backend),
                    cell,
                    "{backend:?} centre {col} {row}"
                );
                // The cell's left edge, and the edge on screen above it,
                // which is its top: both inside it.
                let left = Vec2::new(x0, 62.0 - (y0 + 2.0));
                assert_eq!(pick(left, backend), cell, "{backend:?} left {col} {row}");
                let top = Vec2::new(x0 + 2.0, 62.0 - (y0 + 4.0));
                assert_eq!(pick(top, backend), cell, "{backend:?} top {col} {row}");
            }
        }
    }
}

/// A selected layer is outlined around its occupied cells, wireframe mode
/// traces each cell, and a selected cell takes the sub-object highlight.
#[test]
fn selection_outlines_traces_and_highlights_cells() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Columns 2..=5 and rows 3..=5 of 4-unit cells: plane x 8..24, y 12..24,
    // pixels x 8..24 and y 38..50 of the 62-pixel view.
    let cells = (2..=5).flat_map(|c| (3..=5).map(move |r| (c, r)));
    let id = upload(&mut harness, &sparse_map(8, cells));
    let mut item = TilemapItem::new(id);
    item.cell_size = [4.0, 4.0];
    item.settings.pick_id = PickId(77);
    let base_frame = frame(item.clone(), Vec2::splat(31.0), 62);
    let _ = harness.render(&base_frame, 62, 62);
    let base = harness.render(&base_frame, 62, 62);
    let at = |img: &[u8], x: usize, y: usize| img[(y * 62 + x) * 4..][..3].to_vec();
    let changed = |img: &[u8], inside: bool| {
        (0..62)
            .flat_map(|y| (0..62).map(move |x| (x, y)))
            .filter(|&(x, y)| ((8..24).contains(&x) && (38..50).contains(&y)) == inside)
            .filter(|&(x, y)| at(img, x, y) != at(&base, x, y))
            .count()
    };

    // The outline rings the block and leaves its inside alone.
    item.settings.selected = true;
    let mut outlined = frame(item.clone(), Vec2::splat(31.0), 62);
    outlined.interaction.outline_selected = true;
    let _ = harness.render(&outlined, 62, 62);
    let img = harness.render(&outlined, 62, 62);
    assert!(changed(&img, false) > 40, "the ring around the block");
    assert_eq!(changed(&img, true), 0, "nothing inside the block");
    item.settings.selected = false;

    // Wireframe mode draws each cell's outline instead of its tile.
    let mut wire = frame(item.clone(), Vec2::splat(31.0), 62);
    wire.viewport.wireframe_mode = true;
    let _ = harness.render(&wire, 62, 62);
    let img = harness.render(&wire, 62, 62);
    // The line on plane x 12 lands on pixel column 11 or 12.
    let lines = (38..50)
        .filter(|&y| [11, 12].iter().any(|&x| at(&img, x, y) != at(&img, 14, y)))
        .count();
    assert!(lines >= 8, "a line between columns 2 and 3: {lines} pixels");
    assert!(
        changed(&img, true) > 40,
        "tiles gone from the cells' insides"
    );

    // Cell (3, 4) is pixels x 12..16, y 42..46.
    let mut selection = viewport_lib::SubSelection::new();
    selection.add(77, SubObjectRef::Tile { col: 3, row: 4 });
    let mut highlighted = frame(item, Vec2::splat(31.0), 62);
    highlighted.interaction.sub_selection = Some(viewport_lib::SubSelectionRef::new(
        &selection,
        Default::default(),
        Default::default(),
        Default::default(),
    ));
    let _ = harness.render(&highlighted, 62, 62);
    let img = harness.render(&highlighted, 62, 62);
    assert_ne!(at(&img, 14, 44), at(&base, 14, 44), "the selected cell");
    assert_eq!(at(&img, 21, 40), at(&base, 21, 40), "another cell");
}

/// Tall tiles sorted by row draw on both tiers: the tree's upper cell, and
/// the lower of two trees over the upper one's trunk. On the full tier the
/// trunk also covers the figure behind it and the figure in front covers the
/// trunk; sprites do not draw on the compatibility tier.
#[test]
fn tall_tiles_draw_on_the_compatibility_tier() {
    let scenes = catalogue();
    let scene = scenes
        .iter()
        .find(|s| s.name == "tilemap_tall_tiles")
        .unwrap();
    let render = |mut harness: Harness| {
        let built = harness.build_scene(scene);
        let f = frame_for(&built, &scene.cameras[0].camera, [200.0, 150.0]);
        let _ = harness.render(&f, 200, 150);
        harness.render(&f, 200, 150)
    };
    #[derive(Debug, PartialEq)]
    enum Seen {
        Canopy,
        Trunk,
        Red,
        Blue,
        Other,
    }
    // The 200 x 150 render shows plane (-66, -25.5) to (34, 49.5), two
    // pixels per unit.
    let seen = |img: &[u8], x: f32, y: f32| {
        let (px, py) = ((x + 66.0) * 2.0, (49.5 - y) * 2.0);
        let p = &img[((py as usize) * 200 + px as usize) * 4..][..3];
        let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
        if r > 150 && g < 100 {
            Seen::Red
        } else if b > 150 && r < 100 {
            Seen::Blue
        } else if g > r && g > b && g < 130 {
            Seen::Canopy
        } else if r > g && g > b && r < 160 {
            Seen::Trunk
        } else {
            Seen::Other
        }
    };
    let check = |img: &[u8], tier: &str| {
        assert_eq!(seen(img, -40.0, 24.0), Seen::Canopy, "{tier}: upper cell");
        assert_eq!(
            seen(img, 8.0, 22.0),
            Seen::Canopy,
            "{tier}: lower row in front"
        );
        assert_eq!(seen(img, 8.0, 36.0), Seen::Canopy, "{tier}: upper tree");
    };
    let Some(full) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let img = render(full);
    check(&img, "full");
    assert_eq!(seen(&img, -42.0, 10.0), Seen::Trunk, "trunk over red");
    assert_eq!(seen(&img, -38.0, 4.0), Seen::Blue, "blue over trunk");
    assert_eq!(seen(&img, -46.0, 10.0), Seen::Red, "red beside the trunk");
    let Some(compat) = Harness::with_profile(&DeviceProfile::downlevel_webgl2(
        "tilemap-tall-compatibility",
    )) else {
        return;
    };
    assert_eq!(compat.renderer.tier(), Tier::Compatibility);
    check(&render(compat), "compatibility");
}

/// The `tilemap_orientations` scene built and drawn at its own 400 x 300
/// size, one unit per pixel with plane (0, 0) at the centre.
fn orientations(harness: &mut Harness) -> (FrameData, Vec<TilemapItem>) {
    harness.renderer.set_cpu_pick_cache(true);
    let scenes = catalogue();
    let scene = scenes
        .iter()
        .find(|s| s.name == "tilemap_orientations")
        .unwrap();
    let built = harness.build_scene(scene);
    let f = frame_for(&built, &scene.cameras[0].camera, [400.0, 300.0]);
    let _ = harness.render(&f, 400, 300);
    (f, built.tilemaps.clone())
}

/// Every occupied cell of each orientation's map, with its centre in pixels.
fn orientation_cells(harness: &Harness, items: &[TilemapItem]) -> Vec<(u64, [i32; 2], Vec2)> {
    let mut out = Vec::new();
    for item in items {
        let id = item.settings.pick_id.0;
        if !ORIENTATION_PICK_IDS.contains(&id) {
            continue;
        }
        let map = harness.renderer.tilemap(item.source).unwrap();
        for (coord, chunk) in map.chunks() {
            for (i, tile) in chunk.tiles.iter().enumerate() {
                if tile.is_empty() {
                    continue;
                }
                let [col, row] = map.cell_of(coord, i as u32).unwrap();
                let w = item.cell_to_world(&map.tileset, col, row, glam::DVec2::ZERO);
                // Round trip through the CPU helper.
                assert_eq!(
                    item.world_to_cell(&map.tileset, w, glam::DVec2::ZERO),
                    Some([col, row])
                );
                let pixel = Vec2::new(w.x as f32 + 200.0, 150.0 - w.y as f32);
                out.push((id, [col, row], pixel));
            }
        }
    }
    out
}

/// In each of Tiled's orientations, a pick at a cell's centre names that
/// cell on either backend, turned hexagons included.
#[test]
fn cells_of_every_orientation_pick_at_their_centres() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (f, items) = orientations(&mut harness);
    let cells = orientation_cells(&harness, &items);
    assert_eq!(cells.len(), 35 + 50 + 30 + 28);
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    for (id, [col, row], pixel) in cells {
        for backend in [PickBackend::Gpu, PickBackend::Cpu] {
            let hit = harness
                .renderer
                .pick_object(
                    backend,
                    pixel,
                    &f,
                    &device,
                    &queue,
                    PickMask::OBJECT | PickMask::TILE,
                )
                .unwrap_or_else(|| panic!("{backend:?} misses {id} ({col}, {row})"));
            assert_eq!(hit.id, id, "{backend:?} ({col}, {row})");
            assert_eq!(
                hit.sub_object,
                Some(SubObjectRef::Tile { col, row }),
                "{backend:?} map {id}"
            );
        }
    }
}

/// A rectangle around a few cells' centres of a hexagonal map returns those
/// cells, on either backend.
#[test]
fn a_rectangle_picks_hexagonal_cells_by_their_centres() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (f, items) = orientations(&mut harness);
    let cells = orientation_cells(&harness, &items);
    let pointy = ORIENTATION_PICK_IDS[2];
    // Cells (1, 0), (2, 0), (1, -1) and (2, -1): two rows, the lower one
    // shifted half a cell right.
    let wanted = [[1, 0], [2, 0], [1, -1], [2, -1]];
    let centres: Vec<Vec2> = cells
        .iter()
        .filter(|(id, c, _)| *id == pointy && wanted.contains(c))
        .map(|(_, _, p)| *p)
        .collect();
    assert_eq!(centres.len(), 4);
    let min = centres.iter().fold(Vec2::MAX, |a, &b| a.min(b)) - 2.0;
    let max = centres.iter().fold(Vec2::MIN, |a, &b| a.max(b)) + 2.0;
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    for backend in [PickBackend::Cpu, PickBackend::Gpu] {
        let result = harness.renderer.pick_rect_objects(
            backend,
            min,
            max,
            &f,
            &device,
            &queue,
            PickMask::OBJECT | PickMask::TILE,
        );
        let mut got: Vec<[i32; 2]> = result
            .elements
            .iter()
            .filter(|(id, _)| *id == pointy)
            .filter_map(|(_, s)| match s {
                SubObjectRef::Tile { col, row } => Some([*col, *row]),
                _ => None,
            })
            .collect();
        got.sort();
        got.dedup();
        if backend == PickBackend::Cpu {
            let mut want = wanted.to_vec();
            want.sort();
            assert_eq!(got, want, "{backend:?}");
        } else {
            // The GPU reads every covered pixel, so the cells the rectangle
            // crosses come back too; the four always do.
            assert!(
                wanted.iter().all(|c| got.contains(c)),
                "{backend:?} {got:?}"
            );
        }
    }
}

/// A layer sorted by row on an isometric grid reports one batch per line of
/// cells standing at one height on the plane, and on a grid of staggered
/// columns the shifted columns of a row are a line of their own.
#[test]
fn rows_of_other_grids_are_their_lines_on_the_plane() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    for (orientation, cells, lines) in [
        // A 4 x 4 block of diamonds: seven lines, `row - col` from -3 to 3.
        (TileOrientation::Isometric, (0..4, 0..4), 7),
        // Three columns and two rows, the odd column shifted: four lines.
        (
            TileOrientation::Staggered {
                axis: StaggerAxis::X,
                index: StaggerIndex::Odd,
            },
            (0..3, 0..2),
            4,
        ),
        // Pointy hexagons: each row is a line, shifted or not.
        (
            TileOrientation::Hexagonal {
                side_length: 4,
                axis: StaggerAxis::Y,
                index: StaggerIndex::Odd,
            },
            (0..3, 0..3),
            3,
        ),
    ] {
        let mut map = Tilemap::sparse(Tileset::new(None, [8, 8], 1), 8);
        map.tileset.orientation = orientation;
        for row in cells.1.clone() {
            for col in cells.0.clone() {
                map.set(col, row, Tile::new(0));
            }
        }
        let id = upload(&mut harness, &map);
        let mut item = TilemapItem::new(id);
        item.cell_size = [4.0, 4.0];
        item.draw = TilemapDraw::Blended {
            layer: 0,
            depth_test: false,
            painter: TilemapPainter::PerRow { offset: 0.0 },
        };
        harness.render(&frame(item, Vec2::new(4.0, 0.0), 40), 40, 40);
        assert_eq!(painter_batches(&harness), lines, "{orientation:?}");
    }
}

/// The orientations draw on the compatibility tier as on the full tier: the
/// tiers shade colours differently, but two cell centres of a map show the
/// same colour on one tier exactly when they do on the other.
#[test]
fn orientations_draw_on_the_compatibility_tier() {
    let Some(mut full) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (f, items) = orientations(&mut full);
    let cells = orientation_cells(&full, &items);
    let reference = full.render(&f, 400, 300);
    let Some(mut compat) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("tilemap-orientations"))
    else {
        return;
    };
    assert_eq!(compat.renderer.tier(), Tier::Compatibility);
    let (f, _) = orientations(&mut compat);
    let img = compat.render(&f, 400, 300);
    // The figure is the one item skipped: sprites do not draw there.
    assert_eq!(compat.stats().items_skipped_by_tier, 1);
    let at = |img: &[u8], p: Vec2| -> [u8; 3] {
        let i = ((p.y as usize) * 400 + p.x as usize) * 4;
        [img[i], img[i + 1], img[i + 2]]
    };
    let seen: Vec<(u64, [u8; 3], [u8; 3])> = cells
        .iter()
        .map(|&(id, _, p)| (id, at(&reference, p), at(&img, p)))
        .collect();
    let mut colours = 0;
    for (i, a) in seen.iter().enumerate() {
        for (j, b) in seen.iter().enumerate().skip(i + 1) {
            if a.0 != b.0 {
                continue;
            }
            assert_eq!(
                a.1 == b.1,
                a.2 == b.2,
                "map {}: cells {:?} and {:?}",
                a.0,
                cells[i].1,
                cells[j].1
            );
        }
        colours += seen[..i].iter().all(|b| b.2 != a.2) as usize;
    }
    // Light, dark and a yellow arrow centre or two in every map.
    assert!(colours >= 8, "{colours} colours");
}
