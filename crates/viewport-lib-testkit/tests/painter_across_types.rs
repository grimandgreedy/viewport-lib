//! The painter sequence across item types: sprites placed by every key they
//! hold, meshes and particles in the sequence, composed keys keeping an
//! entity's parts together, picks in draw order, the sprite pick alpha test,
//! and the painter pass after the transparency resolve.

use glam::Vec2;
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, PickBackend, PickId, PickMask, Plane2d, SceneFrame,
    TextureData, painter_key,
};
use viewport_lib_plugins::item_types::sprite::{self, SpriteItem, SpriteOrder, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{
    self, Tile, Tilemap, TilemapDraw, TilemapItem, TilemapPainter, Tileset,
};
use viewport_lib_testkit::{Harness, catalogue, frame_for};

const W: u32 = 200;
const H: u32 = 150;

/// The name the renderer's own painter meshes run under in the sequence.
const MESHES: &str = "vpl.painter_meshes";

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

/// Painter squares at the origin, one per key.
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

/// The sprite type's draw calls and painter batches for the last frame.
fn sprite_counters(harness: &Harness) -> (u32, u32) {
    let (_, c) = harness
        .renderer
        .plugin_frame_counters()
        .find(|(name, _)| *name == sprite::TYPE_NAME)
        .expect("sprites registered");
    (c.draw_calls, c.painter_batches)
}

fn pick(harness: &mut Harness, frame: &FrameData, at: Vec2) -> Option<u64> {
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    harness
        .renderer
        .pick_object(
            PickBackend::Gpu,
            at,
            frame,
            &device,
            &queue,
            PickMask::OBJECT | PickMask::INSTANCE,
        )
        .map(|hit| hit.id)
}

/// The pixel at world `(x, y)` of a `W` x `H` frame.
fn pixel_at(x: f32, y: f32) -> Vec2 {
    Vec2::new(W as f32 * 0.5 + x, H as f32 * 0.5 - y)
}

#[test]
fn interleaved_sprite_items_interleave_per_sprite_and_count_their_draws() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let red = Colour::linear(0.9, 0.1, 0.1, 1.0);
    let blue = Colour::linear(0.1, 0.1, 0.9, 1.0);

    // Interleaved keys: every sprite is a piece of its own and a draw of its
    // own.
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![
        squares(0, &[0.0, 2.0, 4.0], red),
        squares(0, &[1.0, 3.0, 5.0], blue),
    ];
    let interleaved = frame(scene);
    harness.render(&interleaved, W, H);
    let seq = harness.renderer.debug_painter_sequence(&interleaved, true);
    assert_eq!(seq.len(), 1, "one run of the sprite type: {seq:?}");
    assert_eq!(seq[0].2.len(), 6);
    assert_eq!(sprite_counters(&harness), (6, 6));
    assert_eq!(harness.renderer.last_frame_stats().painter_batches, 6);

    // Keys that do not overlap: still one batch per key, but each item's
    // consecutive pieces draw as one call.
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![
        squares(0, &[0.0, 1.0, 2.0], red),
        squares(0, &[3.0, 4.0, 5.0], blue),
    ];
    let apart = frame(scene);
    harness.render(&apart, W, H);
    assert_eq!(sprite_counters(&harness), (2, 6));

    // Equal keys share a batch.
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![squares(0, &[1.0, 1.0, 1.0, 2.0], red)];
    let equal = frame(scene);
    harness.render(&equal, W, H);
    assert_eq!(sprite_counters(&harness), (1, 2));
}

#[test]
fn composed_keys_put_a_mesh_between_two_sprites_of_one_entity() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == "painter_composed_keys")
        .expect("catalogued");
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [240.0, 160.0]);
    harness.render(&frame, 240, 160);
    let seq: Vec<&str> = harness
        .renderer
        .debug_painter_sequence(&frame, true)
        .iter()
        .map(|(name, _, _)| *name)
        .collect();
    assert_eq!(
        seq,
        vec![
            sprite::TYPE_NAME,
            MESHES,
            sprite::TYPE_NAME,
            MESHES,
            sprite::TYPE_NAME
        ],
        "the instanced pair and the flat surface each sit between their entity's sprites"
    );
    let core = harness.renderer.last_frame_stats().painter_batches - sprite_counters(&harness).1;
    assert_eq!(core, 2, "one instanced batch and one surface");

    // Picks follow the stacking. Top row squares are 40 across at
    // (-80 + 22 i, 45 - 10 i): red, green, blue surface, yellow, white. The
    // frame is 240 x 160.
    let px = |x: f32, y: f32| Vec2::new(120.0 + x, 80.0 - y);
    // Inside the green sprite and the blue surface, outside the yellow: the
    // surface (member 1) is over the green sprite (member 0).
    let between_green_and_blue = px(-45.0, 40.0);
    assert_eq!(
        pick(&mut harness, &frame, between_green_and_blue),
        Some(4103)
    );
    // Inside the blue surface and the yellow sprite, outside the white.
    let between_blue_and_yellow = px(-25.0, 20.0);
    assert_eq!(
        pick(&mut harness, &frame, between_blue_and_yellow),
        Some(4101)
    );
    // The instanced pair over the cyan sprite, under the purple one.
    assert_eq!(pick(&mut harness, &frame, px(-45.0, -42.0)), Some(4105));
    assert_eq!(pick(&mut harness, &frame, px(28.0, -42.0)), Some(4104));
}

/// A blended tile layer of one 32-unit cell at the origin, on `layer`.
fn tile_layer(harness: &mut Harness, layer: u32) -> TilemapItem {
    let atlas = harness
        .renderer
        .resources_mut()
        .upload_texture(
            &harness.device,
            &harness.queue,
            TextureData::srgb(1, 1, vec![30, 200, 60, 255]),
        )
        .expect("texture upload");
    let mut map = Tilemap::with_chunk_size(Tileset::new(Some(atlas), [1, 1], 1), [1, 1], 1);
    map.set(0, 0, Tile::new(0));
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    let mut item = TilemapItem::new(id);
    item.origin = [-16.0, -16.0];
    item.cell_size = [32.0, 32.0];
    item.draw = TilemapDraw::Blended {
        layer,
        depth_test: false,
        painter: TilemapPainter::default(),
    };
    item.settings.pick_id = PickId(7);
    item
}

#[test]
fn a_pick_over_a_tile_layer_and_a_sprite_returns_the_one_on_top() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // The sprite type is registered before the tilemap type, so the two
    // stackings put the earlier-registered type on top once and underneath
    // once.
    for (tiles_layer, sprite_layer, expected) in [(1, 0, 7), (0, 1, 8)] {
        let tiles = tile_layer(&mut harness, tiles_layer);
        let mut sprite = squares(sprite_layer, &[0.0], Colour::linear(0.9, 0.2, 0.2, 1.0));
        sprite.settings.pick_id = PickId(8);
        let mut scene = SceneFrame::default();
        *scene.items_mut::<SpriteItem>() = vec![sprite];
        *scene.items_mut::<TilemapItem>() = vec![tiles];
        let frame = frame(scene);
        harness.render(&frame, W, H);
        let names: Vec<&str> = harness
            .renderer
            .debug_painter_sequence(&frame, true)
            .iter()
            .map(|(n, _, _)| *n)
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert_eq!(
            pick(&mut harness, &frame, pixel_at(4.0, 4.0)),
            Some(expected),
            "tiles on layer {tiles_layer}, sprite on layer {sprite_layer}"
        );
    }
    let _ = tilemap::TYPE_NAME;
}

#[test]
fn a_pick_through_a_transparent_texel_misses_under_the_threshold() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Left column transparent, right column opaque.
    let texture = harness
        .renderer
        .resources_mut()
        .upload_texture(
            &harness.device,
            &harness.queue,
            TextureData::srgb(2, 1, vec![255, 255, 255, 0, 255, 255, 255, 255]),
        )
        .expect("texture upload");
    let cutout = |threshold: Option<f32>| {
        let mut item = squares(0, &[0.0], Colour::linear(1.0, 1.0, 1.0, 1.0));
        item.texture_id = Some(texture);
        item.filter = viewport_lib::TextureFilter::Nearest;
        item.pick_alpha_threshold = threshold;
        item.settings.pick_id = PickId(9);
        item
    };
    let transparent = pixel_at(-10.0, 0.0);
    let opaque = pixel_at(10.0, 0.0);
    for (threshold, through_hole) in [(None, Some(9)), (Some(0.5), None)] {
        let mut scene = SceneFrame::default();
        *scene.items_mut::<SpriteItem>() = vec![cutout(threshold)];
        let frame = frame(scene);
        harness.render(&frame, W, H);
        assert_eq!(
            pick(&mut harness, &frame, transparent),
            through_hole,
            "threshold {threshold:?}"
        );
        assert_eq!(pick(&mut harness, &frame, opaque), Some(9));
    }
}

#[test]
fn painter_content_draws_in_its_own_pass_after_the_transparency_resolve() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == "painter_over_transparent")
        .expect("catalogued");
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [240.0, 160.0]);
    let pixels = harness.render(&frame, 240, 160);
    let ran = harness.renderer.debug_viewport_passes_ran(0);
    let at = |name| ran.iter().position(|p| *p == name);
    let (Some(oit), Some(painter)) = (at("oit"), at("painter")) else {
        panic!("both passes run: {ran:?}");
    };
    assert!(oit < painter, "{ran:?}");
    assert!(!ran.contains(&"depth_read"), "{ran:?}");

    // The sprite is the same colour over the cube in front and the one behind.
    let rgba = |x: usize, y: usize| {
        let i = (y * 240 + x) * 4;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    assert_eq!(rgba(120 - 50, 80), rgba(120 + 50, 80));
    assert_eq!(rgba(120 - 50, 80), rgba(120, 80));
}

#[test]
fn particles_sit_between_a_tile_layer_and_sprites_by_layer() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == "painter_particles_layers")
        .expect("catalogued");
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [240.0, 160.0]);
    harness.render(&frame, 240, 160);
    let seq: Vec<(&str, std::ops::RangeInclusive<u32>)> = harness
        .renderer
        .debug_painter_sequence(&frame, true)
        .iter()
        .map(|(n, l, _)| (*n, l.clone()))
        .collect();
    assert_eq!(
        seq,
        vec![
            (tilemap::TYPE_NAME, 1..=1),
            (
                viewport_lib_plugins::item_types::gpu_particles::TYPE_NAME,
                2..=2
            ),
            (sprite::TYPE_NAME, 3..=3),
        ]
    );
}

#[test]
fn painter_keys_compose_in_sort_then_member_order() {
    // A character's parts at one sort value stay together and in order,
    // between the characters above and below it.
    let a = painter_key(2, -10.0, 3);
    let b: Vec<f32> = (0..3).map(|m| painter_key(2, -9.5, m).key).collect();
    let c = painter_key(2, -9.0, 0);
    assert!(a.key < b[0] && b[0] < b[1] && b[1] < b[2] && b[2] < c.key);
}

#[test]
fn the_compatibility_tier_draws_a_painter_surface_between_its_sprites() {
    let Some(mut harness) = Harness::with_profile(
        &viewport_lib_testkit::DeviceProfile::downlevel_webgl2("painter-compatibility"),
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), viewport_lib::Tier::Compatibility);
    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == "painter_composed_keys")
        .expect("catalogued");
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [240.0, 160.0]);
    let pixels = harness.render(&frame, 240, 160);
    let names: Vec<&str> = harness
        .renderer
        .debug_painter_sequence(&frame, false)
        .iter()
        .map(|(n, _, _)| *n)
        .collect();
    assert!(names.contains(&MESHES), "{names:?}");
    let rgba = |x: f32, y: f32| {
        let (x, y) = ((120.0 + x) as usize, (80.0 - y) as usize);
        let i = (y * 240 + x) * 4;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    // The blue surface over the green sprite, under the yellow one.
    let over_green = rgba(-45.0, 40.0);
    assert!(over_green[2] > over_green[1] + 60, "{over_green:?}");
    let under_yellow = rgba(-25.0, 20.0);
    assert!(under_yellow[1] > under_yellow[2] + 60, "{under_yellow:?}");
}
