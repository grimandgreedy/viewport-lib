//! The 2D light layers: they run whenever an item is submitted, so ambient
//! alone darkens lit content, an occluder darkens what lies behind it from
//! the light to the ambient level, lit content samples the layer of its own
//! range only and unlit content draws after it, overlapping items raise a
//! notice, and they light the LDR path and the compatibility tier too, never
//! the background.

use glam::{DVec3, Vec2};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, CameraFrame, Candela, Colour, FrameData, FramePath, LightSource, NoticeDetail,
    OverlayFrame, OverlayShape, OverlayShapeItem, PipelineMode, Plane2d, SceneFrame, Tier,
    WorldOverlayLayer, WorldPlane,
};
use viewport_lib_plugins::item_types::lighting_2d::{
    Light2d, Lighting2dItem, Lighting2dPlugin, Lighting2dStats, Occluder2d, TYPE_NAME,
};
use viewport_lib_plugins::item_types::sprite::{self, SpriteItem, SpriteOrder, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{Tile, Tilemap, TilemapItem, Tileset};
use viewport_lib_testkit::scenes::{frame_for, scene_by_name};
use viewport_lib_testkit::{DeviceProfile, Harness};

const W: u32 = 160;
const H: u32 = 100;

/// A grey floor sprite under the light layer, lit by `lights`.
fn frame(item: Lighting2dItem, lights: Vec<LightSource>) -> FrameData {
    floor_frame(0.5, Some(item), lights)
}

/// A floor sprite of linear grey `grey`, under `item` when there is one.
fn floor_frame(grey: f32, item: Option<Lighting2dItem>, lights: Vec<LightSource>) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    let mut floor = SpriteItem::default();
    floor.positions = vec![[0.0, 0.0, 0.0]];
    floor.sizes = vec![[W as f32, H as f32]];
    floor.size_mode = SpriteSizeMode::WorldSpace;
    floor.default_colour = Colour::linear_rgb(grey, grey, grey);
    floor.depth_write = true;
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![floor];
    if let Some(item) = item {
        *scene.items_mut::<Lighting2dItem>() = vec![item];
    }
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    );
    frame.effects.lighting.lights = lights;
    frame.viewport.show_axes_indicator = false;
    frame
}

fn torch(x: f64) -> LightSource {
    LightSource::point_candela(DVec3::new(x, 0.0, 5.0), Candela(1.0), 120.0, 0.0)
}

fn stats(harness: &Harness) -> Lighting2dStats {
    harness
        .renderer
        .item_type_plugin::<Lighting2dPlugin>(TYPE_NAME)
        .expect("registered")
        .last_stats()
}

fn luma(pixels: &[u8], x: f32, y: f32) -> u32 {
    let px = (W as f32 / 2.0 + x) as usize;
    let py = (H as f32 / 2.0 - y) as usize;
    let i = (py * W as usize + px) * 4;
    pixels[i] as u32 + pixels[i + 1] as u32 + pixels[i + 2] as u32
}

/// The centre luma of an unlit floor of linear grey `grey`: what a lit floor
/// whose light multiplies 0.5 grey to `grey` should read, through the same
/// tone mapping.
fn floor_luma(harness: &mut Harness, grey: f32) -> u32 {
    let pixels = harness.render(&floor_frame(grey, None, Vec::new()), W, H);
    luma(&pixels, 0.0, 0.0)
}

#[test]
fn ambient_alone_darkens_the_scene_with_no_lights() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // An item with no lights and ambient 0.2: the floor at a fifth.
    let expected = floor_luma(&mut harness, 0.1);
    let mut item = Lighting2dItem::default();
    item.ambient = Colour::linear_rgb(0.2, 0.2, 0.2);
    let dark = harness.render(&frame(item, Vec::new()), W, H);
    assert_eq!(
        stats(&harness),
        Lighting2dStats {
            lights: 0,
            edges: 0,
            passes: 1
        }
    );
    for (x, y) in [(0.0, 0.0), (-60.0, 30.0), (70.0, -40.0)] {
        let l = luma(&dark, x, y);
        assert!(
            l.abs_diff(expected) <= 3,
            "at ({x}, {y}): {l}, expected {expected}"
        );
    }
    assert!(floor_luma(&mut harness, 0.5) > expected + 100);

    let lights = vec![torch(-40.0), torch(0.0), torch(40.0)];
    let _ = harness.render(&frame(Lighting2dItem::default(), lights), W, H);
    assert_eq!(
        stats(&harness),
        Lighting2dStats {
            lights: 3,
            edges: 0,
            passes: 1
        }
    );
}

#[test]
fn an_occluder_shadows_what_lies_behind_it() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut item = Lighting2dItem::default();
    item.occluders = vec![Occluder2d::Rect {
        min: [-25.0, -5.0],
        max: [-15.0, 5.0],
    }];
    let pixels = harness.render(&frame(item, vec![torch(-60.0)]), W, H);
    assert_eq!(stats(&harness).edges, 4);
    let behind = luma(&pixels, 10.0, 0.0);
    let beside = luma(&pixels, 10.0, 20.0);
    let in_front = luma(&pixels, -40.0, 0.0);
    // Further into the shadow, where the unshadowed light would be dimmer:
    // the same ambient level, so no light reaches either.
    let deeper = luma(&pixels, 40.0, 0.0);
    assert!(
        behind.abs_diff(deeper) <= 3,
        "behind {behind}, deeper {deeper}"
    );
    assert!(beside > behind + 60, "behind {behind}, beside {beside}");
    assert!(in_front > beside, "beside {beside}, in front {in_front}");
}

#[test]
fn the_light_layers_take_no_pass_of_the_frame() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // The floor is a depth-ordered sprite, lit as layer 0 in the scene pass:
    // no scope and no painter pass is opened for the lighting.
    let _ = harness.render(&frame(Lighting2dItem::default(), vec![torch(0.0)]), W, H);
    let ran = harness.renderer.debug_viewport_passes_ran(0);
    assert!(!ran.contains(&"scope_after_transparent"), "{ran:?}");
    assert!(!ran.contains(&"painter"), "{ran:?}");
}

/// A frame of painter squares, `(layer, x, colour, unlit)` each, over a
/// lighting item.
fn painter_frame(item: Lighting2dItem, squares: &[(u32, f32, Colour, bool)]) -> FrameData {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    let sprites = squares
        .iter()
        .map(|&(layer, x, colour, unlit)| {
            let mut s = SpriteItem::default();
            s.positions = vec![[x, 0.0, 0.0]];
            s.default_size = 20.0;
            s.default_colour = colour;
            s.size_mode = SpriteSizeMode::WorldSpace;
            s.order = SpriteOrder::Painter {
                layer,
                depth_test: false,
            };
            s.unlit = unlit;
            s
        })
        .collect();
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = sprites;
    *scene.items_mut::<Lighting2dItem>() = vec![item];
    let mut frame = FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    );
    frame.viewport.show_axes_indicator = false;
    frame
}

#[test]
fn an_unlit_sprite_draws_after_the_lit_content_and_keeps_its_colour() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let grey = Colour::linear_rgb(0.5, 0.5, 0.5);
    let mut item = Lighting2dItem::default();
    item.ambient = Colour::linear_rgb(0.1, 0.1, 0.1);
    let f = painter_frame(item, &[(1, -30.0, grey, false), (1, 30.0, grey, true)]);
    let pixels = harness.render(&f, W, H);
    let seq: Vec<&str> = harness
        .renderer
        .debug_painter_sequence(&f, true)
        .iter()
        .map(|(n, _, _)| *n)
        .collect();
    // The lit square, then the unlit one lifted to the top of the range.
    assert_eq!(seq, vec![sprite::TYPE_NAME, sprite::TYPE_NAME]);
    // The lit square at a tenth, the unlit one as drawn.
    let dim = Colour::linear_rgb(0.05, 0.05, 0.05);
    let mut bare = painter_frame(
        Lighting2dItem::default(),
        &[(1, -30.0, dim, false), (1, 30.0, grey, false)],
    );
    bare.scene.plugin_items.remove(TYPE_NAME);
    let reference = harness.render(&bare, W, H);
    for x in [-30.0, 30.0] {
        let (got, want) = (luma(&pixels, x, 0.0), luma(&reference, x, 0.0));
        assert!(got.abs_diff(want) <= 3, "at {x}: {got}, expected {want}");
    }
    assert!(luma(&pixels, 30.0, 0.0) > luma(&pixels, -30.0, 0.0) + 100);
}

#[test]
fn a_light_range_splits_the_sequence_and_unlit_content_follows_its_range() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let grey = Colour::linear_rgb(0.5, 0.5, 0.5);
    let mut low = Lighting2dItem::default();
    low.layers = 0..=1;
    let mut scene_frame = painter_frame(
        low,
        &[
            (0, -40.0, grey, true),
            (1, -10.0, grey, false),
            (2, 20.0, grey, false),
            (3, 50.0, grey, true),
        ],
    );
    let mut high = Lighting2dItem::default();
    high.layers = 2..=5;
    scene_frame.scene.items_mut::<Lighting2dItem>().push(high);
    let mut layer_frame = OverlayFrame::default();
    layer_frame.shapes.push(OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [0.0, 0.0],
        [4.0, 4.0],
    ));
    scene_frame.scene.world_overlays.push(
        WorldOverlayLayer::new(WorldPlane::Xy)
            .with_order(0)
            .with_unlit(true)
            .with_frame(layer_frame.clone()),
    );
    scene_frame.scene.world_overlays.push(
        WorldOverlayLayer::new(WorldPlane::Xy)
            .with_order(4)
            .with_frame(layer_frame),
    );
    harness.render(&scene_frame, W, H);
    let seq: Vec<(&str, std::ops::RangeInclusive<u32>)> = harness
        .renderer
        .debug_painter_sequence(&scene_frame, true)
        .iter()
        .map(|(n, l, _)| (*n, l.clone()))
        .collect();
    assert_eq!(
        seq,
        vec![
            // Lit content of 0..=1, then its unlit sprite and world layer.
            (sprite::TYPE_NAME, 1..=1),
            (sprite::TYPE_NAME, 0..=0),
            ("world_overlay", 0..=0),
            // Lit content of 2..=5 with its world layer at 4, then the unlit
            // sprite at the top of the range.
            (sprite::TYPE_NAME, 2..=2),
            ("world_overlay", 4..=4),
            (sprite::TYPE_NAME, 3..=3),
        ]
    );
    assert_eq!(stats(&harness).passes, 2);
}

#[test]
fn a_lower_range_keeps_its_own_ambient_under_a_darker_upper_range() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let grey = Colour::linear_rgb(0.5, 0.5, 0.5);
    let mut low = Lighting2dItem::default();
    low.layers = 0..=1;
    low.ambient = Colour::linear_rgb(0.6, 0.6, 0.6);
    low.use_frame_lights = false;
    let mut high = Lighting2dItem::default();
    high.layers = 2..=3;
    high.ambient = Colour::linear_rgb(0.1, 0.1, 0.1);
    high.use_frame_lights = false;
    // A layer-0 square, a layer-3 square over its right edge, and an unlit
    // layer-1 square.
    let mut f = painter_frame(
        low,
        &[
            (0, -40.0, grey, false),
            (3, -20.0, grey, false),
            (1, 40.0, grey, true),
        ],
    );
    f.scene.items_mut::<Lighting2dItem>().push(high);
    let pixels = harness.render(&f, W, H);
    assert_eq!(stats(&harness).passes, 2);

    // Each square at its own range's ambient: the lower one is not scaled
    // again by the darker range above it, and the unlit one keeps its colour.
    let mut bare = painter_frame(
        Lighting2dItem::default(),
        &[
            (0, -40.0, Colour::linear_rgb(0.3, 0.3, 0.3), false),
            (3, -20.0, Colour::linear_rgb(0.05, 0.05, 0.05), false),
            (1, 40.0, grey, false),
        ],
    );
    bare.scene.plugin_items.remove(TYPE_NAME);
    let reference = harness.render(&bare, W, H);
    for x in [-45.0, -20.0, 40.0] {
        let (got, want) = (luma(&pixels, x, 0.0), luma(&reference, x, 0.0));
        assert!(got.abs_diff(want) <= 3, "at {x}: {got}, expected {want}");
    }
    assert!(luma(&pixels, -45.0, 0.0) > luma(&pixels, -20.0, 0.0) + 100);
}

#[test]
fn overlapping_light_ranges_raise_a_notice_and_the_first_wins() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut f = frame(Lighting2dItem::default(), vec![torch(0.0)]);
    let mut second = Lighting2dItem::default();
    second.layers = 3..=4;
    f.scene.items_mut::<Lighting2dItem>().push(second);
    let _ = harness.renderer.take_notices();
    harness.render(&f, W, H);
    let notices = harness.renderer.take_notices();
    assert!(
        notices.iter().any(|n| matches!(
            n.detail,
            NoticeDetail::OverlappingLightRanges {
                ignored: 1,
                kept: 0
            }
        )),
        "{notices:?}"
    );
    assert_eq!(stats(&harness).passes, 1);

    // Ranges that do not overlap draw one layer each and raise nothing.
    let mut f = frame(Lighting2dItem::default(), vec![torch(0.0)]);
    f.scene.items_mut::<Lighting2dItem>()[0].layers = 0..=2;
    let mut second = Lighting2dItem::default();
    second.layers = 3..=4;
    f.scene.items_mut::<Lighting2dItem>().push(second);
    harness.render(&f, W, H);
    let notices = harness.renderer.take_notices();
    assert!(
        !notices
            .iter()
            .any(|n| matches!(n.detail, NoticeDetail::OverlappingLightRanges { .. })),
        "{notices:?}"
    );
    assert_eq!(stats(&harness).passes, 2);
}

#[test]
fn a_cone_lights_its_wedge_and_a_global_light_the_whole_layer() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut item = Lighting2dItem::default();
    item.use_frame_lights = false;
    item.lights = vec![Light2d::cone([-70.0, 0.0], 150.0, [1.0, 0.0], 0.2, 0.3)];
    let cone = harness.render(&frame(item.clone(), Vec::new()), W, H);
    let (on, off) = (luma(&cone, 0.0, 0.0), luma(&cone, 0.0, 40.0));
    assert!(on > off + 100, "on axis {on}, off axis {off}");
    // Behind the cone's tip nothing is lit.
    assert!(luma(&cone, -75.0, 0.0).abs_diff(off) <= 3);

    item.lights = vec![Light2d::global().with_intensity(0.3)];
    let global = harness.render(&frame(item, Vec::new()), W, H);
    let expected = floor_luma(&mut harness, 0.5 * 0.5);
    for (x, y) in [(0.0, 0.0), (-70.0, 40.0), (70.0, -40.0)] {
        let l = luma(&global, x, y);
        assert!(
            l.abs_diff(expected) <= 3,
            "at ({x}, {y}): {l}, expected {expected}"
        );
    }
}

#[test]
fn the_layer_lights_the_ldr_path() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let ldr = |lights| {
        let mut f = frame(Lighting2dItem::default(), lights);
        f.effects.display.mode = PipelineMode::Direct;
        f
    };
    let unlit = harness.render(&ldr(Vec::new()), W, H);
    assert_eq!(harness.stats().frame_path, FramePath::Ldr);
    assert_eq!(stats(&harness).passes, 1);
    let lit = harness.render(&ldr(vec![torch(0.0)]), W, H);
    assert_eq!(harness.stats().frame_path, FramePath::Ldr);
    assert_eq!(stats(&harness).passes, 1);
    // The default ambient darkens the floor away from the light; the light
    // brightens it near the light.
    let (near, far) = (luma(&lit, 0.0, 0.0), luma(&lit, 75.0, 45.0));
    assert!(near > far + 60, "near {near}, far {far}");
    assert_ne!(unlit, lit);
}

#[test]
fn the_layer_lights_the_compatibility_tier() {
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("lighting-2d-compat"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), Tier::Compatibility);
    assert!(harness.renderer.has_item_type_plugin(TYPE_NAME));
    // Sprites are full-tier only, so a cutout tile layer over the left half
    // of the view is lit, as layer 0; the background on the right is not.
    let mut map = Tilemap::with_chunk_size(Tileset::new(None, [8, 8], 1), [8, 10], 8);
    for row in 0..10 {
        for col in 0..8 {
            map.set(col, row, Tile::new(0));
        }
    }
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    let mut tiles = TilemapItem::new(id);
    tiles.origin = [-80.0, -50.0];
    tiles.cell_size = [10.0, 10.0];
    tiles.tint = Colour::linear_rgb(0.5, 0.5, 0.5);
    let compat = |item: Option<Lighting2dItem>, lights| {
        let mut scene = SceneFrame::default();
        *scene.items_mut::<TilemapItem>() = vec![tiles.clone()];
        if let Some(item) = item {
            *scene.items_mut::<Lighting2dItem>() = vec![item];
        }
        let camera = Camera2d::new(Plane2d::Xy)
            .with_units_per_pixel(1.0)
            .with_viewport_size(Vec2::new(W as f32, H as f32))
            .camera();
        let mut f = FrameData::new(
            CameraFrame::from_camera(&camera, [W as f32, H as f32]),
            scene,
        );
        f.effects.lighting.lights = lights;
        f.viewport.show_axes_indicator = false;
        f.viewport.background_colour = Some(Colour::linear_rgb(0.5, 0.5, 0.5));
        f
    };
    let bare = harness.render(&compat(None, Vec::new()), W, H);
    let unlit = harness.render(&compat(Some(Lighting2dItem::default()), Vec::new()), W, H);
    assert_eq!(stats(&harness).passes, 1);
    let lit = harness.render(
        &compat(Some(Lighting2dItem::default()), vec![torch(-40.0)]),
        W,
        H,
    );
    assert_eq!(stats(&harness).passes, 1);
    let (near, far) = (luma(&lit, -40.0, 0.0), luma(&lit, -75.0, 45.0));
    assert!(near > far + 60, "near {near}, far {far}");
    assert!(luma(&unlit, -40.0, 0.0) < luma(&bare, -40.0, 0.0));
    assert_ne!(unlit, lit);
    // The background keeps its colour on this path too.
    for frame in [&unlit, &lit] {
        let (got, want) = (luma(frame, 40.0, 0.0), luma(&bare, 40.0, 0.0));
        assert_eq!(got, want, "background");
    }
}

/// The summed luma of the `rgba8` pixels at `at`.
fn summed(pixels: &[u8], at: &[usize]) -> u64 {
    at.iter()
        .map(|&i| pixels[i * 4] as u64 + pixels[i * 4 + 1] as u64 + pixels[i * 4 + 2] as u64)
        .sum()
}

#[test]
fn every_painter_content_type_samples_its_light_layer() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (w, h) = (240u32, 160u32);
    let size = [w as f32, h as f32];
    let mut dark = Lighting2dItem::default();
    dark.ambient = Colour::linear_rgb(0.2, 0.2, 0.2);
    dark.use_frame_lights = false;

    // Painter sprites, a painter-placed surface and an instanced pair, and a
    // world overlay layer on the same layer.
    let scene = scene_by_name("painter_composed_keys").expect("catalogued");
    let mut built = harness.build_scene(&scene);
    let camera = scene.cameras[0].camera.clone();
    let mut rect = OverlayFrame::default();
    rect.shapes.push(OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [60.0, 10.0],
        [30.0, 30.0],
    ));
    let world = WorldOverlayLayer::new(WorldPlane::Xy)
        .with_order(1)
        .with_frame(rect);
    let with_world = |built: &viewport_lib_testkit::scenes::BuiltScene,
                      world: Option<WorldOverlayLayer>| {
        let mut f = frame_for(built, &camera, size);
        f.scene.world_overlays.extend(world);
        f.viewport.show_axes_indicator = false;
        f
    };
    let bare = harness.render(&with_world(&built, Some(world.clone())), w, h);
    let no_world = harness.render(&with_world(&built, None), w, h);
    built.lighting_2d = vec![dark.clone()];
    let lit = harness.render(&with_world(&built, Some(world.clone())), w, h);
    let pixel = |x: f32, y: f32| {
        ((h as f32 / 2.0 - y) as usize) * w as usize + (w as f32 / 2.0 + x) as usize
    };
    // A sprite, the surface, an instance of the pair.
    for (what, at) in [
        ("sprite", pixel(-80.0, 45.0)),
        ("surface", pixel(-36.0, 25.0)),
        ("instance", pixel(-40.0, -39.0)),
    ] {
        let (got, was) = (summed(&lit, &[at]), summed(&bare, &[at]));
        assert!(got * 10 < was * 7, "{what}: {got}, unlit {was}");
    }
    let layer: Vec<usize> = (0..(w * h) as usize)
        .filter(|&i| bare[i * 4..i * 4 + 3] != no_world[i * 4..i * 4 + 3])
        .collect();
    assert!(
        layer.len() > 100,
        "the world layer drew {} pixels",
        layer.len()
    );
    let (got, was) = (summed(&lit, &layer), summed(&bare, &layer));
    assert!(got * 10 < was * 7, "world layer: {got}, unlit {was}");
    // Marked unlit, it keeps its colour where it covers the background (it
    // is translucent, and over content it shows that content lit).
    let lit_unlit = harness.render(&with_world(&built, Some(world.with_unlit(true))), w, h);
    let background = &no_world[..4];
    let open: Vec<usize> = layer
        .iter()
        .copied()
        .filter(|&i| &no_world[i * 4..i * 4 + 4] == background)
        .collect();
    assert!(
        open.len() > 100,
        "{} pixels over the background",
        open.len()
    );
    for &i in &open {
        for c in 0..3 {
            let (got, was) = (lit_unlit[i * 4 + c], bare[i * 4 + c]);
            assert!(
                got.abs_diff(was) <= 3,
                "unlit world layer: {got}, unlit {was}"
            );
        }
    }

    // Blended tiles, painter particles and painter sprites.
    let scene = scene_by_name("painter_particles_layers").expect("catalogued");
    let mut built = harness.build_scene(&scene);
    let camera = scene.cameras[0].camera.clone();
    let all: Vec<usize> = (0..(w * h) as usize).collect();
    let mut f = frame_for(&built, &camera, size);
    f.viewport.show_axes_indicator = false;
    let bare = harness.render(&f, w, h);
    built.lighting_2d = vec![dark];
    let mut f = frame_for(&built, &camera, size);
    f.viewport.show_axes_indicator = false;
    let lit = harness.render(&f, w, h);
    let tile = pixel(-90.0, 0.0);
    let (got, was) = (summed(&lit, &[tile]), summed(&bare, &[tile]));
    assert!(got * 10 < was * 7, "tile: {got}, unlit {was}");
    let (got, was) = (summed(&lit, &all), summed(&bare, &all));
    assert!(got * 10 < was * 7, "the frame: {got}, unlit {was}");
}
