//! Sprite capabilities: painter masks over sprites and blended tile layers,
//! sliced and tiled draw modes, the sprite and tile vertex hooks (with picks
//! that follow the moved quad), and screen-space sizes in points.
//!
//! One world unit is one pixel, with the world origin in the middle of the
//! view, so the checks read pixel positions straight off the world layout.

use glam::Vec2;
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, CameraFrame, Colour, FrameData, MaskTest, MaterialPlugin, NoticeDetail, OverlayShape,
    PickBackend, PickId, PickMask, Plane2d, SceneFrame, TextureData, TextureFilter, TextureId,
};
use viewport_lib_plugins::item_types::sprite::{
    MaskSource, SpriteDrawMode, SpriteEffects, SpriteItem, SpriteMaskItem, SpriteMaterial,
    SpriteMaterials, SpriteOrder, SpriteSizeMode, TileMode,
};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, TileMaterial, TileMaterials, Tilemap, TilemapDraw, TilemapItem, Tileset,
};
use viewport_lib_testkit::Harness;

const W: u32 = 240;
const H: u32 = 160;

fn camera_frame(ppp: f32) -> CameraFrame {
    let camera = Camera2d::new(Plane2d::Xy)
        .with_units_per_pixel(1.0)
        .with_viewport_size(Vec2::new(W as f32, H as f32))
        .camera();
    CameraFrame::from_camera(&camera, [W as f32, H as f32]).with_pixels_per_point(ppp)
}

fn frame(
    sprites: Vec<SpriteItem>,
    masks: Vec<SpriteMaskItem>,
    tiles: Vec<TilemapItem>,
) -> FrameData {
    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = sprites;
    *scene.items_mut::<SpriteMaskItem>() = masks;
    *scene.items_mut::<TilemapItem>() = tiles;
    FrameData::new(camera_frame(1.0), scene)
}

/// The RGB of the pixel at world `(x, y)` of a `W` by `H` render.
fn pixel(pixels: &[u8], x: f32, y: f32) -> [u8; 3] {
    let col = (W as f32 * 0.5 + x).floor() as usize;
    let row = (H as f32 * 0.5 - y).floor() as usize;
    let i = (row * W as usize + col) * 4;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

fn is_colour(p: [u8; 3], want: [bool; 3]) -> bool {
    p.iter()
        .zip(want)
        .all(|(&c, on)| if on { c > 180 } else { c < 80 })
}

const RED: [bool; 3] = [true, false, false];
const GREEN: [bool; 3] = [false, true, false];
const BLUE: [bool; 3] = [false, false, true];

/// A world-space sprite of one colour.
fn square(at: [f32; 3], size: [f32; 2], colour: Colour) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![at];
    item.sizes = vec![size];
    item.default_colour = colour;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item
}

fn painter(mut item: SpriteItem, layer: u32, mask: MaskTest) -> SpriteItem {
    item.order = SpriteOrder::Painter {
        layer,
        depth_test: false,
    };
    item.mask = mask;
    item
}

/// A circle mask of radius 20 at the origin in `slot` over `layers`.
fn circle(slot: u8, layers: std::ops::RangeInclusive<u32>) -> SpriteMaskItem {
    SpriteMaskItem::new(
        MaskSource::Shape {
            shape: OverlayShape::Circle,
            sprite: square([0.0, 0.0, 0.0], [40.0, 40.0], Colour::WHITE),
        },
        slot,
        layers,
    )
}

fn upload_texture(
    harness: &mut Harness,
    w: u32,
    h: u32,
    texel: impl Fn(u32, u32) -> [u8; 4],
) -> TextureId {
    let pixels: Vec<u8> = (0..w * h).flat_map(|i| texel(i % w, i / w)).collect();
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    harness
        .renderer
        .resources_mut()
        .upload_texture(&device, &queue, TextureData::srgb(w, h, pixels))
        .expect("texture upload")
}

#[test]
fn a_circle_mask_cuts_sprites_inside_and_outside_on_its_layers_only() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let big = |c| square([0.0, 0.0, 0.0], [60.0, 60.0], c);

    let inside = harness.render(
        &frame(
            vec![painter(
                big(Colour::linear_rgb(1.0, 0.0, 0.0)),
                1,
                MaskTest::VisibleInside(0),
            )],
            vec![circle(0, 1..=1)],
            vec![],
        ),
        W,
        H,
    );
    assert!(
        is_colour(pixel(&inside, 0.0, 0.0), RED),
        "inside the circle shows"
    );
    assert!(
        !is_colour(pixel(&inside, 25.0, 0.0), RED),
        "outside the circle is cut"
    );

    let outside = harness.render(
        &frame(
            vec![painter(
                big(Colour::linear_rgb(0.0, 0.0, 1.0)),
                1,
                MaskTest::VisibleOutside(0),
            )],
            vec![circle(0, 1..=1)],
            vec![],
        ),
        W,
        H,
    );
    assert!(
        !is_colour(pixel(&outside, 0.0, 0.0), BLUE),
        "inside the circle is cut"
    );
    assert!(
        is_colour(pixel(&outside, 25.0, 0.0), BLUE),
        "outside the circle shows"
    );

    // On a layer outside the mask's range, and testing a slot no mask drew
    // into, nothing is cut.
    let unaffected = harness.render(
        &frame(
            vec![
                painter(
                    big(Colour::linear_rgb(0.0, 1.0, 0.0)),
                    2,
                    MaskTest::VisibleInside(0),
                ),
                painter(
                    square(
                        [80.0, 0.0, 0.0],
                        [30.0, 30.0],
                        Colour::linear_rgb(1.0, 0.0, 0.0),
                    ),
                    1,
                    MaskTest::VisibleInside(3),
                ),
            ],
            vec![circle(0, 1..=1)],
            vec![],
        ),
        W,
        H,
    );
    assert!(is_colour(pixel(&unaffected, 25.0, 0.0), GREEN));
    assert!(is_colour(pixel(&unaffected, 80.0, 0.0), RED));
}

#[test]
fn a_mask_from_a_sprites_alpha_cuts_a_blended_tile_layer() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // A 16 x 16 mask texture, opaque in its left half only.
    let half = upload_texture(&mut harness, 16, 16, |x, _| {
        [255, 255, 255, if x < 8 { 255 } else { 0 }]
    });
    let mut source = square([0.0, 0.0, 0.0], [64.0, 64.0], Colour::WHITE);
    source.texture_id = Some(half);
    source.filter = TextureFilter::Nearest;
    let mask = SpriteMaskItem::new(
        MaskSource::Alpha {
            sprite: source,
            threshold: 0.5,
        },
        2,
        0..=4,
    );

    let mut map = Tilemap::new(Tileset::new(None, [8, 8], 1), [120, 80]);
    for row in 0..80 {
        for col in 0..120 {
            map.set(col, row, Tile::new(0));
        }
    }
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    let mut layer = TilemapItem::new(id);
    layer.origin = [-60.0, -40.0];
    layer.tint = Colour::linear_rgb(0.0, 1.0, 0.0);
    layer.draw = TilemapDraw::Blended {
        layer: 3,
        depth_test: false,
    };
    layer.mask = MaskTest::VisibleInside(2);

    let pixels = harness.render(&frame(vec![], vec![mask], vec![layer]), W, H);
    assert!(
        is_colour(pixel(&pixels, -16.0, 0.0), GREEN),
        "under the opaque half"
    );
    assert!(
        !is_colour(pixel(&pixels, 16.0, 0.0), GREEN),
        "under the clear half"
    );
    assert!(
        !is_colour(pixel(&pixels, 50.0, 0.0), GREEN),
        "outside the mask's quad"
    );
}

#[test]
fn a_nine_slice_keeps_its_borders_at_texel_size() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // 32 x 32: an 8-texel red border round a green centre.
    let tex = upload_texture(&mut harness, 32, 32, |x, y| {
        let border = x < 8 || y < 8 || x >= 24 || y >= 24;
        if border {
            [255, 0, 0, 255]
        } else {
            [0, 255, 0, 255]
        }
    });
    let mut item = square([0.0, 0.0, 0.0], [200.0, 60.0], Colour::WHITE);
    item.texture_id = Some(tex);
    item.filter = TextureFilter::Nearest;
    item.draw_mode = SpriteDrawMode::Sliced {
        border: [8.0, 8.0, 8.0, 8.0],
    };
    let pixels = harness.render(&frame(vec![item], vec![], vec![]), W, H);
    // The quad spans x -100..100, y -30..30.
    assert!(
        is_colour(pixel(&pixels, -100.0 + 7.5, 0.0), RED),
        "left border, 8 px"
    );
    assert!(
        is_colour(pixel(&pixels, -100.0 + 8.5, 0.0), GREEN),
        "centre starts at 8 px"
    );
    assert!(
        is_colour(pixel(&pixels, 100.0 - 7.5, 0.0), RED),
        "right border, 8 px"
    );
    assert!(is_colour(pixel(&pixels, 100.0 - 8.5, 0.0), GREEN));
    assert!(
        is_colour(pixel(&pixels, 0.0, 30.0 - 7.5), RED),
        "top border, 8 px"
    );
    assert!(is_colour(pixel(&pixels, 0.0, 30.0 - 8.5), GREEN));
    assert!(
        is_colour(pixel(&pixels, 0.0, -30.0 + 7.5), RED),
        "bottom border, 8 px"
    );
    assert!(
        is_colour(pixel(&pixels, 50.0, 0.0), GREEN),
        "the centre stretches"
    );
}

#[test]
fn tiling_repeats_whole_tiles_or_clips_the_last() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // 16 x 16: left half red, right half blue.
    let tex = upload_texture(&mut harness, 16, 16, |x, _| {
        if x < 8 {
            [255, 0, 0, 255]
        } else {
            [0, 0, 255, 255]
        }
    });
    let tiled = |mode, y| {
        let mut item = square([0.0, y, 0.0], [40.0, 16.0], Colour::WHITE);
        item.texture_id = Some(tex);
        item.filter = TextureFilter::Nearest;
        item.draw_mode = SpriteDrawMode::Tiled {
            tile_size: [16.0, 16.0],
            mode,
        };
        item
    };
    let pixels = harness.render(
        &frame(
            vec![
                tiled(TileMode::Continuous, 20.0),
                tiled(TileMode::Adaptive, -20.0),
            ],
            vec![],
            vec![],
        ),
        W,
        H,
    );
    // The quads span x -20..20. Continuous: 16-unit tiles from the left edge,
    // the third cut at 8 units, so 32..40 across is a left (red) half.
    for (at, want) in [
        (4.0, RED),
        (12.0, BLUE),
        (20.0, RED),
        (28.0, BLUE),
        (36.0, RED),
    ] {
        assert!(
            is_colour(pixel(&pixels, -20.0 + at, 20.0), want),
            "continuous at {at}"
        );
    }
    // Adaptive: round(40 / 16) = 2 tiles of 20 units.
    for (at, want) in [(4.0, RED), (14.0, BLUE), (24.0, RED), (36.0, BLUE)] {
        assert!(
            is_colour(pixel(&pixels, -20.0 + at, -20.0), want),
            "adaptive at {at}"
        );
    }
}

/// Moves the top corners right by `shift`.
struct Lean;

impl MaterialPlugin for Lean {
    fn name(&self) -> &str {
        "lean"
    }

    fn wgsl_body(&self) -> String {
        "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
             return v.world + vec3<f32>(select(0.0, params.shift, v.corner >= 2u), 0.0, 0.0);\n}\n"
            .to_string()
    }

    fn param_layout(&self) -> Vec<viewport_lib::ParamDesc> {
        vec![viewport_lib::ParamDesc::f32("shift", 0.0)]
    }
}

#[test]
fn a_vertex_hook_moves_the_top_edge_and_the_pick_follows() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .register_sprite_material(&device, &Lean)
        .expect("a material with only a vertex hook registers");
    let mut item = square(
        [0.0, 0.0, 0.0],
        [40.0, 40.0],
        Colour::linear_rgb(1.0, 0.0, 0.0),
    );
    let mut fx = SpriteEffects::default();
    fx.material = Some(SpriteMaterial::new(id).with_param("shift", 30.0f32));
    item.effects = Some(fx);
    item.settings.pick_id = PickId(77);
    let f = frame(vec![item], vec![], vec![]);
    let pixels = harness.render(&f, W, H);
    // Bottom edge stays at x -20..20; the top edge moves to x 10..50.
    assert!(
        is_colour(pixel(&pixels, -15.0, -18.0), RED),
        "bottom left stays"
    );
    assert!(
        !is_colour(pixel(&pixels, -15.0, 18.0), RED),
        "top left moved away"
    );
    assert!(
        is_colour(pixel(&pixels, 40.0, 18.0), RED),
        "top right moved in"
    );
    assert!(
        !is_colour(pixel(&pixels, 40.0, -18.0), RED),
        "bottom right did not"
    );

    let pick = |harness: &mut Harness, x: f32, y: f32| {
        harness.renderer.pick_object(
            PickBackend::Gpu,
            Vec2::new(W as f32 * 0.5 + x, H as f32 * 0.5 - y),
            &f,
            &device,
            &queue,
            PickMask::OBJECT,
        )
    };
    assert_eq!(
        pick(&mut harness, 40.0, 18.0).map(|h| h.id),
        Some(77),
        "the moved corner picks"
    );
    assert!(
        pick(&mut harness, -15.0, 18.0).is_none(),
        "where it moved from does not"
    );
}

#[test]
fn a_tile_material_moves_tile_corners() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let id = harness
        .renderer
        .register_tile_material(&device, &Lean)
        .expect("registers for tiles");
    // One 40-unit cell at the origin.
    let mut map = Tilemap::new(Tileset::new(None, [8, 8], 1), [1, 1]);
    map.set(0, 0, Tile::new(0));
    let map_id = harness
        .renderer
        .upload(&device, &queue, &map)
        .expect("upload");
    let mut layer = TilemapItem::new(map_id);
    layer.origin = [-20.0, -20.0];
    layer.cell_size = [40.0, 40.0];
    layer.tint = Colour::linear_rgb(1.0, 0.0, 0.0);
    layer.material = Some(TileMaterial::new(id).with_param("shift", 30.0f32));
    let pixels = harness.render(&frame(vec![], vec![], vec![layer]), W, H);
    assert!(
        is_colour(pixel(&pixels, -15.0, -18.0), RED),
        "bottom left stays"
    );
    assert!(
        !is_colour(pixel(&pixels, -15.0, 18.0), RED),
        "top left moved away"
    );
    assert!(
        is_colour(pixel(&pixels, 40.0, 18.0), RED),
        "top right moved in"
    );
}

#[test]
fn screen_space_sizes_are_points() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut item = SpriteItem::default();
    item.positions = vec![[0.0, 0.0, 0.0]];
    item.default_size = 40.0;
    item.default_colour = Colour::linear_rgb(1.0, 0.0, 0.0);
    let covered = |pixels: &[u8], w: u32, h: u32| {
        let row = (h / 2) as usize;
        (0..w as usize)
            .filter(|&x| {
                let i = (row * w as usize + x) * 4;
                pixels[i] > 180 && pixels[i + 1] < 80
            })
            .count()
    };
    let pixels = harness.render(&frame(vec![item.clone()], vec![], vec![]), W, H);
    assert_eq!(
        covered(&pixels, W, H),
        40,
        "40 points at one pixel per point"
    );

    let mut scene = SceneFrame::default();
    *scene.items_mut::<SpriteItem>() = vec![item];
    let doubled = FrameData::new(camera_frame(2.0), scene);
    let pixels = harness.render(&doubled, W * 2, H * 2);
    assert_eq!(
        covered(&pixels, W * 2, H * 2),
        80,
        "40 points at two pixels per point"
    );
}

#[test]
fn a_screen_sized_painter_sprite_raises_a_notice() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut item = SpriteItem::default();
    item.positions = vec![[0.0, 0.0, 0.0]];
    item.order = SpriteOrder::Painter {
        layer: 0,
        depth_test: false,
    };
    let _ = harness.renderer.take_notices();
    harness.render(&frame(vec![item], vec![], vec![]), W, H);
    let notices = harness.renderer.take_notices();
    assert!(
        notices.iter().any(|n| matches!(
            n.detail,
            NoticeDetail::PainterSpriteScreenSized { batches: 1 }
        )),
        "{notices:?}"
    );
}

/// The catalogue's `sprites_2d_modes` on WebGL2 limits, against
/// `<backend>-compatibility/sprites_2d_modes.png`: the masked tile layer and
/// its mask draw there, the sprites (a full-tier type) do not. Bless with
/// `BLESS=1 cargo test --release --test sprite_capabilities`.
#[test]
fn the_modes_scene_draws_its_mask_and_tiles_on_the_compatibility_tier() {
    use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
    use viewport_lib_testkit::{DeviceProfile, catalogue, frame_for};
    let Some(mut harness) =
        Harness::with_profile(&DeviceProfile::downlevel_webgl2("sprites-2d-modes"))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(harness.renderer.tier(), viewport_lib::Tier::Compatibility);
    let scene = catalogue()
        .into_iter()
        .find(|s| s.name == "sprites_2d_modes")
        .expect("in the catalogue");
    let built = harness.build_scene(&scene);
    let (w, h) = (400, 300);
    let frame = frame_for(&built, &scene.cameras[0].camera, [w as f32, h as f32]);
    let _ = harness.render(&frame, w, h);
    let pixels = harness.render(&frame, w, h);

    let snapshots = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
    let dir = snapshots.join(format!("{}-compatibility", harness.backend_dir_name()));
    let mode = GoldenMode::from_env();
    let actual = RgbaImage::from_raw(w, h, pixels).expect("frame -> image");
    match golden::check(&dir, "sprites_2d_modes", &actual, Tolerance::EXACT, mode) {
        Outcome::Blessed => eprintln!("blessed sprites_2d_modes in {}", dir.display()),
        Outcome::Rendered { path } => eprintln!("render only: {}", path.display()),
        Outcome::Match => {}
        Outcome::Missing { .. } => panic!("{}", golden::missing_message("sprites_2d_modes", &dir)),
        Outcome::Mismatch { comparison, .. } => panic!(
            "sprites_2d_modes: max_channel_delta={} pixels_over={}/{}",
            comparison.max_channel_delta, comparison.pixels_over, comparison.total_pixels
        ),
    }
}
