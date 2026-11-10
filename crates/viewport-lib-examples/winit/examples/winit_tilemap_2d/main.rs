//! A scrolling tilemap with two parallax layers under `Camera2d`, with pixel
//! art sprites walking over it.
//!
//! A far layer of hills scrolls at a third of the camera's speed behind a
//! ground layer of grass, paths and animated water; three hundred critters
//! walk over the ground as painter-ordered sprites sorted by `-y`. A signpost
//! of vector art and a glowing additive path sit on a world overlay layer
//! between the ground and the critters, so they scroll and zoom with the map
//! and take bloom. Damage numbers float up from the critters and fade, and
//! one critter carries a name tag: both are screen labels anchored to world
//! points and sized in world units, so they scale with the zoom while facing
//! the screen, and the name tag never shrinks below 12 pixels. The camera
//! drifts to the right on its own.
//!
//! Drag to pan, scroll or pinch to zoom, `P` to switch between crisp pixel art
//! (nearest-sampled and snapped to the pixel grid) and smooth filtering.

use std::cell::Cell;
use std::rc::Rc;

use glam::Vec2;

use viewport_lib as vpl;
use viewport_lib::plugin_api::Uploads;
use viewport_lib_plugins::item_types::sprite::{
    SpriteFlip, SpriteItem, SpriteOrder, SpriteSizeMode,
};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, TileAnimation, TileFlip, TileFrame, Tilemap, TilemapId, TilemapItem, Tileset,
};
use vpl::{
    Action, Alignment, AnchorX, AnchorY, AppConfig, Camera2d, Camera2dController, Colour, ExtraId,
    FillRule, KeyCode, LabelItem, LineCap, Modifiers, ModifiersMatch, OverlayBlend, OverlayFill,
    OverlayFrame, OverlayPolylineItem, OverlayShape, OverlayShapeItem, OverlayStroke, Plane2d,
    ShadowLayer, SubPath, TextureData, TextureFilter, TextureId, ViewportApp, ViewportBinding,
    ViewportGesture, WindowConfig, WorldOverlayLayer, WorldPlane, WorldTextSize, planar_bindings,
};

const TOGGLE_PIXEL_ART: Action = Action::Custom(1);

/// Tiles across and up the ground map.
const MAP: [u32; 2] = [256, 32];
/// World units per ground cell: one texel is one unit.
const CELL: f32 = 8.0;
const CRITTERS: usize = 300;

// Atlas tiles, four to a row.
const GRASS: u32 = 0;
const PATH: u32 = 1;
const WATER: u32 = 2;
const HILL: u32 = 3;
const SKY: u32 = 4;
const EDGE: u32 = 5;
const WATER_B: u32 = 6;
const CRITTER: u32 = 7;

const ART: [[&str; 8]; 8] = [
    [
        "GGGGGGGG", "GgGGGGgG", "GGGGGGGG", "GGGgGGGG", "GGGGGGGG", "GgGGGgGG", "GGGGGGGG",
        "GGGGGGGG",
    ],
    [
        "DDDDDDDD", "DdDDDDDD", "DDDDDdDD", "DDDDDDDD", "DDdDDDDD", "DDDDDDdD", "DDDDDDDD",
        "DdDDDDDD",
    ],
    [
        "wwwwwwww", "wWwwwwww", "wwwwwwww", "wwwwWwww", "wwwwwwww", "wwwwwwwW", "wwwwwwww",
        "wwwwwwww",
    ],
    [
        "........", "...hh...", "..hhhh..", ".hhhhhh.", "hhhhhhhh", "hhhhhhhh", "hhhhhhhh",
        "hhhhhhhh",
    ],
    [
        "ssssssss", "ssssssss", "ssSSssss", "sSSSSsss", "ssssssss", "ssssssss", "ssssSSss",
        "sssSSSSs",
    ],
    // Grass edge: grass along the top, path below, flipped for the far side.
    [
        "GGGGGGGG", "GGGGGGGG", "GgGGGgGG", "DGDDGDDG", "DDDDDDDD", "DdDDDDDD", "DDDDDdDD",
        "DDDDDDDD",
    ],
    [
        "wwwwwwww", "wwwwwwww", "wwWwwwww", "wwwwwwww", "wwwwwwWw", "wwwwwwww", "Wwwwwwww",
        "wwwwwwww",
    ],
    [
        "..KKKK..", ".KKKKKK.", "KKWBKWBK", "KKKKKKKK", "KRRRRRRK", ".KKKKKK.", ".K....K.",
        "KK....KK",
    ],
];

fn palette(c: char) -> [u8; 4] {
    match c {
        'G' => [70, 160, 70, 255],
        'g' => [110, 200, 90, 255],
        'D' => [150, 110, 70, 255],
        'd' => [120, 85, 55, 255],
        'w' => [40, 90, 190, 255],
        'W' => [220, 235, 255, 255],
        'h' => [70, 100, 120, 255],
        's' => [120, 160, 215, 255],
        'S' => [210, 222, 240, 255],
        'K' => [230, 140, 60, 255],
        'B' => [20, 20, 30, 255],
        'R' => [200, 50, 60, 255],
        _ => [0, 0, 0, 0],
    }
}

/// The atlas, row 0 of the image at the top as a tileset image is drawn.
fn atlas() -> TextureData {
    let (w, h) = (32usize, 16usize);
    let mut pixels = vec![0u8; w * h * 4];
    for (i, art) in ART.iter().enumerate() {
        let (tx, ty) = ((i % 4) * 8, (i / 4) * 8);
        for (y, line) in art.iter().enumerate() {
            for (x, c) in line.chars().enumerate() {
                let at = ((ty + y) * w + tx + x) * 4;
                pixels[at..at + 4].copy_from_slice(&palette(c));
            }
        }
    }
    TextureData::srgb(w as u32, h as u32, pixels)
}

fn tileset(atlas: TextureId, filter: TextureFilter) -> Tileset {
    let mut set = Tileset::new(Some(atlas), [8, 8], 4);
    set.filter = filter;
    set.animations.push(TileAnimation {
        tile: WATER,
        frames: vec![
            TileFrame {
                tile: WATER,
                duration: 0.4,
            },
            TileFrame {
                tile: WATER_B,
                duration: 0.4,
            },
        ],
    });
    set
}

/// A small deterministic generator, so the map is the same every run.
fn next(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed as f32 / u32::MAX as f32
}

/// Grass with a winding path and a few ponds.
fn ground(set: Tileset) -> Tilemap {
    let mut map = Tilemap::new(set, MAP);
    let mut seed = 0x7111_E5ED;
    for col in 0..MAP[0] {
        let path = (12.0 + 6.0 * (col as f32 * 0.07).sin()) as u32;
        for row in 0..MAP[1] {
            let tile = if row == path + 2 {
                Tile::new(EDGE)
            } else if row == path.saturating_sub(2) {
                Tile::new(EDGE).with_flip(TileFlip::VERTICAL)
            } else if row.abs_diff(path) < 2 {
                Tile::new(PATH)
            } else {
                Tile::new(GRASS)
            };
            map.set(col, row, tile);
        }
    }
    for _ in 0..20 {
        let (c, r) = (
            (next(&mut seed) * (MAP[0] - 4) as f32) as u32,
            (next(&mut seed) * 6.0) as u32 + if next(&mut seed) < 0.5 { 1 } else { 24 },
        );
        for dc in 0..3 {
            for dr in 0..2 {
                map.set(c + dc, r + dr, Tile::new(WATER));
            }
        }
    }
    map
}

/// The far layer: sky over a band of hills.
fn hills(set: Tileset) -> Tilemap {
    let mut map = Tilemap::new(set, [128, 24]);
    for col in 0..128 {
        for row in 0..24 {
            map.set(col, row, Tile::new(if row < 10 { HILL } else { SKY }));
        }
    }
    map
}

struct Critter {
    position: Vec2,
    velocity: Vec2,
}

fn critters() -> Vec<Critter> {
    let mut seed = 0x2D5E_ED01;
    let extent = Vec2::new(MAP[0] as f32, MAP[1] as f32) * CELL;
    (0..CRITTERS)
        .map(|_| Critter {
            position: Vec2::new(next(&mut seed), next(&mut seed)) * extent,
            velocity: Vec2::from_angle(next(&mut seed) * std::f32::consts::TAU)
                * (6.0 + next(&mut seed) * 14.0),
        })
        .collect()
}

/// The maps uploaded in setup: each layer once per filter, so the toggle
/// switches handles rather than re-uploading.
#[derive(Clone, Copy)]
struct Layers {
    atlas: TextureId,
    ground: [TilemapId; 2],
    hills: [TilemapId; 2],
}

fn sprites(atlas: TextureId, critters: &[Critter], pixel_art: bool) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = Some(atlas);
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_size = 8.0;
    item.default_pivot = [0.5, 0.0];
    item.filter = if pixel_art {
        TextureFilter::Nearest
    } else {
        TextureFilter::Linear
    };
    item.pixel_snap = pixel_art;
    // Above the signpost's world layer at order 0.
    item.order = SpriteOrder::Painter {
        layer: 1,
        depth_test: false,
    };
    // The critter's tile in the 4 x 2 atlas. A sprite quad maps `v = 0` to
    // its bottom edge and the atlas has row 0 at the top, so the rect runs
    // from the tile's bottom edge up to its top.
    let (col, row) = ((CRITTER % 4) as f32, (CRITTER / 4) as f32);
    let uv = [col / 4.0, (row + 1.0) / 2.0, (col + 1.0) / 4.0, row / 2.0];
    for c in critters {
        item.positions.push([c.position.x, c.position.y, 1.0]);
        item.uv_rects.push(uv);
        item.flips.push(if c.velocity.x < 0.0 {
            SpriteFlip::HORIZONTAL
        } else {
            SpriteFlip::NONE
        });
        item.sort_keys.push(-c.position.y);
    }
    item
}

/// A signpost and a glowing path on a world layer, in world units with
/// layer y running down the plane from the layer's origin.
fn signpost(time: f32) -> WorldOverlayLayer {
    let mut frame = OverlayFrame::default();
    // The post and the board.
    frame.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 1.0 },
            [29.0, 30.0],
            [6.0, 34.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.35, 0.22, 0.12, 1.0))),
    );
    frame.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 3.0 },
            [0.0, 0.0],
            [64.0, 34.0],
        )
        .with_fill(OverlayFill::linear_gradient(
            Colour::srgb(0.95, 0.82, 0.5, 1.0),
            Colour::srgb(0.62, 0.4, 0.2, 1.0),
            1.4,
        )),
    );
    let arrow = SubPath::polygon(&[
        [0.0, 4.0],
        [24.0, 4.0],
        [24.0, 0.0],
        [36.0, 8.0],
        [24.0, 16.0],
        [24.0, 12.0],
        [0.0, 12.0],
    ]);
    frame.shapes.push(
        OverlayShapeItem::vector(vec![arrow], FillRule::NonZero, [14.0, 14.0], [36.0, 16.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.45, 0.85, 1.0)))
            .with_stroke(OverlayStroke::new(1.0, Colour::srgb(0.05, 0.1, 0.25, 1.0))),
    );
    frame.labels.push(
        LabelItem::new("TO THE LAKE")
            .with_position([5.0, 2.5])
            .with_font_size(8.0)
            .with_colour(Colour::srgb(0.18, 0.1, 0.05, 1.0)),
    );
    // A glowing trail along the path, rippling with time: bright enough in
    // linear light for bloom, added over what is below.
    let points = (0..=64)
        .map(|i| {
            let x = i as f32 * 4.0;
            [40.0 + x, 60.0 + 6.0 * (x * 0.08 + time * 2.0).sin()]
        })
        .collect();
    frame.polylines.push(
        OverlayPolylineItem::new(points)
            .with_thickness(2.0)
            .with_cap(LineCap::Round)
            .with_colour(Colour::linear(4.0, 2.6, 0.8, 1.0))
            .with_blend(OverlayBlend::Additive),
    );
    WorldOverlayLayer::new(WorldPlane::Xy)
        .with_origin([240.0, 200.0])
        .with_elevation(0.5)
        .with_order(0)
        .with_frame(frame)
}

/// A damage number floating up from where it was dealt.
struct Damage {
    at: Vec2,
    age: f32,
    amount: u32,
}

/// How long a damage number floats, in seconds.
const DAMAGE_LIFE: f32 = 1.2;

/// A thin dark outline, in world units like the text it backs.
fn outline(spread: f32) -> ShadowLayer {
    ShadowLayer::new(Colour::srgb(0.05, 0.03, 0.02, 1.0), 0.0, [0.0, 0.0]).with_spread(spread)
}

/// The damage numbers as screen labels anchored in the world and sized in
/// world units: they rise ten units and fade over their life.
fn damage_labels(damage: &[Damage]) -> Vec<LabelItem> {
    damage
        .iter()
        .map(|d| {
            let t = d.age / DAMAGE_LIFE;
            LabelItem::new(d.amount.to_string())
                .with_world_anchor([d.at.x, d.at.y + 10.0 + 10.0 * t, 2.0])
                .with_align(Alignment::new(AnchorX::Middle, AnchorY::Baseline))
                .with_world_size(7.0, WorldTextSize::new(WorldPlane::Xy))
                .with_colour(Colour::srgb(1.0, 0.85, 0.25, 1.0))
                .with_shadow(outline(0.6))
                .with_opacity((1.0 - t * t).clamp(0.0, 1.0))
        })
        .collect()
}

/// A name tag over a critter: world-sized, but never smaller than 12 or
/// larger than 32 logical pixels.
fn name_tag(at: Vec2) -> LabelItem {
    LabelItem::new("Bramble")
        .with_world_anchor([at.x, at.y + 10.0, 2.0])
        .with_align(Alignment::new(AnchorX::Middle, AnchorY::Baseline))
        .with_world_size(
            4.0,
            WorldTextSize::new(WorldPlane::Xy)
                .with_min_px(12.0)
                .with_max_px(32.0),
        )
        .with_colour(Colour::srgb(0.85, 0.95, 1.0, 1.0))
        .with_shadow(outline(0.4))
}

fn main() {
    let layers: Rc<Cell<Option<Layers>>> = Rc::new(Cell::new(None));
    let setup_layers = layers.clone();

    let extent = Vec2::new(MAP[0] as f32, MAP[1] as f32) * CELL;
    // Three screen pixels per texel.
    let mut camera = Camera2d::new(Plane2d::Xy)
        .with_centre(Vec2::new(200.0, extent.y * 0.5))
        .with_units_per_pixel(1.0 / 3.0);
    camera.snap_to_pixels = true;
    let mut controller = Camera2dController::new_stateless();
    let mut critters = critters();
    let mut damage: Vec<Damage> = Vec::new();
    let mut damage_seed = 0x5EED_D00D_u32;
    let mut next_hit = 0.0_f32;
    let mut pixel_art = true;
    let mut extras: Vec<ExtraId> = Vec::new();

    ViewportApp::new(AppConfig::default())
        .window_with_input(
            WindowConfig::default()
                .with_title("viewport-lib : tilemap in 2D")
                .with_window_size(1280, 720),
            move |setup| {
                let device = setup.device().clone();
                let queue = setup.queue().clone();
                let declined =
                    viewport_lib_plugins::item_types::install(setup.renderer_mut(), &device);
                for (name, reason) in declined {
                    eprintln!("item type {name} unavailable: {reason}");
                }
                let atlas = setup
                    .resources_mut()
                    .upload_texture(&device, &queue, atlas())
                    .expect("atlas upload");
                let renderer = setup.renderer_mut();
                let mut upload = |map: Tilemap| renderer.upload(&device, &queue, &map).unwrap();
                let filters = [TextureFilter::Nearest, TextureFilter::Linear];
                let ground = filters.map(|f| upload(ground(tileset(atlas, f))));
                let hills = filters.map(|f| upload(hills(tileset(atlas, f))));
                setup_layers.set(Some(Layers {
                    atlas,
                    ground,
                    hills,
                }));
                let mut bindings = planar_bindings();
                bindings.push(ViewportBinding::new(
                    TOGGLE_PIXEL_ART,
                    ViewportGesture::KeyPress {
                        key: KeyCode::P,
                        modifiers: ModifiersMatch::Exact(Modifiers::NONE),
                    },
                ));
                setup.set_bindings(bindings);
            },
            |input| {
                for event in input.events().to_vec() {
                    input.forward(event);
                }
            },
            move |ctx| {
                let size = ctx.viewport_size();
                camera.set_viewport_size(Vec2::from(size));
                camera.pixels_per_point = ctx.surface_size()[0] as f32 / size[0].max(1.0);
                let dt = ctx.dt();
                let frame = ctx.action_frame().clone();
                if frame.is_active(TOGGLE_PIXEL_ART) {
                    pixel_art = !pixel_art;
                    camera.snap_to_pixels = pixel_art;
                }

                for c in &mut critters {
                    c.position += c.velocity * dt;
                    if c.position.x < 0.0 || c.position.x > extent.x {
                        c.velocity.x = -c.velocity.x;
                    }
                    if c.position.y < 0.0 || c.position.y > extent.y {
                        c.velocity.y = -c.velocity.y;
                    }
                    c.position = c.position.clamp(Vec2::ZERO, extent);
                }

                // A hit on a critter near the middle of the view every so
                // often, floating up from where it landed.
                next_hit -= dt;
                if next_hit <= 0.0 {
                    next_hit = 0.15 + next(&mut damage_seed) * 0.3;
                    let centre = camera.centre();
                    if let Some(c) = critters
                        .iter()
                        .filter(|c| (c.position - centre).length() < 120.0)
                        .nth((next(&mut damage_seed) * 8.0) as usize)
                    {
                        damage.push(Damage {
                            at: c.position,
                            age: 0.0,
                            amount: 5 + (next(&mut damage_seed) * 95.0) as u32,
                        });
                    }
                }
                for d in &mut damage {
                    d.age += dt;
                }
                damage.retain(|d| d.age < DAMAGE_LIFE);

                // The setup callback runs again after a lost device with new
                // handles, so the items are rebuilt each frame from them.
                for id in extras.drain(..) {
                    ctx.remove_extra(id);
                }
                if let Some(layers) = layers.get() {
                    let which = if pixel_art { 0 } else { 1 };
                    let mut far = TilemapItem::new(layers.hills[which]);
                    far.cell_size = [CELL * 2.0, CELL * 2.0];
                    far.origin = [-200.0, -40.0];
                    // A side view: the hills lag the camera across the screen
                    // but keep step with it vertically, so they never float
                    // above or sink under the ground when the view pans or
                    // zooms.
                    far.parallax = [1.0 / 3.0, 1.0];
                    far.elevation = -1.0;
                    far.pixel_snap = pixel_art;
                    let mut near = TilemapItem::new(layers.ground[which]);
                    near.cell_size = [CELL, CELL];
                    near.time = ctx.time() as f32;
                    near.pixel_snap = pixel_art;
                    extras.push(ctx.add_item(far));
                    extras.push(ctx.add_item(near));
                    extras.push(ctx.add_item(sprites(layers.atlas, &critters, pixel_art)));
                }
                let layer = signpost(ctx.time());
                let mut labels = damage_labels(&damage);
                labels.push(name_tag(critters[0].position));
                ctx.inject(move |frame| {
                    frame.scene.world_overlays.push(layer);
                    frame.overlays.labels.extend(labels);
                    frame.effects.post_process.bloom.enabled = true;
                });

                // Drift right, then pan and zoom on top.
                let centre = camera.centre() + Vec2::new(20.0 * dt, 0.0);
                camera.set_centre(Vec2::new(centre.x % extent.x, centre.y));
                controller.apply(&mut camera, &frame, dt);
                *ctx.camera_mut() = camera.camera();
            },
        )
        .run();
}
