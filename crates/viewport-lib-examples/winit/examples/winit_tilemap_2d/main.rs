//! A scrolling tilemap with two parallax layers under `Camera2d`, with pixel
//! art sprites walking over it.
//!
//! The level is a Tiled map read through viewport-lib-io: a far layer of hills
//! that scrolls at a third of the camera's speed behind a ground layer of
//! grass, paths and animated water, both cut from one tileset image. The map's
//! cells, flips, tile animation, layer offsets and parallax factors map onto
//! `Tilemap`, `Tileset` and `TilemapItem` here; three hundred critters walk
//! over the ground as painter-ordered sprites sorted by `-y`, cut from the
//! same tileset. A signpost of vector art and a glowing additive path sit on a
//! world overlay layer between the ground and the critters, so they scroll and
//! zoom with the map and take bloom. Damage numbers float up from the critters
//! and fade, and one critter carries a name tag: both are screen labels
//! anchored to world points and sized in world units, so they scale with the
//! zoom while facing the screen, and the name tag never shrinks below 12
//! pixels. The camera drifts to the right on its own.
//!
//! Drag to pan, scroll or pinch to zoom, `P` to switch between crisp pixel art
//! (nearest-sampled and snapped to the pixel grid) and smooth filtering.

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

use glam::Vec2;

use viewport_lib as vpl;
use viewport_lib::plugin_api::Uploads;
use viewport_lib_io::loaders::{png, tiled};
use viewport_lib_io::{
    PropertyValue, TextureSource, TileGrid, TileLayerData, TileLayerKind, TileMapData, TileSetData,
};
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

/// The Tiled map beside this file: its tileset image sits next to it.
const LEVEL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/winit_tilemap_2d/assets/level.tmx"
);
const CRITTERS: usize = 300;
/// The critter's tile in the tileset.
const CRITTER: u32 = 7;

/// The level as loaded: the map, its one tileset's image, and the two tile
/// layers by name.
struct Level {
    map: TileMapData,
    atlas: TextureData,
    /// World units per ground cell: one texel is one unit.
    cell: f32,
    /// Extent of the ground layer in world units.
    extent: Vec2,
}

impl Level {
    fn load() -> Self {
        let map = tiled::tilemap_from_path(Path::new(LEVEL)).expect("level");
        let set = map.tilesets.first().expect("the level has a tileset");
        let image = set.image.as_ref().expect("the tileset has an image");
        let atlas = match &image.source {
            TextureSource::File(path) => png::texture_from_path(path).expect("tileset image"),
            TextureSource::Decoded(data) => data.clone(),
        };
        let cell = map.tile_size[0] as f32;
        let ground = tile_layer(&map, "ground").1;
        let extent = Vec2::new(ground.size[0] as f32, ground.size[1] as f32) * cell;
        Self {
            map,
            atlas,
            cell,
            extent,
        }
    }

    fn tileset(&self) -> &TileSetData {
        &self.map.tilesets[0]
    }

    /// The critter's tile as a sprite UV rect.
    fn critter_uv(&self) -> [f32; 4] {
        let rect = self.tileset().tile_rect(CRITTER).expect("critter tile");
        let (w, h) = (self.atlas.width() as f32, self.atlas.height() as f32);
        [
            rect.x as f32 / w,
            rect.y as f32 / h,
            (rect.x + rect.width) as f32 / w,
            (rect.y + rect.height) as f32 / h,
        ]
    }
}

/// The tile layer called `name` and its grid.
fn tile_layer<'a>(map: &'a TileMapData, name: &str) -> (&'a TileLayerData, &'a TileGrid) {
    map.layers
        .iter()
        .find_map(|layer| match &layer.kind {
            TileLayerKind::Tiles(grid) if layer.name == name => Some((layer, grid)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the level has no tile layer called {name}"))
}

/// The renderer's tileset over the uploaded image: the grid as the file lays
/// it out, and every animated tile's frames with their durations in seconds.
fn tileset(set: &TileSetData, atlas: TextureId, filter: TextureFilter) -> Tileset {
    let mut out = Tileset::new(Some(atlas), set.tile_size, set.columns);
    out.spacing = set.spacing;
    out.margin = set.margin;
    out.filter = filter;
    for tile in set.tiles.iter().filter(|t| !t.animation.is_empty()) {
        out.animations.push(TileAnimation {
            tile: tile.index,
            frames: tile
                .animation
                .iter()
                .map(|f| TileFrame {
                    tile: f.index,
                    duration: f.duration_ms as f32 / 1000.0,
                })
                .collect(),
        });
    }
    out
}

/// A grid's cells as a `Tilemap`. The file lists rows from the top and the
/// map counts them from the bottom, so the rows turn over; the flips carry
/// across as they are.
fn tilemap(grid: &TileGrid, set: Tileset) -> Tilemap {
    let mut map = Tilemap::new(set, grid.size);
    let [cols, rows] = grid.size;
    for (i, cell) in grid.cells.iter().enumerate() {
        if cell.is_empty() || cell.tileset != 0 {
            continue;
        }
        let (col, row) = (i as u32 % cols, rows - 1 - i as u32 / cols);
        let flip = TileFlip::new(cell.flip_h, cell.flip_v, cell.flip_d);
        map.set(
            col as i32,
            row as i32,
            Tile::new(cell.index).with_flip(flip),
        );
    }
    map
}

/// A small deterministic generator, so the map is the same every run.
fn next(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed as f32 / u32::MAX as f32
}

struct Critter {
    position: Vec2,
    velocity: Vec2,
}

fn critters(extent: Vec2) -> Vec<Critter> {
    let mut seed = 0x2D5E_ED01;
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

fn sprites(
    atlas: TextureId,
    uv: [f32; 4],
    size: f32,
    critters: &[Critter],
    pixel_art: bool,
) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = Some(atlas);
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_size = size;
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

    let level = Rc::new(Level::load());
    let setup_level = level.clone();
    let extent = level.extent;
    // Three screen pixels per texel.
    let mut camera = Camera2d::new(Plane2d::Xy)
        .with_centre(Vec2::new(200.0, extent.y * 0.5))
        .with_units_per_pixel(1.0 / 3.0);
    camera.snap_to_pixels = true;
    let mut controller = Camera2dController::new_stateless();
    let mut critters = critters(extent);
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
                    .upload_texture(&device, &queue, setup_level.atlas.clone())
                    .expect("atlas upload");
                let renderer = setup.renderer_mut();
                let mut upload = |map: Tilemap| renderer.upload(&device, &queue, &map).unwrap();
                let filters = [TextureFilter::Nearest, TextureFilter::Linear];
                let set = setup_level.tileset();
                let ground = tile_layer(&setup_level.map, "ground").1;
                let hills = tile_layer(&setup_level.map, "hills").1;
                let ground = filters.map(|f| upload(tilemap(ground, tileset(set, atlas, f))));
                let hills = filters.map(|f| upload(tilemap(hills, tileset(set, atlas, f))));
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
                camera.pixels_per_point = ctx.pixels_per_point();
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
                    let cell = level.cell;
                    // The hills draw at the cell size their layer's `cell`
                    // property names, from the layer's offset (pixels, y down
                    // in the file; world units, y up here), and lag the
                    // camera across the screen by the layer's parallax factor
                    // while keeping step with it vertically, so they never
                    // float above or sink under the ground when the view pans
                    // or zooms.
                    let (far_layer, _) = tile_layer(&level.map, "hills");
                    let far_cell = match far_layer.properties.get("cell") {
                        Some(PropertyValue::Float(v)) => *v as f32,
                        _ => cell,
                    };
                    let mut far = TilemapItem::new(layers.hills[which]);
                    far.cell_size = [far_cell, far_cell];
                    far.origin = [far_layer.offset[0], -far_layer.offset[1]];
                    far.parallax = far_layer.parallax;
                    far.elevation = -1.0;
                    far.pixel_snap = pixel_art;
                    let mut near = TilemapItem::new(layers.ground[which]);
                    near.cell_size = [cell, cell];
                    near.time = ctx.time() as f32;
                    near.pixel_snap = pixel_art;
                    extras.push(ctx.add_item(far));
                    extras.push(ctx.add_item(near));
                    extras.push(ctx.add_item(sprites(
                        layers.atlas,
                        level.critter_uv(),
                        cell,
                        &critters,
                        pixel_art,
                    )));
                }
                // The signpost animates, so it is rebuilt in place each frame.
                let layer = signpost(ctx.time()).with_pixel_snap(pixel_art);
                let world = ctx.world_overlays_mut();
                world.clear();
                world.push(layer);
                let mut labels = damage_labels(&damage);
                labels.push(name_tag(critters[0].position));
                ctx.overlays_mut().labels.extend(labels);
                ctx.inject(move |frame| {
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
