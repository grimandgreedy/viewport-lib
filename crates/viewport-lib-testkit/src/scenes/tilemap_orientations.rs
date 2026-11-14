//! `tilemap_orientations`: one map in each of Tiled's grid orientations,
//! each written as Tiled stores it (rows of global tile ids, top row first,
//! flip and turn bits included) and converted the way a Tiled map maps
//! across, Tiled's cell `(x, y)` to cell `(x, -y)`.
//!
//! One world unit is one pixel of the 400 x 300 golden and one texel of the
//! tiles. Each tile's image is its cell's outline filled light or dark, with
//! a darker rim; an arrow tile carries a yellow half-arrow pointing to the
//! image's top, so its flips and turns read.
//!
//! - Top left, isometric 32 x 16: a checker of diamonds with one cell
//!   empty, and two trees two cells tall in a layer sorted by row, the lower
//!   one's canopy over the upper one's, with a red figure standing between
//!   their lines: over the upper canopy, under the lower one.
//! - Top right, staggered 32 x 16, odd rows shifted: a checker of diamonds
//!   filling a rectangle, with one arrow upright and one under the diagonal
//!   flip, which mirrors it across its diagonal as on a square grid.
//! - Bottom left, hexagonal 28 x 32 with sides of 16, odd rows shifted
//!   (pointy top): a row of six arrows turned by Tiled's hexagonal flags:
//!   none, 60 degrees (the diagonal bit), 120, 180, mirrored, and mirrored
//!   then turned 120.
//! - Bottom right, hexagonal 32 x 28 with sides of 16, even columns shifted
//!   (flat top): a checker with three arrows, upright, turned 60 and turned
//!   120.

use glam::{DVec2, Vec2};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{Camera2d, Colour, PickId, Plane2d, TextureData, TextureFilter};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteOrder, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{
    StaggerAxis, StaggerIndex, Tile, TileOrientation, TileSize, Tilemap, TilemapDraw, TilemapItem,
    TilemapPainter, Tileset,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, rigs};

/// The scene appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "tilemap_orientations",
        cameras: vec![NamedCamera {
            name: "top",
            camera: Camera2d::new(Plane2d::Xy)
                .with_units_per_pixel(1.0)
                .with_viewport_size(Vec2::new(400.0, 300.0))
                .camera(),
        }],
        build,
    }]
}

/// Pick ids of the four ground layers, in panel order.
pub const ORIENTATION_PICK_IDS: [u64; 4] = [3201, 3202, 3203, 3204];

// Global ids, the tileset's first id being 1.
const LIGHT: u32 = 1;
const DARK: u32 = 2;
const ARROW: u32 = 3;
const TREE: u32 = 4;
// Tiled's flag bits.
const FLIP_H: u32 = 1 << 31;
const FLIP_D: u32 = 1 << 29;
const TURN_120: u32 = 1 << 28;

/// A map from Tiled's rows of global ids, top row first, `0` for no tile:
/// Tiled's cell `(x, y)` becomes cell `(x, -y)`, flips and turns decoded.
fn from_tiled(tileset: Tileset, rows: &[Vec<u32>]) -> Tilemap {
    let mut map = Tilemap::sparse(tileset, 8);
    for (y, row) in rows.iter().enumerate() {
        for (x, &gid) in row.iter().enumerate() {
            if gid & 0x0fff_ffff == 0 {
                continue;
            }
            let mut tile = Tile::from_tiled_gid(gid);
            tile.index -= 1;
            map.set(x as i32, -(y as i32), tile);
        }
    }
    map
}

/// `cols` x `rows` of the checker, light where `x + y` is even.
fn checker(cols: usize, rows: usize) -> Vec<Vec<u32>> {
    (0..rows)
        .map(|y| {
            (0..cols)
                .map(|x| if (x + y) % 2 == 0 { LIGHT } else { DARK })
                .collect()
        })
        .collect()
}

fn mix(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    std::array::from_fn(|i| (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).round() as u8)
}

/// Distance from `p` to segment `a`..`b`.
fn to_segment(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let d = b - a;
    let t = ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0);
    (a + d * t - p).length()
}

fn in_triangle(p: DVec2, a: DVec2, b: DVec2, c: DVec2) -> bool {
    let s = [
        (b - a).perp_dot(p - a),
        (c - b).perp_dot(p - b),
        (a - c).perp_dot(p - c),
    ];
    s.iter().all(|&v| v >= 0.0) || s.iter().all(|&v| v <= 0.0)
}

/// The atlas for `set`'s grid: four slots to a row, two rows: light, dark
/// and arrow tiles in the cell's outline, and in slot 3 (running into slot
/// 7) a tree two cells tall standing on the cell's centre.
fn atlas(ctx: &mut BuildCtx<'_>, set: &Tileset, light: [u8; 4], dark: [u8; 4]) -> Tileset {
    let [tw, th] = set.tile_size.map(|s| s as usize);
    let (w, h) = (tw * 4, th * 2);
    let mut pixels = vec![0u8; w * h * 4];
    // The outline relative to its bounding box, in cells.
    let origin = set.cell_origin(0, 1);
    let outline: Vec<DVec2> = set
        .cell_outline(0, 1)
        .into_iter()
        .map(|p| p - origin)
        .collect();
    let texel = DVec2::new(tw as f64, th as f64);
    let n = outline.len();
    let shape = |p: DVec2| -> Option<f64> {
        // Inside the outline: the distance to its rim, in texels.
        let mut rim = f64::MAX;
        for i in 0..n {
            let (a, b) = (outline[i], outline[(i + 1) % n]);
            if (b - a).perp_dot(p - a) < 0.0 {
                return None;
            }
            rim = rim.min(to_segment(p * texel, a * texel, b * texel));
        }
        Some(rim)
    };
    let yellow = [250, 205, 40, 255];
    let arrow = |p: DVec2| {
        in_triangle(
            p,
            DVec2::new(0.5, 0.82),
            DVec2::new(0.26, 0.45),
            DVec2::new(0.5, 0.45),
        ) || (0.43..=0.5).contains(&p.x) && (0.2..=0.45).contains(&p.y)
    };
    let mut put = |x: usize, y: usize, rgba: [u8; 4]| {
        let at = (y * w + x) * 4;
        pixels[at..at + 4].copy_from_slice(&rgba);
    };
    for slot in 0..3 {
        for y in 0..th {
            for x in 0..tw {
                // Texel centre, y up from the bottom of the slot.
                let p = DVec2::new(
                    (x as f64 + 0.5) / tw as f64,
                    1.0 - (y as f64 + 0.5) / th as f64,
                );
                let Some(rim) = shape(p) else {
                    continue;
                };
                let base = if slot == 0 { light } else { dark };
                let mut rgba = if rim < 1.5 {
                    mix(base, [0, 0, 0, 255], 0.45)
                } else {
                    base
                };
                if slot == 2 && arrow(p) {
                    rgba = yellow;
                }
                put(slot * tw + x, y, rgba);
            }
        }
    }
    // The tree: a round canopy on a trunk whose foot is the cell's centre,
    // half a cell above the bottom of its two-cell image.
    for y in 0..2 * th {
        for x in 0..tw {
            let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let foot = (2 * th) as f32 - th as f32 * 0.5;
            let canopy = Vec2::new(tw as f32 * 0.5, foot - th as f32 * 0.95);
            let r = (p - canopy).length();
            let rgba = if r < th as f32 * 0.55 {
                if r < th as f32 * 0.35 {
                    [40, 105, 45, 255]
                } else {
                    [25, 70, 35, 255]
                }
            } else if (p.x - tw as f32 * 0.5).abs() < 2.0 && p.y < foot && p.y > canopy.y {
                [110, 70, 35, 255]
            } else {
                continue;
            };
            put(3 * tw + x, y, rgba);
        }
    }
    let texture = ctx
        .renderer
        .resources_mut()
        .upload_texture(
            ctx.device,
            ctx.queue,
            TextureData::srgb(w as u32, h as u32, pixels),
        )
        .expect("atlas upload");
    let mut out = set.clone();
    out.texture_id = Some(texture);
    out.columns = 4;
    out.filter = TextureFilter::Nearest;
    out
}

fn tileset(size: [u32; 2], orientation: TileOrientation) -> Tileset {
    let mut set = Tileset::new(None, size, 4);
    set.orientation = orientation;
    set
}

/// A cutout ground layer of `map` at `origin`, one unit per texel.
fn ground(ctx: &mut BuildCtx<'_>, map: &Tilemap, origin: [f32; 2], pick_id: u64) -> TilemapItem {
    let id = ctx
        .renderer
        .upload(ctx.device, ctx.queue, map)
        .expect("tilemap upload");
    let mut item = TilemapItem::new(id);
    item.origin = origin;
    item.cell_size = map.tileset.tile_size.map(|s| s as f32);
    item.settings.pick_id = PickId(pick_id);
    item
}

fn build(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut tilemaps = Vec::new();

    // Isometric, 6 x 6.
    let iso = atlas(
        ctx,
        &tileset([32, 16], TileOrientation::Isometric),
        [120, 185, 95, 255],
        [80, 145, 70, 255],
    );
    let mut rows = checker(6, 6);
    rows[4][1] = 0;
    let iso_origin = [-116.0, 107.0];
    tilemaps.push(ground(
        ctx,
        &from_tiled(iso.clone(), &rows),
        iso_origin,
        ORIENTATION_PICK_IDS[0],
    ));
    // Two trees in a layer sorted by row, at Tiled cells (3, 1) and (3, 2):
    // the second one line lower on the plane and half a cell to the left.
    let mut trees = iso.clone();
    trees.tile_sizes.push(TileSize::new(TREE - 1, [32, 32]));
    let mut tree_rows = vec![vec![0; 6]; 3];
    tree_rows[1][3] = TREE;
    tree_rows[2][3] = TREE;
    let tree_map = from_tiled(trees, &tree_rows);
    let id = ctx
        .renderer
        .upload(ctx.device, ctx.queue, &tree_map)
        .expect("tree upload");
    let mut tree_item = TilemapItem::new(id);
    tree_item.origin = iso_origin;
    tree_item.cell_size = [32.0, 16.0];
    tree_item.draw = TilemapDraw::Blended {
        layer: 1,
        depth_test: false,
        painter: TilemapPainter::PerRow { offset: 0.0 },
    };
    tilemaps.push(tree_item);
    // The upper tree's cell has its bottom edge at plane y 75 and the lower
    // one's at 67; the figure's feet are between them, keyed `0 - feet y`
    // like the rows, so it covers the upper canopy and the lower one covers
    // it.
    let mut figure = SpriteItem::default();
    figure.size_mode = SpriteSizeMode::WorldSpace;
    figure.default_pivot = [0.5, 0.0];
    figure.order = SpriteOrder::Painter {
        layer: 1,
        depth_test: false,
    };
    figure.positions.push([-76.0, 71.0, 0.0]);
    figure.sizes.push([10.0, 24.0]);
    figure.colours.push(Colour::srgb(0.9, 0.2, 0.15, 1.0));
    figure.sort_keys.push(-71.0);

    // Staggered, 5 x 10, odd rows shifted.
    let staggered = atlas(
        ctx,
        &tileset(
            [32, 16],
            TileOrientation::Staggered {
                axis: StaggerAxis::Y,
                index: StaggerIndex::Odd,
            },
        ),
        [215, 190, 140, 255],
        [170, 140, 95, 255],
    );
    let mut rows = checker(5, 10);
    rows[3][1] = ARROW;
    rows[3][3] = ARROW | FLIP_D;
    tilemaps.push(ground(
        ctx,
        &from_tiled(staggered, &rows),
        [12.0, 103.0],
        ORIENTATION_PICK_IDS[1],
    ));

    // Hexagonal, pointy top, 6 x 5, odd rows shifted.
    let pointy = atlas(
        ctx,
        &tileset(
            [28, 32],
            TileOrientation::Hexagonal {
                side_length: 16,
                axis: StaggerAxis::Y,
                index: StaggerIndex::Odd,
            },
        ),
        [130, 175, 225, 255],
        [90, 130, 190, 255],
    );
    let mut rows = checker(6, 5);
    rows[2] = vec![
        ARROW,
        ARROW | FLIP_D,
        ARROW | TURN_120,
        ARROW | FLIP_D | TURN_120,
        ARROW | FLIP_H,
        ARROW | FLIP_H | TURN_120,
    ];
    tilemaps.push(ground(
        ctx,
        &from_tiled(pointy, &rows),
        [-191.0, -43.0],
        ORIENTATION_PICK_IDS[2],
    ));

    // Hexagonal, flat top, 7 x 4, even columns shifted.
    let flat = atlas(
        ctx,
        &tileset(
            [32, 28],
            TileOrientation::Hexagonal {
                side_length: 16,
                axis: StaggerAxis::X,
                index: StaggerIndex::Even,
            },
        ),
        [205, 150, 200, 255],
        [160, 105, 160, 255],
    );
    let mut rows = checker(7, 4);
    rows[1][1] = ARROW;
    rows[1][3] = ARROW | FLIP_D;
    rows[1][5] = ARROW | TURN_120;
    tilemaps.push(ground(
        ctx,
        &from_tiled(flat, &rows),
        [12.0, -40.0],
        ORIENTATION_PICK_IDS[3],
    ));

    BuiltScene {
        tilemaps,
        sprite_items: vec![figure],
        lighting: rigs::from_above(),
        background: Some([0.05, 0.06, 0.08, 1.0]),
        ..Default::default()
    }
}
