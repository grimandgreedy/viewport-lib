//! The maps: the side view's ground and hills and the isometric sheet.

use super::{
    ATLAS_COLS, BUSH, CELL, COLS, DIRT, GRASS, GROUND_Z, HILL, HILL_TOP, ISO_COLS, ISO_DARK,
    ISO_LIGHT, ISO_ROWS, ISO_TILE, ISO_TOP_LEFT, ISO_TREE, ISO_TREES, PLANK, ROWS, STONE, TORCH,
    TORCHES,
};
use glam::Vec2;
use viewport_lib as vpl;
use viewport_lib_plugins::item_types::tilemap::{
    Tile, TileOrientation, TileSize, Tilemap, Tileset,
};
use vpl::{TextureFilter, TextureId};

pub(super) fn tileset(atlas: Option<TextureId>, filter: TextureFilter) -> Tileset {
    let mut set = Tileset::new(atlas, [8, 8], ATLAS_COLS as u32);
    set.filter = filter;
    set
}

/// The ground layer: dirt under a grass surface, low ruined walls, plank
/// platforms, torches and bushes, with tall walls at both ends.
pub(super) fn ground_map(set: Tileset) -> Tilemap {
    let mut map = Tilemap::new(set, [COLS, ROWS]);
    let ground_row = (GROUND_Z / CELL) as u32 - 1;
    for col in 0..COLS {
        for row in 0..ground_row {
            map.set(col as i32, row as i32, Tile::new(DIRT));
        }
        map.set(col as i32, ground_row as i32, Tile::new(GRASS));
    }
    let mut fill = |cols: std::ops::RangeInclusive<u32>, rows: std::ops::RangeInclusive<u32>, t| {
        for c in cols {
            for r in rows.clone() {
                map.set(c as i32, r as i32, Tile::new(t));
            }
        }
    };
    fill(0..=0, 5..=ROWS - 1, STONE);
    fill(COLS - 1..=COLS - 1, 5..=ROWS - 1, STONE);
    fill(18..=20, 5..=7, STONE);
    fill(26..=32, 9..=9, PLANK);
    fill(44..=47, 5..=8, STONE);
    fill(48..=54, 10..=10, PLANK);
    fill(64..=65, 5..=6, STONE);
    fill(80..=83, 5..=8, STONE);
    fill(81..=82, 9..=12, STONE);
    fill(92..=100, 9..=9, PLANK);
    fill(110..=112, 5..=7, STONE);
    fill(124..=127, 5..=8, STONE);
    fill(128..=136, 11..=11, PLANK);
    fill(146..=148, 5..=6, STONE);
    for (col, row) in TORCHES {
        map.set(col as i32, row as i32, Tile::new(TORCH));
    }
    for col in [5, 14, 34, 58, 70, 88, 104, 118, 140, 154] {
        map.set(col as i32, 5, Tile::new(BUSH));
    }
    map
}

/// The far layer: a band of rolling hills, in cells twice the ground's size.
pub(super) fn hills_map(set: Tileset) -> Tilemap {
    let (cols, rows) = (96u32, 10u32);
    let mut map = Tilemap::new(set, [cols, rows]);
    for col in 0..cols {
        let x = col as f32;
        let top = (4.5 + 1.6 * (x * 0.21).sin() + 1.0 * (x * 0.53 + 1.0).sin()) as u32;
        for row in 0..top.min(rows) {
            map.set(col as i32, row as i32, Tile::new(HILL));
        }
        if top < rows {
            map.set(col as i32, top as i32, Tile::new(HILL_TOP));
        }
    }
    map
}

/// The isometric sheet's grid: Tiled's isometric layout on 16 x 8 cells.
pub(super) fn iso_tileset(atlas: Option<TextureId>, filter: TextureFilter) -> Tileset {
    let mut set = Tileset::new(atlas, ISO_TILE, 4);
    set.filter = filter;
    set.orientation = TileOrientation::Isometric;
    set.tile_sizes.push(TileSize::new(ISO_TREE, [16, 16]));
    set
}

/// The sheet's ground: a checker of diamonds, Tiled's cell `(x, y)` stored
/// at `(x, -y)` so row 0 is the top of the sheet.
pub(super) fn iso_ground_map(set: Tileset) -> Tilemap {
    let mut map = Tilemap::sparse(set, 8);
    for y in 0..ISO_ROWS {
        for x in 0..ISO_COLS {
            let tile = if (x + y) % 2 == 0 {
                ISO_LIGHT
            } else {
                ISO_DARK
            };
            map.set(x, -y, Tile::new(tile));
        }
    }
    map
}

/// The sheet's trees, each a tall tile standing on its cell.
pub(super) fn iso_trees_map(set: Tileset) -> Tilemap {
    let mut map = Tilemap::sparse(set, 8);
    for (x, y) in ISO_TREES {
        map.set(x, -y, Tile::new(ISO_TREE));
    }
    map
}

/// The origin that puts the sheet's bounding box's top-left corner at
/// [`ISO_TOP_LEFT`], one world unit per texel.
pub(super) fn iso_origin(set: &Tileset) -> Vec2 {
    let (mut min_x, mut max_y) = (f64::MAX, f64::MIN);
    for y in 0..ISO_ROWS {
        for x in 0..ISO_COLS {
            let o = set.cell_origin(x, -y);
            min_x = min_x.min(o.x);
            max_y = max_y.max(o.y + 1.0);
        }
    }
    let [tw, th] = ISO_TILE.map(|s| s as f32);
    ISO_TOP_LEFT - Vec2::new(min_x as f32 * tw, max_y as f32 * th)
}

pub(super) fn solid(tile: Tile) -> bool {
    matches!(tile.index, GRASS | DIRT | STONE | PLANK)
}

/// Plane position of a torch's flame.
pub(super) fn torch_flame(col: u32, row: u32) -> Vec2 {
    Vec2::new(col as f32 * CELL + 4.0, row as f32 * CELL + 7.0)
}
