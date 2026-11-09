//! Tilemap layers on the XY plane under an orthographic `Camera2d`.
//!
//! One world unit is one pixel of the 400 x 300 golden, and each tile is
//! 8 x 8 texels drawn at 16 x 16 units. The camera sits at plane (40, 20), off
//! the origin, so the parallax shift shows:
//!
//! - a background layer at half parallax (shifted by half the camera's
//!   position), lower than the rest, showing through holes in the ground;
//! - the ground layer (parallax 1), with a row of one arrow tile in Tiled's
//!   eight orientations, three tinted tiles, and an animated water tile held
//!   at a fixed clock on its third frame;
//! - a blended decoration layer of soft-edged flowers, drawn in painter's
//!   order over the ground, two of them over the water and an arrow.

use glam::Vec2;
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{Camera2d, Colour, PickId, Plane2d, TextureData, TextureFilter, TextureId};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, TileAnimation, TileFlip, TileFrame, Tilemap, TilemapDraw, TilemapItem, Tileset,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, rigs};

/// The tilemap scene appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "tilemap",
        cameras: vec![NamedCamera {
            name: "top",
            camera: Camera2d::new(Plane2d::Xy)
                .with_centre(Vec2::new(40.0, 20.0))
                .with_units_per_pixel(1.0)
                .with_viewport_size(Vec2::new(400.0, 300.0))
                .camera(),
        }],
        build: build_tilemap,
    }]
}

// Tile indices in the atlas, four to a row.
const GRASS: u32 = 0;
const DIRT: u32 = 1;
const ARROW: u32 = 2;
const WATER: u32 = 3;
const HILL: u32 = 4;
const FLOWER: u32 = 5;
const SKY: u32 = 6;
const WATER_B: u32 = 7;
const WATER_C: u32 = 8;

/// The 8 x 8 art of each tile, top row first, as palette characters.
const ART: [[&str; 8]; 9] = [
    // grass
    [
        "GGGGGGGG", "GgGGGGgG", "GGGGGGGG", "GGGgGGGG", "GGGGGGGG", "GgGGGgGG", "GGGGGGGG",
        "GGGGGGGG",
    ],
    // dirt
    [
        "DDDDDDDD", "DdDDDDDD", "DDDDDdDD", "DDDDDDDD", "DDdDDDDD", "DDDDDDdD", "DDDDDDDD",
        "DdDDDDDD",
    ],
    // arrow: points right, yellow top half, blue bottom, so every one of the
    // eight orientations reads differently.
    [
        "Y.......", "YY..W...", "YYYYWW..", "YYYYYYW.", "BBBBBBBW", "BBBBBBB.", "BBBB....",
        "B.......",
    ],
    // water, frame A
    [
        "wwwwwwww", "wWwwwwww", "wwwwwwww", "wwwwWwww", "wwwwwwww", "wwwwwwwW", "wwwwwwww",
        "wwwwwwww",
    ],
    // hill (background)
    [
        "........", "...hh...", "..hhhh..", ".hhhhhh.", "hhhhhhhh", "hhhhhhhh", "hhhhhhhh",
        "hhhhhhhh",
    ],
    // flower (blended: soft petals)
    [
        "........", "..pPp...", ".pPYPp..", "..pPp...", "...l....", "...l.l..", "...ll...",
        "...l....",
    ],
    // sky (background)
    [
        "ssssssss", "ssssssss", "ssSSssss", "sSSSSsss", "ssssssss", "ssssssss", "ssssSSss",
        "sssSSSSs",
    ],
    // water, frame B
    [
        "wwwwwwww", "wwwwwwww", "wwWwwwww", "wwwwwwww", "wwwwwwWw", "wwwwwwww", "Wwwwwwww",
        "wwwwwwww",
    ],
    // water, frame C: a bright crest, the frame the fixed clock selects
    [
        "wwwwwwww", "wwwwwwww", "WWWWWWWW", "wwwwwwww", "wwwwwwww", "WWWWWWWW", "wwwwwwww",
        "wwwwwwww",
    ],
];

fn palette(c: char) -> [u8; 4] {
    match c {
        'G' => [70, 160, 70, 255],
        'g' => [110, 200, 90, 255],
        'D' => [130, 90, 55, 255],
        'd' => [100, 65, 40, 255],
        'Y' => [250, 200, 40, 255],
        'B' => [40, 110, 220, 255],
        'W' => [235, 245, 255, 255],
        'w' => [40, 90, 190, 255],
        'h' => [60, 90, 110, 255],
        'p' => [240, 120, 180, 120],
        'P' => [250, 90, 160, 220],
        'l' => [60, 150, 60, 200],
        's' => [110, 150, 210, 255],
        'S' => [200, 215, 235, 255],
        _ => [0, 0, 0, 0],
    }
}

/// The atlas: the tiles above in rows of four, row 0 of the image at its top.
fn atlas(ctx: &mut BuildCtx<'_>) -> TextureId {
    let (columns, rows) = (4usize, ART.len().div_ceil(4));
    let (w, h) = (columns * 8, rows * 8);
    let mut pixels = vec![0u8; w * h * 4];
    for (i, art) in ART.iter().enumerate() {
        let (tx, ty) = ((i % columns) * 8, (i / columns) * 8);
        for (y, line) in art.iter().enumerate() {
            for (x, c) in line.chars().enumerate() {
                let at = ((ty + y) * w + tx + x) * 4;
                pixels[at..at + 4].copy_from_slice(&palette(c));
            }
        }
    }
    ctx.renderer
        .resources_mut()
        .upload_texture(
            ctx.device,
            ctx.queue,
            TextureData::srgb(w as u32, h as u32, pixels),
        )
        .expect("atlas upload")
}

fn tileset(atlas: TextureId) -> Tileset {
    let mut set = Tileset::new(Some(atlas), [8, 8], 4);
    set.filter = TextureFilter::Nearest;
    let frame = |tile| TileFrame {
        tile,
        duration: 0.25,
    };
    set.animations.push(TileAnimation {
        tile: WATER,
        frames: vec![frame(WATER), frame(WATER_B), frame(WATER_C)],
    });
    set
}

fn build_tilemap(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let atlas = atlas(ctx);
    let device = ctx.device;
    let queue = ctx.queue;

    // Background: sky over a band of hills, larger than the view so the
    // half-parallax shift never uncovers an edge. Chunks of 8.
    let mut background = Tilemap::with_chunk_size(tileset(atlas), [32, 24], 8);
    for row in 0..24 {
        for col in 0..32 {
            let index = if row < 10 { HILL } else { SKY };
            background.set(col, row, Tile::new(index));
        }
    }
    let background = ctx
        .renderer
        .upload(device, queue, &background)
        .expect("background upload");

    // Ground: 22 x 14 cells of 16 units, its bottom-left at plane (-135, -95).
    let mut ground = Tilemap::with_chunk_size(tileset(atlas), [22, 14], 8);
    for row in 0..14u32 {
        for col in 0..22u32 {
            // Holes in a diagonal band, where the background shows.
            if (col + row) % 9 == 0 || (col + 2 * row) % 13 == 5 {
                continue;
            }
            let index = if row < 3 { DIRT } else { GRASS };
            ground.set(col, row, Tile::new(index));
        }
    }
    // The arrow in Tiled's eight orientations along row 8.
    let flips = [
        TileFlip::NONE,
        TileFlip::HORIZONTAL,
        TileFlip::VERTICAL,
        TileFlip::ROTATE_180,
        TileFlip::DIAGONAL,
        TileFlip::ROTATE_90,
        TileFlip::ROTATE_270,
        TileFlip::ANTI_DIAGONAL,
    ];
    for (i, flip) in flips.into_iter().enumerate() {
        ground.set(4 + 2 * i as u32, 8, Tile::new(ARROW).with_flip(flip));
    }
    // A pond of the animated water tile.
    for row in 4..7 {
        for col in 14..19 {
            ground.set(col, row, Tile::new(WATER));
        }
    }
    // Three tinted tiles.
    for (i, tint) in [
        Colour::linear_rgb(1.0, 0.4, 0.4),
        Colour::linear_rgb(0.5, 0.5, 1.0),
        Colour::linear_rgb(1.0, 1.0, 0.3),
    ]
    .into_iter()
    .enumerate()
    {
        ground.set(3 + i as u32, 5, Tile::new(DIRT).with_tint(tint));
    }
    let ground = ctx
        .renderer
        .upload(device, queue, &ground)
        .expect("ground upload");

    // Decoration: soft flowers scattered over the ground's grass.
    let mut flowers = Tilemap::with_chunk_size(tileset(atlas), [22, 14], 8);
    for (col, row) in [
        (2, 10),
        (5, 11),
        (9, 4),
        (11, 11),
        (12, 3),
        (20, 9),
        (7, 6),
        (15, 5),
        (4, 8),
    ] {
        flowers.set(col, row, Tile::new(FLOWER));
    }
    let flowers = ctx
        .renderer
        .upload(device, queue, &flowers)
        .expect("flowers upload");

    let cell = [16.0, 16.0];
    let mut back = TilemapItem::new(background);
    back.origin = [-280.0, -200.0];
    back.cell_size = [24.0, 24.0];
    back.parallax = [0.5, 0.5];
    back.elevation = -2.0;

    let mut ground_item = TilemapItem::new(ground);
    ground_item.origin = [-135.0, -95.0];
    ground_item.cell_size = cell;
    // On the third water frame: 0.25 s per frame.
    ground_item.time = 0.6;
    ground_item.settings.pick_id = PickId(3001);

    let mut flower_item = TilemapItem::new(flowers);
    flower_item.origin = [-135.0, -95.0];
    flower_item.cell_size = cell;
    flower_item.elevation = 1.0;
    flower_item.draw = TilemapDraw::Blended {
        layer: 0,
        depth_test: true,
    };

    BuiltScene {
        tilemaps: vec![back, ground_item, flower_item],
        lighting: rigs::from_above(),
        background: Some([0.05, 0.06, 0.08, 1.0]),
        ..Default::default()
    }
}
