//! 2D lighting over a tilemap room on the XY plane under an orthographic
//! `Camera2d`, one world unit to one pixel of the 400 x 300 golden.
//!
//! A stone floor with two brick walls, one upright and one across, and two
//! crate sprites. Three coloured point lights (warm, green and blue) are
//! projected onto the plane; the walls and both crates cast hard shadows from
//! each light, and the ambient light is 0.2, so the corners no light reaches
//! keep a fifth of their colour.

use glam::{DVec3, Vec2};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, Candela, Colour, LightSource, LightingSettings, Plane2d, TextureData, TextureFilter,
    TextureId,
};
use viewport_lib_plugins::item_types::lighting_2d::{
    Lighting2dItem, occluders_from_sprites, occluders_from_tilemap,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, Tilemap, TilemapItem, TilemapPlane, Tileset,
};

use super::sprites_2d::pixel_art;
use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene};

/// The 2D lighting scene appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "lighting_2d",
        cameras: vec![NamedCamera {
            name: "top",
            camera: Camera2d::new(Plane2d::Xy)
                .with_units_per_pixel(1.0)
                .with_viewport_size(Vec2::new(400.0, 300.0))
                .camera(),
        }],
        build: build_lighting_2d,
    }]
}

const FLOOR: u32 = 0;
const WALL: u32 = 1;

/// A two-tile atlas: stone floor and brick wall, 8 x 8 texels each.
fn atlas(ctx: &mut BuildCtx<'_>) -> TextureId {
    let floor = [
        "SSSSSSSs", "SsSSSSSs", "SSSSSSSs", "SSSSsSSs", "SSSSSSSs", "SSsSSSSs", "SSSSSSSs",
        "ssssssss",
    ];
    let wall = [
        "BBBmBBBB", "BBBmBBBB", "BBBmBBBB", "mmmmmmmm", "BBBBBBBm", "BBBBBBBm", "BBBBBBBm",
        "mmmmmmmm",
    ];
    let colour = |c: char| match c {
        'S' => [200, 195, 185, 255],
        's' => [150, 145, 140, 255],
        'B' => [170, 80, 60, 255],
        'm' => [90, 80, 75, 255],
        _ => [0, 0, 0, 0],
    };
    let mut pixels = vec![0u8; 16 * 8 * 4];
    for (t, art) in [floor, wall].iter().enumerate() {
        for (y, line) in art.iter().enumerate() {
            for (x, c) in line.chars().enumerate() {
                let at = (y * 16 + t * 8 + x) * 4;
                pixels[at..at + 4].copy_from_slice(&colour(c));
            }
        }
    }
    ctx.renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::srgb(16, 8, pixels))
        .expect("atlas upload")
}

fn point(x: f64, y: f64, colour: Colour, range: f32) -> LightSource {
    let mut light = LightSource::point_candela(DVec3::new(x, y, 10.0), Candela(1.2), range, 0.0);
    light.colour = colour;
    light
}

fn build_lighting_2d(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let atlas = atlas(ctx);
    let mut set = Tileset::new(Some(atlas), [8, 8], 2);
    set.filter = TextureFilter::Nearest;

    // 26 x 20 cells of 16 units, covering the view.
    let (cols, rows) = (26u32, 20u32);
    let origin = [-208.0, -160.0];
    let cell = [16.0, 16.0];
    let mut map = Tilemap::with_chunk_size(set, [cols, rows], 8);
    for row in 0..rows {
        for col in 0..cols {
            let upright = col == 10 && (5..14).contains(&row);
            let across = row == 6 && (15..23).contains(&col);
            map.set(
                col as i32,
                row as i32,
                Tile::new(if upright || across { WALL } else { FLOOR }),
            );
        }
    }
    let mut occluders = occluders_from_tilemap(&map, origin, cell, |t| t.index == WALL);
    let id = ctx
        .renderer
        .upload(ctx.device, ctx.queue, &map)
        .expect("tilemap upload");
    let mut layer = TilemapItem::new(id);
    layer.origin = origin;
    layer.cell_size = cell;
    layer.plane = TilemapPlane::Xy;

    // Two crates, drawn and casting.
    let crate_art = pixel_art(
        ctx,
        &[
            "KKKKKKKK", //
            "KWWWWWWK", //
            "KWKWWKWK", //
            "KWWKKWWK", //
            "KWWKKWWK", //
            "KWKWWKWK", //
            "KWWWWWWK", //
            "KKKKKKKK", //
        ],
        &[('K', [70, 45, 25, 255]), ('W', [190, 140, 80, 255])],
    );
    let mut crates = SpriteItem::default();
    crates.texture_id = Some(crate_art);
    crates.positions = vec![[-60.0, 60.0, 1.0], [70.0, 20.0, 1.0]];
    crates.size_mode = SpriteSizeMode::WorldSpace;
    crates.default_size = 32.0;
    crates.filter = TextureFilter::Nearest;
    crates.depth_write = true;
    occluders.extend(occluders_from_sprites(&crates, TilemapPlane::Xy));

    let mut lighting_2d = Lighting2dItem::default();
    lighting_2d.ambient = Colour::linear_rgb(0.2, 0.2, 0.2);
    lighting_2d.occluders = occluders;

    let mut lighting = LightingSettings::default();
    lighting.lights = vec![
        point(-130.0, 90.0, Colour::linear_rgb(1.0, 0.7, 0.35), 220.0),
        point(10.0, -40.0, Colour::linear_rgb(0.35, 1.0, 0.45), 170.0),
        point(150.0, 90.0, Colour::linear_rgb(0.4, 0.55, 1.0), 200.0),
    ];

    BuiltScene {
        tilemaps: vec![layer],
        sprite_items: vec![crates],
        lighting_2d: vec![lighting_2d],
        lighting,
        background: Some([0.05, 0.05, 0.06, 1.0]),
        ..Default::default()
    }
}
