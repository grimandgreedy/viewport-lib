//! Sprite masks, draw modes and the vertex hook under a 2D camera on the XY
//! plane, one world unit per pixel of the 400 x 300 golden.
//!
//! - top left: a circle mask (slot 0, layers 1 and 2). An orange bar on layer
//!   1 shows only inside it, a blue bar on layer 2 only outside it, and a
//!   green square on layer 3, outside the mask's layers, is not cut although
//!   it tests the same slot.
//! - top right: a blended tile layer of stripes cut to a second circle mask
//!   (slot 1), drawn from a sprite's alpha.
//! - bottom left: a 32 x 32 nine-slice with 8 px borders drawn at 200 x 60
//!   under nearest filtering: the corners stay 8 px and the centre stretches.
//! - bottom right: a 16-unit tile repeated across 120 units, continuous (the
//!   last tile cut) above adaptive (eight whole tiles stretched), and below
//!   them a square whose vertex hook moves its top corners right.
//!
//! The compatibility tier draws the tile layer and its mask only: sprites are
//! a full-tier type.

use glam::Vec2;
use viewport_lib::{
    Camera2d, Colour, MaskTest, MaterialPlugin, OverlayShape, Plane2d, TextureFilter,
};
use viewport_lib_plugins::item_types::sprite::{
    MaskSource, SpriteDrawMode, SpriteEffects, SpriteItem, SpriteMaskItem, SpriteMaterial,
    SpriteMaterials, SpriteOrder, SpriteSizeMode, TileMode,
};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, Tilemap, TilemapDraw, TilemapItem, TilemapPainter, Tileset,
};

use super::sprites_2d::pixel_art;
use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, rigs};

/// The scene appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "sprites_2d_modes",
        cameras: vec![NamedCamera {
            name: "top",
            camera: Camera2d::new(Plane2d::Xy)
                .with_units_per_pixel(1.0)
                .with_viewport_size(Vec2::new(400.0, 300.0))
                .camera(),
        }],
        build: build_sprites_2d_modes,
    }]
}

/// A world-space sprite of one colour.
fn quad(at: [f32; 2], size: [f32; 2], colour: Colour) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = vec![[at[0], at[1], 0.0]];
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

/// Moves the top corners of the quad right by `shift`.
struct Lean;

impl MaterialPlugin for Lean {
    fn name(&self) -> &str {
        "golden_lean"
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

fn build_sprites_2d_modes(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut sprites = Vec::new();
    let mut masks = Vec::new();

    // --- top left: inside, outside and out of range ------------------------
    masks.push(SpriteMaskItem::new(
        MaskSource::Shape {
            shape: OverlayShape::Circle,
            sprite: quad([-110.0, 75.0], [90.0, 90.0], Colour::WHITE),
        },
        0,
        1..=2,
    ));
    sprites.push(painter(
        quad(
            [-110.0, 95.0],
            [150.0, 40.0],
            Colour::linear_rgb(0.95, 0.55, 0.1),
        ),
        1,
        MaskTest::VisibleInside(0),
    ));
    sprites.push(painter(
        quad(
            [-110.0, 55.0],
            [150.0, 40.0],
            Colour::linear_rgb(0.15, 0.35, 0.9),
        ),
        2,
        MaskTest::VisibleOutside(0),
    ));
    sprites.push(painter(
        quad(
            [-110.0, 75.0],
            [24.0, 24.0],
            Colour::linear_rgb(0.2, 0.85, 0.3),
        ),
        3,
        MaskTest::VisibleInside(0),
    ));

    // --- top right: a tile layer under an alpha mask ------------------------
    let disc = pixel_art(
        ctx,
        &[
            "....XXXX....", //
            "..XXXXXXXX..", //
            ".XXXXXXXXXX.", //
            ".XXXXXXXXXX.", //
            "XXXXXXXXXXXX", //
            "XXXXXXXXXXXX", //
            "XXXXXXXXXXXX", //
            "XXXXXXXXXXXX", //
            ".XXXXXXXXXX.", //
            ".XXXXXXXXXX.", //
            "..XXXXXXXX..", //
            "....XXXX....", //
        ],
        &[('X', [255, 255, 255, 255])],
    );
    let mut disc_sprite = quad([100.0, 75.0], [96.0, 96.0], Colour::WHITE);
    disc_sprite.texture_id = Some(disc);
    disc_sprite.filter = TextureFilter::Nearest;
    masks.push(SpriteMaskItem::new(
        MaskSource::Alpha {
            sprite: disc_sprite,
            threshold: 0.5,
        },
        1,
        0..=1,
    ));
    let stripes = pixel_art(
        ctx,
        &["YYYYKKKK"; 8],
        &[('Y', [240, 220, 60, 255]), ('K', [60, 40, 90, 255])],
    );
    let mut map = Tilemap::new(Tileset::new(Some(stripes), [8, 8], 1), [16, 14]);
    map.tileset.filter = TextureFilter::Nearest;
    for row in 0..14 {
        for col in 0..16 {
            map.set(col, row, Tile::new(0));
        }
    }
    let map_id =
        viewport_lib::plugin_api::Uploads::upload(ctx.renderer, ctx.device, ctx.queue, &map)
            .expect("tilemap upload");
    let mut tiles = TilemapItem::new(map_id);
    tiles.origin = [36.0, 19.0];
    tiles.cell_size = [8.0, 8.0];
    tiles.draw = TilemapDraw::Blended {
        layer: 1,
        depth_test: false,
        painter: TilemapPainter::default(),
    };
    tiles.mask = MaskTest::VisibleInside(1);

    // --- bottom left: nine-slice --------------------------------------------
    let frame = pixel_art(
        ctx,
        &[
            "RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR",
            "RWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRGGGGGGGGGGGGGGGGRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWRRRRRRRRRRRRRRRRRRRRRRRRRRRRWR",
            "RWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWR",
            "RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR",
        ],
        &[
            ('R', [200, 50, 60, 255]),
            ('W', [245, 240, 230, 255]),
            ('G', [70, 170, 110, 255]),
        ],
    );
    let mut sliced = quad([-90.0, -60.0], [200.0, 60.0], Colour::WHITE);
    sliced.texture_id = Some(frame);
    sliced.filter = TextureFilter::Nearest;
    sliced.draw_mode = SpriteDrawMode::sliced([8.0, 8.0, 8.0, 8.0]);
    sprites.push(sliced);

    // --- bottom right: tiling and the vertex hook ---------------------------
    let brick = pixel_art(
        ctx,
        &[
            "BBBBBBBBBBBBBBBB",
            "BOOOOOOOOOOOOOOB",
            "BOOOOOOOOOOOOOOB",
            "BOOOOOOOOOOOOOOB",
            "BOOOOOOOOOOOOOOB",
            "BOOOOOOOOOOOOOOB",
            "BOOOOOOOOOOOOOOB",
            "BBBBBBBBBBBBBBBB",
            "BBBBBBBBBBBBBBBB",
            "OOOOOOOBOOOOOOOO",
            "OOOOOOOBOOOOOOOO",
            "OOOOOOOBOOOOOOOO",
            "OOOOOOOBOOOOOOOO",
            "OOOOOOOBOOOOOOOO",
            "OOOOOOOBOOOOOOOO",
            "BBBBBBBBBBBBBBBB",
        ],
        &[('B', [60, 30, 30, 255]), ('O', [210, 110, 60, 255])],
    );
    for (y, mode) in [(-20.0, TileMode::Continuous), (-50.0, TileMode::Adaptive)] {
        let mut bar = quad([110.0, y], [120.0, 20.0], Colour::WHITE);
        bar.texture_id = Some(brick);
        bar.filter = TextureFilter::Nearest;
        bar.draw_mode = SpriteDrawMode::Tiled {
            tile_size: [16.0, 16.0],
            mode,
        };
        sprites.push(bar);
    }
    let lean = ctx.renderer.sprite_material_id("golden_lean").or_else(|| {
        ctx.renderer
            .register_sprite_material(ctx.device, &Lean)
            .ok()
    });
    if let Some(id) = lean {
        let mut leaning = quad(
            [90.0, -105.0],
            [40.0, 40.0],
            Colour::linear_rgb(0.6, 0.3, 0.85),
        );
        let mut fx = SpriteEffects::default();
        fx.material = Some(SpriteMaterial::new(id).with_param("shift", 30.0f32));
        leaning.effects = Some(fx);
        sprites.push(leaning);
    }

    BuiltScene {
        sprite_items: sprites,
        sprite_masks: masks,
        tilemaps: vec![tiles],
        lighting: rigs::from_above(),
        background: Some([0.05, 0.06, 0.08, 1.0]),
        ..Default::default()
    }
}
