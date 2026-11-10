//! Sprite effects under an orthographic `Camera2d` on the XY plane, one world
//! unit to one pixel of the 400 x 300 golden. Rows from the top:
//!
//! - flash: one character at flash 0, 0.5 and 1 in one batch.
//! - dissolve and palette: the character dissolving at thresholds 0.35 and
//!   0.7 against a noise texture with an orange rim, then one index-mapped
//!   sprite drawn with two rows of a four-colour palette.
//! - outline: the character outlined at 1 and 3 device pixels, beside a plain
//!   copy for comparison.

use glam::Vec2;
use viewport_lib::{Camera2d, Colour, Plane2d, TextureData, TextureFilter, TextureId};
use viewport_lib_plugins::item_types::sprite::{
    PaletteIndex, SpriteEffects, SpriteItem, SpriteOutline, SpritePalette, SpriteSizeMode,
};

use super::sprites_2d::pixel_art;
use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, rigs};

/// The sprite effect scenes appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![NamedScene {
        name: "sprites_effects",
        cameras: vec![NamedCamera {
            name: "top",
            camera: Camera2d::new(Plane2d::Xy)
                .with_units_per_pixel(1.0)
                .with_viewport_size(Vec2::new(400.0, 300.0))
                .camera(),
        }],
        build: build_sprites_effects,
    }]
}

/// A 12 x 12 character with a transparent surround.
fn character(ctx: &mut BuildCtx<'_>) -> TextureId {
    pixel_art(
        ctx,
        &[
            "....KKKK....", //
            "...KRRRRK...", //
            "..KRRRRRRK..", //
            "..KSSWSSWK..", //
            "..KSSKSSKK..", //
            "..KSSSSSSK..", //
            "...KSSSSK...", //
            "..KBBBBBBK..", //
            ".KSBBBBBBSK.", //
            ".KSKBBBBKSK.", //
            "...KBK.KBK..", //
            "...KKK.KKK..", //
        ],
        &[
            ('K', [25, 20, 30, 255]),
            ('R', [200, 60, 50, 255]),
            ('S', [240, 190, 150, 255]),
            ('W', [255, 255, 255, 255]),
            ('B', [60, 90, 200, 255]),
        ],
    )
}

/// Deterministic smooth value noise, uploaded linear.
fn noise(ctx: &mut BuildCtx<'_>) -> TextureId {
    let n = 64u32;
    let hash = |x: i32, y: i32| -> f32 {
        let mut h = (x as u32).wrapping_mul(374_761_393) ^ (y as u32).wrapping_mul(668_265_263);
        h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
        ((h ^ (h >> 16)) & 0xffff) as f32 / 65535.0
    };
    let cell = 8.0;
    let pixels: Vec<u8> = (0..n * n)
        .flat_map(|i| {
            let (x, y) = ((i % n) as f32 / cell, (i / n) as f32 / cell);
            let (x0, y0) = (x.floor() as i32, y.floor() as i32);
            let (fx, fy) = (x - x.floor(), y - y.floor());
            let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
            let a = hash(x0, y0) + (hash(x0 + 1, y0) - hash(x0, y0)) * sx;
            let b = hash(x0, y0 + 1) + (hash(x0 + 1, y0 + 1) - hash(x0, y0 + 1)) * sx;
            let v = ((a + (b - a) * sy) * 255.0).round() as u8;
            [v, v, v, 255]
        })
        .collect();
    ctx.renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::linear(n, n, pixels))
        .expect("noise upload")
}

/// An 8 x 8 index map for a four-colour palette (red channel `i / 3`),
/// uploaded linear, and a 4 x 2 palette with two rows.
fn palette_pair(ctx: &mut BuildCtx<'_>) -> (TextureId, TextureId) {
    let rows = [
        "..3333..", //
        ".322223.", //
        "32211223", //
        "32100123", //
        "32100123", //
        "32211223", //
        ".322223.", //
        "..3333..", //
    ];
    let index: Vec<u8> = rows
        .iter()
        .rev()
        .flat_map(|row| {
            row.chars().flat_map(|c| match c.to_digit(10) {
                Some(i) => [(i * 255 / 3) as u8, 0, 0, 255],
                None => [0, 0, 0, 0],
            })
        })
        .collect();
    let index = ctx
        .renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::linear(8, 8, index))
        .expect("index map upload");
    // Row 0 (bottom of the texture) is a warm ramp, row 1 a cool one.
    let palette: Vec<u8> = [
        [255, 245, 200, 255],
        [250, 190, 60, 255],
        [220, 90, 30, 255],
        [90, 25, 20, 255],
        [210, 250, 255, 255],
        [90, 200, 240, 255],
        [40, 90, 200, 255],
        [15, 20, 70, 255],
    ]
    .concat();
    let palette = ctx
        .renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::srgb(4, 2, palette))
        .expect("palette upload");
    (index, palette)
}

/// World-space, nearest-sampled, pixel-snapped sprites with effects.
fn batch(texture: TextureId, positions: Vec<[f32; 3]>, size: f32, fx: SpriteEffects) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = Some(texture);
    item.positions = positions;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_size = size;
    item.filter = TextureFilter::Nearest;
    item.pixel_snap = true;
    item.effects = Some(fx);
    item
}

fn build_sprites_effects(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let hero = character(ctx);
    let noise = noise(ctx);
    let (index, palette) = palette_pair(ctx);
    let mut sprites = Vec::new();

    // Flash at 0, 0.5 and 1, per sprite in one batch.
    let mut flash = SpriteEffects::default();
    flash.flash = vec![0.0, 0.5, 1.0];
    sprites.push(batch(
        hero,
        vec![[-120.0, 95.0, 0.0], [0.0, 95.0, 0.0], [120.0, 95.0, 0.0]],
        72.0,
        flash,
    ));

    // Dissolve at two thresholds, with a rim.
    let mut dissolve = SpriteEffects::default();
    dissolve.dissolve = vec![0.35, 0.7];
    dissolve.dissolve_noise = Some(noise);
    dissolve.dissolve_edge_width = 0.08;
    sprites.push(batch(
        hero,
        vec![[-150.0, 0.0, 0.0], [-70.0, 0.0, 0.0]],
        72.0,
        dissolve,
    ));

    // One index-mapped sprite through two palette rows.
    let mut swap = SpriteEffects::default();
    swap.palette = Some(SpritePalette {
        texture: palette,
        index: PaletteIndex::Red,
    });
    swap.palette_rows = vec![0, 1];
    sprites.push(batch(
        index,
        vec![[60.0, 0.0, 0.0], [140.0, 0.0, 0.0]],
        64.0,
        swap,
    ));

    // A plain copy, then outlines at 1 and 3 device pixels.
    let mut plain = batch(
        hero,
        vec![[-120.0, -95.0, 0.0]],
        72.0,
        SpriteEffects::default(),
    );
    plain.effects = None;
    sprites.push(plain);
    for (x, width) in [(0.0, 1.0), (120.0, 3.0)] {
        let mut fx = SpriteEffects::default();
        fx.outline = Some(SpriteOutline::new(
            Colour::linear_rgb(1.0, 0.85, 0.1),
            width,
        ));
        sprites.push(batch(hero, vec![[x, -95.0, 0.0]], 72.0, fx));
    }

    BuiltScene {
        sprite_items: sprites,
        lighting: rigs::from_above(),
        background: Some([0.18, 0.2, 0.24, 1.0]),
        ..Default::default()
    }
}
