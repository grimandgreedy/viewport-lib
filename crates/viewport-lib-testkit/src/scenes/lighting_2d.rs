//! 2D lighting on the XY plane under an orthographic `Camera2d`, one world
//! unit to one pixel of the 400 x 300 goldens.
//!
//! `lighting_2d`: a tilemap room, a stone floor with two brick walls, one
//! upright and one across, and two crate sprites. Three coloured point lights
//! (warm, green and blue) are projected onto the plane; the walls and both
//! crates cast hard shadows from each light, and the ambient light is 0.2, so
//! the corners no light reaches keep a fifth of their colour.
//!
//! `lighting_2d_kinds`: the item's own lights over a grey floor. A white cone
//! from the top left with a block in its wedge, a faint blue global light, a
//! warm directional light casting parallel shadows from two blocks, a green
//! textured ring at the top right and a magenta polygon light at the bottom
//! left. A warm point light lights layers 1 and 2 only: the white square on
//! layer 1 beside it is lit, the one on layer 3 is left at the ambient level.
//! On layer 3 too, a yellow square and a row of orange particles are unlit
//! and keep their colour. The global and directional lights light layers 0
//! to 2, so the range splits in three parts, each with a light layer of its
//! own at the same ambient, and each square is lit by its own part's layer
//! only.
//!
//! `lighting_2d_normals`: normals and soft shadows. In the top half (layer
//! 0), a white light at the left, 60 units above the plane, lights a row of
//! grey sprites: a sprite whose normals lean left towards the light, its
//! twin without a normal map (lit flat, so as bright as the light allows),
//! the leaning sprite flipped so its normals lean away and it is dark, and a
//! dome bright on its left side and dark on its right. Below them a row of
//! white tiles with dome normals, the fourth flipped across and the fifth
//! upside down, all bright on the left. In the bottom half (layer 1), three
//! cones shining down past three blocks cast shadows with softness 0, 4 and
//! 12 device pixels, the penumbra widening from left to right.

use glam::{DVec3, Vec2};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, Candela, Colour, LightSource, LightingSettings, PainterPlacement, Plane2d,
    SpriteBlend, TextureData, TextureFilter, TextureId,
};
use viewport_lib_plugins::item_types::gpu_particles::{
    GpuParticleSystemConfig, GpuParticleSystemItem, GpuParticleSystems, ParticleRender, SpawnShape,
    VelocityDist,
};
use viewport_lib_plugins::item_types::lighting_2d::{
    Falloff, Light2d, Lighting2dItem, Occluder2d, occluders_from_sprites, occluders_from_tilemap,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteOrder, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, TileFlip, Tilemap, TilemapDraw, TilemapItem, TilemapPainter, TilemapPlane, Tileset,
};

use super::sprites_2d::pixel_art;
use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene};

/// The 2D lighting scene appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    let top = || {
        vec![NamedCamera {
            name: "top",
            camera: Camera2d::new(Plane2d::Xy)
                .with_units_per_pixel(1.0)
                .with_viewport_size(Vec2::new(400.0, 300.0))
                .camera(),
        }]
    };
    vec![
        NamedScene {
            name: "lighting_2d",
            cameras: top(),
            build: build_lighting_2d,
        },
        NamedScene {
            name: "lighting_2d_kinds",
            cameras: top(),
            build: build_lighting_2d_kinds,
        },
        NamedScene {
            name: "lighting_2d_normals",
            cameras: top(),
            build: build_lighting_2d_normals,
        },
    ]
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

/// Painter squares on `layer`, `size` units across.
fn painter_squares(layer: u32, size: f32, at: &[[f32; 2]], colour: Colour) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = at.iter().map(|p| [p[0], p[1], 0.0]).collect();
    item.default_colour = colour;
    item.default_size = size;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.order = SpriteOrder::Painter {
        layer,
        depth_test: false,
    };
    item
}

/// A white ring on a clear 32 x 32 texture, for a textured light.
fn ring(ctx: &mut BuildCtx<'_>) -> TextureId {
    let n = 32u32;
    let pixels: Vec<u8> = (0..n * n)
        .flat_map(|i| {
            let (x, y) = ((i % n) as f32 + 0.5, (i / n) as f32 + 0.5);
            let r = Vec2::new(x, y).distance(Vec2::splat(n as f32 * 0.5)) / (n as f32 * 0.5);
            let a = if (0.55..0.9).contains(&r) { 255 } else { 0 };
            [255, 255, 255, a]
        })
        .collect();
    ctx.renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::srgb(n, n, pixels))
        .expect("ring upload")
}

fn build_lighting_2d_kinds(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let ring = ring(ctx);
    let floor = {
        let mut f = painter_squares(0, 1.0, &[[0.0, 0.0]], Colour::linear_rgb(0.6, 0.6, 0.6));
        f.sizes = vec![[400.0, 300.0]];
        f
    };
    let blocks = painter_squares(
        0,
        24.0,
        &[[-90.0, 50.0], [30.0, 0.0], [-20.0, 40.0]],
        Colour::linear_rgb(0.3, 0.25, 0.2),
    );
    let lit = painter_squares(1, 26.0, &[[100.0, -95.0]], Colour::WHITE);
    let above = painter_squares(3, 26.0, &[[140.0, -95.0]], Colour::WHITE);
    let mut emissive = painter_squares(
        3,
        26.0,
        &[[-10.0, -120.0]],
        Colour::linear_rgb(1.0, 0.85, 0.1),
    );
    emissive.unlit = true;

    let lower = 0..=2;
    let mut item = Lighting2dItem::default();
    item.ambient = Colour::linear_rgb(0.4, 0.4, 0.4);
    item.use_frame_lights = false;
    item.occluders = occluders_from_sprites(&blocks, TilemapPlane::Xy);
    item.occluders.push(Occluder2d::Segment {
        a: [60.0, 60.0],
        b: [90.0, 30.0],
    });
    item.lights = vec![
        Light2d::cone([-170.0, 130.0], 240.0, [1.0, -0.8], 0.2, 0.4)
            .with_colour(Colour::linear_rgb(1.0, 1.0, 0.95))
            .with_intensity(1.4),
        Light2d::global()
            .with_colour(Colour::linear_rgb(0.1, 0.12, 0.3))
            .with_layers(lower.clone()),
        Light2d::directional([0.8, -0.6])
            .with_colour(Colour::linear_rgb(0.45, 0.35, 0.2))
            .with_layers(lower),
        Light2d::textured(ring, [130.0, 80.0], [110.0, 110.0], 0.4)
            .with_colour(Colour::linear_rgb(0.3, 1.0, 0.4))
            .with_intensity(1.2),
        Light2d::polygon(
            [-120.0, -80.0],
            vec![[-45.0, -35.0], [45.0, -35.0], [0.0, 45.0]],
            18.0,
        )
        .with_colour(Colour::linear_rgb(1.0, 0.3, 0.8))
        .with_falloff(Falloff::Linear),
        Light2d::point([120.0, -95.0], 70.0)
            .with_colour(Colour::linear_rgb(1.0, 0.6, 0.2))
            .with_intensity(4.0)
            .with_falloff(Falloff::InverseSquare)
            .with_layers(1..=2),
    ];

    // A row of orange particles on layer 3, unlit.
    let mut config = GpuParticleSystemConfig::default();
    config.capacity = 64;
    config.render = ParticleRender::Sprite {
        texture_id: None,
        blend: SpriteBlend::AlphaBlend,
        size_mode: SpriteSizeMode::WorldSpace,
        depth_write: false,
        lit: false,
        lit_params: Default::default(),
        normal_texture_id: None,
    };
    let system = ctx
        .renderer
        .create_gpu_particle_system(ctx.device, ctx.queue, &config)
        .expect("gpu particles are registered");
    let mut particles = GpuParticleSystemItem::new(system, 0.25);
    particles.emitter.rate = 120.0;
    particles.emitter.lifetime = (20.0, 20.0);
    particles.emitter.size = 6.0;
    particles.emitter.colour = Colour::linear(1.0, 0.5, 0.1, 1.0);
    particles.emitter.spawn_shape = SpawnShape::Box {
        min: [-60.0, -140.0, 0.0],
        max: [60.0, -132.0, 0.0],
    };
    particles.emitter.initial_velocity = VelocityDist::Fixed([0.0, 0.0, 0.0]);
    particles.painter = Some(PainterPlacement::new(3, 0.0).with_unlit(true));

    BuiltScene {
        sprite_items: vec![floor, blocks, lit, above, emissive],
        gpu_particle_systems: vec![particles],
        lighting_2d: vec![item],
        background: Some([0.05, 0.05, 0.06, 1.0]),
        ..Default::default()
    }
}

/// A `size` x `size` tangent-space normal map, each texel's normal from
/// `normal` at its centre in `-1..1` across, `y` up the image.
fn normal_map(
    ctx: &mut BuildCtx<'_>,
    size: u32,
    normal: impl Fn(f32, f32) -> [f32; 3],
) -> TextureId {
    let pixels: Vec<u8> = (0..size * size)
        .flat_map(|i| {
            let x = ((i % size) as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let y = 1.0 - ((i / size) as f32 + 0.5) / size as f32 * 2.0;
            let n = glam::Vec3::from(normal(x, y)).normalize();
            let byte = |c: f32| ((c * 0.5 + 0.5) * 255.0).round() as u8;
            [byte(n.x), byte(n.y), byte(n.z), 255]
        })
        .collect();
    ctx.renderer
        .resources_mut()
        .upload_texture(
            ctx.device,
            ctx.queue,
            TextureData::normal_map(size, size, pixels),
        )
        .expect("normal map upload")
}

/// A hemisphere's normals over a disc, flat outside it.
fn dome(x: f32, y: f32) -> [f32; 3] {
    let r2 = x * x + y * y;
    if r2 >= 1.0 {
        [0.0, 0.0, 1.0]
    } else {
        [x, y, (1.0 - r2).sqrt()]
    }
}

fn build_lighting_2d_normals(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let lean = normal_map(ctx, 16, |_, _| [-0.6, 0.0, 0.8]);
    let round = normal_map(ctx, 16, dome);

    let floor = |layer: u32, y: f32| {
        let mut f = painter_squares(layer, 1.0, &[[0.0, y]], Colour::linear_rgb(0.5, 0.5, 0.5));
        f.sizes = vec![[400.0, 150.0]];
        f
    };
    let grey = Colour::linear_rgb(0.8, 0.8, 0.8);
    let mut leaning = painter_squares(0, 50.0, &[[-110.0, 100.0], [10.0, 100.0]], grey);
    leaning.normal_texture_id = Some(lean);
    // The second leaning sprite is flipped, so its normals lean right.
    leaning.uv_rects = vec![[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
    let flat = painter_squares(0, 50.0, &[[-50.0, 100.0]], grey);
    let mut domed = painter_squares(0, 50.0, &[[90.0, 100.0]], grey);
    domed.normal_texture_id = Some(round);

    // A row of six white tiles with dome normals, the fourth and fifth
    // flipped one way and the other.
    let mut set = Tileset::new(None, [16, 16], 1);
    set.normal_atlas = Some(round);
    let mut map = Tilemap::with_chunk_size(set, [6, 1], 8);
    for col in 0..6 {
        let tile = Tile::new(0).with_flip(TileFlip::new(col == 3, col == 4, false));
        map.set(col, 0, tile);
    }
    let id = ctx
        .renderer
        .upload(ctx.device, ctx.queue, &map)
        .expect("tilemap upload");
    let mut tiles = TilemapItem::new(id);
    tiles.origin = [-150.0, 22.0];
    tiles.cell_size = [40.0, 40.0];
    tiles.plane = TilemapPlane::Xy;
    tiles.draw = TilemapDraw::Blended {
        layer: 0,
        depth_test: false,
        painter: TilemapPainter::Layer { key: 1.0 },
    };

    let blocks = painter_squares(
        1,
        16.0,
        &[[-130.0, -50.0], [0.0, -50.0], [130.0, -50.0]],
        Colour::linear_rgb(0.3, 0.25, 0.2),
    );

    let mut item = Lighting2dItem::default();
    item.ambient = Colour::linear_rgb(0.15, 0.15, 0.15);
    item.use_frame_lights = false;
    item.normals = true;
    item.occluders = occluders_from_sprites(&blocks, TilemapPlane::Xy);
    item.lights = vec![
        Light2d::point([-260.0, 70.0], 520.0)
            .with_height(60.0)
            .with_falloff(Falloff::Linear)
            .with_layers(0..=0),
    ];
    for (x, softness) in [(-130.0, 0.0), (0.0, 4.0), (130.0, 12.0)] {
        item.lights.push(
            Light2d::cone([x, -10.0], 150.0, [0.0, -1.0], 0.3, 0.45)
                .with_colour(Colour::linear_rgb(1.0, 0.85, 0.6))
                .with_intensity(1.5)
                .with_shadow_softness(softness)
                .with_layers(1..=1),
        );
    }

    BuiltScene {
        tilemaps: vec![tiles],
        sprite_items: vec![
            floor(0, 75.0),
            floor(1, -75.0),
            leaning,
            flat,
            domed,
            blocks,
        ],
        lighting_2d: vec![item],
        background: Some([0.05, 0.05, 0.06, 1.0]),
        ..Default::default()
    }
}
