//! The painter sequence across item types under an orthographic `Camera2d` on
//! the XY plane, framing 240 x 160 world units around the origin.
//!
//! - `painter_composed_keys`: two rows of overlapping squares. Top row: a
//!   sprite at sort value 4, then an entity at sort value 5 made of a sprite,
//!   a flat mesh surface and a second sprite at member indices 0, 1 and 2,
//!   then a sprite at sort value 6, each square covering part of the one
//!   before. The sprites are submitted out of order in two items, so the
//!   stacking comes from the composed keys alone. Bottom row: a sprite, an
//!   instanced mesh batch of two squares and a second sprite, members 0, 1
//!   and 2 of one sort value.
//! - `painter_particles_layers`: a blended tile layer on painter layer 1, GPU
//!   particles on layer 2 over it, and sprite bars on layer 3 over both.
//! - `painter_over_transparent`: a painter sprite across two translucent
//!   cubes, one in front of the plane and one behind it. Painter content draws
//!   after the transparent surfaces resolve, so the sprite covers both.

use glam::{Mat4, Vec2, Vec3};
use viewport_lib::plugin_api::Uploads;
use viewport_lib::{
    Camera2d, Colour, Material, MeshInstanceItem, PickId, Plane2d, SceneRenderItem, SpriteBlend,
    TextureData, painter_key, primitives,
};
use viewport_lib_plugins::item_types::gpu_particles::{
    GpuParticleSystemConfig, GpuParticleSystemItem, GpuParticleSystems, ParticleRender, SpawnShape,
    VelocityDist,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteOrder, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, Tilemap, TilemapDraw, TilemapItem, TilemapPainter, Tileset,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, material, rigs};

const W: f32 = 240.0;
const H: f32 = 160.0;

fn camera() -> Vec<NamedCamera> {
    vec![NamedCamera {
        name: "top",
        camera: Camera2d::new(Plane2d::Xy)
            .with_units_per_pixel(1.0)
            .with_viewport_size(Vec2::new(W, H))
            .camera(),
    }]
}

/// The painter scenes appended to the catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        NamedScene {
            name: "painter_composed_keys",
            cameras: camera(),
            build: build_composed_keys,
        },
        NamedScene {
            name: "painter_particles_layers",
            cameras: camera(),
            build: build_particles_layers,
        },
        NamedScene {
            name: "painter_over_transparent",
            cameras: camera(),
            build: build_over_transparent,
        },
    ]
}

/// A painter batch of solid squares `size` across, no depth test.
fn squares(layer: u32, size: f32) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_size = size;
    item.order = SpriteOrder::Painter {
        layer,
        depth_test: false,
    };
    item
}

/// Push one square at `at`, in `colour`, at the composed key.
fn push(item: &mut SpriteItem, at: Vec2, colour: Colour, key: f32) {
    item.positions.push([at.x, at.y, 0.0]);
    item.colours.push(colour);
    item.sort_keys.push(key);
}

fn upload_plane(ctx: &mut BuildCtx<'_>, size: f32) -> viewport_lib::MeshId {
    ctx.renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::plane(size, size))
        .expect("plane upload")
}

fn build_composed_keys(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    const LAYER: u32 = 1;
    let red = Colour::linear_rgb(0.85, 0.1, 0.1);
    let green = Colour::linear_rgb(0.1, 0.7, 0.2);
    let blue = Colour::linear_rgb(0.15, 0.3, 0.95);
    let yellow = Colour::linear_rgb(0.95, 0.85, 0.1);
    let white = Colour::linear_rgb(0.9, 0.9, 0.9);

    // Top row: each square 22 right of and 10 below the previous one.
    let at = |i: f32| Vec2::new(-80.0 + 22.0 * i, 45.0 - 10.0 * i);
    let key = |sort: f32, member: u32| painter_key(LAYER, sort, member).key;
    // Submitted out of order across two items.
    let mut first = squares(LAYER, 40.0);
    push(&mut first, at(4.0), white, key(6.0, 0));
    push(&mut first, at(3.0), yellow, key(5.0, 2));
    push(&mut first, at(0.0), red, key(4.0, 0));
    first.settings.pick_id = PickId(4101);
    let mut second = squares(LAYER, 40.0);
    push(&mut second, at(1.0), green, key(5.0, 0));
    second.settings.pick_id = PickId(4102);

    // The entity's middle part: a flat unlit surface.
    let plane = upload_plane(ctx, 40.0);
    let mut flat = SceneRenderItem::default();
    flat.mesh_id = plane;
    let p = at(2.0);
    flat.model = Mat4::from_translation(Vec3::new(p.x, p.y, 0.0)).into();
    flat.material = material(ctx, Material::from_colour(blue));
    flat.settings.unlit = true;
    flat.settings.pick_id = PickId(4103);
    flat.painter = Some(painter_key(LAYER, 5.0, 1));

    // Bottom row: a sprite, an instanced pair and a sprite at one sort value.
    let low = |x: f32| Vec2::new(x, -45.0);
    let cyan = Colour::linear_rgb(0.1, 0.75, 0.8);
    let orange = Colour::linear_rgb(0.95, 0.45, 0.05);
    let purple = Colour::linear_rgb(0.55, 0.15, 0.75);
    let mut third = squares(LAYER, 44.0);
    push(&mut third, low(40.0), purple, key(-2.0, 2));
    push(&mut third, low(-60.0), cyan, key(-2.0, 0));
    third.settings.pick_id = PickId(4104);
    let small = upload_plane(ctx, 30.0);
    let mut pair = MeshInstanceItem::default();
    pair.mesh_id = small;
    pair.transforms = [low(-40.0), low(20.0)]
        .iter()
        .map(|p| Mat4::from_translation(Vec3::new(p.x, p.y + 6.0, 0.0)).to_cols_array_2d())
        .collect();
    pair.colours = vec![orange; 2];
    pair.blend = SpriteBlend::AlphaBlend;
    pair.settings.pick_id = PickId(4105);
    pair.painter = Some(painter_key(LAYER, -2.0, 1));

    BuiltScene {
        items: vec![flat],
        sprite_items: vec![first, second, third],
        mesh_instances: vec![pair],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_particles_layers(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Two 8 x 8 tiles: a dark and a light checker cell.
    let mut pixels = Vec::with_capacity(16 * 8 * 4);
    for _ in 0..8 {
        for x in 0..16 {
            pixels.extend_from_slice(if x < 8 {
                &[40, 60, 110, 255]
            } else {
                &[90, 120, 170, 255]
            });
        }
    }
    let atlas = ctx
        .renderer
        .resources_mut()
        .upload_texture(ctx.device, ctx.queue, TextureData::srgb(16, 8, pixels))
        .expect("atlas upload");
    let mut map = Tilemap::with_chunk_size(Tileset::new(Some(atlas), [8, 8], 2), [12, 6], 8);
    for row in 0..6 {
        for col in 0..12 {
            map.set(col, row, Tile::new(((col + row) % 2) as u32));
        }
    }
    let map = ctx
        .renderer
        .upload(ctx.device, ctx.queue, &map)
        .expect("tilemap upload");
    let mut tiles = TilemapItem::new(map);
    tiles.origin = [-96.0, -48.0];
    tiles.cell_size = [16.0, 16.0];
    tiles.draw = TilemapDraw::Blended {
        layer: 1,
        depth_test: false,
        painter: TilemapPainter::default(),
    };
    tiles.settings.pick_id = PickId(4201);

    let mut config = GpuParticleSystemConfig::default();
    config.capacity = 512;
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
    particles.emitter.rate = 300.0;
    particles.emitter.lifetime = (20.0, 20.0);
    particles.emitter.size = 7.0;
    particles.emitter.colour = Colour::linear(1.0, 0.75, 0.2, 1.0);
    particles.emitter.spawn_shape = SpawnShape::Box {
        min: [-110.0, -60.0, 0.0],
        max: [110.0, 60.0, 0.0],
    };
    particles.emitter.initial_velocity = VelocityDist::Fixed([0.0, 0.0, 0.0]);
    particles.painter = Some(viewport_lib::PainterPlacement::new(2, 0.0));

    // Two bars across the middle on layer 3.
    let mut bars = squares(3, 1.0);
    bars.positions = vec![[0.0, 20.0, 0.0], [0.0, -20.0, 0.0]];
    bars.sizes = vec![[200.0, 14.0]; 2];
    bars.colours = vec![
        Colour::linear_rgb(0.85, 0.15, 0.3),
        Colour::linear_rgb(0.2, 0.8, 0.4),
    ];
    bars.settings.pick_id = PickId(4202);

    BuiltScene {
        tilemaps: vec![tiles],
        gpu_particle_systems: vec![particles],
        sprite_items: vec![bars],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

fn build_over_transparent(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let cube = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::cuboid(70.0, 70.0, 20.0))
        .expect("cube upload");
    let glass = |ctx: &mut BuildCtx<'_>, colour: Colour, x: f32, z: f32| {
        let mut item = SceneRenderItem::default();
        item.mesh_id = cube;
        item.model = Mat4::from_translation(Vec3::new(x, 0.0, z)).into();
        item.material = material(ctx, Material::from_colour(colour));
        item.settings.unlit = true;
        item.settings.opacity = 0.5;
        item
    };
    // In front of the sprite's plane on the left, behind it on the right.
    let front = glass(ctx, Colour::linear_rgb(0.1, 0.4, 0.95), -50.0, 30.0);
    let behind = glass(ctx, Colour::linear_rgb(0.95, 0.3, 0.1), 50.0, -30.0);
    // A plate behind everything, so the sprite's depth test has something.
    let plate_mesh = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::plane(W, H))
        .expect("plate upload");
    let mut plate = SceneRenderItem::default();
    plate.mesh_id = plate_mesh;
    plate.model = Mat4::from_translation(Vec3::new(0.0, 0.0, -60.0)).into();
    plate.material = material(
        ctx,
        Material::from_colour(Colour::linear_rgb(0.2, 0.2, 0.22)),
    );
    plate.settings.unlit = true;

    let mut sprite = SpriteItem::default();
    sprite.size_mode = SpriteSizeMode::WorldSpace;
    sprite.positions = vec![[0.0, 0.0, 0.0]];
    sprite.sizes = vec![[200.0, 24.0]];
    sprite.default_colour = Colour::linear(0.95, 0.95, 0.3, 1.0);
    sprite.order = SpriteOrder::Painter {
        layer: 0,
        depth_test: true,
    };

    BuiltScene {
        items: vec![plate, front, behind],
        sprite_items: vec![sprite],
        lighting: rigs::from_above(),
        ..Default::default()
    }
}
