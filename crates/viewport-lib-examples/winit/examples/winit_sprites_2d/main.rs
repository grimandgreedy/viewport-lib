//! Animated pixel-art sprites on the XY plane under `Camera2d`.
//!
//! Three hundred critters wander over the plane, every one drawn from one
//! four-frame walk cycle in a single atlas texture. Each sprite picks its frame
//! from the atlas with a UV rect, flips to face the way it walks, and stands on
//! a pivot at its feet. The batch draws in painter's order with `-y` as the sort
//! key, so a critter lower on screen covers the ones behind it. The atlas is
//! nearest-sampled, the camera shows three screen pixels per texel and snaps to
//! the pixel grid, and the sprites snap too, so the art stays crisp.
//!
//! Drag to pan, scroll or pinch to zoom.

use std::cell::Cell;
use std::rc::Rc;

use glam::Vec2;

use viewport_lib as vpl;
use viewport_lib_plugins::item_types::sprite::{
    SpriteFlip, SpriteItem, SpriteOrder, SpriteSizeMode,
};
use vpl::{
    AppConfig, Camera2d, Camera2dController, Colour, ExtraId, Plane2d, TextureData, TextureFilter,
    TextureId, ViewportApp, WindowConfig, planar_bindings,
};

/// How many critters wander the plane.
const COUNT: usize = 300;
/// Frames in the walk cycle, side by side in the atlas.
const FRAMES: usize = 4;
/// The critters stay inside this half extent of the plane, in world units.
const HALF_EXTENT: Vec2 = Vec2::new(200.0, 110.0);

/// One 8 x 8 walk frame, top row first. `L` marks the legs, which move per frame.
const BODY: [&str; 8] = [
    "..GGGG..", //
    ".GGGGGG.", //
    "GGGGWKGG", //
    "GGGGGGGG", //
    "GRRRRGGG", //
    ".GGGGGG.", //
    "........", //
    "........", //
];

/// The two leg rows of each frame.
const LEGS: [[&str; 2]; FRAMES] = [
    [".L....L.", "LL....LL"],
    ["..L..L..", ".LL..LL."],
    [".L....L.", "LL....LL"],
    ["L......L", "L......L"],
];

/// The atlas: the frames side by side, uploaded bottom row first because a
/// sprite quad maps `v = 0` to its bottom edge.
fn atlas() -> TextureData {
    let (w, h) = (8 * FRAMES as u32, 8u32);
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for row in (0..8).rev() {
        for frame in 0..FRAMES {
            let line = if row < 6 {
                BODY[row]
            } else {
                LEGS[frame][row - 6]
            };
            for c in line.chars() {
                pixels.extend_from_slice(&match c {
                    'G' | 'L' => [90, 200, 110, 255],
                    'W' => [255, 255, 255, 255],
                    'K' => [20, 20, 30, 255],
                    'R' => [210, 60, 70, 255],
                    _ => [0, 0, 0, 0],
                });
            }
        }
    }
    TextureData::srgb(w, h, pixels)
}

struct Critter {
    position: Vec2,
    velocity: Vec2,
    phase: f32,
    tint: Colour,
}

/// A small deterministic generator, so the scene is the same every run.
fn next(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed as f32 / u32::MAX as f32
}

fn critters() -> Vec<Critter> {
    let mut seed = 0x2D5E_ED01;
    (0..COUNT)
        .map(|_| {
            let position = Vec2::new(
                (next(&mut seed) * 2.0 - 1.0) * HALF_EXTENT.x,
                (next(&mut seed) * 2.0 - 1.0) * HALF_EXTENT.y,
            );
            let angle = next(&mut seed) * std::f32::consts::TAU;
            let speed = 8.0 + next(&mut seed) * 18.0;
            let shade = 0.7 + next(&mut seed) * 0.3;
            Critter {
                position,
                velocity: Vec2::from_angle(angle) * speed,
                phase: next(&mut seed) * FRAMES as f32,
                tint: Colour::linear_rgb(shade, 1.0, shade),
            }
        })
        .collect()
}

/// This frame's sprite batch for the critters.
fn sprites(atlas: TextureId, critters: &[Critter]) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = Some(atlas);
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_size = 8.0;
    item.filter = TextureFilter::Nearest;
    item.pixel_snap = true;
    // Feet on the position, so the sort key is where the critter stands.
    item.default_pivot = [0.5, 0.0];
    item.order = SpriteOrder::Painter {
        layer: 0,
        depth_test: false,
    };
    for c in critters {
        item.positions.push([c.position.x, c.position.y, 0.0]);
        let frame = (c.phase as usize) % FRAMES;
        let u0 = frame as f32 / FRAMES as f32;
        item.uv_rects.push([u0, 0.0, u0 + 1.0 / FRAMES as f32, 1.0]);
        // The art faces right.
        item.flips.push(if c.velocity.x < 0.0 {
            SpriteFlip::HORIZONTAL
        } else {
            SpriteFlip::NONE
        });
        item.sort_keys.push(-c.position.y);
        item.colours.push(c.tint);
    }
    item
}

fn main() {
    let atlas_id: Rc<Cell<Option<TextureId>>> = Rc::new(Cell::new(None));
    let setup_atlas = atlas_id.clone();

    // Three screen pixels per texel (one texel is one world unit).
    let mut camera = Camera2d::new(Plane2d::Xy).with_units_per_pixel(1.0 / 3.0);
    camera.snap_to_pixels = true;
    let mut controller = Camera2dController::new_stateless();
    let mut critters = critters();
    let mut extra: Option<ExtraId> = None;

    ViewportApp::new(AppConfig::default())
        .window_with_input(
            WindowConfig::default()
                .with_title("viewport-lib : sprites in 2D")
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
                setup_atlas.set(Some(atlas));
                setup.set_bindings(planar_bindings());
            },
            // Everything goes to the viewport: there is no UI over it.
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

                for c in &mut critters {
                    c.position += c.velocity * dt;
                    if c.position.x.abs() > HALF_EXTENT.x {
                        c.velocity.x = -c.velocity.x;
                    }
                    if c.position.y.abs() > HALF_EXTENT.y {
                        c.velocity.y = -c.velocity.y;
                    }
                    c.position = c.position.clamp(-HALF_EXTENT, HALF_EXTENT);
                    c.phase = (c.phase + dt * c.velocity.length() * 0.35) % FRAMES as f32;
                }

                // The setup callback runs again after a lost device, with a new
                // atlas id and a new instance, so the batch is rebuilt each
                // frame from the current id.
                if let Some(atlas) = atlas_id.get() {
                    if let Some(id) = extra.take() {
                        ctx.remove_extra(id);
                    }
                    extra = Some(ctx.add_item(sprites(atlas, &critters)));
                }

                let frame = ctx.action_frame().clone();
                controller.apply(&mut camera, &frame, dt);
                *ctx.camera_mut() = camera.camera();
            },
        )
        .run();
}
