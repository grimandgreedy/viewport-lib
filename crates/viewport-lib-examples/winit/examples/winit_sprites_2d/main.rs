//! Animated pixel-art sprites on the XY plane under `Camera2d`.
//!
//! Three hundred critters wander over the plane, every one drawn from one
//! four-frame walk cycle in a single atlas texture. The atlas and its frames
//! come from an Aseprite sprite sheet read through viewport-lib-io: the
//! `walk` tag names the frames, each frame's rectangle becomes a UV rect, and
//! the `feet` slice's pivot puts the sprite's position at its feet. Each
//! sprite flips to face the way it walks. The batch draws in painter's order
//! with `-y` as the sort key, so a critter lower on screen covers the ones
//! behind it. The atlas is nearest-sampled, the camera shows three screen
//! pixels per texel and snaps to the pixel grid, and the sprites snap too, so
//! the art stays crisp.
//!
//! A torch follows the cursor: it is a point light in the frame's lighting,
//! projected onto the plane by the 2D light layer, and two standing stones cast
//! hard shadows from it over a dim ambient. Clicking a critter flashes it
//! white; `D` dissolves the stones away with a burning rim, and again brings
//! them back.
//!
//! Drag to pan, scroll or pinch to zoom.

use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

use glam::{DVec3, Vec2};

use viewport_lib as vpl;
use viewport_lib_io::TextureSource;
use viewport_lib_io::loaders::{aseprite, png};
use viewport_lib_plugins::item_types::lighting_2d::{Lighting2dItem, occluders_from_sprites};
use viewport_lib_plugins::item_types::sprite::{
    SpriteEffects, SpriteFlip, SpriteItem, SpriteOrder, SpriteSizeMode,
};
use viewport_lib_plugins::item_types::tilemap::TilemapPlane;
use vpl::{
    Action, AppConfig, Camera2d, Camera2dController, Candela, Colour, ExtraId, KeyCode,
    LightSource, Modifiers, ModifiersMatch, Plane2d, TextureData, TextureFilter, TextureId,
    ViewportApp, ViewportBinding, ViewportGesture, WindowConfig, planar_bindings,
};

const TOGGLE_DISSOLVE: Action = Action::Custom(1);
/// The two standing stones, at their feet.
const STONES: [[f32; 2]; 2] = [[-40.0, -10.0], [50.0, 20.0]];

/// How many critters wander the plane.
const COUNT: usize = 300;
/// The critters stay inside this half extent of the plane, in world units.
const HALF_EXTENT: Vec2 = Vec2::new(200.0, 110.0);

/// The sprite sheet beside this file: its atlas image sits next to it.
const SHEET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/winit_sprites_2d/assets/critter.json"
);

/// The walk cycle as loaded from the sheet.
struct Walk {
    atlas: TextureData,
    /// One UV rect per frame of the `walk` tag.
    frames: Vec<[f32; 4]>,
    /// The sprite's size in texels, one texel per world unit.
    size: [f32; 2],
    /// The pivot along the quad's right and up axes, `[0.5, 0.0]` for feet at
    /// the bottom centre.
    pivot: [f32; 2],
}

impl Walk {
    fn load() -> Self {
        let sheet = aseprite::sprite_sheet_from_path(Path::new(SHEET)).expect("sprite sheet");
        let atlas = match sheet.image.as_ref().expect("the sheet names its image") {
            TextureSource::File(path) => png::texture_from_path(path).expect("atlas image"),
            TextureSource::Decoded(data) => data.clone(),
        };
        let walk = sheet
            .animations
            .iter()
            .find(|a| a.name == "walk")
            .expect("a walk tag");
        // A sprite quad maps `v = 0` to its bottom edge and the sheet's `v`
        // runs from the image's top, so each rect runs from the frame's
        // bottom edge up to its top.
        let frames = (walk.first..=walk.last)
            .map(|i| {
                let [u0, v0, u1, v1] = sheet.frames[i as usize].uv;
                [u0, v1, u1, v0]
            })
            .collect();
        let first = &sheet.frames[walk.first as usize];
        let size = [first.source_size[0] as f32, first.source_size[1] as f32];
        // The slice pivot is in pixels from the slice's top left; the sprite
        // pivot is a fraction of the quad from its bottom left.
        let pivot = sheet
            .slices
            .iter()
            .find(|s| s.name == "feet")
            .and_then(|s| s.keys.first())
            .and_then(|k| k.pivot.map(|p| (p, k.bounds)))
            .map(|(p, b)| {
                [
                    p[0] as f32 / b.width as f32,
                    1.0 - p[1] as f32 / b.height as f32,
                ]
            })
            .unwrap_or([0.5, 0.0]);
        Self {
            atlas,
            frames,
            size,
            pivot,
        }
    }
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

fn critters(frames: usize) -> Vec<Critter> {
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
                phase: next(&mut seed) * frames as f32,
                tint: Colour::linear_rgb(shade, 1.0, shade),
            }
        })
        .collect()
}

/// This frame's sprite batch for the critters, each with its flash amount.
fn sprites(atlas: TextureId, walk: &Walk, critters: &[Critter], flashes: &[f32]) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = Some(atlas);
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.sizes = vec![walk.size; critters.len()];
    item.filter = TextureFilter::Nearest;
    item.pixel_snap = true;
    // Feet on the position, so the sort key is where the critter stands.
    item.default_pivot = walk.pivot;
    item.order = SpriteOrder::Painter {
        layer: 0,
        depth_test: false,
    };
    for c in critters {
        item.positions.push([c.position.x, c.position.y, 0.0]);
        item.uv_rects
            .push(walk.frames[(c.phase as usize) % walk.frames.len()]);
        // The art faces right.
        item.flips.push(if c.velocity.x < 0.0 {
            SpriteFlip::HORIZONTAL
        } else {
            SpriteFlip::NONE
        });
        item.sort_keys.push(-c.position.y);
        item.colours.push(c.tint);
    }
    let mut effects = SpriteEffects::default();
    effects.flash = flashes.to_vec();
    item.effects = Some(effects);
    item
}

/// The two stones, drawn with the critters in painter's order and dissolved
/// by `dissolve`.
fn stones(dissolve: f32) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.positions = STONES.iter().map(|p| [p[0], p[1], 0.0]).collect();
    item.sizes = vec![[14.0, 26.0]; STONES.len()];
    item.default_pivot = [0.5, 0.0];
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_colour = Colour::linear_rgb(0.35, 0.33, 0.38);
    item.pixel_snap = true;
    item.order = SpriteOrder::Painter {
        layer: 0,
        depth_test: false,
    };
    item.sort_keys = STONES.iter().map(|p| -p[1]).collect();
    let mut effects = SpriteEffects::default();
    effects.default_dissolve = dissolve;
    effects.noise_scale = 2.0;
    item.effects = Some(effects);
    item
}

fn main() {
    let atlas_id: Rc<Cell<Option<TextureId>>> = Rc::new(Cell::new(None));
    let setup_atlas = atlas_id.clone();

    // Three screen pixels per texel (one texel is one world unit).
    let mut camera = Camera2d::new(Plane2d::Xy).with_units_per_pixel(1.0 / 3.0);
    camera.snap_to_pixels = true;
    let mut controller = Camera2dController::new_stateless();
    let walk = Rc::new(Walk::load());
    let setup_walk = walk.clone();
    let mut critters = critters(walk.frames.len());
    let mut flashes = vec![0.0f32; COUNT];
    let mut extras: Vec<ExtraId> = Vec::new();
    let mut torch = Vec2::ZERO;
    let mut dissolving = false;
    let mut dissolve = 0.0f32;

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
                    .upload_texture(&device, &queue, setup_walk.atlas.clone())
                    .expect("atlas upload");
                setup_atlas.set(Some(atlas));
                let mut bindings = planar_bindings();
                bindings.push(ViewportBinding::new(
                    TOGGLE_DISSOLVE,
                    ViewportGesture::KeyPress {
                        key: KeyCode::D,
                        modifiers: ModifiersMatch::Exact(Modifiers::NONE),
                    },
                ));
                setup.set_bindings(bindings);
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
                camera.pixels_per_point = ctx.pixels_per_point();
                let dt = ctx.dt();
                let frame = ctx.action_frame().clone();

                // The torch follows the cursor and stays where it was left.
                if let Some(cursor) = frame.pointer.cursor {
                    torch = camera.screen_to_world(cursor);
                }
                // A click flashes the critter under it, which fades over a
                // quarter of a second.
                for f in &mut flashes {
                    *f = (*f - dt * 4.0).max(0.0);
                }
                if frame.pointer.clicked {
                    let nearest = critters
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, (c.position + Vec2::new(0.0, 4.0)).distance(torch)))
                        .min_by(|a, b| a.1.total_cmp(&b.1));
                    if let Some((i, d)) = nearest
                        && d < 8.0
                    {
                        flashes[i] = 1.0;
                    }
                }
                if frame.is_active(TOGGLE_DISSOLVE) {
                    dissolving = !dissolving;
                }
                let target = if dissolving { 1.0 } else { 0.0 };
                dissolve += (target - dissolve).clamp(-dt, dt);

                for c in &mut critters {
                    c.position += c.velocity * dt;
                    if c.position.x.abs() > HALF_EXTENT.x {
                        c.velocity.x = -c.velocity.x;
                    }
                    if c.position.y.abs() > HALF_EXTENT.y {
                        c.velocity.y = -c.velocity.y;
                    }
                    c.position = c.position.clamp(-HALF_EXTENT, HALF_EXTENT);
                    c.phase =
                        (c.phase + dt * c.velocity.length() * 0.35) % walk.frames.len() as f32;
                }

                // The setup callback runs again after a lost device, with a new
                // atlas id and a new instance, so the batch is rebuilt each
                // frame from the current id.
                for id in extras.drain(..) {
                    ctx.remove_extra(id);
                }
                if let Some(atlas) = atlas_id.get() {
                    extras.push(ctx.add_item(sprites(atlas, &walk, &critters, &flashes)));
                }
                let stones = stones(dissolve);
                let mut light = Lighting2dItem::default();
                light.ambient = Colour::linear_rgb(0.12, 0.12, 0.16);
                // A stone stops casting once it has mostly burned away.
                if dissolve < 0.6 {
                    light.occluders = occluders_from_sprites(&stones, TilemapPlane::Xy);
                }
                extras.push(ctx.add_item(stones));
                extras.push(ctx.add_item(light));

                let mut lamp = LightSource::point_candela(
                    DVec3::new(torch.x as f64, torch.y as f64, 10.0),
                    Candela(1.4),
                    120.0,
                    0.0,
                );
                lamp.colour = Colour::linear_rgb(1.0, 0.75, 0.45);
                ctx.effects_mut().lighting.lights = vec![lamp];

                controller.apply(&mut camera, &frame, dt);
                *ctx.camera_mut() = camera.camera();
            },
        )
        .run();
}
