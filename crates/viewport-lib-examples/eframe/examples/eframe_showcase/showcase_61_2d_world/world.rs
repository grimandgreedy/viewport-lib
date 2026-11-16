//! The 2D world: its map, its actors and the frame it draws.
//!
//! Everything here depends on `viewport_lib` and `viewport_lib_plugins`
//! alone. [`World::step`] advances the world from a [`WorldInput`],
//! [`World::fill_frame`] writes one frame of it into a `FrameData`, and
//! [`build_assets`] uploads what the frame names, so the world can be driven
//! and rendered without a window.
//!
//! The world is a side view on `Plane2d::Xz`: plane x is world +X and plane y
//! is world +Z, so up on screen is up in the Z-up world, and the camera looks
//! along +Y. Depth runs along world Y, with smaller Y nearer the camera:
//!
//! | y    | what                                                   |
//! |------|--------------------------------------------------------|
//! | +1.0 | the far hills (tilemap elevation -1)                   |
//! | +0.8 | the isometric map sheet (tilemap elevation -0.8)       |
//! | +0.5 | the critters and the guard, depth-tested so walls cover them |
//! |  0.0 | the ground and walls (tilemap elevation 0)             |
//! | -0.5 | the sign (world overlay elevation 0.5) and the stones  |
//! | -1.0 | the player                                             |
//! | -2.0 | the torch flames (GPU particles)                       |
//!
//! Sprites sort for the side view: the critters, the guard, the stones and
//! the player share one painter layer, each keyed by `painter_key` from `-z`
//! (the height of the feet) and a member index, so whoever's feet are lower
//! on screen stands nearer and covers the ones behind it, across batches and
//! item types. The guard is three parts at one sort value: a spear sprite
//! (member 0), a body sprite (1) and a shield mesh (2), so it moves as one
//! unit and critters walk in front of it and behind it. A stone is member 0
//! and the player member 1, so the player standing on the ground is in front
//! of a stone, and behind it in the air.
//!
//! The isometric map sheet is its own painter layer under the actors: a
//! ground layer and a tree layer sorted by row, with a figure walking between
//! the rows keyed `-feet y` like them.

use glam::{DVec2, Vec2};
use viewport_lib as vpl;
use viewport_lib::plugin_api::Uploads;
use viewport_lib_plugins::item_types::gpu_particles::{
    GpuParticleSystemConfig, GpuParticleSystemId, GpuParticleSystems, ParticleRender,
};
use viewport_lib_plugins::item_types::lighting_2d::{
    Lighting2dStats, Occluder2d, occluders_from_tilemap,
};
use viewport_lib_plugins::item_types::sprite::{SpriteLitParams, SpriteSizeMode};
use viewport_lib_plugins::item_types::tilemap::{Tilemap, TilemapId, TilemapItem, Tileset};
use vpl::{
    ActionFrame, Camera2d, Camera2dController, Colour, FontHandle, FrameStats, MeshId,
    OverlayMeshItem, Plane2d, Rect2d, SpriteBlend, TextureFilter, TextureId,
};

mod art;
mod draw;
mod hud;
mod maps;

use art::{atlas, atlas_normals, glow_texture, hud_font, iso_atlas};
use hud::minimap_mesh;
use maps::{
    ground_map, hills_map, iso_ground_map, iso_origin, iso_tileset, iso_trees_map, solid, tileset,
};

/// World units per ground cell. One texel of art is one world unit.
pub(crate) const CELL: f32 = 8.0;
/// Ground map size in cells.
pub(crate) const COLS: u32 = 160;
pub(crate) const ROWS: u32 = 48;
/// Height of the ground's top surface.
const GROUND_Z: f32 = 40.0;
/// The painter layer the critters, the stones and the player share.
const SIDE_LAYER: u32 = 1;
/// How many critters wander the back lane.
pub(crate) const CRITTERS: usize = 300;

// Atlas cells, eight to a row. Row 0 holds the tiles, row 1 the critter walk
// cycle and the wisp, rows 2 and 3 the player's 8 x 16 frames, the stone and
// the guard's spear.
const GRASS: u32 = 0;
const DIRT: u32 = 1;
const STONE: u32 = 2;
const PLANK: u32 = 3;
const HILL: u32 = 4;
const HILL_TOP: u32 = 5;
const TORCH: u32 = 6;
const BUSH: u32 = 7;
const ATLAS_COLS: usize = 8;
const ATLAS_ROWS: usize = 4;

/// The torches, as `(col, row)` of their tile in the ground map.
const TORCHES: [(u32, u32); 8] = [
    (10, 5),
    (22, 5),
    (42, 5),
    (52, 11),
    (78, 5),
    (96, 10),
    (114, 5),
    (132, 12),
];
/// Standing stones in the player's lane, by x. They cast 2D shadows.
const STONES: [f32; 3] = [236.0, 836.0, 1124.0];
/// Light-emitting wisps drifting over the map.
const WISPS: usize = 3;

/// The player's collision box: half its width and its height, feet at the
/// bottom centre.
const PLAYER_HALF_W: f32 = 3.0;
const PLAYER_H: f32 = 15.0;
const WALK_SPEED: f32 = 70.0;
const JUMP_SPEED: f32 = 235.0;
const GRAVITY: f32 = 560.0;

/// How long a damage number floats, in seconds.
const DAMAGE_LIFE: f32 = 1.1;

/// The size the scene draws at under the reference presentation, and the
/// zoom it opens at there: two reference pixels per texel.
const REFERENCE_SIZE: [u32; 2] = [480, 270];
const REFERENCE_ZOOM: f32 = 0.5;

/// The guard's lane, in the middle of the critters' lane, and its patrol.
const GUARD_Z: f32 = GROUND_Z + 3.5;
const GUARD_PATROL: (f32, f32) = (150.0, 300.0);
const GUARD_SPEED: f32 = 16.0;

/// The isometric map sheet: its painter layer (under the actors), its size in
/// cells, its tile size in texels (one texel per world unit), and the plane
/// position of its top-left corner.
const ISO_LAYER: u32 = 0;
const ISO_COLS: i32 = 6;
const ISO_ROWS: i32 = 6;
const ISO_TILE: [u32; 2] = [16, 8];
const ISO_TOP_LEFT: Vec2 = Vec2::new(160.0, 125.0);
// Slots of the isometric atlas: two ground diamonds, and a tree two cells
// tall in slot 3.
const ISO_LIGHT: u32 = 0;
const ISO_DARK: u32 = 1;
const ISO_TREE: u32 = 3;
/// Trees on the sheet, as Tiled's `(x, y)` with row 0 at the top.
const ISO_TREES: [(i32, i32); 6] = [(1, 1), (3, 1), (2, 3), (4, 3), (1, 5), (4, 5)];
/// The figure's walk round the sheet, through Tiled cells.
const ISO_WALK: [(i32, i32); 4] = [(0, 2), (5, 2), (5, 4), (0, 4)];
const ISO_WALK_SPEED: f32 = 10.0;

// ---------------------------------------------------------------------------
// Input and options
// ---------------------------------------------------------------------------

/// What the player asks for this frame.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WorldInput {
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    /// Hit the nearest critter.
    pub attack: bool,
    /// Take a hit: the player flashes and loses health.
    pub hurt: bool,
    pub shake: bool,
}

impl WorldInput {
    fn moving(&self) -> bool {
        self.left || self.right || self.jump
    }
}

/// The section's switches.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WorldOptions {
    /// Nearest-filtered, pixel-snapped art at an integer zoom; off is
    /// linear filtering at any zoom.
    pub pixel_art: bool,
    pub night: bool,
    pub shadows: bool,
    /// The HDR path, which bloom, the vignette and the torch flames need;
    /// off draws the frame on the LDR path (`PipelineMode::Direct`).
    pub hdr: bool,
    pub bloom: bool,
    pub vignette: bool,
    pub hud: bool,
    pub outline: bool,
    /// Walk, jump and fight on a script until a movement key is pressed.
    pub autopilot: bool,
    /// Draw the scene at [`REFERENCE_SIZE`] and scale it up by the largest
    /// whole multiple that fits, with bars; off draws at the window's size.
    pub reference: bool,
    /// The guard: two sprites and a mesh sorted as one by composed keys.
    pub guard: bool,
    /// The isometric map sheet.
    pub iso_sheet: bool,
    /// The player's lantern as a 2D cone light facing the way they walk;
    /// off, a frame point light.
    pub cone_lantern: bool,
    /// A light over the guard that lights the actors' painter layer only.
    pub actor_light: bool,
    /// The HP and score labels in a bitmap font face.
    pub bitmap_hud: bool,
    /// The standing stones shaded by a normal map against the lights.
    pub normal_stones: bool,
    /// The lantern's shadows blurred at their edges.
    pub soft_shadows: bool,
}

impl Default for WorldOptions {
    fn default() -> Self {
        Self {
            pixel_art: true,
            night: false,
            shadows: true,
            hdr: true,
            bloom: true,
            vignette: true,
            hud: true,
            outline: true,
            autopilot: true,
            reference: false,
            guard: true,
            iso_sheet: true,
            cone_lantern: true,
            actor_light: true,
            bitmap_hud: true,
            normal_stones: true,
            soft_shadows: true,
        }
    }
}

/// The previous frame's counters, shown in the HUD's stats strip.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WorldStats {
    pub frame: FrameStats,
    pub lights: Lighting2dStats,
}

// ---------------------------------------------------------------------------
// Assets
// ---------------------------------------------------------------------------

/// What [`build_assets`] uploads: the atlas, each tilemap once per filter so
/// the pixel-art switch picks a handle rather than re-uploading, a glow
/// texture, one particle system per torch, the guard's shield mesh and the
/// HUD's bitmap face.
pub(crate) struct WorldAssets {
    pub atlas: TextureId,
    /// Nearest, then linear.
    pub ground: [TilemapId; 2],
    pub hills: [TilemapId; 2],
    pub iso_ground: [TilemapId; 2],
    pub iso_trees: [TilemapId; 2],
    pub torches: Vec<GpuParticleSystemId>,
    pub shield: Option<MeshId>,
    pub hud_font: Option<FontHandle>,
    /// Normals for the atlas, rounding the standing stone.
    pub normals: TextureId,
}

/// Upload the world's textures and maps and create its particle systems.
pub(crate) fn build_assets(
    renderer: &mut vpl::ViewportRenderer,
    device: &vpl::wgpu::Device,
    queue: &vpl::wgpu::Queue,
) -> WorldAssets {
    let atlas = renderer
        .resources_mut()
        .upload_texture(device, queue, atlas())
        .expect("the atlas is a well-formed sRGB image");
    let normals = renderer
        .resources_mut()
        .upload_texture(device, queue, atlas_normals())
        .expect("the atlas normals are a well-formed linear image");
    let glow = renderer
        .resources_mut()
        .upload_texture(device, queue, glow_texture())
        .expect("the glow is a well-formed sRGB image");
    let iso_atlas = renderer
        .resources_mut()
        .upload_texture(device, queue, iso_atlas())
        .expect("the isometric atlas is a well-formed sRGB image");
    let filters = [TextureFilter::Nearest, TextureFilter::Linear];
    let mut upload = |map: Tilemap| {
        renderer
            .upload(device, queue, &map)
            .expect("tilemap upload")
    };
    let ground = filters.map(|f| upload(ground_map(tileset(Some(atlas), f))));
    let hills = filters.map(|f| upload(hills_map(tileset(Some(atlas), f))));
    let iso_ground = filters.map(|f| upload(iso_ground_map(iso_tileset(Some(iso_atlas), f))));
    let iso_trees = filters.map(|f| upload(iso_trees_map(iso_tileset(Some(iso_atlas), f))));
    let shield = renderer
        .resources_mut()
        .upload_mesh_data(device, &vpl::primitives::disk(3.0, 20))
        .ok();
    let hud_font = renderer.resources_mut().upload_bitmap_font(hud_font()).ok();

    let mut config = GpuParticleSystemConfig::default();
    config.capacity = 512;
    config.render = ParticleRender::Sprite {
        texture_id: Some(glow),
        blend: SpriteBlend::Additive,
        size_mode: SpriteSizeMode::WorldSpace,
        depth_write: false,
        lit: false,
        lit_params: SpriteLitParams::default(),
        normal_texture_id: None,
    };
    // GPU particles need compute; without it the torches draw no flames.
    let torches = TORCHES
        .iter()
        .filter_map(|_| {
            renderer
                .create_gpu_particle_system(device, queue, &config)
                .ok()
        })
        .collect();
    WorldAssets {
        atlas,
        ground,
        hills,
        iso_ground,
        iso_trees,
        torches,
        shield,
        hud_font,
        normals,
    }
}

// ---------------------------------------------------------------------------
// Actors
// ---------------------------------------------------------------------------

/// A small deterministic generator, so the world is the same every run.
fn next(seed: &mut u32) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed as f32 / u32::MAX as f32
}

struct Critter {
    /// Plane position of the feet: x along the map, y the lane height.
    position: Vec2,
    speed: f32,
    phase: f32,
    tint: Colour,
    flash: f32,
}

struct Player {
    /// Plane position of the feet.
    position: Vec2,
    velocity: Vec2,
    on_ground: bool,
    facing_left: bool,
    phase: f32,
    flash: f32,
    health: f32,
}

/// The guard pacing the back lane.
struct Guard {
    x: f32,
    /// `1.0` walking right, `-1.0` left.
    dir: f32,
    phase: f32,
}

struct Damage {
    at: Vec2,
    age: f32,
    amount: u32,
    colour: Colour,
}

/// The scripted walk the autopilot follows.
struct Autopilot {
    dir: f32,
    attack_in: f32,
    hurt_in: f32,
    stuck_for: f32,
    last_x: f32,
}

/// The whole world: map, actors, camera and clock.
pub(crate) struct World {
    pub options: WorldOptions,
    pub camera: Camera2d,
    controller: Camera2dController,
    /// The viewport's size in logical pixels and its pixels per point; under
    /// the reference presentation the camera draws at the reference size
    /// instead.
    window_size: Vec2,
    pixels_per_point: f32,
    /// Whether the camera is set up for the reference presentation, and the
    /// zoom kept for the other presentation.
    presented_reference: bool,
    other_zoom: f32,
    time: f32,
    dt: f32,
    map: Tilemap,
    wall_occluders: Vec<Occluder2d>,
    minimap: OverlayMeshItem,
    critters: Vec<Critter>,
    guard: Guard,
    player: Player,
    /// The isometric sheet's grid (no texture), its origin, and how far the
    /// figure has walked round it.
    iso_set: Tileset,
    iso_origin: Vec2,
    iso_walked: f32,
    autopilot: Autopilot,
    damage: Vec<Damage>,
    score: u32,
    seed: u32,
}

impl World {
    pub(crate) fn new() -> Self {
        let map = ground_map(tileset(None, TextureFilter::Nearest));
        let wall_occluders = occluders_from_tilemap(&map, [0.0, 0.0], [CELL, CELL], |t| {
            matches!(t.index, STONE | PLANK)
        });
        let minimap = minimap_mesh(&map);
        let iso_set = iso_tileset(None, TextureFilter::Nearest);
        let mut seed = 0x2D5E_ED01;
        let extent = COLS as f32 * CELL;
        let critters = (0..CRITTERS)
            .map(|_| {
                let x = 12.0 + next(&mut seed) * (extent - 24.0);
                let lane = GROUND_Z + next(&mut seed) * 7.0;
                let speed =
                    (6.0 + next(&mut seed) * 16.0) * if next(&mut seed) < 0.5 { -1.0 } else { 1.0 };
                let hue = next(&mut seed);
                Critter {
                    position: Vec2::new(x, lane),
                    speed,
                    phase: next(&mut seed) * 4.0,
                    tint: critter_tint(hue),
                    flash: 0.0,
                }
            })
            .collect();
        let start = Vec2::new(120.0, GROUND_Z);
        let mut camera = Camera2d::new(Plane2d::Xz)
            .with_centre(start + Vec2::new(0.0, 50.0))
            .with_units_per_pixel(1.0 / 3.0)
            .with_bounds(Rect2d::new(
                Vec2::ZERO,
                Vec2::new(extent, ROWS as f32 * CELL),
            ));
        camera.dead_zone = Rect2d::new(Vec2::new(-90.0, -70.0), Vec2::new(90.0, 30.0));
        camera.follow_time_constant = 0.25;
        camera.min_units_per_pixel = 1.0 / 8.0;
        camera.max_units_per_pixel = 0.5;
        Self {
            options: WorldOptions::default(),
            camera,
            controller: Camera2dController::new_stateless(),
            window_size: Vec2::ONE,
            pixels_per_point: 1.0,
            presented_reference: false,
            other_zoom: REFERENCE_ZOOM,
            time: 0.0,
            dt: 0.0,
            map,
            wall_occluders,
            minimap,
            critters,
            guard: Guard {
                x: GUARD_PATROL.0,
                dir: 1.0,
                phase: 0.0,
            },
            iso_origin: iso_origin(&iso_set),
            iso_set,
            iso_walked: 0.0,
            player: Player {
                position: start,
                velocity: Vec2::ZERO,
                on_ground: true,
                facing_left: false,
                phase: 0.0,
                flash: 0.0,
                health: 100.0,
            },
            autopilot: Autopilot {
                dir: 1.0,
                attack_in: 0.6,
                hurt_in: 2.5,
                stuck_for: 0.0,
                last_x: start.x,
            },
            damage: Vec::new(),
            score: 0,
            seed: 0x5EED_D00D,
        }
    }

    /// Advance the world by `dt` seconds. `nav` is the frame's resolved
    /// pointer input for the camera's wheel zoom and drag pan, or `None` to
    /// leave the view to the follow alone. `viewport` is in logical pixels.
    pub(crate) fn step(
        &mut self,
        input: &WorldInput,
        nav: Option<&ActionFrame>,
        viewport: Vec2,
        pixels_per_point: f32,
        dt: f32,
    ) {
        let dt = dt.clamp(0.0, 1.0 / 15.0);
        self.dt = dt;
        self.time += dt;
        let mut input = *input;
        if input.moving() {
            self.options.autopilot = false;
        }
        if self.options.autopilot {
            self.drive_autopilot(&mut input, dt);
        }
        self.step_player(&input, dt);
        self.step_critters(dt);
        self.step_guard(dt);
        self.iso_walked += ISO_WALK_SPEED * dt;
        if input.attack {
            self.attack();
        }
        if input.hurt {
            self.hurt();
        }
        if input.shake {
            self.camera.shake(14.0, 0.45);
        }
        for d in &mut self.damage {
            d.age += dt;
        }
        self.damage.retain(|d| d.age < DAMAGE_LIFE);

        // Under the reference presentation the scene draws at the reference
        // size with one pixel per point; each presentation keeps its own zoom.
        self.window_size = viewport.max(Vec2::ONE);
        self.pixels_per_point = pixels_per_point.max(0.25);
        let camera = &mut self.camera;
        if self.options.reference != self.presented_reference {
            self.presented_reference = self.options.reference;
            let zoom = camera.units_per_pixel();
            camera.set_units_per_pixel(self.other_zoom);
            self.other_zoom = zoom;
        }
        if self.options.reference {
            let [w, h] = REFERENCE_SIZE;
            camera.set_viewport_size(Vec2::new(w as f32, h as f32));
            camera.pixels_per_point = 1.0;
        } else {
            camera.set_viewport_size(self.window_size);
            camera.pixels_per_point = self.pixels_per_point;
        }

        // Follow the player's chest with pan and zoom on top. For pixel art
        // the camera draws at a whole number of pixels per texel and snaps
        // its position, while keeping the unsnapped zoom the wheel moves.
        camera.snap_to_pixels = self.options.pixel_art;
        camera.snap_zoom_to_texel = self.options.pixel_art.then_some(1.0);
        camera.follow(self.player.position + Vec2::new(0.0, 12.0));
        match nav {
            Some(frame) => self.controller.apply(camera, frame, dt),
            None => camera.update(dt),
        }
    }

    fn drive_autopilot(&mut self, input: &mut WorldInput, dt: f32) {
        let p = &self.player;
        let extent = COLS as f32 * CELL;
        if p.position.x > extent - 40.0 {
            self.autopilot.dir = -1.0;
        } else if p.position.x < 40.0 {
            self.autopilot.dir = 1.0;
        }
        // Jump when a wall is just ahead; turn round when stuck against one.
        let ahead = p.position + Vec2::new(self.autopilot.dir * (PLAYER_HALF_W + 6.0), 2.0);
        if p.on_ground && self.solid_at(ahead) {
            input.jump = true;
        }
        let p = &self.player;
        let ap = &mut self.autopilot;
        if (p.position.x - ap.last_x).abs() < 0.2 {
            ap.stuck_for += dt;
        } else {
            ap.stuck_for = 0.0;
        }
        ap.last_x = p.position.x;
        if ap.stuck_for > 0.8 {
            ap.dir = -ap.dir;
            ap.stuck_for = 0.0;
        }
        input.left = ap.dir < 0.0;
        input.right = ap.dir > 0.0;
        ap.attack_in -= dt;
        if ap.attack_in <= 0.0 {
            ap.attack_in = 0.7;
            input.attack = true;
        }
        ap.hurt_in -= dt;
        if ap.hurt_in <= 0.0 {
            ap.hurt_in = 3.2;
            input.hurt = true;
        }
    }

    fn step_player(&mut self, input: &WorldInput, dt: f32) {
        let dir = input.right as i32 as f32 - input.left as i32 as f32;
        let p = &mut self.player;
        p.velocity.x = dir * WALK_SPEED;
        if dir != 0.0 {
            p.facing_left = dir < 0.0;
        }
        if input.jump && p.on_ground {
            p.velocity.y = JUMP_SPEED;
        }
        p.velocity.y -= GRAVITY * dt;
        p.flash = (p.flash - dt * 4.0).max(0.0);
        p.health = (p.health + dt * 3.0).min(100.0);

        // Move one axis at a time in steps of under a unit, stopping an axis
        // at the first step that would overlap a solid cell.
        let delta = self.player.velocity * dt;
        let steps = (delta.abs().max_element().ceil() as usize).max(1);
        let step = delta / steps as f32;
        let mut on_ground = false;
        for _ in 0..steps {
            let mut pos = self.player.position;
            pos.x += step.x;
            if self.player_hits(pos) {
                pos.x = self.player.position.x;
                self.player.velocity.x = 0.0;
            }
            pos.y += step.y;
            if self.player_hits(pos) {
                if step.y < 0.0 {
                    on_ground = true;
                }
                pos.y = self.player.position.y;
                self.player.velocity.y = 0.0;
            }
            self.player.position = pos;
        }
        // Standing still on a floor counts as grounded too.
        if !on_ground && self.player_hits(self.player.position - Vec2::new(0.0, 0.5)) {
            on_ground = true;
        }
        let p = &mut self.player;
        p.on_ground = on_ground;
        if p.on_ground && dir != 0.0 {
            p.phase = (p.phase + dt * 10.0) % 4.0;
        } else if p.on_ground {
            p.phase = 0.0;
        }
    }

    fn step_critters(&mut self, dt: f32) {
        let extent = COLS as f32 * CELL;
        for c in &mut self.critters {
            c.position.x += c.speed * dt;
            if c.position.x < 12.0 || c.position.x > extent - 12.0 {
                c.speed = -c.speed;
                c.position.x = c.position.x.clamp(12.0, extent - 12.0);
            }
            c.phase = (c.phase + dt * c.speed.abs() * 0.35) % 4.0;
            c.flash = (c.flash - dt * 4.0).max(0.0);
        }
    }

    fn step_guard(&mut self, dt: f32) {
        let g = &mut self.guard;
        g.x += g.dir * GUARD_SPEED * dt;
        if g.x > GUARD_PATROL.1 {
            g.dir = -1.0;
        } else if g.x < GUARD_PATROL.0 {
            g.dir = 1.0;
        }
        g.phase = (g.phase + dt * 6.0) % 4.0;
    }

    /// The plane position of the isometric figure's feet, and whether it is
    /// walking left.
    fn iso_figure(&self, item: &TilemapItem) -> (Vec2, bool) {
        let corners = ISO_WALK.map(|(x, y)| {
            let p = item.cell_to_world(&self.iso_set, x, -y, DVec2::ZERO);
            Vec2::new(p.x as f32, p.z as f32)
        });
        let lengths =
            std::array::from_fn::<f32, 4, _>(|i| corners[i].distance(corners[(i + 1) % 4]));
        let mut d = self.iso_walked % lengths.iter().sum::<f32>();
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            if d <= lengths[i] {
                return (a.lerp(b, d / lengths[i]), b.x < a.x);
            }
            d -= lengths[i];
        }
        (corners[0], false)
    }

    /// Hit the critter nearest the player's front, if one is close.
    fn attack(&mut self) {
        let p = &self.player;
        let reach = p.position + Vec2::new(if p.facing_left { -8.0 } else { 8.0 }, 0.0);
        let nearest = self
            .critters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, (c.position.x - reach.x).abs()))
            .filter(|(_, d)| *d < 16.0)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, _)) = nearest {
            let amount = 5 + (next(&mut self.seed) * 45.0) as u32;
            let c = &mut self.critters[i];
            c.flash = 1.0;
            self.score += amount;
            self.damage.push(Damage {
                at: c.position + Vec2::new(0.0, 8.0),
                age: 0.0,
                amount,
                colour: Colour::srgb(1.0, 0.86, 0.25, 1.0),
            });
        }
    }

    fn hurt(&mut self) {
        let amount = 6 + (next(&mut self.seed) * 14.0) as u32;
        let p = &mut self.player;
        p.flash = 1.0;
        p.health = (p.health - amount as f32).max(0.0);
        self.damage.push(Damage {
            at: p.position + Vec2::new(0.0, PLAYER_H),
            age: 0.0,
            amount,
            colour: Colour::srgb(1.0, 0.3, 0.25, 1.0),
        });
    }

    fn solid_at(&self, p: Vec2) -> bool {
        if p.x < 0.0 || p.y < 0.0 {
            return true;
        }
        let (col, row) = ((p.x / CELL) as u32, (p.y / CELL) as u32);
        if col >= COLS {
            return true;
        }
        self.map.get(col as i32, row as i32).is_some_and(solid)
    }

    /// Whether the player's box with its feet at `feet` overlaps a solid cell.
    fn player_hits(&self, feet: Vec2) -> bool {
        let (x0, x1) = (feet.x - PLAYER_HALF_W, feet.x + PLAYER_HALF_W - 1e-3);
        let (y0, y1) = (feet.y, feet.y + PLAYER_H - 1e-3);
        let (c0, c1) = ((x0 / CELL).floor(), (x1 / CELL).floor());
        let (r0, r1) = ((y0 / CELL).floor(), (y1 / CELL).floor());
        let mut r = r0;
        while r <= r1 {
            let mut c = c0;
            while c <= c1 {
                if self.solid_at(Vec2::new(c * CELL + 0.5, r * CELL + 0.5)) {
                    return true;
                }
                c += 1.0;
            }
            r += 1.0;
        }
        false
    }

    /// The wisps' plane positions at the current time.
    fn wisps(&self) -> [Vec2; WISPS] {
        let t = self.time;
        std::array::from_fn(|i| {
            let k = i as f32;
            let base = 180.0 + k * 420.0;
            Vec2::new(
                base + 70.0 * (t * (0.23 + 0.05 * k) + k).sin(),
                78.0 + 18.0 * (t * (0.7 + 0.1 * k) + 2.0 * k).sin(),
            )
        })
    }
}

fn critter_tint(hue: f32) -> Colour {
    let tints = [
        Colour::srgb(0.45, 0.85, 0.5, 1.0),
        Colour::srgb(0.95, 0.75, 0.35, 1.0),
        Colour::srgb(0.6, 0.7, 1.0, 1.0),
        Colour::srgb(0.95, 0.55, 0.75, 1.0),
    ];
    tints[((hue * tints.len() as f32) as usize).min(tints.len() - 1)]
}
