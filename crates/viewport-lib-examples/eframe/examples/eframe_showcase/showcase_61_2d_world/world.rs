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
//! | +0.5 | the critters, depth-tested so walls cover them         |
//! |  0.0 | the ground and walls (tilemap elevation 0)             |
//! | -0.5 | the sign (world overlay elevation 0.5) and the stones  |
//! | -1.0 | the player                                             |
//! | -2.0 | the torch flames (GPU particles)                       |
//!
//! Sprites sort for the side view: the critters, the stones and the player
//! share one painter layer with `-z` (the height of the feet) as the key, so
//! whoever's feet are lower on screen stands nearer and covers the ones behind
//! it, across the three batches. At equal keys the later batch draws on top:
//! the player standing on the ground is in front of a stone, and behind it in
//! the air.

use glam::{DVec3, Vec2};

use viewport_lib as vpl;
use viewport_lib::plugin_api::Uploads;
use viewport_lib_plugins::item_types::gpu_particles::{
    ForceField, GpuParticleSystemConfig, GpuParticleSystemId, GpuParticleSystemItem,
    GpuParticleSystems, ParticleRender, SpawnShape, VelocityDist,
};
use viewport_lib_plugins::item_types::lighting_2d::{
    Lighting2dItem, Lighting2dStats, Occluder2d, occluders_from_sprites, occluders_from_tilemap,
};
use viewport_lib_plugins::item_types::sprite::{
    SpriteEffects, SpriteFlip, SpriteItem, SpriteLitParams, SpriteOrder, SpriteOutline,
    SpriteSizeMode,
};
use viewport_lib_plugins::item_types::tilemap::{
    Tile, Tilemap, TilemapId, TilemapItem, TilemapPlane, Tileset,
};
use vpl::{
    ActionFrame, Alignment, AnchorX, AnchorY, Camera2d, Camera2dController, CameraFrame, Candela,
    Colour, FillRule, FrameData, FrameStats, LabelItem, LightSource, LineCap, OverlayBlend,
    OverlayFill, OverlayFrame, OverlayMeshItem, OverlayPolylineItem, OverlayShape,
    OverlayShapeItem, OverlayStroke, Plane2d, Rect2d, ShadowLayer, SpriteBlend, SubPath,
    TextureData, TextureFilter, TextureId, WorldOverlayLayer, WorldPlane, WorldTextSize,
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
// cycle and the wisp, rows 2 and 3 the player's 8 x 16 frames and the stone.
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
/// texture, and one particle system per torch.
pub(crate) struct WorldAssets {
    pub atlas: TextureId,
    /// Nearest, then linear.
    pub ground: [TilemapId; 2],
    pub hills: [TilemapId; 2],
    pub torches: Vec<GpuParticleSystemId>,
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
    let glow = renderer
        .resources_mut()
        .upload_texture(device, queue, glow_texture())
        .expect("the glow is a well-formed sRGB image");
    let filters = [TextureFilter::Nearest, TextureFilter::Linear];
    let mut upload = |map: Tilemap| {
        renderer
            .upload(device, queue, &map)
            .expect("tilemap upload")
    };
    let ground = filters.map(|f| upload(ground_map(tileset(Some(atlas), f))));
    let hills = filters.map(|f| upload(hills_map(tileset(Some(atlas), f))));

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
        torches,
    }
}

// ---------------------------------------------------------------------------
// Art
// ---------------------------------------------------------------------------

const TILE_ART: [[&str; 8]; 8] = [
    // Grass over dirt.
    [
        "gGgGGgGg", "GGGGGGGG", "GgGGGGgG", "dDGDDGDd", "DDDDDdDD", "DdDDDDDD", "DDDDdDDD",
        "DDDDDDDD",
    ],
    [
        "DDDDDDDD", "DdDDDDDD", "DDDDDdDD", "DDDDDDDD", "DDdDDDDD", "DDDDDDdD", "DDDDDDDD",
        "DdDDDdDD",
    ],
    // Stone blocks with mortar.
    [
        "SSSSSSSk", "SsSSSSSk", "SSSSsSSk", "kkkkkkkk", "SSSkSSSS", "SSSkSsSS", "SsSkSSSS",
        "kkkkkkkk",
    ],
    [
        "PPPPPPPP", "PpPPPPpP", "PPPPPPPP", "kkkkkkkk", "PPPPPPPP", "PPpPPPPP", "PPPPPPpP",
        "kkkkkkkk",
    ],
    [
        "hhhhhhhh", "hhHhhhhh", "hhhhhhhh", "hhhhhHhh", "hhhhhhhh", "hHhhhhhh", "hhhhhhhh",
        "hhhhhhHh",
    ],
    [
        "........", "...hh...", "..hhhh..", ".hhhHhh.", "hhhhhhhh", "hHhhhhhh", "hhhhhhHh",
        "hhhhhhhh",
    ],
    // A torch on a bracket; the flame is particles.
    [
        "........", "...FF...", "..bbbb..", "...ww...", "...ww...", "..bwwb..", "...ww...",
        "...ww...",
    ],
    [
        "........", "........", "..LL....", ".LLLL.L.", "LLELLLLL", "LLLLELLL", "LELLLLLE",
        ".LLLLLL.",
    ],
];

/// The critter's body, eight wide, over two leg rows per frame.
const CRITTER_BODY: [&str; 6] = [
    "..cccc..", ".cccccc.", "ccccWKcc", "cccccccc", "cRRRRccc", ".cccccc.",
];
const CRITTER_LEGS: [[&str; 2]; 4] = [
    [".c....c.", "cc....cc"],
    ["..c..c..", ".cc..cc."],
    [".c....c.", "cc....cc"],
    ["c......c", "c......c"],
];

/// The player's head and body, then four rows of legs per frame. Frame 3 is
/// the jump.
const PLAYER_BODY: [&str; 12] = [
    "..aaaa..", ".aaaaaa.", ".aannKa.", ".annnnn.", "..nnnn..", "..tttt..", ".tttttt.", "ntttttTn",
    "ntttttTn", ".BBBBBB.", ".tttttt.", ".tt..tt.",
];
const PLAYER_LEGS: [[&str; 4]; 4] = [
    [".ll..ll.", ".ll..ll.", ".ll..ll.", "ooo..ooo"],
    [".ll..ll.", "ll....ll", "l......l", "o......o"],
    [".ll..ll.", ".ll..ll.", ".ll..ll.", "ooo..ooo"],
    ["..llll..", "..l.ll..", "..l..l..", ".oo..oo."],
];

const STONE_ART: [&str; 16] = [
    "........", "..xxxx..", ".xxXxxx.", ".xxxxxx.", "xxxxxXxx", "xxXxxxxx", "xxxxxxxx", "xmxxxXxx",
    "xxxxxxxx", "xxxXxxxx", "xxxxxxmx", "xXxxxxxx", "xxxxxxxx", "mxxxXxxm", "mmxxxxmm", "mmmxxmmm",
];

fn palette(c: char) -> [u8; 4] {
    match c {
        'g' => [120, 205, 95, 255],
        'G' => [70, 160, 70, 255],
        'D' => [150, 110, 70, 255],
        'd' => [120, 85, 55, 255],
        'S' => [120, 120, 138, 255],
        's' => [152, 152, 168, 255],
        'k' => [52, 48, 60, 255],
        'P' => [172, 122, 72, 255],
        'p' => [138, 94, 54, 255],
        'h' => [62, 92, 116, 255],
        'H' => [84, 118, 140, 255],
        'b' => [60, 60, 72, 255],
        'w' => [112, 76, 46, 255],
        'F' => [255, 205, 90, 255],
        'L' => [52, 132, 62, 255],
        'E' => [36, 96, 46, 255],
        'c' => [235, 235, 235, 255],
        'W' => [255, 255, 255, 255],
        'K' => [20, 20, 30, 255],
        'R' => [210, 60, 70, 255],
        'a' => [92, 56, 30, 255],
        'n' => [240, 192, 152, 255],
        't' => [60, 112, 205, 255],
        'T' => [40, 76, 150, 255],
        'B' => [112, 72, 40, 255],
        'l' => [70, 70, 86, 255],
        'o' => [32, 26, 32, 255],
        'x' => [132, 132, 142, 255],
        'X' => [102, 102, 114, 255],
        'm' => [82, 140, 82, 255],
        _ => [0, 0, 0, 0],
    }
}

/// The one atlas everything draws from, row 0 of the image at the top as a
/// tileset image is drawn.
fn atlas() -> TextureData {
    let (w, h) = (ATLAS_COLS * 8, ATLAS_ROWS * 8);
    let mut pixels = vec![0u8; w * h * 4];
    let mut paint = |cx: usize, cy: usize, rows: &[&str]| {
        for (y, line) in rows.iter().enumerate() {
            for (x, c) in line.chars().enumerate() {
                let at = ((cy * 8 + y) * w + cx * 8 + x) * 4;
                pixels[at..at + 4].copy_from_slice(&palette(c));
            }
        }
    };
    for (i, art) in TILE_ART.iter().enumerate() {
        paint(i, 0, art);
    }
    for (f, legs) in CRITTER_LEGS.iter().enumerate() {
        let rows: Vec<&str> = CRITTER_BODY.iter().chain(legs.iter()).copied().collect();
        paint(f, 1, &rows);
    }
    for (f, legs) in PLAYER_LEGS.iter().enumerate() {
        let rows: Vec<&str> = PLAYER_BODY.iter().chain(legs.iter()).copied().collect();
        paint(f, 2, &rows);
    }
    paint(4, 2, &STONE_ART);
    // The wisp: a soft round glow in cell (4, 1).
    for y in 0..8 {
        for x in 0..8 {
            let d = Vec2::new(x as f32 - 3.5, y as f32 - 3.5).length() / 4.0;
            let a = (1.0 - d).clamp(0.0, 1.0);
            let at = ((8 + y) * w + 32 + x) * 4;
            pixels[at..at + 4].copy_from_slice(&[255, 255, 255, (a * a * 255.0) as u8]);
        }
    }
    TextureData::srgb(w as u32, h as u32, pixels)
}

/// A 16 x 16 radial falloff for the flame particles.
fn glow_texture() -> TextureData {
    let n = 16usize;
    let mut pixels = Vec::with_capacity(n * n * 4);
    for y in 0..n {
        for x in 0..n {
            let d = Vec2::new(x as f32 - 7.5, y as f32 - 7.5).length() / 8.0;
            let a = (1.0 - d).clamp(0.0, 1.0);
            pixels.extend_from_slice(&[255, 255, 255, (a * a * 255.0) as u8]);
        }
    }
    TextureData::srgb(n as u32, n as u32, pixels)
}

/// The UV rect of an atlas cell `w` x `h` cells from `(col, row)`, with row 0
/// at the top of the atlas.
fn cell_uv(col: usize, row: usize, w: usize, h: usize) -> [f32; 4] {
    let (cw, ch) = (1.0 / ATLAS_COLS as f32, 1.0 / ATLAS_ROWS as f32);
    [
        col as f32 * cw,
        row as f32 * ch,
        (col + w) as f32 * cw,
        (row + h) as f32 * ch,
    ]
}

// ---------------------------------------------------------------------------
// Maps
// ---------------------------------------------------------------------------

fn tileset(atlas: Option<TextureId>, filter: TextureFilter) -> Tileset {
    let mut set = Tileset::new(atlas, [8, 8], ATLAS_COLS as u32);
    set.filter = filter;
    set
}

/// The ground layer: dirt under a grass surface, low ruined walls, plank
/// platforms, torches and bushes, with tall walls at both ends.
fn ground_map(set: Tileset) -> Tilemap {
    let mut map = Tilemap::new(set, [COLS, ROWS]);
    let ground_row = (GROUND_Z / CELL) as u32 - 1;
    for col in 0..COLS {
        for row in 0..ground_row {
            map.set(col, row, Tile::new(DIRT));
        }
        map.set(col, ground_row, Tile::new(GRASS));
    }
    let mut fill = |cols: std::ops::RangeInclusive<u32>, rows: std::ops::RangeInclusive<u32>, t| {
        for c in cols {
            for r in rows.clone() {
                map.set(c, r, Tile::new(t));
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
        map.set(col, row, Tile::new(TORCH));
    }
    for col in [5, 14, 34, 58, 70, 88, 104, 118, 140, 154] {
        map.set(col, 5, Tile::new(BUSH));
    }
    map
}

/// The far layer: a band of rolling hills, in cells twice the ground's size.
fn hills_map(set: Tileset) -> Tilemap {
    let (cols, rows) = (96u32, 10u32);
    let mut map = Tilemap::new(set, [cols, rows]);
    for col in 0..cols {
        let x = col as f32;
        let top = (4.5 + 1.6 * (x * 0.21).sin() + 1.0 * (x * 0.53 + 1.0).sin()) as u32;
        for row in 0..top.min(rows) {
            map.set(col, row, Tile::new(HILL));
        }
        if top < rows {
            map.set(col, top, Tile::new(HILL_TOP));
        }
    }
    map
}

fn solid(tile: Tile) -> bool {
    matches!(tile.index, GRASS | DIRT | STONE | PLANK)
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
    pixels_per_point: f32,
    time: f32,
    dt: f32,
    map: Tilemap,
    wall_occluders: Vec<Occluder2d>,
    minimap: OverlayMeshItem,
    critters: Vec<Critter>,
    player: Player,
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
            pixels_per_point: 1.0,
            time: 0.0,
            dt: 0.0,
            map,
            wall_occluders,
            minimap,
            critters,
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

        // Follow the player's chest with pan and zoom on top. For pixel art
        // the camera draws at a whole number of pixels per texel and snaps
        // its position, while keeping the unsnapped zoom the wheel moves.
        self.pixels_per_point = pixels_per_point.max(0.25);
        let camera = &mut self.camera;
        camera.set_viewport_size(viewport.max(Vec2::ONE));
        camera.pixels_per_point = self.pixels_per_point;
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
        self.map.get(col, row).is_some_and(solid)
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

    // -----------------------------------------------------------------------
    // The frame
    // -----------------------------------------------------------------------

    /// The camera this frame draws through.
    pub(crate) fn camera_frame(&self) -> CameraFrame {
        let size = self.camera.viewport_size();
        CameraFrame::from_camera(&self.camera.camera(), [size.x, size.y])
            .with_pixels_per_point(self.pixels_per_point)
    }

    /// Write this frame of the world into `fd`: camera, items, lights,
    /// overlays and effects.
    pub(crate) fn fill_frame(&self, fd: &mut FrameData, assets: &WorldAssets, stats: &WorldStats) {
        let o = self.options;
        fd.camera = self.camera_frame();
        fd.viewport.show_axes_indicator = false;
        fd.viewport.show_grid = false;
        fd.viewport.background_colour = Some(if o.night {
            Colour::srgb(0.02, 0.03, 0.08, 1.0)
        } else {
            Colour::srgb(0.48, 0.68, 0.92, 1.0)
        });
        // The light layer runs on either path; bloom, the vignette and the
        // GPU particle torch flames on the HDR path only.
        fd.effects.display.mode = if o.hdr {
            vpl::PipelineMode::Hdr
        } else {
            vpl::PipelineMode::Direct
        };
        fd.effects.post_process.bloom.enabled = o.bloom;
        fd.effects.post_process.bloom.threshold = 1.0;
        fd.effects.post_process.bloom.intensity = 0.12;
        fd.effects.post_process.vignette.enabled = o.vignette;
        fd.effects.post_process.vignette.amount = 0.55;

        let which = if o.pixel_art { 0 } else { 1 };
        let mut far = TilemapItem::new(assets.hills[which]);
        far.plane = TilemapPlane::Xz;
        far.cell_size = [CELL * 2.0, CELL * 2.0];
        far.origin = [-150.0, 0.0];
        // Lag the camera across the screen but keep step with it vertically,
        // so the hills stay level with the ground under a pan or a zoom.
        far.parallax = [0.4, 1.0];
        far.elevation = -1.0;
        far.pixel_snap = o.pixel_art;
        let mut near = TilemapItem::new(assets.ground[which]);
        near.plane = TilemapPlane::Xz;
        near.cell_size = [CELL, CELL];
        near.time = self.time;
        near.pixel_snap = o.pixel_art;
        let tilemaps = fd.scene.items_mut::<TilemapItem>();
        tilemaps.push(far);
        tilemaps.push(near);

        let stones = self.stones_item(assets.atlas, o);
        let mut light = Lighting2dItem::default();
        light.plane = TilemapPlane::Xz;
        light.shadows = o.shadows;
        if o.night {
            light.ambient = Colour::linear_rgb(0.06, 0.07, 0.13);
            light.intensity = 1.0;
        } else {
            light.ambient = Colour::linear_rgb(0.92, 0.92, 0.95);
            light.intensity = 0.35;
        }
        light.occluders = self.wall_occluders.clone();
        light
            .occluders
            .extend(occluders_from_sprites(&stones, TilemapPlane::Xz));
        fd.scene.items_mut::<Lighting2dItem>().push(light);

        let sprites = fd.scene.items_mut::<SpriteItem>();
        sprites.push(self.critters_item(assets.atlas, o));
        sprites.push(stones);
        sprites.push(self.player_item(assets.atlas, o));
        sprites.push(self.wisps_item(assets.atlas, o));

        let particles = fd.scene.items_mut::<GpuParticleSystemItem>();
        for (i, (&id, &(col, row))) in assets.torches.iter().zip(TORCHES.iter()).enumerate() {
            let flame = torch_flame(col, row);
            let mut item = GpuParticleSystemItem::new(id, self.dt);
            item.emitter.rate = 70.0;
            item.emitter.lifetime = (0.3, 0.75);
            item.emitter.initial_velocity = VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.4,
                min_speed: 6.0,
                max_speed: 16.0 + (i % 3) as f32 * 2.0,
            };
            item.emitter.spawn_shape = SpawnShape::Sphere {
                centre: [flame.x, -2.0, flame.y],
                radius: 1.2,
            };
            item.emitter.colour = Colour::linear(3.2, 1.25, 0.3, 1.0);
            item.emitter.size = 3.0;
            item.forces.push(ForceField::Gravity([0.0, 0.0, 12.0]));
            item.forces.push(ForceField::Drag(1.8));
            particles.push(item);
        }

        fd.effects.lighting.lights = self.lights();
        fd.scene.world_overlays.push(shop_sign(self.time));
        fd.scene.world_overlays.push(ley_line(self.time));
        fd.overlays = self.overlays(stats);
    }

    /// The frame's point lights: the torches, the player's lantern and the
    /// wisps, each placed on the plane in front of the ground.
    fn lights(&self) -> Vec<LightSource> {
        let t = self.time;
        let at = |p: Vec2| DVec3::new(p.x as f64, -4.0, p.y as f64);
        let mut lights = Vec::new();
        for (i, &(col, row)) in TORCHES.iter().enumerate() {
            let k = i as f32;
            let flicker = 1.0 + 0.12 * (t * 13.0 + k).sin() + 0.08 * (t * 7.3 + 2.0 * k).sin();
            let mut l = LightSource::point_candela(
                at(torch_flame(col, row)),
                Candela(1.5 * flicker),
                75.0,
                0.0,
            );
            l.colour = Colour::linear_rgb(1.0, 0.62, 0.3);
            lights.push(l);
        }
        let mut lantern = LightSource::point_candela(
            at(self.player.position + Vec2::new(0.0, 9.0)),
            Candela(1.0),
            60.0,
            0.0,
        );
        lantern.colour = Colour::linear_rgb(1.0, 0.92, 0.75);
        lights.push(lantern);
        for p in self.wisps() {
            let mut l = LightSource::point_candela(at(p), Candela(0.9), 50.0, 0.0);
            l.colour = Colour::linear_rgb(0.35, 0.85, 1.0);
            lights.push(l);
        }
        lights
    }

    fn critters_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let mut item = pixel_sprites(Some(atlas), o);
        // Behind the walls, which cover them through the depth test.
        item.order = SpriteOrder::Painter {
            layer: SIDE_LAYER,
            depth_test: true,
        };
        let mut effects = SpriteEffects::default();
        for c in &self.critters {
            item.positions.push([c.position.x, 0.5, c.position.y]);
            let frame = (c.phase as usize) % 4;
            item.uv_rects.push(cell_uv(frame, 1, 1, 1));
            // The art faces right.
            item.flips.push(if c.speed < 0.0 {
                SpriteFlip::HORIZONTAL
            } else {
                SpriteFlip::NONE
            });
            // Feet lower on screen stand nearer, so they draw later.
            item.sort_keys.push(-c.position.y);
            item.colours.push(c.tint);
            effects.flash.push(c.flash);
        }
        item.effects = Some(effects);
        item
    }

    fn stones_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let mut item = pixel_sprites(Some(atlas), o);
        item.order = SpriteOrder::Painter {
            layer: SIDE_LAYER,
            depth_test: false,
        };
        for &x in &STONES {
            item.positions.push([x, -0.5, GROUND_Z]);
            item.sizes.push([8.0, 16.0]);
            item.uv_rects.push(cell_uv(4, 2, 1, 2));
            item.sort_keys.push(-GROUND_Z);
        }
        item
    }

    fn player_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let p = &self.player;
        let mut item = pixel_sprites(Some(atlas), o);
        item.order = SpriteOrder::Painter {
            layer: SIDE_LAYER,
            depth_test: false,
        };
        let frame = if p.on_ground { p.phase as usize % 4 } else { 3 };
        item.positions.push([p.position.x, -1.0, p.position.y]);
        item.sort_keys.push(-p.position.y);
        item.sizes.push([8.0, 16.0]);
        item.uv_rects.push(cell_uv(frame, 2, 1, 2));
        item.flips.push(if p.facing_left {
            SpriteFlip::HORIZONTAL
        } else {
            SpriteFlip::NONE
        });
        let mut effects = SpriteEffects::default();
        effects.flash = vec![p.flash];
        if o.outline {
            effects.outline = Some(SpriteOutline::new(Colour::srgb(0.1, 0.06, 0.04, 1.0), 2.0));
        }
        item.effects = Some(effects);
        item
    }

    fn wisps_item(&self, atlas: TextureId, o: WorldOptions) -> SpriteItem {
        let mut item = pixel_sprites(Some(atlas), o);
        item.blend = SpriteBlend::Additive;
        item.default_pivot = [0.5, 0.5];
        item.order = SpriteOrder::Painter {
            layer: 4,
            depth_test: false,
        };
        for p in self.wisps() {
            item.positions.push([p.x, -2.0, p.y]);
            item.uv_rects.push(cell_uv(4, 1, 1, 1));
            item.colours.push(Colour::linear(1.2, 3.0, 4.0, 1.0));
        }
        item
    }

    // -----------------------------------------------------------------------
    // Overlays
    // -----------------------------------------------------------------------

    /// The screen overlays: the HUD, the name tag and the damage numbers.
    fn overlays(&self, stats: &WorldStats) -> OverlayFrame {
        let mut ovl = OverlayFrame::default();
        ovl.time = self.time as f64;
        let p = &self.player;
        // World-sized, but never smaller than 12 or larger than 28 pixels.
        ovl.labels.push(
            LabelItem::new("Wren")
                .with_world_anchor([p.position.x, -3.0, p.position.y + PLAYER_H + 5.0])
                .with_align(Alignment::new(AnchorX::Middle, AnchorY::Baseline))
                .with_world_size(
                    5.0,
                    WorldTextSize::new(WorldPlane::Xz)
                        .with_min_px(12.0)
                        .with_max_px(28.0),
                )
                .with_colour(Colour::srgb(0.88, 0.96, 1.0, 1.0))
                .with_shadow(text_outline(0.5)),
        );
        for d in &self.damage {
            let t = d.age / DAMAGE_LIFE;
            ovl.labels.push(
                LabelItem::new(d.amount.to_string())
                    .with_world_anchor([d.at.x, -3.0, d.at.y + 4.0 + 12.0 * t])
                    .with_align(Alignment::new(AnchorX::Middle, AnchorY::Baseline))
                    .with_world_size(7.0, WorldTextSize::new(WorldPlane::Xz))
                    .with_colour(d.colour)
                    .with_shadow(text_outline(0.7))
                    .with_opacity((1.0 - t * t).clamp(0.0, 1.0)),
            );
        }
        if self.options.hud {
            self.hud(&mut ovl, stats);
        }
        ovl
    }

    fn hud(&self, ovl: &mut OverlayFrame, stats: &WorldStats) {
        let size = self.camera.viewport_size();
        let panel = Colour::srgb(0.05, 0.06, 0.1, 0.72);
        let rim = OverlayStroke::new(1.0, Colour::srgb(0.75, 0.8, 0.95, 0.6));

        // Health: a panel, a track and a bar, all shapes.
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 8.0 },
                [16.0, 16.0],
                [232.0, 36.0],
            )
            .with_fill(OverlayFill::Solid(panel))
            .with_stroke(rim.clone()),
        );
        ovl.labels.push(
            LabelItem::new("HP")
                .with_position([28.0, 34.0])
                .with_align_y(AnchorY::Middle)
                .with_font_size(14.0)
                .with_colour(Colour::srgb(1.0, 0.85, 0.8, 1.0))
                .with_z_order(2),
        );
        let frac = self.player.health / 100.0;
        let (bx, by, bw, bh) = (56.0, 27.0, 180.0, 14.0);
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 4.0 },
                [bx, by],
                [bw, bh],
            )
            .with_fill(OverlayFill::Solid(Colour::srgb(0.25, 0.05, 0.06, 1.0)))
            .with_z_order(1),
        );
        if frac > 0.0 {
            ovl.shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 4.0 },
                    [bx, by],
                    [bw * frac, bh],
                )
                .with_fill(OverlayFill::linear_gradient(
                    Colour::srgb(0.85, 0.12, 0.12, 1.0),
                    Colour::srgb(1.0, 0.62, 0.2, 1.0),
                    0.0,
                ))
                .with_z_order(2),
            );
        }

        // Score, top right.
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 8.0 },
                [size.x - 16.0 - 252.0, 16.0],
                [252.0, 30.0],
            )
            .with_fill(OverlayFill::Solid(panel))
            .with_stroke(rim.clone()),
        );
        ovl.labels.push(
            LabelItem::new(format!("SCORE {:06}", self.score))
                .with_position([size.x - 28.0, 31.0])
                .with_align(Alignment::new(AnchorX::Right, AnchorY::Middle))
                .with_font_size(16.0)
                .with_colour(Colour::srgb(1.0, 0.9, 0.45, 1.0))
                .with_z_order(2),
        );

        // The minimap: the map as one mesh, the view as a stroked rect, and
        // the player and torches as dots.
        let (mx, my) = (size.x - 16.0 - 252.0, 54.0);
        let (mw, mh) = (
            MINIMAP_COLS as f32 * MINIMAP_SCALE,
            MINIMAP_ROWS as f32 * MINIMAP_SCALE,
        );
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 6.0 },
                [mx, my],
                [252.0, mh + 12.0],
            )
            .with_fill(OverlayFill::Solid(panel))
            .with_stroke(rim.clone()),
        );
        let (ox, oy) = (mx + (252.0 - mw) * 0.5, my + 6.0);
        ovl.meshes
            .push(self.minimap.clone().with_position([ox, oy]).with_z_order(1));
        let to_map = |p: Vec2| {
            [
                ox + p.x / CELL * MINIMAP_SCALE,
                oy + (MINIMAP_ROWS as f32 - p.y / CELL) * MINIMAP_SCALE,
            ]
        };
        let view = self.camera.visible_rect();
        let (a, b) = (
            to_map(Vec2::new(view.min.x, view.max.y)),
            to_map(Vec2::new(view.max.x, view.min.y)),
        );
        let (a, b) = (
            [a[0].max(ox), a[1].max(oy)],
            [b[0].min(ox + mw), b[1].min(oy + mh)],
        );
        if b[0] > a[0] && b[1] > a[1] {
            ovl.shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 1.0 },
                    a,
                    [b[0] - a[0], b[1] - a[1]],
                )
                .with_fill(OverlayFill::Solid(Colour::srgb(1.0, 1.0, 1.0, 0.08)))
                .with_stroke(OverlayStroke::new(1.0, Colour::srgb(1.0, 1.0, 1.0, 0.85)))
                .with_z_order(3),
            );
        }
        for &(col, row) in &TORCHES {
            let c = to_map(torch_flame(col, row));
            ovl.shapes.push(
                OverlayShapeItem::new(OverlayShape::Circle, [c[0] - 1.5, c[1] - 1.5], [3.0, 3.0])
                    .with_fill(OverlayFill::Solid(Colour::srgb(1.0, 0.6, 0.2, 1.0)))
                    .with_z_order(4),
            );
        }
        let c = to_map(self.player.position + Vec2::new(0.0, 8.0));
        ovl.shapes.push(
            OverlayShapeItem::new(OverlayShape::Circle, [c[0] - 2.5, c[1] - 2.5], [5.0, 5.0])
                .with_fill(OverlayFill::Solid(Colour::srgb(0.4, 0.75, 1.0, 1.0)))
                .with_stroke(OverlayStroke::new(1.0, Colour::WHITE))
                .with_z_order(5),
        );

        // The previous frame's counters.
        let s = &stats.frame;
        let text = format!(
            "draw calls {} (plugin {})   overlay draws {}   layers {}   glyph rasters {}   vector rasters {}   2D lights {}  edges {}",
            s.draw_calls,
            s.plugin_draw_calls,
            s.overlay_draw_calls,
            s.overlay_layers_composited,
            s.overlay_glyph_rasters,
            s.overlay_vector_rasters,
            stats.lights.lights,
            stats.lights.edges,
        );
        ovl.shapes.push(
            OverlayShapeItem::new(
                OverlayShape::Rect { corner_radius: 6.0 },
                [10.0, size.y - 32.0],
                [(size.x - 20.0).min(760.0), 22.0],
            )
            .with_fill(OverlayFill::Solid(panel)),
        );
        ovl.labels.push(
            LabelItem::new(text)
                .with_position([18.0, size.y - 27.0])
                .with_align_y(AnchorY::Top)
                .with_font_size(12.0)
                .with_colour(Colour::srgb(0.85, 0.95, 0.85, 1.0))
                .with_z_order(1),
        );
    }
}

/// Rows of the map the minimap shows, from the bottom.
const MINIMAP_ROWS: u32 = 16;
const MINIMAP_COLS: u32 = COLS;
/// Logical pixels per map cell on the minimap.
const MINIMAP_SCALE: f32 = 1.5;

/// The lower rows of the map as one overlay mesh: a quad per run of like
/// cells along a row, coloured by tile.
fn minimap_mesh(map: &Tilemap) -> OverlayMeshItem {
    let colour = |t: u32| match t {
        GRASS => Some(Colour::srgb(0.3, 0.68, 0.3, 1.0)),
        DIRT => Some(Colour::srgb(0.45, 0.32, 0.2, 1.0)),
        STONE => Some(Colour::srgb(0.62, 0.62, 0.7, 1.0)),
        PLANK => Some(Colour::srgb(0.75, 0.52, 0.3, 1.0)),
        _ => None,
    };
    let mut positions = Vec::new();
    let mut colours = Vec::new();
    let mut indices = Vec::new();
    for row in 0..MINIMAP_ROWS {
        let mut col = 0;
        while col < MINIMAP_COLS {
            let tile = map.get(col, row).map_or(Tile::EMPTY_INDEX, |t| t.index);
            let start = col;
            while col < MINIMAP_COLS
                && map.get(col, row).map_or(Tile::EMPTY_INDEX, |t| t.index) == tile
            {
                col += 1;
            }
            let Some(c) = colour(tile) else {
                continue;
            };
            let s = MINIMAP_SCALE;
            let (x0, x1) = (start as f32 * s, col as f32 * s);
            let (y0, y1) = (
                (MINIMAP_ROWS - row - 1) as f32 * s,
                (MINIMAP_ROWS - row) as f32 * s,
            );
            let base = positions.len() as u32;
            positions.extend_from_slice(&[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
            colours.extend_from_slice(&[c; 4]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    OverlayMeshItem::new(positions, indices).with_colours(colours)
}

/// A batch drawn in world units with its feet on its position, filtered and
/// snapped for the current presentation.
fn pixel_sprites(atlas: Option<TextureId>, o: WorldOptions) -> SpriteItem {
    let mut item = SpriteItem::default();
    item.texture_id = atlas;
    item.size_mode = SpriteSizeMode::WorldSpace;
    item.default_size = 8.0;
    item.default_pivot = [0.5, 0.0];
    item.filter = if o.pixel_art {
        TextureFilter::Nearest
    } else {
        TextureFilter::Linear
    };
    item.pixel_snap = o.pixel_art;
    item
}

/// Plane position of a torch's flame.
fn torch_flame(col: u32, row: u32) -> Vec2 {
    Vec2::new(col as f32 * CELL + 4.0, row as f32 * CELL + 7.0)
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

/// A thin dark outline behind world-sized text, in its own units.
fn text_outline(spread: f32) -> ShadowLayer {
    ShadowLayer::new(Colour::srgb(0.05, 0.03, 0.02, 1.0), 0.0, [0.0, 0.0]).with_spread(spread)
}

/// A shop sign of vector art on a world layer: a post, a board with a rim,
/// a stroked potion-bottle path and a label. Layer y runs down the plane
/// from the layer's origin, which is the board's top left.
fn shop_sign(time: f32) -> WorldOverlayLayer {
    let mut frame = OverlayFrame::default();
    frame.time = time as f64;
    let wood = Colour::srgb(0.36, 0.22, 0.12, 1.0);
    frame.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 1.0 },
            [26.0, 26.0],
            [5.0, 18.0],
        )
        .with_fill(OverlayFill::Solid(wood)),
    );
    frame.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 3.0 },
            [0.0, 0.0],
            [58.0, 27.0],
        )
        .with_fill(OverlayFill::linear_gradient(
            Colour::srgb(0.96, 0.84, 0.54, 1.0),
            Colour::srgb(0.66, 0.43, 0.22, 1.0),
            1.4,
        ))
        .with_stroke(OverlayStroke::new(1.2, Colour::srgb(0.22, 0.12, 0.05, 1.0))),
    );
    // A potion bottle: neck, shoulders and a round belly, stroked over a
    // tinted fill.
    let bottle = SubPath::new([5.0, 0.0])
        .line_to([9.0, 0.0])
        .line_to([9.0, 5.0])
        .cubic_to([15.0, 7.0], [15.0, 17.0], [7.0, 17.0])
        .cubic_to([-1.0, 17.0], [-1.0, 7.0], [5.0, 5.0])
        .close();
    frame.shapes.push(
        OverlayShapeItem::vector(vec![bottle], FillRule::NonZero, [5.0, 5.0], [14.0, 17.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.75, 0.2, 0.55, 0.85)))
            .with_stroke(OverlayStroke::new(1.0, Colour::srgb(0.15, 0.05, 0.12, 1.0))),
    );
    frame.labels.push(
        LabelItem::new("POTIONS")
            .with_position([22.0, 9.0])
            .with_font_size(7.0)
            .with_colour(Colour::srgb(0.2, 0.1, 0.05, 1.0)),
    );
    WorldOverlayLayer::new(WorldPlane::Xz)
        .with_origin([268.0, GROUND_Z + 44.0])
        .with_elevation(0.5)
        .with_order(0)
        .with_frame(frame)
}

/// A glowing trail along the ground, rippling with time: bright enough in
/// linear light to bloom, added over what is below.
fn ley_line(time: f32) -> WorldOverlayLayer {
    let mut frame = OverlayFrame::default();
    let points = (0..=96)
        .map(|i| {
            let x = i as f32 * 4.0;
            [x, 4.0 + 2.5 * (x * 0.08 + time * 2.0).sin()]
        })
        .collect();
    frame.polylines.push(
        OverlayPolylineItem::new(points)
            .with_thickness(1.5)
            .with_cap(LineCap::Round)
            .with_colour(Colour::linear(0.8, 2.6, 4.0, 1.0))
            .with_blend(OverlayBlend::Additive),
    );
    WorldOverlayLayer::new(WorldPlane::Xz)
        .with_origin([360.0, GROUND_Z + 8.0])
        .with_elevation(0.5)
        .with_order(0)
        .with_frame(frame)
}

/// One frame of the world as a whole `FrameData`, for drivers without a host
/// frame to fill.
#[allow(dead_code)]
pub(crate) fn frame(world: &World, assets: &WorldAssets, stats: &WorldStats) -> FrameData {
    let mut fd = FrameData::new(world.camera_frame(), vpl::SceneFrame::default());
    world.fill_frame(&mut fd, assets, stats);
    fd
}
