//! Showcase 41: Sprites and Particles
//!
//! Three sub-modes demonstrating the `SpriteItem` API:
//!
//!   Mode A - Placed sprites: a sphere mesh with world-space-sized textured
//!     billboards scattered around it. Sprites face the camera and scale with
//!     distance as expected for markers or icons.
//!
//!   Mode B - Ring particles: two rings of particles orbiting like rings around
//!     a sphere (one equatorial, one polar). Particles are distributed around
//!     each ring with small random offsets; they rotate in unison to create an
//!     Ouroboros effect -- the leading edge is bright and large, the trailing
//!     tail fades to nothing. Wire outlines show the orbit path; they fade and
//!     reappear on a slower cycle.
//!
//!   Mode C - Sprite atlas: 9 sprites arranged in a 3x3 grid, each sampling
//!     from a 4x4 atlas texture (128x128 px, 16 cells of 32x32). All sprites
//!     cycle through the same frame to show flip-book animation.

use crate::App;
use eframe::egui;
#[allow(unused_imports)]
use viewport_lib::renderer::{SpriteLitParams, SpriteNormalMode};
use viewport_lib::scene::{Scene, build_light_glyphs};
use viewport_lib::Selection;
use viewport_lib::{
    BackfacePolicy, ForceField, FrameData, GpuParticleSystemConfig, GpuParticleSystemId,
    GpuParticleSystemItem, LightKind, LightSource, LightingSettings, MeshId, MeshInstanceItem,
    ParticleRender, PolylineItem, RibbonItem, SceneRenderItem, SpawnShape, SpriteBlend, SpriteItem,
    SpriteOrientation, SpriteSizeMode, VelocityDist, ViewportRenderer, primitives,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpriteSubMode {
    Placed,
    Particles,
    Atlas,
    /// Three columns of overlapping sprites under alpha-blend / additive /
    /// premultiplied blend modes against a solid sphere backdrop.
    Blend,
    /// A flat sprite sheet slicing through the sphere, with `soft_particle_distance`
    /// driving a smooth alpha fade at the intersection rather than a hard line.
    Soft,
    /// A swarm of small cubes orbiting like leaves, drawn through `MeshInstanceItem`
    /// with additive blending and one indexed draw call per blend bucket.
    MeshParticles,
    /// Moving emitters leaving fading ribbon trails behind them. Each emitter
    /// keeps a rolling buffer of recent positions; the ribbon's
    /// `colour_attribute` ramps alpha from zero at the tail to full at the
    /// head, and the per-vertex `width_attribute` tapers in the same direction.
    Trails,
    /// A GPU-simulated particle fountain. The host owns no per-particle state;
    /// the renderer runs the emitter, integrates forces (gravity + an
    /// orbiting attractor), and draws live particles each frame.
    GpuParticles,
    /// Sprite orientation modes side by side: a flock of camera-facing
    /// markers, a swarm of velocity-stretched sparks, and a row of grass
    /// cards locked to the world up axis.
    Orientations,
    /// Refractive sprites: an expanding shockwave ring distorts the textured
    /// scene behind it, demonstrating `SpriteItem::refraction_strength`.
    Distortion,
    /// Lit sprites: a slow-tumbling smoke cluster against a sphere, picking
    /// up directional and hemisphere ambient light as the directional light
    /// rotates. Toggling `lit` off shows the difference against the emissive
    /// path.
    Lit,
}

pub(crate) struct TrailEmitter {
    /// Most recent positions, oldest first. The last entry is the current
    /// emitter position; the first entry is the oldest point still in the
    /// trail. Updated each frame in `update_sprites`.
    pub history: Vec<[f32; 3]>,
    /// Phase offset (radians) so emitters orbit at different starting angles.
    pub phase: f32,
    /// Orbit radius.
    pub radius: f32,
    /// Vertical bob amplitude.
    pub bob: f32,
    /// RGB tint applied to the ribbon.
    pub colour: [f32; 3],
}

pub(crate) struct Ring {
    /// Current rotation angle of particles around the ring (radians).
    pub spin: f32,
    /// How fast the particles advance (rad/s). Sign gives direction.
    pub spin_rate: f32,
    /// Ring normal axis (perpendicular to the ring plane).
    pub spin_axis: [f32; 3],
    /// Ring radius.
    pub radius: f32,
    /// Base colour (RGB).
    pub colour: [f32; 3],
    /// Wire outline lifetime for the fade/reappear cycle.
    pub life: f32,
    pub max_life: f32,
    /// Per-particle base phase (0..TAU), fixed at build time. The Ouroboros
    /// gradient is keyed on these values: phase=0 is the head, phase=TAU is
    /// the tail right behind the head.
    pub particle_phases: Vec<f32>,
    /// Per-particle [radial, axial] perturbation in world units, fixed at build.
    pub particle_perturb: Vec<[f32; 2]>,
}

pub(crate) struct Particle {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub life: f32,
    pub max_life: f32,
}

pub(crate) struct SpriteState {
    pub built: bool,
    pub sub_mode: SpriteSubMode,

    // Mode A
    pub sphere_id: MeshId,
    pub sprite_tex: u64,
    pub placed_positions: Vec<[f32; 3]>,

    // Mode B
    pub particles: Vec<Particle>,
    pub glow_tex: u64,
    pub rings: [Ring; 2],

    // Mode C
    pub atlas_tex: u64,
    pub atlas_positions: Vec<[f32; 3]>,
    pub atlas_frame: u32,
    pub atlas_time: f32,

    // Mesh-particle mode
    pub cube_id: MeshId,
    /// Time used to drive the orbital animation of mesh particles and the
    /// horizontal sweep of the soft-fade sprite plane.
    pub demo_time: f32,

    // Streak texture: a procedural noisy lengthwise stripe used to demonstrate
    // ribbon texturing in the Trails sub-mode (toggle below) and the
    // velocity-stretched rain in the Orientations sub-mode.
    pub streak_tex: u64,
    pub trail_streak_enabled: bool,
    /// Flame texture: tall taper with hot inner gradient. Used by the
    /// axis-locked candle flames in the Orientations sub-mode.
    pub flame_tex: u64,
    /// Shockwave ring texture: a thin ring whose R/G channels encode outward
    /// displacement (heat-haze direction). Drives the refractive sprite in
    /// the Distortion sub-mode.
    pub shockwave_tex: u64,
    /// Flat plane mesh used as the three corner walls in the Distortion
    /// sub-mode.
    pub wall_id: MeshId,
    /// Strength slider for the Distortion sub-mode (NDC pixels).
    pub distortion_strength: f32,
    /// Current directional-light azimuth (radians around Z) for the Lit demo.
    /// Advances each frame when `lit_auto_rotate` is on; otherwise driven by
    /// the UI slider.
    pub lit_angle: f32,
    /// When true, the Lit demo's directional light slowly orbits the pillar.
    pub lit_auto_rotate: bool,
    /// Multiplier on the directional light intensity for the Lit demo.
    pub lit_intensity: f32,
    /// When true, both pillars use `lit: true`. When false, the right pillar
    /// drops back to the emissive path so the demo collapses to a single
    /// shading model for comparison against another scene.
    pub lit_show_both: bool,
    /// Scene-graph that owns the directional light driving the Lit demo. A
    /// scene-graph light is required to render the built-in light glyph
    /// (`build_light_glyphs`), so the rotating arrow shows up in the viewport.
    pub lit_scene: Scene,
    /// Node id for the directional light in `lit_scene`.
    pub lit_light_id: u64,
    /// When true, the right pillar requests `receive_shadows` and the scene
    /// enables the cascade shadow map. A small occluder mesh between the
    /// columns then casts a visible shadow gradient across the lit pillar.
    pub lit_receive_shadows: bool,

    // Orientations sub-mode toggles which orientation gets exclusive focus
    // (so the user can see one mode at a time at a useful scale).
    pub orientation_focus: SpriteOrientation,
    /// Per-rain-drop position. Updated each frame in `update_sprites` when the
    /// Orientations sub-mode is showing velocity-stretched rain.
    pub rain_positions: Vec<[f32; 3]>,
    /// Per-rain-drop velocity. Constant downward with a small per-drop jitter
    /// to keep the streaks from forming a synchronised pulse.
    pub rain_velocities: Vec<[f32; 3]>,

    // GPU particles mode
    pub gpu_particle_system: Option<GpuParticleSystemId>,
    pub gpu_emit_rate: f32,
    pub gpu_lifetime: (f32, f32),
    pub gpu_attractor_enabled: bool,

    // Trails mode
    pub trails: Vec<TrailEmitter>,
    /// Number of points held in each emitter's history ring buffer. Higher
    /// values make the trail longer and the fade smoother.
    pub trail_length: usize,
    /// Maximum ribbon half-width at the head. The tail tapers toward zero.
    pub trail_width: f32,
    pub trail_blend: SpriteBlend,
}

impl Default for SpriteState {
    fn default() -> Self {
        // Particle phase data is populated in build_sprite_scene, not here.
        Self {
            built: false,
            sub_mode: SpriteSubMode::Particles,
            sphere_id: MeshId::from_index(0),
            sprite_tex: 0,
            placed_positions: Vec::new(),
            particles: Vec::new(),
            glow_tex: 0,
            rings: [
                Ring {
                    spin: 0.0,
                    spin_rate: 0.7,
                    spin_axis: [0.0, 1.0, 0.0], // equatorial (XZ plane)
                    radius: 3.0,
                    colour: [0.35, 0.75, 1.0],
                    life: 5.0,
                    max_life: 5.0,
                    particle_phases: Vec::new(),
                    particle_perturb: Vec::new(),
                },
                Ring {
                    spin: 0.0,
                    spin_rate: -0.5,
                    spin_axis: [1.0, 0.0, 0.0], // polar (YZ plane)
                    radius: 3.0,
                    colour: [1.0, 0.5, 0.15],
                    life: 2.5, // staggered so they don't expire together
                    max_life: 5.0,
                    particle_phases: Vec::new(),
                    particle_perturb: Vec::new(),
                },
            ],
            atlas_tex: 0,
            atlas_positions: Vec::new(),
            atlas_frame: 0,
            atlas_time: 0.0,
            cube_id: MeshId::from_index(0),
            demo_time: 0.0,
            trails: Vec::new(),
            trail_length: 80,
            trail_width: 0.18,
            trail_blend: SpriteBlend::Additive,
            streak_tex: 0,
            trail_streak_enabled: false,
            flame_tex: 0,
            shockwave_tex: 0,
            wall_id: MeshId::from_index(0),
            distortion_strength: 60.0,
            lit_angle: 0.0,
            lit_auto_rotate: true,
            lit_intensity: 1.4,
            lit_show_both: true,
            lit_scene: Scene::new(),
            lit_light_id: 0,
            lit_receive_shadows: false,
            orientation_focus: SpriteOrientation::VelocityStretched,
            rain_positions: Vec::new(),
            rain_velocities: Vec::new(),
            gpu_particle_system: None,
            gpu_emit_rate: 20_000.0,
            gpu_lifetime: (1.5, 3.0),
            gpu_attractor_enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

pub(crate) fn build_sprite_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    // Upload a sphere mesh for Mode A.
    let sphere_mesh = primitives::sphere(2.0, 32, 16);
    let sphere_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &sphere_mesh)
        .expect("sprite sphere");

    // Mode A: procedural diamond texture (32x32 RGBA).
    let sprite_tex = {
        let (w, h) = (32u32, 32u32);
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let pixels: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let dx = (x as f32 - cx).abs() / cx;
                    let dy = (y as f32 - cy).abs() / cy;
                    let dist = dx + dy;
                    let a = ((1.0 - dist).max(0.0).powi(2) * 255.0) as u8;
                    [255u8, 200, 80, a]
                })
            })
            .collect();
        renderer
            .resources_mut()
            .upload_texture(&app.device, &app.queue, w, h, &pixels)
            .expect("sprite tex")
    };

    let placed_positions = icosphere_sample_positions(20, 3.5);

    // Flame texture (32x96): wide hot core at the bottom, narrowing toward
    // the tip, with a yellow-white inner gradient and soft alpha falloff.
    let flame_tex = {
        let (w, h) = (32u32, 96u32);
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            // v = 0 at the wide base, 1 at the narrow tip.
            let v = y as f32 / (h - 1) as f32;
            // Half-width of the flame at this height (texture-space).
            let taper = (1.0 - v * 0.85).max(0.05);
            for x in 0..w {
                let cx = (x as f32 - w as f32 * 0.5) / (w as f32 * 0.5); // -1..1
                let dist = cx.abs();
                let inside = (taper - dist).max(0.0) / taper;
                if inside <= 0.0 {
                    continue;
                }
                // Brighter, hotter at the core; cools toward the edge and the tip.
                let core = inside.powf(1.3) * (1.0 - v * 0.4);
                let r = 1.0_f32;
                let g = (0.45 + 0.55 * core).clamp(0.0, 1.0);
                let b = (core * 0.5).clamp(0.0, 1.0);
                let a = (core.powf(1.1) * 255.0).clamp(0.0, 255.0) as u8;
                let i = ((y * w + x) * 4) as usize;
                pixels[i]     = (r * 255.0) as u8;
                pixels[i + 1] = (g * 255.0) as u8;
                pixels[i + 2] = (b * 255.0) as u8;
                pixels[i + 3] = a;
            }
        }
        renderer
            .resources_mut()
            .upload_texture(&app.device, &app.queue, w, h, &pixels)
            .expect("flame tex")
    };

    // Shockwave texture (96x96): a ring whose R/G channels encode outward
    // radial displacement and whose alpha gates the ring intensity. Centred
    // at (0.5, 0.5) of the texture; R/G are mapped from [-1, 1] outward
    // direction into [0, 1] storage.
    let shockwave_tex = {
        let (w, h) = (96u32, 96u32);
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f32 - cx) / (w as f32 * 0.5);
                let dy = (y as f32 - cy) / (h as f32 * 0.5);
                let r = (dx * dx + dy * dy).sqrt();
                // Ring profile centred at r=0.7 with a narrow falloff.
                let ring = (1.0 - ((r - 0.7).abs() * 5.0)).clamp(0.0, 1.0);
                if ring <= 0.0 || r < 1e-4 {
                    continue;
                }
                let dir_x = dx / r;
                let dir_y = dy / r;
                let i = ((y * w + x) * 4) as usize;
                pixels[i]     = ((dir_x * 0.5 + 0.5) * 255.0) as u8;
                pixels[i + 1] = ((dir_y * 0.5 + 0.5) * 255.0) as u8;
                pixels[i + 2] = 128;
                pixels[i + 3] = (ring * 255.0) as u8;
            }
        }
        renderer
            .resources_mut()
            .upload_texture(&app.device, &app.queue, w, h, &pixels)
            .expect("shockwave tex")
    };

    // Streak texture (128x16): a horizontal lengthwise stripe with soft edges
    // and a noisy intensity along its length. Suited to ribbon trails and
    // velocity-stretched sparks.
    let streak_tex = {
        let (w, h) = (128u32, 16u32);
        let mut seed = 0x5eed_u64;
        let mut lcg = move || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32)
        };
        let noise: Vec<f32> = (0..w).map(|_| 0.6 + lcg() * 0.4).collect();
        let pixels: Vec<u8> = (0..h)
            .flat_map(|y| {
                let dy = (y as f32 - h as f32 * 0.5).abs() / (h as f32 * 0.5);
                let edge = (1.0 - dy).clamp(0.0, 1.0).powf(2.5);
                noise
                    .iter()
                    .flat_map(move |n| {
                        let a = (edge * n * 255.0) as u8;
                        [255u8, 230, 180, a]
                    })
                    .collect::<Vec<u8>>()
            })
            .collect();
        renderer
            .resources_mut()
            .upload_texture(&app.device, &app.queue, w, h, &pixels)
            .expect("streak tex")
    };

    // Mode B: soft glow disc (32x32, white, for particle sprites).
    let glow_tex = {
        let (w, h) = (32u32, 32u32);
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let pixels: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let r = (dx * dx + dy * dy).sqrt() / cx;
                    let a = ((1.0 - r * r).max(0.0) * 255.0) as u8;
                    [255u8, 255, 255, a]
                })
            })
            .collect();
        renderer
            .resources_mut()
            .upload_texture(&app.device, &app.queue, w, h, &pixels)
            .expect("glow tex")
    };

    // Generate ring particle data (200 per ring, evenly spaced + small jitter).
    let n_particles: usize = 200;
    let mut seed = 0xc0ffee_u64;
    let mut lcg = move || -> f32 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32)
    };

    for ring in &mut app.sprite_state.rings {
        ring.particle_phases.clear();
        ring.particle_perturb.clear();
        for i in 0..n_particles {
            let base = std::f32::consts::TAU * (i as f32 / n_particles as f32);
            // Small random jitter so particles don't form a perfect grid.
            let jitter = (lcg() - 0.5) * 0.08 * std::f32::consts::TAU / n_particles as f32;
            ring.particle_phases.push(base + jitter);
            let radial = (lcg() - 0.5) * 0.25;
            let axial = (lcg() - 0.5) * 0.20;
            ring.particle_perturb.push([radial, axial]);
        }
    }

    // Mode C: 128x128 atlas with 4x4 grid of 16 procedural cells.
    let atlas_tex = {
        let (aw, ah) = (128u32, 128u32);
        let cell = 32u32;
        let mut pixels = vec![0u8; (aw * ah * 4) as usize];
        for ci in 0..16u32 {
            let col = ci % 4;
            let row = ci / 4;
            let ox = col * cell;
            let oy = row * cell;
            for ly in 0..cell {
                for lx in 0..cell {
                    let cx = lx as f32 / cell as f32;
                    let cy = ly as f32 / cell as f32;
                    let (r, g, b, a) = cell_pixel(ci, cx, cy);
                    let px = ox + lx;
                    let py = oy + ly;
                    let i = ((py * aw + px) * 4) as usize;
                    pixels[i] = r;
                    pixels[i + 1] = g;
                    pixels[i + 2] = b;
                    pixels[i + 3] = a;
                }
            }
        }
        renderer
            .resources_mut()
            .upload_texture(&app.device, &app.queue, aw, ah, &pixels)
            .expect("atlas tex")
    };

    let atlas_positions: Vec<[f32; 3]> = (0..9_i32)
        .map(|i| {
            let x = (i % 3 - 1) as f32 * 3.0;
            let y = (i / 3 - 1) as f32 * 3.0;
            [x, y, 0.0]
        })
        .collect();

    // Wall mesh for the Distortion sub-mode: a flat plane that draws as a
    // clean rectangle rather than the disc cross-section a flattened sphere
    // would give at the corner.
    let wall_mesh = primitives::plane(12.0, 12.0);
    let wall_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &wall_mesh)
        .expect("distortion wall");

    // Mesh particles: small unit cube reused at every instance.
    let cube_mesh = primitives::cube(0.18);
    let cube_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &cube_mesh)
        .expect("particle cube");

    // Trail emitters: four orbiters at different radii and tints, each starts
    // with an empty history that fills up over the first ~1s of running.
    let trails = vec![
        TrailEmitter {
            history: Vec::new(),
            phase: 0.0,
            radius: 2.6,
            bob: 0.9,
            colour: [1.0, 0.45, 0.1],
        },
        TrailEmitter {
            history: Vec::new(),
            phase: std::f32::consts::FRAC_PI_2,
            radius: 2.6,
            bob: 0.9,
            colour: [0.2, 0.8, 1.0],
        },
        TrailEmitter {
            history: Vec::new(),
            phase: std::f32::consts::PI,
            radius: 2.6,
            bob: 0.9,
            colour: [0.7, 1.0, 0.3],
        },
        TrailEmitter {
            history: Vec::new(),
            phase: 3.0 * std::f32::consts::FRAC_PI_2,
            radius: 2.6,
            bob: 0.9,
            colour: [1.0, 0.3, 0.85],
        },
    ];

    // Allocate one persistent GPU particle system. Capacity is fixed at
    // creation; the per-frame item only carries emitter and force parameters.
    let mut particle_cfg = GpuParticleSystemConfig::default();
    particle_cfg.capacity = 60_000;
    particle_cfg.render = ParticleRender::Sprite {
        texture_id: Some(glow_tex),
        blend: SpriteBlend::Additive,
        size_mode: SpriteSizeMode::ScreenSpace,
        depth_write: false,
        lit: false,
        lit_params: SpriteLitParams::default(),
        normal_texture_id: None,
    };
    let particle_sys = renderer
        .resources_mut()
        .create_gpu_particle_system(&app.device, &app.queue, &particle_cfg);
    app.sprite_state.gpu_particle_system = Some(particle_sys);

    app.sprite_state.streak_tex = streak_tex;
    app.sprite_state.flame_tex = flame_tex;
    app.sprite_state.shockwave_tex = shockwave_tex;
    app.sprite_state.wall_id = wall_id;

    // Rain field for the Orientations demo: 600 drops scattered in a column
    // above the ground, each with a roughly downward velocity. The host
    // advances positions each frame in `update_sprites`.
    let mut rain_positions = Vec::with_capacity(600);
    let mut rain_velocities = Vec::with_capacity(600);
    let mut rain_seed = 0x7a1d_u64;
    let mut rain_rand = move || -> f32 {
        rain_seed = rain_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((rain_seed >> 33) as f32) / (u32::MAX as f32)
    };
    for _ in 0..600 {
        let x = (rain_rand() - 0.5) * 12.0;
        let y = (rain_rand() - 0.5) * 12.0;
        let z = rain_rand() * 6.0; // 0..6 above the ground
        rain_positions.push([x, y, z]);
        rain_velocities.push([0.0, 0.0, -6.0 - rain_rand() * 2.0]);
    }
    app.sprite_state.rain_positions = rain_positions;
    app.sprite_state.rain_velocities = rain_velocities;
    app.sprite_state.cube_id = cube_id;
    app.sprite_state.sphere_id = sphere_id;
    app.sprite_state.sprite_tex = sprite_tex;
    app.sprite_state.placed_positions = placed_positions;
    app.sprite_state.particles = spawn_burst(2000);
    app.sprite_state.glow_tex = glow_tex;
    app.sprite_state.atlas_tex = atlas_tex;
    app.sprite_state.atlas_positions = atlas_positions;
    app.sprite_state.atlas_frame = 0;
    app.sprite_state.trails = trails;

    // Lit sub-mode: one scene-graph directional light driving both the
    // shading on the right pillar and the on-screen arrow glyph. Direction
    // and intensity are rewritten in `submit_sprite_items` each frame.
    let mut dir_light = LightSource::default();
    dir_light.kind = LightKind::Directional {
        direction: [1.0, 0.0, 0.55],
    };
    dir_light.colour = [1.0, 1.0, 1.0];
    dir_light.intensity = 1.4;
    app.sprite_state.lit_light_id = app.sprite_state.lit_scene.add_light(dir_light);

    app.sprite_state.built = true;
}

// Generate a simple cell pattern for each of 16 atlas frames.
fn cell_pixel(frame: u32, cx: f32, cy: f32) -> (u8, u8, u8, u8) {
    let hue = frame as f32 / 16.0;
    let (r, g, b) = hsl_to_rgb(hue, 0.85, 0.55);
    let dx = cx - 0.5;
    let dy = cy - 0.5;
    let dist = (dx * dx + dy * dy).sqrt();
    let shape = match frame % 3 {
        0 => {
            let inside = (dist - 0.33).abs() < 0.12;
            if inside { 1.0 } else { 0.0 }
        }
        1 => {
            let arm = 0.08;
            let on_h = (dy.abs() < arm) && dist < 0.48;
            let on_v = (dx.abs() < arm) && dist < 0.48;
            if on_h || on_v { 1.0 } else { 0.0 }
        }
        _ => ((0.42 - dist) * 12.0).clamp(0.0, 1.0),
    };
    let a = (shape * 255.0) as u8;
    (r, g, b, a)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if h6 < 1.0 {
        (c, x, 0.0)
    } else if h6 < 2.0 {
        (x, c, 0.0)
    } else if h6 < 3.0 {
        (0.0, c, x)
    } else if h6 < 4.0 {
        (0.0, x, c)
    } else if h6 < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    let to_u8 = |v: f32| ((v + m) * 255.0).clamp(0.0, 255.0) as u8;
    (to_u8(r1), to_u8(g1), to_u8(b1))
}

fn spawn_burst(count: usize) -> Vec<Particle> {
    let mut seed = 0x4d595df4u64;
    let mut rand_f = move || -> f32 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32) / (u32::MAX as f32)
    };
    (0..count)
        .map(|_| {
            let theta = rand_f() * std::f32::consts::TAU;
            let phi = rand_f() * std::f32::consts::PI;
            let speed = 1.5 + rand_f() * 2.5;
            let life = 0.5 + rand_f() * 1.5;
            Particle {
                pos: [0.0, 0.0, 0.0],
                vel: [
                    phi.sin() * theta.cos() * speed,
                    phi.sin() * theta.sin() * speed,
                    phi.cos() * speed,
                ],
                life,
                max_life: life,
            }
        })
        .collect()
}

fn icosphere_sample_positions(n: usize, radius: f32) -> Vec<[f32; 3]> {
    let golden = std::f32::consts::PI * (3.0 - 5_f32.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - (i as f32 / (n - 1) as f32) * 2.0;
            let r = (1.0 - y * y).sqrt();
            let theta = golden * i as f32;
            [
                r * theta.cos() * radius,
                y * radius,
                r * theta.sin() * radius,
            ]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_sprites(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Sub-mode:");
    // Wrap the sub-mode chips so the sidebar stays narrow even as more modes
    // get added. egui's `horizontal_wrapped` flows children into the next row
    // when they would overflow.
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::Placed,
            "Placed",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::Particles,
            "Particles",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::Atlas,
            "Atlas",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::Blend,
            "Blend",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::Soft,
            "Soft",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::MeshParticles,
            "Mesh Particles",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::Trails,
            "Trails",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::GpuParticles,
            "GPU Particles",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::Orientations,
            "Orientations",
        );
        ui.selectable_value(
            &mut app.sprite_state.sub_mode,
            SpriteSubMode::Distortion,
            "Distortion",
        );
        ui.selectable_value(&mut app.sprite_state.sub_mode, SpriteSubMode::Lit, "Lit");
    });
    ui.separator();

    match app.sprite_state.sub_mode {
        SpriteSubMode::Placed => {
            ui.label("World-space textured billboards around a sphere.");
            ui.label("Sprites face the camera and scale with distance.");
        }
        SpriteSubMode::Particles => {
            ui.label("Two rings of particles orbiting like rings around a sphere.");
            ui.label("Particles chase their tail (Ouroboros): bright head, fading tail.");
            ui.label("Wire outlines show the orbit path and fade on a slow cycle.");
        }
        SpriteSubMode::Atlas => {
            ui.label("9 sprites in a 3x3 grid.");
            ui.label("Each samples a 4x4 atlas texture (16 cells), cycling frames as a flip-book.");
        }
        SpriteSubMode::Blend => {
            ui.label("Three columns of overlapping sprites against a solid sphere.");
            ui.label("Left: SpriteBlend::AlphaBlend (default). Middle: Additive. Right: Premultiplied.");
            ui.label("Additive columns brighten where sprites overlap; alpha columns do not.");
        }
        SpriteSubMode::Soft => {
            ui.label("Two rows of sprites sweep horizontally through the sphere.");
            ui.label("Lower (blue) row uses the batch default `soft_particle_distance = 0.3`.");
            ui.label("Upper (orange) row overrides via `soft_particle_distances = 2.5` per instance.");
            ui.label("The wider fade on the upper row shows per-instance distance taking precedence.");
        }
        SpriteSubMode::MeshParticles => {
            ui.label("A swarm of cubes orbiting like leaves, drawn through `MeshInstanceItem`.");
            ui.label("Each blend bucket renders as one indexed draw call, additive blending picked");
            ui.label("here so overlapping cubes glow brighter at their intersections.");
        }
        SpriteSubMode::GpuParticles => {
            ui.label("GPU-simulated particle fountain. The host owns no per-particle state;");
            ui.label("emit + sim run as compute passes in the renderer each frame and draw");
            ui.label("through a sprite shader sourcing positions from the particle buffer.");
            ui.separator();
            ui.label("Emit rate (particles/s):");
            ui.add(
                egui::Slider::new(&mut app.sprite_state.gpu_emit_rate, 0.0..=40_000.0)
                    .step_by(500.0),
            );
            ui.label("Lifetime (min, max seconds):");
            ui.horizontal(|ui| {
                ui.add(egui::Slider::new(&mut app.sprite_state.gpu_lifetime.0, 0.1..=4.0));
                ui.add(egui::Slider::new(&mut app.sprite_state.gpu_lifetime.1, 0.1..=8.0));
            });
            if app.sprite_state.gpu_lifetime.1 < app.sprite_state.gpu_lifetime.0 {
                app.sprite_state.gpu_lifetime.1 = app.sprite_state.gpu_lifetime.0;
            }
            ui.checkbox(
                &mut app.sprite_state.gpu_attractor_enabled,
                "Orbiting point attractor",
            );
        }
        SpriteSubMode::Trails => {
            ui.label("Four orbiters trace figure-eights and leave fading ribbon trails behind them.");
            ui.label("Each trail is a `RibbonItem` whose `colour_attribute` ramps alpha from 0 at the");
            ui.label("tail to 1 at the head, and whose `width_attribute` tapers the same way.");
            ui.separator();
            ui.label("Trail length:");
            ui.add(
                egui::Slider::new(&mut app.sprite_state.trail_length, 10..=200)
                    .step_by(1.0),
            );
            ui.label("Head width:");
            ui.add(
                egui::Slider::new(&mut app.sprite_state.trail_width, 0.02..=0.5)
                    .step_by(0.01),
            );
            ui.label("Blend mode:");
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut app.sprite_state.trail_blend,
                    SpriteBlend::AlphaBlend,
                    "Alpha",
                );
                ui.selectable_value(
                    &mut app.sprite_state.trail_blend,
                    SpriteBlend::Additive,
                    "Additive",
                );
                ui.selectable_value(
                    &mut app.sprite_state.trail_blend,
                    SpriteBlend::Premultiplied,
                    "Premultiplied",
                );
            });
            ui.checkbox(
                &mut app.sprite_state.trail_streak_enabled,
                "Streak texture (modulates ribbon colour along its length)",
            );
        }
        SpriteSubMode::Lit => {
            ui.label("Two smoke pillars side by side under a single directional light.");
            ui.label("Left pillar: `lit: false` (the emissive baseline).");
            ui.label("Right pillar: `lit: true`, spherical normals; the bright side tracks");
            ui.label("the light direction so the pillar reads as a solid volume.");
            ui.separator();
            ui.checkbox(&mut app.sprite_state.lit_auto_rotate, "Auto-rotate light");
            ui.label("Light azimuth (radians around Z):");
            ui.add(
                egui::Slider::new(
                    &mut app.sprite_state.lit_angle,
                    -std::f32::consts::PI..=std::f32::consts::PI,
                )
                .step_by(0.01),
            );
            ui.label("Directional intensity:");
            ui.add(egui::Slider::new(
                &mut app.sprite_state.lit_intensity,
                0.0..=3.0,
            ));
            ui.checkbox(
                &mut app.sprite_state.lit_show_both,
                "Right pillar lit (uncheck to make both emissive)",
            );
            ui.checkbox(
                &mut app.sprite_state.lit_receive_shadows,
                "Receive shadows (lit pillar darkens behind the wall)",
            );
        }
        SpriteSubMode::Distortion => {
            ui.label("Expanding shockwave ring distorts the textured scene behind it.");
            ui.label("`SpriteItem::refraction_strength` routes the sprite through a post-pass");
            ui.label("that samples the resolved scene colour at an offset driven by the");
            ui.label("sprite's own texture (R/G channels as signed displacement, alpha as mask).");
            ui.separator();
            ui.label("Refraction strength (NDC pixels):");
            ui.add(
                egui::Slider::new(&mut app.sprite_state.distortion_strength, 0.0..=200.0)
                    .step_by(1.0),
            );
            ui.separator();
            ui.label("Note: refractive sprites distort whatever is actually behind them.");
            ui.label("Orbit the camera so the shockwave overlaps one of the textured walls;");
            ui.label("from angles where the wall is to the side, the ring shows the empty");
            ui.label("background instead -- the same behaviour as game-engine heat haze.");
        }
        SpriteSubMode::Orientations => {
            ui.label("Three sprite orientation modes side by side. Pick which one");
            ui.label("the demo draws so it stays at a useful scale.");
            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut app.sprite_state.orientation_focus,
                    SpriteOrientation::CameraFacing,
                    "Camera-facing",
                );
                ui.selectable_value(
                    &mut app.sprite_state.orientation_focus,
                    SpriteOrientation::VelocityStretched,
                    "Velocity-stretched",
                );
                ui.selectable_value(
                    &mut app.sprite_state.orientation_focus,
                    SpriteOrientation::AxisLocked,
                    "Axis-locked (world up)",
                );
            });
            match app.sprite_state.orientation_focus {
                SpriteOrientation::CameraFacing => {
                    ui.label("Category markers in a 3D scatter plot. Each cluster's icon stays");
                    ui.label("readable from every angle as the camera orbits.");
                }
                SpriteOrientation::VelocityStretched => {
                    ui.label("Falling rain. Each drop stretches along its downward velocity");
                    ui.label("vector and lengthens with speed, producing classic streak rain.");
                }
                SpriteOrientation::AxisLocked => {
                    ui.label("Candle flames locked to world up. Each flame stays vertical as");
                    ui.label("the camera orbits instead of pivoting toward the viewer.");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame update
// ---------------------------------------------------------------------------

pub(crate) fn update_sprites(app: &mut App, dt: f32) {
    match app.sprite_state.sub_mode {
        SpriteSubMode::Particles => {
            // Simulate burst particles.
            let mut seed = 0xdeadbeef_u64.wrapping_add(app.sprite_state.particles.len() as u64);
            let mut rand_f = move || -> f32 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32) / (u32::MAX as f32)
            };
            for p in &mut app.sprite_state.particles {
                p.pos[0] += p.vel[0] * dt;
                p.pos[1] += p.vel[1] * dt;
                p.pos[2] += p.vel[2] * dt;
                p.vel[1] -= 1.2 * dt;
                p.life -= dt;
                if p.life <= 0.0 {
                    let theta = rand_f() * std::f32::consts::TAU;
                    let phi = rand_f() * std::f32::consts::PI;
                    let speed = 1.5 + rand_f() * 2.5;
                    p.pos = [0.0, 0.0, 0.0];
                    p.vel = [
                        phi.sin() * theta.cos() * speed,
                        phi.sin() * theta.sin() * speed,
                        phi.cos() * speed,
                    ];
                    p.max_life = 0.5 + rand_f() * 1.5;
                    p.life = p.max_life;
                }
            }

            // Advance ring spins and wire-outline lifetime.
            for ring in &mut app.sprite_state.rings {
                ring.spin += ring.spin_rate * dt;
                ring.life -= dt;
                if ring.life <= 0.0 {
                    ring.life = ring.max_life;
                }
            }
        }
        SpriteSubMode::Atlas => {
            app.sprite_state.atlas_time += dt;
            let fps = 10.0_f32;
            app.sprite_state.atlas_frame = (app.sprite_state.atlas_time * fps) as u32 % 16;
        }
        SpriteSubMode::Soft
        | SpriteSubMode::MeshParticles
        | SpriteSubMode::GpuParticles => {
            app.sprite_state.demo_time += dt;
        }
        SpriteSubMode::Distortion => {
            app.sprite_state.demo_time += dt;
        }
        SpriteSubMode::Lit => {
            app.sprite_state.demo_time += dt;
            if app.sprite_state.lit_auto_rotate {
                app.sprite_state.lit_angle += dt * 1.2;
                if app.sprite_state.lit_angle > std::f32::consts::PI {
                    app.sprite_state.lit_angle -= std::f32::consts::TAU;
                }
            }
        }
        SpriteSubMode::Orientations => {
            app.sprite_state.demo_time += dt;
            // Advance the rain regardless of which orientation is focused so
            // switching between them does not visibly reset the drops.
            for i in 0..app.sprite_state.rain_positions.len() {
                let v = app.sprite_state.rain_velocities[i];
                let p = &mut app.sprite_state.rain_positions[i];
                p[0] += v[0] * dt;
                p[1] += v[1] * dt;
                p[2] += v[2] * dt;
                if p[2] < -0.5 {
                    // Recycle to the top with a fresh xy.
                    let mut seed = 0xabad_u64.wrapping_add(i as u64).wrapping_add(
                        (app.sprite_state.demo_time * 9173.0) as u64,
                    );
                    let mut r = || -> f32 {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((seed >> 33) as f32) / (u32::MAX as f32)
                    };
                    p[0] = (r() - 0.5) * 12.0;
                    p[1] = (r() - 0.5) * 12.0;
                    p[2] = 6.0 + r() * 0.5;
                }
            }
        }
        SpriteSubMode::Trails => {
            app.sprite_state.demo_time += dt;
            let t = app.sprite_state.demo_time;
            let max = app.sprite_state.trail_length;
            for emitter in &mut app.sprite_state.trails {
                // Figure-eight in XY with a bob in Z so the trail crosses
                // itself, making the additive overlap visible.
                let theta = emitter.phase + t * 1.3;
                let pos = [
                    emitter.radius * theta.sin(),
                    emitter.radius * (theta * 2.0).sin() * 0.5,
                    emitter.bob * (theta * 0.5 + emitter.phase).sin(),
                ];
                emitter.history.push(pos);
                if emitter.history.len() > max {
                    let excess = emitter.history.len() - max;
                    emitter.history.drain(0..excess);
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Ring helpers
// ---------------------------------------------------------------------------

/// Compute the two orthonormal basis vectors spanning the plane perpendicular
/// to `axis`. The axis is assumed to be normalised.
fn ring_basis(axis: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let u = if axis.x.abs() < 0.9 {
        axis.cross(glam::Vec3::X).normalize()
    } else {
        axis.cross(glam::Vec3::Y).normalize()
    };
    let v = axis.cross(u).normalize();
    (u, v)
}

// ---------------------------------------------------------------------------
// Ring polylines (wire outlines)
// ---------------------------------------------------------------------------

/// Build a closed circle polyline for the ring outline.
///
/// The ring is drawn in its fixed plane (the particles chase around it).
/// Alpha is driven by life/max_life so the outline fades in and out.
fn ring_polyline(ring: &Ring, segments: usize) -> PolylineItem {
    let axis = glam::Vec3::from(ring.spin_axis).normalize();
    let (u, v) = ring_basis(axis);

    let wire_alpha = (ring.life / ring.max_life).clamp(0.0, 1.0);
    // Keep the wire dim so it reads as a guide rather than the main feature.
    let wire_alpha = wire_alpha * 0.35;
    let [r, g, b] = ring.colour;
    let colour = [r, g, b, wire_alpha];

    let mut positions = Vec::with_capacity(segments + 1);
    for i in 0..=segments {
        let t = (i % segments) as f32 / segments as f32 * std::f32::consts::TAU;
        let pos = u * (t.cos() * ring.radius) + v * (t.sin() * ring.radius);
        positions.push([pos.x, pos.y, pos.z]);
    }

    let n = positions.len() as u32;
    let node_colours = vec![colour; positions.len()];

    let mut item = PolylineItem::default();
    item.positions = positions;
    item.strip_lengths = vec![n];
    item.node_colours = node_colours;
    item.line_width = 1.5;
    item
}

pub(crate) fn ring_polylines(app: &App) -> Vec<PolylineItem> {
    if !app.sprite_state.built || app.sprite_state.sub_mode != SpriteSubMode::Particles {
        return vec![];
    }
    app.sprite_state
        .rings
        .iter()
        .map(|r| ring_polyline(r, 64))
        .collect()
}

// ---------------------------------------------------------------------------
// Sprite items
// ---------------------------------------------------------------------------

pub(crate) fn sprite_items(app: &App) -> Vec<SpriteItem> {
    if !app.sprite_state.built {
        return vec![];
    }

    match app.sprite_state.sub_mode {
        SpriteSubMode::Placed => {
            let mut item = SpriteItem::default();
            item.texture_id = Some(app.sprite_state.sprite_tex);
            item.positions = app.sprite_state.placed_positions.clone();
            item.default_colour = [1.0, 1.0, 1.0, 1.0];
            item.default_size = 0.6;
            item.size_mode = SpriteSizeMode::WorldSpace;
            item.depth_write = true;
            vec![item]
        }

        SpriteSubMode::Particles => {
            let mut items = Vec::with_capacity(3);

            // Burst particles from the origin.
            {
                let positions: Vec<[f32; 3]> =
                    app.sprite_state.particles.iter().map(|p| p.pos).collect();
                let colours: Vec<[f32; 4]> = app
                    .sprite_state
                    .particles
                    .iter()
                    .map(|p| {
                        let t = (p.life / p.max_life).clamp(0.0, 1.0);
                        [1.0, 0.6 * t + 0.2, 0.1, t * t]
                    })
                    .collect();
                let mut item = SpriteItem::default();
                item.texture_id = Some(app.sprite_state.glow_tex);
                item.positions = positions;
                item.colours = colours;
                item.default_size = 14.0;
                item.size_mode = SpriteSizeMode::ScreenSpace;
                item.depth_write = false;
                items.push(item);
            }

            // One SpriteItem per ring so each ring keeps its own colour.

            for ring in &app.sprite_state.rings {
                if ring.particle_phases.is_empty() {
                    continue;
                }

                let axis = glam::Vec3::from(ring.spin_axis).normalize();
                let (u, v) = ring_basis(axis);

                let n = ring.particle_phases.len();
                let mut positions = Vec::with_capacity(n);
                let mut colours = Vec::with_capacity(n);
                let mut sizes = Vec::with_capacity(n);

                for i in 0..n {
                    let phase = ring.particle_phases[i];
                    let angle = ring.spin + phase;
                    let [rad_off, ax_off] = ring.particle_perturb[i];

                    // Position on (or near) the ring.
                    let r = ring.radius + rad_off;
                    let pos = u * (angle.cos() * r) + v * (angle.sin() * r) + axis * ax_off;
                    positions.push([pos.x, pos.y, pos.z]);

                    // Ouroboros gradient: phase=0 is the head (bright, large),
                    // phase=TAU is the tail (dim, small) just behind the head.
                    let t = phase / std::f32::consts::TAU;
                    let alpha = (1.0 - t).powf(1.4) * 0.90 + 0.05;
                    let size = 6.0 + (1.0 - t) * 14.0; // head=20, tail=6

                    let [r, g, b] = ring.colour;
                    colours.push([r, g, b, alpha]);
                    sizes.push(size);
                }

                let mut item = SpriteItem::default();
                item.texture_id = Some(app.sprite_state.glow_tex);
                item.positions = positions;
                item.colours = colours;
                item.sizes = sizes;
                item.size_mode = SpriteSizeMode::ScreenSpace;
                item.depth_write = false;
                items.push(item);
            }

            items
        }

        SpriteSubMode::Blend => {
            // Three columns of overlapping glow sprites against the sphere.
            // Each column uses a different blend mode so the user can compare
            // how the same geometry composites under each. World-space sized
            // sprites are stacked vertically with tight spacing to force
            // overlap, where the modes diverge most.
            let column_x = [-3.5_f32, 0.0, 3.5];
            let blends = [
                SpriteBlend::AlphaBlend,
                SpriteBlend::Additive,
                SpriteBlend::Premultiplied,
            ];
            let mut items = Vec::with_capacity(3);
            for (i, blend) in blends.iter().enumerate() {
                let mut positions = Vec::with_capacity(8);
                let mut colours = Vec::with_capacity(8);
                for j in 0..8 {
                    let z = -2.5 + j as f32 * 0.7;
                    positions.push([column_x[i], 0.0, z]);
                    // Tinted glow, slightly translucent so overlap matters.
                    colours.push([1.0, 0.55, 0.2, 0.55]);
                }
                let mut item = SpriteItem::default();
                item.texture_id = Some(app.sprite_state.glow_tex);
                item.positions = positions;
                item.colours = colours;
                item.default_size = 1.6;
                item.size_mode = SpriteSizeMode::WorldSpace;
                item.blend = *blend;
                item.depth_write = false;
                items.push(item);
            }
            items
        }

        SpriteSubMode::Soft => {
            // Two rows of large glow sprites sweeping horizontally through the
            // central sphere. Both rows share the batch-default
            // soft_particle_distance, but the upper row overrides it with a
            // much longer per-instance distance via soft_particle_distances.
            // The contrast against the sphere makes the per-instance override
            // visible: the upper row fades out far earlier as it approaches
            // the sphere surface.
            let t = app.sprite_state.demo_time;
            let sweep_x = (t * 0.4).sin() * 2.5;
            let mut positions = Vec::with_capacity(32);
            let mut colours = Vec::with_capacity(32);
            let mut distances = Vec::with_capacity(32);
            for row in 0..2 {
                let y = if row == 0 { 1.6 } else { -1.6 };
                let (tint, per_instance) = if row == 0 {
                    ([1.0, 0.55, 0.35, 0.9], 2.5)
                } else {
                    ([0.4, 0.75, 1.0, 0.9], 0.0)
                };
                for i in 0..16 {
                    let z = -3.5 + i as f32 * 0.5;
                    positions.push([sweep_x, y, z]);
                    colours.push(tint);
                    distances.push(per_instance);
                }
            }
            let mut item = SpriteItem::default();
            item.texture_id = Some(app.sprite_state.glow_tex);
            item.positions = positions;
            item.colours = colours;
            item.default_size = 1.6;
            item.size_mode = SpriteSizeMode::WorldSpace;
            item.blend = SpriteBlend::AlphaBlend;
            item.depth_write = false;
            item.soft_particle_distance = Some(0.3);
            item.soft_particle_distances = distances;
            vec![item]
        }

        SpriteSubMode::MeshParticles => {
            // Mesh particles draw the cube instances; no sprite items here.
            Vec::new()
        }

        SpriteSubMode::GpuParticles => {
            // GPU particles draw through their own pipeline; no sprite items here.
            Vec::new()
        }

        SpriteSubMode::Orientations => orientation_demo_items(app),

        SpriteSubMode::Lit => {
            // Two rising smoke columns, side by side, sharing one directional
            // light. Each puff is parameterised by a phase that wraps over a
            // fixed cycle: as `demo_time` advances the puff drifts upward,
            // spreads laterally, swells slightly, and fades in/out so the
            // wrap at the top of the column is invisible. The left column is
            // emissive (`lit: false`); the right is lit, so its bright side
            // tracks the rotating directional light while the left stays flat.
            const PUFFS_PER_PILLAR: usize = 40;
            const CYCLE: f32 = 6.0;        // seconds for a puff to rise and recycle
            const RISE_HEIGHT: f32 = 4.5;  // metres travelled per cycle
            const BASE_Z: f32 = -0.6;
            let t = app.sprite_state.demo_time;
            let pillar_x = 1.8;
            let build_pillar = |x_offset: f32, lit: bool, seed: f32| {
                let mut positions = Vec::with_capacity(PUFFS_PER_PILLAR);
                let mut colours = Vec::with_capacity(PUFFS_PER_PILLAR);
                let mut sizes = Vec::with_capacity(PUFFS_PER_PILLAR);
                for i in 0..PUFFS_PER_PILLAR {
                    let stagger = (i as f32 / PUFFS_PER_PILLAR as f32) * CYCLE;
                    let age = (t + stagger + seed).rem_euclid(CYCLE);
                    let u = age / CYCLE;                       // 0..1 over one cycle
                    let h = BASE_Z + u * RISE_HEIGHT;

                    // Lateral spread grows with age: a sharp jet at the base
                    // billows out as it rises, like a real smoke plume.
                    let spread = 0.1 + u * 0.6;
                    let swirl = (i as f32 * 1.7 + age * 1.3 + seed * 4.1).sin();
                    let swirl2 = (i as f32 * 2.3 + age * 0.9 + seed * 2.7).cos();
                    let dx = swirl * spread;
                    let dy = swirl2 * spread;
                    positions.push([x_offset + dx, dy, h]);

                    // Cosine-shaped alpha hides the recycle: fade in at birth,
                    // fade out near the top of the rise.
                    let fade = (u * std::f32::consts::PI).sin();
                    colours.push([0.92, 0.90, 0.86, 0.55 * fade]);
                    sizes.push(1.0 + u * 1.2);
                }
                let mut item = SpriteItem::default();
                item.texture_id = Some(app.sprite_state.glow_tex);
                item.positions = positions;
                item.colours = colours;
                item.sizes = sizes;
                item.default_size = 1.5;
                item.size_mode = SpriteSizeMode::WorldSpace;
                item.blend = SpriteBlend::AlphaBlend;
                item.depth_write = false;
                item.lit = lit;
                item.lit_params = SpriteLitParams {
                    roughness: 0.95,
                    normal_mode: SpriteNormalMode::Spherical,
                    receive_shadows: app.sprite_state.lit_receive_shadows,
                    ambient_scale: 0.7,
                };
                item
            };
            // Different seeds so the two columns churn out of sync.
            vec![
                build_pillar(-pillar_x, false, 0.0),
                build_pillar(pillar_x, app.sprite_state.lit_show_both, 1.7),
            ]
        }

        SpriteSubMode::Distortion => {
            // Single shockwave sprite expanding in front of the textured
            // wall. World-space sizing keeps the ring at a physical diameter
            // so the distortion looks anchored in the scene; the wall sits
            // at z=-2.5 and the shockwave at z=0 stays in front of it as
            // long as the camera is on the +Z side of the wall.
            let t = app.sprite_state.demo_time;
            let cycle = (t % 3.0) / 3.0;
            // Radius grows from 1.5 to 7 world units over the cycle so the
            // ring covers a good chunk of the textured backdrop even at the
            // head-on view distance.
            let radius = 1.5 + cycle * 5.5;
            let fade = (1.0 - cycle).clamp(0.0, 1.0);
            let mut item = SpriteItem::default();
            item.texture_id = Some(app.sprite_state.shockwave_tex);
            item.positions = vec![[0.0, 0.0, 0.0]];
            item.colours = vec![[1.0, 1.0, 1.0, fade]];
            item.sizes = vec![radius * 2.0];
            item.size_mode = SpriteSizeMode::WorldSpace;
            item.depth_write = false;
            item.refraction_strength = Some(app.sprite_state.distortion_strength);
            vec![item]
        }

        SpriteSubMode::Trails => {
            // A small glowing dot at the head of each emitter makes the
            // ribbon read as a comet rather than a free-floating brushstroke.
            let mut positions = Vec::new();
            let mut colours = Vec::new();
            for emitter in &app.sprite_state.trails {
                if let Some(&head) = emitter.history.last() {
                    positions.push(head);
                    colours.push([emitter.colour[0], emitter.colour[1], emitter.colour[2], 1.0]);
                }
            }
            if positions.is_empty() {
                return Vec::new();
            }
            let mut item = SpriteItem::default();
            item.texture_id = Some(app.sprite_state.glow_tex);
            item.positions = positions;
            item.colours = colours;
            item.default_size = 24.0;
            item.size_mode = SpriteSizeMode::ScreenSpace;
            item.blend = SpriteBlend::Additive;
            item.depth_write = false;
            vec![item]
        }

        SpriteSubMode::Atlas => {
            let frame = app.sprite_state.atlas_frame;
            let col = frame % 4;
            let row = frame / 4;
            let cell = 1.0 / 4.0_f32;
            let uv_rect = [
                col as f32 * cell,
                row as f32 * cell,
                (col + 1) as f32 * cell,
                (row + 1) as f32 * cell,
            ];
            let n = app.sprite_state.atlas_positions.len();
            let uv_rects = vec![uv_rect; n];
            let mut item = SpriteItem::default();
            item.texture_id = Some(app.sprite_state.atlas_tex);
            item.positions = app.sprite_state.atlas_positions.clone();
            item.uv_rects = uv_rects;
            item.default_colour = [1.0, 1.0, 1.0, 1.0];
            item.default_size = 1.2;
            item.size_mode = SpriteSizeMode::WorldSpace;
            item.depth_write = true;
            vec![item]
        }
    }
}

// ---------------------------------------------------------------------------
// Scene items (sphere mesh for Mode A)
// ---------------------------------------------------------------------------

pub(crate) fn sprite_scene_items(app: &App) -> Vec<SceneRenderItem> {
    if !app.sprite_state.built {
        return vec![];
    }
    match app.sprite_state.sub_mode {
        SpriteSubMode::Placed | SpriteSubMode::Blend | SpriteSubMode::Soft => {
            // The sphere acts as both a backdrop (Placed, Blend) and the
            // opaque geometry the soft-particle fade is measured against.
            let mut item = SceneRenderItem::default();
            item.mesh_id = app.sprite_state.sphere_id;
            item.material.base_colour = [0.3, 0.45, 0.7];
            item.material.specular = 0.2;
            vec![item]
        }
        SpriteSubMode::Lit => {
            // A small slab between the two columns acts as a shadow caster.
            // It sits at z=2 (mid-pillar) so the orbiting directional light
            // throws a gradient across the right (lit) pillar but never
            // touches the left (emissive) one. The slab is hidden when
            // shadows are off because there's nothing for it to do.
            if !app.sprite_state.lit_receive_shadows {
                return vec![];
            }
            let mut item = SceneRenderItem::default();
            item.mesh_id = app.sprite_state.sphere_id;
            item.model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.05, 1.4, 1.0),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, 2.0),
            )
            .to_cols_array_2d();
            item.material.base_colour = [0.35, 0.3, 0.25];
            item.material.specular = 0.1;
            vec![item]
        }
        SpriteSubMode::Distortion => {
            // Three flat textured planes meeting at the (-X, -Y, -Z) corner.
            // Each plane primitive sits in the XY plane facing +Z by default;
            // the rotations move two of them into the XZ and YZ planes so the
            // three meet at a clean corner without the disc cross-sections a
            // flattened sphere would produce.
            let walls = [
                // Floor (XY plane at z=-2.5, normal +Z).
                glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -2.5)),
                // Back wall (XZ plane at y=-2.5, normal +Y).
                glam::Mat4::from_translation(glam::Vec3::new(0.0, -2.5, 0.0))
                    * glam::Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                // Side wall (YZ plane at x=-2.5, normal +X).
                glam::Mat4::from_translation(glam::Vec3::new(-2.5, 0.0, 0.0))
                    * glam::Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ];
            walls
                .iter()
                .map(|m| {
                    let mut item = SceneRenderItem::default();
                    item.mesh_id = app.sprite_state.wall_id;
                    item.model = m.to_cols_array_2d();
                    item.material.base_colour = [0.85, 0.85, 0.85];
                    item.material.specular = 0.1;
                    item.material.texture_id = Some(app.sprite_state.atlas_tex);
                    // Render both sides of each wall so the textured surface
                    // is visible from every orbit angle, not just the side
                    // the primitive normal points toward. Without this the
                    // shockwave sees an empty scene behind it from half the
                    // orbit and the distortion samples the clear colour.
                    item.material.backface_policy = BackfacePolicy::Identical;
                    item
                })
                .collect()
        }
        SpriteSubMode::Orientations => match app.sprite_state.orientation_focus {
            // Scatter plot doesn't need backdrop geometry; the cluster
            // markers and points are the scene.
            SpriteOrientation::CameraFacing => vec![],
            // Candle flames sit on a dark slab so the additive glow reads.
            SpriteOrientation::AxisLocked => {
                let mut item = SceneRenderItem::default();
                item.mesh_id = app.sprite_state.sphere_id;
                item.model = glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::new(5.0, 5.0, 0.02),
                    glam::Quat::IDENTITY,
                    glam::Vec3::new(0.0, 0.0, -0.05),
                )
                .to_cols_array_2d();
                item.material.base_colour = [0.18, 0.12, 0.08];
                vec![item]
            }
            // Rain reads against a darker disc that acts as a puddle.
            SpriteOrientation::VelocityStretched => {
                let mut item = SceneRenderItem::default();
                item.mesh_id = app.sprite_state.sphere_id;
                item.model = glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::new(5.0, 5.0, 0.02),
                    glam::Quat::IDENTITY,
                    glam::Vec3::new(0.0, 0.0, -0.5),
                )
                .to_cols_array_2d();
                item.material.base_colour = [0.18, 0.22, 0.28];
                vec![item]
            }
        },
        _ => vec![],
    }
}

/// Mesh-instance batches for the MeshParticles sub-mode.
///
/// Builds a single batch of 200 cubes orbiting two interleaved rings, tinted
/// by a slow hue cycle. All instances share one mesh and one additive blend
/// bucket, so the renderer issues exactly one indexed draw call for the batch.
pub(crate) fn mesh_instance_items(app: &App) -> Vec<MeshInstanceItem> {
    if !app.sprite_state.built || app.sprite_state.sub_mode != SpriteSubMode::MeshParticles {
        return vec![];
    }
    let t = app.sprite_state.demo_time;
    let count = 200;
    let mut transforms = Vec::with_capacity(count);
    let mut colours = Vec::with_capacity(count);
    for i in 0..count {
        let frac = i as f32 / count as f32;
        let ring_axis = if i % 2 == 0 {
            glam::Vec3::Y
        } else {
            glam::Vec3::Z
        };
        let (u, v) = ring_basis(ring_axis);
        let phase = frac * std::f32::consts::TAU + t * 0.6 * ring_axis.dot(glam::Vec3::Y).signum();
        let radius = 3.0 + (t * 0.4 + frac * 5.0).sin() * 0.4;
        let pos = u * (phase.cos() * radius) + v * (phase.sin() * radius);
        let yaw = t + frac * std::f32::consts::TAU;
        let rot = glam::Mat4::from_rotation_y(yaw) * glam::Mat4::from_rotation_x(yaw * 0.7);
        let model = glam::Mat4::from_translation(pos) * rot;
        transforms.push(model.to_cols_array_2d());
        let hue = (frac + t * 0.05) % 1.0;
        let (r, g, b) = hsl_to_rgb(hue, 0.85, 0.55);
        colours.push([
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            0.8,
        ]);
    }
    let mut item = MeshInstanceItem::default();
    item.mesh_id = app.sprite_state.cube_id.index() as u64;
    item.transforms = transforms;
    item.colours = colours;
    item.blend = SpriteBlend::Additive;
    vec![item]
}

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------

pub(crate) fn sprite_lighting(app: &App) -> LightingSettings {
    let mut settings = LightingSettings::default();
    match app.sprite_state.sub_mode {
        SpriteSubMode::Lit => {
            // The Lit sub-mode owns its directional light on the scene graph
            // (so the on-screen arrow glyph shows up). Here we only set the
            // ambient sky/ground tint and leave `lights` empty so the
            // scene-graph light is not duplicated. Shadows follow the demo's
            // `receive_shadows` toggle.
            settings.lights = vec![];
            settings.shadows_enabled = app.sprite_state.lit_receive_shadows;
            settings.hemisphere_intensity = 0.35;
            settings.sky_colour = [0.65, 0.72, 0.85];
            settings.ground_colour = [0.18, 0.16, 0.14];
        }
        _ => {
            let mut light = LightSource::default();
            light.kind = LightKind::Directional {
                direction: [0.4, 0.7, 0.6],
            };
            light.colour = [1.0, 1.0, 1.0];
            light.intensity = 0.8;
            settings.lights = vec![light];
            settings.shadows_enabled = false;
            settings.hemisphere_intensity = 0.4;
            settings.sky_colour = [0.85, 0.9, 1.0];
            settings.ground_colour = [0.4, 0.4, 0.5];
        }
    }
    settings
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn submit_sprite_items(app: &mut App, fd: &mut FrameData, dt: f32) {
    if !app.sprite_state.built {
        return;
    }
    fd.scene.sprite_items.extend(sprite_items(app));
    fd.scene.polylines.extend(ring_polylines(app));
    fd.scene.mesh_instances.extend(mesh_instance_items(app));
    fd.scene.ribbon_items.extend(trail_ribbon_items(app));
    if let Some(item) = gpu_particle_item(app, dt) {
        fd.scene.gpu_particle_systems.push(item);
    }

    // Lit sub-mode: rewrite the scene-graph directional light each frame so
    // it tracks the rotating azimuth and the chosen intensity, then collect
    // the light into the frame and append its glyph so the arrow shows up.
    if app.sprite_state.sub_mode == SpriteSubMode::Lit {
        // Orbit the light around the smoke columns: the node sits on a
        // horizontal ring of radius ~4.5 at half-column height, and its
        // direction points from the centre of the columns up toward the
        // node, so the bright side of the lit pillar tracks the glyph as it
        // orbits.
        let a = app.sprite_state.lit_angle;
        let orbit_radius = 4.5;
        let orbit_height = 2.5;
        let target = glam::Vec3::new(0.0, 0.0, 1.5);
        let light_pos = glam::Vec3::new(
            a.cos() * orbit_radius,
            a.sin() * orbit_radius,
            orbit_height,
        );
        let dir = (light_pos - target).normalize_or_zero();

        let mut src = LightSource::default();
        src.kind = LightKind::Directional {
            direction: dir.into(),
        };
        src.colour = [1.0, 1.0, 1.0];
        src.intensity = app.sprite_state.lit_intensity;
        app.sprite_state
            .lit_scene
            .set_light(app.sprite_state.lit_light_id, Some(src));
        app.sprite_state.lit_scene.set_local_transform(
            app.sprite_state.lit_light_id,
            glam::Mat4::from_translation(light_pos),
        );

        fd.scene
            .lights
            .extend(app.sprite_state.lit_scene.collect_lights());
        let (glyphs, polylines) =
            build_light_glyphs(&app.sprite_state.lit_scene, &Selection::new());
        fd.scene.glyphs.extend(glyphs);
        fd.scene.polylines.extend(polylines);
    }
}

/// Build a per-frame `GpuParticleSystemItem` for the GpuParticles sub-mode.
///
/// The emitter is a vertical cone at the origin. Forces are gravity plus an
/// optional point attractor that orbits around the system on a slow cycle, so
/// the trail bends visibly even though every particle is host-stateless.
fn gpu_particle_item(app: &App, dt: f32) -> Option<GpuParticleSystemItem> {
    if app.sprite_state.sub_mode != SpriteSubMode::GpuParticles {
        return None;
    }
    let sys = app.sprite_state.gpu_particle_system?;

    let mut item = GpuParticleSystemItem::new(sys, dt);
    item.emitter.rate = app.sprite_state.gpu_emit_rate;
    item.emitter.lifetime = app.sprite_state.gpu_lifetime;
    item.emitter.initial_velocity = VelocityDist::UniformCone {
        axis: [0.0, 0.0, 1.0],
        half_angle: 0.35,
        min_speed: 2.0,
        max_speed: 4.5,
    };
    item.emitter.spawn_shape = SpawnShape::Sphere {
        center: [0.0, 0.0, 0.0],
        radius: 0.2,
    };
    item.emitter.colour = [1.0, 0.55, 0.15, 1.0];
    item.emitter.size = 18.0;
    item.forces.push(ForceField::Gravity([0.0, 0.0, -2.5]));
    if app.sprite_state.gpu_attractor_enabled {
        let t = app.sprite_state.demo_time;
        let r = 2.6;
        item.forces.push(ForceField::PointAttractor {
            position: [r * t.cos(), r * t.sin(), 1.0],
            strength: 6.0,
            falloff: 0.5,
        });
    }
    Some(item)
}

/// Sprite items for the `Orientations` sub-mode.
///
/// Each focus picks the scene that orientation is actually for:
/// camera-facing for world-space labels over a centerpiece, velocity-stretched
/// for falling rain, axis-locked for vertical grass on a ground plane.
fn orientation_demo_items(app: &App) -> Vec<SpriteItem> {
    match app.sprite_state.orientation_focus {
        SpriteOrientation::CameraFacing => {
            // 3D scatter plot: three clusters of data points with a large
            // category marker pinned at each cluster centroid. The cluster
            // markers sample three different cells of the atlas texture so
            // they read as distinct categories; small data points use the
            // glow texture tinted per cluster.
            let cluster_centres = [
                [ 2.4_f32,  0.0,  1.0_f32],
                [-1.8,       2.0,  0.5],
                [ 0.0,      -2.4, -0.5],
            ];
            let cluster_rgb = [
                [0.30, 0.65, 1.00], // blue
                [1.00, 0.55, 0.20], // orange
                [0.55, 0.85, 0.30], // green
            ];
            let mut items = Vec::with_capacity(2);

            // Small data points (one batch, per-instance tint per cluster).
            let mut positions = Vec::with_capacity(cluster_centres.len() * 80);
            let mut colours = Vec::with_capacity(cluster_centres.len() * 80);
            let mut seed = 0xda7au64;
            let mut rand = move || -> f32 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32) / (u32::MAX as f32)
            };
            for (ci, c) in cluster_centres.iter().enumerate() {
                let [r, g, b] = cluster_rgb[ci];
                for _ in 0..80 {
                    // Approximate gaussian via three-step uniform sum.
                    let nx = (rand() + rand() + rand()) / 3.0 - 0.5;
                    let ny = (rand() + rand() + rand()) / 3.0 - 0.5;
                    let nz = (rand() + rand() + rand()) / 3.0 - 0.5;
                    let spread = 1.2;
                    positions.push([
                        c[0] + nx * spread,
                        c[1] + ny * spread,
                        c[2] + nz * spread,
                    ]);
                    colours.push([r, g, b, 0.65]);
                }
            }
            let mut points = SpriteItem::default();
            points.texture_id = Some(app.sprite_state.glow_tex);
            points.positions = positions;
            points.colours = colours;
            points.default_size = 9.0;
            points.size_mode = SpriteSizeMode::ScreenSpace;
            points.blend = SpriteBlend::Additive;
            points.depth_write = false;
            points.orientation = SpriteOrientation::CameraFacing;
            items.push(points);

            // Category markers at the centroids: large atlas icons, world-
            // space sized so they scale with distance like real labels.
            let cell = 1.0_f32 / 4.0;
            let mut marker_pos = Vec::with_capacity(3);
            let mut marker_uv = Vec::with_capacity(3);
            let mut marker_col = Vec::with_capacity(3);
            for (ci, c) in cluster_centres.iter().enumerate() {
                marker_pos.push(*c);
                // Pick three distinct atlas cells with a clear shape each.
                let cell_idx = match ci {
                    0 => 0_u32,
                    1 => 4_u32,
                    _ => 8_u32,
                };
                let col_x = (cell_idx % 4) as f32 * cell;
                let row_y = (cell_idx / 4) as f32 * cell;
                marker_uv.push([col_x, row_y, col_x + cell, row_y + cell]);
                let [r, g, b] = cluster_rgb[ci];
                marker_col.push([r, g, b, 1.0]);
            }
            let mut markers = SpriteItem::default();
            markers.texture_id = Some(app.sprite_state.atlas_tex);
            markers.positions = marker_pos;
            markers.colours = marker_col;
            markers.uv_rects = marker_uv;
            markers.default_size = 1.1;
            markers.size_mode = SpriteSizeMode::WorldSpace;
            markers.blend = SpriteBlend::AlphaBlend;
            markers.depth_write = true;
            markers.orientation = SpriteOrientation::CameraFacing;
            items.push(markers);

            items
        }

        SpriteOrientation::VelocityStretched => {
            // Rain. Positions are advanced by the host each frame in
            // `update_sprites`; the constant downward velocity drives the
            // stretch direction and length so the drops read as streaks.
            let mut item = SpriteItem::default();
            item.texture_id = Some(app.sprite_state.streak_tex);
            item.positions = app.sprite_state.rain_positions.clone();
            item.velocities = app.sprite_state.rain_velocities.clone();
            item.default_colour = [0.75, 0.85, 1.0, 0.85];
            item.default_size = 18.0;
            item.size_mode = SpriteSizeMode::ScreenSpace;
            item.blend = SpriteBlend::AlphaBlend;
            item.orientation = SpriteOrientation::VelocityStretched;
            vec![item]
        }

        SpriteOrientation::AxisLocked => {
            // Candle flames on a ring around the origin. Axis-locked to world
            // up means each flame stays vertical as the camera orbits, the
            // way real fire does. Additive blend so overlapping flames
            // brighten the way overlapping fire would.
            let count = 14usize;
            let mut positions = Vec::with_capacity(count + 8);
            let mut sizes = Vec::with_capacity(count + 8);
            // Outer ring of larger candles.
            for i in 0..count {
                let a = i as f32 / count as f32 * std::f32::consts::TAU;
                let r = 2.6_f32;
                let x = a.cos() * r;
                let y = a.sin() * r;
                // Position is the flame centre; size_y/2 lifts so the base
                // hovers just above the ground.
                positions.push([x, y, 0.8]);
                sizes.push(1.6);
            }
            // A few smaller flickers in the middle for visual density.
            let mut seed = 0xf1a3_u64;
            let mut rand = move || -> f32 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32) / (u32::MAX as f32)
            };
            for _ in 0..8 {
                let r = rand() * 1.5;
                let a = rand() * std::f32::consts::TAU;
                positions.push([a.cos() * r, a.sin() * r, 0.55]);
                sizes.push(0.8 + rand() * 0.4);
            }

            let mut item = SpriteItem::default();
            item.texture_id = Some(app.sprite_state.flame_tex);
            item.positions = positions;
            item.sizes = sizes;
            item.default_colour = [1.0, 0.85, 0.55, 1.0];
            item.size_mode = SpriteSizeMode::WorldSpace;
            item.blend = SpriteBlend::Additive;
            item.depth_write = false;
            item.orientation = SpriteOrientation::AxisLocked;
            item.axis = [0.0, 0.0, 1.0];
            vec![item]
        }
    }
}

/// One `RibbonItem` per emitter so each trail keeps its own RGB tint while
/// sharing the same blend mode and fade shape. Returns an empty list outside
/// the `Trails` sub-mode.
pub(crate) fn trail_ribbon_items(app: &App) -> Vec<RibbonItem> {
    if !app.sprite_state.built || app.sprite_state.sub_mode != SpriteSubMode::Trails {
        return Vec::new();
    }

    let blend = app.sprite_state.trail_blend;
    let head_width = app.sprite_state.trail_width;

    app.sprite_state
        .trails
        .iter()
        .filter_map(|emitter| {
            // A ribbon needs at least two points to define a tangent.
            if emitter.history.len() < 2 {
                return None;
            }
            let n = emitter.history.len();
            let mut colours = Vec::with_capacity(n);
            let mut widths = Vec::with_capacity(n);
            for k in 0..n {
                // k=0 is the oldest point (tail), k=n-1 is the newest (head).
                let t = k as f32 / (n - 1) as f32;
                let alpha = t * t;
                colours.push([emitter.colour[0], emitter.colour[1], emitter.colour[2], alpha]);
                widths.push(head_width * t);
            }
            let mut item = RibbonItem::default();
            item.positions = emitter.history.clone();
            item.strip_lengths = vec![n as u32];
            item.width = head_width;
            item.width_attribute = Some(widths);
            item.colour_attribute = colours;
            item.blend = blend;
            if app.sprite_state.trail_streak_enabled {
                item.texture_id = Some(app.sprite_state.streak_tex);
            }
            // Trails read more like emitter passes than solid surfaces, so
            // suppress lighting; the per-vertex RGBA carries all the visible
            // information.
            item.settings.unlit = true;
            Some(item)
        })
        .collect()
}
