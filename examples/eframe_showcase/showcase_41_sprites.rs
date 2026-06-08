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
use viewport_lib::{
    ForceField, FrameData, GpuParticleSystemConfig, GpuParticleSystemId, GpuParticleSystemItem,
    LightKind, LightSource, LightingSettings, MeshId, MeshInstanceItem, ParticleRender,
    PolylineItem, RibbonItem, SceneRenderItem, SpawnShape, SpriteBlend, SpriteItem, SpriteSizeMode,
    VelocityDist, ViewportRenderer, primitives,
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
    };
    let particle_sys = renderer
        .resources_mut()
        .create_gpu_particle_system(&app.device, &app.queue, &particle_cfg);
    app.sprite_state.gpu_particle_system = Some(particle_sys);

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
    ui.horizontal(|ui| {
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
            "MeshParticles",
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
            ui.label("A flat sprite sheet sweeps horizontally through the sphere.");
            ui.label("`soft_particle_distance` ramps alpha to zero where the sprite meets opaque");
            ui.label("geometry, replacing the hard intersection line with a smooth fade.");
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
        SpriteSubMode::Soft | SpriteSubMode::MeshParticles | SpriteSubMode::GpuParticles => {
            app.sprite_state.demo_time += dt;
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
            // Lay a row of large glow sprites in a flat plane sweeping
            // horizontally through the central sphere. With
            // soft_particle_distance enabled, the alpha ramps to zero where
            // the sprite plane meets the sphere instead of cutting a hard
            // intersection line.
            let t = app.sprite_state.demo_time;
            let sweep_x = (t * 0.4).sin() * 2.5;
            let mut positions = Vec::with_capacity(16);
            let mut colours = Vec::with_capacity(16);
            for i in 0..16 {
                let z = -3.5 + i as f32 * 0.5;
                positions.push([sweep_x, 0.0, z]);
                colours.push([0.5, 0.8, 1.0, 0.9]);
            }
            let mut item = SpriteItem::default();
            item.texture_id = Some(app.sprite_state.glow_tex);
            item.positions = positions;
            item.colours = colours;
            item.default_size = 2.2;
            item.size_mode = SpriteSizeMode::WorldSpace;
            item.blend = SpriteBlend::AlphaBlend;
            item.depth_write = false;
            item.soft_particle_distance = Some(0.6);
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

pub(crate) fn sprite_lighting() -> LightingSettings {
    {
        let mut _t = LightingSettings::default();
        _t.lights = vec![{
            let mut _t = LightSource::default();
            _t.kind = LightKind::Directional {
                direction: [0.4, 0.7, 0.6],
            };
            _t.colour = [1.0, 1.0, 1.0];
            _t.intensity = 0.8;
            _t
        }];
        _t.shadows_enabled = false;
        _t.hemisphere_intensity = 0.4;
        _t.sky_colour = [0.85, 0.9, 1.0];
        _t.ground_colour = [0.4, 0.4, 0.5];
        _t
    }
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn submit_sprite_items(app: &App, fd: &mut FrameData, dt: f32) {
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
            // Trails read more like emitter passes than solid surfaces, so
            // suppress lighting; the per-vertex RGBA carries all the visible
            // information.
            item.settings.unlit = true;
            Some(item)
        })
        .collect()
}
