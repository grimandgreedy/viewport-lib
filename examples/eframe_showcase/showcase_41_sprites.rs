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
    LightKind, LightSource, LightingSettings, MeshId, PolylineItem, SceneRenderItem, SpriteItem,
    SpriteSizeMode, ViewportRenderer, primitives,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpriteSubMode {
    Placed,
    Particles,
    Atlas,
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

    app.sprite_state.sphere_id = sphere_id;
    app.sprite_state.sprite_tex = sprite_tex;
    app.sprite_state.placed_positions = placed_positions;
    app.sprite_state.particles = spawn_burst(2000);
    app.sprite_state.glow_tex = glow_tex;
    app.sprite_state.atlas_tex = atlas_tex;
    app.sprite_state.atlas_positions = atlas_positions;
    app.sprite_state.atlas_frame = 0;
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
    if !app.sprite_state.built || app.sprite_state.sub_mode != SpriteSubMode::Placed {
        return vec![];
    }
    let mut item = SceneRenderItem::default();
    item.mesh_id = app.sprite_state.sphere_id;
    item.material.base_colour = [0.3, 0.45, 0.7];
    item.material.specular = 0.2;
    vec![item]
}

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------

pub(crate) fn sprite_lighting() -> LightingSettings {
    LightingSettings {
        lights: vec![LightSource {
            kind: LightKind::Directional {
                direction: [0.4, 0.7, 0.6],
            },
            colour: [1.0, 1.0, 1.0],
            intensity: 0.8,
        }],
        shadows_enabled: false,
        hemisphere_intensity: 0.4,
        sky_colour: [0.85, 0.9, 1.0],
        ground_colour: [0.4, 0.4, 0.5],
        ..LightingSettings::default()
    }
}
