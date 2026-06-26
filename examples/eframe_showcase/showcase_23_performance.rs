//! Showcase 23: Performance at Scale : build + controls.
//!
//! Demonstrates GPU-driven instanced rendering and culling:
//! - 125 000 boxes (50x50x50 grid) sharing a single mesh
//! - GPU-driven culling via compute cull pass + indirect draw
//! - Toggle GPU culling on/off to compare paths live
//! - Optional HiZ occlusion culling: drop boxes hidden behind nearer ones
//! - Full FrameStats readout: CPU/GPU timings, culling state, draw counts
//! - BVH-accelerated picking: click to select objects

use std::sync::atomic::{AtomicU32, Ordering};

use eframe::egui;
use viewport_lib::{
    Aabb, FrameStats, ItemSettings, Material, MeshId, PickAccelerator, SceneRenderItem,
    scene::Scene, selection::Selection,
};

use crate::App;

/// Number of distinct textures generated for the box grid. Each box picks one
/// at random, so this is also the number of instanced batches the grid splits
/// into (boxes are batched by texture). Raise it for more variety at the cost
/// of more draw calls; set it to 0 to fall back to flat-coloured boxes. Change
/// it here and watch the "Instanced batches" and timing rows react.
pub(crate) const TEXTURE_POOL_SIZE: usize = 128;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct PerfState {
    pub scene: Scene,
    pub selection: Selection,
    pub pick_accelerator: Option<PickAccelerator>,
    /// Per-shape geometry (mesh index, positions, indices) for CPU picking,
    /// one entry per cycled shape.
    pub pick_geometry: Vec<(u64, Vec<[f32; 3]>, Vec<u32>)>,
    pub last_stats: FrameStats,
    pub total_objects: u32,
    pub scene_items_cache: std::sync::Arc<[SceneRenderItem]>,
    pub scene_items_version: (u64, u64),
    pub built: bool,
    pub gpu_culling: bool,
    pub occlusion_culling: bool,
    /// Receives the completed (Scene, PickAccelerator) from the background build thread.
    pub build_rx: Option<std::sync::mpsc::Receiver<(Scene, PickAccelerator)>>,
    /// Shared progress counter written by the background build thread (objects placed so far).
    pub build_progress: Option<std::sync::Arc<AtomicU32>>,
}

impl Default for PerfState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            selection: Selection::new(),
            pick_accelerator: None,
            pick_geometry: Vec::new(),
            last_stats: FrameStats::default(),
            total_objects: 0,
            scene_items_cache: std::sync::Arc::from([]),
            scene_items_version: (u64::MAX, u64::MAX),
            built: false,
            gpu_culling: true,
            occlusion_culling: false,
            build_rx: None,
            build_progress: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Background scene build
// ---------------------------------------------------------------------------

/// Build the 125K-box scene on a background thread.
///
/// The mesh is already uploaded on the main thread before this is called.
/// `mesh_aabb` is passed in so the BVH closure doesn't need GPU access.
/// `progress` is incremented every 10 000 objects so the main thread can
/// display a live loading bar.
pub(crate) fn build_perf_scene_threaded(
    meshes: Vec<(MeshId, Option<Aabb>)>,
    texture_pool: Vec<u64>,
    progress: &AtomicU32,
) -> (Scene, PickAccelerator) {
    let spacing = 2.5_f32;
    let colours: [[f32; 3]; 6] = [
        [0.9, 0.3, 0.3],
        [0.3, 0.9, 0.3],
        [0.3, 0.3, 0.9],
        [0.9, 0.9, 0.3],
        [0.9, 0.5, 0.2],
        [0.5, 0.3, 0.9],
    ];

    let mut scene = Scene::new();
    let (nx, ny, nz) = (50u32, 50u32, 50u32);
    let mut count = 0u32;
    for y in 0..ny {
        for z in 0..nz {
            for x in 0..nx {
                let pos = glam::Vec3::new(
                    (x as f32 - nx as f32 / 2.0) * spacing,
                    (z as f32 - nz as f32 / 2.0) * spacing,
                    (y as f32) * spacing,
                );
                let transform = glam::Mat4::from_translation(pos);
                // Cycle through the shapes (box, sphere, cylinder, spring).
                let mesh = meshes[count as usize % meshes.len()].0;
                let mat = if texture_pool.is_empty() {
                    // Pool size 0: fall back to the flat-colour boxes.
                    Material::flat(colours[count as usize % colours.len()])
                } else {
                    // Pick one pool texture at random. White base so the
                    // texture colours show through unmodulated.
                    let ti = hash_index(count, texture_pool.len());
                    let mut m = Material::flat([1.0, 1.0, 1.0]);
                    m.texture_id = Some(texture_pool[ti]);
                    m
                };
                let id = scene.add(Some(mesh), transform, mat);
                let mut appearance = ItemSettings::default();
                appearance.unlit = false;
                scene.set_appearance(id, appearance);
                count += 1;
                if count % 10_000 == 0 {
                    progress.store(count, Ordering::Relaxed);
                }
            }
        }
    }
    progress.store(count, Ordering::Relaxed);

    let pick_acc = PickAccelerator::build_from_scene(&scene, |mid| {
        meshes.iter().find(|(m, _)| *m == mid).and_then(|(_, a)| *a)
    });

    (scene, pick_acc)
}

/// The shapes the box grid cycles through, each sized to fit a 1x1x1 box
/// (diameter 1, height 1, centred on the origin).
pub(crate) fn shape_meshes() -> Vec<viewport_lib::resources::MeshData> {
    use viewport_lib::primitives;
    // Spring: solve coil-tube radius so the total height is exactly 1, then set
    // the helix radius so the outer diameter is exactly 1.
    let turns = 3.0_f32;
    let coil = 1.0 / (2.5 * turns + 2.0);
    vec![
        primitives::cube(1.0),
        primitives::sphere(0.5, 20, 14),
        primitives::cylinder(0.5, 1.0, 24),
        primitives::spring(0.5 - coil, coil, turns, 10),
    ]
}

// ---------------------------------------------------------------------------
// Texture pool
// ---------------------------------------------------------------------------

/// Pick a pool index from a box index with a cheap integer hash, so the random
/// texture assignment is varied but reproducible (no rng dependency).
fn hash_index(seed: u32, modulo: usize) -> usize {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9e37_79b9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    (x as usize) % modulo
}

/// Generate one 64x64 RGBA texture for pool slot `index`. Hue is spread around
/// the wheel and the pattern (checker / stripes / gradient / rings) varies by
/// index so neighbouring slots look clearly different.
pub(crate) fn make_box_texture(index: usize) -> (u32, Vec<u8>) {
    const SIZE: u32 = 64;
    let n = TEXTURE_POOL_SIZE.max(1) as f32;
    // Golden-ratio hue stride keeps successive slots far apart on the wheel.
    let hue = (index as f32 / n + 0.618_034 * index as f32).fract();
    let c0 = hsv_to_rgb(hue, 0.65, 0.95);
    let c1 = hsv_to_rgb((hue + 0.5).fract(), 0.70, 0.55);
    let pattern = index % 4;

    let mut data = vec![0u8; (SIZE * SIZE * 4) as usize];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let u = x as f32 / SIZE as f32;
            let v = y as f32 / SIZE as f32;
            let t = match pattern {
                0 => (((x / 8) % 2) ^ ((y / 8) % 2)) as f32, // checker
                1 => ((x / 6) % 2) as f32,                   // stripes
                2 => (u + v) * 0.5,                          // diagonal gradient
                _ => {
                    let (dx, dy) = (u - 0.5, v - 0.5);
                    let d = (dx * dx + dy * dy).sqrt() * 5.0;
                    if d.fract() < 0.5 { 0.0 } else { 1.0 } // rings
                }
            };
            let col = [
                lerp(c0[0], c1[0], t),
                lerp(c0[1], c1[1], t),
                lerp(c0[2], c1[2], t),
            ];
            let i = ((y * SIZE + x) * 4) as usize;
            data[i] = (col[0] * 255.0) as u8;
            data[i + 1] = (col[1] * 255.0) as u8;
            data[i + 2] = (col[2] * 255.0) as u8;
            data[i + 3] = 255;
        }
    }
    (SIZE, data)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match (i as i32).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_performance(app: &mut App, ui: &mut egui::Ui) {
    let s = app.perf_state.last_stats;

    // --- GPU culling toggle ---
    ui.heading("GPU-Driven Culling");
    let culling_label = if s.gpu_culling_active {
        egui::RichText::new("Active").color(egui::Color32::from_rgb(100, 220, 100))
    } else {
        egui::RichText::new("Disabled").color(egui::Color32::from_rgb(220, 120, 80))
    };
    ui.horizontal(|ui| {
        ui.label("Status:");
        ui.label(culling_label);
    });
    ui.checkbox(&mut app.perf_state.gpu_culling, "Enable GPU-driven culling");
    ui.add_enabled(
        app.perf_state.gpu_culling,
        egui::Checkbox::new(
            &mut app.perf_state.occlusion_culling,
            "Enable HiZ occlusion culling",
        ),
    );
    ui.small("Drops boxes hidden behind nearer ones, on top of the frustum cull.");
    ui.add_space(4.0);

    // --- Culling stats ---
    ui.separator();
    ui.heading("Culling");
    perf_stat_row(
        ui,
        "Total instances",
        &format_count(app.perf_state.total_objects),
    );
    if s.gpu_culling_active {
        // GPU readback gives the exact post-cull count (one frame lag).
        let gpu_vis = s.gpu_visible_instances.unwrap_or(s.visible_objects);
        if app.perf_state.occlusion_culling {
            // Split the cull into its two stages so the occlusion contribution
            // is visible: total -> frustum survivors -> drawn.
            let total = s.gpu_culled_total.unwrap_or(app.perf_state.total_objects);
            let frustum_vis = s.gpu_frustum_visible.unwrap_or(gpu_vis);
            let frustum_culled = total.saturating_sub(frustum_vis);
            let occlusion_culled = frustum_vis.saturating_sub(gpu_vis);
            perf_stat_row(ui, "Frustum culled", &format_count(frustum_culled));
            perf_stat_row(ui, "Occlusion culled", &format_count(occlusion_culled));
            perf_stat_row(ui, "Visible (GPU)", &format_count(gpu_vis));
        } else {
            let gpu_culled = app.perf_state.total_objects.saturating_sub(gpu_vis);
            perf_stat_row(ui, "Visible (GPU)", &format_count(gpu_vis));
            perf_stat_row(ui, "Culled (GPU)", &format_count(gpu_culled));
        }
    } else {
        perf_stat_row(ui, "Visible (CPU)", &format_count(s.visible_objects));
        perf_stat_row(
            ui,
            "Culled (CPU)",
            &format_count(
                app.perf_state
                    .total_objects
                    .saturating_sub(s.visible_objects),
            ),
        );
    }
    ui.add_space(4.0);

    // --- Draw path ---
    ui.separator();
    ui.heading("Draw Path");
    perf_stat_row(ui, "Draw calls", &format_count(s.draw_calls));
    perf_stat_row(ui, "Instanced batches", &format_count(s.instanced_batches));
    perf_stat_row(ui, "Shadow draw calls", &format_count(s.shadow_draw_calls));
    perf_stat_row(
        ui,
        "Triangles submitted",
        &format_large(s.triangles_submitted),
    );
    ui.add_space(4.0);

    // --- Timings ---
    ui.separator();
    ui.heading("Timings");
    perf_stat_row(ui, "CPU prepare", &format!("{:.2} ms", s.cpu_prepare_ms));
    perf_stat_row(
        ui,
        "GPU scene",
        &s.gpu_frame_ms
            .map(|ms| format!("{ms:.2} ms"))
            .unwrap_or_else(|| "n/a".into()),
    );
    // GPU cull dispatch cost. With one mesh + flat materials the whole grid is a
    // single instanced batch, so every visible instance contends on one atomic
    // counter; this row shows whether the cull pass is paying for itself.
    let cull_ms = s.gpu_breakdown.cull_ms;
    perf_stat_row(
        ui,
        "GPU cull",
        &if s.gpu_culling_active && cull_ms > 0.0 {
            format!("{cull_ms:.2} ms")
        } else {
            "n/a".into()
        },
    );
    perf_stat_row(ui, "Frame total", &format!("{:.2} ms", s.total_frame_ms));
    let fps = if s.total_frame_ms > 0.0 {
        format!("{:.0}", 1000.0 / s.total_frame_ms)
    } else {
        "-".into()
    };
    perf_stat_row(ui, "FPS (approx)", &fps);
    ui.add_space(4.0);

    // --- Renderer state ---
    ui.separator();
    ui.heading("Renderer");
    perf_stat_row(
        ui,
        "Render scale",
        &format!("{:.0}%", s.render_scale * 100.0),
    );
    perf_stat_row(
        ui,
        "Budget missed",
        if s.missed_budget { "yes" } else { "no" },
    );
    perf_stat_row(ui, "Upload bytes", &format_bytes(s.upload_bytes));
    ui.add_space(4.0);

    // --- Picking ---
    ui.separator();
    ui.label("Click objects to select them.");
    if ui.button("Clear Selection").clicked() {
        app.perf_state.selection.clear();
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn perf_stat_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).weak());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(value).monospace());
        });
    });
}

pub(crate) fn format_count(n: u32) -> String {
    format_large(n as u64)
}

fn format_large(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn format_bytes(b: u64) -> String {
    if b >= 1024 * 1024 {
        format!("{:.1} MB", b as f64 / (1024.0 * 1024.0))
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}
