//! Showcase 23: Performance at Scale : build + controls.
//!
//! Demonstrates GPU-driven instanced rendering and culling:
//! - 1 000 000 boxes (100x100x100 grid) sharing a single mesh
//! - GPU-driven culling via compute cull pass + indirect draw
//! - Toggle GPU culling on/off to compare paths live
//! - Full FrameStats readout: CPU/GPU timings, culling state, draw counts
//! - BVH-accelerated picking: click to select objects

use std::sync::atomic::{AtomicU32, Ordering};

use eframe::egui;
use viewport_lib::{
    Aabb, ItemSettings, FrameStats, Material, MeshId, PickAccelerator, SceneRenderItem,
    scene::Scene, selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct PerfState {
    pub scene: Scene,
    pub selection: Selection,
    pub pick_accelerator: Option<PickAccelerator>,
    pub mesh: Option<MeshId>,
    pub last_stats: FrameStats,
    pub total_objects: u32,
    pub scene_items_cache: std::sync::Arc<[SceneRenderItem]>,
    pub scene_items_version: (u64, u64),
    pub built: bool,
    pub gpu_culling: bool,
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
            mesh: None,
            last_stats: FrameStats::default(),
            total_objects: 0,
            scene_items_cache: std::sync::Arc::from([]),
            scene_items_version: (u64::MAX, u64::MAX),
            built: false,
            gpu_culling: true,
            build_rx: None,
            build_progress: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Background scene build
// ---------------------------------------------------------------------------

/// Build the 1M-box scene on a background thread.
///
/// The mesh is already uploaded on the main thread before this is called.
/// `mesh_aabb` is passed in so the BVH closure doesn't need GPU access.
/// `progress` is incremented every 10 000 objects so the main thread can
/// display a live loading bar.
pub(crate) fn build_perf_scene_threaded(
    mesh: MeshId,
    mesh_aabb: Option<Aabb>,
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
    let (nx, ny, nz) = (100u32, 100u32, 100u32);
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
                let colour = colours[count as usize % colours.len()];
                let mat = Material::from_colour(colour);
                let id = scene.add(Some(mesh), transform, mat);
                let mut appearance = ItemSettings::default();
                appearance.unlit = true;
                scene.set_appearance(id, appearance);
                count += 1;
                if count % 10_000 == 0 {
                    progress.store(count, Ordering::Relaxed);
                }
            }
        }
    }
    progress.store(count, Ordering::Relaxed);

    let pick_acc =
        PickAccelerator::build_from_scene(&scene, |mid| if mid == mesh { mesh_aabb } else { None });

    (scene, pick_acc)
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
        let gpu_culled = app.perf_state.total_objects.saturating_sub(gpu_vis);
        perf_stat_row(ui, "Visible (GPU)", &format_count(gpu_vis));
        perf_stat_row(ui, "Culled (GPU)", &format_count(gpu_culled));
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
