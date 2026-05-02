//! Showcase 3: Performance at Scale : build + controls.
//!
//! Demonstrates GPU-driven instanced rendering and culling:
//! - 1 000 000 boxes (100x100x100 grid) sharing a single mesh
//! - GPU-driven culling via compute cull pass + indirect draw
//! - Toggle GPU culling on/off to compare paths live
//! - Full FrameStats readout: CPU/GPU timings, culling state, draw counts
//! - BVH-accelerated picking: click to select objects

use eframe::egui;
use viewport_lib::{FrameStats, Material, PickAccelerator, ViewportRenderer, scene::Scene};

use crate::App;

impl App {
    /// Build a large grid of boxes (all sharing one mesh) to demonstrate GPU instancing.
    pub(crate) fn build_perf_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.perf_scene = Scene::new();
        self.perf_selection.clear();

        let mesh = self.upload_box(renderer);
        self.perf_mesh = Some(mesh);

        let spacing = 2.5_f32;
        let colors: [[f32; 3]; 6] = [
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.3],
            [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3],
            [0.9, 0.5, 0.2],
            [0.5, 0.3, 0.9],
        ];

        let (nx, ny, nz) = (100, 100, 100);
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
                    let color = colors[count as usize % colors.len()];
                    let mat = Material::from_color(color);
                    let name = format!("Perf {count}");
                    self.perf_scene.add_named(&name, Some(mesh), transform, mat);
                    count += 1;
                }
            }
        }

        self.perf_total_objects = count;

        let resources = renderer.resources();
        self.pick_accelerator = Some(PickAccelerator::build_from_scene(&self.perf_scene, |mid| {
            resources.mesh(mid).map(|m| m.aabb)
        }));

        self.perf_built = true;
    }

    /// Side-panel controls for Showcase 3.
    pub(crate) fn perf_controls(&mut self, ui: &mut egui::Ui) {
        let s = self.last_stats;

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
        ui.checkbox(&mut self.perf_gpu_culling, "Enable GPU-driven culling");
        ui.add_space(4.0);

        // --- Culling stats ---
        ui.separator();
        ui.heading("Culling");
        perf_stat_row(ui, "Total instances", &format_count(self.perf_total_objects));
        perf_stat_row(ui, "Visible", &format_count(s.visible_objects));
        perf_stat_row(
            ui,
            "Culled",
            &format_count(self.perf_total_objects.saturating_sub(s.visible_objects)),
        );
        ui.add_space(4.0);

        // --- Draw path ---
        ui.separator();
        ui.heading("Draw Path");
        perf_stat_row(ui, "Draw calls", &format_count(s.draw_calls));
        perf_stat_row(ui, "Instanced batches", &format_count(s.instanced_batches));
        perf_stat_row(ui, "Shadow draw calls", &format_count(s.shadow_draw_calls));
        perf_stat_row(ui, "Triangles submitted", &format_large(s.triangles_submitted));
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
            "—".into()
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
        perf_stat_row(ui, "Budget missed", if s.missed_budget { "yes" } else { "no" });
        perf_stat_row(ui, "Upload bytes", &format_bytes(s.upload_bytes));
        ui.add_space(4.0);

        // --- Picking ---
        ui.separator();
        ui.label("Click objects to select them.");
        if ui.button("Clear Selection").clicked() {
            self.perf_selection.clear();
        }
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

fn format_count(n: u32) -> String {
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
