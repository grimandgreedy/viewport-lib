//! Showcase 36: Playback Runtime Control
//!
//! Demonstrates the full runtime-control layer against a deliberate workload:
//! - Deforming mesh: NxN sine-wave grid re-uploaded every frame in Playback mode
//! - Static instanced box grid: independent draw-call load
//! - Controls: RuntimeMode, PerformancePolicy sliders/toggles, load selectors
//! - Stats: live FrameStats readout with 60-frame sparkline

use std::collections::VecDeque;

use eframe::egui;
use viewport_lib::{
    BackfacePolicy, Material, MeshData, RuntimeMode, SceneRenderItem, ViewportRenderer,
    scene::Scene,
    selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// Load selectors
// ---------------------------------------------------------------------------

pub(crate) const GRID_RESOLUTIONS: &[(usize, &str)] = &[
    (50, "50x50"),
    (100, "100x100"),
    (200, "200x200"),
];

pub(crate) const INSTANCE_COUNTS: &[(usize, &str)] = &[
    (100, "100"),
    (1000, "1 000"),
    (10000, "10 000"),
];

// ---------------------------------------------------------------------------
// Scene construction (called once from ensure_scene_built)
// ---------------------------------------------------------------------------

pub(crate) fn build_pb_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    // Upload placeholder deforming mesh (replaced each frame in Playback mode).
    let deform_mesh = build_sine_grid(app.pb_grid_resolution, 0.0);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &deform_mesh)
        .expect("pb deforming mesh upload");
    app.pb_mesh_id = Some(mesh_id);
    app.pb_last_grid_resolution = app.pb_grid_resolution;

    // Upload one shared box mesh for the static grid.
    let box_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &app.box_mesh_data)
        .expect("pb static box mesh upload");
    app.pb_static_mesh_id = Some(box_id);

    rebuild_pb_static_scene(app);
    app.pb_built = true;
}

/// Rebuild the static instanced box scene to match `pb_instance_count`.
/// Call whenever pb_instance_count changes (and after pb_static_mesh_id is set).
pub(crate) fn rebuild_pb_static_scene(app: &mut App) {
    app.pb_scene = Scene::new();
    let Some(mid) = app.pb_static_mesh_id else {
        return;
    };
    let n = app.pb_instance_count;
    let per_row = (n as f32).sqrt().ceil() as usize;
    let spacing = 2.5_f32;
    let mat = Material::from_color([0.4, 0.55, 0.85]);
    for i in 0..n {
        let col = i % per_row;
        let row = i / per_row;
        let x = (col as f32 - per_row as f32 * 0.5) * spacing;
        let y = (row as f32 - per_row as f32 * 0.5) * spacing;
        let z = -4.0; // below the deforming grid
        let t = glam::Mat4::from_translation(glam::Vec3::new(x, y, z));
        app.pb_scene.add(Some(mid), t, mat);
    }
}

// ---------------------------------------------------------------------------
// Per-frame scene items
// ---------------------------------------------------------------------------

pub(crate) fn pb_scene_items(app: &mut App) -> Vec<SceneRenderItem> {
    let mut items = Vec::new();

    // Deforming mesh: single item placed at origin.
    if let Some(mid) = app.pb_mesh_id {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mid;
        let mut mat = Material::from_color([0.9, 0.55, 0.2]);
        mat.backface_policy = BackfacePolicy::Identical;
        item.material = mat;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        items.push(item);
    }

    // Static instanced boxes.
    let empty_sel = Selection::new();
    items.extend(app.pb_scene.collect_render_items(&empty_sel));

    items
}

// ---------------------------------------------------------------------------
// Deforming grid mesh
// ---------------------------------------------------------------------------

/// Build a `resolution x resolution` sine-wave grid mesh at time `t`.
///
/// Each vertex has an analytical normal derived from the gradient of the
/// displacement field, so normals update correctly every frame.
pub(crate) fn build_sine_grid(resolution: usize, t: f32) -> MeshData {
    let n = resolution.max(2);
    let half = (n as f32 - 1.0) * 0.5;
    let scale = 4.0_f32; // half-extent of the grid in world units

    let vert_count = (n + 1) * (n + 1);
    let mut positions = Vec::with_capacity(vert_count);
    let mut normals = Vec::with_capacity(vert_count);
    let mut indices = Vec::with_capacity(n * n * 6);

    for row in 0..=n {
        for col in 0..=n {
            let x = (col as f32 - half) / half * scale;
            let y = (row as f32 - half) / half * scale;

            let ax = x * 1.2 + t * 2.0;
            let ay = y * 0.9 + t * 1.7;
            let z = ax.sin() * 0.6 + ay.sin() * 0.4;

            positions.push([x, y, z]);

            // Gradient: dz/dx, dz/dy.
            let dzdx = ax.cos() * 1.2 * 0.6;
            let dzdy = ay.cos() * 0.9 * 0.4;
            let nv = glam::Vec3::new(-dzdx, -dzdy, 1.0).normalize();
            normals.push([nv.x, nv.y, nv.z]);
        }
    }

    let stride = (n + 1) as u32;
    for row in 0..n as u32 {
        for col in 0..n as u32 {
            let tl = row * stride + col;
            let tr = tl + 1;
            let bl = tl + stride;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

pub(crate) fn controls_pb(app: &mut App, ui: &mut egui::Ui, frame: &eframe::Frame) {
    // Refresh stats for display.
    if let Some(rs) = frame.wgpu_render_state() {
        let guard = rs.renderer.read();
        if let Some(renderer) = guard.callback_resources.get::<ViewportRenderer>() {
            app.pb_last_stats = renderer.last_frame_stats();
        }
    }

    egui::ScrollArea::vertical().show(ui, |ui| {
        // --- Runtime mode ---
        ui.label("Mode:");
        ui.horizontal(|ui| {
            for (label, mode) in [
                ("Interactive", RuntimeMode::Interactive),
                ("Playback", RuntimeMode::Playback),
                ("Paused", RuntimeMode::Paused),
            ] {
                if ui.radio(app.pb_mode == mode, label).clicked() {
                    app.pb_mode = mode;
                }
            }
        });

        ui.separator();

        // --- Performance policy ---
        ui.label("Target FPS:");
        ui.horizontal(|ui| {
            if ui
                .radio(app.pb_policy.target_fps.is_none(), "Uncapped")
                .clicked()
            {
                app.pb_policy.target_fps = None;
            }
            for fps in [60.0_f32, 30.0, 15.0] {
                if ui
                    .radio(app.pb_policy.target_fps == Some(fps), format!("{fps:.0}"))
                    .clicked()
                {
                    app.pb_policy.target_fps = Some(fps);
                }
            }
        });

        ui.add(egui::Checkbox::new(
            &mut app.pb_policy.allow_dynamic_resolution,
            "Dynamic resolution",
        ));

        if app.pb_policy.allow_dynamic_resolution {
            ui.add(
                egui::Slider::new(&mut app.pb_policy.min_render_scale, 0.25..=1.0)
                    .text("Min scale")
                    .step_by(0.05),
            );
        } else {
            ui.add(
                egui::Slider::new(&mut app.pb_manual_render_scale, 0.25..=1.0)
                    .text("Render scale")
                    .step_by(0.05),
            );
        }

        ui.add(egui::Checkbox::new(
            &mut app.pb_policy.allow_shadow_reduction,
            "Allow shadow reduction",
        ));
        ui.add(egui::Checkbox::new(
            &mut app.pb_policy.allow_volume_quality_reduction,
            "Allow volume quality reduction",
        ));
        ui.add(egui::Checkbox::new(
            &mut app.pb_policy.allow_effect_throttling,
            "Allow effect throttling",
        ));

        ui.separator();

        // --- Load controls ---
        ui.label("Grid resolution:");
        for &(res, label) in GRID_RESOLUTIONS {
            if ui.radio(app.pb_grid_resolution == res, label).clicked() {
                app.pb_grid_resolution = res;
            }
        }

        let old_count = app.pb_instance_count;
        ui.label("Instance count:");
        for &(count, label) in INSTANCE_COUNTS {
            if ui.radio(app.pb_instance_count == count, label).clicked() {
                app.pb_instance_count = count;
            }
        }
        if app.pb_instance_count != old_count && app.pb_built {
            rebuild_pb_static_scene(app);
        }

        ui.separator();

        // --- Stats readout ---
        let s = app.pb_last_stats;
        ui.label("Frame stats:");

        egui::Grid::new("pb_stats_grid")
            .num_columns(2)
            .spacing([8.0, 2.0])
            .show(ui, |ui| {
                ui.label("cpu_prepare:");
                ui.label(format!("{:.2} ms", s.cpu_prepare_ms));
                ui.end_row();

                ui.label("gpu_frame:");
                match s.gpu_frame_ms {
                    Some(g) => ui.label(format!("{:.2} ms", g)),
                    None => ui.label("n/a"),
                };
                ui.end_row();

                ui.label("total_frame:");
                ui.label(format!("{:.2} ms", s.total_frame_ms));
                ui.end_row();

                ui.label("fps:");
                let fps = if s.total_frame_ms > 0.0 {
                    1000.0 / s.total_frame_ms
                } else {
                    0.0
                };
                ui.label(format!("{:.1}", fps));
                ui.end_row();

                ui.label("upload_bytes:");
                ui.label(format!("{}", s.upload_bytes));
                ui.end_row();
            });

        // Render scale bar.
        ui.horizontal(|ui| {
            ui.label(format!("scale: {:.2}", s.render_scale));
            let bar_w = 60.0_f32;
            let bar_h = 10.0_f32;
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(bar_w, bar_h), egui::Sense::empty());
            let painter = ui.painter();
            painter.rect_filled(rect, 2.0, egui::Color32::from_gray(40));
            let fill_w = (bar_w * s.render_scale).min(bar_w);
            let fill_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, bar_h));
            painter.rect_filled(fill_rect, 2.0, egui::Color32::from_rgb(70, 160, 70));
        });

        // missed_budget dot.
        ui.horizontal(|ui| {
            ui.label("missed_budget:");
            let color = if s.missed_budget {
                egui::Color32::from_rgb(220, 55, 55)
            } else {
                egui::Color32::from_gray(90)
            };
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::empty());
            ui.painter().circle_filled(rect.center(), 5.0, color);
        });

        ui.separator();

        // Per-flag degradation dots.
        ui.label("Degradation:");
        for (label, flag_on, active) in [
            (
                "shadows",
                app.pb_policy.allow_shadow_reduction,
                app.pb_policy.allow_shadow_reduction && s.missed_budget,
            ),
            (
                "volumes",
                app.pb_policy.allow_volume_quality_reduction,
                app.pb_policy.allow_volume_quality_reduction && s.missed_budget,
            ),
            (
                "effects",
                app.pb_policy.allow_effect_throttling,
                app.pb_policy.allow_effect_throttling && s.missed_budget,
            ),
        ] {
            let color = if active {
                egui::Color32::from_rgb(220, 55, 55)
            } else if flag_on {
                egui::Color32::from_gray(140)
            } else {
                egui::Color32::from_gray(60)
            };
            ui.horizontal(|ui| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::empty());
                ui.painter().circle_filled(rect.center(), 5.0, color);
                ui.label(label);
            });
        }

        ui.separator();

        // Sparkline: last 60 frames of total_frame_ms.
        ui.label("Frame time (60f):");
        draw_sparkline(ui, &app.pb_stats_history, app.pb_policy.target_fps);
    });
}

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

fn draw_sparkline(
    ui: &mut egui::Ui,
    history: &VecDeque<f32>,
    target_fps: Option<f32>,
) {
    let w = 200.0_f32;
    let h = 50.0_f32;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::empty());
    let painter = ui.painter();

    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(22));

    let budget_ms = target_fps.map(|fps| 1000.0 / fps).unwrap_or(f32::INFINITY);
    let max_ms = history
        .iter()
        .cloned()
        .fold(1.0_f32, f32::max)
        .max(if budget_ms < f32::INFINITY { budget_ms * 1.2 } else { 50.0 });

    let to_y = |v: f32| rect.bottom() - (v / max_ms).clamp(0.0, 1.0) * rect.height();

    // Red shading above budget.
    if budget_ms < f32::INFINITY {
        let by = to_y(budget_ms).clamp(rect.top(), rect.bottom());
        if by > rect.top() {
            let over = egui::Rect::from_min_max(
                egui::pos2(rect.left(), rect.top()),
                egui::pos2(rect.right(), by),
            );
            painter.rect_filled(
                over,
                0.0,
                egui::Color32::from_rgba_unmultiplied(180, 40, 40, 35),
            );
        }
        // Budget line.
        painter.line_segment(
            [egui::pos2(rect.left(), by), egui::pos2(rect.right(), by)],
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(230, 80, 80, 180)),
        );
    }

    if history.len() < 2 {
        return;
    }

    let n = history.len();
    let points: Vec<egui::Pos2> = history
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = rect.left() + i as f32 / (n - 1).max(1) as f32 * rect.width();
            let y = to_y(v).clamp(rect.top(), rect.bottom());
            egui::pos2(x, y)
        })
        .collect();

    for w in points.windows(2) {
        painter.line_segment(
            [w[0], w[1]],
            egui::Stroke::new(1.5, egui::Color32::from_rgb(90, 190, 90)),
        );
    }
}
