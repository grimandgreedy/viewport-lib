//! Showcase 36: Playback Runtime Control
//!
//! Demonstrates the full runtime-control layer against a deliberate workload:
//! - Deforming mesh: NxNxL sine-wave grid re-uploaded every frame in Playback mode
//!   (L=1 is a flat 2D sheet; L>1 stacks multiple sheets for heavier upload load)
//! - Static instanced box grid: independent draw-call load
//! - Controls: RuntimeMode, PerformancePolicy sliders/toggles, load selectors
//! - Stats: live FrameStats readout with 60-frame sparkline

use std::collections::VecDeque;

use eframe::egui;
use viewport_lib::{
    BackfacePolicy, FrameStats, Material, MeshData, MeshId, PerformancePolicy, QualityPreset,
    RuntimeMode, SceneRenderItem, ViewportRenderer, scene::Scene, selection::Selection,
};

use crate::App;

// ---------------------------------------------------------------------------
// Load selectors
// ---------------------------------------------------------------------------

pub(crate) const GRID_RESOLUTIONS: &[(usize, &str)] = &[
    (50, "50x50"),
    (100, "100x100"),
    (200, "200x200"),
    (300, "300x300"),
];

pub(crate) const GRID_LAYERS: &[(usize, &str)] = &[(1, "1 (2D)"), (3, "3"), (5, "5"), (10, "10")];

pub(crate) const INSTANCE_COUNTS: &[(usize, &str)] = &[
    (100, "100"),
    (1000, "1 000"),
    (10000, "10 000"),
    (25000, "25 000"),
    (50000, "50 000"),
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct PbState {
    pub built: bool,
    pub mode: RuntimeMode,
    pub policy: PerformancePolicy,
    pub manual_render_scale: f32,
    pub grid_resolution: usize,
    pub last_grid_resolution: usize,
    pub grid_layers: usize,
    pub last_grid_layers: usize,
    pub instance_count: usize,
    pub time: f32,
    pub mesh_id: Option<MeshId>,
    pub static_mesh_id: Option<MeshId>,
    pub scene: Scene,
    pub last_stats: FrameStats,
    pub stats_history: VecDeque<f32>,
    pub upload_ms: f32,
}

impl Default for PbState {
    fn default() -> Self {
        Self {
            built: false,
            mode: RuntimeMode::Interactive,
            policy: PerformancePolicy::default(),
            manual_render_scale: 1.0,
            grid_resolution: 50,
            last_grid_resolution: 0,
            grid_layers: 1,
            last_grid_layers: 0,
            instance_count: 1000,
            time: 0.0,
            mesh_id: None,
            static_mesh_id: None,
            scene: Scene::new(),
            last_stats: FrameStats::default(),
            stats_history: VecDeque::new(),
            upload_ms: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Scene construction (called once from ensure_scene_built)
// ---------------------------------------------------------------------------

pub(crate) fn build_pb_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    // Upload placeholder deforming mesh (replaced each frame in Playback mode).
    let deform_mesh = build_sine_grid(app.pb_state.grid_resolution, app.pb_state.grid_layers, 0.0);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &deform_mesh)
        .expect("pb deforming mesh upload");
    app.pb_state.mesh_id = Some(mesh_id);
    app.pb_state.last_grid_resolution = app.pb_state.grid_resolution;
    app.pb_state.last_grid_layers = app.pb_state.grid_layers;

    // Upload one shared box mesh for the static grid.
    let box_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &app.box_mesh_data)
        .expect("pb static box mesh upload");
    app.pb_state.static_mesh_id = Some(box_id);

    rebuild_pb_static_scene(app);
    app.pb_state.built = true;
}

/// Rebuild the static instanced box scene to match `pb_instance_count`.
/// Call whenever pb_instance_count changes (and after pb_static_mesh_id is set).
pub(crate) fn rebuild_pb_static_scene(app: &mut App) {
    app.pb_state.scene = Scene::new();
    let Some(mid) = app.pb_state.static_mesh_id else {
        return;
    };
    let n = app.pb_state.instance_count;
    let per_row = (n as f32).sqrt().ceil() as usize;
    let spacing = 2.5_f32;
    let mat = Material::from_colour([0.4, 0.55, 0.85]);
    for i in 0..n {
        let col = i % per_row;
        let row = i / per_row;
        let x = (col as f32 - per_row as f32 * 0.5) * spacing;
        let y = (row as f32 - per_row as f32 * 0.5) * spacing;
        let z = -4.0; // below the deforming grid
        let t = glam::Mat4::from_translation(glam::Vec3::new(x, y, z));
        app.pb_state.scene.add(Some(mid), t, mat);
    }
}

// ---------------------------------------------------------------------------
// Per-frame scene items
// ---------------------------------------------------------------------------

pub(crate) fn pb_scene_items(app: &mut App) -> Vec<SceneRenderItem> {
    let mut items = Vec::new();

    // Deforming mesh: single item placed at origin.
    if let Some(mid) = app.pb_state.mesh_id {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mid;
        let mut mat = Material::from_colour([0.9, 0.55, 0.2]);
        mat.backface_policy = BackfacePolicy::Identical;
        item.material = mat;
        item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
        items.push(item);
    }

    // Static instanced boxes.
    let empty_sel = Selection::new();
    items.extend(app.pb_state.scene.collect_render_items(&empty_sel));

    items
}

// ---------------------------------------------------------------------------
// Deforming grid mesh
// ---------------------------------------------------------------------------

/// Build a `resolution x resolution x layers` sine-wave grid mesh at time `t`.
///
/// `layers = 1` produces a single flat 2D sheet. `layers > 1` stacks multiple
/// sheets along Z, each with a phase offset of `layer * 0.5` radians, to
/// increase upload cost without changing the visual footprint much.
///
/// Each vertex has an analytical normal derived from the gradient of the
/// displacement field, so normals update correctly every frame.
pub(crate) fn build_sine_grid(resolution: usize, layers: usize, t: f32) -> MeshData {
    let n = resolution.max(2);
    let l = layers.max(1);
    let half = (n as f32 - 1.0) * 0.5;
    let scale = 4.0_f32; // half-extent of each sheet in world units
    let layer_spacing = 2.5_f32;

    let verts_per_layer = (n + 1) * (n + 1);
    let tris_per_layer = n * n * 6;
    let mut positions = Vec::with_capacity(verts_per_layer * l);
    let mut normals = Vec::with_capacity(verts_per_layer * l);
    let mut indices = Vec::with_capacity(tris_per_layer * l);

    for layer in 0..l {
        let phase = layer as f32 * 0.5;
        let z_offset = (layer as f32 - (l as f32 - 1.0) * 0.5) * layer_spacing;
        let vert_base = (layer * verts_per_layer) as u32;
        let stride = (n + 1) as u32;

        for row in 0..=n {
            for col in 0..=n {
                let x = (col as f32 - half) / half * scale;
                let y = (row as f32 - half) / half * scale;

                let ax = x * 1.2 + t * 2.0 + phase;
                let ay = y * 0.9 + t * 1.7 + phase;
                let z = z_offset + ax.sin() * 0.6 + ay.sin() * 0.4;

                positions.push([x, y, z]);

                // Gradient: dz/dx, dz/dy.
                let dzdx = ax.cos() * 1.2 * 0.6;
                let dzdy = ay.cos() * 0.9 * 0.4;
                let nv = glam::Vec3::new(-dzdx, -dzdy, 1.0).normalize();
                normals.push([nv.x, nv.y, nv.z]);
            }
        }

        for row in 0..n as u32 {
            for col in 0..n as u32 {
                let tl = vert_base + row * stride + col;
                let tr = tl + 1;
                let bl = tl + stride;
                let br = bl + 1;
                indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
            }
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
            app.pb_state.last_stats = renderer.last_frame_stats();
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
                if ui.radio(app.pb_state.mode == mode, label).clicked() {
                    app.pb_state.mode = mode;
                }
            }
        });

        ui.separator();

        // --- Performance policy ---
        ui.label("Target FPS:");
        ui.horizontal(|ui| {
            if ui
                .radio(app.pb_state.policy.target_fps.is_none(), "Uncapped")
                .clicked()
            {
                app.pb_state.policy.target_fps = None;
            }
            for fps in [60.0_f32, 30.0, 15.0] {
                if ui
                    .radio(
                        app.pb_state.policy.target_fps == Some(fps),
                        format!("{fps:.0}"),
                    )
                    .clicked()
                {
                    app.pb_state.policy.target_fps = Some(fps);
                }
            }
        });

        ui.add(egui::Checkbox::new(
            &mut app.pb_state.policy.allow_dynamic_resolution,
            "Dynamic resolution",
        ));

        // Scale slider: hidden when a preset is active because the preset owns the
        // scale bounds. Show the preset's range as informational text instead.
        match app.pb_state.policy.preset {
            Some(preset) => {
                let (lo, hi) = match preset {
                    QualityPreset::High => (1.0_f32, 1.0_f32),
                    QualityPreset::Medium => (0.75, 1.0),
                    QualityPreset::Low => (0.5, 0.75),
                };
                if (lo - hi).abs() < 0.001 {
                    ui.label(format!("Scale: {lo:.2} (preset)"));
                } else {
                    ui.label(format!("Scale: {lo:.2} - {hi:.2} (preset)"));
                }
            }
            None => {
                if app.pb_state.policy.allow_dynamic_resolution {
                    ui.add(
                        egui::Slider::new(&mut app.pb_state.policy.min_render_scale, 0.25..=1.0)
                            .text("Min scale")
                            .step_by(0.05),
                    );
                } else {
                    ui.add(
                        egui::Slider::new(&mut app.pb_state.manual_render_scale, 0.25..=1.0)
                            .text("Render scale")
                            .step_by(0.05),
                    );
                }
            }
        }

        // --- Quality preset ---
        // High/Medium/Low configure scale bounds and degradation flags as a unit.
        // Custom exposes the individual knobs directly.
        ui.label("Quality preset:");
        ui.horizontal(|ui| {
            for (label, value) in [
                ("High", Some(QualityPreset::High)),
                ("Medium", Some(QualityPreset::Medium)),
                ("Low", Some(QualityPreset::Low)),
                ("Custom", None),
            ] {
                if ui
                    .radio(app.pb_state.policy.preset == value, label)
                    .clicked()
                {
                    app.pb_state.policy.preset = value;
                }
            }
        });

        // Individual degradation knobs: enabled only in Custom mode.
        let custom = app.pb_state.policy.preset.is_none();
        ui.add_enabled(
            custom,
            egui::Checkbox::new(
                &mut app.pb_state.policy.allow_shadow_reduction,
                "Allow shadow reduction",
            ),
        );
        ui.add_enabled(
            custom,
            egui::Checkbox::new(
                &mut app.pb_state.policy.allow_volume_quality_reduction,
                "Allow volume quality reduction",
            ),
        );
        ui.add_enabled(
            custom,
            egui::Checkbox::new(
                &mut app.pb_state.policy.allow_effect_throttling,
                "Allow effect throttling",
            ),
        );

        ui.separator();

        // --- Load controls ---
        // Grid resolution drives upload cost: the mesh is rebuilt and re-uploaded
        // every frame in Playback mode. At high resolutions this can dominate
        // frame time and quality controls will have little visible effect.
        ui.label("Grid (upload load):");
        for &(res, label) in GRID_RESOLUTIONS {
            if ui
                .radio(app.pb_state.grid_resolution == res, label)
                .clicked()
            {
                app.pb_state.grid_resolution = res;
            }
        }

        // Layers stacks multiple NxN sheets to multiply upload cost.
        ui.label("Layers:");
        for &(layers, label) in GRID_LAYERS {
            if ui
                .radio(app.pb_state.grid_layers == layers, label)
                .clicked()
            {
                app.pb_state.grid_layers = layers;
            }
        }

        // Instance count drives render cost: a single instanced draw call, but
        // more geometry and shadow work for the GPU to process.
        let old_count = app.pb_state.instance_count;
        ui.label("Instances (render load):");
        for &(count, label) in INSTANCE_COUNTS {
            if ui
                .radio(app.pb_state.instance_count == count, label)
                .clicked()
            {
                app.pb_state.instance_count = count;
            }
        }
        if app.pb_state.instance_count != old_count && app.pb_state.built {
            rebuild_pb_static_scene(app);
        }

        ui.separator();

        // --- Stats readout ---
        let s = app.pb_state.last_stats;
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

                ui.label("upload_ms:");
                ui.label(format!("{:.2} ms", app.pb_state.upload_ms));
                ui.end_row();

                // Controller signal: gpu_frame_ms when available, else total_frame_ms.
                ui.label("controller:");
                match s.gpu_frame_ms {
                    Some(g) => ui.label(format!("gpu {g:.2} ms")),
                    None => ui.label(format!("wall-clock {:.2} ms", s.total_frame_ms)),
                };
                ui.end_row();

                // Degradation tier derived from FrameStats flags.
                let tier = if s.effects_throttled {
                    "effects"
                } else if s.volume_quality_reduced {
                    "volumes"
                } else if s.shadows_skipped {
                    "shadows"
                } else if s.render_scale < 0.999 {
                    "scale"
                } else {
                    "none"
                };
                ui.label("tier:");
                ui.label(tier);
                ui.end_row();
            });

        // Render scale bar.
        ui.horizontal(|ui| {
            ui.label(format!("scale: {:.2}", s.render_scale));
            let bar_w = 60.0_f32;
            let bar_h = 10.0_f32;
            let (rect, _) = ui.allocate_exact_size(egui::vec2(bar_w, bar_h), egui::Sense::empty());
            let painter = ui.painter();
            painter.rect_filled(rect, 2.0, egui::Color32::from_gray(40));
            let fill_w = (bar_w * s.render_scale).min(bar_w);
            let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, bar_h));
            painter.rect_filled(fill_rect, 2.0, egui::Color32::from_rgb(70, 160, 70));
        });

        // missed_budget dot.
        ui.horizontal(|ui| {
            ui.label("missed_budget:");
            let colour = if s.missed_budget {
                egui::Color32::from_rgb(220, 55, 55)
            } else {
                egui::Color32::from_gray(90)
            };
            let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::empty());
            ui.painter().circle_filled(rect.center(), 5.0, colour);
        });

        ui.separator();

        // Per-flag degradation dots.
        // `active` reads directly from FrameStats so the dot reflects what
        // actually fired, not an approximation based on missed_budget alone.
        // `flag_on` reflects the effective policy (preset-derived when a preset is set).
        ui.label("Degradation:");
        let (eff_allow_shadows, eff_allow_volumes, eff_allow_effects) =
            match app.pb_state.policy.preset {
                Some(QualityPreset::High) => (false, false, false),
                Some(QualityPreset::Medium) => (true, false, true),
                Some(QualityPreset::Low) => (true, true, true),
                None => (
                    app.pb_state.policy.allow_shadow_reduction,
                    app.pb_state.policy.allow_volume_quality_reduction,
                    app.pb_state.policy.allow_effect_throttling,
                ),
            };
        for (label, flag_on, active) in [
            ("shadows", eff_allow_shadows, s.shadows_skipped),
            ("volumes", eff_allow_volumes, s.volume_quality_reduced),
            ("effects", eff_allow_effects, s.effects_throttled),
        ] {
            let colour = if active {
                egui::Color32::from_rgb(220, 55, 55)
            } else if flag_on {
                egui::Color32::from_gray(140)
            } else {
                egui::Color32::from_gray(60)
            };
            ui.horizontal(|ui| {
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::empty());
                ui.painter().circle_filled(rect.center(), 5.0, colour);
                ui.label(label);
            });
        }

        ui.separator();

        // Sparkline: last 60 frames of total_frame_ms.
        ui.label("Frame time (60f):");
        draw_sparkline(
            ui,
            &app.pb_state.stats_history,
            app.pb_state.policy.target_fps,
        );
    });
}

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

fn draw_sparkline(ui: &mut egui::Ui, history: &VecDeque<f32>, target_fps: Option<f32>) {
    let w = 200.0_f32;
    let h = 50.0_f32;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::empty());
    let painter = ui.painter();

    painter.rect_filled(rect, 2.0, egui::Color32::from_gray(22));

    let budget_ms = target_fps.map(|fps| 1000.0 / fps).unwrap_or(f32::INFINITY);
    let max_ms =
        history
            .iter()
            .cloned()
            .fold(1.0_f32, f32::max)
            .max(if budget_ms < f32::INFINITY {
                budget_ms * 1.2
            } else {
                50.0
            });

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
