//! Showcase 52: Level of detail.
//!
//! A field of instanced spheres recedes into the distance. Each instance picks
//! its own LOD level from how large it appears on screen, so near instances draw
//! the full mesh and far ones drop to cheaper meshes, all from one submitted
//! `MeshInstanceItem`. A short row of standalone `SceneRenderItem` spheres shows
//! the same on the surface-mesh path.
//!
//! The LOD work happens in the library: the demo uploads three icosphere
//! resolutions as one `LodGroup` and submits items that reference it. The
//! colourise option tints each instance by the level it lands on, computed with
//! the same `projected_screen_size` the renderer uses, so the bands line up with
//! the meshes actually drawn.

use crate::App;
use eframe::egui;
use viewport_lib::{
    Aabb, FrameData, FrameStats, LodGroupId, Material, MeshId, MeshInstanceItem, PickId,
    RenderCamera, SceneRenderItem, SpriteBlend, ViewportRenderer, primitives,
    projected_screen_size,
};

const GRID_N: usize = 18;
const GRID_SPACING: f32 = 3.0;

pub(crate) struct LodState {
    pub built: bool,
    pub group: Option<LodGroupId>,
    /// One mesh per level, kept so the LOD-off mode can force the full mesh.
    pub level_meshes: Vec<MeshId>,
    /// Lower screen-size bound per level, used to tint instances by level.
    pub thresholds: Vec<f32>,
    /// Local-space bounds shared by every level, for the tint sizing math.
    pub aabb: Aabb,
    pub lod_enabled: bool,
    pub colourise: bool,
    pub auto_dolly: bool,
    pub cull_enabled: bool,
    pub cull_below: f32,
    pub dolly_time: f32,
    pub generation: u64,
    pub last_stats: FrameStats,
}

impl Default for LodState {
    fn default() -> Self {
        Self {
            built: false,
            group: None,
            level_meshes: Vec::new(),
            thresholds: Vec::new(),
            aabb: Aabb::default(),
            lod_enabled: true,
            colourise: true,
            auto_dolly: true,
            cull_enabled: false,
            cull_below: 0.04,
            dolly_time: 0.0,
            generation: 0,
            last_stats: FrameStats::default(),
        }
    }
}

impl App {
    pub(crate) fn build_lod_scene(&mut self, renderer: &mut ViewportRenderer) {
        // Three detail levels of the same unit sphere. The screen-size
        // thresholds are tuned so the grid and the dolly span all three.
        let full = primitives::icosphere(1.0, 3);
        let aabb = full.compute_aabb();
        let thresholds = [0.22_f32, 0.10, 0.0];

        let level_data = [
            full,
            primitives::icosphere(1.0, 1),
            primitives::icosphere(1.0, 0),
        ];
        let mut level_meshes = Vec::with_capacity(level_data.len());
        for data in &level_data {
            let id = renderer
                .resources_mut()
                .upload_mesh_data(&self.device, data)
                .expect("lod level mesh");
            level_meshes.push(id);
        }
        let group = renderer
            .resources_mut()
            .register_lod_group(&level_meshes, &thresholds)
            .expect("lod group");

        self.lod_state.group = Some(group);
        self.lod_state.level_meshes = level_meshes;
        self.lod_state.thresholds = thresholds.to_vec();
        self.lod_state.aabb = aabb;
        self.lod_state.built = true;
    }
}

/// Per-frame tick: advance the dolly and bump the scene generation.
pub(crate) fn update_lod(app: &mut App, dt: f32) {
    app.lod_state.generation = app.lod_state.generation.wrapping_add(1);
    if app.lod_state.auto_dolly {
        app.lod_state.dolly_time += dt;
        let t = (app.lod_state.dolly_time * 0.35).sin() * 0.5 + 0.5;
        app.camera.distance = 16.0 + t * (78.0 - 16.0);
    }
}

/// Standalone surface-mesh items: a row of spheres at increasing distance, each
/// referencing the LOD group so the renderer picks their level per frame.
pub(crate) fn lod_scene_items(app: &App) -> Vec<SceneRenderItem> {
    let st = &app.lod_state;
    if !st.built {
        return Vec::new();
    }
    let Some(group) = st.group else {
        return Vec::new();
    };

    let half = (GRID_N as f32 - 1.0) * GRID_SPACING * 0.5;
    let x = half + 10.0;
    let count = 5;
    (0..count)
        .map(|i| {
            let y = -half + i as f32 * (2.0 * half / (count as f32 - 1.0));
            let mut item = SceneRenderItem::default();
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(x, y, 0.0)).to_cols_array_2d();
            item.material = Material::from_colour([0.6, 0.65, 0.8]);
            item.settings.pick_id = PickId(1000 + i as u64);
            if st.lod_enabled {
                item.lod_group = Some(group);
            } else {
                item.mesh_id = st.level_meshes[0];
            }
            item
        })
        .collect()
}

/// The instanced field. Added to `mesh_instances` each frame.
pub(crate) fn submit_lod_items(app: &mut App, fd: &mut FrameData) {
    let st = &app.lod_state;
    if !st.built {
        return;
    }
    let Some(group) = st.group else {
        return;
    };

    let camera = RenderCamera::from_camera(&app.camera);
    let half = (GRID_N as f32 - 1.0) * GRID_SPACING * 0.5;
    let count = GRID_N * GRID_N;
    let mut transforms = Vec::with_capacity(count);
    let mut colours = Vec::with_capacity(count);

    for iy in 0..GRID_N {
        for ix in 0..GRID_N {
            let x = ix as f32 * GRID_SPACING - half;
            let y = iy as f32 * GRID_SPACING - half;
            let model = glam::Mat4::from_translation(glam::Vec3::new(x, y, 0.0));

            let colour = if st.lod_enabled && st.colourise {
                let size = projected_screen_size(&st.aabb, &model, &camera);
                level_tint(level_for_size(&st.thresholds, size))
            } else {
                [0.74, 0.74, 0.78, 1.0]
            };

            transforms.push(model.to_cols_array_2d());
            colours.push(colour);
        }
    }

    let mut item = MeshInstanceItem::default();
    item.transforms = transforms;
    item.colours = colours;
    item.blend = SpriteBlend::AlphaBlend;
    if st.lod_enabled {
        item.lod_group = Some(group);
    } else {
        item.mesh_id = st.level_meshes[0].index() as u64;
    }
    fd.scene.mesh_instances.push(item);
}

/// Push the current cull setting onto the group. Cheap to call each frame: it
/// just sets a field on the registered group.
pub(crate) fn apply_lod_cull(st: &LodState, renderer: &mut ViewportRenderer) {
    let Some(group) = st.group else {
        return;
    };
    let cull = if st.cull_enabled {
        Some(st.cull_below)
    } else {
        None
    };
    let _ = renderer.resources_mut().set_lod_cull_below(group, cull);
}

pub(crate) fn controls_lod(app: &mut App, ui: &mut egui::Ui) {
    ui.label("A field of spheres receding into the distance. Each picks a mesh");
    ui.label("from how large it appears on screen.");
    ui.separator();

    ui.checkbox(&mut app.lod_state.lod_enabled, "LOD enabled");
    ui.label("Off forces the full mesh everywhere, for the cost comparison.");
    ui.add_enabled(
        app.lod_state.lod_enabled,
        egui::Checkbox::new(&mut app.lod_state.colourise, "Colourise by level"),
    );
    ui.checkbox(
        &mut app.lod_state.auto_dolly,
        "Animate camera (dolly in/out)",
    );
    ui.add_enabled_ui(app.lod_state.lod_enabled, |ui| {
        ui.checkbox(&mut app.lod_state.cull_enabled, "Cull below a size");
        ui.add_enabled(
            app.lod_state.cull_enabled,
            egui::Slider::new(&mut app.lod_state.cull_below, 0.0..=0.1).text("Cull size"),
        );
    });
    ui.separator();

    if app.lod_state.colourise && app.lod_state.lod_enabled {
        ui.label("green = full detail, yellow = mid, red = crude");
        ui.separator();
    }

    let s = app.lod_state.last_stats;
    stat_row(
        ui,
        "Camera distance",
        &format!("{:.1}", app.camera.distance),
    );
    stat_row(ui, "LOD resolved", &s.lod_items_resolved.to_string());
    stat_row(ui, "LOD reduced", &s.lod_items_reduced.to_string());
    stat_row(ui, "LOD switches", &s.lod_switches.to_string());
    stat_row(ui, "LOD culled", &s.lod_culled.to_string());
    stat_row(ui, "Draw calls", &s.draw_calls.to_string());
    stat_row(ui, "Triangles", &format_count(s.triangles_submitted));
}

fn stat_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(format!("{label}:"));
        ui.monospace(value);
    });
}

fn format_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// The level a given screen size maps to. Mirrors `LodGroup::level_for_size`:
/// the finest level whose threshold the size clears, else the crudest.
fn level_for_size(thresholds: &[f32], size: f32) -> usize {
    for (i, &t) in thresholds.iter().enumerate() {
        if size >= t {
            return i;
        }
    }
    thresholds.len().saturating_sub(1)
}

fn level_tint(level: usize) -> [f32; 4] {
    match level {
        0 => [0.32, 0.78, 0.36, 1.0],
        1 => [0.92, 0.80, 0.22, 1.0],
        _ => [0.86, 0.32, 0.28, 1.0],
    }
}
