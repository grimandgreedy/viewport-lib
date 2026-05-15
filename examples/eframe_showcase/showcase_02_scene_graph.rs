//! Showcase 2: Scene Graph + Materials.

use crate::App;
use eframe::egui;
use viewport_lib::{Material, ViewportRenderer, scene::Scene, selection::Selection};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct SgState {
    pub scene: Scene,
    pub selection: Selection,
    pub material_cycle: usize,
    pub bg_cycle: usize,
    pub outline_width: f32,
    pub layer_b: Option<viewport_lib::scene::LayerId>,
    pub layer_b_visible: bool,
    pub built: bool,
}

impl Default for SgState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            selection: Selection::new(),
            material_cycle: 0,
            bg_cycle: 0,
            outline_width: 4.0,
            layer_b: None,
            layer_b_visible: true,
            built: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_scene_graph(&mut self, renderer: &mut ViewportRenderer) {
        self.sg_state.scene = Scene::new();
        self.sg_state.layer_b = Some(self.sg_state.scene.add_layer("Layer B"));
        self.sg_state.layer_b_visible = true;
        self.sg_state.selection.clear();

        let positions = [
            [-1.5, -1.5, 0.0],
            [1.5, -1.5, 0.0],
            [-1.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
        ];
        let colors = [
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.3],
            [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3],
        ];
        for (i, (pos, color)) in positions.iter().zip(&colors).enumerate() {
            let mesh = self.upload_box(renderer);
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material::from_color(*color);
            let name = format!("Box {}", i + 1);
            let id = self
                .sg_state
                .scene
                .add_named(&name, Some(mesh), transform, mat);
            if i >= 2 {
                self.sg_state
                    .scene
                    .set_layer(id, self.sg_state.layer_b.unwrap());
            }
        }

        self.sg_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_scene_graph(app: &mut App, ui: &mut egui::Ui, frame: &eframe::Frame) {
    let sel = app.sg_state.selection.len();
    let nodes = app.sg_state.scene.node_count();
    ui.label(format!("Nodes: {nodes}  Selected: {sel}"));
    ui.separator();

    if ui.button("Cycle Material").clicked() {
        app.sg_state.material_cycle += 1;
        let mat = material_preset(app.sg_state.material_cycle);
        for &id in app.sg_state.selection.iter() {
            app.sg_state.scene.set_material(id, mat);
        }
    }

    if ui.button("Toggle Transparency").clicked() {
        for &id in app.sg_state.selection.iter() {
            if let Some(node) = app.sg_state.scene.node(id) {
                let mut mat = *node.material();
                mat.opacity = if mat.opacity < 1.0 { 1.0 } else { 0.4 };
                app.sg_state.scene.set_material(id, mat);
            }
        }
    }

    if ui.button("Toggle Normal Vis").clicked() {
        for &id in app.sg_state.selection.iter() {
            if let Some(node) = app.sg_state.scene.node(id) {
                let show = !node.show_normals();
                app.sg_state.scene.set_show_normals(id, show);
            }
        }
    }

    ui.separator();

    ui.label("Outline width (px):");
    ui.add(egui::Slider::new(&mut app.sg_state.outline_width, 1.0..=8.0).step_by(0.5));

    ui.separator();

    if ui.button("Cycle Background").clicked() {
        app.sg_state.bg_cycle += 1;
    }

    ui.separator();

    if ui.button("Add Child to Selected").clicked() {
        if let Some(parent_id) = app.sg_state.selection.primary() {
            let rs = frame.wgpu_render_state().unwrap();
            let mut guard = rs.renderer.write();
            let renderer = guard
                .callback_resources
                .get_mut::<viewport_lib::ViewportRenderer>()
                .unwrap();
            let mesh = app.upload_box(renderer);
            let local = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(0.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(1.5, 0.0, 1.5),
            );
            let child_id = app.sg_state.scene.add_named(
                "Child",
                Some(mesh),
                local,
                Material::from_color([1.0, 0.6, 0.2]),
            );
            app.sg_state.scene.set_parent(child_id, Some(parent_id));
            app.sg_state.selection.select_one(child_id);
        }
    }

    if ui.button("Remove Selected").clicked() {
        if let Some(id) = app.sg_state.selection.primary() {
            let removed = app.sg_state.scene.remove(id);
            for rid in &removed {
                app.sg_state.selection.remove(*rid);
            }
        }
    }

    ui.separator();

    if ui
        .checkbox(&mut app.sg_state.layer_b_visible, "Layer B Visible")
        .changed()
    {
        app.sg_state
            .scene
            .set_layer_visible(app.sg_state.layer_b.unwrap(), app.sg_state.layer_b_visible);
    }

    ui.separator();

    if ui.button("Cycle Selection (Tab)").clicked() {
        let walk = app.sg_state.scene.walk_depth_first();
        if !walk.is_empty() {
            let current = app.sg_state.selection.primary();
            let next_idx = match current {
                Some(id) => {
                    let pos = walk.iter().position(|(nid, _)| *nid == id);
                    pos.map(|i| (i + 1) % walk.len()).unwrap_or(0)
                }
                None => 0,
            };
            app.sg_state.selection.select_one(walk[next_idx].0);
        }
    }

    if ui.button("Clear Selection").clicked() {
        app.sg_state.selection.clear();
    }
}

// ---------------------------------------------------------------------------
// Material and background presets
// ---------------------------------------------------------------------------

pub(crate) fn material_preset(index: usize) -> Material {
    match index % 4 {
        0 => Material::default(),
        1 => {
            let mut m = Material::from_color([0.8, 0.2, 0.2]);
            m.specular = 0.8;
            m.shininess = 64.0;
            m.ambient = 0.1;
            m
        }
        2 => {
            let mut m = Material::from_color([0.2, 0.4, 0.9]);
            m.opacity = 0.5;
            m.specular = 0.9;
            m.shininess = 128.0;
            m
        }
        3 => {
            let mut m = Material::from_color([0.3, 0.7, 0.3]);
            m.specular = 0.1;
            m.shininess = 8.0;
            m.diffuse = 0.9;
            m
        }
        _ => unreachable!(),
    }
}

pub(crate) fn background_color(index: usize) -> [f32; 4] {
    match index % 3 {
        0 => crate::BG_COLOR,
        1 => [0.05, 0.08, 0.15, 1.0],
        2 => [0.18, 0.16, 0.14, 1.0],
        _ => unreachable!(),
    }
}
