//! Showcase 5: Materials and Visibility: build and controls.

use crate::App;
use eframe::egui;
use viewport_lib::{Material, Selection, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct MaterialsVisibilityState {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub clip_enabled: bool,
    pub outline_on: bool,
    pub xray_on: bool,
}

impl Default for MaterialsVisibilityState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            clip_enabled: false,
            outline_on: true,
            xray_on: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 5 (materials, clipping, outlines, x-ray).
    pub(crate) fn build_materials_visibility_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.materials_visibility_state.scene = Scene::new();
        self.materials_visibility_state.selection.clear();

        let m = self.upload_box(renderer);
        let id = self.materials_visibility_state.scene.add_named(
            "Gold (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, -1.5, 0.0)),
            Material::pbr([1.0, 0.78, 0.2], 0.95, 0.05),
        );
        self.materials_visibility_state.selection.select_one(id);

        let m = self.upload_box(renderer);
        self.materials_visibility_state.scene.add_named(
            "Brushed Steel (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(2.5, 1.5, 0.0)),
            Material::pbr([0.82, 0.82, 0.86], 0.75, 0.35),
        );

        let m = self.upload_box(renderer);
        self.materials_visibility_state.scene.add_named(
            "Shiny Blue (Blinn-Phong)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, -1.5, 0.0)),
            {
                let mut mat = Material::from_colour([0.2, 0.4, 0.9]);
                mat.specular = 0.9;
                mat.shininess = 128.0;
                mat
            },
        );

        let m = self.upload_box(renderer);
        self.materials_visibility_state.scene.add_named(
            "Matte Green (Blinn-Phong)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-2.5, 1.5, 0.0)),
            {
                let mut mat = Material::from_colour([0.2, 0.7, 0.3]);
                mat.specular = 0.05;
                mat.diffuse = 0.95;
                mat.shininess = 4.0;
                mat
            },
        );

        let m = self.upload_box(renderer);
        self.materials_visibility_state.scene.add_named(
            "Wall (occluder)",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(7.5, 0.25, 2.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 3.5, 0.25),
            ),
            Material::from_colour([0.55, 0.55, 0.55]),
        );

        let m = self.upload_box(renderer);
        self.materials_visibility_state.scene.add_named(
            "Hidden Magenta (x-ray target)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 5.5, 0.0)),
            Material::from_colour([0.9, 0.3, 0.7]),
        );

        self.materials_visibility_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_materials_visibility(app: &mut App, ui: &mut egui::Ui) {
    let sel = app.materials_visibility_state.selection.len();
    ui.label(format!("Selected: {sel}"));
    ui.separator();

    ui.checkbox(
        &mut app.materials_visibility_state.clip_enabled,
        "Clip plane (x < 0)",
    );
    ui.checkbox(
        &mut app.materials_visibility_state.outline_on,
        "Selection outline",
    );
    ui.checkbox(
        &mut app.materials_visibility_state.xray_on,
        "X-ray selected",
    );

    ui.separator();

    if ui.button("Cycle Selection (Tab)").clicked() {
        let walk = app.materials_visibility_state.scene.walk_depth_first();
        if !walk.is_empty() {
            let current = app.materials_visibility_state.selection.primary();
            let next_idx = match current {
                Some(id) => {
                    let pos = walk.iter().position(|(nid, _)| *nid == id);
                    pos.map(|i| (i + 1) % walk.len()).unwrap_or(0)
                }
                None => 0,
            };
            app.materials_visibility_state
                .selection
                .select_one(walk[next_idx].0);
        }
    }

    if ui.button("Clear Selection").clicked() {
        app.materials_visibility_state.selection.clear();
    }
}
