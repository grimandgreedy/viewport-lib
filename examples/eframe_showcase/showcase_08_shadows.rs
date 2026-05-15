//! Showcase 8: Shadow Demo : build and controls.

use crate::App;
use crate::geometry::{make_box_with_uvs, make_uv_sphere};
use eframe::egui;
use viewport_lib::{Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ShadowsState {
    pub built: bool,
    pub scene: Scene,
    pub cascade_count: u32,
    pub pcss_on: bool,
    pub contact_on: bool,
}

impl Default for ShadowsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            cascade_count: 4,
            pcss_on: false,
            contact_on: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_shadow_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.shd_state.scene = Scene::new();

        let ground_mesh = make_box_with_uvs(20.0, 20.0, 0.2);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("ground mesh upload");
        self.shd_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.1)),
            Material::pbr([1.0, 1.0, 1.0], 0.0, 0.9),
        );

        let sphere_mesh = make_uv_sphere(24, 12, 0.5);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("sphere mesh upload");

        let sphere_dense_mesh = make_uv_sphere(64, 32, 0.5);
        let sphere_dense_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_dense_mesh)
            .expect("dense sphere mesh upload");

        let box_mesh = make_box_with_uvs(1.0, 1.0, 1.0);
        let box_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &box_mesh)
            .expect("box mesh upload");

        let object_data: &[(&str, glam::Vec3, [f32; 3])] = &[
            (
                "Sphere Near",
                glam::Vec3::new(-1.5, 1.0, 0.5),
                [0.85, 0.35, 0.25],
            ),
            (
                "Box Near",
                glam::Vec3::new(1.5, 1.0, 0.5),
                [0.25, 0.55, 0.85],
            ),
            (
                "Sphere Mid",
                glam::Vec3::new(-3.0, -3.0, 0.5),
                [0.8, 0.7, 0.3],
            ),
            (
                "Box Mid",
                glam::Vec3::new(3.0, -3.0, 0.5),
                [0.35, 0.75, 0.45],
            ),
        ];
        for (name, pos, colour) in object_data {
            let mesh_id = if *name == "Sphere Near" {
                sphere_dense_id
            } else if name.contains("Sphere") {
                sphere_id
            } else {
                box_id
            };
            self.shd_state.scene.add_named(
                name,
                Some(mesh_id),
                glam::Mat4::from_translation(*pos),
                Material::pbr(*colour, 0.0, 0.5),
            );
        }

        let pillar_mesh = make_box_with_uvs(0.4, 0.4, 3.0);
        let pillar_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &pillar_mesh)
            .expect("pillar mesh upload");
        self.shd_state.scene.add_named(
            "Tall Pillar",
            Some(pillar_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, -6.0, 1.5)),
            Material::pbr([0.65, 0.65, 0.70], 0.0, 0.6),
        );

        self.shd_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_shadows(app: &mut App, ui: &mut egui::Ui) {
    ui.label(format!("Cascades: {}", app.shd_state.cascade_count));
    ui.horizontal(|ui| {
        if ui
            .button("-")
            .on_hover_text("Decrease cascade count")
            .clicked()
        {
            app.shd_state.cascade_count = (app.shd_state.cascade_count - 1).max(1);
        }
        if ui
            .button("+")
            .on_hover_text("Increase cascade count")
            .clicked()
        {
            app.shd_state.cascade_count = (app.shd_state.cascade_count + 1).min(4);
        }
    });

    ui.separator();
    ui.label("Filter:");
    ui.horizontal(|ui| {
        if ui.radio(!app.shd_state.pcss_on, "PCF").clicked() {
            app.shd_state.pcss_on = false;
        }
        if ui.radio(app.shd_state.pcss_on, "PCSS").clicked() {
            app.shd_state.pcss_on = true;
        }
    });

    ui.separator();
    ui.checkbox(&mut app.shd_state.contact_on, "Contact Shadows");
}
