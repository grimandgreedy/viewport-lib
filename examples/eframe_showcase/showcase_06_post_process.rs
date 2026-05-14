//! Showcase 6: Post-Processing : build and controls.
//!
//! Post-processing showcase: bloom, SSAO, FXAA, tone mapping, and EDL controls.
//! Uses `prepare_callback` + `paint_callback` which dispatches to the full HDR
//! pipeline when `PostProcessSettings::enabled` is true (the default).

use crate::App;
use crate::geometry::make_uv_sphere;
use eframe::egui;
use viewport_lib::{Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct PostProcessState {
    pub built:          bool,
    pub scene:          Scene,
    pub shadow_pcss:    bool,
    pub point_light_on: bool,
    pub dir_intensity:  f32,
    pub dof_enabled:    bool,
    pub dof_focal_dist: f32,
    pub dof_focal_range: f32,
    pub dof_max_blur:   f32,
}

impl Default for PostProcessState {
    fn default() -> Self {
        Self {
            built:          false,
            scene:          Scene::new(),
            shadow_pcss:    true,
            point_light_on: true,
            dir_intensity:  0.6,
            dof_enabled:    false,
            dof_focal_dist: 5.0,
            dof_focal_range: 1.0,
            dof_max_blur:   8.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 6 (post-processing / PBR scene).
    pub(crate) fn build_pp_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.pp_state.scene = Scene::new();

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Ground",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(10.0, 10.0, 0.15),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, -0.575),
            ),
            Material::pbr([1.0, 1.0, 1.0], 0.0, 0.9),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Gold (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, -1.2, 0.0)),
            Material::pbr([1.0, 0.78, 0.2], 0.95, 0.05),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Brushed Steel (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, -1.2, 0.0)),
            Material::pbr([0.82, 0.82, 0.86], 0.75, 0.35),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Chrome (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, 1.2, 0.0)),
            Material::pbr([0.9, 0.9, 0.95], 1.0, 0.02),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Ceramic (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, 1.2, 0.0)),
            Material::pbr([0.85, 0.7, 0.6], 0.0, 0.9),
        );

        let sphere = make_uv_sphere(32, 16, 0.6);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("pp sphere upload");
        self.pp_state.scene.add_named(
            "Sphere Test",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.1)),
            Material::pbr([0.85, 0.45, 0.35], 0.0, 0.55),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Pillar",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.5, 0.5, 2.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, 1.0),
            ),
            Material::pbr([0.7, 0.7, 0.75], 0.1, 0.7),
        );

        self.pp_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_post_process(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Lighting:");
    ui.add(egui::Slider::new(&mut app.pp_state.dir_intensity, 0.0..=5.0).text("Dir. intensity"));
    ui.checkbox(&mut app.pp_state.point_light_on, "Point light");

    ui.separator();
    ui.label("Shadows:");
    ui.checkbox(&mut app.pp_state.shadow_pcss, "PCSS (soft shadows)");

    ui.separator();
    ui.label("Depth of Field:");
    ui.checkbox(&mut app.pp_state.dof_enabled, "Enable DoF");
    if app.pp_state.dof_enabled {
        ui.add(
            egui::Slider::new(&mut app.pp_state.dof_focal_dist, 0.5..=30.0)
                .text("Focal distance"),
        );
        ui.add(
            egui::Slider::new(&mut app.pp_state.dof_focal_range, 0.1..=10.0)
                .text("Focal range"),
        );
        ui.add(
            egui::Slider::new(&mut app.pp_state.dof_max_blur, 1.0..=20.0)
                .text("Max blur (px)"),
        );
    }

    ui.separator();
    ui.weak("Bloom, SSAO, and FXAA can be enabled via PostProcessSettings\nbut are not wired to controls in this showcase.");
}
