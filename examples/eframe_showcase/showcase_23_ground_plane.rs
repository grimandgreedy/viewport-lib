//! Showcase 23: Ground Plane.
//!
//! Demonstrates all four ground-plane modes:
//!   - None       : plane disabled (zero overhead)
//!   - ShadowOnly : invisible plane that receives and displays shadows
//!   - Tile       : procedural checkerboard
//!   - SolidColor : flat-colored plane
//!
//! Layout: three spheres at y = -3, 0, +3 (along X-axis), floating at Z = 1.5
//! above a ground plane at Z = 0.

use crate::App;
use eframe::egui;
use viewport_lib::{Material, ViewportRenderer, scene::Scene};

impl App {
    pub(crate) fn build_ground_plane_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.gp_scene = Scene::new();

        let sphere = viewport_lib::geometry::primitives::sphere(1.0, 32, 16);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("gp sphere mesh upload");

        // Three spheres in a row along X.
        let positions: [(f32, &str, [f32; 3]); 3] = [
            (-3.5, "Left", [0.8, 0.3, 0.3]),
            (0.0, "Centre", [0.3, 0.7, 0.3]),
            (3.5, "Right", [0.3, 0.4, 0.8]),
        ];

        for (x, name, color) in positions {
            let mut mat = Material::from_color(color);
            mat.roughness = 0.5;
            mat.metallic = 0.1;
            self.gp_scene.add_named(
                name,
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 1.5)),
                mat,
            );
        }

        self.gp_built = true;
    }

    pub(crate) fn controls_ground_plane(&mut self, ui: &mut egui::Ui) {
        ui.label("Ground plane mode:");
        ui.horizontal_wrapped(|ui| {
            for (label, mode) in [
                ("None", GpMode::None),
                ("ShadowOnly", GpMode::ShadowOnly),
                ("Tile", GpMode::Tile),
                ("SolidColor", GpMode::SolidColor),
            ] {
                if ui.selectable_label(self.gp_mode == mode, label).clicked() {
                    self.gp_mode = mode;
                }
            }
        });

        ui.separator();
        ui.label("Height (Z):");
        ui.add(egui::Slider::new(&mut self.gp_height, -3.0..=3.0).step_by(0.1));

        match self.gp_mode {
            GpMode::Tile => {
                ui.separator();
                ui.label("Tile color:");
                ui.color_edit_button_rgba_unmultiplied(&mut self.gp_color);
                ui.label("Tile size:");
                ui.add(egui::Slider::new(&mut self.gp_tile_size, 0.1..=5.0).step_by(0.1));
            }
            GpMode::SolidColor => {
                ui.separator();
                ui.label("Surface color:");
                ui.color_edit_button_rgba_unmultiplied(&mut self.gp_color);
            }
            GpMode::ShadowOnly => {
                ui.separator();
                ui.label("Shadow color:");
                ui.color_edit_button_rgba_unmultiplied(&mut self.gp_shadow_color);
                ui.label("Shadow opacity:");
                ui.add(egui::Slider::new(&mut self.gp_shadow_opacity, 0.0..=1.0).step_by(0.05));
            }
            GpMode::None => {}
        }
    }
}

/// Ground plane mode selection (mirrors `viewport_lib::GroundPlaneMode`).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GpMode {
    #[default]
    None,
    ShadowOnly,
    Tile,
    SolidColor,
}
