//! Showcase 10: Camera Tools : build and controls.
//!
//! Six coloured boxes arranged along the cardinal axes so that every named
//! view preset shows a clearly different face of the layout:
//!   Front (+Y): red/green split visible
//!   Right (+X): green/blue split visible
//!   Top  (+Z): full cross visible from above

use crate::App;
use eframe::egui;
use viewport_lib::{Easing, Material, Projection, ViewPreset, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct CameraToolsState {
    pub built: bool,
    pub scene: Scene,
}

impl Default for CameraToolsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 10 (Camera Tools demo).
    pub(crate) fn build_camera_tools_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.ct_state.scene = Scene::new();

        let objects: &[(&str, glam::Vec3, [f32; 3])] = &[
            ("Origin", glam::Vec3::ZERO, [0.70, 0.70, 0.70]),
            (
                "+X Right",
                glam::Vec3::new(4.0, 0.0, 0.0),
                [0.85, 0.22, 0.22],
            ),
            (
                "-X Left",
                glam::Vec3::new(-4.0, 0.0, 0.0),
                [0.50, 0.10, 0.10],
            ),
            (
                "+Y Front",
                glam::Vec3::new(0.0, 4.0, 0.0),
                [0.22, 0.78, 0.22],
            ),
            (
                "-Y Back",
                glam::Vec3::new(0.0, -4.0, 0.0),
                [0.10, 0.40, 0.10],
            ),
            ("+Z Up", glam::Vec3::new(0.0, 0.0, 4.0), [0.25, 0.50, 0.90]),
        ];

        for (name, pos, colour) in objects {
            let mesh = self.upload_box(renderer);
            self.ct_state.scene.add_named(
                name,
                Some(mesh),
                glam::Mat4::from_translation(*pos),
                Material::from_colour(*colour),
            );
        }

        self.ct_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_camera_tools(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Named Views:");
    egui::Grid::new("cam_view_presets")
        .num_columns(4)
        .show(ui, |ui| {
            for (label, preset) in [
                ("Front", ViewPreset::Front),
                ("Back", ViewPreset::Back),
                ("Left", ViewPreset::Left),
                ("Right", ViewPreset::Right),
                ("Top", ViewPreset::Top),
                ("Bottom", ViewPreset::Bottom),
                ("Iso", ViewPreset::Isometric),
            ] {
                if ui.button(label).clicked() {
                    app.cam_animator.fly_to_full(
                        &app.camera,
                        app.camera.center,
                        app.camera.distance,
                        preset.orientation(),
                        preset.preferred_projection(),
                        0.6,
                        Easing::EaseInOutCubic,
                    );
                }
            }
        });
    ui.separator();
    ui.label("Projection:");
    ui.horizontal(|ui| {
        if ui
            .radio(
                app.camera.projection == Projection::Perspective,
                "Perspective",
            )
            .clicked()
        {
            app.camera.projection = Projection::Perspective;
        }
        if ui
            .radio(
                app.camera.projection == Projection::Orthographic,
                "Orthographic",
            )
            .clicked()
        {
            app.camera.projection = Projection::Orthographic;
        }
    });
    if app.camera.projection == Projection::Perspective {
        ui.separator();
        let mut fov_deg = app.camera.fov_y.to_degrees();
        ui.label(format!("FOV: {fov_deg:.0}\u{b0}"));
        if ui
            .add(egui::Slider::new(&mut fov_deg, 20.0_f32..=120.0_f32).suffix("\u{b0}"))
            .changed()
        {
            app.camera.fov_y = fov_deg.to_radians();
        }
    }
    ui.separator();
    ui.label("The coloured boxes identify each axis:");
    ui.label("Red = +X,  Green = +Y,  Blue = +Z");
}
