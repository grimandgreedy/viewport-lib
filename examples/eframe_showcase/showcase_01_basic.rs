//! Showcase 1: Basic rendering -- four boxes with directional or point light.

use crate::{App, MeshId};
use eframe::egui;
use viewport_lib::{Projection, SceneRenderItem, ViewportRenderer};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct BasicState {
    pub mesh_id: Option<MeshId>,
    pub use_point_light: bool,
}

impl Default for BasicState {
    fn default() -> Self {
        Self {
            mesh_id: None,
            use_point_light: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_basic_scene(&mut self, renderer: &mut ViewportRenderer) {
        let mesh = viewport_lib::primitives::cube(1.0);
        self.basic_state.mesh_id = Some(
            renderer
                .resources_mut()
                .upload_mesh_data(&self.device, &mesh)
                .expect("basic box mesh"),
        );
    }
}

// ---------------------------------------------------------------------------
// Render items
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn basic_scene_items(&self) -> Vec<SceneRenderItem> {
        let Some(mesh_id) = self.basic_state.mesh_id else {
            return vec![];
        };
        let positions = [
            [-1.5f32, -1.5, 0.0],
            [1.5, -1.5, 0.0],
            [-1.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
        ];
        positions
            .iter()
            .map(|pos| {
                let mut item = SceneRenderItem::default();
                item.mesh_id = mesh_id;
                item.model =
                    glam::Mat4::from_translation(glam::Vec3::from(*pos)).to_cols_array_2d();
                item
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_basic(app: &mut App, ui: &mut egui::Ui) {
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
    ui.separator();
    ui.label("Light:");
    ui.horizontal(|ui| {
        if ui
            .radio(!app.basic_state.use_point_light, "Directional")
            .clicked()
        {
            app.basic_state.use_point_light = false;
        }
        if ui.radio(app.basic_state.use_point_light, "Point").clicked() {
            app.basic_state.use_point_light = true;
        }
    });
}
