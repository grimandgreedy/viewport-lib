//! Showcase 1: Basic rendering -- four boxes with directional or point light.

use crate::{App, MeshId};
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{Projection, SceneRenderItem, ViewportRenderer};

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
        let mesh = vpl::primitives::cube(1.0);
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

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    app.basic_state.mesh_id.is_none()
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_basic_scene(renderer);
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. `_out` carries
/// the few extra frame settings a showcase can set alongside its items.
pub(crate) fn scene(
    app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    _out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    let (items, bg_colour, lighting, scene_gen, sel_gen) = {
        let items = app.basic_scene_items();

        let lights = if app.basic_state.use_point_light {
            vec![{
                let mut _t = vpl::LightSource::default();
                _t.kind = vpl::LightKind::Point {
                    position: [5.0, 5.0, 5.0],
                    range: 30.0,
                    radius: 0.1,
                };
                // Candela-scale key light: inverse-square over ~8 units to
                // the origin objects needs a large intensity to read.
                _t.intensity = 150.0;
                _t
            }]
        } else {
            vec![vpl::LightSource::default()]
        };
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.lights = lights;
            _t.hemisphere_intensity = 0.25;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        (items, None, lighting, 0u64, 0u64)
    };
    crate::SceneContents {
        items,
        bg_colour,
        lighting,
        scene_gen,
        sel_gen,
    }
}
