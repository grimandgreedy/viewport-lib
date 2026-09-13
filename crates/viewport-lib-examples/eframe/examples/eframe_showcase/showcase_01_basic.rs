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

// ---------------------------------------------------------------------------
// Per-frame frame-data tweaks
// ---------------------------------------------------------------------------

/// Fold this showcase's own contributions into the assembled frame: extra
/// render items, overlays, and effect settings that are re-submitted every
/// frame rather than baked into the scene.
pub(crate) fn frame(
    _app: &mut crate::App,
    _fd: &mut vpl::FrameData,
    _ctx: &crate::FrameCtx,
) {}

// ---------------------------------------------------------------------------
// Viewport overlay and per-frame tick
// ---------------------------------------------------------------------------

/// Draw this showcase's own egui overlay on top of the rendered viewport:
/// selection rectangles, mode readouts, and in-scene labels.
pub(crate) fn overlay(_app: &mut crate::App, _ui: &mut crate::eframe::egui::Ui, _cx: &crate::ViewportCtx) {}

/// Advance this showcase's animation and ask for another frame. Runs after the
/// viewport has been drawn, so it only affects the next frame.
pub(crate) fn tick(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Route a viewport click for this showcase. The host calls this for a plain
/// click that no gizmo or widget has already consumed; `pos` is in viewport
/// pixels.
pub(crate) fn on_click(_app: &mut crate::App, _cx: &crate::ClickCtx) {}

/// Handle drag gestures this showcase owns, before the camera controller runs.
pub(crate) fn drag_input(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Advance this showcase's own camera animation or object motion for the frame.
pub(crate) fn advance(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Update this showcase's interactive widgets for the frame.
pub(crate) fn widgets(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Flush any per-frame GPU writes this showcase has queued.
pub(crate) fn flush_gpu(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Cache gizmo placement for next frame's hit-testing.
pub(crate) fn cache_gizmo(_app: &mut crate::App, _cx: &crate::ViewportCtx) {}

/// Take over the whole viewport for this frame. Returning false leaves the
/// host's normal single-viewport path in charge.
pub(crate) fn viewport_override(
    _app: &mut crate::App,
    _ui: &mut crate::eframe::egui::Ui,
    _cx: &crate::ViewportCtx,
) -> bool {
    false
}
