//! Showcase 22: Parameterization Visualization.
//!
//! Demonstrates the `ParamVis` API for inspecting UV quality on four mesh types:
//!   - Torus   (z =  4.5)
//!   - Sphere  (z =  1.5)
//!   - Cube    (z = -1.5)
//!   - Plane   (z = -4.5, two-sided)
//!
//! Within each row, columns correspond to the four `ParamVisMode` variants:
//!   Left -> Checker . Grid . LocalChecker . LocalRadial <- Right
//!
//! The controls panel lets you:
//! - Toggle UV-vis on/off to compare patterns against plain PBR shading.
//! - Adjust the tile scale shared across all objects.

use crate::App;
use crate::geometry::{make_box_with_uvs, make_uv_sphere};
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{
    Material, MeshData, MeshId, NodeId, ParamVis, ParamVisMode, ViewportRenderer,
    scene::{Scene, material::BackfacePolicy},
};

/// One entry per `ParamVisMode` variant (column order).
const MODES: [(ParamVisMode, &str); 4] = [
    (ParamVisMode::Checker, "Checker"),
    (ParamVisMode::Grid, "Grid"),
    (ParamVisMode::LocalChecker, "LocalChecker"),
    (ParamVisMode::LocalRadial, "LocalRadial"),
];

const X_POSITIONS: [f32; 4] = [-4.5, -1.5, 1.5, 4.5];

// Row Z offsets (front-to-back order in UI: torus at back, plane at front).
const Z_TORUS: f32 = 4.5;
const Z_SPHERE: f32 = 1.5;
const Z_CUBE: f32 = -1.5;
const Z_PLANE: f32 = -4.5;

// Node-ID layout: [0..4] = torus, [4..8] = sphere, [8..12] = cube, [12..16] = plane.
const IDX_TORUS: usize = 0;
const IDX_SPHERE: usize = 4;
const IDX_CUBE: usize = 8;
const IDX_PLANE: usize = 12;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ParamVisState {
    pub scene: Scene,
    pub built: bool,
    pub node_ids: [NodeId; 16],
    pub scale: f32,
    pub on: bool,
}

impl Default for ParamVisState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            built: false,
            node_ids: [0u64; 16],
            scale: 8.0,
            on: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build Showcase 22: Parameterization Visualization.
    ///
    /// Four rows x four columns = 16 scene nodes.
    /// Each object needs its own uploaded mesh slot for independent per-object GPU state.
    pub(crate) fn build_param_vis_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.param_vis_state.scene = Scene::new();

        let torus_data = vpl::primitives::torus(1.1, 0.45, 48, 24);
        let sphere_data = make_uv_sphere(48, 24, 1.0);
        let cube_data = make_box_with_uvs(1.6, 1.6, 1.6);
        let plane_data = vpl::primitives::plane(2.8, 2.8);

        let upload_mesh = |renderer: &mut ViewportRenderer, data: &MeshData| -> MeshId {
            renderer
                .resources_mut()
                .upload_mesh_data(&self.device, data)
                .expect("param-vis mesh upload")
        };

        let scale = self.param_vis_state.scale;

        let add_row = |scene: &mut Scene,
                       renderer: &mut ViewportRenderer,
                       mesh_data: &MeshData,
                       z: f32,
                       base_idx: usize,
                       node_ids: &mut [NodeId; 16],
                       colour: [f32; 3],
                       two_sided: bool,
                       rotate_x_90: bool| {
            for (col, (mode, label)) in MODES.iter().enumerate() {
                let mesh_id = upload_mesh(renderer, mesh_data);
                let mat = {
                    let mut m = Material::pbr(colour, 0.0, 0.4);
                    m.param_vis = Some(ParamVis { mode: *mode, scale });
                    m.backface_policy = if two_sided {
                        BackfacePolicy::Identical
                    } else {
                        BackfacePolicy::Cull
                    };
                    m
                };
                let mut transform =
                    glam::Mat4::from_translation(glam::Vec3::new(X_POSITIONS[col], 0.0, z));
                if rotate_x_90 {
                    transform *= glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);
                }
                let node_id = scene.add_named(*label, Some(mesh_id), transform, mat);
                node_ids[base_idx + col] = node_id;
            }
        };

        add_row(
            &mut self.param_vis_state.scene,
            renderer,
            &torus_data,
            Z_TORUS,
            IDX_TORUS,
            &mut self.param_vis_state.node_ids,
            [0.55, 0.70, 0.65],
            false,
            true,
        );
        add_row(
            &mut self.param_vis_state.scene,
            renderer,
            &sphere_data,
            Z_SPHERE,
            IDX_SPHERE,
            &mut self.param_vis_state.node_ids,
            [0.7, 0.7, 0.7],
            false,
            false,
        );
        add_row(
            &mut self.param_vis_state.scene,
            renderer,
            &cube_data,
            Z_CUBE,
            IDX_CUBE,
            &mut self.param_vis_state.node_ids,
            [0.72, 0.65, 0.55],
            false,
            false,
        );
        add_row(
            &mut self.param_vis_state.scene,
            renderer,
            &plane_data,
            Z_PLANE,
            IDX_PLANE,
            &mut self.param_vis_state.node_ids,
            [0.65, 0.65, 0.80],
            true,
            true,
        );

        self.param_vis_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_param_vis(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Four mesh rows, four ParamVisMode columns:");
    ui.label("  Back row    : torus");
    ui.label("  .           : sphere");
    ui.label("  .           : cube");
    ui.label("  Front row   : plane (two-sided)");
    ui.separator();
    ui.label("Columns (left -> right):");
    ui.label("  Checker . Grid . LocalChecker . LocalRadial");

    ui.separator();

    let vis_changed = ui
        .checkbox(&mut app.param_vis_state.on, "UV vis on")
        .changed();

    ui.separator();

    ui.label("Scale (tiles per UV unit):");
    let scale_changed = ui
        .add(
            egui::Slider::new(&mut app.param_vis_state.scale, 1.0..=32.0)
                .step_by(0.5)
                .logarithmic(false),
        )
        .changed();

    if (vis_changed || scale_changed) && app.param_vis_state.built {
        update_param_vis_materials(app);
    }

    ui.separator();
    ui.weak("Checker: alternating black/white in UV space.");
    ui.weak("Grid: thin lines at UV integer boundaries.");
    ui.weak("LocalChecker: polar checkerboard at UV (0.5, 0.5).");
    ui.weak("LocalRadial: concentric rings at UV (0.5, 0.5).");
}

fn update_param_vis_materials(app: &mut App) {
    let s = &mut app.param_vis_state;
    let rows: [(usize, [f32; 3], bool); 4] = [
        (IDX_TORUS, [0.55, 0.70, 0.65], false),
        (IDX_SPHERE, [0.7, 0.7, 0.7], false),
        (IDX_CUBE, [0.72, 0.65, 0.55], false),
        (IDX_PLANE, [0.65, 0.65, 0.80], true),
    ];
    for (base_idx, colour, two_sided) in rows {
        for (col, (mode, _)) in MODES.iter().enumerate() {
            let mat = {
                let mut m = Material::pbr(colour, 0.0, 0.4);
                m.param_vis = if s.on {
                    Some(ParamVis {
                        mode: *mode,
                        scale: s.scale,
                    })
                } else {
                    None
                };
                m.backface_policy = if two_sided {
                    BackfacePolicy::Identical
                } else {
                    BackfacePolicy::Cull
                };
                m
            };
            s.scene.set_material(s.node_ids[base_idx + col], mat);
        }
    }
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.param_vis_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_param_vis_scene(renderer);
    app.camera = vpl::Camera {
        center: glam::Vec3::ZERO,
        distance: 22.0,
        orientation: glam::Quat::from_rotation_z(0.3)
            * glam::Quat::from_rotation_x(1.1),
        ..vpl::Camera::default()
    };
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
        let items = app
            .param_vis_state
            .scene
            .collect_render_items(&vpl::Selection::new());
        let sg = app.param_vis_state.scene.version();
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.hemisphere_intensity = 0.5;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        (items, None, lighting, sg, 0)
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

/// Drive the orbit controller for this showcase. Returning false leaves the
/// host to run the usual suppress-or-apply path.
pub(crate) fn drive_camera(_app: &mut crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

/// Whether the orbit controller should resolve without moving the camera this
/// frame. This showcase never suppresses it.
pub(crate) fn suppress_orbit(_app: &crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}
