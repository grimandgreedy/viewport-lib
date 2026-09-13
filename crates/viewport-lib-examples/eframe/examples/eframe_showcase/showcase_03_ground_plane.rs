//! Showcase 3: Ground Plane.
//!
//! Demonstrates all four ground-plane modes:
//!   - None       : plane disabled (zero overhead)
//!   - ShadowOnly : invisible plane that receives and displays shadows
//!   - Tile       : procedural checkerboard
//!   - SolidColour : flat-coloured plane
//!
//! Layout: three spheres at y = -3, 0, +3 (along X-axis), floating at Z = 1.5
//! above a ground plane at Z = 0.

use crate::App;
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct GroundPlaneState {
    pub scene: Scene,
    pub built: bool,
    pub mode: GpMode,
    pub height: f32,
    pub colour: [f32; 4],
    pub tile_colour2: [f32; 4],
    pub tile_size: f32,
    pub shadow_colour: [f32; 4],
    pub shadow_opacity: f32,
    pub grid_colour: [f32; 3],
}

impl Default for GroundPlaneState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            built: false,
            mode: GpMode::Grid,
            height: 0.0,
            colour: [0.85, 0.15, 0.10, 1.0].into(),
            tile_colour2: [1.0, 1.0, 1.0, 1.0].into(),
            tile_size: 1.0,
            shadow_colour: [0.0, 0.0, 0.0, 1.0].into(),
            shadow_opacity: 0.5,
            grid_colour: [0.55, 0.55, 0.55],
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_ground_plane_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.gp_state.scene = Scene::new();

        let sphere = vpl::geometry::primitives::sphere(1.0, 32, 16);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("gp sphere mesh upload");

        // Three spheres in a row along X.
        let positions: [(f32, &str, [f32; 3]); 3] = [
            (-3.5, "Left", [0.65, 0.09, 0.07]),
            (0.0, "Centre", [0.10, 0.52, 0.18]),
            (3.5, "Right", [0.10, 0.26, 0.68]),
        ];

        for (x, name, colour) in positions {
            let mut mat = Material::from_colour(colour);
            mat.roughness = 0.5;
            mat.metallic = 0.1;
            self.gp_state.scene.add_named(
                name,
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 1.5)),
                mat,
            );
        }

        self.gp_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_ground_plane(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.gp_state;

    ui.label("Ground plane mode:");
    ui.horizontal_wrapped(|ui| {
        for (label, mode) in [
            ("None", GpMode::None),
            ("Grid", GpMode::Grid),
            ("ShadowOnly", GpMode::ShadowOnly),
            ("Tile", GpMode::Tile),
            ("SolidColour", GpMode::SolidColour),
        ] {
            if ui.selectable_label(s.mode == mode, label).clicked() {
                s.mode = mode;
            }
        }
    });

    ui.separator();
    ui.label("Height (Z):");
    ui.add(egui::Slider::new(&mut s.height, -3.0..=3.0).step_by(0.1));

    match s.mode {
        GpMode::Grid => {
            ui.separator();
            ui.label("Grid colour:");
            ui.color_edit_button_rgb(&mut s.grid_colour);
        }
        GpMode::Tile => {
            ui.separator();
            ui.label("Tile colour A:");
            ui.color_edit_button_rgba_unmultiplied(&mut s.colour);
            ui.label("Tile colour B:");
            ui.color_edit_button_rgba_unmultiplied(&mut s.tile_colour2);
            ui.label("Tile size:");
            ui.add(egui::Slider::new(&mut s.tile_size, 0.1..=5.0).step_by(0.1));
        }
        GpMode::SolidColour => {
            ui.separator();
            ui.label("Surface colour:");
            ui.color_edit_button_rgba_unmultiplied(&mut s.colour);
        }
        GpMode::ShadowOnly => {
            ui.separator();
            ui.label("Shadow colour:");
            ui.color_edit_button_rgba_unmultiplied(&mut s.shadow_colour);
            ui.label("Shadow opacity:");
            ui.add(egui::Slider::new(&mut s.shadow_opacity, 0.0..=1.0).step_by(0.05));
        }
        GpMode::None => {}
    }
}

// ---------------------------------------------------------------------------
// Ground plane mode enum
// ---------------------------------------------------------------------------

/// Ground plane mode selection (mirrors `viewport_lib::GroundPlaneMode`).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum GpMode {
    /// Wire grid only, no ground plane.
    #[default]
    Grid,
    /// Tile checkerboard ground plane, no wire grid.
    Tile,
    /// Invisible plane that receives shadows.
    ShadowOnly,
    /// Flat-coloured ground plane.
    SolidColour,
    /// Nothing.
    None,
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.gp_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_ground_plane_scene(renderer);
    app.camera = vpl::Camera {
        center: glam::Vec3::new(0.0, 0.0, 1.0),
        distance: 14.0,
        orientation: glam::Quat::from_rotation_z(0.4)
            * glam::Quat::from_rotation_x(1.0),
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
        let items = app.gp_state.scene.collect_render_items(&vpl::Selection::new());
        let sg = app.gp_state.scene.version();
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.lights = vec![{
                let mut _t = vpl::LightSource::default();
                _t.kind = vpl::LightKind::Directional {
                    direction: [0.4, 0.6, 1.0],
                };
                _t.intensity = 1.5;
                _t
            }];
            _t.shadows.enabled = true;
            _t.hemisphere_intensity = 0.3;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [0.3, 0.3, 0.3];
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
pub(crate) fn on_click(_app: &mut crate::App, _pos: glam::Vec2, _w: f32, _h: f32) {}
