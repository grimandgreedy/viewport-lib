//! Showcase 5: Materials and Visibility: build and controls.

use crate::App;
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{Material, Selection, ViewportRenderer, scene::Scene};

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
            Material::pbr_with_ao([1.0, 0.78, 0.2], 0.95, 0.05, None),
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
            Material::from_colour([0.35, 0.35, 0.35]),
        );

        let m = self.upload_box(renderer);
        self.materials_visibility_state.scene.add_named(
            "Hidden Magenta (x-ray target)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 5.5, 0.0)),
            Material::from_colour([0.65, 0.06, 0.45]),
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

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.materials_visibility_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_materials_visibility_scene(renderer);
    app.camera = vpl::Camera {
        center: glam::Vec3::new(0.0, 2.0, 0.5),
        distance: 14.0,
        orientation: glam::Quat::from_rotation_z(0.4)
            * glam::Quat::from_rotation_x(1.1),
        ..vpl::Camera::default()
    };
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. `out` carries
/// the few extra frame settings a showcase can set alongside its items.
pub(crate) fn scene(
    app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    let (items, bg_colour, lighting, scene_gen, sel_gen) = {
        let items = app
            .materials_visibility_state
            .scene
            .collect_render_items(&app.materials_visibility_state.selection);
        if app.materials_visibility_state.clip_enabled {
            out.clip_objects.push(vpl::ClipObject::plane([1.0, 0.0, 0.0], 0.0));
        }
        out.outline = app.materials_visibility_state.outline_on
            && !app.materials_visibility_state.selection.is_empty();
        out.xray = app.materials_visibility_state.xray_on
            && !app.materials_visibility_state.selection.is_empty();
        let sg = app.materials_visibility_state.scene.version();
        let ss = app.materials_visibility_state.selection.version();
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.hemisphere_intensity = 0.5;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        (items, None, lighting, sg, ss)
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


// ---------------------------------------------------------------------------
// Viewport overlay and per-frame tick
// ---------------------------------------------------------------------------

/// Draw this showcase's own egui overlay on top of the rendered viewport:
/// selection rectangles, mode readouts, and in-scene labels.


/// Advance this showcase's animation and ask for another frame. Runs after the
/// viewport has been drawn, so it only affects the next frame.


/// Route a viewport click for this showcase. The host calls this for a plain
/// click that no gizmo or widget has already consumed; `pos` is in viewport
/// pixels.
pub(crate) fn on_click(app: &mut crate::App, cx: &crate::ClickCtx) {
    // Object-level selection: defer the pick to the render site, where the
    // renderer and the on-screen `FrameData` are in scope, and resolve it with
    // the unified GPU picker. See `apply_pending_pick`.
    app.pending_pick = Some(cx.pos);
}

/// Handle drag gestures this showcase owns, before the camera controller runs.


/// Advance this showcase's own camera animation or object motion for the frame.


/// Update this showcase's interactive widgets for the frame.


/// Flush any per-frame GPU writes this showcase has queued.


/// Cache gizmo placement for next frame's hit-testing.


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

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScMaterialsAndVisibility;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScMaterialsAndVisibility = ScMaterialsAndVisibility;

impl crate::Showcase for ScMaterialsAndVisibility {
    fn needs_build(&self, app: &crate::App) -> bool {
        needs_build(app)
    }
    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        build(app, renderer)
    }
    fn scene(&self, app: &mut crate::App, frame: &crate::eframe::Frame, out: &mut crate::SceneOverrides) -> crate::SceneContents {
        scene(app, frame, out)
    }
    fn on_click(&self, app: &mut crate::App, cx: &crate::ClickCtx) {
        on_click(app, cx)
    }
    fn viewport_override(&self, app: &mut crate::App, ui: &mut crate::eframe::egui::Ui, cx: &crate::ViewportCtx) -> bool {
        viewport_override(app, ui, cx)
    }
    fn drive_camera(&self, app: &mut crate::App, cx: &crate::ViewportCtx) -> bool {
        drive_camera(app, cx)
    }
    fn suppress_orbit(&self, app: &crate::App, cx: &crate::ViewportCtx) -> bool {
        suppress_orbit(app, cx)
    }
    fn controls(&self, app: &mut crate::App, ui: &mut crate::eframe::egui::Ui, _frame: &crate::eframe::Frame) {
        controls_materials_visibility(app, ui)
    }
}
