//! Showcase 9: Annotation Labels : build and render methods.
//!
//! Labels now render natively via `LabelItem` in `OverlayFrame`, replacing the
//! previous egui painter approach.

use crate::App;
use crate::geometry::make_box_with_uvs;
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{Camera, LabelItem, Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct AnnotationState {
    pub built: bool,
    pub scene: Scene,
    pub labels: Vec<LabelItem>,
}

impl Default for AnnotationState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            labels: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 9 (Annotation Labels demo).
    pub(crate) fn build_annotation_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.ann_state.scene = Scene::new();

        let mut place_marker =
            |renderer: &mut ViewportRenderer, pos: glam::Vec3, colour: [f32; 3]| {
                let mesh = make_box_with_uvs(0.3, 0.3, 0.3);
                let id = renderer
                    .resources_mut()
                    .upload_mesh_data(&self.device, &mesh)
                    .expect("annotation marker upload");
                self.ann_state.scene.add_named(
                    "Marker",
                    Some(id),
                    glam::Mat4::from_translation(pos),
                    Material::from_colour(colour),
                );
            };

        place_marker(renderer, glam::Vec3::ZERO, [1.0, 1.0, 1.0]);
        place_marker(renderer, glam::Vec3::new(2.0, 3.0, 0.0), [1.0, 0.9, 0.1]);
        place_marker(renderer, glam::Vec3::new(-3.0, 2.0, 0.0), [0.4, 0.8, 1.0]);
        place_marker(renderer, glam::Vec3::new(0.0, 300.0, 0.0), [1.0, 0.0, 0.0]);

        self.ann_state.labels = vec![
            LabelItem::new("Origin (0,0,0)")
                .with_world_anchor([0.0, 0.0, 0.0])
                .with_colour([1.0, 1.0, 1.0, 1.0])
                .with_background(true),
            LabelItem::new("Peak Pressure: 101.3 kPa")
                .with_world_anchor([2.0, 3.0, 0.0])
                .with_colour([1.0, 0.9, 0.1, 1.0])
                .with_leader_line(true)
                .with_background(true),
            LabelItem::new("Outlet")
                .with_world_anchor([-3.0, 2.0, 0.0])
                .with_colour([0.4, 0.8, 1.0, 1.0])
                .with_background(true),
            LabelItem::new("Behind camera (clipped)")
                .with_world_anchor([0.0, 300.0, 0.0])
                .with_colour([1.0, 0.0, 0.0, 1.0])
                .with_background(true),
        ];

        self.ann_state.built = true;
    }

    /// Reset the camera to a good viewing angle for the annotation demo.
    pub(crate) fn reset_annotation_camera(&mut self) {
        self.camera = Camera {
            center: glam::Vec3::new(0.0, 0.5, 1.0),
            distance: 12.0,
            orientation: glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.1),
            ..Camera::default()
        };
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_annotation(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Labels render natively via OverlayFrame.");
    ui.separator();
    for (i, label) in app.ann_state.labels.iter().enumerate() {
        let status = if let vpl::OverlayAnchor::World(wa) = label.anchor {
            let view = app.camera.view_matrix();
            let proj = app.camera.proj_matrix();
            let pos = glam::Vec3::from(wa);
            let clip = proj * view * pos.extend(1.0);
            let visible = clip.w > 0.0 && {
                let ndc = glam::Vec3::new(clip.x, clip.y, clip.z) / clip.w;
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0
            };
            if visible { "visible" } else { "clipped" }
        } else {
            "screen-anchored"
        };
        ui.label(format!("L{i}: \"{}\" : {status}", label.text));
    }
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.ann_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_annotation_scene(renderer);
    app.reset_annotation_camera();
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
        let items = app.ann_state.scene.collect_render_items(&vpl::Selection::new());
        let sg = app.ann_state.scene.version();
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.hemisphere_intensity = 0.5;
            _t.sky_colour = [1.0, 1.0, 1.0].into();
            _t.ground_colour = [1.0, 1.0, 1.0].into();
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
    app: &mut crate::App,
    fd: &mut vpl::FrameData,
    _ctx: &crate::FrameCtx,
) {
    // Overlay labels (Showcase 9 and 34): populate OverlayFrame.
    if app.ann_state.built {
        fd.overlays.labels = app.ann_state.labels.clone();
    }
}

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
pub(crate) struct ScAnnotation;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScAnnotation = ScAnnotation;

impl crate::Showcase for ScAnnotation {
    fn needs_build(&self, app: &crate::App) -> bool {
        needs_build(app)
    }
    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        build(app, renderer)
    }
    fn scene(&self, app: &mut crate::App, frame: &crate::eframe::Frame, out: &mut crate::SceneOverrides) -> crate::SceneContents {
        scene(app, frame, out)
    }
    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
        frame(app, fd, ctx)
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
        controls_annotation(app, ui)
    }
}
