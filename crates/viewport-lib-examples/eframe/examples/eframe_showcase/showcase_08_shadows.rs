//! Showcase 8: Shadow Demo : build and controls.

use crate::App;
use crate::geometry::{make_box_with_uvs, make_uv_sphere};
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ShadowsState {
    pub built: bool,
    pub scene: Scene,
    pub cascade_count: u32,
    pub pcss_on: bool,
    pub contact_on: bool,
}

impl Default for ShadowsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            cascade_count: 4,
            pcss_on: false,
            contact_on: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_shadow_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.shd_state.scene = Scene::new();

        let ground_mesh = make_box_with_uvs(20.0, 20.0, 0.2);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("ground mesh upload");
        self.shd_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.1)),
            Material::pbr([1.0, 1.0, 1.0], 0.0, 0.9),
        );

        let sphere_mesh = make_uv_sphere(24, 12, 0.5);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("sphere mesh upload");

        let sphere_dense_mesh = make_uv_sphere(64, 32, 0.5);
        let sphere_dense_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_dense_mesh)
            .expect("dense sphere mesh upload");

        let box_mesh = make_box_with_uvs(1.0, 1.0, 1.0);
        let box_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &box_mesh)
            .expect("box mesh upload");

        let object_data: &[(&str, glam::Vec3, [f32; 3])] = &[
            (
                "Sphere Near",
                glam::Vec3::new(-1.5, 1.0, 0.5),
                [0.75, 0.28, 0.05],
            ),
            (
                "Box Near",
                glam::Vec3::new(1.5, 1.0, 0.5),
                [0.10, 0.26, 0.68],
            ),
            (
                "Sphere Mid",
                glam::Vec3::new(-3.0, -3.0, 0.5),
                [0.60, 0.50, 0.05],
            ),
            (
                "Box Mid",
                glam::Vec3::new(3.0, -3.0, 0.5),
                [0.10, 0.52, 0.18],
            ),
        ];
        for (name, pos, colour) in object_data {
            let mesh_id = if *name == "Sphere Near" {
                sphere_dense_id
            } else if name.contains("Sphere") {
                sphere_id
            } else {
                box_id
            };
            self.shd_state.scene.add_named(
                name,
                Some(mesh_id),
                glam::Mat4::from_translation(*pos),
                Material::pbr(*colour, 0.0, 0.5),
            );
        }

        let pillar_mesh = make_box_with_uvs(0.4, 0.4, 3.0);
        let pillar_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &pillar_mesh)
            .expect("pillar mesh upload");
        self.shd_state.scene.add_named(
            "Tall Pillar",
            Some(pillar_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, -6.0, 1.5)),
            Material::pbr([0.35, 0.35, 0.40], 0.0, 0.6),
        );

        self.shd_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_shadows(app: &mut App, ui: &mut egui::Ui) {
    ui.label(format!("Cascades: {}", app.shd_state.cascade_count));
    ui.horizontal(|ui| {
        if ui
            .button("-")
            .on_hover_text("Decrease cascade count")
            .clicked()
        {
            app.shd_state.cascade_count = (app.shd_state.cascade_count - 1).max(1);
        }
        if ui
            .button("+")
            .on_hover_text("Increase cascade count")
            .clicked()
        {
            app.shd_state.cascade_count = (app.shd_state.cascade_count + 1).min(4);
        }
    });

    ui.separator();
    ui.label("Filter:");
    ui.horizontal(|ui| {
        if ui.radio(!app.shd_state.pcss_on, "PCF").clicked() {
            app.shd_state.pcss_on = false;
        }
        if ui.radio(app.shd_state.pcss_on, "PCSS").clicked() {
            app.shd_state.pcss_on = true;
        }
    });

    ui.separator();
    ui.checkbox(&mut app.shd_state.contact_on, "Contact Shadows");
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.shd_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_shadow_scene(renderer);
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
        let items = app.shd_state.scene.collect_render_items(&vpl::Selection::new());
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.lights = vec![{
                let mut _t = vpl::LightSource::default();
                _t.kind = vpl::LightKind::Directional {
                    direction: [0.5, 0.2, 1.2],
                };
                _t.intensity = 2.0;
                _t
            }];
            _t.shadows.enabled = true;
            _t.shadows.cascade_count = app.shd_state.cascade_count;
            _t.shadows.filter = if app.shd_state.pcss_on {
                vpl::ShadowFilter::Pcss
            } else {
                vpl::ShadowFilter::Pcf
            };
            _t.hemisphere_intensity = 0.5;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        let sg = app.shd_state.scene.version();
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
    fd.effects.display.mode = vpl::PipelineMode::Direct;
    fd.effects.post_process = {
        let mut _t = vpl::PostProcessSettings::default();
        _t.contact_shadows.enabled = app.shd_state.contact_on;
        _t.contact_shadows.max_distance = 0.18;
        _t.contact_shadows.steps = 32;
        _t.contact_shadows.thickness = 0.04;
        _t
    };
    // Cap far plane for better cascade distribution, but track orbit
    // distance so the scene doesn't disappear when zooming out.
    let mut rc = vpl::RenderCamera::from_camera(&app.camera);
    rc.far = (app.camera.distance * 3.0).max(60.0);
    rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
    fd.camera.render_camera = rc;
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
pub(crate) struct ScShadows;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScShadows = ScShadows;

impl crate::Showcase for ScShadows {
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
        controls_shadows(app, ui)
    }
}
