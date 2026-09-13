//! Showcase 6: Post-Processing : build and controls.
//!
//! Post-processing showcase: bloom, SSAO, FXAA, tone mapping, and EDL controls.
//! Uses `prepare_callback` + `paint_callback` which dispatches to the full HDR
//! pipeline when `PostProcessSettings::enabled` is true (the default).

use crate::App;
use crate::geometry::make_uv_sphere;
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct PostProcessState {
    pub built: bool,
    pub scene: Scene,
    pub shadow_pcss: bool,
    pub point_light_on: bool,
    pub dir_intensity: f32,
    pub dof_enabled: bool,
    pub dof_focal_dist: f32,
    pub dof_focal_range: f32,
    pub dof_max_blur: f32,
}

impl Default for PostProcessState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            shadow_pcss: true,
            point_light_on: true,
            dir_intensity: 0.4,
            dof_enabled: false,
            dof_focal_dist: 5.0,
            dof_focal_range: 1.0,
            dof_max_blur: 8.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 6 (post-processing / PBR scene).
    pub(crate) fn build_pp_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.pp_state.scene = Scene::new();

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Ground",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(10.0, 10.0, 0.15),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, -0.575),
            ),
            Material::pbr([1.0, 1.0, 1.0], 0.0, 0.9),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Gold (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, -1.2, 0.0)),
            Material::pbr([1.0, 0.72, 0.06], 0.95, 0.05),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Brushed Steel (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, -1.2, 0.0)),
            Material::pbr([0.82, 0.82, 0.86], 0.75, 0.35),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Chrome (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(-1.2, 1.2, 0.0)),
            Material::pbr([0.9, 0.9, 0.95], 1.0, 0.02),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Ceramic (PBR)",
            Some(m),
            glam::Mat4::from_translation(glam::Vec3::new(1.2, 1.2, 0.0)),
            Material::pbr([0.82, 0.58, 0.42], 0.0, 0.9),
        );

        let sphere = make_uv_sphere(32, 16, 0.6);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("pp sphere upload");
        self.pp_state.scene.add_named(
            "Sphere Test",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(3.0, 0.0, 0.1)),
            Material::pbr([0.75, 0.28, 0.05], 0.0, 0.55),
        );

        let m = self.upload_box(renderer);
        self.pp_state.scene.add_named(
            "Pillar",
            Some(m),
            glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::new(0.5, 0.5, 2.5),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.0, 0.0, 1.0),
            ),
            Material::pbr([0.4, 0.4, 0.45], 0.1, 0.7),
        );

        self.pp_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_post_process(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Lighting:");
    ui.add(egui::Slider::new(&mut app.pp_state.dir_intensity, 0.0..=5.0).text("Dir. intensity"));
    ui.checkbox(&mut app.pp_state.point_light_on, "Point light");

    ui.separator();
    ui.label("Shadows:");
    ui.checkbox(&mut app.pp_state.shadow_pcss, "PCSS (soft shadows)");

    ui.separator();
    ui.label("Depth of Field:");
    ui.checkbox(&mut app.pp_state.dof_enabled, "Enable DoF");
    if app.pp_state.dof_enabled {
        ui.add(
            egui::Slider::new(&mut app.pp_state.dof_focal_dist, 0.5..=30.0).text("Focal distance"),
        );
        ui.add(
            egui::Slider::new(&mut app.pp_state.dof_focal_range, 0.1..=10.0).text("Focal range"),
        );
        ui.add(egui::Slider::new(&mut app.pp_state.dof_max_blur, 1.0..=20.0).text("Max blur (px)"));
    }

    ui.separator();
    ui.weak("Bloom, SSAO, and FXAA can be enabled via PostProcessSettings\nbut are not wired to controls in this showcase.");
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.pp_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_pp_scene(renderer);
    app.camera = vpl::Camera {
        center: glam::Vec3::new(0.0, 0.0, 0.5),
        distance: 8.0,
        orientation: glam::Quat::from_rotation_z(0.6)
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
        let items = app.pp_state.scene.collect_render_items(&vpl::Selection::new());
        let mut lights = vec![{
            let mut _t = vpl::LightSource::default();
            _t.kind = vpl::LightKind::Directional {
                direction: [0.6, 0.4, 1.0],
            };
            _t.intensity = app.pp_state.dir_intensity;
            _t
        }];
        if app.pp_state.point_light_on {
            lights.push({
                let mut _t = vpl::LightSource::default();
                _t.kind = vpl::LightKind::Point {
                    position: [3.0, 3.0, 3.0],
                    range: 15.0,
                    radius: 0.1,
                };
                _t.colour = [1.0, 0.9, 0.7].into();
                _t.intensity = 20.0;
                // Warm fill only. With two hard casters the shadows
                // overlap as a two-tone shape with a seam; one key
                // caster plus non-casting fill is the intended
                // lighting pattern.
                _t.cast_shadows = false;
                _t
            });
        }
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.lights = lights;
            _t.shadows.enabled = true;
            _t.shadows.filter = if app.pp_state.shadow_pcss {
                vpl::ShadowFilter::Pcss
            } else {
                vpl::ShadowFilter::Pcf
            };
            _t.hemisphere_intensity = 0.4;
            _t.sky_colour = [1.0, 1.0, 1.0];
            _t.ground_colour = [1.0, 1.0, 1.0];
            _t
        };
        let sg = app.pp_state.scene.version();
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
    // Cap far plane for better cascade distribution, but track orbit
    // distance so the scene doesn't disappear when zooming out.
    let mut rc = vpl::RenderCamera::from_camera(&app.camera);
    rc.far = (app.camera.distance * 3.0).max(60.0);
    rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
    fd.camera.render_camera = rc;
    if app.pp_state.dof_enabled {
        fd.effects.post_process = {
            let mut _t = vpl::PostProcessSettings::default();
            _t.dof.enabled = true;
            _t.dof.focal_distance = app.pp_state.dof_focal_dist;
            _t.dof.focal_range = app.pp_state.dof_focal_range;
            _t.dof.max_blur_radius = app.pp_state.dof_max_blur;
            _t
        };
    }
}

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
