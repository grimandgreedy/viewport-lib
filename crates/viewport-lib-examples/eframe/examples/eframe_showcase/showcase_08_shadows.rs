//! Showcase 8: Shadows.
//!
//! Directional cascaded shadows and a shadowed spot light on one scene of
//! spheres, boxes and a tall pillar.
//!
//! - The cascade stepper sets `ShadowSettings::cascade_count`; the cascade
//!   colour view (`DebugVis` with `DebugQuantity::CascadeIndex`) tints each
//!   cascade so its reach reads directly.
//! - Shadows filter with PCSS, which sizes the directional penumbra from
//!   `ShadowSettings::sun_angular_radius`: 0 is a hard edge, the real sun
//!   (about 0.0047 rad) is close to it at shadow-map resolution, and wider
//!   discs soften the pillar's shadow with distance from its base.
//! - The cascade fit: `shadow_distance` (fit to the casters, or end at a set
//!   distance), `cascade_split` (`CascadeSplit::Practical` with its `lambda`)
//!   and `fade_fraction`.
//! - A spot light with its own shadow aims at the pillar; a radio picks
//!   which light casts, so the two shadows of the same pillar compare. A spot
//!   penumbra comes from the light's `radius`.

use crate::App;
use crate::eframe::egui;
use crate::geometry::{make_box_with_uvs, make_uv_sphere};
use crate::shared::upload_material;
use viewport_lib as vpl;
use viewport_lib::Colour;
use vpl::{CascadeSplit, Material, ViewportRenderer, scene::Scene};

/// Which light casts the shadows.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Caster {
    Sun,
    Spot,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ShadowsState {
    pub built: bool,
    pub scene: Scene,
    pub cascade_count: u32,
    pub sun_angular_radius: f32,
    pub fit_to_casters: bool,
    pub shadow_distance: f32,
    pub split_lambda: f32,
    pub fade_fraction: f32,
    pub show_cascades: bool,
    pub caster: Caster,
    pub spot_radius: f32,
    pub contact_on: bool,
}

impl Default for ShadowsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            cascade_count: 4,
            sun_angular_radius: 0.01,
            fit_to_casters: true,
            shadow_distance: 20.0,
            split_lambda: 0.75,
            fade_fraction: 0.1,
            show_cascades: false,
            caster: Caster::Sun,
            spot_radius: 0.1,
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
            upload_material(
                renderer,
                Material::pbr(Colour::linear_rgb(0.7, 0.7, 0.7), 0.0, 0.9),
            ),
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
                upload_material(
                    renderer,
                    Material::pbr(Colour::from_linear_rgb_array(*colour), 0.0, 0.5),
                ),
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
            upload_material(
                renderer,
                Material::pbr(Colour::linear_rgb(0.35, 0.35, 0.40), 0.0, 0.6),
            ),
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

    let st = &mut app.shd_state;
    ui.checkbox(&mut st.show_cascades, "Colour the cascades");

    ui.separator();
    ui.label("Casting light:");
    ui.horizontal(|ui| {
        ui.radio_value(&mut st.caster, Caster::Sun, "Sun");
        ui.radio_value(&mut st.caster, Caster::Spot, "Spot");
    });

    ui.separator();
    match st.caster {
        Caster::Sun => {
            ui.label("Sun angular radius (PCSS penumbra):");
            ui.add(
                egui::Slider::new(&mut st.sun_angular_radius, 0.0..=0.06)
                    .fixed_decimals(4)
                    .text("rad"),
            );
            ui.horizontal(|ui| {
                for (label, r) in [("Hard", 0.0), ("Sun", 0.0047), ("Wide", 0.02)] {
                    if ui.small_button(label).clicked() {
                        st.sun_angular_radius = r;
                    }
                }
            });
            ui.weak(
                "A blocker h above a receiver casts a penumbra about 2 h tan(radius) wide, \
                 so the pillar's shadow hardens at its base and softens along its length.",
            );
        }
        Caster::Spot => {
            ui.add(egui::Slider::new(&mut st.spot_radius, 0.0..=0.5).text("Spot radius"));
            ui.weak("A spot or point penumbra is sized by the light's radius.");
        }
    }

    ui.separator();
    ui.label("Cascade fit:");
    ui.checkbox(&mut st.fit_to_casters, "Fit range to the casters");
    ui.add_enabled(
        !st.fit_to_casters,
        egui::Slider::new(&mut st.shadow_distance, 2.0..=60.0).text("Shadow distance"),
    );
    ui.add(egui::Slider::new(&mut st.split_lambda, 0.0..=1.0).text("Split lambda (log vs even)"));
    ui.add(egui::Slider::new(&mut st.fade_fraction, 0.0..=0.5).text("Fade fraction"));

    ui.separator();
    ui.checkbox(&mut st.contact_on, "Contact shadows");
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
    app.camera = vpl::Camera::orbit(
        glam::DVec3::new(0.0, 0.0, 1.0),
        14.0,
        glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.0),
    );
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
            .shd_state
            .scene
            .collect_render_items(&vpl::Selection::new());
        let st = &app.shd_state;
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            let mut sun = vpl::LightSource::default();
            sun.kind = vpl::LightKind::Directional {
                direction: [0.5, 0.2, 1.2],
            };
            // With the spot casting, the sun drops to a dim fill so the
            // spot's pool and its shadow of the pillar read.
            sun.intensity = if st.caster == Caster::Sun { 2.0 } else { 0.3 };
            sun.cast_shadows = st.caster == Caster::Sun;
            // A spot behind the pillar, aimed at its middle, so its shadow
            // falls across the scene towards the spheres.
            let mut spot = vpl::LightSource::default();
            spot.kind = vpl::LightKind::Spot {
                position: glam::DVec3::new(1.5, -10.0, 4.0),
                direction: [-1.5, 4.0, -2.5],
                range: 25.0,
                inner_angle: 0.35,
                outer_angle: 0.55,
                radius: st.spot_radius,
            };
            spot.colour = Colour::linear_rgb(1.0, 0.85, 0.65);
            spot.intensity = 25.0;
            spot.cast_shadows = st.caster == Caster::Spot;
            _t.lights = vec![sun, spot];
            _t.shadows.enabled = true;
            _t.shadows.cascade_count = st.cascade_count;
            _t.shadows.filter = vpl::ShadowFilter::Pcss;
            _t.shadows.sun_angular_radius = st.sun_angular_radius;
            _t.shadows.shadow_distance = (!st.fit_to_casters).then_some(st.shadow_distance);
            _t.shadows.cascade_split = CascadeSplit::Practical {
                lambda: st.split_lambda,
            };
            _t.shadows.fade_fraction = st.fade_fraction;
            _t.hemisphere_intensity = 0.5;
            _t.sky_colour = Colour::linear_rgb(1.0, 1.0, 1.0);
            _t.ground_colour = Colour::linear_rgb(1.0, 1.0, 1.0);
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
pub(crate) fn frame(app: &mut crate::App, fd: &mut vpl::FrameData, _ctx: &crate::FrameCtx) {
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
    let cam = &app.camera;
    let far = (cam.focus_distance() * 3.0).max(60.0);
    fd.camera.render_camera =
        vpl::RenderCamera::from_camera(cam).with_depth_range(cam.depth_range().0, far);
    if app.shd_state.show_cascades {
        fd.effects.debug.debug_vis = vpl::DebugVis::new(
            vpl::DebugOutputMode::TintOverlay,
            vpl::DebugQuantity::CascadeIndex,
            vpl::DebugQuantity::Zero,
            vpl::DebugQuantity::One,
            1.0,
        );
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
    fn scene(
        &self,
        app: &mut crate::App,
        frame: &crate::eframe::Frame,
        out: &mut crate::SceneOverrides,
    ) -> crate::SceneContents {
        scene(app, frame, out)
    }
    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
        frame(app, fd, ctx)
    }
    fn viewport_override(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        cx: &crate::ViewportCtx,
    ) -> bool {
        viewport_override(app, ui, cx)
    }
    fn drive_camera(&self, app: &mut crate::App, cx: &crate::ViewportCtx) -> bool {
        drive_camera(app, cx)
    }
    fn suppress_orbit(&self, app: &crate::App, cx: &crate::ViewportCtx) -> bool {
        suppress_orbit(app, cx)
    }
    fn controls(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        _frame: &crate::eframe::Frame,
    ) {
        controls_shadows(app, ui)
    }
    fn tour(&self, app: &mut crate::App, step: u32) -> bool {
        let st = &mut app.shd_state;
        match step {
            0 => {}
            1 => st.show_cascades = true,
            2 => {
                st.show_cascades = false;
                st.sun_angular_radius = 0.04;
            }
            3 => st.caster = Caster::Spot,
            _ => return false,
        }
        true
    }
}
