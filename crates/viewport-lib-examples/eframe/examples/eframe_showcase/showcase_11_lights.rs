//! Showcase 11: Lights : build and controls.
//!
//! A flat ground plane plus a 3x3 grid of white spheres : neutral surfaces that make
//! light colour, cone angle, and attenuation directly visible.
//! One additional sphere in the corner uses `Material::unlit` to show the raw base
//! colour without any lighting contribution.

use crate::App;
use crate::geometry::make_box_with_uvs;
use crate::eframe::egui;
use viewport_lib as vpl;
use vpl::{LightKind, LightSource, Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct LightsState {
    pub built: bool,
    pub scene: Scene,
    pub sources: Vec<LightSource>,
    pub hemi_on: bool,
    pub hemi_intensity: f32,
    pub sky_colour: [f32; 3],
    pub ground_colour: [f32; 3],
    pub edl_enabled: bool,
    pub edl_radius: f32,
    pub edl_strength: f32,
    pub unlit_sphere: bool,
}

impl Default for LightsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            sources: vec![LightSource::default()],
            hemi_on: true,
            hemi_intensity: 0.7,
            sky_colour: [1.0, 1.0, 1.0],
            ground_colour: [1.0, 1.0, 1.0],
            edl_enabled: false,
            edl_radius: 1.0,
            edl_strength: 1.0,
            unlit_sphere: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 11 (Lights demo).
    pub(crate) fn build_lights_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.lights_state.scene = Scene::new();

        // Ground plane : thin slab, Z-up.
        let ground_mesh = make_box_with_uvs(16.0, 16.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("lights ground mesh");
        self.lights_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.05)),
            {
                let mut m = Material::from_colour([0.45, 0.45, 0.48]);
                m.roughness = 0.9;
                m
            },
        );

        // 3x3 grid of lit spheres.
        let sphere_mesh = vpl::primitives::sphere(0.6, 32, 16);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("lights sphere mesh");

        for row in 0..3i32 {
            for col in 0..3i32 {
                let x = (col - 1) as f32 * 4.0;
                let y = (row - 1) as f32 * 4.0;
                let z = 0.6_f32; // rest on ground
                let name = format!("Sphere ({row},{col})");
                self.lights_state.scene.add_named(
                    &name,
                    Some(sphere_id),
                    glam::Mat4::from_translation(glam::Vec3::new(x, y, z)),
                    {
                        let mut m = Material::from_colour([0.92, 0.92, 0.92]);
                        m.roughness = 0.35;
                        m
                    },
                );
            }
        }

        // Unlit sphere in the corner : shows the raw base colour regardless of scene
        // lighting. A lit sphere of the same colour sits beside it for comparison.
        // Visibility is controlled at frame time via the toggle, not by rebuilding.
        let unlit_id = self.lights_state.scene.add_named(
            "Unlit Sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(6.0, -6.0, 0.6)),
            Material::from_colour([0.12, 0.38, 0.82]),
        );
        {
            let mut a = vpl::ItemSettings::default();
            a.unlit = true;
            self.lights_state.scene.set_appearance(unlit_id, a);
        }
        self.lights_state.scene.add_named(
            "Lit Sphere (same colour)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(6.0, -2.0, 0.6)),
            {
                let mut m = Material::from_colour([0.12, 0.38, 0.82]);
                m.roughness = 0.35;
                m
            },
        );

        self.lights_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_lights(app: &mut App, ui: &mut egui::Ui) {
    ui.label(format!("Lights: {}", app.lights_state.sources.len()));
    ui.separator();

    ui.horizontal(|ui| {
        if ui.button("+ Directional").clicked() && app.lights_state.sources.len() < 8 {
            app.lights_state.sources.push({
                let mut _t = LightSource::default();
                _t.kind = LightKind::Directional {
                    direction: [0.4, 0.3, 1.5],
                };
                _t.colour = [1.0, 1.0, 1.0].into();
                _t.intensity = 1.0;
                _t
            });
        }
        if ui.button("+ Point").clicked() && app.lights_state.sources.len() < 8 {
            app.lights_state.sources.push({
                let mut _t = LightSource::default();
                _t.kind = LightKind::Point {
                    position: [0.0, 3.0, 3.0],
                    range: 15.0,
                    radius: 0.1,
                };
                _t.colour = [1.0, 0.9, 0.7].into();
                // Candela-scale: inverse-square falloff means intensity ~= E * d^2,
                // so lighting spheres a few units away needs tens, not single digits.
                _t.intensity = 30.0;
                _t
            });
        }
        if ui.button("+ Spot").clicked() && app.lights_state.sources.len() < 8 {
            app.lights_state.sources.push({
                let mut _t = LightSource::default();
                _t.kind = LightKind::Spot {
                    position: [0.0, 3.0, 6.0],
                    direction: [0.0, 0.0, -1.0],
                    range: 20.0,
                    inner_angle: 0.25,
                    outer_angle: 0.45,
                    radius: 0.1,
                };
                _t.colour = [0.8, 0.95, 1.0].into();
                _t.intensity = 70.0;
                _t
            });
        }
    });

    if ui.button("Reset to Default").clicked() {
        app.lights_state.sources = vec![LightSource::default()];
    }

    ui.separator();

    egui::ScrollArea::vertical()
        .max_height(300.0)
        .show(ui, |ui| {
            let mut to_remove: Option<usize> = None;
            let count = app.lights_state.sources.len();
            for i in 0..count {
                let kind_label = match app.lights_state.sources[i].kind {
                    LightKind::Directional { .. } => "Directional",
                    LightKind::Point { .. } => "Point",
                    LightKind::Spot { .. } => "Spot",
                    _ => "Unknown",
                };
                egui::CollapsingHeader::new(format!("Light {i} ({kind_label})"))
                    .id_salt(i)
                    .show(ui, |ui| {
                        let src = &mut app.lights_state.sources[i];

                        // Colour
                        ui.horizontal(|ui| {
                            ui.label("Colour:");
                            let mut c = src.colour;
                            if ui.color_edit_button_rgb(&mut c).changed() {
                                src.colour = c.into();
                            }
                        });

                        // Intensity
                        ui.horizontal(|ui| {
                            ui.label("Intensity:");
                            // Wide range: directional reads in single digits (lux-like),
                            // but point/spot need tens under inverse-square (candela-like).
                            ui.add(egui::Slider::new(&mut src.intensity, 0.0..=100.0));
                        });

                        // Kind-specific params
                        #[allow(clippy::match_wildcard_for_catch_all)]
                        match &mut src.kind {
                            LightKind::Directional { direction } => {
                                ui.label("Direction (toward light):");
                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    ui.add(egui::DragValue::new(&mut direction[0]).speed(0.02));
                                    ui.label("Y:");
                                    ui.add(egui::DragValue::new(&mut direction[1]).speed(0.02));
                                    ui.label("Z:");
                                    ui.add(egui::DragValue::new(&mut direction[2]).speed(0.02));
                                });
                            }
                            LightKind::Point {
                                position, range, ..
                            } => {
                                ui.label("Position:");
                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    ui.add(egui::DragValue::new(&mut position[0]).speed(0.1));
                                    ui.label("Y:");
                                    ui.add(egui::DragValue::new(&mut position[1]).speed(0.1));
                                    ui.label("Z:");
                                    ui.add(egui::DragValue::new(&mut position[2]).speed(0.1));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Range:");
                                    ui.add(egui::Slider::new(range, 1.0..=50.0));
                                });
                            }
                            LightKind::Spot {
                                position,
                                direction,
                                range,
                                inner_angle,
                                outer_angle,
                                ..
                            } => {
                                ui.label("Position:");
                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    ui.add(egui::DragValue::new(&mut position[0]).speed(0.1));
                                    ui.label("Y:");
                                    ui.add(egui::DragValue::new(&mut position[1]).speed(0.1));
                                    ui.label("Z:");
                                    ui.add(egui::DragValue::new(&mut position[2]).speed(0.1));
                                });
                                ui.label("Direction:");
                                ui.horizontal(|ui| {
                                    ui.label("X:");
                                    ui.add(egui::DragValue::new(&mut direction[0]).speed(0.02));
                                    ui.label("Y:");
                                    ui.add(egui::DragValue::new(&mut direction[1]).speed(0.02));
                                    ui.label("Z:");
                                    ui.add(egui::DragValue::new(&mut direction[2]).speed(0.02));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Range:");
                                    ui.add(egui::Slider::new(range, 1.0..=50.0));
                                });
                                let mut inner_deg = inner_angle.to_degrees();
                                let mut outer_deg = outer_angle.to_degrees();
                                ui.horizontal(|ui| {
                                    ui.label("Inner cone:");
                                    if ui
                                        .add(
                                            egui::Slider::new(&mut inner_deg, 1.0..=45.0)
                                                .suffix("\u{b0}"),
                                        )
                                        .changed()
                                    {
                                        *inner_angle = inner_deg.to_radians();
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Outer cone:");
                                    if ui
                                        .add(
                                            egui::Slider::new(&mut outer_deg, 2.0..=89.0)
                                                .suffix("\u{b0}"),
                                        )
                                        .changed()
                                    {
                                        *outer_angle = outer_deg.to_radians();
                                    }
                                });
                            }
                            _ => {}
                        }

                        if ui.button("Remove").clicked() {
                            to_remove = Some(i);
                        }
                    });
            }
            if let Some(idx) = to_remove {
                app.lights_state.sources.remove(idx);
            }
        });

    ui.separator();
    ui.checkbox(&mut app.lights_state.hemi_on, "Hemisphere Ambient");
    if app.lights_state.hemi_on {
        ui.add(
            egui::Slider::new(&mut app.lights_state.hemi_intensity, 0.0..=1.0).text("Intensity"),
        );
        ui.horizontal(|ui| {
            ui.label("Sky:");
            ui.color_edit_button_rgb(&mut app.lights_state.sky_colour);
        });
        ui.horizontal(|ui| {
            ui.label("Ground:");
            ui.color_edit_button_rgb(&mut app.lights_state.ground_colour);
        });
    }

    ui.separator();
    ui.label("Unlit shading");
    ui.checkbox(&mut app.lights_state.unlit_sphere, "Unlit sphere (corner)");

    ui.separator();
    ui.label("Eye-Dome Lighting");
    ui.checkbox(&mut app.lights_state.edl_enabled, "EDL enabled");
    if app.lights_state.edl_enabled {
        ui.add(
            egui::Slider::new(&mut app.lights_state.edl_radius, 0.5..=8.0)
                .text("Radius (px)")
                .step_by(0.5),
        );
        ui.add(egui::Slider::new(&mut app.lights_state.edl_strength, 0.0..=5.0).text("Strength"));
    }
}

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.lights_state.built
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    app.build_lights_scene(renderer);
    app.camera = vpl::Camera {
        center: glam::Vec3::ZERO,
        distance: 14.0,
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
        let mut items = app
            .lights_state
            .scene
            .collect_render_items(&vpl::Selection::new());
        if !app.lights_state.unlit_sphere {
            items.retain(|item| !item.settings.unlit);
        }
        let lighting = {
            let mut _t = vpl::LightingSettings::default();
            _t.lights = app.lights_state.sources.clone();
            _t.hemisphere_intensity = if app.lights_state.hemi_on {
                app.lights_state.hemi_intensity
            } else {
                0.0
            };
            _t.sky_colour = app.lights_state.sky_colour;
            _t.ground_colour = app.lights_state.ground_colour;
            _t
        };
        let sg = app.lights_state.scene.version();
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
    // Cap the far plane so depth values span a useful range for EDL.
    // Without this, all scene geometry clusters near depth 0.99, making
    // the log-space neighbor differences too small to see.
    let mut rc = vpl::RenderCamera::from_camera(&app.camera);
    rc.far = (app.camera.distance * 3.0).max(30.0);
    rc.projection = glam::Mat4::perspective_rh(rc.fov, rc.aspect, rc.near, rc.far);
    fd.camera.render_camera = rc;
    if app.lights_state.edl_enabled {
        fd.effects.post_process = {
            let mut _t = vpl::PostProcessSettings::default();
            _t.edl.enabled = true;
            _t.edl.radius = app.lights_state.edl_radius;
            _t.edl.strength = app.lights_state.edl_strength;
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
