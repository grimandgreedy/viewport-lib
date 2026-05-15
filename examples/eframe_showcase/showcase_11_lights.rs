//! Showcase 11: Lights : build and controls.
//!
//! A flat ground plane plus a 3x3 grid of white spheres : neutral surfaces that make
//! light color, cone angle, and attenuation directly visible.
//! One additional sphere in the corner uses `Material::unlit` to show the raw base
//! color without any lighting contribution.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{LightKind, LightSource, Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct LightsState {
    pub built: bool,
    pub scene: Scene,
    pub sources: Vec<LightSource>,
    pub hemi_on: bool,
    pub hemi_intensity: f32,
    pub sky_color: [f32; 3],
    pub ground_color: [f32; 3],
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
            hemi_intensity: 1.0,
            sky_color: [1.0, 1.0, 1.0],
            ground_color: [1.0, 1.0, 1.0],
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
                let mut m = Material::from_color([0.45, 0.45, 0.48]);
                m.roughness = 0.9;
                m
            },
        );

        // 3x3 grid of lit spheres.
        let sphere_mesh = viewport_lib::primitives::sphere(0.6, 32, 16);
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
                        let mut m = Material::from_color([0.92, 0.92, 0.92]);
                        m.roughness = 0.35;
                        m
                    },
                );
            }
        }

        // Unlit sphere in the corner : shows the raw base color regardless of scene
        // lighting. A lit sphere of the same color sits beside it for comparison.
        // Visibility is controlled at frame time via the toggle, not by rebuilding.
        self.lights_state.scene.add_named(
            "Unlit Sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(6.0, -6.0, 0.6)),
            {
                let mut m = Material::from_color([0.2, 0.7, 1.0]);
                m.unlit = true;
                m
            },
        );
        self.lights_state.scene.add_named(
            "Lit Sphere (same color)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(6.0, -2.0, 0.6)),
            {
                let mut m = Material::from_color([0.2, 0.7, 1.0]);
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
            app.lights_state.sources.push(LightSource {
                kind: LightKind::Directional {
                    direction: [0.4, 0.3, 1.5],
                },
                color: [1.0, 1.0, 1.0],
                intensity: 1.0,
            });
        }
        if ui.button("+ Point").clicked() && app.lights_state.sources.len() < 8 {
            app.lights_state.sources.push(LightSource {
                kind: LightKind::Point {
                    position: [0.0, 3.0, 3.0],
                    range: 15.0,
                },
                color: [1.0, 0.9, 0.7],
                intensity: 2.0,
            });
        }
        if ui.button("+ Spot").clicked() && app.lights_state.sources.len() < 8 {
            app.lights_state.sources.push(LightSource {
                kind: LightKind::Spot {
                    position: [0.0, 3.0, 6.0],
                    direction: [0.0, 0.0, -1.0],
                    range: 20.0,
                    inner_angle: 0.25,
                    outer_angle: 0.45,
                },
                color: [0.8, 0.95, 1.0],
                intensity: 3.0,
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

                        // Color
                        ui.horizontal(|ui| {
                            ui.label("Color:");
                            let mut c = src.color;
                            if ui.color_edit_button_rgb(&mut c).changed() {
                                src.color = c;
                            }
                        });

                        // Intensity
                        ui.horizontal(|ui| {
                            ui.label("Intensity:");
                            ui.add(egui::Slider::new(&mut src.intensity, 0.0..=10.0));
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
                            LightKind::Point { position, range } => {
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
            ui.color_edit_button_rgb(&mut app.lights_state.sky_color);
        });
        ui.horizontal(|ui| {
            ui.label("Ground:");
            ui.color_edit_button_rgb(&mut app.lights_state.ground_color);
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
