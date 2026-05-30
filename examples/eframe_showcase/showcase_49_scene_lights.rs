//! Showcase 49: Scene-graph lights.
//!
//! Lights placed via `Scene::add_light` and collected into `SceneFrame::lights`
//! each frame. The renderer unions them with `EffectsFrame::lighting.lights`.
//!
//! The scene mixes surface meshes and instanced glyphs so both pipelines
//! respond to point and spot lights.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    GlyphItem, GlyphType, LightKind, LightSource, LightingSettings, Material,
    SceneRenderItem, Selection, ViewportRenderer,
    scene::Scene,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct SlState {
    pub built: bool,
    pub scene: Scene,
    pub light_ids: [u64; 3],
    pub lights: [LightSource; 3],
    pub animate: bool,
    pub time: f32,
    pub show_glyphs: bool,
    pub hemi_intensity: f32,
}

fn default_lights() -> [LightSource; 3] {
    let point = {
        let mut s = LightSource::default();
        s.kind = LightKind::Point { position: [5.0, 0.0, 3.0], range: 14.0 };
        s.colour = [1.0, 0.6, 0.2];
        s.intensity = 5.0;
        s
    };
    let spot = {
        let mut s = LightSource::default();
        s.kind = LightKind::Spot {
            position: [-5.0, 0.0, 6.0],
            direction: [0.0, 0.0, -1.0],
            range: 18.0,
            inner_angle: 0.2,
            outer_angle: 0.4,
        };
        s.colour = [0.4, 0.7, 1.0];
        s.intensity = 6.0;
        s
    };
    let dir = {
        let mut s = LightSource::default();
        s.kind = LightKind::Directional { direction: [0.3, 0.2, 1.0] };
        s.colour = [1.0, 1.0, 0.9];
        s.intensity = 0.4;
        s
    };
    [point, spot, dir]
}

impl Default for SlState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            light_ids: [0; 3],
            lights: default_lights(),
            animate: true,
            time: 0.0,
            show_glyphs: true,
            hemi_intensity: 0.15,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_sl_scene(&mut self, renderer: &mut ViewportRenderer) {
        let state = &mut self.sl_state;
        state.scene = Scene::new();

        let ground_mesh = make_box_with_uvs(18.0, 18.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("sl ground");
        state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.05)),
            { let mut m = Material::from_colour([0.28, 0.28, 0.3]); m.roughness = 0.95; m },
        );

        let sphere_mesh = viewport_lib::primitives::sphere(0.7, 32, 16);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("sl sphere");
        for row in 0..3i32 {
            for col in 0..3i32 {
                let x = (col - 1) as f32 * 4.5;
                let y = (row - 1) as f32 * 4.5;
                state.scene.add_named(
                    &format!("Sphere {row}{col}"),
                    Some(sphere_id),
                    glam::Mat4::from_translation(glam::Vec3::new(x, y, 0.7)),
                    { let mut m = Material::from_colour([0.88, 0.88, 0.9]); m.roughness = 0.3; m },
                );
            }
        }

        let lights = state.lights.clone();
        for (i, src) in lights.iter().enumerate() {
            let id = state.scene.add_light(src.clone());
            state.light_ids[i] = id;
        }

        state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Per-frame submission
// ---------------------------------------------------------------------------

pub(crate) fn submit_sl_items(app: &mut App, fd: &mut viewport_lib::FrameData) {
    if app.sl_state.animate {
        app.sl_state.time += 0.016;
    }
    let t = app.sl_state.time;
    let radius = 5.5_f32;

    // Orbit point and spot lights; update both the node transform and the
    // cached LightSource so the UI sliders show current values.
    let px = radius * t.cos();
    let py = radius * t.sin();
    app.sl_state.scene.set_local_transform(
        app.sl_state.light_ids[0],
        glam::Mat4::from_translation(glam::Vec3::new(px, py, 3.0)),
    );
    if let LightKind::Point { ref mut position, .. } = app.sl_state.lights[0].kind {
        *position = [px, py, 3.0];
    }

    let sx = radius * (t + std::f32::consts::PI).cos();
    let sy = radius * (t + std::f32::consts::PI).sin();
    app.sl_state.scene.set_local_transform(
        app.sl_state.light_ids[1],
        glam::Mat4::from_translation(glam::Vec3::new(sx, sy, 6.0)),
    );
    if let LightKind::Spot { ref mut position, .. } = app.sl_state.lights[1].kind {
        *position = [sx, sy, 6.0];
    }

    // Push edited colour/intensity back into each scene-graph light node.
    for i in 0..3 {
        let id = app.sl_state.light_ids[i];
        let src = app.sl_state.lights[i].clone();
        app.sl_state.scene.set_light(id, Some(src));
    }

    fd.scene.lights = app.sl_state.scene.collect_lights();

    // Glyph markers at each light position (non-mesh pipeline, also lit by scene lights).
    if app.sl_state.show_glyphs {
        let point_pos = if let LightKind::Point { position, .. } = app.sl_state.lights[0].kind {
            Some(position)
        } else { None };
        let spot_pos = if let LightKind::Spot { position, .. } = app.sl_state.lights[1].kind {
            Some(position)
        } else { None };

        if let Some(pos) = point_pos {
            let c = app.sl_state.lights[0].colour;
            let mut g = GlyphItem::default();
            g.glyph_type = GlyphType::Sphere;
            g.positions.push(pos);
            g.vectors.push([0.0, 0.0, 0.28]);
            g.scale = 0.28;
            g.scale_by_magnitude = false;
            g.use_default_colour = true;
            g.default_colour = [c[0], c[1], c[2], 1.0];
            fd.scene.glyphs.push(g);
        }
        if let Some(pos) = spot_pos {
            let c = app.sl_state.lights[1].colour;
            let mut g = GlyphItem::default();
            g.glyph_type = GlyphType::Arrow;
            g.positions.push(pos);
            g.vectors.push([0.0, 0.0, -0.28]);
            g.scale = 1.0;
            g.scale_by_magnitude = true;
            g.use_default_colour = true;
            g.default_colour = [c[0], c[1], c[2], 1.0];
            fd.scene.glyphs.push(g);
        }
    }
}

// ---------------------------------------------------------------------------
// Scene item collection
// ---------------------------------------------------------------------------

pub(crate) fn sl_collect(
    app: &mut App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64) {
    let items = app.sl_state.scene.collect_render_items(&Selection::new());
    let lighting = {
        let mut l = LightingSettings::default();
        l.lights = vec![]; // all lights come from the scene graph via SceneFrame::lights
        l.hemisphere_intensity = app.sl_state.hemi_intensity;
        l.sky_colour = [0.7, 0.8, 1.0];
        l.ground_colour = [0.4, 0.35, 0.3];
        l.shadows_enabled = false;
        l
    };
    let sg = app.sl_state.scene.version();
    (items, lighting, sg)
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_sl(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Lights live in the scene graph (scene.add_light). Each frame scene.collect_lights() feeds SceneFrame::lights; the renderer unions them with EffectsFrame::lighting.");
    ui.separator();

    ui.checkbox(&mut app.sl_state.animate, "Animate lights");
    ui.checkbox(&mut app.sl_state.show_glyphs, "Show position glyphs");
    ui.add(egui::Slider::new(&mut app.sl_state.hemi_intensity, 0.0..=1.0).text("Hemisphere ambient"));
    ui.separator();

    let names = ["Point (warm, orbits)", "Spot (cool, orbits)", "Directional (fill)"];
    for i in 0..3 {
        egui::CollapsingHeader::new(names[i])
            .id_salt(i + 200)
            .show(ui, |ui| {
                let src = &mut app.sl_state.lights[i];
                ui.horizontal(|ui| {
                    ui.label("Colour:");
                    ui.color_edit_button_rgb(&mut src.colour);
                });
                ui.add(egui::Slider::new(&mut src.intensity, 0.0..=12.0).text("Intensity"));
                #[allow(clippy::match_wildcard_for_catch_all)]
                match &mut src.kind {
                    LightKind::Point { range, .. } => {
                        ui.add(egui::Slider::new(range, 1.0..=40.0).text("Range"));
                    }
                    LightKind::Spot { range, inner_angle, outer_angle, .. } => {
                        ui.add(egui::Slider::new(range, 1.0..=40.0).text("Range"));
                        let mut id = inner_angle.to_degrees();
                        let mut od = outer_angle.to_degrees();
                        if ui.add(egui::Slider::new(&mut id, 1.0..=44.0).suffix("deg").text("Inner")).changed() {
                            *inner_angle = id.to_radians();
                        }
                        if ui.add(egui::Slider::new(&mut od, 2.0..=89.0).suffix("deg").text("Outer")).changed() {
                            *outer_angle = od.to_radians();
                        }
                    }
                    LightKind::Directional { direction } => {
                        ui.horizontal(|ui| {
                            ui.label("Direction:");
                            ui.add(egui::DragValue::new(&mut direction[0]).speed(0.02).prefix("X "));
                            ui.add(egui::DragValue::new(&mut direction[1]).speed(0.02).prefix("Y "));
                            ui.add(egui::DragValue::new(&mut direction[2]).speed(0.02).prefix("Z "));
                        });
                    }
                    _ => {}
                }
            });
    }
}
