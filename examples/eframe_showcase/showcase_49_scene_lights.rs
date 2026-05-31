//! Showcase 49: Scene-graph lights.
//!
//! Lights placed via `Scene::add_light` and collected into `SceneFrame::lights`
//! each frame. The renderer unions them with `EffectsFrame::lighting.lights`.
//!
//! Two tabs:
//!   - Basics: three orbiting lights (point, spot, directional) over a small
//!     grid of spheres. Built-in glyphs mark each light position; per-light
//!     colour / intensity / range sliders.
//!   - Stress: a dense grid of point lights at variable density, demonstrating
//!     the storage-buffer path. The `importance` field controls which lights
//!     survive when the count exceeds the renderer's per-frame cap.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    LightKind, LightSource, LightingSettings, Material,
    SceneRenderItem, Selection, ViewportRenderer,
    scene::{Scene, build_light_glyphs},
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum SlTab {
    Basics,
    Stress,
}

pub(crate) struct SlState {
    pub built: bool,
    pub tab: SlTab,
    pub active_tab: SlTab,

    pub scene: Scene,
    pub time: f32,

    // -- Basics tab --
    pub light_ids: [u64; 3],
    pub lights: [LightSource; 3],
    pub animate: bool,
    pub show_glyphs: bool,
    pub hemi_intensity: f32,

    // -- Stress tab --
    pub stress_count: u32,
    pub stress_radius: f32,
    pub stress_intensity: f32,
    pub stress_animate: bool,
    pub stress_show_glyphs: bool,
    pub stress_importance_falloff: f32,
    pub stress_light_ids: Vec<u64>,
    pub stress_sources: Vec<LightSource>,
    pub stress_seed: u32,
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
            tab: SlTab::Basics,
            active_tab: SlTab::Basics,
            scene: Scene::new(),
            time: 0.0,

            light_ids: [0; 3],
            lights: default_lights(),
            animate: true,
            show_glyphs: true,
            hemi_intensity: 0.15,

            stress_count: 80,
            stress_radius: 4.5,
            stress_intensity: 5.0,
            stress_animate: true,
            stress_show_glyphs: true,
            stress_importance_falloff: 0.6,
            stress_light_ids: Vec::new(),
            stress_sources: Vec::new(),
            stress_seed: 0xC0FFEE,
        }
    }
}

// ---------------------------------------------------------------------------
// Tiny deterministic LCG so the demo doesn't pull in a `rand` dependency.
// ---------------------------------------------------------------------------

fn lcg_next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn rand_unit(state: &mut u32) -> f32 {
    (lcg_next(state) >> 8) as f32 / (1u32 << 24) as f32
}

fn rand_range(state: &mut u32, lo: f32, hi: f32) -> f32 {
    lo + rand_unit(state) * (hi - lo)
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_sl_scene(&mut self, renderer: &mut ViewportRenderer) {
        let active = self.sl_state.tab;
        self.sl_state.scene = Scene::new();
        self.sl_state.active_tab = active;
        self.sl_state.time = 0.0;

        match active {
            SlTab::Basics => self.build_sl_basics(renderer),
            SlTab::Stress => self.build_sl_stress(renderer),
        }

        self.sl_state.built = true;
    }

    fn build_sl_basics(&mut self, renderer: &mut ViewportRenderer) {
        let ground_mesh = make_box_with_uvs(18.0, 18.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("sl ground");
        self.sl_state.scene.add_named(
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
                self.sl_state.scene.add_named(
                    &format!("Sphere {row}{col}"),
                    Some(sphere_id),
                    glam::Mat4::from_translation(glam::Vec3::new(x, y, 0.7)),
                    { let mut m = Material::from_colour([0.88, 0.88, 0.9]); m.roughness = 0.3; m },
                );
            }
        }

        let lights = self.sl_state.lights.clone();
        for (i, src) in lights.iter().enumerate() {
            let id = self.sl_state.scene.add_light(src.clone());
            self.sl_state.light_ids[i] = id;
        }
    }

    fn build_sl_stress(&mut self, renderer: &mut ViewportRenderer) {
        // Large dark ground so the per-light pools of illumination read.
        let ground_mesh = make_box_with_uvs(40.0, 40.0, 0.2);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("sl stress ground");
        self.sl_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.1)),
            { let mut m = Material::from_colour([0.10, 0.10, 0.11]); m.roughness = 0.95; m },
        );

        // A scattered grid of small white pillars to catch the light from
        // different directions and angles.
        let pillar_mesh = make_box_with_uvs(0.8, 0.8, 1.6);
        let pillar_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &pillar_mesh)
            .expect("sl stress pillar");
        for row in -4..=4i32 {
            for col in -4..=4i32 {
                let x = col as f32 * 3.0;
                let y = row as f32 * 3.0;
                self.sl_state.scene.add_named(
                    &format!("Pillar {row}{col}"),
                    Some(pillar_id),
                    glam::Mat4::from_translation(glam::Vec3::new(x, y, 0.8)),
                    { let mut m = Material::from_colour([0.85, 0.85, 0.9]); m.roughness = 0.6; m },
                );
            }
        }

        rebuild_stress_lights(&mut self.sl_state);
    }
}

/// Replace the stress-tab light set in-place. Called when count, seed,
/// radius, intensity, or the importance falloff changes.
fn rebuild_stress_lights(state: &mut SlState) {
    for id in state.stress_light_ids.drain(..) {
        state.scene.remove(id);
    }
    state.stress_sources.clear();

    let mut rng = state.stress_seed;

    // One always-on directional fill so the scene isn't pitch black when
    // every point light gets dropped by the importance fallback.
    let mut dir = LightSource::default();
    dir.kind = LightKind::Directional { direction: [0.25, 0.3, 1.0] };
    dir.colour = [0.6, 0.7, 0.9];
    dir.intensity = 0.15;
    dir.importance = 10.0; // Always survive the cap.
    let dir_id = state.scene.add_light(dir.clone());
    state.stress_light_ids.push(dir_id);
    state.stress_sources.push(dir);

    let extent = 14.0;
    let count = state.stress_count;
    for i in 0..count {
        let x = rand_range(&mut rng, -extent, extent);
        let y = rand_range(&mut rng, -extent, extent);
        let z = rand_range(&mut rng, 0.4, 2.8);
        let r = rand_unit(&mut rng).powf(0.5);
        let g = rand_unit(&mut rng).powf(0.5);
        let b = rand_unit(&mut rng).powf(0.5);
        let max = r.max(g).max(b).max(0.0001);
        // Saturate the colour so individual pools read as distinct hues.
        let colour = [r / max, g / max, b / max];

        // Importance hint: lights near the centre matter more so the camera
        // (which orbits the origin) sees the surviving subset when the cap
        // is hit. Falloff slider lets the user lerp between flat and
        // strongly-prioritised distributions.
        let dist_from_origin = (x * x + y * y).sqrt() / extent;
        let importance =
            (1.0 - state.stress_importance_falloff * dist_from_origin).max(0.05);

        let mut src = LightSource::default();
        src.kind = LightKind::Point {
            position: [x, y, z],
            range: state.stress_radius,
        };
        src.colour = colour;
        src.intensity = state.stress_intensity;
        src.importance = importance;

        let id = state.scene.add_light(src.clone());
        state.stress_light_ids.push(id);
        state.stress_sources.push(src);

        // Keep the seed deterministic across slider tweaks: don't advance
        // the global seed here, but interleave more entropy into rng so
        // each light's colour and position are distinct.
        let _ = i;
    }
}

// ---------------------------------------------------------------------------
// Per-frame submission
// ---------------------------------------------------------------------------

pub(crate) fn submit_sl_items(app: &mut App, fd: &mut viewport_lib::FrameData) {
    match app.sl_state.active_tab {
        SlTab::Basics => submit_basics(app, fd),
        SlTab::Stress => submit_stress(app, fd),
    }
}

fn submit_basics(app: &mut App, fd: &mut viewport_lib::FrameData) {
    if app.sl_state.animate {
        app.sl_state.time += 0.016;
    }
    let t = app.sl_state.time;
    let radius = 5.5_f32;

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

    for i in 0..3 {
        let id = app.sl_state.light_ids[i];
        let src = app.sl_state.lights[i].clone();
        app.sl_state.scene.set_light(id, Some(src));
    }

    fd.scene.lights = app.sl_state.scene.collect_lights();

    if app.sl_state.show_glyphs {
        let (glyphs, polylines) = build_light_glyphs(&app.sl_state.scene, &Selection::new());
        fd.scene.glyphs.extend(glyphs);
        fd.scene.polylines.extend(polylines);
    }
}

fn submit_stress(app: &mut App, fd: &mut viewport_lib::FrameData) {
    if app.sl_state.stress_animate {
        app.sl_state.time += 0.012;
    }
    let t = app.sl_state.time;

    // Drift the lights in a circular pattern so the importance fallout is
    // visible at the rim. Skip index 0 (the directional fill).
    let bob_amp = 0.6;
    let bob_freq = 0.8;
    for (idx, src) in app.sl_state.stress_sources.iter_mut().enumerate().skip(1) {
        if let LightKind::Point { ref mut position, .. } = src.kind {
            let base_z = (position[2] - bob_amp).max(0.3);
            position[2] = base_z + bob_amp * (t * bob_freq + idx as f32 * 0.3).sin().abs();
            app.sl_state.scene.set_local_transform(
                app.sl_state.stress_light_ids[idx],
                glam::Mat4::from_translation(glam::Vec3::from(*position)),
            );
            app.sl_state.scene.set_light(
                app.sl_state.stress_light_ids[idx],
                Some(src.clone()),
            );
        }
    }

    fd.scene.lights = app.sl_state.scene.collect_lights();

    if app.sl_state.stress_show_glyphs {
        let (glyphs, polylines) = build_light_glyphs(&app.sl_state.scene, &Selection::new());
        fd.scene.glyphs.extend(glyphs);
        fd.scene.polylines.extend(polylines);
    }
}

// ---------------------------------------------------------------------------
// Scene item collection
// ---------------------------------------------------------------------------

pub(crate) fn sl_collect(
    app: &mut App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64) {
    let items = app.sl_state.scene.collect_render_items(&Selection::new());
    let mut l = LightingSettings::default();
    l.lights = vec![];
    l.shadows_enabled = false;
    match app.sl_state.active_tab {
        SlTab::Basics => {
            l.hemisphere_intensity = app.sl_state.hemi_intensity;
            l.sky_colour = [0.7, 0.8, 1.0];
            l.ground_colour = [0.4, 0.35, 0.3];
        }
        SlTab::Stress => {
            // Near-black ambient so the per-light pools dominate.
            l.hemisphere_intensity = 0.03;
            l.sky_colour = [0.1, 0.12, 0.18];
            l.ground_colour = [0.02, 0.02, 0.03];
        }
    }
    let sg = app.sl_state.scene.version();
    (items, l, sg)
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_sl(app: &mut App, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        if ui
            .selectable_label(app.sl_state.tab == SlTab::Basics, "Basics")
            .clicked()
        {
            app.sl_state.tab = SlTab::Basics;
        }
        if ui
            .selectable_label(app.sl_state.tab == SlTab::Stress, "Stress")
            .clicked()
        {
            app.sl_state.tab = SlTab::Stress;
        }
    });
    ui.separator();

    if app.sl_state.tab != app.sl_state.active_tab {
        app.sl_state.built = false;
    }

    match app.sl_state.tab {
        SlTab::Basics => controls_basics(app, ui),
        SlTab::Stress => controls_stress(app, ui),
    }
}

fn controls_basics(app: &mut App, ui: &mut egui::Ui) {
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

fn controls_stress(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Stress test: many point lights pushed into the scene at once. When the count exceeds the renderer's per-frame cap, lights are ranked by `LightSource::importance * proximity_weight` and the tail is dropped.");
    ui.separator();

    let mut dirty = false;
    ui.horizontal(|ui| {
        ui.label("Light count:");
        let resp = ui.add(egui::Slider::new(&mut app.sl_state.stress_count, 1..=1024));
        if resp.changed() {
            dirty = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Per-light range:");
        let resp = ui.add(egui::Slider::new(&mut app.sl_state.stress_radius, 1.0..=12.0));
        if resp.changed() {
            dirty = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Per-light intensity:");
        let resp = ui.add(egui::Slider::new(&mut app.sl_state.stress_intensity, 0.5..=15.0));
        if resp.changed() {
            dirty = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Importance falloff:");
        let resp = ui.add(
            egui::Slider::new(&mut app.sl_state.stress_importance_falloff, 0.0..=1.0)
                .text("0 = flat, 1 = strongly favour centre"),
        );
        if resp.changed() {
            dirty = true;
        }
    });
    ui.horizontal(|ui| {
        if ui.button("Reseed").clicked() {
            app.sl_state.stress_seed = app.sl_state.stress_seed.wrapping_mul(2_654_435_761).wrapping_add(1);
            dirty = true;
        }
        ui.checkbox(&mut app.sl_state.stress_animate, "Animate");
        ui.checkbox(&mut app.sl_state.stress_show_glyphs, "Show glyphs");
    });

    if dirty {
        rebuild_stress_lights(&mut app.sl_state);
    }
}
