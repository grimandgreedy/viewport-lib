//! Showcase 57: Light Falloff.
//!
//! A row of identical white spheres receding from a single punctual light, so
//! the physical inverse-square falloff is visible directly: each sphere down the
//! row is dimmer than the last. The controls expose the three knobs that shape a
//! punctual light:
//!
//!   - `intensity`: overall brightness (a plain linear multiplier for now; it
//!     becomes a photometric unit in a later phase).
//!   - `range`: reach. The falloff is windowed smoothly to zero here and the
//!     light is culled beyond it; it does NOT scale brightness. Drop it below
//!     the row length and the far spheres go dark.
//!   - `radius`: source size. It clamps the inverse-square term near the light,
//!     so raising it softens and caps the brightness of the closest spheres
//!     instead of letting them blow out.
//!
//! Point and spot lights share one falloff formula (see
//! `scene_lighting.wgsl::eval_light`); the directional option is a flat
//! reference with no distance term.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    LightKind, LightSource, LightingSettings, Material, ViewportRenderer, scene::Scene,
};

// Number of spheres in the receding row and their spacing (world units).
const ROW_COUNT: i32 = 10;
const ROW_SPACING: f32 = 2.0;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
pub(crate) enum LfType {
    Point,
    Spot,
    Directional,
}

pub(crate) struct LfState {
    pub built: bool,
    pub scene: Scene,
    pub light_type: LfType,
    pub intensity: f32,
    pub range: f32,
    pub radius: f32,
    pub height: f32,
    pub hemi_intensity: f32,
}

impl Default for LfState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            light_type: LfType::Point,
            intensity: 40.0,
            range: 30.0,
            radius: 0.1,
            height: 4.0,
            hemi_intensity: 0.05,
        }
    }
}

impl LfState {
    /// The single light described by the current controls.
    fn light(&self) -> LightSource {
        let mut s = LightSource::default();
        s.cast_shadows = false;
        s.kind = match self.light_type {
            // Point sits above the near end of the row so the falloff runs down +X.
            LfType::Point => LightKind::Point {
                position: [0.0, 0.0, self.height],
                range: self.range,
                radius: self.radius,
            },
            LfType::Spot => LightKind::Spot {
                position: [0.0, 0.0, self.height],
                direction: [1.0, 0.0, -0.3],
                range: self.range,
                inner_angle: 0.25,
                outer_angle: 0.55,
                radius: self.radius,
            },
            LfType::Directional => LightKind::Directional {
                direction: [-0.3, 0.0, 1.0],
            },
        };
        s.intensity = self.intensity;
        s
    }

    /// Full lighting settings for this frame: the one light plus a low
    /// hemisphere fill so the far, dark end of the row is not pure black.
    pub(crate) fn lighting(&self) -> LightingSettings {
        let mut l = LightingSettings::default();
        l.lights = vec![self.light()];
        l.shadows.enabled = false;
        l.hemisphere_intensity = self.hemi_intensity;
        l.sky_colour = [0.7, 0.8, 1.0];
        l.ground_colour = [0.3, 0.3, 0.35];
        l
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 57 (Light Falloff): a ground slab and a row
    /// of identical spheres receding in +X.
    pub(crate) fn build_light_falloff_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.lf_state.scene = Scene::new();

        let row_len = (ROW_COUNT - 1) as f32 * ROW_SPACING;

        // Ground slab spanning the row, Z-up.
        let ground_mesh = make_box_with_uvs(row_len + 6.0, 6.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("falloff ground mesh");
        self.lf_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(row_len * 0.5, 0.0, -0.05)),
            {
                let mut m = Material::from_colour([0.5, 0.5, 0.52]);
                m.roughness = 0.9;
                m
            },
        );

        // Row of identical neutral spheres so brightness differences read as
        // falloff, not as material variation.
        let sphere_mesh = viewport_lib::primitives::sphere(0.6, 32, 16);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("falloff sphere mesh");
        for i in 0..ROW_COUNT {
            let x = i as f32 * ROW_SPACING;
            self.lf_state.scene.add_named(
                &format!("Sphere {i}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, 0.6)),
                {
                    let mut m = Material::from_colour([0.9, 0.9, 0.9]);
                    m.roughness = 0.4;
                    m
                },
            );
        }

        self.lf_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_lf(app: &mut App, ui: &mut egui::Ui) {
    let st = &mut app.lf_state;
    ui.label("A row of identical spheres recedes in +X. Watch how each knob shapes the falloff.");
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Light:");
        ui.selectable_value(&mut st.light_type, LfType::Point, "Point");
        ui.selectable_value(&mut st.light_type, LfType::Spot, "Spot");
        ui.selectable_value(&mut st.light_type, LfType::Directional, "Directional");
    });

    let punctual = st.light_type != LfType::Directional;
    ui.add(egui::Slider::new(&mut st.intensity, 0.0..=200.0).text("Intensity"));
    ui.add_enabled(
        punctual,
        egui::Slider::new(&mut st.range, 1.0..=40.0).text("Range (reach, not brightness)"),
    );
    ui.add_enabled(
        punctual,
        egui::Slider::new(&mut st.radius, 0.0..=3.0).text("Source radius (near clamp)"),
    );
    ui.add_enabled(
        punctual,
        egui::Slider::new(&mut st.height, 0.5..=10.0).text("Light height"),
    );
    ui.add(egui::Slider::new(&mut st.hemi_intensity, 0.0..=0.5).text("Hemisphere fill"));

    ui.separator();
    ui.label(match st.light_type {
        LfType::Directional => {
            "Directional: no distance term, every sphere lit equally. Intensity ~1 is plenty."
        }
        _ => {
            "Point/Spot: brightness falls as 1/d^2. Lower Range to cut the far end; raise Radius to soften the near spheres."
        }
    });
}
