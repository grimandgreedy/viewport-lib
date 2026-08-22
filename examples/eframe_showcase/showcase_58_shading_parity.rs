//! Showcase 58: Shading Model Parity.
//!
//! Two rows of identical spheres under one directional light: the back row is
//! PBR, the front row is Blinn-Phong, and each column shares a roughness. The
//! point is that the two rows read at the same overall brightness: the diffuse
//! energy is now normalised the same way in both models (`albedo / pi`), so
//! switching a material between `Pbr` and `Phong` no longer changes how light or
//! dark it is. What still differs, by design, is the highlight: Phong keeps its
//! cheaper Blinn specular lobe (now energy-bounded), so its highlight has a
//! different shape from PBR's Cook-Torrance one even though the diffuse matches.
//!
//! For a fair comparison the Phong materials use `diffuse = 1.0` so their
//! diffuse albedo weight matches PBR's (the default Phong `diffuse` is lower,
//! which would just make the row uniformly dimmer, not a model difference).

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    LightKind, LightSource, LightingSettings, Material, ShadingModel, ViewportRenderer,
    scene::Scene,
};

const COLUMNS: usize = 5;
const COL_SPACING: f32 = 2.6;
const ROW_Y: f32 = 1.6;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ParityState {
    pub built: bool,
    pub scene: Scene,
    pub intensity: f32,
    pub metallic: f32,
    pub hemi_intensity: f32,
}

impl Default for ParityState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            // Directional reflected radiance is albedo/pi * intensity, so ~pi
            // lands a white surface near display white before any exposure.
            intensity: 3.0,
            metallic: 0.0,
            hemi_intensity: 0.05,
        }
    }
}

impl ParityState {
    pub(crate) fn lighting(&self) -> LightingSettings {
        let mut s = LightSource::default();
        s.cast_shadows = false;
        s.kind = LightKind::Directional {
            direction: [0.35, 0.2, 1.0],
        };
        s.intensity = self.intensity;

        let mut l = LightingSettings::default();
        l.lights = vec![s];
        l.shadows_enabled = false;
        l.hemisphere_intensity = self.hemi_intensity;
        l.sky_colour = [0.7, 0.8, 1.0];
        l.ground_colour = [0.3, 0.3, 0.35];
        l
    }
}

/// Roughness for column `c`, from smooth on the left to rough on the right.
fn column_roughness(c: usize) -> f32 {
    let t = c as f32 / (COLUMNS - 1) as f32;
    0.05 + t * 0.9
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 58 (Shading Model Parity): a PBR row and a
    /// Phong row of spheres sharing per-column roughness.
    pub(crate) fn build_shading_parity_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.parity_state.scene = Scene::new();
        let metallic = self.parity_state.metallic;

        let span = (COLUMNS - 1) as f32 * COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 8.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("parity ground mesh");
        self.parity_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(span * 0.5, 0.0, -0.05)),
            {
                let mut m = Material::from_colour([0.5, 0.5, 0.52]);
                m.roughness = 0.9;
                m
            },
        );

        let sphere_mesh = viewport_lib::primitives::sphere(0.7, 48, 24);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("parity sphere mesh");

        let base = [0.85, 0.85, 0.88];
        for c in 0..COLUMNS {
            let x = c as f32 * COL_SPACING;
            let rough = column_roughness(c);

            // PBR row (back).
            let pbr = {
                let mut m = Material::from_colour(base);
                m.shading_model = ShadingModel::Pbr;
                m.roughness = rough;
                m.metallic = metallic;
                m
            };
            self.parity_state.scene.add_named(
                &format!("PBR {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, ROW_Y, 0.7)),
                pbr,
            );

            // Phong row (front). shininess runs sharp -> broad with roughness;
            // diffuse = 1.0 so the diffuse albedo weight matches PBR.
            let phong = {
                let mut m = Material::from_colour(base);
                m.shading_model = ShadingModel::Phong;
                m.diffuse = 1.0;
                m.specular = 0.5;
                m.shininess = 2.0 + (1.0 - rough) * 126.0;
                m
            };
            self.parity_state.scene.add_named(
                &format!("Phong {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, -ROW_Y, 0.7)),
                phong,
            );
        }

        self.parity_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_parity(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Back row: PBR. Front row: Phong. Same roughness per column, one directional light.");
    ui.label("The two rows should read at the same brightness; only the highlight shape differs.");
    ui.separator();

    let mut rebuild = false;
    ui.add(egui::Slider::new(&mut app.parity_state.intensity, 0.0..=10.0).text("Light intensity"));
    ui.add(
        egui::Slider::new(&mut app.parity_state.hemi_intensity, 0.0..=0.5).text("Hemisphere fill"),
    );
    if ui
        .add(egui::Slider::new(&mut app.parity_state.metallic, 0.0..=1.0).text("PBR metallic"))
        .changed()
    {
        // Metallic changes the PBR material, which is baked into the scene at
        // build time, so rebuild when it moves.
        rebuild = true;
    }
    if rebuild {
        app.parity_state.built = false;
    }
    ui.separator();
    ui.label("Metallic drives the PBR row only; Phong has no metallic term (its highlight is a fixed specular colour).");
}
