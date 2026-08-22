//! Showcase 61: Emissive & Environment (nits).
//!
//! The two light sources that carry **luminance** rather than illuminance:
//!
//! - **Emissive surfaces.** Each glowing sphere is authored with
//!   [`Material::emissive`] at a fixed luminance in nits (20 -> 20,000, a ladder
//!   of 10x steps). Under auto-exposure the dim ones read as a faint tint while
//!   the bright ones blow past the white point and bloom - exactly how a real
//!   emitter behaves relative to the exposed scene.
//! - **Image-based lighting.** A gradient environment lights the scene and is
//!   reflected by the chrome sphere. [`EnvironmentSettings::intensity`] is an absolute
//!   nits scale: slide it and the sky, its ambient contribution, and the
//!   reflection all brighten together, then the exposure re-balances.
//!
//! Both feed the same linear HDR radiance that exposure maps down, so they live
//! on one physical scale with the lux/candela lights of showcase 60.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    AutoExposure, EnvironmentSettings, ExposureMode, ExposureSettings, LightSource,
    LightingSettings, Lux, Material, ViewportRenderer, scene::Scene,
};

/// Fixed emissive ladder (hue, luminance in nits) - a 10x step each.
const EMITTERS: [([f32; 3], f32); 4] = [
    ([1.0, 0.2, 0.2], 20.0),     // deep red, barely glowing
    ([1.0, 0.7, 0.2], 200.0),    // amber, a soft lamp
    ([0.3, 0.9, 1.0], 2_000.0),  // cyan, clearly lit
    ([1.0, 1.0, 1.0], 20_000.0), // white, blows out and blooms
];
const COL_SPACING: f32 = 2.4;

/// Equirectangular sky/ground gradient (RGBA f32), sky at the top row. The stored
/// values are relative; [`EnvironmentSettings::intensity`] scales them to nits.
fn equirect_gradient(sky: [f32; 3], ground: [f32; 3], w: u32, h: u32) -> Vec<f32> {
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let v = y as f32 / (h - 1).max(1) as f32;
        let t = v * v * (3.0 - 2.0 * v); // soft horizon
        for _x in 0..w {
            px.push(sky[0] + (ground[0] - sky[0]) * t);
            px.push(sky[1] + (ground[1] - sky[1]) * t);
            px.push(sky[2] + (ground[2] - sky[2]) * t);
            px.push(1.0);
        }
    }
    px
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct EmissiveEnvState {
    pub built: bool,
    pub scene: Scene,
    /// Absolute environment luminance in nits (drives IBL + skybox).
    pub env_intensity: f32,
    pub show_skybox: bool,
    /// Frame time captured from egui, fed to auto-exposure smoothing.
    pub frame_dt: f32,
}

impl Default for EmissiveEnvState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            env_intensity: 4_000.0,
            show_skybox: true,
            frame_dt: 0.0,
        }
    }
}

impl EmissiveEnvState {
    pub(crate) fn lighting(&self) -> LightingSettings {
        // A soft overcast key gives the matte surfaces shape; the environment
        // carries the ambient and the reflections, so no hemisphere fill is
        // needed here.
        let mut sun = LightSource::directional_lux([0.3, 0.25, 1.2], Lux::OVERCAST);
        sun.cast_shadows = true;

        let mut l = LightingSettings::default();
        l.lights = vec![sun];
        l.shadows.enabled = true;
        l.hemisphere_intensity = 0.0;
        l
    }

    pub(crate) fn environment(&self) -> EnvironmentSettings {
        EnvironmentSettings {
            intensity: self.env_intensity,
            rotation: 0.0,
            show_skybox: self.show_skybox,
        }
    }

    pub(crate) fn exposure(&self) -> ExposureSettings {
        // Full adaptation so the scene stays readable as the environment nits and
        // the emitters push the metered luminance around.
        let auto = AutoExposure {
            adaptation: 1.0,
            dt: self.frame_dt,
            ..AutoExposure::default()
        };
        ExposureSettings::from_mode(ExposureMode::Automatic(auto))
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 61 (Emissive & Environment): a ground plane,
    /// a ladder of emissive spheres, and a chrome sphere that mirrors the
    /// environment. Uploads a gradient environment for the IBL.
    pub(crate) fn build_emissive_env_scene(&mut self, renderer: &mut ViewportRenderer) {
        // Gradient environment: cool sky -> warm ground. Uploaded once; it bakes
        // the irradiance and prefiltered reflection maps the IBL samples.
        let px = equirect_gradient([0.35, 0.5, 0.85], [0.35, 0.28, 0.22], 64, 32);
        renderer
            .upload_environment_map(&self.device, &self.queue, &px, 64, 32)
            .expect("environment upload");

        self.emissive_env_state.scene = Scene::new();

        let count = EMITTERS.len();
        let span = count as f32 * COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 10.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("emissive ground mesh");
        self.emissive_env_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(span * 0.5 - COL_SPACING, 0.0, -0.05)),
            {
                let mut m = Material::from_colour([0.5, 0.5, 0.52]);
                m.roughness = 0.9;
                m
            },
        );

        let sphere_mesh = viewport_lib::primitives::sphere(0.8, 48, 24);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("emissive sphere mesh");

        // The emissive ladder.
        for (i, (hue, nits)) in EMITTERS.iter().enumerate() {
            self.emissive_env_state.scene.add_named(
                &format!("Emissive {nits:.0} nits"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(i as f32 * COL_SPACING, 0.8, 0.85)),
                Material::emissive(*hue, *nits),
            );
        }

        // A chrome sphere to show the reflected environment.
        self.emissive_env_state.scene.add_named(
            "Chrome (reflects environment)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(count as f32 * COL_SPACING, 0.8, 0.85)),
            Material::pbr([0.95, 0.95, 0.95], 1.0, 0.06),
        );

        self.emissive_env_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_emissive_env(app: &mut App, ui: &mut egui::Ui) {
    app.emissive_env_state.frame_dt = ui.ctx().input(|i| i.stable_dt).min(0.1);
    let st = &mut app.emissive_env_state;

    ui.label(
        "Emissive surfaces and image-based lighting, both authored in nits. The spheres glow at a \
         fixed 20 -> 20,000 nit ladder; the environment's nits scale is live below. Auto-exposure \
         (full adaptation) re-balances as either changes.",
    );
    ui.separator();

    ui.label("Emissive ladder (fixed, nits):");
    for (_, nits) in EMITTERS.iter() {
        ui.label(format!("  - {nits:.0} nits"));
    }
    ui.label("Dim emitters read as a faint tint; bright ones exceed the white point and bloom.");
    ui.separator();

    ui.label("Environment (image-based lighting):");
    ui.add(
        egui::Slider::new(&mut st.env_intensity, 0.0..=20_000.0)
            .text("Environment intensity (nits)"),
    );
    ui.checkbox(&mut st.show_skybox, "Show environment as skybox");
    ui.label("Intensity is an absolute nits scale: the sky, its ambient light, and the chrome sphere's reflection all track it, then exposure re-balances.");
    ui.separator();

    ui.label("Exposure: automatic, full adaptation.");
}
