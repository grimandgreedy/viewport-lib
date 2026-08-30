//! Showcase 58: Physically-Based Surfaces.
//!
//! **Emissive & IBL** - the two sources that carry **luminance** rather than
//! illuminance: emissive surfaces at a fixed nits ladder, and an image-based
//! environment whose [`EnvironmentSettings::intensity`] is an absolute nits
//! scale. Auto-exposure re-balances as either changes.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib as vpl;
use vpl::{
    AutoExposure, EnvironmentSettings, ExposureMode, ExposureSettings, LightSource,
    LightingSettings, Lux, Material, ViewportRenderer, scene::Scene,
};

// ===========================================================================
// Emissive & IBL
// ===========================================================================

/// Fixed emissive ladder (hue, luminance in nits) - a 10x step each.
const EMITTERS: [([f32; 3], f32); 4] = [
    ([1.0, 0.2, 0.2], 20.0),     // deep red, barely glowing
    ([1.0, 0.7, 0.2], 200.0),    // amber, a soft lamp
    ([0.3, 0.9, 1.0], 2_000.0),  // cyan, clearly lit
    ([1.0, 1.0, 1.0], 20_000.0), // white, blows out and blooms
];
const EMISSIVE_COL_SPACING: f32 = 2.4;

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

pub(crate) struct PhysicallyBasedSurfacesState {
    pub built: bool,
    pub scene: Scene,
    /// Absolute environment luminance in nits (drives IBL + skybox).
    pub env_intensity: f32,
    pub show_skybox: bool,
    /// Frame time captured from egui, fed to auto-exposure smoothing.
    pub frame_dt: f32,
}

impl Default for PhysicallyBasedSurfacesState {
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

impl PhysicallyBasedSurfacesState {
    pub(crate) fn built(&self) -> bool {
        self.built
    }

    pub(crate) fn scene(&self) -> &Scene {
        &self.scene
    }

    pub(crate) fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    pub(crate) fn lighting(&self) -> LightingSettings {
        // A soft overcast key gives the matte surfaces shape; the environment
        // carries the ambient and the reflections, so no hemisphere fill here.
        let mut sun = LightSource::directional_lux([0.3, 0.25, 1.2], Lux::OVERCAST);
        sun.cast_shadows = true;

        let mut l = LightingSettings::default();
        l.lights = vec![sun];
        l.shadows.enabled = true;
        l.hemisphere_intensity = 0.0;
        l
    }

    pub(crate) fn environment(&self) -> Option<EnvironmentSettings> {
        Some(EnvironmentSettings {
            intensity: self.env_intensity,
            rotation: 0.0,
            show_skybox: self.show_skybox,
        })
    }

    pub(crate) fn exposure_override(&self) -> Option<ExposureSettings> {
        let auto = AutoExposure {
            adaptation: 1.0,
            dt: self.frame_dt,
            ..AutoExposure::default()
        };
        Some(ExposureSettings::from_mode(ExposureMode::Automatic(auto)))
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 58 and frame the camera for it.
    pub(crate) fn build_physically_based_surfaces_scene(
        &mut self,
        renderer: &mut ViewportRenderer,
    ) {
        self.build_emissive_scene(renderer);
        self.camera = vpl::Camera {
            center: glam::Vec3::new(4.8, 0.5, 0.8),
            distance: 17.0,
            orientation: glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.2),
            ..vpl::Camera::default()
        };
    }

    fn build_emissive_scene(&mut self, renderer: &mut ViewportRenderer) {
        // Gradient environment: cool sky -> warm ground. Uploaded once; it bakes
        // the irradiance and prefiltered reflection maps the IBL samples.
        let px = equirect_gradient([0.35, 0.5, 0.85], [0.35, 0.28, 0.22], 64, 32);
        renderer
            .upload_environment_map(&self.device, &self.queue, &px, 64, 32)
            .expect("environment upload");

        self.surfaces_state.scene = Scene::new();

        let count = EMITTERS.len();
        let span = count as f32 * EMISSIVE_COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 10.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("emissive ground mesh");
        self.surfaces_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(span * 0.5, 0.0, -0.05)),
            {
                let mut m = Material::from_colour([0.5, 0.5, 0.52]);
                m.roughness = 0.9;
                m
            },
        );

        let sphere_mesh = vpl::primitives::sphere(0.8, 48, 24);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("emissive sphere mesh");

        // The emissive ladder.
        for (i, (hue, nits)) in EMITTERS.iter().enumerate() {
            self.surfaces_state.scene.add_named(
                &format!("Emissive {nits:.0} nits"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(
                    i as f32 * EMISSIVE_COL_SPACING,
                    0.8,
                    0.85,
                )),
                Material::emissive(*hue, *nits),
            );
        }

        // A chrome sphere to show the reflected environment.
        self.surfaces_state.scene.add_named(
            "Chrome (reflects environment)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(
                count as f32 * EMISSIVE_COL_SPACING,
                0.8,
                0.85,
            )),
            Material::pbr([0.95, 0.95, 0.95], 1.0, 0.06),
        );

        self.surfaces_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_physically_based_surfaces(app: &mut App, ui: &mut egui::Ui) {
    app.surfaces_state.frame_dt = ui.ctx().input(|i| i.stable_dt).min(0.1);
    let st = &mut app.surfaces_state;

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
