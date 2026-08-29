//! Showcase 58: Physically-Based Surfaces.
//!
//! How surfaces turn light into pixels, in two sub-tabs:
//!
//! - **Shading Parity** - two rows of identical spheres under one directional
//!   light: PBR (back) and Blinn-Phong (front), sharing per-column roughness. The
//!   diffuse energy is normalised the same way in both (`albedo / pi`), so the
//!   rows read at the same brightness; only the highlight shape differs.
//! - **Emissive & IBL** - the two sources that carry **luminance** rather than
//!   illuminance: emissive surfaces at a fixed nits ladder, and an image-based
//!   environment whose [`EnvironmentSettings::intensity`] is an absolute nits
//!   scale. Auto-exposure re-balances as either changes.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    AutoExposure, EnvironmentSettings, ExposureMode, ExposureSettings, LightKind, LightSource,
    LightingSettings, Lux, Material, ShadingModel, ViewportRenderer, scene::Scene,
};

/// Which facet of surface response is shown.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SurfacesSub {
    Parity,
    EmissiveIbl,
}

// ===========================================================================
// Shading Parity sub
// ===========================================================================

const PARITY_COLUMNS: usize = 5;
const PARITY_COL_SPACING: f32 = 2.6;
const PARITY_ROW_Y: f32 = 1.6;

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
        l.shadows.enabled = false;
        l.hemisphere_intensity = self.hemi_intensity;
        l.sky_colour = [0.7, 0.8, 1.0];
        l.ground_colour = [0.3, 0.3, 0.35];
        l
    }
}

/// Roughness for column `c`, from smooth on the left to rough on the right.
fn column_roughness(c: usize) -> f32 {
    let t = c as f32 / (PARITY_COLUMNS - 1) as f32;
    0.05 + t * 0.9
}

// ===========================================================================
// Emissive & IBL sub
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
        // carries the ambient and the reflections, so no hemisphere fill here.
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
        let auto = AutoExposure {
            adaptation: 1.0,
            dt: self.frame_dt,
            ..AutoExposure::default()
        };
        ExposureSettings::from_mode(ExposureMode::Automatic(auto))
    }
}

// ===========================================================================
// Container
// ===========================================================================

pub(crate) struct PhysicallyBasedSurfacesState {
    pub sub: SurfacesSub,
    pub parity: ParityState,
    pub emissive: EmissiveEnvState,
}

impl Default for PhysicallyBasedSurfacesState {
    fn default() -> Self {
        Self {
            sub: SurfacesSub::Parity,
            parity: ParityState::default(),
            emissive: EmissiveEnvState::default(),
        }
    }
}

impl PhysicallyBasedSurfacesState {
    pub(crate) fn built(&self) -> bool {
        match self.sub {
            SurfacesSub::Parity => self.parity.built,
            SurfacesSub::EmissiveIbl => self.emissive.built,
        }
    }

    fn mark_unbuilt(&mut self) {
        match self.sub {
            SurfacesSub::Parity => self.parity.built = false,
            SurfacesSub::EmissiveIbl => self.emissive.built = false,
        }
    }

    pub(crate) fn scene(&self) -> &Scene {
        match self.sub {
            SurfacesSub::Parity => &self.parity.scene,
            SurfacesSub::EmissiveIbl => &self.emissive.scene,
        }
    }

    pub(crate) fn scene_mut(&mut self) -> &mut Scene {
        match self.sub {
            SurfacesSub::Parity => &mut self.parity.scene,
            SurfacesSub::EmissiveIbl => &mut self.emissive.scene,
        }
    }

    pub(crate) fn lighting(&self) -> LightingSettings {
        match self.sub {
            SurfacesSub::Parity => self.parity.lighting(),
            SurfacesSub::EmissiveIbl => self.emissive.lighting(),
        }
    }

    /// Exposure override for the active sub, or `None` to leave the frame default
    /// (the Parity sub reads at the neutral default exposure).
    pub(crate) fn exposure_override(&self) -> Option<ExposureSettings> {
        match self.sub {
            SurfacesSub::EmissiveIbl => Some(self.emissive.exposure()),
            SurfacesSub::Parity => None,
        }
    }

    /// Environment for the active sub, or `None` (the Parity sub has no IBL).
    pub(crate) fn environment(&self) -> Option<EnvironmentSettings> {
        match self.sub {
            SurfacesSub::EmissiveIbl => Some(self.emissive.environment()),
            SurfacesSub::Parity => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the active sub-scene for Showcase 58 and frame the camera for it.
    pub(crate) fn build_physically_based_surfaces_scene(
        &mut self,
        renderer: &mut ViewportRenderer,
    ) {
        match self.surfaces_state.sub {
            SurfacesSub::Parity => {
                self.build_parity_scene(renderer);
                self.camera = viewport_lib::Camera {
                    center: glam::Vec3::new(5.2, 0.0, 0.6),
                    distance: 16.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.15),
                    ..viewport_lib::Camera::default()
                };
            }
            SurfacesSub::EmissiveIbl => {
                self.build_emissive_scene(renderer);
                self.camera = viewport_lib::Camera {
                    center: glam::Vec3::new(4.8, 0.5, 0.8),
                    distance: 17.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.2),
                    ..viewport_lib::Camera::default()
                };
            }
        }
    }

    fn build_parity_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.surfaces_state.parity.scene = Scene::new();
        let metallic = self.surfaces_state.parity.metallic;

        let span = (PARITY_COLUMNS - 1) as f32 * PARITY_COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 8.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("parity ground mesh");
        self.surfaces_state.parity.scene.add_named(
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
        for c in 0..PARITY_COLUMNS {
            let x = c as f32 * PARITY_COL_SPACING;
            let rough = column_roughness(c);

            // PBR row (back).
            let pbr = {
                let mut m = Material::from_colour(base);
                m.shading_model = ShadingModel::Pbr;
                m.roughness = rough;
                m.metallic = metallic;
                m
            };
            self.surfaces_state.parity.scene.add_named(
                &format!("PBR {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, PARITY_ROW_Y, 0.7)),
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
            self.surfaces_state.parity.scene.add_named(
                &format!("Phong {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, -PARITY_ROW_Y, 0.7)),
                phong,
            );
        }

        self.surfaces_state.parity.built = true;
    }

    fn build_emissive_scene(&mut self, renderer: &mut ViewportRenderer) {
        // Gradient environment: cool sky -> warm ground. Uploaded once; it bakes
        // the irradiance and prefiltered reflection maps the IBL samples.
        let px = equirect_gradient([0.35, 0.5, 0.85], [0.35, 0.28, 0.22], 64, 32);
        renderer
            .upload_environment_map(&self.device, &self.queue, &px, 64, 32)
            .expect("environment upload");

        self.surfaces_state.emissive.scene = Scene::new();

        let count = EMITTERS.len();
        let span = count as f32 * EMISSIVE_COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 10.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("emissive ground mesh");
        self.surfaces_state.emissive.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(
                span * 0.5 - EMISSIVE_COL_SPACING,
                0.0,
                -0.05,
            )),
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
            self.surfaces_state.emissive.scene.add_named(
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
        self.surfaces_state.emissive.scene.add_named(
            "Chrome (reflects environment)",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(
                count as f32 * EMISSIVE_COL_SPACING,
                0.8,
                0.85,
            )),
            Material::pbr([0.95, 0.95, 0.95], 1.0, 0.06),
        );

        self.surfaces_state.emissive.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_physically_based_surfaces(app: &mut App, ui: &mut egui::Ui) {
    let prev = app.surfaces_state.sub;
    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut app.surfaces_state.sub,
            SurfacesSub::Parity,
            "Shading Parity",
        );
        ui.selectable_value(
            &mut app.surfaces_state.sub,
            SurfacesSub::EmissiveIbl,
            "Emissive & IBL",
        );
    });
    if app.surfaces_state.sub != prev {
        app.surfaces_state.mark_unbuilt();
    }
    ui.separator();

    match app.surfaces_state.sub {
        SurfacesSub::Parity => controls_parity(app, ui),
        SurfacesSub::EmissiveIbl => controls_emissive(app, ui),
    }
}

fn controls_parity(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Back row: PBR. Front row: Phong. Same roughness per column, one directional light.");
    ui.label("The two rows should read at the same brightness; only the highlight shape differs.");
    ui.separator();

    let mut rebuild = false;
    ui.add(
        egui::Slider::new(&mut app.surfaces_state.parity.intensity, 0.0..=10.0)
            .text("Light intensity"),
    );
    ui.add(
        egui::Slider::new(&mut app.surfaces_state.parity.hemi_intensity, 0.0..=0.5)
            .text("Hemisphere fill"),
    );
    if ui
        .add(
            egui::Slider::new(&mut app.surfaces_state.parity.metallic, 0.0..=1.0)
                .text("PBR metallic"),
        )
        .changed()
    {
        // Metallic is baked into the PBR material at build time, so rebuild.
        rebuild = true;
    }
    if rebuild {
        app.surfaces_state.parity.built = false;
    }
    ui.separator();
    ui.label("Metallic drives the PBR row only; Phong has no metallic term (its highlight is a fixed specular colour).");
}

fn controls_emissive(app: &mut App, ui: &mut egui::Ui) {
    app.surfaces_state.emissive.frame_dt = ui.ctx().input(|i| i.stable_dt).min(0.1);
    let st = &mut app.surfaces_state.emissive;

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
