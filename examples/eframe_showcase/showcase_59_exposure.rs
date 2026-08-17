//! Showcase 59: Exposure & Auto-Exposure.
//!
//! One lit scene (a ground plane and a row of spheres running dark -> bright
//! albedo, with cast shadows for a genuinely dark region) viewed under the three
//! exposure modes:
//!
//! - **Manual EV**: a fixed EV100. Crank the scene brightness and the image
//!   clips or crushes, exactly like a camera on manual.
//! - **Physical camera**: aperture / shutter / ISO drive EV100 through the
//!   camera triangle; the readout shows the resulting EV.
//! - **Automatic**: the HDR target is metered each frame and the exposure adapts
//!   to hold the image at a sensible mid-grey no matter how bright the scene
//!   gets. With smoothing off (`dt = 0`) it snaps; with smoothing on it eases
//!   ("the eye adjusting").
//!
//! The scene-brightness slider is the thing to play with: under Manual it
//! changes how bright the image is; under Automatic the image stays put while
//! the metered EV moves.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    AutoExposure, ExposureMode, ExposureSettings, LightKind, LightSource, LightingSettings,
    Material, ViewportRenderer, scene::Scene,
};

const COLUMNS: usize = 6;
const COL_SPACING: f32 = 2.4;

/// Which exposure mode the radio selects. Kept separate from
/// [`ExposureSettings`] so per-mode parameters survive switching between modes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModeSel {
    Manual,
    Physical,
    Automatic,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ExposureShowcaseState {
    pub built: bool,
    pub scene: Scene,
    /// Scene brightness (directional light intensity). The demo knob.
    pub light_intensity: f32,

    pub mode: ModeSel,
    pub compensation: f32,

    // Manual
    pub manual_ev: f32,
    // Physical camera
    pub aperture: f32,
    pub shutter_denom: f32,
    pub iso: f32,
    // Automatic
    pub auto: AutoExposure,
    pub smooth: bool,
    /// Frame time captured from egui, fed to the auto-exposure `dt` when
    /// smoothing is on (this host is not `ViewportApp`, so it fills `dt` itself).
    pub frame_dt: f32,
}

impl Default for ExposureShowcaseState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            light_intensity: 4.0,
            mode: ModeSel::Automatic,
            compensation: 0.0,
            manual_ev: 0.0,
            // Low-light defaults (wide aperture, slow shutter, high ISO): the
            // scene is still pre-photometric (intensities ~1, not lux/candela),
            // so daylight f-stops read black until units are pinned in Phase 4.
            aperture: 1.4,
            shutter_denom: 30.0,
            iso: 3200.0,
            auto: AutoExposure::default(),
            smooth: false,
            frame_dt: 0.0,
        }
    }
}

impl ExposureShowcaseState {
    pub(crate) fn lighting(&self) -> LightingSettings {
        let mut s = LightSource::default();
        s.cast_shadows = true;
        s.kind = LightKind::Directional {
            direction: [0.4, 0.5, 0.9],
        };
        s.intensity = self.light_intensity;

        let mut l = LightingSettings::default();
        l.lights = vec![s];
        l.shadows_enabled = true;
        // A little sky fill so shadows are dark but not pure black.
        l.hemisphere_intensity = 0.05;
        l.sky_colour = [0.6, 0.7, 0.9];
        l.ground_colour = [0.25, 0.22, 0.2];
        l
    }

    /// The exposure settings this frame, assembled from the current mode.
    pub(crate) fn settings(&self) -> ExposureSettings {
        let mode = match self.mode {
            ModeSel::Manual => ExposureMode::Manual { ev: self.manual_ev },
            ModeSel::Physical => ExposureMode::PhysicalCamera {
                aperture: self.aperture,
                shutter: 1.0 / self.shutter_denom.max(1.0),
                iso: self.iso.max(1.0),
            },
            ModeSel::Automatic => {
                let mut a = self.auto;
                a.dt = if self.smooth { self.frame_dt } else { 0.0 };
                ExposureMode::Automatic(a)
            }
        };
        ExposureSettings::from_mode(mode).with_compensation(self.compensation)
    }

    /// The EV100 the non-automatic modes resolve to, for the readout.
    fn readout_ev(&self) -> Option<f32> {
        match self.mode {
            ModeSel::Manual => Some(self.manual_ev),
            ModeSel::Physical => {
                let n = self.aperture;
                let t = 1.0 / self.shutter_denom.max(1.0);
                let iso = self.iso.max(1.0);
                Some((n * n / t).log2() + (100.0 / iso).log2())
            }
            ModeSel::Automatic => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 59 (Exposure): a ground plane and a row of
    /// spheres with albedo running dark -> bright, lit by one shadowing light.
    pub(crate) fn build_exposure_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.exposure_state.scene = Scene::new();

        let span = (COLUMNS - 1) as f32 * COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 10.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("exposure ground mesh");
        self.exposure_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(span * 0.5, 0.0, -0.05)),
            {
                let mut m = Material::from_colour([0.45, 0.45, 0.48]);
                m.roughness = 0.95;
                m
            },
        );

        let sphere_mesh = viewport_lib::primitives::sphere(0.8, 48, 24);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("exposure sphere mesh");

        for c in 0..COLUMNS {
            let x = c as f32 * COL_SPACING;
            // Albedo dark on the left, bright on the right.
            let a = 0.05 + (c as f32 / (COLUMNS - 1) as f32) * 0.85;
            let mut m = Material::from_colour([a, a, a]);
            m.roughness = 0.6;
            m.metallic = 0.0;
            self.exposure_state.scene.add_named(
                &format!("Sphere {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.8, 0.85)),
                m,
            );
        }

        self.exposure_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_exposure(app: &mut App, ui: &mut egui::Ui) {
    // Capture this host's frame time for smooth auto-exposure adaptation.
    app.exposure_state.frame_dt = ui.ctx().input(|i| i.stable_dt).min(0.1);

    ui.label("One lit scene under three exposure modes. Drive the scene-brightness slider and watch how each mode responds.");
    ui.separator();

    ui.add(
        egui::Slider::new(&mut app.exposure_state.light_intensity, 0.1..=40.0)
            .logarithmic(true)
            .text("Scene brightness (light intensity)"),
    );
    ui.add(
        egui::Slider::new(&mut app.exposure_state.compensation, -3.0..=3.0)
            .text("Exposure compensation (stops)"),
    );
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Mode:");
        ui.selectable_value(&mut app.exposure_state.mode, ModeSel::Manual, "Manual EV");
        ui.selectable_value(
            &mut app.exposure_state.mode,
            ModeSel::Physical,
            "Physical camera",
        );
        ui.selectable_value(
            &mut app.exposure_state.mode,
            ModeSel::Automatic,
            "Automatic",
        );
    });
    ui.separator();

    match app.exposure_state.mode {
        ModeSel::Manual => {
            ui.add(egui::Slider::new(&mut app.exposure_state.manual_ev, -6.0..=16.0).text("EV100"));
            ui.label("Fixed exposure. Higher EV = darker image. Change the scene brightness and the image clips or crushes.");
        }
        ModeSel::Physical => {
            ui.add(
                egui::Slider::new(&mut app.exposure_state.aperture, 1.0..=22.0).text("Aperture f/"),
            );
            ui.add(
                egui::Slider::new(&mut app.exposure_state.shutter_denom, 4.0..=4000.0)
                    .logarithmic(true)
                    .text("Shutter 1/x s"),
            );
            ui.add(
                egui::Slider::new(&mut app.exposure_state.iso, 50.0..=6400.0)
                    .logarithmic(true)
                    .text("ISO"),
            );
            ui.label("The camera triangle: EV100 = log2(N^2 / t) + log2(100 / ISO).");
            ui.label("Note: f-stops are calibrated for photometric magnitudes (Phase 4). Until units are pinned, the scene is dim, so use fast/wide/high-ISO (low EV) settings to expose it; daylight settings read black.");
        }
        ModeSel::Automatic => {
            ui.checkbox(&mut app.exposure_state.smooth, "Smooth adaptation (dt > 0)");
            ui.add(
                egui::Slider::new(&mut app.exposure_state.auto.min_ev, -10.0..=6.0).text("EV min"),
            );
            ui.add(
                egui::Slider::new(&mut app.exposure_state.auto.max_ev, 4.0..=20.0).text("EV max"),
            );
            ui.add(
                egui::Slider::new(&mut app.exposure_state.auto.speed_up, 0.1..=10.0)
                    .text("Adapt speed (brighten)"),
            );
            ui.add(
                egui::Slider::new(&mut app.exposure_state.auto.speed_down, 0.1..=10.0)
                    .text("Adapt speed (darken)"),
            );
            ui.add(
                egui::Slider::new(&mut app.exposure_state.auto.low_percent, 0.0..=0.9)
                    .text("Meter low clip"),
            );
            ui.add(
                egui::Slider::new(&mut app.exposure_state.auto.high_percent, 0.1..=1.0)
                    .text("Meter high clip"),
            );
            ui.add(
                egui::Slider::new(&mut app.exposure_state.auto.center_weight, 0.0..=1.0)
                    .text("Center weighting"),
            );
            ui.label("Metering the HDR target holds the image steady while the scene brightness moves. dt <= 0 snaps; smoothing eases over time. Center weighting keeps exposure stable when you zoom or pan (set it to 0 to meter the whole frame and see the framing-driven swing).");
        }
    }

    ui.separator();
    match app.exposure_state.readout_ev() {
        Some(ev) => {
            ui.label(format!(
                "EV100: {ev:.2}  (before {:+.1} stops compensation)",
                app.exposure_state.compensation
            ));
        }
        None => {
            ui.label("EV100: metered on the GPU each frame (adapts live).");
        }
    }
}
