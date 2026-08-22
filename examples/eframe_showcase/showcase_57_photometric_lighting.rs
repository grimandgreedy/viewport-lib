//! Showcase 57: Photometric Lighting.
//!
//! The light-authoring and camera side of the photometric pipeline, in one scene
//! with three sub-tabs:
//!
//! - **Units & Presets** - lights authored in real units (directional in **lux**,
//!   bulbs in **lumens** -> candela). Pick a sky preset (full daylight down to
//!   full moon, a ~400,000x range) and an optional indoor bulb; auto-exposure
//!   adapts every preset to mid-grey, or a fixed daylight camera exposes daylight
//!   and leaves dimmer presets dark.
//! - **Falloff** - a row of identical spheres receding from a punctual light, so
//!   the physical inverse-square falloff reads directly. `range` is reach (not
//!   brightness); `radius` clamps the near-field.
//! - **Exposure** - one lit scene under Manual EV / Physical camera / Automatic,
//!   with the resulting EV100 read out.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    AutoExposure, ExposureMode, ExposureSettings, LightKind, LightSource, LightingSettings, Lumen,
    Lux, Material, ViewportRenderer, scene::Scene,
};

/// Which facet of photometric lighting is shown.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LightingSub {
    Presets,
    Falloff,
    Exposure,
}

// ===========================================================================
// Units & Presets sub
// ===========================================================================

const PRESET_COLUMNS: usize = 5;
const PRESET_COL_SPACING: f32 = 2.4;

/// A directional-light preset: a real-world illuminance and a matching tint and
/// elevation. Authored with [`LightSource::directional_lux`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkyPreset {
    FullDaylight,
    Overcast,
    SunriseSunset,
    FullMoon,
}

impl SkyPreset {
    fn label(self) -> &'static str {
        match self {
            Self::FullDaylight => "Full daylight (~100,000 lx)",
            Self::Overcast => "Overcast sky (~1,000 lx)",
            Self::SunriseSunset => "Sunrise / sunset (~400 lx)",
            Self::FullMoon => "Full moon (~0.25 lx)",
        }
    }

    fn illuminance(self) -> Lux {
        match self {
            Self::FullDaylight => Lux::FULL_DAYLIGHT,
            Self::Overcast => Lux::OVERCAST,
            Self::SunriseSunset => Lux::SUNRISE_SUNSET,
            Self::FullMoon => Lux::FULL_MOON,
        }
    }

    /// Surface-to-light direction (z-up): high and frontal for midday, low and
    /// grazing for sunrise/sunset.
    fn direction(self) -> [f32; 3] {
        match self {
            Self::FullDaylight => [0.35, 0.3, 1.5],
            Self::Overcast => [0.2, 0.1, 1.5],
            Self::SunriseSunset => [1.3, 0.25, 0.35],
            Self::FullMoon => [-0.6, -0.3, 0.9],
        }
    }

    /// Linear-RGB tint of the light.
    fn colour(self) -> [f32; 3] {
        match self {
            Self::FullDaylight => [1.0, 0.98, 0.95],
            Self::Overcast => [0.9, 0.94, 1.0],
            Self::SunriseSunset => [1.0, 0.6, 0.35],
            Self::FullMoon => [0.6, 0.72, 1.0],
        }
    }
}

/// An indoor bulb preset, authored from its rated luminous flux with
/// [`LightSource::point_lumens`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BulbPreset {
    Candle,
    Incandescent60W,
    Led10W,
}

impl BulbPreset {
    fn label(self) -> &'static str {
        match self {
            Self::Candle => "Candle (~12 lm)",
            Self::Incandescent60W => "60 W incandescent (~800 lm)",
            Self::Led10W => "10 W LED (~900 lm)",
        }
    }

    fn flux(self) -> Lumen {
        match self {
            Self::Candle => Lumen::CANDLE,
            Self::Incandescent60W => Lumen::INCANDESCENT_60W,
            Self::Led10W => Lumen::LED_10W,
        }
    }

    fn colour(self) -> [f32; 3] {
        match self {
            Self::Candle => [1.0, 0.55, 0.25],
            Self::Incandescent60W => [1.0, 0.82, 0.62],
            Self::Led10W => [1.0, 0.98, 0.95],
        }
    }
}

pub(crate) struct PresetsState {
    pub built: bool,
    pub scene: Scene,
    pub sky: SkyPreset,
    pub bulb_on: bool,
    pub bulb: BulbPreset,
    pub auto_exposure: bool,
    pub frame_dt: f32,
}

impl Default for PresetsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            sky: SkyPreset::FullDaylight,
            bulb_on: false,
            bulb: BulbPreset::Incandescent60W,
            auto_exposure: true,
            frame_dt: 0.0,
        }
    }
}

impl PresetsState {
    fn bulb_position(&self) -> [f32; 3] {
        let span = (PRESET_COLUMNS - 1) as f32 * PRESET_COL_SPACING;
        [span * 0.5, 0.9, 3.2]
    }

    pub(crate) fn lighting(&self) -> LightingSettings {
        let sun = {
            let mut s = LightSource::directional_lux(self.sky.direction(), self.sky.illuminance());
            s.colour = self.sky.colour();
            s.cast_shadows = true;
            s
        };

        let mut lights = vec![sun];
        if self.bulb_on {
            let mut bulb =
                LightSource::point_lumens(self.bulb_position(), self.bulb.flux(), 25.0, 0.15);
            bulb.colour = self.bulb.colour();
            bulb.cast_shadows = true;
            lights.push(bulb);
        }

        let mut l = LightingSettings::default();
        l.lights = lights;
        l.shadows.enabled = true;
        // A modest sky fill (~5% of the direct illuminance) so shadows stay
        // readable. Ambient is added without the diffuse 1/pi, so the fraction is
        // smaller than it looks. (Provisional ambient until IBL carries nits.)
        l.hemisphere_intensity = self.sky.illuminance().0 * 0.05;
        l.sky_colour = self.sky.colour();
        l.ground_colour = [0.28, 0.26, 0.24];
        l
    }

    pub(crate) fn settings(&self) -> ExposureSettings {
        if self.auto_exposure {
            // Full adaptation so every preset - across a ~400,000x lux range - is
            // driven to mid-grey. The library default (0.5) deliberately does not
            // normalise a wide brightness range.
            let auto = AutoExposure {
                adaptation: 1.0,
                dt: self.frame_dt,
                ..AutoExposure::default()
            };
            ExposureSettings::from_mode(ExposureMode::Automatic(auto))
        } else {
            // A fixed sunny-daylight camera (roughly "sunny 16").
            ExposureSettings::physical(16.0, 1.0 / 125.0, 100.0)
        }
    }
}

// ===========================================================================
// Falloff sub
// ===========================================================================

const FALLOFF_ROW_COUNT: i32 = 10;
const FALLOFF_ROW_SPACING: f32 = 2.0;

#[derive(PartialEq, Clone, Copy)]
pub(crate) enum LfType {
    Point,
    Spot,
    Directional,
}

pub(crate) struct FalloffState {
    pub built: bool,
    pub scene: Scene,
    pub light_type: LfType,
    pub intensity: f32,
    pub range: f32,
    pub radius: f32,
    pub height: f32,
    pub hemi_intensity: f32,
}

impl Default for FalloffState {
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

impl FalloffState {
    fn light(&self) -> LightSource {
        let mut s = LightSource::default();
        s.cast_shadows = false;
        s.kind = match self.light_type {
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

// ===========================================================================
// Exposure sub
// ===========================================================================

const EXPOSURE_COLUMNS: usize = 6;
const EXPOSURE_COL_SPACING: f32 = 2.4;

/// Which exposure mode the radio selects. Kept separate from
/// [`ExposureSettings`] so per-mode parameters survive switching between modes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModeSel {
    Manual,
    Physical,
    Automatic,
}

pub(crate) struct ExposureSubState {
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
    pub frame_dt: f32,
}

impl Default for ExposureSubState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            light_intensity: 4.0,
            mode: ModeSel::Automatic,
            compensation: 0.0,
            manual_ev: 0.0,
            // Low-light defaults: this demo's "scene brightness" is a small
            // abstract value, not a physical lux level, so daylight f-stops read
            // black here. See the Units & Presets tab for real lux/candela.
            aperture: 1.4,
            shutter_denom: 30.0,
            iso: 3200.0,
            auto: AutoExposure::default(),
            smooth: false,
            frame_dt: 0.0,
        }
    }
}

impl ExposureSubState {
    pub(crate) fn lighting(&self) -> LightingSettings {
        let mut s = LightSource::default();
        s.cast_shadows = true;
        s.kind = LightKind::Directional {
            direction: [0.4, 0.5, 0.9],
        };
        s.intensity = self.light_intensity;

        let mut l = LightingSettings::default();
        l.lights = vec![s];
        l.shadows.enabled = true;
        l.hemisphere_intensity = 0.05;
        l.sky_colour = [0.6, 0.7, 0.9];
        l.ground_colour = [0.25, 0.22, 0.2];
        l
    }

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

// ===========================================================================
// Container
// ===========================================================================

pub(crate) struct PhotometricLightingState {
    pub sub: LightingSub,
    pub presets: PresetsState,
    pub falloff: FalloffState,
    pub exposure: ExposureSubState,
}

impl Default for PhotometricLightingState {
    fn default() -> Self {
        Self {
            sub: LightingSub::Presets,
            presets: PresetsState::default(),
            falloff: FalloffState::default(),
            exposure: ExposureSubState::default(),
        }
    }
}

impl PhotometricLightingState {
    /// Whether the active sub's scene is built.
    pub(crate) fn built(&self) -> bool {
        match self.sub {
            LightingSub::Presets => self.presets.built,
            LightingSub::Falloff => self.falloff.built,
            LightingSub::Exposure => self.exposure.built,
        }
    }

    /// Force a rebuild of the active sub (used on sub switch so the scene and
    /// camera framing refresh).
    fn mark_unbuilt(&mut self) {
        match self.sub {
            LightingSub::Presets => self.presets.built = false,
            LightingSub::Falloff => self.falloff.built = false,
            LightingSub::Exposure => self.exposure.built = false,
        }
    }

    pub(crate) fn scene(&self) -> &Scene {
        match self.sub {
            LightingSub::Presets => &self.presets.scene,
            LightingSub::Falloff => &self.falloff.scene,
            LightingSub::Exposure => &self.exposure.scene,
        }
    }

    pub(crate) fn scene_mut(&mut self) -> &mut Scene {
        match self.sub {
            LightingSub::Presets => &mut self.presets.scene,
            LightingSub::Falloff => &mut self.falloff.scene,
            LightingSub::Exposure => &mut self.exposure.scene,
        }
    }

    pub(crate) fn lighting(&self) -> LightingSettings {
        match self.sub {
            LightingSub::Presets => self.presets.lighting(),
            LightingSub::Falloff => self.falloff.lighting(),
            LightingSub::Exposure => self.exposure.lighting(),
        }
    }

    /// Exposure to apply this frame. The Falloff sub uses the neutral default;
    /// Presets and Exposure drive it from their controls.
    pub(crate) fn exposure_settings(&self) -> ExposureSettings {
        match self.sub {
            LightingSub::Presets => self.presets.settings(),
            LightingSub::Exposure => self.exposure.settings(),
            LightingSub::Falloff => ExposureSettings::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the active sub-scene for Showcase 57 and frame the camera for it.
    pub(crate) fn build_photometric_lighting_scene(&mut self, renderer: &mut ViewportRenderer) {
        match self.lighting_state.sub {
            LightingSub::Presets => {
                self.build_presets_scene(renderer);
                self.camera = viewport_lib::Camera {
                    center: glam::Vec3::new(4.8, 0.4, 0.7),
                    distance: 16.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.2),
                    ..viewport_lib::Camera::default()
                };
            }
            LightingSub::Falloff => {
                self.build_falloff_scene(renderer);
                // Frame the row from the side so the near-to-far dimming reads.
                self.camera = viewport_lib::Camera {
                    center: glam::Vec3::new(9.0, 0.0, 0.5),
                    distance: 22.0,
                    orientation: glam::Quat::from_rotation_z(0.5)
                        * glam::Quat::from_rotation_x(1.15),
                    ..viewport_lib::Camera::default()
                };
            }
            LightingSub::Exposure => {
                self.build_exposure_scene(renderer);
                self.camera = viewport_lib::Camera {
                    center: glam::Vec3::new(6.0, 0.4, 0.7),
                    distance: 17.0,
                    orientation: glam::Quat::from_rotation_z(0.4)
                        * glam::Quat::from_rotation_x(1.2),
                    ..viewport_lib::Camera::default()
                };
            }
        }
    }

    fn build_presets_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.lighting_state.presets.scene = Scene::new();

        let span = (PRESET_COLUMNS - 1) as f32 * PRESET_COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 10.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("photometric ground mesh");
        self.lighting_state.presets.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(span * 0.5, 0.0, -0.05)),
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
            .expect("photometric sphere mesh");

        for c in 0..PRESET_COLUMNS {
            let x = c as f32 * PRESET_COL_SPACING;
            // A neutral 60%-grey card so brightness reads from the light.
            let mut m = Material::from_colour([0.6, 0.6, 0.6]);
            m.roughness = 0.55;
            m.metallic = 0.0;
            self.lighting_state.presets.scene.add_named(
                &format!("Sphere {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.8, 0.85)),
                m,
            );
        }

        self.lighting_state.presets.built = true;
    }

    fn build_falloff_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.lighting_state.falloff.scene = Scene::new();

        let row_len = (FALLOFF_ROW_COUNT - 1) as f32 * FALLOFF_ROW_SPACING;

        let ground_mesh = make_box_with_uvs(row_len + 6.0, 6.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("falloff ground mesh");
        self.lighting_state.falloff.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(row_len * 0.5, 0.0, -0.05)),
            {
                let mut m = Material::from_colour([0.5, 0.5, 0.52]);
                m.roughness = 0.9;
                m
            },
        );

        let sphere_mesh = viewport_lib::primitives::sphere(0.6, 32, 16);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("falloff sphere mesh");
        for i in 0..FALLOFF_ROW_COUNT {
            let x = i as f32 * FALLOFF_ROW_SPACING;
            self.lighting_state.falloff.scene.add_named(
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

        self.lighting_state.falloff.built = true;
    }

    fn build_exposure_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.lighting_state.exposure.scene = Scene::new();

        let span = (EXPOSURE_COLUMNS - 1) as f32 * EXPOSURE_COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 10.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("exposure ground mesh");
        self.lighting_state.exposure.scene.add_named(
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

        for c in 0..EXPOSURE_COLUMNS {
            let x = c as f32 * EXPOSURE_COL_SPACING;
            // Albedo dark on the left, bright on the right.
            let a = 0.05 + (c as f32 / (EXPOSURE_COLUMNS - 1) as f32) * 0.85;
            let mut m = Material::from_colour([a, a, a]);
            m.roughness = 0.6;
            m.metallic = 0.0;
            self.lighting_state.exposure.scene.add_named(
                &format!("Sphere {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.8, 0.85)),
                m,
            );
        }

        self.lighting_state.exposure.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_photometric_lighting(app: &mut App, ui: &mut egui::Ui) {
    let prev = app.lighting_state.sub;
    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut app.lighting_state.sub,
            LightingSub::Presets,
            "Units & Presets",
        );
        ui.selectable_value(&mut app.lighting_state.sub, LightingSub::Falloff, "Falloff");
        ui.selectable_value(
            &mut app.lighting_state.sub,
            LightingSub::Exposure,
            "Exposure",
        );
    });
    if app.lighting_state.sub != prev {
        // Switching sub swaps the scene and camera framing; force a rebuild.
        app.lighting_state.mark_unbuilt();
    }
    ui.separator();

    match app.lighting_state.sub {
        LightingSub::Presets => controls_presets(app, ui),
        LightingSub::Falloff => controls_falloff(app, ui),
        LightingSub::Exposure => controls_exposure(app, ui),
    }
}

fn controls_presets(app: &mut App, ui: &mut egui::Ui) {
    app.lighting_state.presets.frame_dt = ui.ctx().input(|i| i.stable_dt).min(0.1);
    let st = &mut app.lighting_state.presets;

    ui.label(
        "Lights authored in real units: directional in lux, bulbs in lumens. Auto-exposure adapts \
         to every preset; turn it off to see the same physical scale under a fixed daylight camera.",
    );
    ui.separator();

    ui.label("Sky (directional, lux):");
    for preset in [
        SkyPreset::FullDaylight,
        SkyPreset::Overcast,
        SkyPreset::SunriseSunset,
        SkyPreset::FullMoon,
    ] {
        ui.selectable_value(&mut st.sky, preset, preset.label());
    }
    ui.separator();

    ui.checkbox(&mut st.bulb_on, "Indoor bulb (point light, lumens)");
    ui.add_enabled_ui(st.bulb_on, |ui| {
        for preset in [
            BulbPreset::Candle,
            BulbPreset::Incandescent60W,
            BulbPreset::Led10W,
        ] {
            ui.selectable_value(&mut st.bulb, preset, preset.label());
        }
    });
    ui.separator();

    ui.checkbox(
        &mut st.auto_exposure,
        "Auto-exposure (off = fixed daylight camera)",
    );
    ui.separator();

    ui.label(format!("Sun illuminance: {:.0} lx", st.sky.illuminance().0));
    if st.bulb_on {
        let flux = st.bulb.flux();
        let cd = flux.to_point_candela();
        ui.label(format!(
            "Bulb: {:.0} lm -> {:.1} cd (over the full sphere)",
            flux.0, cd.0
        ));
    }
    if st.auto_exposure {
        ui.label("Exposure: automatic, full adaptation (each preset -> mid-grey).");
    } else {
        ui.label("Exposure: fixed f/16, 1/125 s, ISO 100 (~EV 15).");
    }
}

fn controls_falloff(app: &mut App, ui: &mut egui::Ui) {
    let st = &mut app.lighting_state.falloff;
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

fn controls_exposure(app: &mut App, ui: &mut egui::Ui) {
    app.lighting_state.exposure.frame_dt = ui.ctx().input(|i| i.stable_dt).min(0.1);

    ui.label("One lit scene under three exposure modes. Drive the scene-brightness slider and watch how each mode responds.");
    ui.separator();

    let st = &mut app.lighting_state.exposure;
    ui.add(
        egui::Slider::new(&mut st.light_intensity, 0.1..=40.0)
            .logarithmic(true)
            .text("Scene brightness (light intensity)"),
    );
    ui.add(
        egui::Slider::new(&mut st.compensation, -3.0..=3.0).text("Exposure compensation (stops)"),
    );
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Mode:");
        ui.selectable_value(&mut st.mode, ModeSel::Manual, "Manual EV");
        ui.selectable_value(&mut st.mode, ModeSel::Physical, "Physical camera");
        ui.selectable_value(&mut st.mode, ModeSel::Automatic, "Automatic");
    });
    ui.separator();

    match st.mode {
        ModeSel::Manual => {
            ui.add(egui::Slider::new(&mut st.manual_ev, -6.0..=16.0).text("EV100"));
            ui.label("Fixed exposure. Higher EV = darker image. Change the scene brightness and the image clips or crushes.");
        }
        ModeSel::Physical => {
            ui.add(egui::Slider::new(&mut st.aperture, 1.0..=22.0).text("Aperture f/"));
            ui.add(
                egui::Slider::new(&mut st.shutter_denom, 4.0..=4000.0)
                    .logarithmic(true)
                    .text("Shutter 1/x s"),
            );
            ui.add(
                egui::Slider::new(&mut st.iso, 50.0..=6400.0)
                    .logarithmic(true)
                    .text("ISO"),
            );
            ui.label("The camera triangle: EV100 = log2(N^2 / t) + log2(100 / ISO).");
            ui.label("Note: this tab's brightness slider is a small abstract value, not a physical lux level, so use fast/wide/high-ISO (low EV) settings; the Units & Presets tab authors lights in real lux/candela.");
        }
        ModeSel::Automatic => {
            ui.checkbox(&mut st.smooth, "Smooth adaptation (dt > 0)");
            ui.add(egui::Slider::new(&mut st.auto.min_ev, -10.0..=6.0).text("EV min"));
            ui.add(egui::Slider::new(&mut st.auto.max_ev, 4.0..=20.0).text("EV max"));
            ui.add(
                egui::Slider::new(&mut st.auto.speed_up, 0.1..=10.0).text("Adapt speed (brighten)"),
            );
            ui.add(
                egui::Slider::new(&mut st.auto.speed_down, 0.1..=10.0).text("Adapt speed (darken)"),
            );
            ui.add(egui::Slider::new(&mut st.auto.low_percent, 0.0..=0.9).text("Meter low clip"));
            ui.add(egui::Slider::new(&mut st.auto.high_percent, 0.1..=1.0).text("Meter high clip"));
            ui.add(
                egui::Slider::new(&mut st.auto.center_weight, 0.0..=1.0).text("Center weighting"),
            );
            ui.label("Metering the HDR target holds the image steady while the scene brightness moves. dt <= 0 snaps; smoothing eases over time.");
        }
    }

    ui.separator();
    match app.lighting_state.exposure.readout_ev() {
        Some(ev) => {
            ui.label(format!(
                "EV100: {ev:.2}  (before {:+.1} stops compensation)",
                app.lighting_state.exposure.compensation
            ));
        }
        None => {
            ui.label("EV100: metered on the GPU each frame (adapts live).");
        }
    }
}
