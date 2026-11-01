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
//! - **Exposure** - one lit scene under Passthrough / Manual EV / Physical
//!   camera / Automatic, with the resulting EV100 read out.
//! - **Exposure contract** - which values the camera's exposure scales and
//!   which it leaves alone, on a row of subjects under full daylight: a
//!   Lambert 18% card, a PBR sphere, a 200 nit emitter at
//!   `Material::emissive_exposure_weight` 1, 0.5 and 0, an
//!   `ItemSettings::unlit` card, a colour-mapped point cloud, a matcap sphere
//!   and a sphere shaded through a `recolour` material plugin. Moving from
//!   Passthrough to sunny-16 darkens the lit subjects and the weight-1
//!   emitter; the weight-0 emitter, the unlit card, the colour map and the
//!   matcap stay put. The tab also renders through `PipelineMode::Direct`
//!   (where `Automatic` falls back and reports
//!   `NoticeDetail::AutoExposureOnLdr`), splits the view across two viewports
//!   with their own exposures (`create_viewport`, `prepare_scene`,
//!   `prepare_viewport`, `render_viewport`), and captures a panorama with
//!   `capture_equirect`, which renders at the passthrough exposure whatever
//!   the camera's.
//!
//! The physical camera is the camera's own `Camera::exposure`
//! (`PhysicalExposure`), handed to the frame through
//! `PhysicalExposure::to_exposure_mode` on `DisplaySettings::exposure`.

use crate::App;
use crate::eframe::egui;
use crate::geometry::make_box_with_uvs;
use crate::shared::{two_sided, upload_material};
use viewport_lib as vpl;
use viewport_lib::Colour;
use vpl::{
    AutoExposure, ExposureMode, ExposureSettings, LightKind, LightSource, LightingSettings, Lumen,
    Lux, Material, PhysicalExposure, ViewportRenderer, scene::Scene,
};

#[path = "../plugins/recolour_tint_plugin.rs"]
mod recolour_tint_plugin;
use recolour_tint_plugin::RecolourTintPlugin;

/// Which facet of photometric lighting is shown.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LightingSub {
    Presets,
    Falloff,
    Exposure,
    Contract,
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
            s.colour = Colour::from_linear_rgb_array(self.sky.colour());
            s.cast_shadows = true;
            s
        };

        let mut lights = vec![sun];
        if self.bulb_on {
            let mut bulb =
                LightSource::point_lumens(self.bulb_position(), self.bulb.flux(), 25.0, 0.15);
            bulb.colour = Colour::from_linear_rgb_array(self.bulb.colour());
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
        l.sky_colour = Colour::from_linear_rgb_array(self.sky.colour());
        l.ground_colour = Colour::linear_rgb(0.28, 0.26, 0.24);
        l
    }

    pub(crate) fn settings(&self, camera: &PhysicalExposure) -> ExposureSettings {
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
            // The camera's own physical exposure: sunny 16 unless changed.
            ExposureSettings::from_mode(camera.to_exposure_mode())
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
                position: glam::DVec3::new(0.0, 0.0, self.height.into()),
                range: self.range,
                radius: self.radius,
            },
            LfType::Spot => LightKind::Spot {
                position: glam::DVec3::new(0.0, 0.0, self.height.into()),
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
        l.sky_colour = Colour::linear_rgb(0.7, 0.8, 1.0);
        l.ground_colour = Colour::linear_rgb(0.3, 0.3, 0.35);
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
    Passthrough,
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
        l.sky_colour = Colour::linear_rgb(0.6, 0.7, 0.9);
        l.ground_colour = Colour::linear_rgb(0.25, 0.22, 0.2);
        l
    }

    pub(crate) fn settings(&self, camera: &PhysicalExposure) -> ExposureSettings {
        let mode = exposure_mode(
            self.mode,
            self.manual_ev,
            camera,
            self.auto,
            self.smooth,
            self.frame_dt,
        );
        ExposureSettings::from_mode(mode).with_compensation(self.compensation)
    }
}

/// The exposure mode the shared selector picks. The physical camera is the
/// camera's own `PhysicalExposure`.
fn exposure_mode(
    sel: ModeSel,
    manual_ev: f32,
    camera: &PhysicalExposure,
    auto: AutoExposure,
    smooth: bool,
    frame_dt: f32,
) -> ExposureMode {
    match sel {
        ModeSel::Passthrough => ExposureMode::Passthrough,
        ModeSel::Manual => ExposureMode::Manual { ev: manual_ev },
        ModeSel::Physical => camera.to_exposure_mode(),
        ModeSel::Automatic => {
            let mut a = auto;
            a.dt = if smooth { frame_dt } else { 0.0 };
            ExposureMode::Automatic(a)
        }
    }
}

/// EV100 of a physical camera: `log2(N^2 / t) + log2(100 / ISO)`.
fn physical_ev100(camera: &PhysicalExposure) -> f32 {
    let n = camera.aperture;
    let t = camera.shutter_seconds.max(1e-6);
    (n * n / t).log2() + (100.0 / camera.iso.max(1.0)).log2()
}

/// The linear multiplier a fixed EV100 resolves to: `1 / (1.2 * 2^EV100)`.
fn ev_multiplier(ev100: f32) -> f32 {
    1.0 / (1.2 * ev100.exp2())
}

/// The shared mode selector: Passthrough, Manual EV, Physical camera, Automatic.
fn mode_selector(ui: &mut egui::Ui, mode: &mut ModeSel) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Mode:");
        ui.selectable_value(mode, ModeSel::Passthrough, "Passthrough");
        ui.selectable_value(mode, ModeSel::Manual, "Manual EV");
        ui.selectable_value(mode, ModeSel::Physical, "Physical camera");
        ui.selectable_value(mode, ModeSel::Automatic, "Automatic");
    });
}

/// Sliders for the camera's physical exposure (aperture, shutter, ISO).
fn physical_sliders(ui: &mut egui::Ui, camera: &mut PhysicalExposure) {
    ui.add(egui::Slider::new(&mut camera.aperture, 1.0..=22.0).text("Aperture f/"));
    let mut denom = 1.0 / camera.shutter_seconds.max(1e-6);
    if ui
        .add(
            egui::Slider::new(&mut denom, 4.0..=4000.0)
                .logarithmic(true)
                .text("Shutter 1/x s"),
        )
        .changed()
    {
        camera.shutter_seconds = 1.0 / denom.max(1.0);
    }
    ui.add(
        egui::Slider::new(&mut camera.iso, 50.0..=6400.0)
            .logarithmic(true)
            .text("ISO"),
    );
    if ui
        .small_button("Sunny 16 (f/16, 1/125 s, ISO 100)")
        .clicked()
    {
        *camera = PhysicalExposure::default().with_focus_distance(camera.focus_distance);
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
    pub contract: ContractState,
}

impl Default for PhotometricLightingState {
    fn default() -> Self {
        Self {
            sub: LightingSub::Presets,
            presets: PresetsState::default(),
            falloff: FalloffState::default(),
            exposure: ExposureSubState::default(),
            contract: ContractState::default(),
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
            LightingSub::Contract => self.contract.built,
        }
    }

    /// Force a rebuild of the active sub (used on sub switch so the scene and
    /// camera framing refresh).
    fn mark_unbuilt(&mut self) {
        match self.sub {
            LightingSub::Presets => self.presets.built = false,
            LightingSub::Falloff => self.falloff.built = false,
            LightingSub::Exposure => self.exposure.built = false,
            LightingSub::Contract => self.contract.built = false,
        }
    }

    pub(crate) fn scene(&self) -> &Scene {
        match self.sub {
            LightingSub::Presets => &self.presets.scene,
            LightingSub::Falloff => &self.falloff.scene,
            LightingSub::Exposure => &self.exposure.scene,
            LightingSub::Contract => &self.contract.scene,
        }
    }

    pub(crate) fn scene_mut(&mut self) -> &mut Scene {
        match self.sub {
            LightingSub::Presets => &mut self.presets.scene,
            LightingSub::Falloff => &mut self.falloff.scene,
            LightingSub::Exposure => &mut self.exposure.scene,
            LightingSub::Contract => &mut self.contract.scene,
        }
    }

    pub(crate) fn lighting(&self) -> LightingSettings {
        match self.sub {
            LightingSub::Presets => self.presets.lighting(),
            LightingSub::Falloff => self.falloff.lighting(),
            LightingSub::Exposure => self.exposure.lighting(),
            LightingSub::Contract => self.contract.lighting(),
        }
    }

    /// Exposure to apply this frame. The Falloff sub uses the neutral default;
    /// the others drive it from their controls and the camera's physical
    /// exposure.
    pub(crate) fn exposure_settings(&self, camera: &PhysicalExposure) -> ExposureSettings {
        match self.sub {
            LightingSub::Presets => self.presets.settings(camera),
            LightingSub::Exposure => self.exposure.settings(camera),
            LightingSub::Contract => self.contract.settings(camera),
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
                self.camera = vpl::Camera::orbit(
                    glam::DVec3::new(4.8, 0.4, 0.7),
                    16.0,
                    glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.2),
                );
            }
            LightingSub::Falloff => {
                self.build_falloff_scene(renderer);
                // Frame the row from the side so the near-to-far dimming reads.
                self.camera = vpl::Camera::orbit(
                    glam::DVec3::new(9.0, 0.0, 0.5),
                    22.0,
                    glam::Quat::from_rotation_z(0.5) * glam::Quat::from_rotation_x(1.15),
                );
            }
            LightingSub::Exposure => {
                self.build_exposure_scene(renderer);
                self.camera = vpl::Camera::orbit(
                    glam::DVec3::new(6.0, 0.4, 0.7),
                    17.0,
                    glam::Quat::from_rotation_z(0.4) * glam::Quat::from_rotation_x(1.2),
                );
                // Low-light defaults: this tab's "scene brightness" is a small
                // abstract value, not a physical lux level, so daylight
                // f-stops read black here. See Units & Presets for real lux.
                self.camera.exposure = PhysicalExposure::new(1.4, 1.0 / 30.0, 3200.0);
            }
            LightingSub::Contract => {
                build_contract_scene(self, renderer);
                self.camera = vpl::Camera::orbit(
                    glam::DVec3::new(f64::from(CONTRACT_SPAN) * 0.5, 0.0, 0.5),
                    20.0,
                    glam::Quat::from_rotation_x(1.05),
                );
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
                let mut m = Material::from_colour(Colour::linear_rgb(0.5, 0.5, 0.52));
                m.roughness = 0.9;
                upload_material(renderer, m)
            },
        );

        let sphere_mesh = vpl::primitives::sphere(0.8, 48, 24);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("photometric sphere mesh");

        for c in 0..PRESET_COLUMNS {
            let x = c as f32 * PRESET_COL_SPACING;
            // A neutral 60%-grey card so brightness reads from the light.
            let mut m = Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.6));
            m.roughness = 0.55;
            m.metallic = 0.0;
            self.lighting_state.presets.scene.add_named(
                &format!("Sphere {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.8, 0.85)),
                upload_material(renderer, m),
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
                let mut m = Material::from_colour(Colour::linear_rgb(0.5, 0.5, 0.52));
                m.roughness = 0.9;
                upload_material(renderer, m)
            },
        );

        let sphere_mesh = vpl::primitives::sphere(0.6, 32, 16);
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
                    let mut m = Material::from_colour(Colour::linear_rgb(0.9, 0.9, 0.9));
                    m.roughness = 0.4;
                    upload_material(renderer, m)
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
                let mut m = Material::from_colour(Colour::linear_rgb(0.45, 0.45, 0.48));
                m.roughness = 0.95;
                upload_material(renderer, m)
            },
        );

        let sphere_mesh = vpl::primitives::sphere(0.8, 48, 24);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("exposure sphere mesh");

        for c in 0..EXPOSURE_COLUMNS {
            let x = c as f32 * EXPOSURE_COL_SPACING;
            // Albedo dark on the left, bright on the right.
            let a = 0.05 + (c as f32 / (EXPOSURE_COLUMNS - 1) as f32) * 0.85;
            let mut m = Material::from_colour(Colour::linear_rgb(a, a, a));
            m.roughness = 0.6;
            m.metallic = 0.0;
            self.lighting_state.exposure.scene.add_named(
                &format!("Sphere {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.8, 0.85)),
                upload_material(renderer, m),
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
        ui.selectable_value(
            &mut app.lighting_state.sub,
            LightingSub::Contract,
            "Exposure contract",
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
        LightingSub::Contract => controls_contract(app, ui),
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

    ui.label("One lit scene under each exposure mode. Drive the scene-brightness slider and watch how each mode responds.");
    ui.separator();

    let camera = &mut app.camera.exposure;
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

    mode_selector(ui, &mut st.mode);
    ui.separator();

    match st.mode {
        ModeSel::Passthrough => {
            ui.label("A multiplier of exactly 1: lights read as authored.");
        }
        ModeSel::Manual => {
            ui.add(egui::Slider::new(&mut st.manual_ev, -6.0..=16.0).text("EV100"));
            ui.label("Fixed exposure. Higher EV = darker image. Change the scene brightness and the image clips or crushes.");
        }
        ModeSel::Physical => {
            ui.label("The camera's own exposure (Camera::exposure):");
            physical_sliders(ui, camera);
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
                egui::Slider::new(&mut st.auto.centre_weight, 0.0..=1.0).text("Centre weighting"),
            );
            ui.label("Metering the HDR target holds the image steady while the scene brightness moves. dt <= 0 snaps; smoothing eases over time.");
        }
    }

    ui.separator();
    let readout = match app.lighting_state.exposure.mode {
        ModeSel::Passthrough => Some(vpl::PASSTHROUGH_EV100),
        ModeSel::Manual => Some(app.lighting_state.exposure.manual_ev),
        ModeSel::Physical => Some(physical_ev100(&app.camera.exposure)),
        ModeSel::Automatic => None,
    };
    match readout {
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

// ---------------------------------------------------------------------------
// Lazy scene build
// ---------------------------------------------------------------------------

/// Whether the host should call [`build`] before the next frame.
pub(crate) fn needs_build(app: &crate::App) -> bool {
    !app.lighting_state.built()
}

/// Build this showcase's scene and frame its opening camera. Called once, on
/// the first frame after it becomes the active showcase.
pub(crate) fn build(app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
    // Builds the active sub-scene and frames the camera for it.
    app.build_photometric_lighting_scene(renderer);
}

// ---------------------------------------------------------------------------
// Per-frame scene contents
// ---------------------------------------------------------------------------

/// Collect this showcase's render items and lighting for the frame. `_out` carries
/// the few extra frame settings a showcase can set alongside its items.
pub(crate) fn scene(
    app: &mut crate::App,
    _frame: &crate::eframe::Frame,
    _out: &mut crate::SceneOverrides,
) -> crate::SceneContents {
    let (items, bg_colour, lighting, scene_gen, sel_gen) = {
        let items = app
            .lighting_state
            .scene_mut()
            .collect_render_items(&vpl::Selection::new());
        let lighting = app.lighting_state.lighting();
        let sg = app.lighting_state.scene().version();
        (items, None, lighting, sg, 0)
    };
    crate::SceneContents {
        items,
        bg_colour,
        lighting,
        scene_gen,
        sel_gen,
    }
}

// ---------------------------------------------------------------------------
// Per-frame frame-data tweaks
// ---------------------------------------------------------------------------

/// Fold this showcase's own contributions into the assembled frame: extra
/// render items, overlays, and effect settings that are re-submitted every
/// frame rather than baked into the scene.
pub(crate) fn frame(app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
    // Drive the exposure model from the active sub's controls. The HDR
    // pipeline stays enabled (default) so the tone map / exposure
    // buffer path runs.
    fd.effects.display.exposure = app.lighting_state.exposure_settings(&app.camera.exposure);
    if app.lighting_state.sub == LightingSub::Contract {
        contract_frame(app, fd, ctx);
    }
    let cam = &app.camera;
    let far = (cam.focus_distance() * 3.0).max(60.0);
    fd.camera.render_camera =
        vpl::RenderCamera::from_camera(cam).with_depth_range(cam.depth_range().0, far);
}

// ---------------------------------------------------------------------------
// Viewport overlay and per-frame tick
// ---------------------------------------------------------------------------

/// Draw this showcase's own egui overlay on top of the rendered viewport:
/// selection rectangles, mode readouts, and in-scene labels.

/// Advance this showcase's animation and ask for another frame. Runs after the
/// viewport has been drawn, so it only affects the next frame.
pub(crate) fn tick(_app: &mut crate::App, cx: &crate::ViewportCtx) {
    // ----- Exposure (59): keep repainting so exposure-control changes
    // take effect and auto-exposure adaptation runs. Exposure lives in
    // `EffectsFrame`, not the scene, so it never dirties the scene; the
    // renderer is on-demand, so without a repaint request a slider change
    // is dropped until something else re-renders. Auto-exposure smoothing
    // also needs continuous frames.
    cx.egui.request_repaint();
}

/// Route a viewport click for this showcase. The host calls this for a plain
/// click that no gizmo or widget has already consumed; `pos` is in viewport
/// pixels.

/// Handle drag gestures this showcase owns, before the camera controller runs.

/// Advance this showcase's own camera animation or object motion for the frame.

/// Update this showcase's interactive widgets for the frame.

/// Flush any per-frame GPU writes this showcase has queued.

/// Cache gizmo placement for next frame's hit-testing.

/// Take over the whole viewport for this frame. Returning false leaves the
/// host's normal single-viewport path in charge.
pub(crate) fn viewport_override(
    app: &mut crate::App,
    ui: &mut crate::eframe::egui::Ui,
    cx: &crate::ViewportCtx,
) -> bool {
    if app.lighting_state.sub != LightingSub::Contract
        || !app.lighting_state.contract.two_viewports
        || !app.lighting_state.contract.built
    {
        return false;
    }
    contract_two_viewports(app, ui, cx);
    true
}

/// Drive the orbit controller for this showcase. Returning false leaves the
/// host to run the usual suppress-or-apply path.
pub(crate) fn drive_camera(_app: &mut crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

/// Whether the orbit controller should resolve without moving the camera this
/// frame. This showcase never suppresses it.
pub(crate) fn suppress_orbit(_app: &crate::App, _cx: &crate::ViewportCtx) -> bool {
    false
}

// ---------------------------------------------------------------------------
// Showcase entry point
// ---------------------------------------------------------------------------

/// Stateless handle for this showcase; the scene state lives on [`crate::App`].
pub(crate) struct ScPhotometricLighting;

/// The registry's handle to this showcase.
pub(crate) static SHOWCASE: ScPhotometricLighting = ScPhotometricLighting;

impl crate::Showcase for ScPhotometricLighting {
    fn needs_build(&self, app: &crate::App) -> bool {
        needs_build(app)
    }
    fn build(&self, app: &mut crate::App, renderer: &mut vpl::ViewportRenderer) {
        build(app, renderer)
    }
    fn scene(
        &self,
        app: &mut crate::App,
        frame: &crate::eframe::Frame,
        out: &mut crate::SceneOverrides,
    ) -> crate::SceneContents {
        scene(app, frame, out)
    }
    fn frame(&self, app: &mut crate::App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
        frame(app, fd, ctx)
    }
    fn tick(&self, app: &mut crate::App, cx: &crate::ViewportCtx) {
        tick(app, cx)
    }
    fn viewport_override(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        cx: &crate::ViewportCtx,
    ) -> bool {
        viewport_override(app, ui, cx)
    }
    fn drive_camera(&self, app: &mut crate::App, cx: &crate::ViewportCtx) -> bool {
        drive_camera(app, cx)
    }
    fn suppress_orbit(&self, app: &crate::App, cx: &crate::ViewportCtx) -> bool {
        suppress_orbit(app, cx)
    }
    fn controls(
        &self,
        app: &mut crate::App,
        ui: &mut crate::eframe::egui::Ui,
        _frame: &crate::eframe::Frame,
    ) {
        controls_photometric_lighting(app, ui)
    }
    fn overlay(&self, app: &mut crate::App, ui: &mut egui::Ui, cx: &crate::ViewportCtx) {
        if app.lighting_state.sub == LightingSub::Contract {
            contract_overlay(app, ui, cx.rect);
        }
    }
    fn tour(&self, app: &mut crate::App, step: u32) -> bool {
        let st = &mut app.lighting_state;
        let to_contract = |st: &mut PhotometricLightingState| {
            if st.sub != LightingSub::Contract {
                st.sub = LightingSub::Contract;
                st.mark_unbuilt();
            }
        };
        match step {
            0 => {}
            1 => {
                to_contract(st);
                st.contract.mode = ModeSel::Passthrough;
            }
            2 => st.contract.mode = ModeSel::Physical,
            3 => {
                st.contract.mode = ModeSel::Automatic;
                st.contract.direct_path = true;
            }
            _ => return false,
        }
        true
    }
}

// ===========================================================================
// Exposure contract sub
// ===========================================================================

/// Spacing of the subjects along X.
const CONTRACT_STEP: f32 = 1.6;
/// Number of subjects in the row.
const CONTRACT_SUBJECTS: usize = 9;
/// Length of the row.
const CONTRACT_SPAN: f32 = (CONTRACT_SUBJECTS - 1) as f32 * CONTRACT_STEP;
/// Luminance of the emissive cards, in nits.
const EMITTER_NITS: f32 = 200.0;
/// Albedo of the Lambert card.
const GREY_CARD: f32 = 0.18;

/// How a subject responds to the camera's exposure.
#[derive(Clone, Copy)]
enum Response {
    /// Lit radiance: scales by the full multiplier.
    Lit,
    /// An emitter at this exposure weight: scales by `mix(1, k, w)`.
    Emissive(f32),
    /// A resolved colour: exposure does not touch it.
    Resolved,
}

/// One subject of the row: its label, how it responds and why.
const SUBJECTS: [(&str, Response, &str); CONTRACT_SUBJECTS] = [
    (
        "Lambert 18% card",
        Response::Lit,
        "Lit: every light is exposed before shading.",
    ),
    (
        "PBR sphere",
        Response::Lit,
        "Lit: sun, fill and reflections all exposed.",
    ),
    (
        "Emitter, weight 1",
        Response::Emissive(1.0),
        "A light source: exposed like the sun.",
    ),
    (
        "Emitter, weight 0.5",
        Response::Emissive(0.5),
        "Half way, in multiplier terms.",
    ),
    (
        "Emitter, weight 0",
        Response::Emissive(0.0),
        "Display-relative: shows its value whatever the EV.",
    ),
    (
        "Unlit orange card",
        Response::Resolved,
        "ItemSettings::unlit: a colour, not light.",
    ),
    (
        "Colour-mapped points",
        Response::Resolved,
        "A colour map is data, so the colour bar stays true.",
    ),
    (
        "Matcap sphere",
        Response::Resolved,
        "A matcap is captured lighting, shown as is.",
    ),
    (
        "Recolour sphere",
        Response::Lit,
        "The hook sees and returns exposed radiance.",
    ),
];

/// A captured panorama, kept as display pixels for the sidebar and as
/// radiance for comparing two captures.
pub(crate) struct ProbeCapture {
    radiance: Vec<f32>,
    rgba8: Vec<u8>,
    size: [usize; 2],
    /// The camera EV100 at capture time, or `None` under automatic exposure.
    ev: Option<f32>,
    /// Largest per-channel difference from the capture before, if any.
    max_diff: Option<f32>,
}

pub(crate) struct ContractState {
    pub built: bool,
    pub scene: Scene,
    pub mode: ModeSel,
    pub manual_ev: f32,
    pub direct_path: bool,
    pub two_viewports: bool,
    pub right_ev: f32,
    viewports: Option<[vpl::ViewportId; 2]>,
    targets: Vec<crate::Target>,
    recolour: Option<vpl::MaterialPluginId>,
    cloud_positions: Vec<[f32; 3]>,
    cloud_scalars: Vec<f32>,
    /// Notices taken from the renderer, newest frame first.
    notices: Vec<String>,
    capture_requested: bool,
    probe: Option<ProbeCapture>,
    probe_texture: Option<egui::TextureHandle>,
    frame_dt: f32,
}

impl Default for ContractState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            mode: ModeSel::Physical,
            manual_ev: 15.0,
            direct_path: false,
            two_viewports: false,
            right_ev: 13.0,
            viewports: None,
            targets: Vec::new(),
            recolour: None,
            cloud_positions: Vec::new(),
            cloud_scalars: Vec::new(),
            notices: Vec::new(),
            capture_requested: false,
            probe: None,
            probe_texture: None,
            frame_dt: 0.0,
        }
    }
}

impl ContractState {
    fn lighting(&self) -> LightingSettings {
        let mut l = LightingSettings::daylight();
        l.lights[0].kind = LightKind::Directional {
            direction: [0.35, -0.8, 1.0],
        };
        l.lights[0].cast_shadows = true;
        l.shadows.enabled = true;
        l
    }

    fn settings(&self, camera: &PhysicalExposure) -> ExposureSettings {
        let auto = AutoExposure {
            adaptation: 1.0,
            ..AutoExposure::default()
        };
        ExposureSettings::from_mode(exposure_mode(
            self.mode,
            self.manual_ev,
            camera,
            auto,
            true,
            self.frame_dt,
        ))
    }

    /// The multiplier this frame's fixed exposure resolves to, or `None` under
    /// automatic exposure, which is metered on the GPU.
    fn multiplier(&self, camera: &PhysicalExposure) -> Option<f32> {
        self.ev(camera).map(ev_multiplier)
    }

    fn ev(&self, camera: &PhysicalExposure) -> Option<f32> {
        match self.mode {
            ModeSel::Passthrough => Some(vpl::PASSTHROUGH_EV100),
            ModeSel::Manual => Some(self.manual_ev),
            ModeSel::Physical => Some(physical_ev100(camera)),
            ModeSel::Automatic => None,
        }
    }
}

/// A black surface emitting white at [`EMITTER_NITS`] with exposure weight `w`.
fn contract_emitter(w: f32) -> Material {
    let mut m = Material::pbr(Colour::linear_rgb(0.0, 0.0, 0.0), 0.0, 1.0);
    m.specular_factor = 0.0;
    m.emissive = Colour::linear_rgb(1.0, 1.0, 1.0);
    m.emissive_strength = EMITTER_NITS;
    m.with_emissive_exposure_weight(w)
}

fn build_contract_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    let st = &mut app.lighting_state.contract;
    st.scene = Scene::new();
    let device = &app.device;
    let upload = |renderer: &mut ViewportRenderer, mesh: &vpl::MeshData| {
        renderer
            .resources_mut()
            .upload_mesh_data(device, mesh)
            .expect("contract mesh upload")
    };
    let ground = upload(
        renderer,
        &vpl::primitives::cuboid(CONTRACT_SPAN + 8.0, 16.0, 0.1),
    );
    let card = upload(renderer, &vpl::primitives::plane(1.0, 1.0));
    let sphere = upload(renderer, &vpl::primitives::sphere(0.5, 48, 24));

    let lambert = |albedo: [f32; 3]| {
        let mut m = Material::from_colour(Colour::from_linear_rgb_array(albedo));
        m.specular = 0.0;
        m
    };
    let ground_mat = upload_material(renderer, lambert([0.3, 0.3, 0.3]));
    st.scene.add_named(
        "Ground",
        Some(ground),
        glam::Mat4::from_translation(glam::Vec3::new(CONTRACT_SPAN * 0.5, 0.0, -0.05)),
        ground_mat,
    );

    renderer
        .resources_mut()
        .ensure_matcaps_initialized(&app.device, &app.queue);
    let matcap = renderer
        .resources()
        .builtin_matcap_id(vpl::BuiltinMatcap::Ceramic);
    if st.recolour.is_none() {
        st.recolour = Some(
            renderer
                .resources_mut()
                .register_material_plugin(&app.device, &RecolourTintPlugin)
                .expect("register recolour plugin"),
        );
    }

    let x = |i: usize| i as f32 * CONTRACT_STEP;
    // Cards lean back towards the camera; spheres sit on the ground.
    let card_at = |i: usize| {
        glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(1.1),
            glam::Quat::from_rotation_x(1.0),
            glam::Vec3::new(x(i), 0.0, 0.5),
        )
    };
    let sphere_at = |i: usize| glam::Mat4::from_translation(glam::Vec3::new(x(i), 0.0, 0.5));

    let mut add = |name: &str, mesh, at, m: Material, unlit: bool| {
        let mat = upload_material(renderer, two_sided(m));
        let id = st.scene.add_named(name, Some(mesh), at, mat);
        if unlit {
            let mut settings = vpl::ItemSettings::default();
            settings.unlit = true;
            st.scene.set_appearance(id, settings);
        }
    };
    add(
        SUBJECTS[0].0,
        card,
        card_at(0),
        lambert([GREY_CARD; 3]),
        false,
    );
    add(
        SUBJECTS[1].0,
        sphere,
        sphere_at(1),
        Material::pbr(Colour::linear_rgb(0.5, 0.5, 0.5), 0.0, 0.45),
        false,
    );
    for (i, w) in [(2, 1.0), (3, 0.5), (4, 0.0)] {
        add(SUBJECTS[i].0, card, card_at(i), contract_emitter(w), false);
    }
    add(
        SUBJECTS[5].0,
        card,
        card_at(5),
        Material::from_colour(Colour::linear_rgb(0.9, 0.35, 0.05)),
        true,
    );
    let mut m = Material::from_colour(Colour::linear_rgb(0.8, 0.8, 0.8));
    m.shading_model = vpl::ShadingModel::Matcap(matcap);
    add(SUBJECTS[7].0, sphere, sphere_at(7), m, false);
    let recolour = Material::pbr(Colour::linear_rgb(0.6, 0.6, 0.6), 0.0, 0.45)
        .with_plugin(st.recolour.expect("registered above"))
        .with_param("tint", [1.0f32, 0.45, 0.15]);
    add(SUBJECTS[8].0, sphere, sphere_at(8), recolour, false);

    // The point cloud: a 16 x 16 upright patch, coloured by its height.
    st.cloud_positions.clear();
    st.cloud_scalars.clear();
    for i in 0..16 {
        for j in 0..16 {
            let u = i as f32 / 15.0;
            let v = j as f32 / 15.0;
            st.cloud_positions
                .push([x(6) - 0.5 + u, 0.0, 0.05 + v * 1.0]);
            st.cloud_scalars.push(v);
        }
    }

    if st.viewports.is_none() {
        st.viewports = Some([
            renderer.create_viewport(&app.device),
            renderer.create_viewport(&app.device),
        ]);
    }
    app.camera.exposure = PhysicalExposure::default();
    st.built = true;
}

/// The point cloud and the per-frame renderer work: notices, probe capture.
fn contract_frame(app: &mut App, fd: &mut vpl::FrameData, ctx: &crate::FrameCtx) {
    let st = &mut app.lighting_state.contract;
    if st.direct_path {
        fd.effects.display.mode = vpl::PipelineMode::Direct;
    }
    let mut cloud = viewport_lib_plugins::item_types::point_cloud::PointCloudItem::default();
    cloud.positions = st.cloud_positions.clone();
    cloud.colour = vpl::ColourSource::Scalar {
        values: st.cloud_scalars.clone(),
        range: Some((0.0, 1.0)),
        colourmap: Some(vpl::ColourmapId(vpl::BuiltinColourmap::Viridis as usize)),
    };
    cloud.size = vpl::SizeSource::Uniform(7.0);
    fd.scene
        .items_mut::<viewport_lib_plugins::item_types::point_cloud::PointCloudItem>()
        .push(cloud);

    let Some(rs) = ctx.frame.wgpu_render_state() else {
        return;
    };
    let mut guard = rs.renderer.write();
    let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() else {
        return;
    };
    // Notices reported by the last frame's render.
    let notices = renderer.take_notices();
    st.notices = notices
        .iter()
        .map(|n| format!("{:?} x{}: {}", n.severity, n.count, n.detail))
        .collect();
    if st.capture_requested {
        st.capture_requested = false;
        let ev = st.ev(&app.camera.exposure);
        let centre = [CONTRACT_SPAN * 0.5, -2.0, 1.2];
        let pano = renderer.capture_equirect(&app.device, &app.queue, fd, centre, 64, 64);
        let (w, h) = (pano.width as usize, pano.height as usize);
        let max_diff = st.probe.as_ref().and_then(|prev| {
            (prev.radiance.len() == pano.rgba.len()).then(|| {
                prev.radiance
                    .iter()
                    .zip(&pano.rgba)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            })
        });
        // A simple Reinhard and sRGB encode for the thumbnail.
        let rgba8 = pano
            .rgba
            .chunks_exact(4)
            .flat_map(|p| {
                let enc = |c: f32| {
                    let t = (c / (1.0 + c)).clamp(0.0, 1.0);
                    (t.powf(1.0 / 2.2) * 255.0).round() as u8
                };
                [
                    enc(p[0] / 4000.0),
                    enc(p[1] / 4000.0),
                    enc(p[2] / 4000.0),
                    255,
                ]
            })
            .collect();
        st.probe = Some(ProbeCapture {
            radiance: pano.rgba,
            rgba8,
            size: [w, h],
            ev,
            max_diff,
        });
        st.probe_texture = None;
    }
}

fn controls_contract(app: &mut App, ui: &mut egui::Ui) {
    app.lighting_state.contract.frame_dt = ui.ctx().input(|i| i.stable_dt).min(0.1);
    ui.label(
        "Full daylight (100 000 lx sun, 8 000 nit fill) on a row of subjects. Move from \
         Passthrough to the physical camera (sunny 16): the lit subjects and the weight-1 \
         emitter come down into range; the weight-0 emitter, the unlit card, the colour map \
         and the matcap do not move.",
    );
    ui.separator();
    let camera = &mut app.camera.exposure;
    let st = &mut app.lighting_state.contract;
    mode_selector(ui, &mut st.mode);
    match st.mode {
        ModeSel::Manual => {
            ui.add(egui::Slider::new(&mut st.manual_ev, -2.0..=18.0).text("EV100"));
        }
        ModeSel::Physical => physical_sliders(ui, camera),
        _ => {}
    }
    let k = st.multiplier(camera);
    match (st.ev(camera), k) {
        (Some(ev), Some(k)) => ui.label(format!("EV100 {ev:.2}: multiplier k = {k:.3e}")),
        _ => ui.label("Automatic: k is metered on the GPU each frame."),
    };

    ui.separator();
    ui.label("Per subject (value before the tone map):");
    egui::Grid::new("contract_table")
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Subject");
            ui.strong("Scale");
            ui.strong("Value");
            ui.end_row();
            // Sun lux and fill nits of the daylight preset, at normal incidence.
            let sun_lux = Lux::FULL_DAYLIGHT.0;
            for (name, response, why) in SUBJECTS {
                ui.label(name).on_hover_text(why);
                let (scale, value) = match (response, k) {
                    (Response::Lit, Some(k)) => (
                        format!("k = {k:.2e}"),
                        if name.starts_with("Lambert") {
                            format!("{:.3}", GREY_CARD * sun_lux / std::f32::consts::PI * k)
                        } else {
                            "lit x k".to_string()
                        },
                    ),
                    (Response::Emissive(w), Some(k)) => {
                        let m = 1.0 + (k - 1.0) * w;
                        (format!("{m:.3}"), format!("{:.3}", EMITTER_NITS * m))
                    }
                    (Response::Resolved, _) => ("1".to_string(), "as authored".to_string()),
                    (_, None) => ("metered".to_string(), "-".to_string()),
                };
                ui.label(scale);
                ui.label(value);
                ui.end_row();
            }
        });
    ui.weak(
        "Lambert value: 0.18 x 100 000 lx / pi x k at normal incidence, before the fill. \
         Hover a subject for why it scales the way it does.",
    );

    ui.separator();
    ui.checkbox(&mut st.direct_path, "Direct path (PipelineMode::Direct)");
    if st.direct_path {
        ui.weak("No HDR target: Automatic cannot meter, so it draws at passthrough.");
    }
    ui.label("Notices:");
    if st.notices.is_empty() {
        ui.weak("(none)");
    }
    for n in &st.notices {
        ui.add(egui::Label::new(egui::RichText::new(n).small()).wrap());
    }

    ui.separator();
    ui.checkbox(&mut st.two_viewports, "Two viewports");
    ui.add_enabled(
        st.two_viewports,
        egui::Slider::new(&mut st.right_ev, 8.0..=18.0).text("Right half EV100"),
    );

    ui.separator();
    if ui.button("Capture probe").clicked() {
        st.capture_requested = true;
    }
    if let Some(probe) = &st.probe {
        let tex = st.probe_texture.get_or_insert_with(|| {
            ui.ctx().load_texture(
                "contract_probe",
                egui::ColorImage::from_rgba_unmultiplied(probe.size, &probe.rgba8),
                egui::TextureOptions::LINEAR,
            )
        });
        ui.image((tex.id(), egui::vec2(200.0, 100.0)));
        match probe.ev {
            Some(ev) => ui.label(format!("Captured with the camera at EV100 {ev:.2}.")),
            None => ui.label("Captured under automatic exposure."),
        };
        if let Some(d) = probe.max_diff {
            ui.label(format!("Max difference from the capture before: {d:.3e}"));
        }
        ui.weak("A capture holds absolute radiance: change the EV and capture again.");
    }
}

/// The colour bar for the point cloud, drawn over the bottom-left corner.
fn contract_overlay(_app: &mut App, ui: &mut egui::Ui, rect: egui::Rect) {
    let bar = egui::Rect::from_min_size(
        rect.left_bottom() + egui::vec2(60.0, -34.0),
        egui::vec2(140.0, 12.0),
    );
    let painter = ui.painter_at(rect);
    let steps = 32;
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        let c = vpl::BuiltinColourmap::Viridis.sample(t);
        let x0 = bar.left() + bar.width() * i as f32 / steps as f32;
        let x1 = bar.left() + bar.width() * (i + 1) as f32 / steps as f32;
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, bar.top()), egui::pos2(x1, bar.bottom())),
            0.0,
            egui::Color32::from_rgb(
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
            ),
        );
    }
    let text = |pos, s: &str, align| {
        painter.text(
            pos,
            align,
            s,
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(220),
        );
    };
    text(bar.left_top(), "0", egui::Align2::LEFT_BOTTOM);
    text(bar.right_top(), "1", egui::Align2::RIGHT_BOTTOM);
    text(
        bar.center_top(),
        "point height",
        egui::Align2::CENTER_BOTTOM,
    );
}

/// Feed the host's input to the orbit controller, for the frames this
/// showcase draws the viewport itself.
fn drive_orbit(app: &mut App, ui: &egui::Ui, cx: &crate::ViewportCtx) {
    use vpl::input::adapters::{AdapterCtx, egui as vpl_egui};
    use vpl::{ViewportContext, ViewportEvent};
    let rect = cx.rect;
    let ownership = vpl_egui::ownership(cx.response, ui.ctx(), rect, false);
    app.input.begin_frame(ViewportContext::new(
        ownership,
        [rect.width(), rect.height()],
    ));
    app.controller
        .set_viewport_size([rect.width(), rect.height()]);
    let actx = AdapterCtx::new(1.0, glam::Vec2::new(rect.left(), rect.top()));
    let mut events = Vec::new();
    ui.input(|i| {
        vpl_egui::begin_frame(i, &actx, &mut events);
        for event in &i.events {
            vpl_egui::translate(event, &actx, &mut events);
        }
    });
    // The orbit takes the pointer only; this showcase binds no keys.
    for ev in events {
        if !matches!(ev, ViewportEvent::Key { .. }) {
            app.input.push_event(ev);
        }
    }
    let action = app.input.resolve();
    let dt = ui.ctx().input(|i| i.stable_dt);
    app.controller.apply(&mut app.camera, &action, dt);
    app.camera.set_aspect_ratio(rect.width(), rect.height());
}

/// Draw the contract scene through two viewports over one prepared scene:
/// the left half at the selected exposure, the right half at a manual EV.
fn contract_two_viewports(app: &mut App, ui: &mut egui::Ui, cx: &crate::ViewportCtx) {
    drive_orbit(app, ui, cx);
    let rect = cx.rect;
    let ppp = ui.ctx().pixels_per_point();
    let dt = ui.ctx().input(|i| i.stable_dt.min(1.0 / 15.0));
    let Some([vp_left, vp_right]) = app.lighting_state.contract.viewports else {
        return;
    };
    let mut left = app.build_frame_data(rect.width(), rect.height(), ppp, cx.frame, dt);
    let mut right = app.build_frame_data(rect.width(), rect.height(), ppp, cx.frame, dt);
    left.camera = left.camera.with_viewport_id(vp_left);
    right.camera = right.camera.with_viewport_id(vp_right);
    right.effects.display.exposure = ExposureSettings::manual(app.lighting_state.contract.right_ev);

    let rs = cx.frame.wgpu_render_state().expect("wgpu required");
    let size_px = [
        (rect.width() * ppp).round().max(1.0) as u32,
        (rect.height() * ppp).round().max(1.0) as u32,
    ];
    let mut guard = rs.renderer.write();
    let st = &mut app.lighting_state.contract;
    if st.targets.len() != 2 || st.targets.iter().any(|t| t.inner.size() != size_px) {
        for old in st.targets.drain(..) {
            guard.free_texture(&old.id);
        }
        for _ in 0..2 {
            let inner = vpl::OffscreenViewportTarget::new(&app.device, rs.target_format, size_px);
            let id = guard.register_native_texture(
                &app.device,
                inner.sample_view(),
                crate::eframe::wgpu::FilterMode::Linear,
            );
            st.targets.push(crate::Target { inner, id });
        }
    }
    if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
        let mut owned = renderer.owned();
        let (scene_fx, _) = left.effects.split();
        let token = owned.prepare_scene(&app.device, &app.queue, &left, &scene_fx);
        owned.prepare_viewport(&app.device, &app.queue, &token, vp_left, &left);
        owned.prepare_viewport(&app.device, &app.queue, &token, vp_right, &right);
        for (vp, fd, target) in [
            (vp_left, &left, &st.targets[0]),
            (vp_right, &right, &st.targets[1]),
        ] {
            let cmd =
                owned.render_viewport(&app.device, &app.queue, target.inner.render_view(), vp, fd);
            app.queue.submit(std::iter::once(cmd));
        }
    }
    let ids = [st.targets[0].id, st.targets[1].id];
    drop(guard);

    let mid = rect.center().x;
    let painter = ui.painter_at(rect);
    painter.image(
        ids[0],
        egui::Rect::from_min_max(rect.min, egui::pos2(mid, rect.max.y)),
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(0.5, 1.0)),
        egui::Color32::WHITE,
    );
    painter.image(
        ids[1],
        egui::Rect::from_min_max(egui::pos2(mid, rect.min.y), rect.max),
        egui::Rect::from_min_max(egui::pos2(0.5, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    painter.line_segment(
        [egui::pos2(mid, rect.top()), egui::pos2(mid, rect.bottom())],
        egui::Stroke::new(1.0, egui::Color32::from_gray(200)),
    );
    painter.text(
        egui::pos2(mid + 6.0, rect.top() + 6.0),
        egui::Align2::LEFT_TOP,
        format!("EV100 {:.1}", st.right_ev),
        egui::FontId::proportional(13.0),
        egui::Color32::WHITE,
    );
    contract_overlay(app, ui, rect);
    ui.ctx().request_repaint();
}
