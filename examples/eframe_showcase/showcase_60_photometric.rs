//! Showcase 60: Photometric Presets.
//!
//! Lights are authored in real photometric units - directional lights in **lux**
//! ([`Lux`]) and point/spot lights in **candela** ([`Candela`]), with lumen
//! constructors for bulbs ([`Lumen`]). This scene picks a sky preset (full
//! daylight down to full moon, a ~400,000x range) and an optional indoor bulb,
//! and shows two things:
//!
//! - **With auto-exposure on**, every preset resolves to a sensible mid-grey, the
//!   way an eye or camera adapts: a moonlit scene and a sunlit one both read.
//! - **With auto-exposure off**, a fixed daylight camera exposes full daylight
//!   correctly and leaves the dimmer presets dark - proof the units are on one
//!   physical scale rather than arbitrary multipliers.
//!
//! The takeaway: author the light the world actually emits, and let exposure map
//! it to the display.

use crate::App;
use crate::geometry::make_box_with_uvs;
use eframe::egui;
use viewport_lib::{
    AutoExposure, ExposureMode, ExposureSettings, LightSource, LightingSettings, Lumen, Lux,
    Material, ViewportRenderer, scene::Scene,
};

const COLUMNS: usize = 5;
const COL_SPACING: f32 = 2.4;

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
            // Warm candle, warm-white tungsten, neutral LED.
            Self::Candle => [1.0, 0.55, 0.25],
            Self::Incandescent60W => [1.0, 0.82, 0.62],
            Self::Led10W => [1.0, 0.98, 0.95],
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct PhotometricState {
    pub built: bool,
    pub scene: Scene,
    pub sky: SkyPreset,
    pub bulb_on: bool,
    pub bulb: BulbPreset,
    /// Auto-exposure adapts to every preset; off uses a fixed daylight camera.
    pub auto_exposure: bool,
    /// Frame time captured from egui, fed to auto-exposure smoothing.
    pub frame_dt: f32,
}

impl Default for PhotometricState {
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

impl PhotometricState {
    /// World position of the indoor bulb (above the middle of the row).
    fn bulb_position(&self) -> [f32; 3] {
        let span = (COLUMNS - 1) as f32 * COL_SPACING;
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
            // A bare bulb: total lumens spread over the full sphere become
            // candela, then inverse-square falloff. Range bounds reach, not
            // brightness; a small radius keeps it point-like.
            let mut bulb =
                LightSource::point_lumens(self.bulb_position(), self.bulb.flux(), 25.0, 0.15);
            bulb.colour = self.bulb.colour();
            bulb.cast_shadows = true;
            lights.push(bulb);
        }

        let mut l = LightingSettings::default();
        l.lights = lights;
        l.shadows.enabled = true;
        // A modest sky fill (~15% of the direct illuminance) so shadows stay
        // readable without washing the scene flat. Ambient is added without the
        // diffuse 1/pi, so the fraction here is smaller than it looks.
        // (Provisional ambient until IBL carries nits.)
        l.hemisphere_intensity = self.sky.illuminance().0 * 0.05;
        l.sky_colour = self.sky.colour();
        l.ground_colour = [0.28, 0.26, 0.24];
        l
    }

    pub(crate) fn settings(&self) -> ExposureSettings {
        if self.auto_exposure {
            // Full adaptation (strength 1.0) so every preset - across a
            // ~400,000x lux range - is driven to mid-grey. The library default
            // is partial (0.5) for framing stability, which deliberately does
            // NOT normalise a wide brightness range; a photometric scene that
            // must stay viewable from moonlight to noon wants full adaptation.
            let auto = AutoExposure {
                adaptation: 1.0,
                dt: self.frame_dt,
                ..AutoExposure::default()
            };
            ExposureSettings::from_mode(ExposureMode::Automatic(auto))
        } else {
            // A fixed sunny-daylight camera (roughly the "sunny 16" rule):
            // exposes full daylight correctly and leaves dimmer presets dark.
            ExposureSettings::physical(16.0, 1.0 / 125.0, 100.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 60 (Photometric Presets): a ground plane and
    /// a row of mid-grey spheres, lit by presets authored in lux and lumens.
    pub(crate) fn build_photometric_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.photometric_state.scene = Scene::new();

        let span = (COLUMNS - 1) as f32 * COL_SPACING;
        let ground_mesh = make_box_with_uvs(span + 6.0, 10.0, 0.1);
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("photometric ground mesh");
        self.photometric_state.scene.add_named(
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

        for c in 0..COLUMNS {
            let x = c as f32 * COL_SPACING;
            // A neutral 60%-grey card so brightness reads from the light, not
            // varying albedo.
            let mut m = Material::from_colour([0.6, 0.6, 0.6]);
            m.roughness = 0.55;
            m.metallic = 0.0;
            self.photometric_state.scene.add_named(
                &format!("Sphere {c}"),
                Some(sphere_id),
                glam::Mat4::from_translation(glam::Vec3::new(x, 0.8, 0.85)),
                m,
            );
        }

        self.photometric_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_photometric(app: &mut App, ui: &mut egui::Ui) {
    app.photometric_state.frame_dt = ui.ctx().input(|i| i.stable_dt).min(0.1);
    let st = &mut app.photometric_state;

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

    // Readout of the authored magnitudes so the units are legible.
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
