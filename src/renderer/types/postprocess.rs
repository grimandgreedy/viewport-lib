/// Tone mapping operator used by the HDR pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToneMapping {
    /// Reinhard tone mapping (simple, good for scenes without extreme HDR).
    Reinhard,
    /// ACES filmic tone mapping (cinematic look, strong colour shift).
    Aces,
    /// Khronos Neutral tone mapping (perceptually uniform, minimal colour shift).
    ///
    /// Values in [0, 1] pass through with little change. Only meaningful work
    /// is done on HDR values above 1.0. This is the default.
    #[default]
    KhronosNeutral,
}

/// Neutral exposure numerator. Temporary: cancels the `1.2` maxLuminance
/// divisor so `ExposureMode::Manual { ev: 0.0 }` maps to a linear multiplier of
/// exactly `1.0`, matching the retired `PostProcessSettings.exposure = 1.0`
/// default while light intensities are still unitless (`~1`, not yet
/// lux/candela). It goes away once lights carry photometric units, when the full
/// `exposure = 1 / (1.2 * 2^EV)` form becomes correct.
pub(crate) const INTERIM_EXPOSURE_BOOST: f32 = 1.2;

/// Reflected-light meter calibration constant `K` used to convert an average
/// scene luminance to EV100 (`EV100 = log2(L * 100 / K)`). `12.5` is the
/// standard reflected-light value.
pub(crate) const METER_CALIBRATION_K: f32 = 12.5;

/// How the pre-tone-map exposure multiplier is derived each frame.
///
/// Every mode resolves to a single linear multiplier applied to scene radiance
/// before tone mapping (see `tone_map.wgsl`). The multiplier is routed through a
/// small GPU buffer so auto-exposure can write it from a compute pass in the
/// same submission as the tone map, keeping a one-shot (dirty-only) render
/// correctly exposed on its own frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExposureMode {
    /// Manual EV100. `ev = 0.0` is the neutral default (a `1.0` multiplier
    /// during the interim; see [`INTERIM_EXPOSURE_BOOST`]). Higher EV darkens.
    Manual {
        /// Exposure value at ISO 100. Higher is darker.
        ev: f32,
    },
    /// Physical-camera triangle. `EV100 = log2(N^2 / t) + log2(100 / ISO)`.
    PhysicalCamera {
        /// f-number `N` (aperture). Larger = smaller aperture = darker.
        aperture: f32,
        /// Shutter time `t` in seconds. Longer = brighter.
        shutter: f32,
        /// Sensor sensitivity (ISO). Higher = brighter.
        iso: f32,
    },
    /// Auto-exposure: meter the HDR target's log-luminance histogram each frame
    /// and adapt an internal EV toward it (see [`AutoExposure`]).
    Automatic(AutoExposure),
}

impl ExposureMode {
    /// The physically-neutral default: a fixed `EV 0` manual exposure.
    pub const NEUTRAL: ExposureMode = ExposureMode::Manual { ev: 0.0 };
}

/// Auto-exposure metering and adaptation parameters (used by
/// [`ExposureMode::Automatic`]).
///
/// Metering reduces the HDR target to a percentile-clipped average
/// log-luminance, converts it to a target EV100, and eases an internal EV
/// toward it. `dt` is the smoothing control: `dt <= 0` snaps to the target this
/// frame (the default, correct for consumers that render only when dirty);
/// `dt > 0` applies exponential smoothing (the "eye adjusting" look).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoExposure {
    /// Lower clamp on the adapted EV100. Default: `-8.0`.
    pub min_ev: f32,
    /// Upper clamp on the adapted EV100. Default: `18.0`.
    pub max_ev: f32,
    /// Adaptation rate (per second) when the scene gets brighter (EV rising).
    /// Larger adapts faster. Default: `3.0`.
    pub speed_up: f32,
    /// Adaptation rate (per second) when the scene gets darker (EV falling).
    /// Default: `1.0`.
    pub speed_down: f32,
    /// Frame time in seconds for temporal smoothing. `<= 0` snaps to the target
    /// this frame (no adaptation animation). `ViewportApp` auto-fills this from
    /// `ctx.dt`; hand-written hosts pass it (or `0.0`). Default: `0.0`.
    pub dt: f32,
    /// Lower edge of the metering window: the cumulative pixel fraction below
    /// which luminances are discarded before averaging, in `[0, 1)`. Together
    /// with `high_percent` this meters an upper-mid luminance *band* rather than
    /// the whole frame. Metering that band (not the bulk) is what keeps exposure
    /// stable as the camera orbits: the brightest lit surfaces have a nearly
    /// framing-invariant luminance, whereas the frame *average* swings as a large
    /// surface (a floor, a wall) grows or shrinks in view. Default: `0.65`.
    pub low_percent: f32,
    /// Upper edge of the metering window: the cumulative fraction above which the
    /// brightest pixels are discarded, in `(low_percent, 1]`. Rejects fireflies /
    /// specular highlights so a stray hot pixel does not drive exposure.
    /// Default: `0.95`.
    pub high_percent: f32,
    /// Center-weighting of the meter, in `[0, 1]`. `0` meters the whole frame
    /// uniformly (exposure then swings as bright content enters/leaves the frame
    /// on zoom/pan); `1` weights the centre strongly so a centred subject drives
    /// exposure and framing barely moves it. Default: `0.85`.
    pub center_weight: f32,
    /// Adaptation strength in `[0, 1]`: how fully the exposure follows the
    /// metered scene. `1` is full auto-exposure (drives the metered value to
    /// middle grey; maximally sensitive to what the camera frames). `0` holds a
    /// fixed exposure. Values below `1` are more eye-like - a dim view stays
    /// somewhat dim rather than being lifted to grey - and proportionally shrink
    /// framing-driven brightness swings (orbit, zoom onto a bright/dark object,
    /// the flare when facing a dimly-lit surface). Default: `0.5`.
    pub adaptation: f32,
}

impl Default for AutoExposure {
    fn default() -> Self {
        Self {
            min_ev: -8.0,
            max_ev: 18.0,
            speed_up: 3.0,
            speed_down: 1.0,
            dt: 0.0,
            low_percent: 0.65,
            high_percent: 0.95,
            center_weight: 0.85,
            adaptation: 0.5,
        }
    }
}

/// Physical-camera exposure configuration for the HDR pipeline.
///
/// Passed via [`EffectsFrame::exposure`](crate::EffectsFrame). Replaces the old
/// `PostProcessSettings.exposure` scalar. `compensation` is a stops-of-bias
/// applied on top of every mode (positive brightens the image).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureSettings {
    /// How the base exposure multiplier is derived. Default:
    /// [`ExposureMode::Automatic`] - lights are authored in physical photometric
    /// units (lux/candela), whose magnitudes only read correctly once exposure
    /// maps them down, so the default adapts to the scene rather than clipping to
    /// white. Set [`ExposureMode::Manual`] or [`ExposureMode::PhysicalCamera`] for
    /// a fixed exposure.
    pub mode: ExposureMode,
    /// Exposure compensation in stops, applied on top of `mode`. Positive
    /// values brighten the image (lower the effective EV). Default: `0.0`.
    pub compensation: f32,
}

impl Default for ExposureSettings {
    fn default() -> Self {
        Self {
            mode: ExposureMode::Automatic(AutoExposure::default()),
            compensation: 0.0,
        }
    }
}

impl ExposureSettings {
    /// An exposure from a mode, with no compensation.
    pub fn from_mode(mode: ExposureMode) -> Self {
        Self {
            mode,
            compensation: 0.0,
        }
    }

    /// Set the exposure compensation (stops); positive brightens.
    pub fn with_compensation(mut self, stops: f32) -> Self {
        self.compensation = stops;
        self
    }

    /// Convenience: a manual EV100 exposure with no compensation.
    pub fn manual(ev: f32) -> Self {
        Self {
            mode: ExposureMode::Manual { ev },
            compensation: 0.0,
        }
    }

    /// Convenience: a physical-camera exposure with no compensation.
    pub fn physical(aperture: f32, shutter: f32, iso: f32) -> Self {
        Self {
            mode: ExposureMode::PhysicalCamera {
                aperture,
                shutter,
                iso,
            },
            compensation: 0.0,
        }
    }

    /// Convenience: auto-exposure with default metering/adaptation parameters.
    pub fn automatic() -> Self {
        Self {
            mode: ExposureMode::Automatic(AutoExposure::default()),
            compensation: 0.0,
        }
    }

    /// The EV100 for the non-automatic modes. Returns `None` for `Automatic`,
    /// whose EV is metered on the GPU.
    pub(crate) fn base_ev100(&self) -> Option<f32> {
        match self.mode {
            ExposureMode::Manual { ev } => Some(ev),
            ExposureMode::PhysicalCamera {
                aperture,
                shutter,
                iso,
            } => Some(ev100_from_physical(aperture, shutter, iso)),
            ExposureMode::Automatic(_) => None,
        }
    }

    /// The linear exposure multiplier for the non-automatic modes, including
    /// `compensation`. Returns `None` for `Automatic`.
    pub(crate) fn manual_multiplier(&self) -> Option<f32> {
        self.base_ev100()
            .map(|ev| ev100_to_exposure(ev - self.compensation))
    }
}

/// A snapshot of a viewport's exposure state, read back from the GPU via
/// [`ViewportRenderer::exposure_state`](crate::ViewportRenderer::exposure_state).
///
/// EVs are EV100. For Manual / PhysicalCamera, `current_ev == target_ev` and
/// `adapting` is always `false`; the fields are only interesting under
/// [`ExposureMode::Automatic`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureReadback {
    /// The linear exposure multiplier currently applied before tone mapping.
    pub exposure: f32,
    /// The adapted EV100 in effect this frame.
    pub current_ev: f32,
    /// The metered target EV100 (what adaptation is easing toward).
    pub target_ev: f32,
    /// Whether adaptation is still easing toward the target (Automatic only).
    pub adapting: bool,
}

/// EV100 from a physical-camera triangle:
/// `EV100 = log2(N^2 / t) + log2(100 / ISO)`.
pub(crate) fn ev100_from_physical(aperture: f32, shutter: f32, iso: f32) -> f32 {
    (aperture * aperture / shutter).log2() + (100.0 / iso).log2()
}

/// EV100 to a linear exposure multiplier via the maxLuminance form:
/// `maxLum = 1.2 * 2^EV100`, `exposure = boost / maxLum`. `boost` is the
/// temporary neutral numerator (see [`INTERIM_EXPOSURE_BOOST`]); it is `1.0` once
/// photometric units are pinned.
pub(crate) fn ev100_to_exposure(ev100: f32) -> f32 {
    let max_lum = 1.2 * 2.0_f32.powf(ev100);
    INTERIM_EXPOSURE_BOOST / max_lum
}

/// Post-processing settings for the HDR render pipeline.
///
/// Passed via `EffectsFrame::post_process` each frame. When `enabled` is
/// `false`, the viewport renders directly to the output surface and all other
/// fields are ignored. The `render()` and `render_viewport()` entry points
/// support both paths; the `paint_to` / `paint_viewport_to` entry points are
/// always LDR regardless of this setting.
///
/// Transparent volume meshes (`SceneFrame::transparent_volume_meshes`) require
/// `enabled = true` -- they are rendered via the OIT pass which only exists in
/// the HDR pipeline.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PostProcessSettings {
    /// Enable the HDR render target and tone mapping pipeline. Default: `true`.
    pub enabled: bool,
    /// Tone mapping operator. Default: `KhronosNeutral`.
    pub tone_mapping: ToneMapping,
    /// Enable screen-space ambient occlusion.
    pub ssao: bool,
    /// Enable bloom.
    pub bloom: bool,
    /// HDR luminance threshold for bloom extraction. Default: `1.0`.
    pub bloom_threshold: f32,
    /// Bloom contribution multiplier. Default: `0.1`.
    pub bloom_intensity: f32,
    /// Firefly cap for bloom extraction: each pixel's luminance is scaled
    /// down to at most this value before thresholding, so a single very
    /// bright HDR texel (a tight specular highlight) cannot bloom into a
    /// large blob after the blur chain. Default: `4.0`. Set to `f32::MAX`
    /// to disable the cap.
    pub bloom_max_brightness: f32,
    /// Enable FXAA (Fast Approximate Anti-Aliasing) fullscreen pass.
    pub fxaa: bool,
    /// Supersampling anti-aliasing factor. 1 = off, 2 = 2x, 4 = 4x.
    ///
    /// When `> 1`, scene geometry is rendered at `ssaa_factor x resolution` and
    /// downsampled before post-processing. Produces sharper edges than FXAA at
    /// the cost of rendering `ssaa_factor^2` times more pixels. Intended for
    /// offline or screenshot use, not interactive rendering.
    pub ssaa_factor: u32,
    /// Enable screen-space contact shadows (thin shadows at object-ground contact).
    pub contact_shadows: bool,
    /// Maximum ray-march distance in view space. Default: `0.5`.
    pub contact_shadow_max_distance: f32,
    /// Number of ray-march steps. Default: `16`.
    pub contact_shadow_steps: u32,
    /// Depth thickness threshold for occlusion test. Default: `0.1`.
    pub contact_shadow_thickness: f32,
    /// Enable Eye-Dome Lighting depth enhancement.
    ///
    /// Samples a ring of 8 depth neighbors and darkens pixels at depth
    /// discontinuities, making point clouds and surface edges easier to read
    /// at any viewing distance.
    pub edl_enabled: bool,
    /// EDL sample ring radius in pixels. Default: `1.0`.
    pub edl_radius: f32,
    /// EDL darkening strength (0.0 = none, higher = stronger). Default: `1.0`.
    pub edl_strength: f32,
    /// Enable depth of field bokeh blur.
    ///
    /// Pixels whose linearized depth is outside
    /// `[dof_focal_distance - dof_focal_range, dof_focal_distance + dof_focal_range]`
    /// are blurred with a disc kernel whose radius scales up to `dof_max_blur_radius`
    /// pixels.
    pub dof_enabled: bool,
    /// View-space depth of the in-focus plane (same units as the scene). Default: `5.0`.
    pub dof_focal_distance: f32,
    /// Half-width of the sharp band around the focal plane (view-space units). Default: `1.0`.
    pub dof_focal_range: f32,
    /// Maximum blur kernel radius in pixels at maximum defocus. Default: `8.0`.
    pub dof_max_blur_radius: f32,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            tone_mapping: ToneMapping::KhronosNeutral,
            ssao: false,
            bloom: false,
            bloom_threshold: 1.0,
            bloom_intensity: 0.1,
            bloom_max_brightness: 4.0,
            fxaa: false,
            ssaa_factor: 1,
            contact_shadows: false,
            contact_shadow_max_distance: 0.5,
            contact_shadow_steps: 16,
            contact_shadow_thickness: 0.1,
            edl_enabled: false,
            edl_radius: 1.0,
            edl_strength: 1.0,
            dof_enabled: false,
            dof_focal_distance: 5.0,
            dof_focal_range: 1.0,
            dof_max_blur_radius: 8.0,
        }
    }
}

#[cfg(test)]
mod exposure_tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn physical_ev100_matches_hand_computed() {
        // Reference exposure (f/1, 1s, ISO 100) is EV 0.
        assert!(approx(ev100_from_physical(1.0, 1.0, 100.0), 0.0, 1e-4));
        // Stopping down two stops (f/2) adds 2 EV.
        assert!(approx(ev100_from_physical(2.0, 1.0, 100.0), 2.0, 1e-4));
        // Doubling ISO (200) removes one stop.
        assert!(approx(ev100_from_physical(2.0, 1.0, 200.0), 1.0, 1e-4));
        // Halving the shutter (1/2 s) adds one stop.
        assert!(approx(ev100_from_physical(1.0, 0.5, 100.0), 1.0, 1e-4));
    }

    #[test]
    fn sunny_sixteen_is_about_ev15() {
        // Sunny 16: f/16 at shutter ~= 1/ISO. At ISO 100, 1/125s ~= EV 15.
        let ev = ev100_from_physical(16.0, 1.0 / 125.0, 100.0);
        assert!(approx(ev, 15.0, 0.1), "sunny-16 EV100 was {ev}");
    }

    #[test]
    fn ev0_is_unit_multiplier() {
        // The neutral boost is chosen so EV 0 lands on a 1.0 multiplier,
        // reproducing the retired `exposure = 1.0` default. It goes away once
        // lights carry photometric units.
        assert!(approx(ev100_to_exposure(0.0), 1.0, 1e-4));
    }

    #[test]
    fn exposure_halves_per_stop_and_is_monotonic() {
        let e0 = ev100_to_exposure(0.0);
        let e1 = ev100_to_exposure(1.0);
        let e2 = ev100_to_exposure(2.0);
        // Each +1 EV halves the multiplier (darker).
        assert!(approx(e1, e0 * 0.5, 1e-4));
        assert!(approx(e2, e0 * 0.25, 1e-4));
        assert!(e2 < e1 && e1 < e0);
    }

    #[test]
    fn compensation_brightens() {
        let base = ExposureSettings::manual(0.0);
        let plus = ExposureSettings {
            mode: ExposureMode::Manual { ev: 0.0 },
            compensation: 1.0,
        };
        // +1 stop of compensation doubles the exposure multiplier.
        let b = base.manual_multiplier().unwrap();
        let p = plus.manual_multiplier().unwrap();
        assert!(approx(p, b * 2.0, 1e-4), "b={b} p={p}");
    }

    #[test]
    fn automatic_has_no_manual_multiplier() {
        let auto = ExposureSettings::automatic();
        assert!(auto.manual_multiplier().is_none());
        assert!(auto.base_ev100().is_none());
        assert!(matches!(auto.mode, ExposureMode::Automatic(_)));
    }

    #[test]
    fn physical_mode_multiplier_tracks_ev() {
        // f/2, 1s, ISO 100 -> EV 2 -> exposure = ev0 * 0.25.
        let s = ExposureSettings::physical(2.0, 1.0, 100.0);
        let expected = ev100_to_exposure(2.0);
        assert!(approx(s.manual_multiplier().unwrap(), expected, 1e-4));
    }
}
