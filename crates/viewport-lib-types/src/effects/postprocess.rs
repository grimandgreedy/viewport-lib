//! Post-process configuration such as tone mapping applied at display time.

/// Tone mapping operator used by the HDR pipeline.
///
/// A named operator keeps its curve: a corrected or retuned curve is a new
/// variant. The AgX and ACES operators follow the reference implementations'
/// constants; [`tone_operators`](crate::effects::tone_operators) holds a CPU
/// version of each.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ToneMapping {
    /// Reinhard tone mapping (simple, good for scenes without extreme HDR).
    Reinhard,
    /// Narkowicz's fit of the ACES filmic curve, applied per channel in the
    /// working space: a cheap approximation with a strong colour shift in
    /// saturated highlights. See [`AcesRrt`](Self::AcesRrt) for the fuller
    /// model.
    Aces,
    /// Khronos Neutral tone mapping (perceptually uniform, minimal colour shift).
    ///
    /// Values in [0, 1] pass through with little change. Only meaningful work
    /// is done on HDR values above 1.0. This is the default.
    #[default]
    KhronosNeutral,
    /// AgX: a log encoding and sigmoid in a set of inset Rec.2020 primaries,
    /// which desaturates bright colours smoothly towards white. Takes a look
    /// from [`PostProcessSettings::agx_look`].
    Agx,
    /// An approximation of the ACES reference rendering and output
    /// transforms (RRT and ODT, 100 nit display) with the glow, red
    /// modifier and desaturation stages, evaluated in AP0 and AP1.
    ///
    /// Colours move to AP0 without adapting the white point, as the
    /// reference does, so a neutral grey comes out slightly blue in the
    /// shadows and warm in the highlights.
    AcesRrt,
    /// A configurable per-channel curve through a mid-grey point and a
    /// white point, shaped by [`PostProcessSettings::generic_tone_curve`].
    Generic,
    /// A curve supplied as a view-transform LUT: a strip LUT baked with
    /// [`bake_view_transform_lut`] and uploaded through the renderer's
    /// texture API like any texture.
    ///
    /// The composite shapes exposed radiance with [`VIEW_LUT_SHAPER`],
    /// samples the LUT and uses the result in place of a built-in operator;
    /// the background composite and display-referred grading follow as
    /// usual. Under [`WorkingSpace::Rec2020`] the LUT is looked up with
    /// Rec.2020 values and its output is converted back to Rec.709, so bake
    /// a curve that expects Rec.2020 input.
    ///
    /// A [`PostProcessSettings::grade_lut`] under [`GradingSpace::Scene`]
    /// still applies first, and this LUT replaces only the operator. For one
    /// LUT holding the whole view transform, bake the grade in through
    /// [`bake_view_transform_lut`]'s `grade` argument and leave `grade_lut`
    /// unset.
    ///
    /// An id that does not name an uploaded texture (or names an external
    /// view, whose size the renderer cannot read) applies no operator: the
    /// exposed colour reaches the display unchanged.
    ///
    /// [`bake_view_transform_lut`]: crate::effects::tone_curve::bake_view_transform_lut
    /// [`VIEW_LUT_SHAPER`]: crate::effects::tone_curve::VIEW_LUT_SHAPER
    Custom {
        /// The baked LUT.
        lut: crate::ids::TextureId,
    },
}

impl ToneMapping {
    /// The operator's index in the renderer's composite uniform.
    #[doc(hidden)]
    pub const fn composite_index(self) -> u32 {
        match self {
            ToneMapping::Reinhard => 0,
            ToneMapping::Aces => 1,
            ToneMapping::KhronosNeutral => 2,
            ToneMapping::Agx => 3,
            ToneMapping::AcesRrt => 4,
            ToneMapping::Generic => 5,
            ToneMapping::Custom { .. } => 6,
        }
    }
}

/// A creative look applied inside the AgX operator, after its sigmoid
/// ([`ToneMapping::Agx`]).
///
/// A look is a slope and power per channel (an ASC CDL) followed by a
/// saturation scale about Rec.709 luma.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AgxLook {
    /// The base curve with no look.
    #[default]
    None,
    /// More contrast and saturation: power 1.35, saturation 1.4.
    Punchy,
    /// A warm grade: slope (1.0, 0.9, 0.5), power 0.8, saturation 1.3.
    Golden,
}

impl AgxLook {
    /// The `(slope, power, saturation)` of the look's CDL.
    pub fn cdl(self) -> ([f32; 3], [f32; 3], f32) {
        match self {
            AgxLook::None => ([1.0; 3], [1.0; 3], 1.0),
            AgxLook::Punchy => ([1.0; 3], [1.35; 3], 1.4),
            AgxLook::Golden => ([1.0, 0.9, 0.5], [0.8; 3], 1.3),
        }
    }
}

/// The curve of [`ToneMapping::Generic`].
///
/// Each channel `x` maps to `output_scale * x^c / (x^c + input_scale)`, with
/// the two scales chosen so `mid_grey_in` maps to `mid_grey_out` and
/// `hdr_max` maps to 1. See [`scales`](Self::scales).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenericToneCurve {
    /// The exponent `c`: higher is more contrast. Default: `1.55`.
    pub contrast: f32,
    /// The input value treated as mid-grey. Default: `0.18`.
    pub mid_grey_in: f32,
    /// The display value mid-grey maps to. Default: `0.215`.
    pub mid_grey_out: f32,
    /// The input value that maps to display white. Default: `10.0`.
    pub hdr_max: f32,
}

impl Default for GenericToneCurve {
    fn default() -> Self {
        Self {
            contrast: 1.55,
            mid_grey_in: 0.18,
            mid_grey_out: 0.215,
            hdr_max: 10.0,
        }
    }
}

impl GenericToneCurve {
    /// A curve from its four parameters.
    pub fn new(contrast: f32, mid_grey_in: f32, mid_grey_out: f32, hdr_max: f32) -> Self {
        Self {
            contrast,
            mid_grey_in,
            mid_grey_out,
            hdr_max,
        }
    }

    /// The contrast exponent and the `(input_scale, output_scale)` pair, after
    /// clamping the parameters to their valid ranges: contrast at least
    /// `1e-5`, both mid-grey values in `[1e-5, 1]`, `hdr_max` at least 1.
    ///
    /// With `a = mid_grey_in^c`, `b = hdr_max^c` and
    /// `k = a - mid_grey_out * b`, `input_scale = a * b * (mid_grey_out - 1) / k`
    /// and `output_scale = mid_grey_out * (a - b) / k`.
    pub fn scales(&self) -> (f32, f32, f32) {
        let c = self.contrast.max(1e-5);
        let mid_in = self.mid_grey_in.clamp(1e-5, 1.0);
        let mid_out = self.mid_grey_out.clamp(1e-5, 1.0);
        let hdr_max = self.hdr_max.max(1.0);
        let a = mid_in.powf(c);
        let b = hdr_max.powf(c);
        let k = a - mid_out * b;
        (c, a * b * (mid_out - 1.0) / k, mid_out * (a - b) / k)
    }
}

/// The primaries the scene-referred grade and the tone-map operator work in.
///
/// Exposed radiance arrives in Rec.709 (sRGB) primaries and the display
/// output is Rec.709. [`ToneMapping::Agx`] and [`ToneMapping::AcesRrt`]
/// always convert to their own internal space and back, clamping negative
/// channels to 0 on the way back, whatever the working space.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WorkingSpace {
    /// Grade and tone map in Rec.709 primaries. The default.
    #[default]
    Rec709,
    /// Convert exposed radiance to Rec.2020 before the scene-referred grade,
    /// evaluate the grade's log shaper and the operator there, and convert
    /// the result back to Rec.709 (negative channels clamped to 0) before
    /// the background composite. A per-channel operator such as
    /// [`ToneMapping::Reinhard`] then compresses saturated colours
    /// differently, and a scene-referred LUT reads Rec.2020 values.
    Rec2020,
}

/// The EV100 whose exposure multiplier is exactly `1.0`: `-log2(1.2)`.
///
/// [`ExposureMode::Passthrough`] exposes at this EV, and auto-exposure scales
/// partial adaptation from it.
pub const PASSTHROUGH_EV100: f32 = -0.263_034_4;

/// Reflected-light meter calibration constant `K` used to convert an average
/// scene luminance to EV100 (`EV100 = log2(L * 100 / K)`). `12.5` is the
/// standard reflected-light value.
#[doc(hidden)]
pub const METER_CALIBRATION_K: f32 = 12.5;

/// How the camera's exposure multiplier is derived each frame.
///
/// Every mode resolves to a single linear multiplier. The shaders apply it
/// where they read radiance (lights, the environment, the hemisphere fill,
/// probes, lightmaps, emissive), not in the composite, so shading, plugin
/// hooks and post effects see exposed radiance and colours drawn without
/// lighting are left alone; see
/// [`light_transport`](crate::effects::light_transport).
///
/// The fixed modes resolve on the CPU and apply on the frame they are set.
/// [`ExposureMode::Automatic`] meters the HDR target on the GPU, and a frame
/// shades with the exposure the frame before it metered, so a viewport that
/// switches to it, or renders only when dirty, needs a second frame to show
/// the adapted exposure.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExposureMode {
    /// No exposure scaling: scene radiance in nits goes to the tone map as-is,
    /// so a white surface reflecting 1 nit displays as white. The default, for
    /// lighting authored at the faithful scale. Equivalent to `Manual` at
    /// [`PASSTHROUGH_EV100`].
    Passthrough,
    /// Manual EV100, mapped to a multiplier as `1 / (1.2 * 2^EV100)` (the
    /// saturation-based form HDRP and Filament use). Higher EV darkens.
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
    /// and adapt an internal EV toward it (see [`AutoExposure`]). The next frame
    /// shades with the adapted EV. Needs [`PipelineMode::Hdr`]: on
    /// [`PipelineMode::Direct`] there is no target to meter, so the frame draws
    /// at the passthrough exposure and the renderer reports a notice.
    Automatic(AutoExposure),
}

impl ExposureMode {
    /// The neutral default: [`ExposureMode::Passthrough`].
    pub const NEUTRAL: ExposureMode = ExposureMode::Passthrough;
}

/// Auto-exposure metering and adaptation parameters (used by
/// [`ExposureMode::Automatic`]).
///
/// Metering reduces the HDR target to a percentile-clipped average
/// log-luminance, converts it to a target EV100, and eases an internal EV
/// toward it. `dt` is the smoothing control: `dt <= 0` snaps to the target this
/// frame (the default), so the next frame shades at the metered exposure;
/// `dt > 0` applies exponential smoothing (the "eye adjusting" look).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    /// this frame (no adaptation animation). `ViewportApp` auto-fills this
    /// from `ctx.dt`; hand-written hosts pass it (or `0.0`). Default: `0.0`.
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
    /// metered scene. `1` (the default) is full auto-exposure, driving the metered
    /// value to middle grey so the scene is correctly exposed regardless of its
    /// magnitude. `0` holds the passthrough exposure. The correction is scaled
    /// from that passthrough anchor ([`PASSTHROUGH_EV100`]), so values below `1`
    /// leave scenes far from it permanently under-corrected: a real-daylight
    /// scene at `0.5` stays several stops overexposed. Lower it only for near-neutral lighting where you want a
    /// touch of brightness constancy and smaller framing-driven swings (orbit,
    /// zoom onto a bright/dark object). Default: `1.0`.
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
            adaptation: 1.0,
        }
    }
}

/// Physical-camera exposure configuration.
///
/// Passed via `DisplaySettings::exposure`. `compensation` is a stops-of-bias
/// applied on top of every mode (positive brightens the image). The resolved
/// multiplier is applied where the shaders read radiance, on both pipelines;
/// which values it scales and which it leaves alone is set out in
/// [`light_transport`](crate::effects::light_transport).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExposureSettings {
    /// How the base exposure multiplier is derived. Default:
    /// [`ExposureMode::NEUTRAL`] ([`ExposureMode::Passthrough`]) - no exposure
    /// scaling, so the faithful default lighting reads as authored ("colour is
    /// data"). Switch to
    /// [`ExposureMode::Automatic`] (e.g. via [`ExposureSettings::automatic`]) or
    /// [`ExposureMode::PhysicalCamera`] when authoring real photometric magnitudes
    /// that need mapping down.
    pub mode: ExposureMode,
    /// Exposure compensation in stops, applied on top of `mode`. Positive
    /// values brighten the image (lower the effective EV). Default: `0.0`.
    pub compensation: f32,
}

impl Default for ExposureSettings {
    fn default() -> Self {
        Self {
            mode: ExposureMode::NEUTRAL,
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

    /// Convenience: the passthrough exposure with no compensation.
    pub fn passthrough() -> Self {
        Self {
            mode: ExposureMode::Passthrough,
            compensation: 0.0,
        }
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
    #[doc(hidden)]
    pub fn base_ev100(&self) -> Option<f32> {
        match self.mode {
            ExposureMode::Passthrough => Some(PASSTHROUGH_EV100),
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
    #[doc(hidden)]
    pub fn manual_multiplier(&self) -> Option<f32> {
        self.base_ev100()
            .map(|ev| ev100_to_exposure(ev - self.compensation))
    }
}

/// EV100 from a physical-camera triangle:
/// `EV100 = log2(N^2 / t) + log2(100 / ISO)`.
pub(crate) fn ev100_from_physical(aperture: f32, shutter: f32, iso: f32) -> f32 {
    (aperture * aperture / shutter).log2() + (100.0 / iso).log2()
}

/// EV100 to a linear exposure multiplier via the saturation-based maxLuminance
/// form: `maxLum = 1.2 * 2^EV100`, `exposure = 1 / maxLum`.
pub(crate) fn ev100_to_exposure(ev100: f32) -> f32 {
    1.0 / (1.2 * 2.0_f32.powf(ev100))
}

/// Which render pipeline turns scene geometry into display pixels.
///
/// `Hdr` is the main path; `Direct` is a constrained fallback for
/// host-owned render passes (`paint_to` / `paint_viewport`, which always use it)
/// and cheap inline rendering. `Direct` omits the HDR target and therefore
/// automatic exposure, tone mapping, every post effect, OIT (so transparency is
/// order-dependent), the skybox, and item-type plugins that have not opted
/// into LDR painting via `ItemTypePlugin::draws_ldr`. A fixed exposure
/// (`Manual`, `PhysicalCamera`) applies on `Direct` as on `Hdr`, with lit
/// radiance clamped to 1.0. Transparent volume
/// meshes (`SceneFrame::transparent_volume_meshes`) require `Hdr`.
// NOT #[non_exhaustive]: a closed, consumer-selected binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PipelineMode {
    /// Full HDR pipeline: HDR render target, exposure, tone mapping, all post
    /// effects, OIT, skybox, and item-type plugins. The default.
    #[default]
    Hdr,
    /// Constrained LDR passthrough: draw directly to the output surface with no
    /// post chain. Everything the HDR pipeline adds is skipped.
    Direct,
}

/// The display transform: how linear HDR radiance becomes display pixels.
///
/// Passed via `EffectsFrame::display`. Groups the three
/// controls of the HDR-to-display stage - the [`PipelineMode`] switch, the
/// [`ExposureSettings`] the shaders apply where they read radiance, and the
/// [`ToneMapping`] operator. The operator is inert on [`PipelineMode::Direct`],
/// and so is an automatic exposure.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DisplaySettings {
    /// Which render pipeline runs. Default: [`PipelineMode::Hdr`].
    pub mode: PipelineMode,
    /// The camera's exposure, applied to every radiance source where the shaders
    /// read it. Default: [`ExposureMode::Passthrough`].
    pub exposure: ExposureSettings,
    /// Tone-map operator mapping exposed HDR radiance into display range.
    /// Default: [`ToneMapping::KhronosNeutral`].
    pub operator: ToneMapping,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            mode: PipelineMode::Hdr,
            exposure: ExposureSettings::default(),
            operator: ToneMapping::default(),
        }
    }
}

impl DisplaySettings {
    /// True when the full HDR pipeline is active (i.e. not [`PipelineMode::Direct`]).
    pub fn is_hdr(&self) -> bool {
        matches!(self.mode, PipelineMode::Hdr)
    }
}

/// Post-processing effects for the HDR render pipeline.
///
/// Passed via `EffectsFrame::post_process` each frame. Every effect here requires
/// the HDR pipeline ([`PipelineMode::Hdr`], set on
/// `EffectsFrame::display`); on [`PipelineMode::Direct`]
/// the viewport renders directly to the output surface and these are all ignored.
#[non_exhaustive]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostProcessSettings {
    /// Enable screen-space ambient occlusion.
    pub ssao: bool,
    /// Bloom (glow) settings.
    pub bloom: BloomSettings,
    /// Enable FXAA (Fast Approximate Anti-Aliasing) fullscreen pass.
    pub fxaa: bool,
    /// Supersampling anti-aliasing factor. 1 = off, 2 = 2x, 4 = 4x.
    ///
    /// When `> 1`, scene geometry is rendered at `ssaa_factor x resolution` and
    /// downsampled before post-processing. Produces sharper edges than FXAA at
    /// the cost of rendering `ssaa_factor^2` times more pixels. Intended for
    /// offline or screenshot use, not interactive rendering.
    pub ssaa_factor: u32,
    /// Screen-space contact shadow settings.
    pub contact_shadows: ContactShadowSettings,
    /// Eye-Dome Lighting (depth-discontinuity darkening) settings.
    pub edl: EdlSettings,
    /// Depth-of-field bokeh settings.
    pub dof: DofSettings,
    /// Vignette (corner darkening) settings.
    pub vignette: VignetteSettings,
    /// Colour-grading lookup texture. `None` = no grading. Where it applies
    /// and what its axes mean is set by [`grading_space`](Self::grading_space).
    ///
    /// The id must reference a strip LUT uploaded through the renderer's
    /// texture API: `n` slices of `n` x `n` laid out horizontally (width
    /// `n * n`, height `n`), where red indexes across a slice, green down it,
    /// and blue selects the slice. Upload it in a non-sRGB format so the
    /// values pass through unmodified; texture ids registered as external
    /// views are ignored (the renderer cannot read their dimensions).
    pub grade_lut: Option<crate::ids::TextureId>,
    /// Where [`grade_lut`](Self::grade_lut) applies. Default:
    /// [`GradingSpace::Scene`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub grading_space: GradingSpace,
    /// The look [`ToneMapping::Agx`] applies. Default: [`AgxLook::None`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub agx_look: AgxLook,
    /// The curve [`ToneMapping::Generic`] applies. Default:
    /// [`GenericToneCurve::default`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub generic_tone_curve: GenericToneCurve,
    /// The primaries the scene-referred grade and the operator work in.
    /// Default: [`WorkingSpace::Rec709`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub working_space: WorkingSpace,
}

/// Where the colour-grading LUT ([`PostProcessSettings::grade_lut`]) applies
/// in the view transform.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GradingSpace {
    /// Grade exposed radiance before the tone-map operator, through a fixed
    /// log shaper. Each channel `c` (nits after exposure) is shaped as
    ///
    /// ```text
    /// s = clamp((log2(max(c, 2^-14)) + 14) / 24, 0, 1)
    /// ```
    ///
    /// which maps `[2^-14, 2^10]` to `[0, 1]`. The LUT is looked up at `s`
    /// and its output, in the same shaped space, is mapped back to radiance
    /// with `2^(s * 24 - 14)` before the operator. A LUT whose texels hold
    /// their own coordinates is an identity. Values outside the range clamp
    /// to its ends, negative channels included.
    ///
    /// A LUT's output steps are steps of the shaped value, so an 8-bit LUT
    /// quantises radiance to about 1/11 of a stop; upload a scene-referred
    /// LUT as a float texture ([`TextureData::hdr`]) to keep it smooth. The
    /// background colour is a display-space value and is not graded.
    ///
    /// [`TextureData::hdr`]: crate::data::texture::TextureData::hdr
    #[default]
    Scene,
    /// Grade the display-space result after the tone-map operator and the
    /// background composite, with the LUT's axes in `[0, 1]` display values.
    /// The background colour is graded with everything else.
    Display,
}

impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            ssao: false,
            bloom: BloomSettings::default(),
            fxaa: false,
            ssaa_factor: 1,
            contact_shadows: ContactShadowSettings::default(),
            edl: EdlSettings::default(),
            dof: DofSettings::default(),
            vignette: VignetteSettings::default(),
            grade_lut: None,
            grading_space: GradingSpace::Scene,
            agx_look: AgxLook::None,
            generic_tone_curve: GenericToneCurve::default(),
            working_space: WorkingSpace::Rec709,
        }
    }
}

/// Vignette settings, grouped on [`PostProcessSettings::vignette`].
///
/// Darkens the image toward the corners, applied at the very end of the
/// tone-map composite (after grading in either space), so it affects the
/// background too.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VignetteSettings {
    /// Enable the vignette.
    pub enabled: bool,
    /// Darkening strength at full falloff, `0.0..=1.0`. Default: `0.5`.
    pub amount: f32,
    /// Distance from the image centre where darkening starts, as a fraction
    /// of the centre-to-corner distance (0 = centre, 1 = corner).
    /// Default: `0.6`.
    pub radius: f32,
    /// Width of the falloff band past `radius`, in the same units.
    /// Default: `0.45`.
    pub softness: f32,
}

impl Default for VignetteSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            amount: 0.5,
            radius: 0.6,
            softness: 0.45,
        }
    }
}

/// Bloom (glow) settings, grouped on [`PostProcessSettings::bloom`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BloomSettings {
    /// Enable bloom. Default: `false`.
    pub enabled: bool,
    /// HDR luminance threshold for bloom extraction. Default: `1.0`.
    pub threshold: f32,
    /// Bloom contribution multiplier. Default: `0.1`.
    pub intensity: f32,
    /// Firefly cap for bloom extraction: each pixel's luminance is scaled
    /// down to at most this value before thresholding, so a single very
    /// bright HDR texel (a tight specular highlight) cannot bloom into a
    /// large blob after the blur chain. Default: `4.0`. Set to `f32::MAX`
    /// to disable the cap.
    pub max_brightness: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 1.0,
            intensity: 0.1,
            max_brightness: 4.0,
        }
    }
}

/// Screen-space contact shadow settings, grouped on
/// [`PostProcessSettings::contact_shadows`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContactShadowSettings {
    /// Enable screen-space contact shadows (thin shadows at object-ground contact).
    pub enabled: bool,
    /// Maximum ray-march distance in view space. Default: `0.5`.
    pub max_distance: f32,
    /// Number of ray-march steps. Default: `16`.
    pub steps: u32,
    /// Depth thickness threshold for occlusion test. Default: `0.1`.
    pub thickness: f32,
}

impl Default for ContactShadowSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            max_distance: 0.5,
            steps: 16,
            thickness: 0.1,
        }
    }
}

/// Eye-Dome Lighting settings, grouped on [`PostProcessSettings::edl`].
///
/// Samples a ring of 8 depth neighbours and darkens pixels at depth
/// discontinuities, making point clouds and surface edges easier to read at any
/// viewing distance.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdlSettings {
    /// Enable Eye-Dome Lighting depth enhancement.
    pub enabled: bool,
    /// EDL sample ring radius in pixels. Default: `1.0`.
    pub radius: f32,
    /// EDL darkening strength (0.0 = none, higher = stronger). Default: `1.0`.
    pub strength: f32,
}

impl Default for EdlSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 1.0,
            strength: 1.0,
        }
    }
}

/// Depth-of-field bokeh settings, grouped on [`PostProcessSettings::dof`].
///
/// Pixels whose linearised depth is outside
/// `[focal_distance - focal_range, focal_distance + focal_range]` are blurred
/// with a disc kernel whose radius scales up to `max_blur_radius` pixels.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DofSettings {
    /// Enable depth of field bokeh blur.
    pub enabled: bool,
    /// View-space depth of the in-focus plane (same units as the scene). Default: `5.0`.
    pub focal_distance: f32,
    /// Half-width of the sharp band around the focal plane (view-space units). Default: `1.0`.
    pub focal_range: f32,
    /// Maximum blur kernel radius in pixels at maximum defocus. Default: `8.0`.
    pub max_blur_radius: f32,
}

impl Default for DofSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            focal_distance: 5.0,
            focal_range: 1.0,
            max_blur_radius: 8.0,
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
    fn ev0_is_the_saturation_based_multiplier() {
        assert!(approx(ev100_to_exposure(0.0), 1.0 / 1.2, 1e-4));
    }

    #[test]
    fn passthrough_is_unit_multiplier() {
        assert!(approx(ev100_to_exposure(PASSTHROUGH_EV100), 1.0, 1e-5));
        let s = ExposureSettings::default();
        assert!(matches!(s.mode, ExposureMode::Passthrough));
        assert!(approx(s.manual_multiplier().unwrap(), 1.0, 1e-5));
        // Compensation still applies: +1 stop doubles it.
        let plus = ExposureSettings::passthrough().with_compensation(1.0);
        assert!(approx(plus.manual_multiplier().unwrap(), 2.0, 1e-4));
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
