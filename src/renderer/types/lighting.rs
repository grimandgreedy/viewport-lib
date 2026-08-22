// ---------------------------------------------------------------------------
// Lighting configuration types
// ---------------------------------------------------------------------------

use std::f32::consts::PI;

/// Illuminance in lux (lumens per square metre): the photometric unit for
/// [`LightKind::Directional`] lights, which deliver the same illuminance to
/// every lit surface regardless of distance.
///
/// Reference values are available as associated constants ([`Lux::FULL_DAYLIGHT`]
/// and friends). Direct midday sun is around `100_000`; a bright overcast sky is
/// around `1_000`; a well-lit office is a few hundred.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Lux(pub f32);

/// Luminous intensity in candela (lumens per steradian): the photometric unit
/// for [`LightKind::Point`] and [`LightKind::Spot`] lights. Unlike lux this is
/// flux per solid angle, so it becomes an illuminance at a surface only after
/// the inverse-square falloff. Build one from a bulb's total output with
/// [`Lumen::to_point_candela`] or [`Lumen::to_spot_candela`].
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Candela(pub f32);

/// Luminous flux in lumens: the total light a source emits in all directions,
/// as printed on a bulb's packaging. Convert to [`Candela`] on construction with
/// [`Lumen::to_point_candela`] (isotropic, over the full sphere) or
/// [`Lumen::to_spot_candela`] (concentrated into a cone).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Lumen(pub f32);

impl Lux {
    /// Direct midday sun on a clear day.
    pub const FULL_DAYLIGHT: Lux = Lux(100_000.0);
    /// Overcast daytime sky.
    pub const OVERCAST: Lux = Lux(1_000.0);
    /// Sun near the horizon at sunrise or sunset.
    pub const SUNRISE_SUNSET: Lux = Lux(400.0);
    /// A typical brightly-lit office or classroom.
    pub const OFFICE: Lux = Lux(400.0);
    /// Comfortable domestic living-room lighting.
    pub const LIVING_ROOM: Lux = Lux(50.0);
    /// Full moon on a clear night.
    pub const FULL_MOON: Lux = Lux(0.25);
}

impl Candela {
    /// A single candle flame, viewed from the side (the historical definition).
    pub const CANDLE: Candela = Candela(1.0);
}

impl Lumen {
    /// A candle's total luminous flux.
    pub const CANDLE: Lumen = Lumen(12.0);
    /// A 40 W incandescent bulb.
    pub const INCANDESCENT_40W: Lumen = Lumen(450.0);
    /// A 60 W incandescent bulb.
    pub const INCANDESCENT_60W: Lumen = Lumen(800.0);
    /// A 100 W incandescent bulb.
    pub const INCANDESCENT_100W: Lumen = Lumen(1_600.0);
    /// A ~10 W LED bulb (60 W-incandescent equivalent).
    pub const LED_10W: Lumen = Lumen(900.0);

    /// Candela for an isotropic point source: the flux is spread evenly over the
    /// full sphere (`4*pi` steradians), so `cd = lm / (4*pi)`.
    pub fn to_point_candela(self) -> Candela {
        Candela(self.0 / (4.0 * PI))
    }

    /// Candela for a spotlight whose flux is confined to a cone of the given
    /// outer half-angle (radians): `cd = lm / (2*pi*(1 - cos(outer)))`. A
    /// narrower cone concentrates the same lumens into a brighter beam.
    pub fn to_spot_candela(self, outer_angle: f32) -> Candela {
        let solid_angle = 2.0 * PI * (1.0 - outer_angle.cos());
        Candela(self.0 / solid_angle.max(1e-6))
    }
}

impl From<Lux> for f32 {
    fn from(v: Lux) -> f32 {
        v.0
    }
}
impl From<Candela> for f32 {
    fn from(v: Candela) -> f32 {
        v.0
    }
}
impl From<Lumen> for f32 {
    fn from(v: Lumen) -> f32 {
        v.0
    }
}

/// Light source type.
///
/// `Directional` emits parallel rays from a fixed direction (infinite distance).
/// `Point` emits rays from a position with distance-based falloff.
/// `Spot` emits a cone of light with inner (full-intensity) and outer (cutoff) angles.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum LightKind {
    /// Infinitely distant light with parallel rays (e.g. the sun).
    Directional {
        /// World-space surface-to-light direction: it points from the scene
        /// toward the light source, so an overhead sun in a Z-up scene has a
        /// large positive Z component. Both the shading and the shadow
        /// matrices use this convention.
        direction: [f32; 3],
    },
    /// Omnidirectional point light with physical inverse-square falloff.
    Point {
        /// World-space position of the light source.
        position: [f32; 3],
        /// Reach (world units): the falloff is windowed smoothly to zero at
        /// this distance and the light is culled beyond it. This bounds how far
        /// the light reaches; it does not scale brightness. Brightness comes
        /// from `intensity` and the inverse-square falloff.
        range: f32,
        /// Source radius (world units). The inverse-square term is clamped by
        /// this radius, so it both removes the near-field singularity and models
        /// a finite-size emitter (a larger radius is a softer, less point-like
        /// light). `0.0` is a mathematical point source (clamped internally to a
        /// tiny epsilon). It also sizes the soft-shadow penumbra.
        radius: f32,
    },
    /// Cone-shaped spotlight with physical inverse-square falloff.
    Spot {
        /// World-space position of the light source.
        position: [f32; 3],
        /// World-space direction the cone points toward.
        direction: [f32; 3],
        /// Reach (world units): see [`LightKind::Point::range`]. Bounds reach,
        /// not brightness.
        range: f32,
        /// Inner cone half-angle (radians) : full intensity within this cone.
        inner_angle: f32,
        /// Outer cone half-angle (radians) : light fades to zero at this angle.
        outer_angle: f32,
        /// Source radius (world units). See [`LightKind::Point::radius`].
        radius: f32,
    },
}

/// A single light source with colour and intensity.
#[non_exhaustive]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LightSource {
    /// The type and geometric parameters of this light.
    pub kind: LightKind,
    /// RGB light colour in linear 0..1. Default [1.0, 1.0, 1.0].
    pub colour: [f32; 3],
    /// Photometric brightness, in the unit that matches [`Self::kind`]:
    /// **lux** ([`Lux`]) for [`LightKind::Directional`] (illuminance delivered to
    /// every surface), and **candela** ([`Candela`]) for [`LightKind::Point`] and
    /// [`LightKind::Spot`] (luminous intensity, which becomes illuminance only
    /// after the inverse-square falloff). The two are different physical
    /// quantities, so the same number does not read alike across light types.
    ///
    /// Prefer the typed constructors ([`LightSource::directional_lux`],
    /// [`LightSource::point_candela`], [`LightSource::point_lumens`],
    /// [`LightSource::spot_candela`], [`LightSource::spot_lumens`]) over setting
    /// this field directly. Default: a modest key light (`~pi`) that reads as a
    /// plain lit surface at neutral exposure (EV 0), not a physical daylight
    /// magnitude. Real photometric values ([`Lux::FULL_DAYLIGHT`] and friends)
    /// are an opt-in paired with auto or physical exposure.
    pub intensity: f32,
    /// Importance hint used by the renderer when more lights are pushed
    /// than fit under the per-frame cap. Higher values are kept; lower
    /// values are dropped first. Default 1.0.
    ///
    /// The renderer ranks lights by `importance * proximity_weight` where
    /// proximity_weight scales the contribution of points and spots by
    /// distance to the active camera. Directional lights are treated as
    /// infinitely close (proximity_weight = 1).
    pub importance: f32,
    /// When true, this light contributes to shadow casting.
    ///
    /// Directional lights cast cascaded shadows into the shared atlas (only
    /// the first directional light is treated as the CSM caster). Point
    /// lights with `cast_shadows = true` acquire a slot in the cubemap
    /// shadow pool, up to `MAX_POINT_SHADOW_LIGHTS` active casters. Spot
    /// lights cast a single-face perspective shadow.
    ///
    /// Default: true. Disable per-light to skip the shadow render work.
    pub cast_shadows: bool,
}

impl Default for LightSource {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional {
                // Surface-to-light direction. Z is up in the default coordinate system.
                // ~65 deg elevation: mostly overhead, slight front-right bias.
                direction: [0.4, 0.3, 1.5],
            },
            colour: [1.0, 1.0, 1.0],
            // Faithful default: a modest key light, not a physical daylight
            // magnitude. With the energy-normalised (albedo/pi) diffuse, an
            // illuminance of ~pi reproduces the classic `albedo * intensity` look
            // at neutral exposure (EV 0), so a surface reads as its own colour
            // ("colour is data"). Physical lux/candela values are an opt-in paired
            // with auto or physical exposure.
            intensity: core::f32::consts::PI,
            importance: 1.0,
            cast_shadows: true,
        }
    }
}

impl LightSource {
    /// A directional (sun-like) light of the given surface-to-light `direction`
    /// and illuminance in [`Lux`]. Colour is white; use struct update on the
    /// result to change colour, shadows, or importance.
    pub fn directional_lux(direction: [f32; 3], illuminance: Lux) -> Self {
        Self {
            kind: LightKind::Directional { direction },
            intensity: illuminance.0,
            ..Self::default()
        }
    }

    /// A point light at `position` with luminous intensity in [`Candela`].
    /// `range` bounds reach (not brightness) and `radius` is the source size
    /// (clamps the near-field and sizes the penumbra). See [`LightKind::Point`].
    pub fn point_candela(position: [f32; 3], intensity: Candela, range: f32, radius: f32) -> Self {
        Self {
            kind: LightKind::Point {
                position,
                range,
                radius,
            },
            intensity: intensity.0,
            ..Self::default()
        }
    }

    /// A point light specified by its total luminous flux in [`Lumen`] (as
    /// printed on a bulb). The flux is converted to candela over the full sphere
    /// via [`Lumen::to_point_candela`].
    pub fn point_lumens(position: [f32; 3], flux: Lumen, range: f32, radius: f32) -> Self {
        Self::point_candela(position, flux.to_point_candela(), range, radius)
    }

    /// A spotlight at `position` aimed along `direction`, with luminous intensity
    /// in [`Candela`]. `inner_angle`/`outer_angle` are cone half-angles (radians);
    /// `range`/`radius` behave as on [`LightKind::Spot`].
    #[allow(clippy::too_many_arguments)]
    pub fn spot_candela(
        position: [f32; 3],
        direction: [f32; 3],
        intensity: Candela,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
        radius: f32,
    ) -> Self {
        Self {
            kind: LightKind::Spot {
                position,
                direction,
                range,
                inner_angle,
                outer_angle,
                radius,
            },
            intensity: intensity.0,
            ..Self::default()
        }
    }

    /// A spotlight specified by its total luminous flux in [`Lumen`]. The flux is
    /// converted to candela over the outer cone via [`Lumen::to_spot_candela`], so
    /// a narrower `outer_angle` yields a brighter beam for the same lumens.
    #[allow(clippy::too_many_arguments)]
    pub fn spot_lumens(
        position: [f32; 3],
        direction: [f32; 3],
        flux: Lumen,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
        radius: f32,
    ) -> Self {
        Self::spot_candela(
            position,
            direction,
            flux.to_spot_candela(outer_angle),
            range,
            inner_angle,
            outer_angle,
            radius,
        )
    }
}

/// Point-light shadow technique.
///
/// `Cube` (default) renders six cubemap faces per shadow-casting point light,
/// giving omnidirectional coverage. `Cone` is the legacy single-perspective
/// path: a 90 degree view from the light toward the scene centre. The cone
/// path is kept temporarily for sanity comparison; objects outside the cone
/// receive no shadow data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PointShadowMode {
    /// Legacy single-face perspective from the light toward the scene centre.
    Cone,
    /// Cubemap-array point shadows. Default.
    #[default]
    Cube,
}

/// Maximum number of point lights that can cast shadows simultaneously.
///
/// Bounds the size of the cubemap shadow texture array. Lights beyond this
/// count are evicted from the pool by LRU on `last_frame_used`.
pub const MAX_POINT_SHADOW_LIGHTS: u32 = 8;

/// Resolution of a single cubemap face for point-light shadows (pixels).
///
/// Total VRAM cost is `6 * MAX_POINT_SHADOW_LIGHTS * size^2 * 4` bytes for
/// the Depth32Float format. Default 1024 -> 48 MB at MAX=8.
pub const POINT_SHADOW_FACE_SIZE: u32 = 1024;

/// Shadow filtering mode.
///
/// Two axes in one enum: the penumbra model (hard edge, fixed-width PCF,
/// or variable-width PCSS with contact hardening) and the sampling budget
/// within that model. Receiver-side filtering runs per lit fragment, so
/// the tap count is a whole-scene cost multiplier on shadowed frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShadowFilter {
    /// One hardware-compare tap (2x2 bilinear). Crisp edges; the cheapest
    /// tier.
    Hard,
    /// 8-tap rotated-Poisson PCF over a fixed 1.5-texel radius. The
    /// default: visually close to [`PcfHigh`](Self::PcfHigh) at a quarter
    /// of the sampling cost.
    #[default]
    Pcf,
    /// 32-tap rotated-Poisson PCF over the same radius. This was the
    /// `Pcf` behaviour before v0.20.0; the extra taps slightly smooth the
    /// penumbra dither.
    PcfHigh,
    /// Percentage-Closer Soft Shadows: a blocker search sizes a variable
    /// penumbra per fragment (contact hardening), then a wide filter
    /// samples it. 16 blocker + 32 filter taps; the most expensive tier.
    Pcss,
    /// PCSS with halved loops (8 blocker + 16 filter taps). Same contact
    /// hardening; noisier in the widest penumbras.
    PcssFast,
}

/// Per-frame lighting configuration for the viewport.
///
/// Supports up to 8 light sources. Only `lights[0]` casts shadows.
/// Blinn-Phong shading coefficients (ambient, diffuse, specular, shininess) have
/// moved to per-object [`Material`] structs.
#[non_exhaustive]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LightingSettings {
    /// Active light sources (max 8). Default: one directional light.
    pub lights: Vec<LightSource>,
    /// Sky colour for hemisphere ambient. Default [0.8, 0.9, 1.0].
    pub sky_colour: [f32; 3],
    /// Ground colour for hemisphere ambient. Default [0.5, 0.55, 0.6].
    pub ground_colour: [f32; 3],
    /// Hemisphere (sky/ground) ambient fill, in the same linear scale as the
    /// lights' lux/candela. `0.0` disables it. The default is a modest fill
    /// (~half the default key light) so shadowed surfaces stay readable rather
    /// than pitch black; scale it with your key light. (This term is a provisional
    /// ambient approximation until image-based lighting carries absolute nits.)
    pub hemisphere_intensity: f32,
    /// Shadow-map configuration (cascades, atlas, filtering, bias, ...).
    pub shadows: ShadowSettings,
    /// Debug visualization configuration. Off by default (zero overhead when inactive).
    pub debug_vis: crate::renderer::types::debug::DebugVis,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            lights: vec![LightSource::default()],
            sky_colour: [0.8, 0.9, 1.0],
            ground_colour: [0.5, 0.55, 0.6],
            hemisphere_intensity: 1.5,
            shadows: ShadowSettings::default(),
            debug_vis: crate::renderer::types::debug::DebugVis::default(),
        }
    }
}

/// Shadow-map configuration, grouped on [`LightingSettings::shadows`].
///
/// Split out of [`LightingSettings`] so the shadow knobs travel as one unit; the
/// field names drop the `shadow_` prefix they carried when they were flat
/// (`lighting.shadow_filter` -> `lighting.shadows.filter`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShadowSettings {
    /// Whether shadow maps are computed and sampled. Default: true.
    pub enabled: bool,
    /// Constant NDC depth bias subtracted from the receiver comparison depth. Default: 0.0.
    pub bias: f32,
    /// Override the shadow frustum half-extent (world units). None = auto (20.0).
    /// Tighter values improve shadow map texel density and reduce contact-shadow penumbra.
    pub extent_override: Option<f32>,
    /// Number of cascaded shadow map splits (1-4). Default: 4.
    pub cascade_count: u32,
    /// Shadow atlas resolution (width = height). Default: 4096.
    /// Each cascade tile is `atlas_resolution / 2`.
    pub atlas_resolution: u32,
    /// Shadow filtering mode. Default: PCF.
    pub filter: ShadowFilter,
    /// PCSS light source radius in shadow-map UV space. Controls penumbra width. Default: 0.02.
    pub pcss_light_radius: f32,
    /// Point-light shadow technique. Default: cubemap.
    pub point_shadow_mode: PointShadowMode,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            bias: 0.0,
            extent_override: None,
            cascade_count: 4,
            atlas_resolution: 4096,
            filter: ShadowFilter::Pcf,
            pcss_light_radius: 0.02,
            point_shadow_mode: PointShadowMode::default(),
        }
    }
}

impl LightingSettings {
    /// A physically-scaled daylight preset: a [`Lux::FULL_DAYLIGHT`] sun with a
    /// proportional sky fill. These are real photometric magnitudes and clip to
    /// white under the neutral default exposure, so pair this with an exposure
    /// that maps daylight down - typically [`ExposureSettings::automatic`] (or a
    /// [`ExposureSettings::physical`] daylight camera). To set this preset and a
    /// matching camera together in one call, prefer
    /// [`EffectsFrame::with_posture`](crate::EffectsFrame::with_posture) with
    /// [`LightingPosture::PhysicalDaylight`](crate::LightingPosture); pair them by
    /// hand only when you need a non-default exposure:
    ///
    /// ```no_run
    /// # use viewport_lib::{EffectsFrame, LightingSettings, ExposureSettings};
    /// let mut effects = EffectsFrame::default();
    /// effects.lighting = LightingSettings::daylight();
    /// effects.display.exposure = ExposureSettings::automatic();
    /// ```
    pub fn daylight() -> Self {
        let mut sun = LightSource::default();
        sun.intensity = Lux::FULL_DAYLIGHT.0;
        Self {
            lights: vec![sun],
            // Clear-sky fill proportional to the daylight key (~8% of the sun),
            // so shadowed surfaces stay readable once exposure maps the scene down.
            hemisphere_intensity: 8_000.0,
            ..Self::default()
        }
    }
}
