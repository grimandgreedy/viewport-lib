//! Lighting configuration: light sources and their photometric units.

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
        /// World-space position of the light source, in f64 so a light far
        /// from the world origin keeps its place.
        position: glam::DVec3,
        /// Reach (world units): the falloff is windowed smoothly to zero at
        /// this distance and the light is culled beyond it. This bounds how far
        /// the light reaches; it does not scale brightness. Brightness comes
        /// from `intensity` and the inverse-square falloff.
        range: f32,
        /// Source radius (world units). The inverse-square term is clamped by
        /// this radius, so it both removes the near-field singularity and models
        /// a finite-size emitter (a larger radius is a softer, less point-like
        /// light). `0.0` is a mathematical point source (clamped internally to a
        /// tiny epsilon). Under the PCSS shadow filters it also sizes the
        /// penumbra: a blocker `h` in front of a receiver `d` from the light
        /// casts one about `2 * radius * h / (d - h)` wide. Radii below 0.01
        /// shadow as 0.01.
        radius: f32,
    },
    /// Cone-shaped spotlight with physical inverse-square falloff.
    Spot {
        /// World-space position of the light source, in f64 as for
        /// [`LightKind::Point::position`].
        position: glam::DVec3,
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
    /// Light colour: pure chromaticity, with the photometric magnitude carried
    /// separately by [`intensity`](Self::intensity). Default white.
    ///
    /// Build it however the value arrives: [`Colour::rgb`](crate::colour::Colour::rgb)
    /// or [`Colour::hex`](crate::colour::Colour::hex) for a value off a colour
    /// picker or a Kelvin table, or
    /// [`Colour::linear_rgb`](crate::colour::Colour::linear_rgb) for one already
    /// in linear space.
    pub colour: crate::colour::Colour,
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
    /// plain lit surface at passthrough exposure, not a physical daylight
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
    /// The first directional light with `cast_shadows` set casts cascaded
    /// shadows into the shared atlas; other directional lights cast none.
    /// Point and spot lights take layers of the punctual shadow array, 6 for
    /// a point light and 1 for a spot, in rank order until
    /// [`ShadowSettings::punctual_shadow_layers`] is spent. A light that does
    /// not fit renders unshadowed. A spot's shadow covers cones up to 85
    /// degrees from the axis; past that the rim is unshadowed.
    ///
    /// Default: true. Disable per-light to skip the shadow render work.
    pub cast_shadows: bool,
    /// Which layers this light illuminates, as a 32-bit mask. Default `!0`
    /// (every layer). A light only contributes to an item when this mask
    /// shares a bit with the item's `ItemSettings::visibility_mask`
    /// (`channel_mask & visibility_mask != 0`), so the default lights
    /// everything. Use it to confine a light to a subset of the scene (a rig
    /// light that must not touch the environment, a highlight light for the
    /// selected layer). Honoured by the mesh-family lit paths; the AND-test
    /// runs per light in the shader.
    pub channel_mask: u32,
}

impl Default for LightSource {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional {
                // Surface-to-light direction. Z is up in the default coordinate system.
                // ~65 deg elevation: mostly overhead, slight front-right bias.
                direction: [0.4, 0.3, 1.5],
            },
            colour: crate::colour::Colour::WHITE,
            // Faithful default: a modest key light, not a physical daylight
            // magnitude. With the energy-normalised (albedo/pi) diffuse, an
            // illuminance of ~pi reproduces the classic `albedo * intensity` look
            // at passthrough exposure, so a surface reads as its own colour
            // ("colour is data"). Physical lux/candela values are an opt-in paired
            // with auto or physical exposure.
            intensity: core::f32::consts::PI,
            importance: 1.0,
            cast_shadows: true,
            channel_mask: !0,
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
    pub fn point_candela(
        position: impl crate::scene::IntoWorldPosition,
        intensity: Candela,
        range: f32,
        radius: f32,
    ) -> Self {
        Self {
            kind: LightKind::Point {
                position: position.into_world_position(),
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
    pub fn point_lumens(
        position: impl crate::scene::IntoWorldPosition,
        flux: Lumen,
        range: f32,
        radius: f32,
    ) -> Self {
        Self::point_candela(position, flux.to_point_candela(), range, radius)
    }

    /// A spotlight at `position` aimed along `direction`, with luminous intensity
    /// in [`Candela`]. `inner_angle`/`outer_angle` are cone half-angles (radians);
    /// `range`/`radius` behave as on [`LightKind::Spot`].
    #[allow(clippy::too_many_arguments)]
    pub fn spot_candela(
        position: impl crate::scene::IntoWorldPosition,
        direction: [f32; 3],
        intensity: Candela,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
        radius: f32,
    ) -> Self {
        Self {
            kind: LightKind::Spot {
                position: position.into_world_position(),
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
        position: impl crate::scene::IntoWorldPosition,
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

/// Shadow filtering mode.
///
/// Two axes in one enum: the penumbra model (hard edge, fixed-width PCF,
/// or variable-width PCSS with contact hardening) and the sampling budget
/// within that model. Receiver-side filtering runs per lit fragment, so
/// the tap count is a whole-scene cost multiplier on shadowed frames.
///
/// Point and spot light shadows follow the same tiers, with the disk laid on
/// the cube face around the light-to-receiver direction for a point light and
/// on the spot's map for a spot. The PCSS tiers size a directional penumbra by
/// [`ShadowSettings::sun_angular_radius`] and a point or spot penumbra by the
/// light's `radius`.
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
    /// 32-tap rotated-Poisson PCF over the same radius as [`Pcf`](Self::Pcf).
    /// The extra taps slightly smooth the penumbra dither.
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
/// Lights are clustered, so a frame can carry up to 512 of them on the full
/// tier (16 on the compatibility tier); past that the renderer keeps the
/// first directional light and the highest-ranked of the rest. Each light casts
/// shadows when its `cast_shadows` flag is set: the first casting directional
/// light owns the cascades, and point and spot lights take layers of the
/// punctual shadow array in rank order. Blinn-Phong shading coefficients
/// (ambient, diffuse, specular, shininess) live on per-object
/// [`Material`](crate::material::Material) structs.
#[non_exhaustive]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LightingSettings {
    /// Active light sources. Default: one directional light.
    pub lights: Vec<LightSource>,
    /// Sky colour of the hemisphere fill: the tint of light arriving from
    /// above, with the magnitude in
    /// [`hemisphere_intensity`](Self::hemisphere_intensity). Default white.
    ///
    /// This is not the skybox, which comes from the environment map. With
    /// image-based lighting off, a surface's ambient blends from
    /// `ground_colour` to this colour by the Z of its normal.
    pub sky_colour: crate::colour::Colour,
    /// Ground colour of the hemisphere fill: the tint of light arriving from
    /// below, reached by a surface whose normal points down (-Z). Default a
    /// neutral grey, a little darker than the sky.
    pub ground_colour: crate::colour::Colour,
    /// Radiance of the hemisphere fill in nits, scaling `sky_colour` and
    /// `ground_colour`. With image-based lighting off it is the whole ambient
    /// of a PBR material, so an upward-facing surface under a white sky
    /// receives this times its albedo; a Blinn-Phong material adds its own
    /// [`Material::ambient`](crate::material::Material::ambient) to it.
    /// Default 0.55. `0.0` disables the fill, leaving surfaces dark where no
    /// light reaches.
    pub hemisphere_intensity: f32,
    /// Shadow-map configuration (cascades, atlas, filtering, bias, ...).
    pub shadows: ShadowSettings,
    /// Brightness of the environment that lights the scene
    /// (`EffectsFrame::environment`), and of the background unless the viewport
    /// sets its own. Default: a multiplier of 1.0; [`daylight`](Self::daylight)
    /// sets a lux target to match its sun.
    #[cfg_attr(feature = "serde", serde(default))]
    pub environment_intensity: crate::effects::environment::EnvironmentIntensity,
    /// Brighten material ambient occlusion by the light that bounces between
    /// the occluding surfaces (Jimenez's multi-bounce fit), coloured by the
    /// surface's albedo, so AO on bright materials does not read as dirt.
    /// Applies to the environment, hemisphere, probe and lightmap ambient.
    /// Default: true.
    #[cfg_attr(feature = "serde", serde(default = "default_multi_bounce_ao"))]
    pub multi_bounce_ao: bool,
    /// Let material ambient occlusion shadow directional lights too: a cavity
    /// darkens where the light grazes it (micro-shadowing). Point and spot
    /// lights are not affected. Default: false.
    #[cfg_attr(feature = "serde", serde(default))]
    pub ao_micro_shadowing: bool,
}

#[cfg(feature = "serde")]
fn default_multi_bounce_ao() -> bool {
    true
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            lights: vec![LightSource::default()],
            sky_colour: crate::colour::Colour::WHITE,
            ground_colour: crate::colour::Colour::linear_rgb(0.6, 0.6, 0.6),
            hemisphere_intensity: 0.55,
            shadows: ShadowSettings::default(),
            environment_intensity: Default::default(),
            multi_bounce_ao: true,
            ao_micro_shadowing: false,
        }
    }
}

/// Shadow-map configuration, grouped on [`LightingSettings::shadows`] so the
/// shadow knobs travel as one unit.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShadowSettings {
    /// Whether shadow maps are computed and sampled. Default: true.
    pub enabled: bool,
    /// Constant NDC depth bias subtracted from the receiver comparison depth. Default: 0.0.
    pub bias: f32,
    /// Half-extent in world units of the single shadow map a directional
    /// light gets when the camera has no view matrix to fit cascades to.
    /// `None` sizes it to the shadow casters. Does not affect the cascades.
    pub extent_override: Option<f32>,
    /// Number of cascaded shadow map splits (1-4). Default: 4.
    pub cascade_count: u32,
    /// Shadow atlas resolution (width = height). Default: 4096.
    /// Each cascade tile is `atlas_resolution / 2`. The atlas texture is made at
    /// this size, clamped to 256..=8192 and the device's largest 2D texture and
    /// rounded down to even (a clamp is reported as a notice), and is recreated
    /// when the value changes. Its memory is `4 * atlas_resolution^2` bytes.
    pub atlas_resolution: u32,
    /// Shadow filtering mode. Default: PCF.
    pub filter: ShadowFilter,
    /// Angular radius of the directional light's disc, in radians, for the
    /// PCSS tiers. A blocker `h` world units above a receiver casts a
    /// penumbra about `2 * h * tan(sun_angular_radius)` wide, so the same
    /// pair gives the same penumbra in every cascade. The real sun is about
    /// 0.0047, which at shadow-map resolution reads close to PCF; the
    /// default is wider so the PCSS tiers look soft. Point and spot lights
    /// size their penumbra by their own `radius` instead. Default: 0.01.
    #[cfg_attr(feature = "serde", serde(default = "default_sun_angular_radius"))]
    pub sun_angular_radius: f32,
    /// Layers of the punctual shadow array shared by point and spot lights.
    /// A shadow-casting point light takes 6 layers (one per cube face) and a
    /// spot light 1. Lights are given layers in rank order until the budget
    /// is spent; a light that does not fit renders unshadowed. Default: 48,
    /// room for 8 point lights. Rounded down to a multiple of 6 and clamped
    /// to the device's `max_texture_array_layers`.
    ///
    /// The array grows in steps of 6 layers as casters need them, so the
    /// memory taken is `4 * punctual_shadow_resolution^2` bytes per layer in
    /// use, not per layer of budget.
    #[cfg_attr(feature = "serde", serde(default = "default_punctual_shadow_layers"))]
    pub punctual_shadow_layers: u32,
    /// Side of each punctual shadow layer in texels. Default: 1024 (24 MB per
    /// point light, 4 MB per spot light). Clamped to 64..=4096 and the device's
    /// largest 2D texture.
    #[cfg_attr(
        feature = "serde",
        serde(default = "default_punctual_shadow_resolution")
    )]
    pub punctual_shadow_resolution: u32,
    /// How far along the view direction directional shadows reach. `None`
    /// fits the cascades to the depth range in which the scene's casters
    /// can throw a visible shadow; `Some(d)` ends them `d` world units in
    /// front of the camera, clamped to its far plane, whatever the scene.
    /// Default: `None`.
    ///
    /// Fitted cascades spend all their resolution on that range, which
    /// sharpens shadows most when the camera views the scene from a
    /// distance. Each cascade then holds more casters, so the shadow pass
    /// costs more in that view than cascades starting at the camera would; a
    /// camera inside the scene sees no difference. A set distance or a lower
    /// [`cascade_count`](Self::cascade_count) trades the resolution back.
    #[cfg_attr(feature = "serde", serde(default))]
    pub shadow_distance: Option<f32>,
    /// Where the cascade boundaries fall inside that range. Default:
    /// [`CascadeSplit::Practical`] with `lambda` 0.75.
    #[cfg_attr(feature = "serde", serde(default))]
    pub cascade_split: CascadeSplit,
    /// The fraction of the shadow distance, at its far end, over which
    /// directional shadows fade out where the cascades stop short of the
    /// scene: at a set [`shadow_distance`](Self::shadow_distance), or when
    /// there are no mesh bounds to fit them to. Past the end shadows are
    /// off. A range fitted to the scene already reaches every shadow in
    /// view and is not faded. `0.0` leaves a hard edge where the last
    /// cascade's map runs out. Clamped to `0.0..=1.0`. Default: 0.1.
    #[cfg_attr(feature = "serde", serde(default = "default_fade_fraction"))]
    pub fade_fraction: f32,
}

/// Where the boundaries between directional shadow cascades fall, within
/// the view depth range the cascades cover.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CascadeSplit {
    /// A blend of logarithmic and even splits: `lambda` 1.0 is fully
    /// logarithmic (dense near the camera), 0.0 fully even.
    Practical {
        /// Weight of the logarithmic split, `0.0..=1.0`.
        lambda: f32,
    },
    /// Boundaries as fractions of the covered range, nearest first: cascade
    /// `i` ends at `near + (far - near) * fractions[i]` and the last ends at
    /// `far`. With fewer than four cascades only the first
    /// `cascade_count - 1` are read.
    Fractions([f32; 3]),
}

#[cfg(feature = "serde")]
fn default_fade_fraction() -> f32 {
    ShadowSettings::default().fade_fraction
}

impl Default for CascadeSplit {
    fn default() -> Self {
        Self::Practical { lambda: 0.75 }
    }
}

#[cfg(feature = "serde")]
fn default_sun_angular_radius() -> f32 {
    ShadowSettings::default().sun_angular_radius
}

#[cfg(feature = "serde")]
fn default_punctual_shadow_layers() -> u32 {
    ShadowSettings::default().punctual_shadow_layers
}

#[cfg(feature = "serde")]
fn default_punctual_shadow_resolution() -> u32 {
    ShadowSettings::default().punctual_shadow_resolution
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
            sun_angular_radius: 0.01,
            punctual_shadow_layers: 48,
            punctual_shadow_resolution: 1024,
            shadow_distance: None,
            cascade_split: CascadeSplit::default(),
            fade_fraction: 0.1,
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
    /// `EffectsFrame::with_posture` with
    /// `LightingPosture::PhysicalDaylight`; pair them by
    /// hand only when you need a non-default exposure:
    ///
    /// [`ExposureSettings::automatic`]: crate::effects::postprocess::ExposureSettings::automatic
    /// [`ExposureSettings::physical`]: crate::effects::postprocess::ExposureSettings::physical
    ///
    /// ```
    /// # use viewport_lib_types::effects::lighting::LightingSettings;
    /// # use viewport_lib_types::effects::postprocess::ExposureSettings;
    /// let lighting = LightingSettings::daylight();
    /// let exposure = ExposureSettings::automatic();
    /// ```
    pub fn daylight() -> Self {
        let mut sun = LightSource::default();
        sun.intensity = Lux::FULL_DAYLIGHT.0;
        Self {
            lights: vec![sun],
            // Clear-sky fill proportional to the daylight key (~8% of the sun),
            // so shadowed surfaces stay readable once exposure maps the scene down.
            hemisphere_intensity: 8_000.0,
            // The illuminance that fill gives an upward-facing surface, so an
            // HDRI environment sits beside the sun at a real sky's brightness.
            environment_intensity: crate::effects::environment::EnvironmentIntensity::Lux(
                8_000.0 * PI,
            ),
            ..Self::default()
        }
    }
}
