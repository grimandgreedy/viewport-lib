//! Environment settings: which uploaded environment lights the scene, and what
//! a viewport draws behind it.

use crate::ids::EnvironmentMapId;

/// Image-based lighting from an uploaded environment, set on
/// `EffectsFrame::environment`.
///
/// The environment lights every surface through its diffuse irradiance and
/// prefiltered specular reflections, scaled by
/// `LightingSettings::environment_intensity`. By default a viewport also draws
/// it as the background (see [`EnvironmentBackground`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvironmentLighting {
    /// The environment, from the environment upload. A freed handle lights
    /// nothing.
    pub environment: EnvironmentMapId,
    /// Rotation of the environment about the world up axis (+Z), in radians.
    /// Default: 0.0.
    pub rotation: f32,
    /// Scale on the diffuse (irradiance) term. Default: 1.0.
    pub diffuse_scale: f32,
    /// Scale on the specular (reflection) term. Default: 1.0.
    pub specular_scale: f32,
}

impl EnvironmentLighting {
    /// Light the scene with `environment`, unrotated and unscaled.
    pub fn new(environment: EnvironmentMapId) -> Self {
        Self {
            environment,
            rotation: 0.0,
            diffuse_scale: 1.0,
            specular_scale: 1.0,
        }
    }

    /// Set [`rotation`](Self::rotation).
    pub fn with_rotation(mut self, radians: f32) -> Self {
        self.rotation = radians;
        self
    }
}

/// How bright an environment is drawn or lights the scene.
///
/// The stored environment holds relative radiance; this turns it into the
/// absolute luminance the renderer works in, on the same nits scale as the
/// lights and emissive surfaces.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EnvironmentIntensity {
    /// Multiply the stored radiance by this factor, so a stored `1.0` reads as
    /// that many nits.
    Multiplier(f32),
}

impl Default for EnvironmentIntensity {
    fn default() -> Self {
        Self::Multiplier(1.0)
    }
}

/// What a viewport draws behind the scene, set on
/// `ViewportFrame::environment_background`.
///
/// The default draws the lighting environment exactly as it lights the scene,
/// or the flat background colour when there is none.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnvironmentBackground {
    /// The environment drawn, or the flat colour. Default:
    /// [`BackgroundSource::LightingEnvironment`].
    pub source: BackgroundSource,
    /// Brightness of the background. `None` follows
    /// `LightingSettings::environment_intensity`. Default: `None`.
    pub intensity: Option<EnvironmentIntensity>,
    /// Rotation about +Z in radians. `None` follows the lighting rotation.
    /// Default: `None`.
    pub rotation: Option<f32>,
    /// `0.0` draws the sharp source. Above zero draws the environment's
    /// prefiltered reflection chain instead, `1.0` being the roughest level.
    /// Default: 0.0.
    pub blur: f32,
}

impl Default for EnvironmentBackground {
    fn default() -> Self {
        Self {
            source: BackgroundSource::LightingEnvironment,
            intensity: None,
            rotation: None,
            blur: 0.0,
        }
    }
}

impl EnvironmentBackground {
    /// The flat `ViewportFrame::background_colour`, with no environment drawn.
    pub fn colour() -> Self {
        Self {
            source: BackgroundSource::Colour,
            ..Self::default()
        }
    }

    /// Draw `environment`, which need not be the one lighting the scene.
    pub fn environment(environment: EnvironmentMapId) -> Self {
        Self {
            source: BackgroundSource::Environment(environment),
            ..Self::default()
        }
    }
}

/// The source of [`EnvironmentBackground`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BackgroundSource {
    /// The environment named by `EffectsFrame::environment`, if any; otherwise
    /// the flat colour.
    LightingEnvironment,
    /// The flat `ViewportFrame::background_colour`.
    Colour,
    /// A specific environment. A freed handle draws the flat colour.
    Environment(EnvironmentMapId),
}
