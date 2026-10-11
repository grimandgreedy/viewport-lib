//! Participating-media volume primitive: per-pixel ray-marched fog / smoke / clouds.
//!
//! A `ScatterVolume` is a box- or sphere-bounded region of participating media.
//! The renderer rasterises each visible volume through its screen-space
//! bounding rectangle, ray-marches the bound region per covered fragment,
//! accumulates absorption (Beer-Lambert) plus a lit / emissive scattered
//! colour, and composites the result over the opaque scene.

use viewport_lib_types::colour::Colour;
use viewport_lib_types::scene::aabb::Aabb;

/// A ray-marched participating-media region.
///
/// Add to a frame via [`ScatterVolumeItem`](crate::item_types::scatter_volume::ScatterVolumeItem) and
/// `frame.scene.submit::<ScatterVolumeItem>(..)`. No upload step is required;
/// the item type packs visible volumes into a storage buffer each frame.
/// Hard cap on the number of scatter volumes drawn in one frame. Volumes past
/// this many are skipped for the frame; the per-volume draw flow handles up to
/// this many active volumes.
pub const MAX_SCATTER_VOLUMES: usize = 16;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScatterVolume {
    /// Spatial bounds: axis-aligned box or sphere.
    pub shape: ScatterShape,
    /// Beer-Lambert extinction coefficient in world units.
    ///
    /// Typical range 0.05 to 1.0. A value of 0 disables the volume.
    pub density: f32,
    /// Colour source: a flat RGB or a colourmap LUT indexed by local density.
    pub colour: ColourSource,
    /// Henyey-Greenstein phase anisotropy in [-1, 1].
    ///
    /// 0.0 = isotropic (fog), positive = forward scattering (clouds, ~0.7),
    /// negative = back scattering.
    pub anisotropy: f32,
    /// Self-emission. `Emission::None` disables emissive contribution.
    pub emission: Emission,
    /// Density curve applied before colour and emission sampling.
    pub density_remap: DensityRemap,
    /// Procedural noise driver. None disables the noise modulation; the
    /// density at each march step then comes only from the base `density`
    /// times the active remap and (if set) the density texture.
    pub noise: Option<NoiseDriver>,
    /// Per-volume ray-march step count override. `None` uses the global
    /// step count from `ScatterSettings::quality`. Use a higher value for
    /// volumes that need extra detail (clouds, fire); a lower value for
    /// cheap background fog.
    pub step_budget: Option<u32>,
    /// External 3D density texture (typically baked sim output) that
    /// modulates per-step density instead of procedural noise. Uploaded
    /// via [`upload_volume`](viewport_lib::resources::DeviceResources::upload_volume).
    /// The texture is sampled at normalized coordinates inside the volume's
    /// world-space AABB. When both `noise` and `density_texture` are set
    /// the texture takes precedence (noise is ignored for that volume).
    /// Only one density texture can be bound per frame; the first volume
    /// in `SceneFrame::scatter_volumes` with a texture wins for the pass.
    pub density_texture: Option<viewport_lib_types::ids::VolumeId>,
    /// Refractive distortion of the scene behind the volume. `None` is the
    /// default and skips the refraction pass entirely. When `Some`, the
    /// renderer copies the scene colour, samples it at a UV offset derived
    /// from the local density gradient, and writes the distorted result
    /// back over the volume's screen footprint before the scatter pass
    /// runs. Used for heat haze and similar shimmer effects.
    pub refraction: Option<RefractionParams>,
}

impl Default for ScatterVolume {
    fn default() -> Self {
        Self {
            shape: ScatterShape::Box(Aabb {
                min: glam::Vec3::splat(-0.5),
                max: glam::Vec3::splat(0.5),
            }),
            density: 0.0,
            colour: ColourSource::Flat(Colour::linear_rgb(0.8, 0.85, 0.9)),
            anisotropy: 0.0,
            emission: Emission::None,
            density_remap: DensityRemap::Identity,
            noise: None,
            step_budget: None,
            density_texture: None,
            refraction: None,
        }
    }
}

impl ScatterVolume {
    /// Convenience: a uniform-density box volume with flat colour.
    pub fn box_uniform(aabb: Aabb, density: f32, colour: impl Into<Colour>) -> Self {
        Self {
            shape: ScatterShape::Box(aabb),
            density,
            colour: ColourSource::Flat(colour.into()),
            ..Default::default()
        }
    }

    /// Convenience: a uniform-density sphere volume with flat colour.
    pub fn sphere_uniform(
        center: [f32; 3],
        radius: f32,
        density: f32,
        colour: impl Into<Colour>,
    ) -> Self {
        Self {
            shape: ScatterShape::Sphere { center, radius },
            density,
            colour: ColourSource::Flat(colour.into()),
            ..Default::default()
        }
    }

    /// Conservative world-space AABB enclosing the volume.
    pub fn world_aabb(&self) -> Aabb {
        match self.shape {
            ScatterShape::Box(b) => b,
            ScatterShape::Sphere { center, radius } => {
                let c = glam::Vec3::from(center);
                let r = glam::Vec3::splat(radius);
                Aabb {
                    min: c - r,
                    max: c + r,
                }
            }
        }
    }

    /// World-space centre of the volume's shape.
    pub fn shape_centre(&self) -> [f32; 3] {
        match self.shape {
            ScatterShape::Box(b) => {
                let c = (b.min + b.max) * 0.5;
                [c.x, c.y, c.z]
            }
            ScatterShape::Sphere { center, .. } => center,
        }
    }
}

/// Spatial bounds of a [`ScatterVolume`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ScatterShape {
    /// Axis-aligned box.
    Box(Aabb),
    /// Sphere defined by world-space center and radius.
    Sphere {
        /// World-space center.
        center: [f32; 3],
        /// World-space radius.
        radius: f32,
    },
}

/// How a volume's colour is determined at each ray-march step.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ColourSource {
    /// Single RGB colour applied uniformly throughout the volume.
    Flat(viewport_lib_types::colour::Colour),
    /// Density-indexed lookup through a colourmap LUT.
    Ramp(viewport_lib_types::colourmap::ColourmapId),
}

/// Self-emission specification for a [`ScatterVolume`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Emission {
    /// No emission.
    None,
    /// Emission proportional to a function of local density.
    Strength {
        /// Multiplier on the volume's colour to produce emitted radiance.
        strength: f32,
        /// Function mapping local density (after remap) to an emission scalar.
        curve: EmissionCurve,
    },
}

/// Curve mapping local density to an emission multiplier.
#[derive(Debug, Clone, Copy)]
pub enum EmissionCurve {
    /// Linear in density.
    Linear,
    /// `density^exponent`.
    Power(f32),
    /// Hard threshold; emit only where density exceeds the cutoff.
    Threshold(f32),
}

/// Remap of the raw density value before colour and emission sampling.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum DensityRemap {
    /// Pass-through.
    Identity,
    /// `smoothstep(lo, hi, density)`.
    Smoothstep {
        /// Lower edge.
        lo: f32,
        /// Upper edge.
        hi: f32,
    },
    /// Exponential falloff from a centre point.
    ExpFalloff {
        /// World-space falloff origin.
        center: [f32; 3],
        /// Falloff coefficient in inverse world units.
        falloff: f32,
    },
}

/// Procedural noise driver: fbm value noise modulates per-step density and,
/// when `time_scale` is non-zero, evolves over time.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct NoiseDriver {
    /// Base frequency in inverse world units. Higher = finer detail.
    pub scale: f32,
    /// Number of fbm octaves (clamped to 1..=6 by the shader).
    pub octaves: u32,
    /// Animation scroll velocity (world units per second). Adds to the
    /// sample position each frame for drifting smoke / flowing clouds.
    pub scroll_velocity: [f32; 3],
    /// Per-second domain-warp rate. Non-zero values make the noise field
    /// evolve in place without drifting in any direction (good for fire
    /// flicker). 0 = static (only `scroll_velocity` animates).
    pub time_scale: f32,
    /// Fbm frequency multiplier per octave. Typical values 1.8..2.2; 2.0
    /// is the standard. Clamped to 1.1..=4.0 by the shader.
    pub lacunarity: f32,
}

impl Default for NoiseDriver {
    fn default() -> Self {
        Self {
            scale: 1.0,
            octaves: 3,
            scroll_velocity: [0.0; 3],
            time_scale: 0.0,
            lacunarity: 2.0,
        }
    }
}

/// Refractive distortion parameters for a [`ScatterVolume`].
///
/// The renderer samples the scene colour at a UV offset taken from the
/// local density gradient and writes the result over the volume's screen
/// footprint. The scatter pass then runs on top of the distorted scene,
/// so absorption and in-scattering still apply normally.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RefractionParams {
    /// Maximum screen-space displacement, in normalized UV units. Typical
    /// values are 0.005 to 0.04. Larger values look like strong heat haze.
    pub strength: f32,
    /// Density threshold below which a sample contributes no distortion.
    /// Useful for keeping wispy edges quiet while the hot core shimmers.
    pub density_threshold: f32,
    /// Frequency multiplier on the noise field that drives the gradient.
    /// Higher values make the shimmer finer.
    pub noise_scale: f32,
}

impl Default for RefractionParams {
    fn default() -> Self {
        Self {
            strength: 0.015,
            density_threshold: 0.05,
            noise_scale: 1.5,
        }
    }
}
