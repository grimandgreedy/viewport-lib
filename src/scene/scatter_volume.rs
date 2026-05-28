//! Participating-media volume primitive: per-pixel ray-marched fog / smoke / clouds.
//!
//! A `ScatterVolume` is a box- or sphere-bounded region of participating media.
//! The renderer ray-marches every visible volume per fragment, accumulating
//! absorption (Beer-Lambert) and a scattered colour, and composites the result
//! over the opaque scene.
//!
//! V1 ships shape + uniform density + flat colour. Future phases (see
//! `docs/plans/volumetric-effects-plan.md`) extend the parameter set without
//! changing the public API: V2 adds anisotropic phase functions and shadow-map
//! sampling, V3 adds colour ramps and emission, V4 adds noise drivers.

use crate::scene::aabb::Aabb;

/// A ray-marched participating-media region.
///
/// Add to a frame via [`ScatterVolumeItem`](crate::renderer::ScatterVolumeItem)
/// and push into `SceneFrame::scatter_volumes`. No upload step is required;
/// the renderer packs visible volumes into a storage buffer each frame.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScatterVolume {
    /// Spatial bounds: axis-aligned box or sphere.
    pub shape: ScatterShape,
    /// Beer-Lambert extinction coefficient in world units.
    ///
    /// Typical range 0.05 to 1.0. A value of 0 disables the volume.
    pub density: f32,
    /// Colour source. V1 reads only `ColourSource::Flat`; ramps land in V3.
    pub colour: ColourSource,
    /// Henyey-Greenstein phase anisotropy in [-1, 1].
    ///
    /// V1 has no lighting integration so this is unused. V2 honours it.
    /// 0.0 = isotropic (fog), positive = forward scattering (clouds, ~0.7),
    /// negative = back scattering.
    pub anisotropy: f32,
    /// Self-emission. V1 reads only `Emission::None`; emissive volumes land in V3.
    pub emission: Emission,
    /// Density curve applied before colour and emission sampling. V1 reads only
    /// `DensityRemap::Identity`.
    pub density_remap: DensityRemap,
    /// Procedural / external density driver. V1 ignores this field; V4 honours it.
    pub noise: Option<NoiseDriver>,
}

impl Default for ScatterVolume {
    fn default() -> Self {
        Self {
            shape: ScatterShape::Box(Aabb {
                min: glam::Vec3::splat(-0.5),
                max: glam::Vec3::splat(0.5),
            }),
            density: 0.0,
            colour: ColourSource::Flat([0.8, 0.85, 0.9]),
            anisotropy: 0.0,
            emission: Emission::None,
            density_remap: DensityRemap::Identity,
            noise: None,
        }
    }
}

impl ScatterVolume {
    /// Convenience: a uniform-density box volume with flat colour.
    pub fn box_uniform(aabb: Aabb, density: f32, colour: [f32; 3]) -> Self {
        Self {
            shape: ScatterShape::Box(aabb),
            density,
            colour: ColourSource::Flat(colour),
            ..Default::default()
        }
    }

    /// Convenience: a uniform-density sphere volume with flat colour.
    pub fn sphere_uniform(center: [f32; 3], radius: f32, density: f32, colour: [f32; 3]) -> Self {
        Self {
            shape: ScatterShape::Sphere { center, radius },
            density,
            colour: ColourSource::Flat(colour),
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
}

/// Spatial bounds of a [`ScatterVolume`].
#[derive(Debug, Clone, Copy)]
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
    Flat([f32; 3]),
    /// Density-indexed lookup through a colourmap LUT. Honoured in V3+.
    Ramp(crate::resources::ColourmapId),
}

/// Self-emission specification for a [`ScatterVolume`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Emission {
    /// No emission. V1 default.
    None,
    /// Emission proportional to a function of local density. Honoured in V3+.
    Strength {
        /// Multiplier on the volume's colour to produce emitted radiance.
        strength: f32,
        /// Function mapping local density (after remap) to an emission scalar.
        curve: EmissionCurve,
    },
}

/// Curve mapping local density to an emission multiplier. Honoured in V3+.
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
/// Honoured in V3+.
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

/// Procedural / external density driver. Stub for V4; V1 ignores this entirely.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct NoiseDriver {
    /// Base frequency in world units.
    pub scale: f32,
    /// Number of fbm octaves.
    pub octaves: u32,
    /// Animation scroll velocity (world units per second).
    pub scroll_velocity: [f32; 3],
    /// Per-second time scale on the noise domain warp.
    pub time_scale: f32,
}

impl Default for NoiseDriver {
    fn default() -> Self {
        Self {
            scale: 1.0,
            octaves: 3,
            scroll_velocity: [0.0; 3],
            time_scale: 0.0,
        }
    }
}

/// CPU representation of the GPU storage entry. Public so consumers writing
/// custom render paths can pack their own buffers; ordinary use does not need
/// to touch this.
///
/// Layout: 80 bytes, 16-byte aligned. Matches `GpuScatterVolume` in
/// `src/shaders/scatter_volume.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuScatterVolume {
    /// 0 = Box, 1 = Sphere. Future variants extend this number.
    pub shape_kind: u32,
    /// Bit flags: 1 = unlit (skip in-scattering), 2 = receive_shadows.
    pub flags: u32,
    /// Padding to bring `p0` to a 16-byte boundary.
    pub _pad0: [u32; 2],
    /// Box: `min.xyz`, `_`. Sphere: `center.xyz, radius`.
    pub p0: [f32; 4],
    /// Box: `max.xyz, _`. Sphere: unused.
    pub p1: [f32; 4],
    /// RGB scattered colour and density (a = density).
    pub colour_density: [f32; 4],
    /// V2 scalar parameters: `x` = Henyey-Greenstein anisotropy g, others reserved.
    pub params: [f32; 4],
}

/// Flag bit: skip in-scattering (treat the volume as `unlit`).
pub const SCATTER_FLAG_UNLIT: u32 = 1;
/// Flag bit: sample the shadow map at each march step.
pub const SCATTER_FLAG_RECEIVE_SHADOWS: u32 = 2;

impl GpuScatterVolume {
    /// Pack a CPU `ScatterVolume` into the GPU layout. `density_multiplier`
    /// folds `ItemSettings::opacity` into the effective density. `flags` is
    /// the per-volume settings bitfield (see `SCATTER_FLAG_*` constants).
    /// Returns `None` if the resulting density is non-positive.
    pub fn pack(volume: &ScatterVolume, density_multiplier: f32, flags: u32) -> Option<Self> {
        let density = volume.density * density_multiplier;
        if !(density > 0.0) {
            return None;
        }
        let colour = match volume.colour {
            ColourSource::Flat(rgb) => rgb,
            // V3 will resolve a ramp at sample time. Until then a Ramp volume
            // renders as mid-grey so it is visible but obviously placeholder.
            ColourSource::Ramp(_) => [0.5, 0.5, 0.5],
        };
        let (shape_kind, p0, p1) = match volume.shape {
            ScatterShape::Box(b) => (
                0u32,
                [b.min.x, b.min.y, b.min.z, 0.0],
                [b.max.x, b.max.y, b.max.z, 0.0],
            ),
            ScatterShape::Sphere { center, radius } => (
                1u32,
                [center[0], center[1], center[2], radius],
                [0.0; 4],
            ),
        };
        let anisotropy = volume.anisotropy.clamp(-0.95, 0.95);
        Some(Self {
            shape_kind,
            flags,
            _pad0: [0; 2],
            p0,
            p1,
            colour_density: [colour[0], colour[1], colour[2], density],
            params: [anisotropy, 0.0, 0.0, 0.0],
        })
    }
}

/// CPU ray-vs-shape intersection used by picking and verified by tests.
///
/// Returns `Some((t_enter, t_exit))` in ray parameter units. Both are clamped
/// to be non-negative (camera-inside case sets `t_enter = 0`). Returns `None`
/// when the ray misses the shape or exits before entering.
pub fn ray_intersect(
    shape: &ScatterShape,
    origin: glam::Vec3,
    dir: glam::Vec3,
) -> Option<(f32, f32)> {
    match shape {
        ScatterShape::Box(b) => ray_box(b, origin, dir),
        ScatterShape::Sphere { center, radius } => {
            ray_sphere(glam::Vec3::from(*center), *radius, origin, dir)
        }
    }
}

fn ray_box(b: &Aabb, o: glam::Vec3, d: glam::Vec3) -> Option<(f32, f32)> {
    let inv = glam::Vec3::new(
        if d.x.abs() > 1e-8 { 1.0 / d.x } else { f32::INFINITY },
        if d.y.abs() > 1e-8 { 1.0 / d.y } else { f32::INFINITY },
        if d.z.abs() > 1e-8 { 1.0 / d.z } else { f32::INFINITY },
    );
    let t0 = (b.min - o) * inv;
    let t1 = (b.max - o) * inv;
    let t_min = t0.min(t1);
    let t_max = t0.max(t1);
    let t_enter = t_min.x.max(t_min.y).max(t_min.z).max(0.0);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);
    if t_enter >= t_exit || t_exit <= 0.0 {
        None
    } else {
        Some((t_enter, t_exit))
    }
}

fn ray_sphere(c: glam::Vec3, r: f32, o: glam::Vec3, d: glam::Vec3) -> Option<(f32, f32)> {
    let oc = o - c;
    let a = d.dot(d);
    let b = 2.0 * oc.dot(d);
    let cc = oc.dot(oc) - r * r;
    let disc = b * b - 4.0 * a * cc;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t0 = (-b - sq) / (2.0 * a);
    let t1 = (-b + sq) / (2.0 * a);
    let t_enter = t0.max(0.0);
    let t_exit = t1;
    if t_enter >= t_exit || t_exit <= 0.0 {
        None
    } else {
        Some((t_enter, t_exit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_volume_has_zero_density() {
        let v = ScatterVolume::default();
        assert_eq!(v.density, 0.0);
        assert!(matches!(v.colour, ColourSource::Flat(_)));
        assert!(matches!(v.emission, Emission::None));
        assert!(matches!(v.density_remap, DensityRemap::Identity));
        assert!(v.noise.is_none());
    }

    #[test]
    fn pack_zero_density_returns_none() {
        let v = ScatterVolume::default();
        assert!(GpuScatterVolume::pack(&v, 1.0, 0).is_none());
    }

    #[test]
    fn pack_box_round_trips() {
        let v = ScatterVolume::box_uniform(
            Aabb {
                min: glam::Vec3::new(-1.0, -2.0, -3.0),
                max: glam::Vec3::new(4.0, 5.0, 6.0),
            },
            0.2,
            [0.1, 0.2, 0.3],
        );
        let g = GpuScatterVolume::pack(&v, 1.0, 0).unwrap();
        assert_eq!(g.shape_kind, 0);
        assert_eq!(&g.p0[..3], &[-1.0, -2.0, -3.0]);
        assert_eq!(&g.p1[..3], &[4.0, 5.0, 6.0]);
        assert_eq!(g.colour_density, [0.1, 0.2, 0.3, 0.2]);
    }

    #[test]
    fn pack_sphere_round_trips() {
        let v = ScatterVolume::sphere_uniform([1.0, 2.0, 3.0], 4.0, 0.5, [0.4, 0.5, 0.6]);
        let g = GpuScatterVolume::pack(&v, 1.0, 0).unwrap();
        assert_eq!(g.shape_kind, 1);
        assert_eq!(g.p0, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(g.colour_density, [0.4, 0.5, 0.6, 0.5]);
    }

    #[test]
    fn opacity_multiplier_scales_density() {
        let v = ScatterVolume::box_uniform(
            Aabb {
                min: glam::Vec3::ZERO,
                max: glam::Vec3::ONE,
            },
            0.4,
            [1.0; 3],
        );
        let g = GpuScatterVolume::pack(&v, 0.5, 0).unwrap();
        assert!((g.colour_density[3] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn ray_box_hits_from_outside() {
        let b = Aabb {
            min: glam::Vec3::new(-1.0, -1.0, -1.0),
            max: glam::Vec3::new(1.0, 1.0, 1.0),
        };
        let hit = ray_intersect(
            &ScatterShape::Box(b),
            glam::Vec3::new(0.0, 0.0, -5.0),
            glam::Vec3::Z,
        );
        let (enter, exit) = hit.unwrap();
        assert!((enter - 4.0).abs() < 1e-4);
        assert!((exit - 6.0).abs() < 1e-4);
    }

    #[test]
    fn ray_box_camera_inside_starts_at_zero() {
        let b = Aabb {
            min: glam::Vec3::new(-1.0, -1.0, -1.0),
            max: glam::Vec3::new(1.0, 1.0, 1.0),
        };
        let hit = ray_intersect(&ScatterShape::Box(b), glam::Vec3::ZERO, glam::Vec3::Z);
        let (enter, exit) = hit.unwrap();
        assert_eq!(enter, 0.0);
        assert!((exit - 1.0).abs() < 1e-4);
    }

    #[test]
    fn ray_sphere_misses() {
        let hit = ray_intersect(
            &ScatterShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            glam::Vec3::new(2.0, 0.0, -5.0),
            glam::Vec3::Z,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn ray_sphere_camera_inside_starts_at_zero() {
        let hit = ray_intersect(
            &ScatterShape::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            },
            glam::Vec3::ZERO,
            glam::Vec3::Z,
        );
        let (enter, exit) = hit.unwrap();
        assert_eq!(enter, 0.0);
        assert!((exit - 1.0).abs() < 1e-4);
    }

    #[test]
    fn world_aabb_sphere_matches_bounds() {
        let v = ScatterVolume::sphere_uniform([0.0, 0.0, 0.0], 2.0, 0.1, [1.0; 3]);
        let b = v.world_aabb();
        assert_eq!(b.min, glam::Vec3::splat(-2.0));
        assert_eq!(b.max, glam::Vec3::splat(2.0));
    }
}
