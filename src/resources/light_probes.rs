//! Spherical-harmonic light probes.
//!
//! A light probe stores the low-frequency incoming radiance at a point as
//! order-2 real spherical harmonics (9 coefficients per RGB channel). Dynamic
//! objects sample the probe field to pick up per-position indirect light instead
//! of the single global environment, so an object indoors is lit by the room
//! rather than by the sky.
//!
//! This module is pure CPU: the projection that turns a captured panorama into
//! SH ([`project_equirect_to_sh`]), the blend that interpolates the probes
//! around a point ([`LightProbeSet::blend_sh_at`]), and the evaluation used to
//! test them ([`evaluate_sh`]). The blended per-object result is carried to the
//! shader in the per-object uniform, so the probe set itself never reaches the
//! GPU.
//!
//! The interpolation in [`LightProbeSet::blend_sh_at`] is capped-radius
//! k-nearest weighting. Barycentric interpolation over a Delaunay
//! tetrahedralisation is a planned quality upgrade (phase `LP-tet`); it swaps the
//! body of that one method and changes nothing else here.

/// Order-2 real spherical harmonics: 9 coefficients per RGB channel.
///
/// These are radiance coefficients (the raw projection of incoming light).
/// [`evaluate_sh`] applies the cosine-lobe convolution that turns them into the
/// diffuse irradiance a surface receives.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SHCoefficients {
    /// Red-channel coefficients, band order 0, 1(-1,0,1), 2(-2,-1,0,1,2).
    pub r: [f32; 9],
    /// Green-channel coefficients.
    pub g: [f32; 9],
    /// Blue-channel coefficients.
    pub b: [f32; 9],
}

impl Default for SHCoefficients {
    fn default() -> Self {
        Self {
            r: [0.0; 9],
            g: [0.0; 9],
            b: [0.0; 9],
        }
    }
}

/// A single light probe: a world position and the SH radiance sampled there.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LightProbe {
    /// World-space position the probe was sampled at.
    pub position: [f32; 3],
    /// SH radiance at that position.
    pub sh: SHCoefficients,
}

/// A set of light probes with the interpolation used to sample them at an
/// arbitrary point.
#[derive(Clone, Debug, Default)]
pub struct LightProbeSet {
    probes: Vec<LightProbe>,
}

impl LightProbeSet {
    /// Build a probe set from probes. Order is irrelevant.
    pub fn new(probes: Vec<LightProbe>) -> Self {
        Self { probes }
    }

    /// The probes, in insertion order.
    pub fn probes(&self) -> &[LightProbe] {
        &self.probes
    }

    /// Whether the set has no probes (blends return zero SH).
    pub fn is_empty(&self) -> bool {
        self.probes.is_empty()
    }

    /// Interpolate the probe SH at a world position.
    ///
    /// Capped-radius k-nearest inverse-distance weighting: the nearest
    /// [`K_NEAREST`] probes are weighted by `1 / (distance^2 + eps)` and blended.
    /// A point sitting on a probe returns that probe's SH. An empty set returns
    /// zero SH. This is the swap point for the barycentric-tet upgrade (LP-tet).
    pub fn blend_sh_at(&self, position: [f32; 3]) -> SHCoefficients {
        if self.probes.is_empty() {
            return SHCoefficients::default();
        }

        // Find the K nearest probes by squared distance. Probe counts are small
        // (hundreds), so a partial-sort scan is cheaper than a spatial index.
        let mut nearest: Vec<(f32, usize)> = self
            .probes
            .iter()
            .enumerate()
            .map(|(i, p)| (dist_sq(p.position, position), i))
            .collect();
        let k = K_NEAREST.min(nearest.len());
        nearest.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        nearest.truncate(k);

        // Inverse-distance weights. eps keeps a probe exactly at `position` from
        // producing an infinite weight while still dominating the blend.
        const EPS: f32 = 1e-6;
        let mut total = 0.0f32;
        let mut weights: Vec<(f32, usize)> = Vec::with_capacity(k);
        for &(d2, i) in &nearest {
            let w = 1.0 / (d2 + EPS);
            total += w;
            weights.push((w, i));
        }
        let inv_total = 1.0 / total;

        let mut out = SHCoefficients::default();
        for (w, i) in weights {
            let f = w * inv_total;
            let sh = &self.probes[i].sh;
            for c in 0..9 {
                out.r[c] += sh.r[c] * f;
                out.g[c] += sh.g[c] * f;
                out.b[c] += sh.b[c] * f;
            }
        }
        out
    }
}

/// Number of nearest probes blended by [`LightProbeSet::blend_sh_at`].
pub const K_NEAREST: usize = 4;

/// Evaluate the order-2 SH basis (9 functions) for a unit direction.
fn sh_basis(d: [f32; 3]) -> [f32; 9] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

/// Per-band cosine-lobe convolution factors, pre-divided by PI so that a
/// constant environment of radiance `L` evaluates back to `L` (the quantity a
/// diffuse surface multiplies by its albedo). Band 0: PI/PI = 1; band 1:
/// (2PI/3)/PI; band 2: (PI/4)/PI.
const SH_COSINE_A: [f32; 9] = [
    1.0,
    2.0 / 3.0,
    2.0 / 3.0,
    2.0 / 3.0,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
];

/// Evaluate SH radiance as diffuse irradiance for a surface normal.
///
/// Applies the cosine-lobe convolution, so the result is the irradiance the
/// surface receives divided by PI: a value ready to multiply by base colour,
/// matching how the IBL irradiance map is consumed. Negative lobes from SH
/// ringing are clamped to zero.
pub fn evaluate_sh(sh: &SHCoefficients, normal: [f32; 3]) -> [f32; 3] {
    let n = normalize(normal);
    let yb = sh_basis(n);
    let mut out = [0.0f32; 3];
    for i in 0..9 {
        let s = yb[i] * SH_COSINE_A[i];
        out[0] += sh.r[i] * s;
        out[1] += sh.g[i] * s;
        out[2] += sh.b[i] * s;
    }
    [out[0].max(0.0), out[1].max(0.0), out[2].max(0.0)]
}

/// Project an equirectangular HDR panorama to order-2 SH radiance.
///
/// `rgba` is row-major linear RGBA, `width * height * 4` floats, using the same
/// direction convention as `dir_to_equirect_uv` in `helpers/ambient.wgsl`
/// (`phi = atan2(y, x)` around +Z, `theta = asin(z)`). This is the input a
/// [`CapturedHdr`](crate::renderer::CapturedHdr) from
/// [`capture_equirect`](crate::renderer::ViewportRenderer::capture_equirect)
/// produces, so a captured probe feeds straight in.
pub fn project_equirect_to_sh(rgba: &[f32], width: u32, height: u32) -> SHCoefficients {
    let (w, h) = (width as usize, height as usize);
    let mut sh = SHCoefficients::default();
    let dphi = std::f32::consts::TAU / w as f32;
    let dtheta = std::f32::consts::PI / h as f32;
    for y in 0..h {
        let v = (y as f32 + 0.5) / h as f32;
        let theta = (0.5 - v) * std::f32::consts::PI; // latitude, +Z polar
        let (st, ct) = theta.sin_cos();
        // Solid angle of this texel row: cos(latitude) d_phi d_theta.
        let domega = dphi * dtheta * ct;
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let phi = (u - 0.5) * std::f32::consts::TAU;
            let (sp, cp) = phi.sin_cos();
            let dir = [ct * cp, ct * sp, st];
            let yb = sh_basis(dir);
            let o = (y * w + x) * 4;
            let (rr, gg, bb) = (rgba[o], rgba[o + 1], rgba[o + 2]);
            for i in 0..9 {
                let wgt = yb[i] * domega;
                sh.r[i] += rr * wgt;
                sh.g[i] += gg * wgt;
                sh.b[i] += bb * wgt;
            }
        }
    }
    sh
}

fn dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-8 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform white environment projects to a DC-only SH whose evaluation is
    /// the same constant in every direction (rotational invariance) and equal to
    /// the input radiance. This pins the projection/evaluation convention and
    /// normalisation without any GPU.
    #[test]
    fn constant_environment_evaluates_flat() {
        let (w, h) = (64u32, 32u32);
        let rgba = vec![1.0f32; (w * h * 4) as usize];
        let sh = project_equirect_to_sh(&rgba, w, h);

        for n in [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.577, 0.577, 0.577],
        ] {
            let e = evaluate_sh(&sh, n);
            for c in 0..3 {
                assert!(
                    (e[c] - 1.0).abs() < 0.05,
                    "constant env normal {n:?} channel {c} evaluated to {} (expected ~1.0)",
                    e[c]
                );
            }
        }
    }

    /// A bright cap in the +X hemisphere must make a +X-facing normal brighter
    /// than a -X-facing one. This pins the directional sign of the projection.
    #[test]
    fn directional_environment_is_brighter_toward_the_light() {
        let (w, h) = (128u32, 64u32);
        let mut rgba = vec![0.0f32; (w * h * 4) as usize];
        for y in 0..h {
            let v = (y as f32 + 0.5) / h as f32;
            let theta = (0.5 - v) * std::f32::consts::PI;
            let (st, ct) = theta.sin_cos();
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let phi = (u - 0.5) * std::f32::consts::TAU;
                let (sp, cp) = phi.sin_cos();
                let dir = [ct * cp, ct * sp, st];
                // Bright only for directions near +X.
                if dir[0] > 0.7 {
                    let o = (y as usize * w as usize + x as usize) * 4;
                    rgba[o] = 5.0;
                    rgba[o + 1] = 5.0;
                    rgba[o + 2] = 5.0;
                }
            }
        }
        let sh = project_equirect_to_sh(&rgba, w, h);
        let toward = evaluate_sh(&sh, [1.0, 0.0, 0.0])[0];
        let away = evaluate_sh(&sh, [-1.0, 0.0, 0.0])[0];
        assert!(
            toward > away + 0.1,
            "+X normal ({toward}) should be brighter than -X normal ({away})"
        );
    }

    /// The blend returns a probe's own SH at its position and an interpolated
    /// value between two probes, and is zero for an empty set.
    #[test]
    fn blend_interpolates_between_probes() {
        let mut a = SHCoefficients::default();
        a.r[0] = 1.0;
        let mut b = SHCoefficients::default();
        b.r[0] = 3.0;
        let set = LightProbeSet::new(vec![
            LightProbe {
                position: [0.0, 0.0, 0.0],
                sh: a,
            },
            LightProbe {
                position: [10.0, 0.0, 0.0],
                sh: b,
            },
        ]);

        // On top of probe A.
        assert!((set.blend_sh_at([0.0, 0.0, 0.0]).r[0] - 1.0).abs() < 1e-3);
        // On top of probe B.
        assert!((set.blend_sh_at([10.0, 0.0, 0.0]).r[0] - 3.0).abs() < 1e-3);
        // Between them the value is strictly inside (1, 3).
        let mid = set.blend_sh_at([5.0, 0.0, 0.0]).r[0];
        assert!(mid > 1.0 && mid < 3.0, "midpoint blended to {mid}");

        assert_eq!(
            LightProbeSet::default().blend_sh_at([0.0; 3]),
            SHCoefficients::default()
        );
    }
}
