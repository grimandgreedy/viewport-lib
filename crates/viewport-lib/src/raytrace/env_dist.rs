//! Importance-sampling distribution for an equirect HDR environment.
//!
//! Builds a 2D piecewise-constant distribution over the environment image so the
//! path tracer can sample directions toward bright regions (next-event
//! estimation on the environment) instead of finding them only by chance through
//! BSDF sampling. This is the classic marginal/conditional CDF pair (pbrt's
//! `Distribution2D`), weighted by `sin(theta)` so the density is uniform in solid
//! angle rather than in the distorted equirect parameterisation.
//!
//! The tables are consumed on the GPU by `sample_env` / `env_pdf` in
//! `raytrace.wgsl`; the [`sample`](EnvDistribution::sample) and
//! [`pdf`](EnvDistribution::pdf) methods here are the CPU reference the shader
//! mirrors, and are what the unit tests check.

// `build`/`tables`/`width`/`height`/`integral` feed the GPU upload; `sample` and
// `pdf` are the CPU reference the shader mirrors, exercised by the unit tests.
#![allow(dead_code)]

use std::f32::consts::PI;

/// Perceptual luminance of a linear RGB triple.
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// A sampled environment direction in parameter space.
pub(crate) struct EnvSample {
    /// Horizontal parameter in [0, 1): longitude, wrapping.
    pub u: f32,
    /// Vertical parameter in [0, 1]: latitude, 0 at the top (+Z).
    pub v: f32,
    /// Density in the unit square (`func / integral`); convert to a solid-angle
    /// pdf by dividing by `2 * PI^2 * sin(theta)`.
    pub pdf_uv: f32,
}

/// Marginal + conditional CDFs over an equirect environment's sin-weighted
/// luminance. Row-major, matching the image's `height` rows of `width` texels.
pub(crate) struct EnvDistribution {
    width: u32,
    height: u32,
    /// Sin-weighted luminance per texel, `height * width`. Kept for pdf lookups.
    func: Vec<f32>,
    /// Per-row conditional CDFs, each `width + 1` long and ending at 1.0,
    /// concatenated as `height * (width + 1)`.
    conditional_cdf: Vec<f32>,
    /// Marginal CDF over rows, `height + 1` long, ending at 1.0.
    marginal_cdf: Vec<f32>,
    /// Mean of `func` over the image: the normaliser that makes `func / integral`
    /// integrate to 1 over the unit square.
    integral: f32,
}

impl EnvDistribution {
    /// Build the distribution from linear RGBA f32 pixels (`width * height * 4`).
    /// A black image yields a uniform distribution so sampling still terminates.
    pub(crate) fn build(pixels: &[f32], width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;
        debug_assert_eq!(pixels.len(), w * h * 4);

        // Sin-weighted luminance: the equirect row at latitude theta subtends a
        // solid angle proportional to sin(theta), so weighting by it makes the
        // density uniform in solid angle.
        let mut func = vec![0.0f32; w * h];
        for y in 0..h {
            let theta = PI * (y as f32 + 0.5) / h as f32;
            let sin_t = theta.sin();
            for x in 0..w {
                let i = (y * w + x) * 4;
                let lum = luminance(pixels[i], pixels[i + 1], pixels[i + 2]).max(0.0);
                func[y * w + x] = lum * sin_t;
            }
        }

        // Conditional CDF per row, and each row's integral.
        let mut conditional_cdf = vec![0.0f32; h * (w + 1)];
        let mut row_integral = vec![0.0f32; h];
        for y in 0..h {
            let base = y * (w + 1);
            for x in 0..w {
                conditional_cdf[base + x + 1] =
                    conditional_cdf[base + x] + func[y * w + x] / w as f32;
            }
            let integ = conditional_cdf[base + w];
            row_integral[y] = integ;
            if integ > 0.0 {
                for x in 1..=w {
                    conditional_cdf[base + x] /= integ;
                }
            } else {
                // Uniform fallback for a black row.
                for x in 1..=w {
                    conditional_cdf[base + x] = x as f32 / w as f32;
                }
            }
        }

        // Marginal CDF over the row integrals.
        let mut marginal_cdf = vec![0.0f32; h + 1];
        for y in 0..h {
            marginal_cdf[y + 1] = marginal_cdf[y] + row_integral[y] / h as f32;
        }
        let integral = marginal_cdf[h];
        if integral > 0.0 {
            for y in 1..=h {
                marginal_cdf[y] /= integral;
            }
        } else {
            for y in 1..=h {
                marginal_cdf[y] = y as f32 / h as f32;
            }
        }

        Self {
            width,
            height,
            func,
            conditional_cdf,
            marginal_cdf,
            integral: integral.max(1.0e-8),
        }
    }

    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    pub(crate) fn height(&self) -> u32 {
        self.height
    }

    pub(crate) fn integral(&self) -> f32 {
        self.integral
    }

    /// The flattened tables, in the layout the GPU buffers use:
    /// `(func, conditional_cdf, marginal_cdf)`.
    pub(crate) fn tables(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.func, &self.conditional_cdf, &self.marginal_cdf)
    }

    /// Find the interval `x` falls in: the largest `i` with `cdf[off + i] <= x`,
    /// clamped to `[0, n - 2]` so both `cdf[off + i]` and `cdf[off + i + 1]` are
    /// valid (matching pbrt's `FindInterval`). `cdf` here has `n` entries, a
    /// leading 0 and trailing 1.
    fn find_interval(cdf: &[f32], off: usize, n: usize, x: f32) -> usize {
        // Linear scan is fine for the CPU reference; the shader binary-searches.
        let mut i = 0usize;
        while i + 1 < n && cdf[off + i + 1] <= x {
            i += 1;
        }
        i.min(n - 2)
    }

    /// Sample a direction parameter from two uniform variates, returning `(u, v)`
    /// and the density in the unit square. Mirrors the WGSL `sample_env`.
    pub(crate) fn sample(&self, u1: f32, u2: f32) -> EnvSample {
        let w = self.width as usize;
        let h = self.height as usize;

        // Marginal: pick a row.
        let y = Self::find_interval(&self.marginal_cdf, 0, h + 1, u2);
        let my0 = self.marginal_cdf[y];
        let my1 = self.marginal_cdf[y + 1];
        let dy = (my1 - my0).max(1.0e-12);
        let v = (y as f32 + (u2 - my0) / dy) / h as f32;

        // Conditional: pick a column within that row.
        let base = y * (w + 1);
        let x = Self::find_interval(&self.conditional_cdf, base, w + 1, u1);
        let cx0 = self.conditional_cdf[base + x];
        let cx1 = self.conditional_cdf[base + x + 1];
        let dx = (cx1 - cx0).max(1.0e-12);
        let u = (x as f32 + (u1 - cx0) / dx) / w as f32;

        EnvSample {
            u,
            v,
            pdf_uv: self.func[y * w + x] / self.integral,
        }
    }

    /// Density in the unit square for a parameter `(u, v)`, matching WGSL
    /// `env_pdf` before the solid-angle Jacobian. `u`/`v` are in `[0, 1]`.
    pub(crate) fn pdf(&self, u: f32, v: f32) -> f32 {
        let w = self.width as usize;
        let h = self.height as usize;
        let x = ((u.fract() + 1.0).fract() * w as f32) as usize;
        let y = (v.clamp(0.0, 1.0) * h as f32) as usize;
        let x = x.min(w - 1);
        let y = y.min(h - 1);
        self.func[y * w + x] / self.integral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a WxH black image with a single bright texel at (bx, by).
    fn one_bright(w: u32, h: u32, bx: u32, by: u32, val: f32) -> Vec<f32> {
        let mut px = vec![0.0f32; (w * h * 4) as usize];
        let i = ((by * w + bx) * 4) as usize;
        px[i] = val;
        px[i + 1] = val;
        px[i + 2] = val;
        px[i + 3] = 1.0;
        px
    }

    #[test]
    fn cdfs_are_normalised() {
        let (w, h) = (16, 8);
        let px = one_bright(w, h, 5, 3, 10.0);
        let d = EnvDistribution::build(&px, w, h);
        // Every conditional row ends at 1, and the marginal ends at 1.
        for y in 0..h as usize {
            let end = d.conditional_cdf[y * (w as usize + 1) + w as usize];
            assert!((end - 1.0).abs() < 1e-5, "row {y} cdf ends at {end}");
        }
        let m = d.marginal_cdf[h as usize];
        assert!((m - 1.0).abs() < 1e-5, "marginal ends at {m}");
    }

    #[test]
    fn sampling_concentrates_on_the_bright_texel() {
        let (w, h) = (16, 8);
        let (bx, by) = (5u32, 3u32);
        let px = one_bright(w, h, bx, by, 10.0);
        let d = EnvDistribution::build(&px, w, h);

        // A low-discrepancy-ish grid of variates; all should land in the bright
        // texel since it is the only non-zero weight.
        let n = 40;
        let mut hits = 0;
        for i in 0..n {
            for j in 0..n {
                let u1 = (i as f32 + 0.5) / n as f32;
                let u2 = (j as f32 + 0.5) / n as f32;
                let s = d.sample(u1, u2);
                let sx = (s.u * w as f32) as u32;
                let sy = (s.v * h as f32) as u32;
                if sx == bx && sy == by {
                    hits += 1;
                }
                assert!(s.pdf_uv > 0.0, "sampled a zero-pdf direction");
            }
        }
        assert_eq!(hits, n * n, "every sample should hit the one bright texel");
    }

    #[test]
    fn uniform_image_is_solid_angle_uniform() {
        // A constant image should be uniform in solid angle, i.e. its solid-angle
        // pdf is 1/(4*PI) everywhere. The unit-square density is not flat: it is
        // proportional to sin(theta), which the Jacobian 2*PI^2*sin(theta) cancels.
        let (w, h) = (32, 16);
        let px = vec![1.0f32; (w * h * 4) as usize];
        let d = EnvDistribution::build(&px, w, h);
        let uniform_sphere = 1.0 / (4.0 * PI);
        for v in [0.2f32, 0.5, 0.8] {
            for u in [0.1f32, 0.5, 0.9] {
                let sin_t = (PI * v).sin();
                let p_omega = d.pdf(u, v) / (2.0 * PI * PI * sin_t);
                assert!(
                    (p_omega - uniform_sphere).abs() < 0.02,
                    "solid-angle pdf at ({u},{v}) = {p_omega}, expected {uniform_sphere}"
                );
            }
        }
    }

    #[test]
    fn pdf_matches_sample_density() {
        let (w, h) = (32, 16);
        // A smooth gradient so several texels carry weight.
        let mut px = vec![0.0f32; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let c = (x as f32 / w as f32) + 0.05;
                px[i] = c;
                px[i + 1] = c;
                px[i + 2] = c;
                px[i + 3] = 1.0;
            }
        }
        let d = EnvDistribution::build(&px, w, h);
        // The pdf at a sampled point equals the density the sampler reports there.
        for (u1, u2) in [(0.2f32, 0.3f32), (0.7, 0.6), (0.9, 0.1)] {
            let s = d.sample(u1, u2);
            let p = d.pdf(s.u, s.v);
            assert!(
                (p - s.pdf_uv).abs() < 1e-3,
                "pdf {p} != sample pdf {} at ({},{})",
                s.pdf_uv,
                s.u,
                s.v
            );
        }
    }
}
