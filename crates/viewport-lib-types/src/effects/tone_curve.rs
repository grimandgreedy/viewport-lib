//! Tone curves as data: bake any curve into a view-transform LUT the
//! composite applies in place of a built-in operator.
//!
//! A [`ToneCurve`] maps exposed scene-linear radiance to a display value.
//! [`bake_view_transform_lut`] samples one (optionally after a scene-referred
//! grade) over the [`VIEW_LUT_SHAPER`] log domain into a strip LUT. Upload the
//! result like any texture and name it with
//! [`ToneMapping::Custom`](crate::effects::postprocess::ToneMapping::Custom).

use super::postprocess::{PostProcessSettings, ToneMapping};
use super::tone_operators::operator_in_working_space;
use crate::data::texture::TextureData;

/// A curve from exposed scene-linear radiance to display-linear values.
///
/// Implemented for [`ToneMapping`] (with the default settings), for
/// [`OperatorCurve`] (an operator with explicit settings), and for any
/// `Fn([f32; 3]) -> [f32; 3]`.
pub trait ToneCurve {
    /// Map one exposed scene-linear colour.
    fn map(&self, scene_linear: [f32; 3]) -> [f32; 3];
}

impl<F: Fn([f32; 3]) -> [f32; 3]> ToneCurve for F {
    fn map(&self, scene_linear: [f32; 3]) -> [f32; 3] {
        self(scene_linear)
    }
}

/// The operator with [`PostProcessSettings::default`]. See [`OperatorCurve`].
impl ToneCurve for ToneMapping {
    fn map(&self, scene_linear: [f32; 3]) -> [f32; 3] {
        OperatorCurve::new(*self, &PostProcessSettings::default()).map(scene_linear)
    }
}

/// A built-in operator with the settings it reads (the AgX look, the generic
/// curve and the working space).
///
/// Maps a colour already in `post.working_space`, as the composite's operator
/// step sees it: the conversion into and out of the working space is not
/// applied. [`ToneMapping::Custom`] passes the colour through, since its curve
/// lives in a texture.
#[derive(Clone, Copy, Debug)]
pub struct OperatorCurve<'a> {
    operator: ToneMapping,
    post: &'a PostProcessSettings,
}

impl<'a> OperatorCurve<'a> {
    /// `operator` reading its settings from `post`.
    pub fn new(operator: ToneMapping, post: &'a PostProcessSettings) -> Self {
        Self { operator, post }
    }
}

impl ToneCurve for OperatorCurve<'_> {
    fn map(&self, scene_linear: [f32; 3]) -> [f32; 3] {
        operator_in_working_space(scene_linear, self.operator, self.post)
    }
}

/// A log2 shaper: maps radiance in `[2^log2_min, 2^(log2_min + log2_range)]`
/// to `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogShaper {
    /// The log2 of the lowest radiance the shaper resolves.
    pub log2_min: f32,
    /// The number of stops the shaper spans.
    pub log2_range: f32,
}

impl LogShaper {
    /// Shape one channel: `clamp((log2(max(x, 2^log2_min)) - log2_min) / log2_range, 0, 1)`.
    pub fn shape(&self, x: f32) -> f32 {
        let floor = self.log2_min.exp2();
        ((x.max(floor).log2() - self.log2_min) / self.log2_range).clamp(0.0, 1.0)
    }

    /// The radiance a shaped value stands for: `2^(s * log2_range + log2_min)`.
    pub fn unshape(&self, s: f32) -> f32 {
        (s * self.log2_range + self.log2_min).exp2()
    }
}

/// The shaper the composite applies before a scene-referred LUT lookup, both
/// for [`GradingSpace::Scene`] and for [`ToneMapping::Custom`]: 24 stops from
/// `2^-14` to `2^10`, with mid-grey (0.18) at about 0.48. Radiance outside the
/// range clamps to its ends, negative channels included.
///
/// [`GradingSpace::Scene`]: crate::effects::postprocess::GradingSpace::Scene
pub const VIEW_LUT_SHAPER: LogShaper = LogShaper {
    log2_min: -14.0,
    log2_range: 24.0,
};

/// Bake `curve`, after `grade` when one is given, into a view-transform strip
/// LUT of `size` slices for [`ToneMapping::Custom`].
///
/// The layout is the grading LUT's: `size` slices of `size` x `size` laid out
/// horizontally (width `size * size`, height `size`), red across a slice,
/// green down it, blue selecting the slice. Each lattice coordinate `s` in
/// `[0, 1]` stands for the radiance [`VIEW_LUT_SHAPER`] unshapes it to; the
/// texel holds `curve(grade(radiance))`, a display-linear value. The
/// composite shapes exposed radiance the same way, samples the LUT with
/// linear filtering and uses the result in place of the operator.
///
/// `grade` maps scene-linear radiance to scene-linear radiance. Baking the
/// grade in gives one LUT for the whole view transform; leave
/// [`PostProcessSettings::grade_lut`] unset when doing so.
///
/// The texture holds float texels and uploads as `Rgba16Float`. At 32 slices
/// (a 1024 x 32 texture) a per-channel curve such as Reinhard or the generic
/// curve comes within a few 8-bit display steps of the operator. A curve
/// that mixes channels through their minimum or maximum, as Khronos Neutral
/// does, interpolates less well near the grey axis: its error shrinks only
/// in proportion to the slice count.
///
/// # Panics
///
/// When `size` is less than 2.
pub fn bake_view_transform_lut(
    curve: &dyn ToneCurve,
    grade: Option<&dyn ToneCurve>,
    size: u32,
) -> TextureData {
    assert!(size >= 2, "a view-transform LUT needs at least 2 slices");
    let n = size;
    let lattice = |v: u32| VIEW_LUT_SHAPER.unshape(v as f32 / (n - 1) as f32);
    let mut texels = Vec::with_capacity((n * n * n * 4) as usize);
    for g in 0..n {
        for b in 0..n {
            for r in 0..n {
                let radiance = [lattice(r), lattice(g), lattice(b)];
                let graded = grade.map_or(radiance, |grade| grade.map(radiance));
                let out = curve.map(graded);
                texels.extend_from_slice(&[out[0], out[1], out[2], 1.0]);
            }
        }
    }
    TextureData::hdr(n * n, n, texels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::texture::TexturePayload;
    use crate::effects::tone_operators::reinhard;

    /// The composite's lookup on the CPU: shape, then trilinear in the strip.
    fn sample(lut: &TextureData, colour: [f32; 3]) -> [f32; 3] {
        let n = lut.height() as usize;
        let TexturePayload::Rgba32F(texels) = lut.payload() else {
            panic!("float payload expected");
        };
        let texel = |r: usize, g: usize, b: usize| {
            let i = (g * n * n + b * n + r) * 4;
            [texels[i], texels[i + 1], texels[i + 2]]
        };
        let pos = colour.map(|c| VIEW_LUT_SHAPER.shape(c) * (n - 1) as f32);
        let lo = pos.map(|p| (p.floor() as usize).min(n - 2));
        let f = [0, 1, 2].map(|i| pos[i] - lo[i] as f32);
        let mut out = [0.0; 3];
        for (dr, wr) in [(0, 1.0 - f[0]), (1, f[0])] {
            for (dg, wg) in [(0, 1.0 - f[1]), (1, f[1])] {
                for (db, wb) in [(0, 1.0 - f[2]), (1, f[2])] {
                    let t = texel(lo[0] + dr, lo[1] + dg, lo[2] + db);
                    for c in 0..3 {
                        out[c] += t[c] * wr * wg * wb;
                    }
                }
            }
        }
        out
    }

    /// Test inputs across the shaper's range, off the lattice.
    fn inputs() -> Vec<[f32; 3]> {
        (0..97)
            .map(|i| {
                let s = 0.12 + 0.85 * (i as f32 + 0.37) / 97.0;
                let x = VIEW_LUT_SHAPER.unshape(s);
                [x, x * 0.55, x * 0.2 + 1e-4]
            })
            .collect()
    }

    #[test]
    fn shaper_round_trips() {
        for i in 0..=48 {
            let s = i as f32 / 48.0;
            let back = VIEW_LUT_SHAPER.shape(VIEW_LUT_SHAPER.unshape(s));
            assert!((back - s).abs() < 1e-5, "{s}: {back}");
        }
        assert_eq!(VIEW_LUT_SHAPER.shape(-1.0), 0.0);
        assert_eq!(VIEW_LUT_SHAPER.shape(1e6), 1.0);
    }

    #[test]
    fn baked_lut_has_the_strip_layout() {
        let lut = bake_view_transform_lut(&|c: [f32; 3]| c, None, 8);
        assert_eq!((lut.width(), lut.height()), (64, 8));
        // The lattice corners hold the shaper's ends.
        let lo = sample(&lut, [0.0; 3]);
        let hi = sample(&lut, [2048.0; 3]);
        assert!((lo[0] - VIEW_LUT_SHAPER.unshape(0.0)).abs() < 1e-9);
        assert!((hi[2] - VIEW_LUT_SHAPER.unshape(1.0)).abs() < 1e-3);
    }

    /// Linear interpolation of the linear output between lattice points
    /// overshoots the curve by at most `max_t (1 - t + t 2^h) / 2^(t h) - 1`
    /// for a lattice step of `h` stops: about 3.7% at 32 slices.
    #[test]
    fn baked_identity_round_trips_within_the_lattice_step() {
        let size = 32;
        let lut = bake_view_transform_lut(&|c: [f32; 3]| c, None, size);
        let h = VIEW_LUT_SHAPER.log2_range / (size - 1) as f32;
        let bound = (0..=1000)
            .map(|i| {
                let t = i as f32 / 1000.0;
                (1.0 - t + t * h.exp2()) / (t * h).exp2() - 1.0
            })
            .fold(0.0f32, f32::max);
        assert!(bound < 0.04, "{bound}");
        for c in inputs() {
            let got = sample(&lut, c);
            for i in 0..3 {
                let rel = (got[i] - c[i]).abs() / c[i];
                assert!(rel <= bound + 1e-4, "{c:?}: {got:?} ({rel} > {bound})");
            }
        }
    }

    #[test]
    fn baked_reinhard_round_trips() {
        let lut = bake_view_transform_lut(&reinhard, None, 32);
        let mut worst = 0.0f32;
        for c in inputs() {
            let got = sample(&lut, c);
            let want = reinhard(c);
            for i in 0..3 {
                worst = worst.max((got[i] - want[i]).abs());
            }
        }
        // Second-derivative bound of x / (x + 1) over a 24/31-stop step.
        eprintln!("baked Reinhard at 32: worst {worst}");
        assert!(worst < 4e-3, "{worst}");
    }

    #[test]
    fn grade_applies_before_the_curve() {
        let double = |c: [f32; 3]| c.map(|x| x * 2.0);
        let lut = bake_view_transform_lut(&reinhard, Some(&double), 32);
        let got = sample(&lut, [0.5; 3]);
        let want = reinhard([1.0; 3]);
        assert!((got[0] - want[0]).abs() < 4e-3, "{got:?} {want:?}");
    }

    #[test]
    fn operator_curve_matches_tone_map_in_rec709() {
        let post = PostProcessSettings::default();
        for op in [
            ToneMapping::Reinhard,
            ToneMapping::Agx,
            ToneMapping::Generic,
        ] {
            let curve = OperatorCurve::new(op, &post);
            for c in inputs() {
                let a = curve.map(c);
                let b = super::super::tone_operators::tone_map(c, op, &post);
                assert_eq!(a, b, "{op:?}");
            }
        }
    }
}
