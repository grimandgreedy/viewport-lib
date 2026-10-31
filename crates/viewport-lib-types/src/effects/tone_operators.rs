//! CPU versions of the tone-map operators, matching the renderer's composite.
//!
//! Each operator is a plain function on a linear RGB triple. The per-channel
//! operators ([`reinhard`], [`aces_fit`], [`khronos_neutral`], [`generic`])
//! work in whatever primaries they are given; [`agx`] and [`aces_rrt`] take
//! and return linear Rec.2020. [`tone_map`] runs the composite's operator
//! block on exposed Rec.709 radiance: the working-space conversion, the
//! operator and the conversion back, without the colour grade.

use super::colour_space::{
    AP0_TO_AP1, AP1_LUMINANCE, AP1_TO_REC2020, AP1_TO_XYZ, Mat3, REC709_LUMA, REC2020_TO_AP0,
    REC2020_TO_SRGB, SRGB_TO_REC2020, XYZ_TO_AP1, apply, inverse,
};
use super::postprocess::{
    AgxLook, GenericToneCurve, PostProcessSettings, ToneMapping, WorkingSpace,
};

/// AgX inset: Rec.2020 into the AgX log-encoding primaries. Rows sum to 1.
pub const AGX_INSET: Mat3 = [
    [0.856627153315983, 0.0951212405381588, 0.0482516061458583],
    [0.137318972929847, 0.761241990602591, 0.101439036467562],
    [0.11189821299995, 0.0767994186031903, 0.811302368396859],
];

/// The inverse of [`AGX_OUTSET`], as the reference publishes it.
pub const AGX_OUTSET_INVERSE: Mat3 = [
    [0.899796955911611, 0.0871996192028351, 0.013003424885555],
    [0.11142098895748, 0.875575586156966, 0.0130034248855548],
    [0.11142098895748, 0.0871996192028349, 0.801379391839686],
];

/// AgX outset: the sigmoid's output back to Rec.2020.
pub const AGX_OUTSET: Mat3 = inverse(&AGX_OUTSET_INVERSE);

/// The lower end of AgX's log2 encoding, in stops about 1.
pub const AGX_MIN_EV: f32 = -12.47393;
/// The upper end of AgX's log2 encoding, in stops about 1.
pub const AGX_MAX_EV: f32 = 4.026069;

fn map(v: [f32; 3], f: impl Fn(f32) -> f32) -> [f32; 3] {
    [f(v[0]), f(v[1]), f(v[2])]
}

fn dot(a: [f32; 3], w: [f64; 3]) -> f32 {
    a[0] * w[0] as f32 + a[1] * w[1] as f32 + a[2] * w[2] as f32
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// `x / (x + 1)` per channel.
pub fn reinhard(c: [f32; 3]) -> [f32; 3] {
    map(c, |x| x / (x + 1.0))
}

/// Narkowicz's ACES fit per channel, clamped to `[0, 1]`
/// ([`ToneMapping::Aces`]).
pub fn aces_fit(c: [f32; 3]) -> [f32; 3] {
    map(c, |x| {
        ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
    })
}

/// Khronos PBR Neutral ([`ToneMapping::KhronosNeutral`]).
pub fn khronos_neutral(colour: [f32; 3]) -> [f32; 3] {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;
    let x = colour[0].min(colour[1].min(colour[2]));
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    let c = map(colour, |v| v - offset);
    let peak = c[0].max(c[1].max(c[2]));
    if peak < start_compression {
        return c;
    }
    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    let scaled = map(c, |v| v * (new_peak / peak));
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    mix(scaled, [new_peak; 3], g)
}

/// The [`ToneMapping::Generic`] curve per channel. Negative channels map to
/// 0.
pub fn generic(c: [f32; 3], curve: &GenericToneCurve) -> [f32; 3] {
    let (contrast, input_scale, output_scale) = curve.scales();
    generic_with_scales(c, contrast, input_scale, output_scale)
}

/// [`generic`] from precomputed [`GenericToneCurve::scales`].
pub fn generic_with_scales(
    c: [f32; 3],
    contrast: f32,
    input_scale: f32,
    output_scale: f32,
) -> [f32; 3] {
    map(c, |x| {
        let p = x.max(0.0).powf(contrast);
        output_scale * p / (p + input_scale)
    })
}

/// AgX's 7th-order fit of its default-contrast sigmoid, on `[0, 1]`.
fn agx_sigmoid(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    -17.86 * x6 * x + 78.01 * x6 - 126.7 * x4 * x + 92.06 * x4 - 28.72 * x2 * x + 4.361 * x2
        - 0.1718 * x
        + 0.002857
}

fn agx_look(v: [f32; 3], look: AgxLook) -> [f32; 3] {
    if look == AgxLook::None {
        return v;
    }
    let luma = dot(v, REC709_LUMA);
    let (slope, power, sat) = look.cdl();
    let graded = [
        (v[0] * slope[0]).powf(power[0]),
        (v[1] * slope[1]).powf(power[1]),
        (v[2] * slope[2]).powf(power[2]),
    ];
    map(graded, |g| luma + sat * (g - luma))
}

/// AgX ([`ToneMapping::Agx`]) on linear Rec.2020, returning linear Rec.2020
/// (before any clamp).
pub fn agx(rec2020: [f32; 3], look: AgxLook) -> [f32; 3] {
    let v = apply(&AGX_INSET, map(rec2020, |x| x.max(0.0)));
    let v = map(v, |x| {
        let encoded = (x.max(1e-10).log2() - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);
        agx_sigmoid(encoded.clamp(0.0, 1.0))
    });
    let v = agx_look(v, look);
    let v = apply(&AGX_OUTSET, v);
    map(v, |x| x.max(0.0).powf(2.2))
}

/// The largest value a half float holds, where the ACES stages clamp.
const HALF_MAX: f32 = 65504.0;

fn rgb_to_saturation(rgb: [f32; 3]) -> f32 {
    const TINY: f32 = 1e-5;
    let mi = rgb[0].min(rgb[1].min(rgb[2]));
    let ma = rgb[0].max(rgb[1].max(rgb[2]));
    (ma.max(TINY) - mi.max(TINY)) / ma.max(1e-2)
}

fn rgb_to_yc(rgb: [f32; 3]) -> f32 {
    const YC_RADIUS_WEIGHT: f32 = 1.75;
    let [r, g, b] = rgb;
    let chroma = (b * (b - g) + g * (g - r) + r * (r - b)).max(0.0).sqrt();
    (b + g + r + YC_RADIUS_WEIGHT * chroma) / 3.0
}

fn sigmoid_shaper(x: f32) -> f32 {
    let t = (1.0 - (x / 2.0).abs()).max(0.0);
    let y = 1.0 + sign(x) * (1.0 - t * t);
    y / 2.0
}

/// WGSL's `sign`: 0 at 0.
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn glow_fwd(yc_in: f32, glow_gain_in: f32, glow_mid: f32) -> f32 {
    if yc_in <= 2.0 / 3.0 * glow_mid {
        glow_gain_in
    } else if yc_in >= 2.0 * glow_mid {
        0.0
    } else {
        glow_gain_in * (glow_mid / yc_in - 0.5)
    }
}

fn rgb_to_hue(rgb: [f32; 3]) -> f32 {
    let mut hue = 0.0;
    if !(rgb[0] == rgb[1] && rgb[1] == rgb[2]) {
        hue = (3.0_f32.sqrt() * (rgb[1] - rgb[2]))
            .atan2(2.0 * rgb[0] - rgb[1] - rgb[2])
            .to_degrees();
    }
    if hue < 0.0 { hue + 360.0 } else { hue }
}

fn centre_hue(hue: f32, centre: f32) -> f32 {
    let h = hue - centre;
    if h < -180.0 {
        h + 360.0
    } else if h > 180.0 {
        h - 360.0
    } else {
        h
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn dark_to_dim_surround(linear_cv: [f32; 3]) -> [f32; 3] {
    const DIM_SURROUND_GAMMA: f32 = 0.9811;
    let xyz = apply(&AP1_TO_XYZ, linear_cv);
    let sum = (xyz[0] + xyz[1] + xyz[2]).max(1e-5);
    let x = xyz[0] / sum;
    let y = xyz[1] / sum;
    let big_y = xyz[1].clamp(0.0, HALF_MAX).powf(DIM_SURROUND_GAMMA);
    let m = big_y / y.max(1e-5);
    let xyz = [x * m, big_y, (1.0 - x - y) * m];
    apply(&XYZ_TO_AP1, xyz)
}

/// The ACES RRT and ODT approximation ([`ToneMapping::AcesRrt`]) on linear
/// Rec.2020, returning linear Rec.2020 (before any clamp).
pub fn aces_rrt(rec2020: [f32; 3]) -> [f32; 3] {
    aces_rrt_with_brightness(rec2020, 1.0)
}

/// [`aces_rrt`] with the AP1 signal scaled by `brightness` before the fitted
/// curve.
pub fn aces_rrt_with_brightness(rec2020: [f32; 3], brightness: f32) -> [f32; 3] {
    const RRT_GLOW_GAIN: f32 = 0.05;
    const RRT_GLOW_MID: f32 = 0.08;
    const RRT_RED_SCALE: f32 = 0.82;
    const RRT_RED_PIVOT: f32 = 0.03;
    const RRT_RED_HUE: f32 = 0.0;
    const RRT_RED_WIDTH: f32 = 135.0;
    const RRT_SAT_FACTOR: f32 = 0.96;
    const ODT_SAT_FACTOR: f32 = 0.93;

    let mut ap0 = apply(&REC2020_TO_AP0, rec2020);

    // Glow.
    let saturation = rgb_to_saturation(ap0);
    let yc_in = rgb_to_yc(ap0);
    let s = sigmoid_shaper((saturation - 0.4) / 0.2);
    let added_glow = 1.0 + glow_fwd(yc_in, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
    ap0 = map(ap0, |c| c * added_glow);

    // Red modifier.
    let hue = rgb_to_hue(ap0);
    let centred = centre_hue(hue, RRT_RED_HUE);
    let mut hue_weight = smoothstep(0.0, 1.0, 1.0 - (2.0 * centred / RRT_RED_WIDTH).abs());
    hue_weight *= hue_weight;
    ap0[0] += hue_weight * saturation * (RRT_RED_PIVOT - ap0[0]) * (1.0 - RRT_RED_SCALE);

    // To the rendering space, then global desaturation.
    let ap1 = map(apply(&AP0_TO_AP1, ap0), |c| c.clamp(0.0, HALF_MAX));
    let ap1 = mix([dot(ap1, AP1_LUMINANCE); 3], ap1, RRT_SAT_FACTOR);
    let ap1 = map(ap1, |c| c * brightness);

    // Fitted RRT and ODT (100 nit dim-surround display).
    const A: f32 = 2.785085;
    const B: f32 = 0.107772;
    const C: f32 = 2.936045;
    const D: f32 = 0.887122;
    const E: f32 = 0.806889;
    let post = map(ap1, |x| (x * (A * x + B)) / (x * (C * x + D) + E));

    let linear_cv = dark_to_dim_surround(post);
    let linear_cv = mix(
        [dot(linear_cv, AP1_LUMINANCE); 3],
        linear_cv,
        ODT_SAT_FACTOR,
    );
    apply(&AP1_TO_REC2020, linear_cv)
}

/// The composite's operator block on exposed Rec.709 radiance: convert to
/// `post.working_space`, apply `operator`, convert back to Rec.709. AgX and
/// ACES RRT convert to Rec.2020 and back, clamping negative channels to 0.
/// The colour grade is not applied.
pub fn tone_map(colour: [f32; 3], operator: ToneMapping, post: &PostProcessSettings) -> [f32; 3] {
    let rec2020 = post.working_space == WorkingSpace::Rec2020;
    let c = if rec2020 {
        apply(&SRGB_TO_REC2020, colour)
    } else {
        colour
    };
    // Into Rec.2020 and back out to the working space, for the operators that
    // run there.
    let wide = |c: [f32; 3], f: &dyn Fn([f32; 3]) -> [f32; 3]| {
        if rec2020 {
            map(f(c), |x| x.max(0.0))
        } else {
            map(
                apply(&REC2020_TO_SRGB, f(apply(&SRGB_TO_REC2020, c))),
                |x| x.max(0.0),
            )
        }
    };
    let c = match operator {
        ToneMapping::Reinhard => reinhard(c),
        ToneMapping::Aces => aces_fit(c),
        ToneMapping::KhronosNeutral => khronos_neutral(c),
        ToneMapping::Agx => wide(c, &|v| agx(v, post.agx_look)),
        ToneMapping::AcesRrt => wide(c, &aces_rrt),
        ToneMapping::Generic => generic(c, &post.generic_tone_curve),
    };
    if rec2020 {
        map(apply(&REC2020_TO_SRGB, c), |x| x.max(0.0))
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::colour_space::mul;

    #[test]
    fn agx_matrices_keep_neutral() {
        for m in [AGX_INSET, AGX_OUTSET] {
            for row in m {
                assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{m:?}");
            }
        }
        let id = mul(&AGX_OUTSET, &AGX_OUTSET_INVERSE);
        for (i, row) in id.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn generic_hits_its_anchors() {
        let curve = GenericToneCurve::default();
        let mid = generic([curve.mid_grey_in; 3], &curve)[0];
        let white = generic([curve.hdr_max; 3], &curve)[0];
        assert!((mid - curve.mid_grey_out).abs() < 1e-5, "{mid}");
        assert!((white - 1.0).abs() < 1e-5, "{white}");
        assert_eq!(generic([0.0; 3], &curve), [0.0; 3]);
    }

    #[test]
    fn neutral_stays_neutral_through_agx() {
        let post = PostProcessSettings::default();
        for x in [0.01, 0.18, 1.0, 4.0, 50.0] {
            let c = tone_map([x; 3], ToneMapping::Agx, &post);
            assert!(
                (c[0] - c[1]).abs() < 1e-4 && (c[1] - c[2]).abs() < 1e-4,
                "{x}: {c:?}"
            );
        }
    }

    /// AP0 and AP1 use the ACES white point and the conversion does not adapt
    /// it, so a D65 grey leaves the curve slightly blue in the shadows and
    /// warm in the highlights (about 0.92, 0.87, 0.81 at 4). The spread stays
    /// bounded.
    #[test]
    fn aces_rrt_tint_on_neutral_is_bounded() {
        let post = PostProcessSettings::default();
        for x in [0.01, 0.18, 1.0, 4.0, 50.0] {
            let c = tone_map([x; 3], ToneMapping::AcesRrt, &post);
            let spread = c[0].max(c[1].max(c[2])) - c[0].min(c[1].min(c[2]));
            assert!(spread <= 0.15 * c[1].max(0.01), "{x}: {c:?}");
        }
    }

    #[test]
    fn wide_operators_are_monotonic_and_bounded() {
        let post = PostProcessSettings::default();
        for op in [ToneMapping::Agx, ToneMapping::AcesRrt, ToneMapping::Generic] {
            let mut last = -1.0;
            for i in 0..200 {
                let x = 2f32.powf(-10.0 + i as f32 * 0.1);
                let y = tone_map([x; 3], op, &post)[1];
                assert!(
                    y.is_finite() && y >= last - 1e-6,
                    "{op:?} at {x}: {y} < {last}"
                );
                assert!(y <= 1.05, "{op:?} at {x}: {y}");
                last = y;
            }
        }
    }

    #[test]
    fn black_is_finite() {
        let post = PostProcessSettings::default();
        for op in [ToneMapping::Agx, ToneMapping::AcesRrt, ToneMapping::Generic] {
            let c = tone_map([0.0; 3], op, &post);
            assert!(
                c.iter().all(|v| v.is_finite() && *v < 0.01),
                "{op:?}: {c:?}"
            );
        }
    }

    #[test]
    fn rec2020_working_space_leaves_neutral_per_channel_operators_alone() {
        let mut post = PostProcessSettings::default();
        let a = tone_map([0.5; 3], ToneMapping::Reinhard, &post);
        post.working_space = WorkingSpace::Rec2020;
        let b = tone_map([0.5; 3], ToneMapping::Reinhard, &post);
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() < 1e-5, "{a:?} {b:?}");
        }
    }
}
