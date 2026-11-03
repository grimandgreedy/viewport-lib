//! Linear RGB primaries conversions used by the view transform.
//!
//! Matrices are row-major (`out[i] = sum_j m[i][j] * in[j]`) and in f64, so
//! the renderer's shader constants can be checked against them. sRGB and
//! Rec.2020 use the D65 white point; ACES AP0 and AP1 use the ACES white
//! point and are converted to and from XYZ without chromatic adaptation, as
//! the reference implementations do.

/// A row-major 3x3 matrix.
pub type Mat3 = [[f64; 3]; 3];

/// Linear sRGB (Rec.709 primaries) to CIE XYZ.
pub const SRGB_TO_XYZ: Mat3 = [
    [0.4123907992659593, 0.357584339383878, 0.1804807884018343],
    [0.2126390058715103, 0.715168678767756, 0.07219231536073371],
    [0.01933081871559182, 0.119194779794626, 0.9505321522496607],
];

/// Linear Rec.2020 to CIE XYZ.
pub const REC2020_TO_XYZ: Mat3 = [
    [0.6369580483012911, 0.1446169035862084, 0.1688809751641721],
    [0.262700212011267, 0.677998071518871, 0.05930171646986196],
    [0.0, 0.02807269304908744, 1.060985057710791],
];

/// ACES AP0 to CIE XYZ.
pub const AP0_TO_XYZ: Mat3 = [
    [0.9525523959381857, 0.0, 0.00009367863166046855],
    [0.3439664497650751, 0.7281660966134857, -0.07213254637856079],
    [0.0, 0.0, 1.008825184351586],
];

/// ACES AP1 to CIE XYZ.
pub const AP1_TO_XYZ: Mat3 = [
    [0.6624541811085055, 0.1340042064564331, 0.1561876870049078],
    [0.2722287167809146, 0.6740817658111483, 0.05368951740793705],
    [
        -0.005574649490394109,
        0.004060733528982825,
        1.010339100312997,
    ],
];

/// CIE XYZ to linear sRGB.
pub const XYZ_TO_SRGB: Mat3 = inverse(&SRGB_TO_XYZ);
/// CIE XYZ to linear Rec.2020.
pub const XYZ_TO_REC2020: Mat3 = inverse(&REC2020_TO_XYZ);
/// CIE XYZ to ACES AP0.
pub const XYZ_TO_AP0: Mat3 = inverse(&AP0_TO_XYZ);
/// CIE XYZ to ACES AP1.
pub const XYZ_TO_AP1: Mat3 = inverse(&AP1_TO_XYZ);

/// Linear sRGB to linear Rec.2020.
pub const SRGB_TO_REC2020: Mat3 = mul(&XYZ_TO_REC2020, &SRGB_TO_XYZ);
/// Linear Rec.2020 to linear sRGB.
pub const REC2020_TO_SRGB: Mat3 = mul(&XYZ_TO_SRGB, &REC2020_TO_XYZ);
/// Linear Rec.2020 to ACES AP0.
pub const REC2020_TO_AP0: Mat3 = mul(&XYZ_TO_AP0, &REC2020_TO_XYZ);
/// ACES AP0 to ACES AP1.
pub const AP0_TO_AP1: Mat3 = mul(&XYZ_TO_AP1, &AP0_TO_XYZ);
/// ACES AP1 to linear Rec.2020.
pub const AP1_TO_REC2020: Mat3 = mul(&XYZ_TO_REC2020, &AP1_TO_XYZ);

/// Luminance weights of AP1: the Y row of [`AP1_TO_XYZ`].
pub const AP1_LUMINANCE: [f64; 3] = AP1_TO_XYZ[1];

/// Rec.709 luma weights.
pub const REC709_LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// The product `a * b`.
pub const fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    let mut i = 0;
    while i < 3 {
        let mut j = 0;
        while j < 3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            j += 1;
        }
        i += 1;
    }
    out
}

/// The inverse of `m`, by cofactors. `m` must be invertible.
pub const fn inverse(m: &Mat3) -> Mat3 {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    let inv = 1.0 / det;
    [
        [
            c00 * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            c01 * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            c02 * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ]
}

/// `m * v` in f32, the precision the shaders run at.
pub fn apply(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    let row = |r: &[f64; 3]| r[0] as f32 * v[0] + r[1] as f32 * v[1] + r[2] as f32 * v[2];
    [row(&m[0]), row(&m[1]), row(&m[2])]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RGB to XYZ from chromaticities and a white point.
    fn from_primaries(p: [[f64; 2]; 3], white: [f64; 2]) -> Mat3 {
        let col = |xy: [f64; 2]| [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]];
        let cols = [col(p[0]), col(p[1]), col(p[2])];
        let unscaled = [
            [cols[0][0], cols[1][0], cols[2][0]],
            [cols[0][1], cols[1][1], cols[2][1]],
            [cols[0][2], cols[1][2], cols[2][2]],
        ];
        let w = col(white);
        let inv = inverse(&unscaled);
        let s: Vec<f64> = (0..3)
            .map(|i| inv[i][0] * w[0] + inv[i][1] * w[1] + inv[i][2] * w[2])
            .collect();
        let mut out = unscaled;
        for row in &mut out {
            for (j, v) in row.iter_mut().enumerate() {
                *v *= s[j];
            }
        }
        out
    }

    fn assert_close(a: &Mat3, b: &Mat3, eps: f64) {
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (a[i][j] - b[i][j]).abs() <= eps,
                    "[{i}][{j}]: {} vs {}",
                    a[i][j],
                    b[i][j]
                );
            }
        }
    }

    const D65: [f64; 2] = [0.3127, 0.3290];
    const ACES_WHITE: [f64; 2] = [0.32168, 0.33767];

    #[test]
    fn matrices_match_their_primaries() {
        let srgb = from_primaries([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], D65);
        let rec2020 = from_primaries([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], D65);
        let ap0 = from_primaries(
            [[0.7347, 0.2653], [0.0, 1.0], [0.0001, -0.0770]],
            ACES_WHITE,
        );
        let ap1 = from_primaries([[0.713, 0.293], [0.165, 0.830], [0.128, 0.044]], ACES_WHITE);
        assert_close(&SRGB_TO_XYZ, &srgb, 1e-12);
        assert_close(&REC2020_TO_XYZ, &rec2020, 1e-12);
        assert_close(&AP0_TO_XYZ, &ap0, 1e-12);
        assert_close(&AP1_TO_XYZ, &ap1, 1e-12);
    }

    #[test]
    fn round_trips_are_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert_close(&mul(&SRGB_TO_REC2020, &REC2020_TO_SRGB), &id, 1e-12);
        assert_close(&mul(&XYZ_TO_AP1, &AP1_TO_XYZ), &id, 1e-12);
    }

    #[test]
    fn white_maps_to_white_between_d65_spaces() {
        let w = apply(&SRGB_TO_REC2020, [1.0, 1.0, 1.0]);
        for c in w {
            assert!((c - 1.0).abs() < 1e-6, "{w:?}");
        }
    }
}
