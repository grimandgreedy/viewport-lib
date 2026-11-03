//! The tone-map shader's matrices and constants agree with the f64 constants
//! the CPU operators use.

use viewport_lib::vplt::effects::colour_space::{self as cs, Mat3};
use viewport_lib::vplt::effects::tone_operators as ops;

const WGSL: &str = include_str!("../src/shaders/helpers/tone_operators.wgsl");

/// The numbers inside the declaration of `name`, in source order.
fn numbers(name: &str) -> Vec<f64> {
    let start = WGSL
        .find(&format!("const {name} "))
        .or_else(|| WGSL.find(&format!("const {name}:")))
        .unwrap_or_else(|| panic!("{name} not declared"));
    let body = &WGSL[start..start + WGSL[start..].find(';').unwrap()];
    let body = &body[body.find('=').unwrap() + 1..];
    body.replace("mat3x3<f32>", "")
        .replace("vec3<f32>", "")
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .unwrap_or_else(|_| panic!("{name}: bad number {s}"))
        })
        .collect()
}

fn assert_matrix(name: &str, rust: &Mat3) {
    let wgsl = numbers(name);
    assert_eq!(wgsl.len(), 9, "{name}");
    // WGSL lists columns.
    for col in 0..3 {
        for row in 0..3 {
            let (w, r) = (wgsl[col * 3 + row], rust[row][col]);
            assert!(
                (w - r).abs() <= 1e-6,
                "{name}[{row}][{col}]: shader {w}, Rust {r}"
            );
        }
    }
}

#[test]
fn shader_matrices_match_the_rust_constants() {
    assert_matrix("SRGB_TO_REC2020", &cs::SRGB_TO_REC2020);
    assert_matrix("REC2020_TO_SRGB", &cs::REC2020_TO_SRGB);
    assert_matrix("REC2020_TO_AP0", &cs::REC2020_TO_AP0);
    assert_matrix("AP0_TO_AP1", &cs::AP0_TO_AP1);
    assert_matrix("AP1_TO_XYZ", &cs::AP1_TO_XYZ);
    assert_matrix("XYZ_TO_AP1", &cs::XYZ_TO_AP1);
    assert_matrix("AP1_TO_REC2020", &cs::AP1_TO_REC2020);
    assert_matrix("AGX_INSET", &ops::AGX_INSET);
    assert_matrix("AGX_OUTSET", &ops::AGX_OUTSET);
}

#[test]
fn shader_scalars_match_the_rust_constants() {
    let close = |name: &str, want: &[f64]| {
        let got = numbers(name);
        assert_eq!(got.len(), want.len(), "{name}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-6, "{name}: shader {g}, Rust {w}");
        }
    };
    close("AP1_LUMINANCE", &cs::AP1_LUMINANCE);
    close("REC709_LUMA", &cs::REC709_LUMA);
    close("AGX_MIN_EV", &[ops::AGX_MIN_EV as f64]);
    close("AGX_MAX_EV", &[ops::AGX_MAX_EV as f64]);
}
