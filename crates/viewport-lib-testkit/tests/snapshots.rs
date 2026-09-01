//! Golden-image snapshot tests.
//!
//! Each catalogue scene is rendered headlessly at 400x300 from its first camera
//! and compared to a committed reference PNG with a perceptual tolerance (mean
//! absolute delta plus a cap on how many pixels may differ noticeably), not
//! exact equality, because GPUs differ. On mismatch the actual frame and a diff
//! image are written next to the reference for inspection.
//!
//! References live in `tests/snapshots/` as committed PNG files. They reflect the
//! machine that blessed them; a GPU or driver change needs a re-bless. Regenerate
//! with:
//!
//!     BLESS=1 cargo test --test snapshots
//!
//! On a machine with no GPU adapter the test skips. The first run on a fresh
//! checkout with no references generates them (and passes); subsequent runs
//! compare against them.

use image::{Rgba, RgbaImage};
use std::path::{Path, PathBuf};
use viewport_lib_testkit::{Harness, catalogue, frame_for};

const W: u32 = 400;
const H: u32 = 300;

/// Per-channel difference (0..255) above which a pixel counts as "noticeably
/// different".
const PIXEL_DELTA: i32 = 8;
/// Maximum mean absolute per-channel delta across the whole image.
const MAX_MEAN_ABS: f64 = 2.0;
/// Maximum fraction of pixels allowed to exceed `PIXEL_DELTA`.
const MAX_OVER_FRACTION: f64 = 0.001;

fn snapshot_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots")
}

/// (mean absolute per-channel delta, fraction of pixels over `PIXEL_DELTA`).
fn compare(reference: &RgbaImage, actual: &RgbaImage) -> (f64, f64) {
    let mut sum_abs: u64 = 0;
    let mut over: u64 = 0;
    let total = (reference.width() * reference.height()) as u64;
    for (r, a) in reference.pixels().zip(actual.pixels()) {
        let mut pixel_over = false;
        for c in 0..4 {
            let d = (r.0[c] as i32 - a.0[c] as i32).abs();
            sum_abs += d as u64;
            if d > PIXEL_DELTA {
                pixel_over = true;
            }
        }
        if pixel_over {
            over += 1;
        }
    }
    let mean_abs = sum_abs as f64 / (total * 4) as f64;
    let over_frac = over as f64 / total as f64;
    (mean_abs, over_frac)
}

/// Build a diff image: red where any channel differs by more than `PIXEL_DELTA`,
/// dimmed greyscale of the reference elsewhere.
fn diff_image(reference: &RgbaImage, actual: &RgbaImage) -> RgbaImage {
    let mut out = RgbaImage::new(reference.width(), reference.height());
    for (x, y, p) in out.enumerate_pixels_mut() {
        let r = reference.get_pixel(x, y);
        let a = actual.get_pixel(x, y);
        let changed = (0..4).any(|c| (r.0[c] as i32 - a.0[c] as i32).abs() > PIXEL_DELTA);
        *p = if changed {
            Rgba([255, 0, 0, 255])
        } else {
            let g = (r.0[0] as u32 + r.0[1] as u32 + r.0[2] as u32) / 6;
            Rgba([g as u8, g as u8, g as u8, 255])
        };
    }
    out
}

#[test]
fn scene_snapshots_match_references() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let bless = std::env::var_os("BLESS").is_some();
    let dir = snapshot_dir();
    std::fs::create_dir_all(&dir).expect("create snapshot dir");

    let mut failures: Vec<String> = Vec::new();
    let mut generated = 0usize;

    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        // Render twice; snapshot the settled second frame.
        let _ = harness.render(&frame, W, H);
        let pixels = harness.render(&frame, W, H);
        let actual = RgbaImage::from_raw(W, H, pixels).expect("frame -> image");

        let ref_path = dir.join(format!("{}.png", scene.name));
        if bless || !ref_path.exists() {
            actual.save(&ref_path).expect("write reference");
            generated += 1;
            continue;
        }

        let reference = image::open(&ref_path).expect("open reference").to_rgba8();
        if reference.dimensions() != actual.dimensions() {
            failures.push(format!("{}: dimension mismatch", scene.name));
            continue;
        }

        let (mean_abs, over_frac) = compare(&reference, &actual);
        if mean_abs > MAX_MEAN_ABS || over_frac > MAX_OVER_FRACTION {
            actual
                .save(dir.join(format!("{}.actual.png", scene.name)))
                .expect("write actual");
            diff_image(&reference, &actual)
                .save(dir.join(format!("{}.diff.png", scene.name)))
                .expect("write diff");
            failures.push(format!(
                "{}: mean_abs={:.3} (max {:.1}), over_frac={:.5} (max {:.5})",
                scene.name, mean_abs, MAX_MEAN_ABS, over_frac, MAX_OVER_FRACTION
            ));
        }
    }

    if generated > 0 {
        eprintln!(
            "generated {generated} reference image(s) in {}",
            dir.display()
        );
    }
    assert!(
        failures.is_empty(),
        "snapshot mismatches (actual/diff written next to references):\n  {}",
        failures.join("\n  ")
    );
}
