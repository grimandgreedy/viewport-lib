//! Golden-image snapshot tests over the catalogue: exact by default.
//!
//! Each catalogue scene is rendered headlessly at 400x300 from its first camera
//! and compared to a committed reference PNG. The comparison is exact unless the
//! scene has a measured tolerance in `tests/snapshots/tolerances.txt`; the policy
//! and the comparison live in the `golden` module.
//!
//! References live in `tests/snapshots/` as committed PNG files, blessed on the
//! pinned reference adapter. A GPU or driver change needs a re-bless:
//!
//!     BLESS=1 cargo test --test snapshots
//!
//! On a machine with no GPU adapter the test skips. The first run on a fresh
//! checkout with no references generates them (and passes); subsequent runs
//! compare against them.

use std::path::{Path, PathBuf};
use viewport_lib_testkit::golden::{self, Outcome, RgbaImage, Tolerance};
use viewport_lib_testkit::{Harness, catalogue, frame_for};

const W: u32 = 400;
const H: u32 = 300;

fn snapshot_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots")
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
    let tolerances = golden::load_tolerances(&dir.join("tolerances.txt"));

    let mut failures: Vec<String> = Vec::new();
    let mut generated = 0usize;

    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        // Render twice; snapshot the settled second frame.
        let _ = harness.render(&frame, W, H);
        let pixels = harness.render(&frame, W, H);
        let actual = RgbaImage::from_raw(W, H, pixels).expect("frame -> image");

        let tol = tolerances
            .get(scene.name)
            .copied()
            .unwrap_or(Tolerance::EXACT);
        match golden::check(&dir, scene.name, &actual, tol, bless) {
            Outcome::Blessed => generated += 1,
            Outcome::Match => {}
            Outcome::Mismatch {
                comparison,
                tolerance,
            } => failures.push(format!(
                "{}: max_channel_delta={} pixels_over={}/{} (allowed: delta<={}, over<={})",
                scene.name,
                comparison.max_channel_delta,
                comparison.pixels_over,
                comparison.total_pixels,
                tolerance.max_channel_delta,
                tolerance.max_pixels_over,
            )),
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
