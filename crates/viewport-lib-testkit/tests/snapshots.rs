//! Golden-image snapshot tests over the catalogue: exact by default.
//!
//! Each catalogue scene is rendered headlessly at 400x300 from its first camera
//! and compared to a committed reference PNG. The comparison is exact unless the
//! scene has a measured tolerance in `tests/snapshots/tolerances.txt`; the policy
//! and the comparison live in the `golden` module.
//!
//! References live in `tests/snapshots/<adapter>/` (`metal/`,
//! `vulkan-lavapipe/`, ...; see `Harness::backend_dir_name`) as committed PNG
//! files, blessed per adapter on its pinned reference machine: different
//! backends and rasterisers draw the same frame differently, so each keeps its
//! own reference set and its own `tolerances.txt`. A GPU or driver change needs
//! a re-bless:
//!
//!     BLESS=1 cargo test --release --test snapshots
//!
//! A scene with no reference fails. On an adapter with no blessed set, check
//! that every scene renders without comparing pixels:
//!
//!     GOLDEN_RENDER_ONLY=1 cargo test --release --test snapshots
//!
//! which writes the frames under `<target>/golden-render/<adapter>/`. A set
//! blessed on a software rasteriser may carry a `MESA_VERSION` stamp; when the
//! live driver differs the test fails asking for a re-bless. On a machine with
//! no GPU adapter the test skips.

use std::path::{Path, PathBuf};
use viewport_lib_testkit::golden::{self, GoldenMode, Outcome, RgbaImage, Tolerance};
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
    let mode = GoldenMode::from_env();
    let dir = snapshot_dir().join(harness.backend_dir_name());
    if mode != GoldenMode::Bless {
        if let Err(e) = golden::check_driver_stamp(&dir, harness.driver_info().unwrap_or("")) {
            panic!("{e}");
        }
    }
    let tolerances = golden::load_tolerances(&dir.join("tolerances.txt"));

    let mut failures: Vec<String> = Vec::new();
    let mut generated = 0usize;
    let mut rendered: Option<PathBuf> = None;

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
        match golden::check(&dir, scene.name, &actual, tol, mode) {
            Outcome::Blessed => generated += 1,
            Outcome::Rendered { path } => rendered = path.parent().map(Path::to_path_buf),
            Outcome::Match => {}
            Outcome::Missing { .. } => failures.push(golden::missing_message(scene.name, &dir)),
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
            "blessed {generated} reference image(s) in {}",
            dir.display()
        );
    }
    if let Some(out) = rendered {
        eprintln!(
            "render only: frames written to {}, not compared",
            out.display()
        );
    }
    assert!(
        failures.is_empty(),
        "snapshot failures (actual/diff written next to references):\n  {}",
        failures.join("\n  ")
    );
}
