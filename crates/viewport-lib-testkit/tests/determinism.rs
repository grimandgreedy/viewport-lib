//! Double-render determinism probe.
//!
//! Renders each catalogue scene in two separately built harnesses (a new
//! device, renderer and scene each time) and diffs the two raw frames. This
//! measures run-to-run determinism, which is what the golden policy rests on: a
//! scene that renders bit-identically in two fresh harnesses can be held to an
//! exact golden; one that does not (an order-dependent transparency
//! pass, atomic light binning, ...) has a real non-determinism floor and needs a
//! measured per-scene tolerance rather than exact.
//!
//! The probe reports every non-bit-identical scene with its worst relative
//! difference and how many pixels moved, so the numbers can be transcribed into
//! `tests/snapshots/<backend>/tolerances.txt`. It is deliberately loud: it renders the
//! same frame twice, so any difference is the renderer's own non-determinism on
//! this adapter, not a code change.
//!
//! It renders into a half-float target rather than the 8-bit one the goldens
//! use, and that matters more than it sounds. An 8-bit readback only reports a
//! pixel whose value happens to cross a quantisation step, so a difference
//! smaller than a step shows up on a fraction of the pixels it affects and only
//! on some runs: a real, every-frame order dependence can read as an occasional
//! one-pixel flicker. Comparing floats reports it wherever it occurs, every
//! time.
//!
//! A scene with an entry in the adapter's `tolerances.txt` (the same reference
//! directory the snapshot suite uses) is held to that tolerance instead of
//! bit-identity: a pixel counts as moved when a channel differs by more than
//! `max_channel_delta` on the 0..255 scale, and the scene passes while no more
//! than `max_pixels_over` pixels move. Scenes without an entry stay exact.

use std::path::Path;
use viewport_lib_testkit::golden::{self, Tolerance};
use viewport_lib_testkit::{Harness, NamedScene, catalogue, frame_for};

const W: u32 = 400;
const H: u32 = 300;

/// (worst relative difference in any channel, number of differing pixels)
/// between two float frames.
///
/// Pixels are compared on their exact bits, so a difference of a single
/// half-float step counts. The relative figure is what separates one of those
/// steps (around 1e-3) from a genuinely different value reaching the target.
fn diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> (f32, u64) {
    let mut worst = 0f32;
    let mut moved = 0u64;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let mut pixel_moved = false;
        for c in 0..4 {
            if pa[c].to_bits() == pb[c].to_bits() {
                continue;
            }
            pixel_moved = true;
            let denom = pa[c].abs().max(pb[c].abs()).max(1e-6);
            worst = worst.max((pa[c] - pb[c]).abs() / denom);
        }
        if pixel_moved {
            moved += 1;
        }
    }
    (worst, moved)
}

/// Pixels with any channel differing by more than `tol.max_channel_delta / 255`.
fn pixels_over(a: &[[f32; 4]], b: &[[f32; 4]], tol: Tolerance) -> u64 {
    let limit = f32::from(tol.max_channel_delta) / 255.0;
    a.iter()
        .zip(b.iter())
        .filter(|(pa, pb)| (0..4).any(|c| (pa[c] - pb[c]).abs() > limit))
        .count() as u64
}

/// The settled frame of `scene` from a freshly built harness, rendered the way
/// the snapshot suite renders it: one warm frame, then the frame that counts.
/// A scene with a simulation steps it the same number of times in every
/// harness, so two harnesses agree unless the renderer itself does not.
fn settled_frame(scene: &NamedScene) -> Vec<[f32; 4]> {
    let mut harness =
        Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT).expect("adapter went away");
    let built = harness.build_scene(scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let _ = harness.render_float(&frame, W, H);
    harness.render_float(&frame, W, H)
}

// Ignored by default: this is a diagnostic to run deliberately on the reference
// adapter when measuring or updating the tolerance manifest, not a per-commit
// gate. Its result is adapter-specific and it renders the whole catalogue twice,
// so it is too slow and too machine-dependent to run on every commit. A scene
// that moves here is worth chasing rather than absorbing into a tolerance: the
// renderer has no accepted non-determinism floor outside the order-dependent
// transparency pass. Run it with:
//
//     cargo test --test determinism -- --ignored --nocapture
#[test]
#[ignore = "diagnostic probe; run explicitly on the reference adapter"]
fn scenes_render_deterministically() {
    let Some(backend_dir) =
        Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT).map(|h| h.backend_dir_name())
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let tolerances = golden::load_tolerances(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/snapshots")
            .join(backend_dir)
            .join("tolerances.txt"),
    );
    let mut nondeterministic: Vec<String> = Vec::new();

    for scene in catalogue() {
        let first = settled_frame(&scene);
        let second = settled_frame(&scene);

        let (worst, moved) = diff(&first, &second);
        if moved == 0 {
            continue;
        }
        if let Some(&tol) = tolerances.get(scene.name) {
            let over = pixels_over(&first, &second, tol);
            if over <= tol.max_pixels_over {
                eprintln!(
                    "{}: within tolerance (pixels_moved={moved}, over={over}, allowed: \
                     delta<={}, over<={})",
                    scene.name, tol.max_channel_delta, tol.max_pixels_over
                );
            } else {
                nondeterministic.push(format!(
                    "{}: pixels_over={over}/{} exceeds tolerance (delta<={}, over<={})",
                    scene.name,
                    (W * H) as u64,
                    tol.max_channel_delta,
                    tol.max_pixels_over
                ));
            }
        } else {
            nondeterministic.push(format!(
                "{}: worst_relative_difference={worst:.6} pixels_moved={moved}/{}",
                scene.name,
                (W * H) as u64
            ));
        }
    }

    if nondeterministic.is_empty() {
        eprintln!(
            "all catalogue scenes render bit-identically twice on this adapter, or \
             within their tolerance (half-float target)"
        );
    }
    assert!(
        nondeterministic.is_empty(),
        "non-deterministic scenes on this adapter (each needs a measured tolerance \
         in tests/snapshots/<backend>/tolerances.txt):\n  {}",
        nondeterministic.join("\n  ")
    );
}
