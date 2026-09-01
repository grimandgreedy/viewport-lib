//! Double-render determinism probe.
//!
//! Renders each catalogue scene twice on the same device and diffs the two raw
//! frames. This measures run-to-run determinism, which is what the golden policy
//! rests on: a scene that renders bit-identically twice on one device can be
//! held to an exact golden; one that does not (an order-dependent transparency
//! pass, atomic light binning, ...) has a real non-determinism floor and needs a
//! measured per-scene tolerance rather than exact.
//!
//! The probe reports every non-bit-identical scene with its worst per-channel
//! delta and how many pixels moved, so the numbers can be transcribed into
//! `tests/snapshots/tolerances.txt`. It is deliberately loud, not a hard gate:
//! it renders the same frame twice, so any difference is the renderer's own
//! non-determinism on this adapter, not a code change.

use viewport_lib_testkit::{Harness, catalogue, frame_for};

const W: u32 = 400;
const H: u32 = 300;

/// (worst per-channel delta, number of differing pixels) between two RGBA frames.
fn diff(a: &[u8], b: &[u8]) -> (u8, u64) {
    let mut worst = 0u8;
    let mut moved = 0u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let mut pixel_moved = false;
        for c in 0..4 {
            let d = (i16::from(pa[c]) - i16::from(pb[c])).unsigned_abs() as u8;
            worst = worst.max(d);
            if d != 0 {
                pixel_moved = true;
            }
        }
        if pixel_moved {
            moved += 1;
        }
    }
    (worst, moved)
}

// Ignored by default: this is a diagnostic to run deliberately on the reference
// adapter when measuring or updating the tolerance manifest, not a per-commit
// gate. Its result is adapter-specific, and a tiny non-determinism floor (a
// pixel or two off by 1 in an instanced scene) is expected on some GPUs, so
// gating every commit on it would be noise. Run it with:
//
//     cargo test --test determinism -- --ignored --nocapture
#[test]
#[ignore = "diagnostic probe; run explicitly on the reference adapter"]
fn scenes_render_deterministically() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut nondeterministic: Vec<String> = Vec::new();

    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        // Warm once so first-frame uploads and cache building settle, then take
        // two settled frames and compare them.
        let _ = harness.render(&frame, W, H);
        let first = harness.render(&frame, W, H);
        let second = harness.render(&frame, W, H);

        let (worst, moved) = diff(&first, &second);
        if moved != 0 {
            nondeterministic.push(format!(
                "{}: worst_channel_delta={worst} pixels_moved={moved}/{}",
                scene.name,
                (W * H) as u64
            ));
        }
    }

    if nondeterministic.is_empty() {
        eprintln!("all catalogue scenes render bit-identically twice on this adapter");
    }
    assert!(
        nondeterministic.is_empty(),
        "non-deterministic scenes on this adapter (each needs a measured tolerance \
         in tests/snapshots/tolerances.txt):\n  {}",
        nondeterministic.join("\n  ")
    );
}
