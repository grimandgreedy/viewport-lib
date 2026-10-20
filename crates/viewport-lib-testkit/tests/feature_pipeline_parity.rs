//! Feature-set pipelines against the universal one, over the catalogue.
//!
//! A per-object mesh batch draws through a pipeline built with only the shader
//! features its items use. Every feature block also tests its per-draw flag,
//! so the pipeline with every feature compiled in draws the same surfaces.
//! The two are not bit-identical: compiling a block out changes how the
//! backend compiler schedules the code that is left, and on Metal (fast math
//! on) that moves a few pixels across a mip boundary. So this renders each
//! catalogue scene both ways and allows scattered pixels to differ, but no
//! region: a missing feature block or a wrong feature set changes a whole
//! surface, which moves every tile it covers.
//!
//! The exact check is the other test: every record's features fall inside its
//! batch's set.

use viewport_lib_testkit::golden::{RgbaImage, max_block_mean_delta};
use viewport_lib_testkit::{Harness, catalogue, frame_for};

const W: u32 = 400;
const H: u32 = 300;

/// Pixels that may differ, as a fraction of the frame.
const MAX_PIXELS_MOVED: f32 = 0.01;
/// Tile size and the largest mean channel delta any tile may carry.
const BLOCK: u32 = 8;
const MAX_BLOCK_MEAN: f32 = 4.0;

fn image(pixels: Vec<u8>) -> RgbaImage {
    RgbaImage::from_raw(W, H, pixels).expect("frame -> image")
}

#[test]
fn feature_set_pipelines_draw_the_universal_surfaces() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut failures = Vec::new();
    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        harness.renderer.set_mesh_universal(false);
        let first = image(harness.render(&frame, W, H));
        let reduced = image(harness.render(&frame, W, H));
        harness.renderer.set_mesh_universal(true);
        let _ = harness.render(&frame, W, H);
        let universal = image(harness.render(&frame, W, H));
        harness.renderer.set_mesh_universal(false);
        // A scene that does not repeat its own frame (an animated particle
        // system) says nothing about the pipelines.
        if first != reduced {
            eprintln!("{}: not deterministic, skipped", scene.name);
            continue;
        }
        let moved = reduced
            .pixels()
            .zip(universal.pixels())
            .filter(|(a, b)| a != b)
            .count();
        let block = max_block_mean_delta(&universal, &reduced, BLOCK);
        if moved > 0 {
            eprintln!(
                "{}: {moved} pixels moved, worst tile mean {block:.2}",
                scene.name
            );
        }
        if moved as f32 > MAX_PIXELS_MOVED * (W * H) as f32 || block > MAX_BLOCK_MEAN {
            failures.push(format!(
                "{}: {moved} pixels moved, worst {BLOCK}x{BLOCK} tile mean delta {block:.2}",
                scene.name
            ));
        }
    }
    // The gate means nothing unless the catalogue reaches the reduced
    // pipelines at all.
    assert!(
        harness.stats().feature_pipelines_built > 0,
        "no catalogue scene drew through a feature-set pipeline"
    );
    assert!(
        failures.is_empty(),
        "feature-set pipelines drew different surfaces from the universal one:\n  {}",
        failures.join("\n  ")
    );
}

/// Every record's features fall inside its batch's feature set, in every
/// catalogue scene. Exact, and the check that catches a feature-set
/// derivation that misses something.
#[test]
fn every_record_draws_with_the_features_it_uses() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut failures = Vec::new();
    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
        let _ = harness.render(&frame, W, H);
        for line in harness.renderer.feature_set_violations() {
            failures.push(format!("{}: {line}", scene.name));
        }
    }
    assert!(
        failures.is_empty(),
        "records outside their batch's feature set:\n  {}",
        failures.join("\n  ")
    );
}
