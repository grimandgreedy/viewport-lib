//! Smoke test: every catalogue scene builds and renders headlessly.
//!
//! This is the end-to-end check that the catalogue and harness work together. It
//! does not assert on counters; it only proves each scene uploads its assets,
//! produces a frame, and issues draw calls. Skips cleanly on machines with no
//! GPU adapter.

use viewport_lib_testkit::{Harness, catalogue, frame_for};

#[test]
fn every_scene_builds_and_renders() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let (w, h) = (320u32, 240u32);
    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let has_content = !built.items.is_empty()
            || !built.point_clouds.is_empty()
            || !built.polylines.is_empty()
            || !built.glyphs.is_empty();
        assert!(has_content, "{}: built no content", scene.name);
        assert!(!scene.cameras.is_empty(), "{}: no cameras", scene.name);

        let camera = &scene.cameras[0].camera;
        let frame = frame_for(&built, camera, [w as f32, h as f32]);
        let pixels = harness.render(&frame, w, h);
        assert_eq!(
            pixels.len(),
            (w * h * 4) as usize,
            "{}: unexpected pixel buffer size",
            scene.name
        );

        // Mesh scenes must issue draw calls; non-mesh item types (point clouds,
        // polylines, glyphs) are not counted in `draw_calls`, so their
        // correctness is covered by the snapshot test instead.
        if !built.items.is_empty() {
            let stats = harness.stats();
            assert!(
                stats.draw_calls > 0,
                "{}: rendered zero draw calls",
                scene.name
            );
        }
    }
}
