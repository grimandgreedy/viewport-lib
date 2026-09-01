//! Deterministic `FrameStats` counter assertions.
//!
//! These counters are exact, not timings, so they carry no noise and are stable
//! across machines (they are CPU-side: object counts, batch counts, triangle
//! counts, draw calls). Locking them catches a whole class of regressions the
//! instant they happen: batching silently degrading to the per-object path, a
//! static scene re-uploading every frame, culling that stops culling.
//!
//! The expected values are recorded literals, reviewed on change. A deliberate
//! batching or mesh-generation change shows up as a visible diff here; an
//! accidental one shows up as a failure. Regenerate the table with
//! `cargo run --example dump_counters` after an intended change.

use viewport_lib_testkit::{Harness, catalogue, frame_for};

/// Expected per-frame counts for a settled (second-frame) render of a scene's
/// first camera at 320x240.
struct Expected {
    total_objects: u32,
    visible_objects: u32,
    draw_calls: u32,
    instanced_batches: u32,
    per_object_items: u32,
    triangles_submitted: u64,
}

fn expected(name: &str) -> Option<Expected> {
    let e = |total, visible, draws, batches, per_obj, tris| Expected {
        total_objects: total,
        visible_objects: visible,
        draw_calls: draws,
        instanced_batches: batches,
        per_object_items: per_obj,
        triangles_submitted: tris,
    };
    Some(match name {
        "primitives_trio" => e(3, 3, 3, 3, 0, 1996),
        "torus_knot" => e(1, 1, 1, 0, 0, 7680),
        "gear" => e(1, 1, 1, 0, 0, 224),
        "bowl" => e(1, 1, 1, 0, 0, 2784),
        "castellated_bar" => e(2, 2, 2, 2, 0, 96),
        "heightfield" => e(1, 1, 1, 0, 0, 18432),
        "thin_sheet" => e(1, 1, 1, 0, 0, 4608),
        "stress_sphere" => e(1, 1, 1, 0, 0, 81920),
        "concave_shadows" => e(3, 3, 3, 3, 0, 8732),
        "textured_checker" => e(1, 1, 1, 0, 0, 2208),
        "textured_normalmap" => e(1, 1, 1, 0, 0, 3968),
        "transparent" => e(3, 3, 1, 1, 0, 2880),
        "materials_pbr" => e(25, 25, 1, 1, 0, 55200),
        "many_objects" => e(144, 144, 2, 2, 0, 16992),
        "lights_eight" => e(4, 4, 2, 2, 0, 2892),
        "game_mix" => e(85, 85, 4, 4, 0, 2972),
        // Non-mesh item scenes: the mesh-oriented counters are all zero (their
        // draws are not tracked in draw_calls/triangles). Locking the zeros still
        // catches a regression that accidentally routes them through the mesh
        // path; visual correctness is covered by the snapshot test.
        "point_cloud" => e(0, 0, 0, 0, 0, 0),
        "polyline" => e(0, 0, 0, 0, 0, 0),
        "glyphs" => e(0, 0, 0, 0, 0, 0),
        _ => return None,
    })
}

#[test]
fn scene_counters_match_recorded_values() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let (w, h) = (320u32, 240u32);
    for scene in catalogue() {
        let exp = expected(scene.name).unwrap_or_else(|| {
            panic!(
                "no recorded counters for scene '{}'; add it to `expected()` \
                 (regenerate with `cargo run --example dump_counters`)",
                scene.name
            )
        });

        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [w as f32, h as f32]);
        let s = harness.render_two_frames(&frame, w, h);

        assert_eq!(
            s.total_objects, exp.total_objects,
            "{}: total_objects",
            scene.name
        );
        assert_eq!(
            s.visible_objects, exp.visible_objects,
            "{}: visible_objects",
            scene.name
        );
        assert_eq!(s.draw_calls, exp.draw_calls, "{}: draw_calls", scene.name);
        assert_eq!(
            s.instanced_batches, exp.instanced_batches,
            "{}: instanced_batches (batching regressed?)",
            scene.name
        );
        assert_eq!(
            s.per_object_items, exp.per_object_items,
            "{}: per_object_items (items fell off the instanced path?)",
            scene.name
        );
        assert_eq!(
            s.triangles_submitted, exp.triangles_submitted,
            "{}: triangles_submitted (mesh generation changed?)",
            scene.name
        );
    }
}

/// A settled frame must not re-upload instance data or rebuild per-object bind
/// groups: rendering the identical frame twice should leave the second frame's
/// caches untouched. A regression here means per-frame churn that tanks the
/// frame rate even on a static scene.
#[test]
fn settled_frame_does_no_reupload() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (w, h) = (320u32, 240u32);
    for scene in catalogue() {
        let built = harness.build_scene(&scene);
        let frame = frame_for(&built, &scene.cameras[0].camera, [w as f32, h as f32]);
        let s = harness.render_two_frames(&frame, w, h);
        assert_eq!(
            s.batches_reuploaded, 0,
            "{}: re-uploaded a static batch",
            scene.name
        );
        assert_eq!(
            s.per_object_bind_groups_built, 0,
            "{}: rebuilt per-object bind groups on a settled frame",
            scene.name
        );
    }
}

/// Moving the camera over a static scene must not re-upload geometry. The
/// downstream engine's stationary-fast / moving-slow gap is exactly this class
/// of bug, so this turns the symptom into a deterministic check.
#[test]
fn camera_motion_does_no_reupload() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (w, h) = (320u32, 240u32);
    for scene in catalogue() {
        if scene.cameras.len() < 2 {
            continue;
        }
        let built = harness.build_scene(&scene);
        // Settle at the first camera.
        let f0 = frame_for(&built, &scene.cameras[0].camera, [w as f32, h as f32]);
        let _ = harness.render_two_frames(&f0, w, h);
        // Move to a different camera; geometry is unchanged.
        let f1 = frame_for(&built, &scene.cameras[1].camera, [w as f32, h as f32]);
        let _ = harness.render(&f1, w, h);
        let s = harness.stats();
        assert_eq!(
            s.batches_reuploaded, 0,
            "{}: camera motion re-uploaded static batches",
            scene.name
        );
        assert_eq!(
            s.per_object_bind_groups_built, 0,
            "{}: camera motion rebuilt per-object bind groups",
            scene.name
        );
    }
}
