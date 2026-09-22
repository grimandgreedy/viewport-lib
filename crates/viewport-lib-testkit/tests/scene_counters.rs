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
        "tensor_glyphs" => e(0, 0, 0, 0, 0, 0),
        "tubes" => e(0, 0, 0, 0, 0, 0),
        "streamtubes" => e(0, 0, 0, 0, 0, 0),
        "ribbons" => e(0, 0, 0, 0, 0, 0),
        "sprites" => e(0, 0, 0, 0, 0, 0),
        // The extra sprite scenes carry mesh geometry for the sprites to draw
        // against: a ground slab and a cube for the soft fade, a cube for the
        // OIT overlaps, and a textured ground for the refraction to distort.
        "sprites_soft" => e(2, 2, 2, 2, 0, 24),
        "sprites_oit" => e(1, 1, 1, 0, 0, 12),
        "sprites_refraction" => e(1, 1, 1, 0, 0, 12),
        // Same content as `sprites_soft`, supersampled: the counters are
        // resolution-independent, so they match it exactly.
        "supersampled_sprites" => e(2, 2, 2, 2, 0, 24),
        "supersampled_sprite_refraction" => e(1, 1, 1, 0, 0, 12),
        // No mesh geometry: the particles are the whole scene.
        "gpu_particles" => e(0, 0, 0, 0, 0, 0),
        "volume" => e(0, 0, 0, 0, 0, 0),
        "gaussian_splats" => e(0, 0, 0, 0, 0, 0),
        "image_slice" => e(0, 0, 0, 0, 0, 0),
        "volume_surface_slice" => e(0, 0, 0, 0, 0, 0),
        "gpu_implicit" => e(0, 0, 0, 0, 0, 0),
        "gpu_marching_cubes" => e(0, 0, 0, 0, 0, 0),
        "mesh_instances" => e(0, 0, 0, 0, 0, 0),
        // These item-type scenes include mesh geometry (ground and receivers
        // for scatter and decals), so the mesh counters are live for them.
        "scatter_volume" => e(2, 2, 2, 2, 0, 972),
        // A fog box containing a dense sphere, downsampled: two volumes over a
        // slab and a pillar, so the back-to-front order has something to get
        // wrong.
        "scatter_layered" => e(2, 2, 2, 2, 0, 24),
        // A texture-driven volume beside a noise-driven one, against a single
        // backdrop slab.
        "scatter_textured" => e(1, 1, 1, 0, 0, 12),
        // Scrolling noise and heat-haze refraction at a pinned clock, over a
        // wall and four struts for the shimmer to bend.
        "scatter_animated" => e(5, 5, 2, 2, 0, 60),
        // Wireframe-only scene: a volume box, splat rings and sprite quads,
        // all drawn as lines through the shared substrate. No mesh geometry.
        "item_wireframes" => e(0, 0, 0, 0, 0, 0),
        "decals" => e(2, 2, 2, 2, 0, 24),
        // The decal scene again with supersampling on: same content and same
        // draw structure, only the resolution differs.
        "supersampled_decals" => e(2, 2, 2, 2, 0, 24),
        // One slab receiving a decal, with soft-particle sprites over it: the
        // scene that pins decal ordering against the depth-read pass.
        "decal_under_soft_sprite" => e(1, 1, 1, 0, 0, 12),
        // One slab with soft-particle sprites and refractive sprites over it:
        // the scene that pins what the refraction samples.
        "refraction_over_soft_sprite" => e(1, 1, 1, 0, 0, 12),
        // A mesh that opted out of decals beside a GPU implicit surface that
        // cannot: pins that decals land on any depth writer, not just meshes.
        "decal_on_non_mesh" => e(1, 1, 1, 0, 0, 960),
        // Tube, streamtube and ribbon under one decal. No mesh geometry: the
        // curve types are the whole scene.
        "decal_on_curves" => e(0, 0, 0, 0, 0, 0),
        // The same scene from below the projection plane, where the shader's
        // view-direction check currently removes the decal outright.
        "decal_from_below" => e(1, 1, 1, 0, 0, 960),
        // The overlay scenes: one sphere backdrop each, so the scene-side
        // counters are identical across all of them. They exist to gate overlay
        // pixels, not scene structure; a change here means the backdrop moved.
        "overlay_shapes" => e(1, 1, 1, 0, 0, 960),
        "overlay_vector" => e(1, 1, 1, 0, 0, 960),
        "overlay_polylines" => e(1, 1, 1, 0, 0, 960),
        "overlay_labels" => e(1, 1, 1, 0, 0, 960),
        "overlay_glyph_runs" => e(1, 1, 1, 0, 0, 960),
        "overlay_shadows" => e(1, 1, 1, 0, 0, 960),
        "overlay_rotation" => e(1, 1, 1, 0, 0, 960),
        "overlay_clipping" => e(1, 1, 1, 0, 0, 960),
        "overlay_retained" => e(1, 1, 1, 0, 0, 960),
        "overlay_composition" => e(1, 1, 1, 0, 0, 960),
        "overlay_text_fill" => e(1, 1, 1, 0, 0, 960),
        "overlay_shadow_parity" => e(1, 1, 1, 0, 0, 960),
        "overlay_group_anchor" => e(1, 1, 1, 0, 0, 960),
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
