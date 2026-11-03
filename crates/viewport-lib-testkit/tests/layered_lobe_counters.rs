//! Layered lobes compile into a mesh pipeline only for the batches that use
//! them. A scene without clear coat, sheen or anisotropy builds the same
//! feature-set pipelines as it did before the lobes existed; a lobe grid,
//! whose spheres share one batch, builds one set for the union of its lobes.

use viewport_lib_testkit::{Harness, frame_for, scenes::scene_by_name};

const W: u32 = 320;
const H: u32 = 240;

/// Feature-set pipelines a fresh harness builds to draw `name`, settled.
fn feature_pipelines(name: &str) -> Option<u32> {
    let mut harness = Harness::new()?;
    let scene = scene_by_name(name).expect("scene");
    let built = harness.build_scene(&scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    Some(
        harness
            .render_two_frames(&frame, W, H)
            .feature_pipelines_built,
    )
}

#[test]
fn lobe_free_scenes_keep_their_pipeline_count() {
    for (name, expected) in [
        ("primitives_trio", 1),
        ("materials_pbr", 1),
        ("materials_ibl", 1),
        ("materials_textures", 3),
    ] {
        let Some(built) = feature_pipelines(name) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(built, expected, "{name}: feature-set pipelines built");
    }
}

#[test]
fn each_lobe_grid_builds_one_feature_set() {
    for name in ["lobes_clearcoat", "lobes_sheen", "lobes_anisotropy"] {
        let Some(built) = feature_pipelines(name) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(built, 1, "{name}: feature-set pipelines built");
    }
}
