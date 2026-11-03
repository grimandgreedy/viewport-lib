//! Transmission compiles into a mesh pipeline only for the batches that use
//! it. A scene without a transmissive material builds the same feature-set
//! pipelines as it did before transmission existed; the transmission scene
//! builds one set for its floor and one transmission pipeline its four glass
//! spheres share.

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
fn transmission_free_scenes_keep_their_pipeline_count() {
    for (name, expected) in [
        ("primitives_trio", 1),
        ("materials_pbr", 1),
        ("materials_ibl", 1),
        ("materials_textures", 3),
        ("lobes_clearcoat", 1),
        ("transparent", 1),
    ] {
        let Some(built) = feature_pipelines(name) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(built, expected, "{name}: feature-set pipelines built");
    }
}

#[test]
fn the_glass_spheres_share_one_transmission_pipeline() {
    let Some(built) = feature_pipelines("transmission_checker") else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(built, 2, "floor set plus one transmission set");
}
