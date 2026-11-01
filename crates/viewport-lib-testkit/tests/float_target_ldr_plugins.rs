//! An item-type plugin drawn on the `Direct` path into an `Rgba16Float`
//! output picks its LDR pipeline: the pass is the LDR pass even though its
//! format is the HDR scene format.

use viewport_lib::PipelineMode;
use viewport_lib_testkit::{Harness, frame_for, scene_by_name};

const W: u32 = 160;
const H: u32 = 120;

#[test]
fn point_cloud_draws_on_the_direct_path_into_a_float_target() {
    let Some(mut h) = Harness::with_target_format(Harness::FLOAT_TARGET_FORMAT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let scene = scene_by_name("point_cloud").expect("point cloud in the catalogue");
    let built = h.build_scene(&scene);
    let mut frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    frame.effects.display.mode = PipelineMode::Direct;
    let empty = {
        let mut f = frame_for(
            &viewport_lib_testkit::BuiltScene::default(),
            &scene.cameras[0].camera,
            [W as f32, H as f32],
        );
        f.effects.display.mode = PipelineMode::Direct;
        f.scene.generation = 1;
        h.render_float(&f, W, H)
    };
    let _ = h.render_float(&frame, W, H);
    let px = h.render_float(&frame, W, H);
    let drawn = px.iter().zip(&empty).filter(|(a, b)| a != b).count();
    assert!(drawn > 100, "only {drawn} pixels show the point cloud");
}
