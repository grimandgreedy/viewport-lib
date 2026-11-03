//! CPU picking of item types against the frame last prepared.

use viewport_lib::{PickBackend, PickMask};
use viewport_lib_testkit::{Harness, catalogue, frame_for};

const W: u32 = 256;
const H: u32 = 192;

/// An item type whose items leave the frame is not prepared, so it still holds
/// the last items it was given. A pick or a box select after such a frame
/// finds nothing of them.
#[test]
fn a_cpu_pick_ignores_item_types_left_out_of_the_frame() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    harness.renderer.set_cpu_pick_cache(true);
    let scenes = catalogue();
    let scene = scenes
        .iter()
        .find(|s| s.name == "ribbons")
        .expect("ribbons is in the catalogue");
    let built = harness.build_scene(scene);
    let frame = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    let _ = harness.render(&frame, W, H);

    let device = harness.device.clone();
    let queue = harness.queue.clone();
    let pick = |harness: &mut Harness, frame: &viewport_lib::FrameData, at: glam::Vec2| {
        harness
            .renderer
            .pick_object(
                PickBackend::Cpu,
                at,
                frame,
                &device,
                &queue,
                PickMask::OBJECT,
            )
            .map(|hit| hit.id)
    };
    let (at, id) = (0..H)
        .step_by(2)
        .flat_map(|y| {
            (0..W)
                .step_by(2)
                .map(move |x| glam::Vec2::new(x as f32, y as f32))
        })
        .find_map(|at| pick(&mut harness, &frame, at).map(|id| (at, id)))
        .expect("a ribbon is pickable");

    let mut empty = frame_for(&built, &scene.cameras[0].camera, [W as f32, H as f32]);
    empty.scene.plugin_items.clear();
    let _ = harness.render(&empty, W, H);
    assert_eq!(
        pick(&mut harness, &empty, at),
        None,
        "ribbon {id} was left out"
    );
    let rect = harness.renderer.pick_rect(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32, H as f32),
        glam::Vec2::new(W as f32, H as f32),
        empty.camera.render_camera.view_proj(),
        PickMask::OBJECT,
    );
    assert!(rect.objects.is_empty(), "{:?}", rect.objects);
}
