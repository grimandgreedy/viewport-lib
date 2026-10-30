//! GPU picks that must answer whatever the renderer is doing: a blocking pick
//! in `RuntimeMode::Playback`, and a pick on one transform of a
//! `MeshInstanceItem` batch.

use viewport_lib::{
    CameraFrame, Colour, FrameData, Material, MeshId, MeshInstanceItem, PickBackend, PickId,
    PickMask, RuntimeMode, SceneFrame, SceneRenderItem, SubObjectRef, primitives,
};
use viewport_lib_testkit::{Harness, orbit_camera};

const W: u32 = 200;
const H: u32 = 150;

fn frame(scene: SceneFrame) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, 10.0, 0.0, 0.0);
    FrameData::new(
        CameraFrame::from_camera(&camera, [W as f32, H as f32]),
        scene,
    )
}

fn sphere(harness: &mut Harness) -> MeshId {
    harness
        .renderer
        .resources_mut()
        .upload_mesh_data(&harness.device, &primitives::sphere(0.6, 24, 12))
        .expect("sphere upload")
}

/// Where `world` lands on screen, in logical pixels.
fn screen_of(frame: &FrameData, world: glam::Vec3) -> glam::Vec2 {
    let ndc = frame.camera.render_camera.view_proj().project_point3(world);
    glam::Vec2::new(
        (ndc.x + 1.0) * 0.5 * W as f32,
        (1.0 - ndc.y) * 0.5 * H as f32,
    )
}

fn gpu_pick(
    harness: &mut Harness,
    frame: &FrameData,
    at: glam::Vec2,
    mask: PickMask,
) -> Option<viewport_lib::PickHit> {
    let device = harness.device.clone();
    let queue = harness.queue.clone();
    harness
        .renderer
        .pick_object(PickBackend::Gpu, at, frame, &device, &queue, mask)
}

/// A blocking pick is a one-shot request: every one of four consecutive
/// frames' picks hits the object under the cursor in Playback mode.
#[test]
fn blocking_picks_hit_on_every_playback_frame() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mesh = sphere(&mut harness);
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh;
    item.material = harness
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material upload");
    item.settings.pick_id = PickId(7);
    let frame = frame(SceneFrame::from_surface_items(vec![item]));
    let centre = screen_of(&frame, glam::Vec3::ZERO);

    harness.renderer.set_runtime_mode(RuntimeMode::Playback);
    for n in 0..4 {
        let _ = harness.render(&frame, W, H);
        let hit = gpu_pick(&mut harness, &frame, centre, PickMask::OBJECT);
        assert_eq!(
            hit.map(|h| h.id),
            Some(7),
            "Playback pick {n} missed the object under the cursor"
        );
        let hit = gpu_pick(&mut harness, &frame, centre, PickMask::OBJECT);
        assert_eq!(
            hit.map(|h| h.id),
            Some(7),
            "a second Playback pick on frame {n} missed"
        );
    }
}

/// Every transform of a mesh-instance batch is picked under the batch's id,
/// and the hit names the transform.
#[test]
fn a_mesh_instance_pick_names_the_transform() {
    let Some(mut harness) = Harness::new() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mesh = sphere(&mut harness);
    let positions = [
        glam::Vec3::new(-2.0, 0.0, 0.0),
        glam::Vec3::ZERO,
        glam::Vec3::new(2.0, 0.0, 0.0),
    ];
    let mut item = MeshInstanceItem::default();
    item.mesh_id = mesh;
    item.transforms = positions
        .iter()
        .map(|p| glam::Mat4::from_translation(*p).to_cols_array_2d())
        .collect();
    item.colours = vec![Colour::linear_rgb(0.8, 0.4, 0.2); positions.len()];
    item.settings.pick_id = PickId(31);
    let mut scene = SceneFrame::default();
    scene.mesh_instances.push(item);
    let frame = frame(scene);
    let _ = harness.render(&frame, W, H);

    let middle = screen_of(&frame, positions[1]);
    let hit = gpu_pick(
        &mut harness,
        &frame,
        middle,
        PickMask::OBJECT | PickMask::INSTANCE,
    )
    .expect("the middle instance is under the cursor");
    assert_eq!(hit.id, 31);
    assert_eq!(hit.sub_object, Some(SubObjectRef::Instance(1)));

    for (i, p) in positions.iter().enumerate() {
        let at = screen_of(&frame, *p);
        let hit = gpu_pick(&mut harness, &frame, at, PickMask::INSTANCE)
            .unwrap_or_else(|| panic!("instance {i} is under the cursor"));
        assert_eq!(hit.id, 31);
        assert_eq!(hit.sub_object, Some(SubObjectRef::Instance(i as u32)));
    }

    // Object level only: the batch is hit, with no sub-object.
    let hit = gpu_pick(&mut harness, &frame, middle, PickMask::OBJECT)
        .expect("the batch is under the cursor");
    assert_eq!(hit.id, 31);
    assert_eq!(hit.sub_object, None);
}
