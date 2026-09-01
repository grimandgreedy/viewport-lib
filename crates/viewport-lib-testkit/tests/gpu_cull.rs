//! GPU frustum cull effectiveness: a line of cubes with the camera in the
//! middle of the line must cull the cubes behind the eye. Guards against
//! the cull pass silently passing everything.

use glam::Vec3;
use viewport_lib::primitives::cube;
use viewport_lib::wgpu;
use viewport_lib::{CameraFrame, FrameData, Material, SceneFrame, SceneRenderItem};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::scenes::orbit_camera;

#[test]
fn frustum_cull_rejects_behind_camera() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    if !h
        .device
        .features()
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        eprintln!("skipping: no INDIRECT_FIRST_INSTANCE");
        return;
    }

    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &cube(1.0))
        .expect("cube upload");

    // 20 cubes along x = -47.5 .. 47.5, all in one instanced batch.
    let items: Vec<SceneRenderItem> = (0..20)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = glam::Mat4::from_translation(Vec3::new(-47.5 + 5.0 * i as f32, 0.0, 0.0))
                .to_cols_array_2d();
            item.material = Material::from_colour([0.7, 0.7, 0.7]);
            item
        })
        .collect();

    // Orbit convention: the eye sits at centre + distance on the +X side
    // (yaw pi/2, pitch pi/2 = horizontal) looking toward -X. Centre at
    // (-30, 0, 0) puts the eye at the origin, so the 9+ cubes at x > 0 sit
    // behind the eye and must frustum-cull.
    let camera = orbit_camera(Vec3::new(-30.0, 0.0, 0.0), 30.0, 1.5708, 1.5708);
    let fd = FrameData::new(
        CameraFrame::from_camera(&camera, [320.0, 240.0]),
        SceneFrame::from_surface_items(items),
    );

    // The visible-count readback lands a few frames later.
    let mut stats = h.render_two_frames(&fd, 320, 240);
    for _ in 0..6 {
        let _ = h.render(&fd, 320, 240);
        stats = h.stats();
        if stats.gpu_visible_instances.is_some() {
            break;
        }
    }

    assert!(
        stats.gpu_culling_active,
        "GPU culling inactive despite INDIRECT_FIRST_INSTANCE"
    );
    let visible = stats
        .gpu_visible_instances
        .expect("no gpu_visible_instances readback after 8 frames");
    assert!(
        visible < 20,
        "frustum cull rejected nothing: {visible} of 20 instances visible \
         with half the line behind the camera"
    );
    assert!(
        visible >= 10,
        "frustum cull rejected too much: {visible} of 20 visible"
    );
}
