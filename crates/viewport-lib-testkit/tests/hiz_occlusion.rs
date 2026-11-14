//! HiZ occlusion cull: a box hidden behind a wall is culled and a box in front
//! of the wall is not. Guards the depth-direction contract between the HiZ
//! pyramid and the cull shader: comparing the wrong way culls everything or
//! nothing, and both show up here as a wrong visible count.

use glam::{Mat4, Vec3};
use viewport_lib::Colour;
use viewport_lib::primitives::cube;
use viewport_lib::wgpu;
use viewport_lib::{CameraFrame, FrameData, Material, MeshItem, SceneFrame};
use viewport_lib_testkit::Harness;
use viewport_lib_testkit::scenes::orbit_camera;

#[test]
fn box_behind_wall_is_occluded_and_box_in_front_is_not() {
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
    h.renderer.set_occlusion_culling(true);

    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &cube(1.0))
        .expect("cube upload");
    let grey = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.7, 0.7, 0.7)))
        .expect("material upload");

    // The eye sits on +X at distance 20 looking toward -X. The wall is thin in
    // X and covers the middle of the view; the background shows around it so
    // the wall never occludes itself.
    let placed = |m: Mat4| {
        let mut item = MeshItem::default();
        item.mesh_id = mesh_id;
        item.model = m.into();
        item.material = grey;
        item
    };
    let items = vec![
        // Wall.
        placed(Mat4::from_scale(Vec3::new(0.2, 8.0, 8.0))),
        // Behind the wall: hidden.
        placed(Mat4::from_translation(Vec3::new(-5.0, 0.0, 0.0))),
        // In front of the wall: visible.
        placed(Mat4::from_translation(Vec3::new(5.0, 1.0, 1.0))),
    ];
    let camera = orbit_camera(Vec3::ZERO, 20.0, 1.5708, 1.5708);
    let fd = FrameData::new(
        CameraFrame::from_camera(&camera, [320.0, 240.0]),
        SceneFrame::from_surface_items(items),
    );

    // The first frame stores depth, later frames reproject it and cull; the
    // counter readback lags a few frames behind.
    let mut stats = h.render_two_frames(&fd, 320, 240);
    for _ in 0..8 {
        let _ = h.render(&fd, 320, 240);
        stats = h.stats();
    }

    assert!(stats.gpu_culling_active, "GPU culling inactive");
    let total = stats.gpu_culled_total.expect("no total readback");
    let frustum = stats.gpu_frustum_visible.expect("no frustum readback");
    let visible = stats.gpu_visible_instances.expect("no visible readback");
    eprintln!("hiz occlusion counts: total={total} frustum={frustum} visible={visible}");
    assert_eq!(total, 3, "every instance enters the cull");
    assert_eq!(frustum, 3, "every instance is inside the frustum");
    assert_eq!(
        visible, 2,
        "the box behind the wall is occluded and the wall and the front box are not"
    );
}
