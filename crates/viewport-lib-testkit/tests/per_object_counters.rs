//! `FrameStats::draw_calls` and `triangles_submitted` cover every batch. Four
//! items on one mesh share one batch: one draw of four instances.

use viewport_lib::Colour;
use viewport_lib::{CameraFrame, FrameData, Material, SceneFrame, SceneRenderItem};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

#[test]
fn batch_draws_are_counted() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let data = meshes::heightfield(16, 16, 4.0, 0.5);
    let tris_per_item = (data.indices.len() / 3) as u64;
    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &data)
        .expect("upload");

    let material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.6, 0.6, 0.7)))
        .expect("material upload");
    let count = 4usize;
    let items: Vec<SceneRenderItem> = (0..count)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = id;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(i as f32 * 5.0, 0.0, 0.0))
                .to_cols_array_2d();
            item.material = material;
            item
        })
        .collect();

    let camera = orbit_camera(glam::Vec3::new(7.5, 0.0, 0.0), 30.0, 0.6, 1.0);
    let fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(items),
    );
    let stats = h.render_two_frames(&fd, 200, 150);

    assert_eq!(stats.instanced_batches, 1, "one batch of four instances");
    assert_eq!(stats.draw_calls, 1);
    assert_eq!(stats.triangles_submitted, tris_per_item * count as u64);
}
