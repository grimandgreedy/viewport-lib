//! `FrameStats::draw_calls` and `triangles_submitted` must cover the
//! per-object render path, not just the instanced one. Items with a styled
//! backface policy miss the instanced fast path and used to leave both
//! counters at 0 even while issuing one draw per item.

use viewport_lib::{BackfacePolicy, CameraFrame, FrameData, Material, SceneFrame, SceneRenderItem};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

#[test]
fn per_object_draws_are_counted() {
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

    let count = 4usize;
    let items: Vec<SceneRenderItem> = (0..count)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = id;
            item.model = glam::Mat4::from_translation(glam::Vec3::new(i as f32 * 5.0, 0.0, 0.0))
                .to_cols_array_2d();
            let mut m = Material::from_colour([0.6, 0.6, 0.7]);
            // A styled backface policy keeps the item off the instanced path.
            m.backface_policy = BackfacePolicy::Tint(0.4);
            item.material = m;
            item
        })
        .collect();

    let camera = orbit_camera(glam::Vec3::new(7.5, 0.0, 0.0), 30.0, 0.6, 1.0);
    let fd = FrameData::new(
        CameraFrame::from_camera(&camera, [200.0, 150.0]),
        SceneFrame::from_surface_items(items),
    );
    let stats = h.render_two_frames(&fd, 200, 150);

    assert_eq!(stats.per_object_items, count as u32, "test premise");
    assert_eq!(stats.instanced_batches, 0, "test premise");
    assert_eq!(stats.draw_calls, count as u32);
    assert_eq!(stats.triangles_submitted, tris_per_item * count as u64);
}
