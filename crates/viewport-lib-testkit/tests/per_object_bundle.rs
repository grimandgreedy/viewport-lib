//! The cached batch render bundle must draw exactly what the immediate batch
//! loops draw, for lean and fat batches alike. The first frame after renderer
//! creation misses the bundle (it was recorded against the fallback camera
//! bind group before the viewport slot existed), so frame 1 exercises the
//! immediate path and later frames the bundle replay; their images must match.

use viewport_lib::Colour;
use viewport_lib::{CameraFrame, FrameData, Material, SceneFrame, SceneRenderItem};
use viewport_lib_testkit::{Harness, orbit_camera};

#[test]
fn bundle_replay_matches_immediate_draws() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let data = viewport_lib::primitives::icosphere(4.0, 2);
    // One mesh per item, so every item is its own batch and the list is
    // long enough to be bundled. Every other item takes the fat path.
    let ids: Vec<_> = (0..100)
        .map(|_| {
            h.renderer
                .resources_mut()
                .upload_mesh_data(&h.device, &data)
                .expect("upload")
        })
        .collect();

    // 100 items in a grid, some overlapping in depth so wrong draw content
    // would show.
    let items: Vec<SceneRenderItem> = (0..100)
        .map(|i| {
            let mut it = SceneRenderItem::default();
            it.mesh_id = ids[i];
            let (x, y) = ((i % 10) as f32 * 7.0 - 31.5, (i / 10) as f32 * 7.0 - 31.5);
            it.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, (i % 3) as f32 * 3.0))
                .to_cols_array_2d();
            it.material = Material::from_colour(Colour::linear_rgb(
                0.3 + (i % 7) as f32 * 0.1,
                0.4,
                0.8 - (i % 5) as f32 * 0.1,
            ));
            it.settings.force_per_object = i % 2 == 0;
            it
        })
        .collect();

    let camera = orbit_camera(glam::Vec3::ZERO, 120.0, 0.6, 1.0);
    let mut fd = FrameData::new(
        CameraFrame::from_camera(&camera, [320.0, 240.0]),
        SceneFrame::from_surface_items(items),
    );
    // The bundle only covers the LDR path.
    fd.effects.display.mode = viewport_lib::PipelineMode::Direct;

    let first = h.render(&fd, 320, 240);
    let _ = h.render(&fd, 320, 240);
    let bundled = h.render(&fd, 320, 240);
    assert!(
        h.stats().per_object_bundle_cached,
        "test premise: a 100-batch scene should cache a bundle"
    );
    assert_eq!(
        first, bundled,
        "bundle replay must produce the same image as immediate batch draws"
    );
}
