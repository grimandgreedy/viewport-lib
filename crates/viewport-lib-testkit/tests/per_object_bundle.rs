//! The cached per-object render bundle must draw exactly what the immediate
//! per-object loop draws. The first frame after renderer creation misses the
//! bundle (it was recorded against the fallback camera bind group before the
//! viewport slot existed), so frame 1 exercises the immediate path and later
//! frames the bundle replay; their images must match.

use viewport_lib::{BackfacePolicy, CameraFrame, FrameData, Material, SceneFrame, SceneRenderItem};
use viewport_lib_testkit::{Harness, orbit_camera};

#[test]
fn bundle_replay_matches_immediate_draws() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let data = viewport_lib::primitives::icosphere(4.0, 2);
    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &data)
        .expect("upload");

    // 100 unbatchable items (styled backface forces the per-object path) in a
    // grid, some overlapping in depth so wrong draw content would show.
    let items: Vec<SceneRenderItem> = (0..100)
        .map(|i| {
            let mut it = SceneRenderItem::default();
            it.mesh_id = id;
            let (x, y) = ((i % 10) as f32 * 7.0 - 31.5, (i / 10) as f32 * 7.0 - 31.5);
            it.model = glam::Mat4::from_translation(glam::Vec3::new(x, y, (i % 3) as f32 * 3.0))
                .to_cols_array_2d();
            let mut m = Material::from_colour([
                0.3 + (i % 7) as f32 * 0.1,
                0.4,
                0.8 - (i % 5) as f32 * 0.1,
            ]);
            m.backface_policy = BackfacePolicy::Tint(0.4);
            it.material = m;
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
        "test premise: a 100-item per-object scene should cache a bundle"
    );
    assert_eq!(
        first, bundled,
        "bundle replay must produce the same image as immediate per-object draws"
    );
}
