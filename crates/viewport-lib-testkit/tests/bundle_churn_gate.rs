//! Per-object bundle churn gate: sustained item-set churn must back the
//! bundle off to immediate draws (re-recording every frame doubles the
//! encode and leaks the dropped bundle in wgpu 27, gfx-rs/wgpu#8656), an
//! isolated change must still re-record immediately, and a set that
//! stabilises again must get the bundle back.

use viewport_lib::{CameraFrame, FrameData, Material, PickId, SceneFrame, SceneRenderItem};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

fn items(mesh_id: viewport_lib::MeshId, count: u32, id_base: u64) -> Vec<SceneRenderItem> {
    (0..count)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.material = Material::from_colour([0.6, 0.65, 0.7]);
            let x = (i % 16) as f32 * 2.0;
            let y = (i / 16) as f32 * 2.0;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(x, y, 0.0)).to_cols_array_2d();
            item.settings.pick_id = PickId(id_base + u64::from(i));
            // Styled backface policy forces the per-object path.
            item.material.backface_policy = viewport_lib::BackfacePolicy::Tint(0.5);
            item
        })
        .collect()
}

#[test]
fn bundle_backs_off_under_churn_and_rearms() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mesh_id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &meshes::stress_sphere(0.8, 2).into())
        .expect("mesh upload");

    let count = 100u32; // above the bundle's 64-item minimum
    let camera = orbit_camera(glam::Vec3::new(16.0, 12.0, 0.0), 60.0, 0.6, 1.0);
    let size = [96.0f32, 64.0];
    let fd = |id_base: u64| {
        FrameData::new(
            CameraFrame::from_camera(&camera, size),
            SceneFrame::from_surface_items(items(mesh_id, count, id_base)),
        )
    };

    // Stable set: the bundle engages.
    for _ in 0..3 {
        let _ = h.render(&fd(1), 96, 64);
    }
    assert!(
        h.stats().per_object_bundle_cached,
        "stable set must be served by the bundle"
    );

    // Sustained churn: fresh pick ids every frame. The first change may
    // still re-record; by the third churn frame the gate must have backed
    // off to immediate draws.
    let mut id_base = 1000u64;
    for _ in 0..3 {
        id_base += u64::from(count);
        let _ = h.render(&fd(id_base), 96, 64);
    }
    assert!(
        !h.stats().per_object_bundle_cached,
        "sustained churn must back the bundle off"
    );

    // Stability again: after the re-arm stretch the bundle returns.
    let mut rearmed_at = None;
    for f in 0..40 {
        let _ = h.render(&fd(id_base), 96, 64);
        if h.stats().per_object_bundle_cached {
            rearmed_at = Some(f);
            break;
        }
    }
    let rearmed_at = rearmed_at.expect("bundle must re-arm once the set is stable again");
    assert!(
        rearmed_at >= 25,
        "re-arm should wait out the stability stretch (re-armed after {rearmed_at} frames)"
    );

    // Isolated change on a stable set: the bundle re-records immediately,
    // so the change frame is still served by a bundle.
    let _ = h.render(&fd(500_000), 96, 64);
    assert!(
        h.stats().per_object_bundle_cached,
        "an isolated change must re-record immediately, not fall to immediate draws"
    );
}
