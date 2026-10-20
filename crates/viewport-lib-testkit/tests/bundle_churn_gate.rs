//! Batch bundle churn gate: sustained batch-list churn must back the
//! bundle off to immediate draws (re-recording every frame doubles the
//! encode and leaks the dropped bundle in wgpu 27, gfx-rs/wgpu#8656), an
//! isolated change must still re-record immediately, and a set that
//! stabilises again must get the bundle back.

use viewport_lib::Colour;
use viewport_lib::{CameraFrame, FrameData, Material, PickId, SceneFrame, SceneRenderItem};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

/// One item per mesh, so one batch each. `hidden` names the item left out,
/// which changes the batch list.
fn items(
    mesh_ids: &[viewport_lib::MeshId],
    material: viewport_lib::MaterialId,
    hidden: Option<u32>,
) -> Vec<SceneRenderItem> {
    mesh_ids
        .iter()
        .enumerate()
        .map(|(i, &mesh_id)| {
            let i = i as u32;
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.material = material;
            let x = (i % 16) as f32 * 2.0;
            let y = (i / 16) as f32 * 2.0;
            item.model =
                glam::Mat4::from_translation(glam::Vec3::new(x, y, 0.0)).to_cols_array_2d();
            item.settings.pick_id = PickId(1 + u64::from(i));
            item.settings.hidden = hidden == Some(i);
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
    // Under GPU-driven culling the batches draw from the cull's indirect
    // args and the bundle holds none of them; without it, the bundle draws
    // every opaque batch.
    h.renderer.disable_gpu_driven_culling();
    let count = 100u32; // above the bundle's 64-draw minimum
    let data: viewport_lib::MeshData = meshes::stress_sphere(0.8, 2).into();
    let mesh_ids: Vec<_> = (0..count)
        .map(|_| {
            h.renderer
                .resources_mut()
                .upload_mesh_data(&h.device, &data)
                .expect("mesh upload")
        })
        .collect();
    let material = h
        .renderer
        .resources_mut()
        .upload_material(Material::from_colour(Colour::linear_rgb(0.6, 0.65, 0.7)))
        .expect("material upload");
    let camera = orbit_camera(glam::Vec3::new(16.0, 12.0, 0.0), 60.0, 0.6, 1.0);
    let size = [96.0f32, 64.0];
    let fd = |hidden: Option<u32>| {
        FrameData::new(
            CameraFrame::from_camera(&camera, size),
            SceneFrame::from_surface_items(items(&mesh_ids, material, hidden)),
        )
    };

    // Stable set: the bundle engages.
    for _ in 0..3 {
        let _ = h.render(&fd(None), 96, 64);
    }
    assert!(
        h.stats().per_object_bundle_cached,
        "stable set must be served by the bundle"
    );

    // Sustained churn: a different item hidden every frame, so the batch
    // list changes each time. The first change may still re-record; by the
    // third churn frame the gate must have backed off to immediate draws.
    let mut hidden = 0u32;
    for _ in 0..3 {
        hidden += 1;
        let _ = h.render(&fd(Some(hidden)), 96, 64);
    }
    assert!(
        !h.stats().per_object_bundle_cached,
        "sustained churn must back the bundle off"
    );

    // Stability again: after the re-arm stretch the bundle returns.
    let mut rearmed_at = None;
    for f in 0..40 {
        let _ = h.render(&fd(Some(hidden)), 96, 64);
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
    let _ = h.render(&fd(Some(hidden + 10)), 96, 64);
    assert!(
        h.stats().per_object_bundle_cached,
        "an isolated change must re-record immediately, not fall to immediate draws"
    );
}
