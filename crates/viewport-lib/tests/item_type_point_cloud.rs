//! The point cloud item type: the clouds it holds on the consumer's behalf,
//! plus picking (GPU pick-id, per-point sub-objects, CPU proximity picking,
//! rect select, and the pre-uploaded reference form).
//!
//! One file per item type, so a type's coverage travels with it.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

#[test]
fn gpu_pick_point_cloud_resolves_point() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // Three points spread along X. The centre point (index 1) sits at world
    // origin, under the cursor. CLOUD_POINT picking reads the forwarded
    // instance_index, which needs no device feature.
    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    cloud.point_size = 20.0;
    cloud.settings.pick_id = PickId(444);
    frame.scene.point_clouds.push(cloud);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::CLOUD_POINT,
    );
    let hit = hit.expect("centre point should be hit");
    assert_eq!(hit.id, 444);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
    // The snap position comes back from the point cloud itself, which indexes
    // its own positions: point 1 is at the origin, not merely somewhere on the
    // splat the ray struck.
    let snap = hit
        .sub_object_world_pos
        .expect("point pick should fill the snap position");
    assert!(
        snap.length() < 1e-4,
        "point 1 should be at the origin, got {snap:?}"
    );
}

#[test]
fn gpu_pick_rect_resolves_point_cloud_elements() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // Three fat points spread across the centre of the view.
    let mut pc = PointCloudItem::default();
    pc.positions = vec![[-1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]];
    pc.point_size = 24.0;
    pc.settings.pick_id = PickId(500);
    frame.scene.point_clouds.push(pc);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // A CLOUD_POINT rect over the whole frame collects point sub-objects and no
    // objects (the mask carries no OBJECT bit). Point sub-objects come from the
    // instance index, so this needs no device feature.
    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::CLOUD_POINT,
    );
    assert!(
        result.objects.is_empty(),
        "CLOUD_POINT mask carries no OBJECT bit"
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect point sub-objects"
    );
    assert!(
        result
            .elements
            .iter()
            .all(|(id, sub)| *id == 500 && matches!(sub, viewport_lib::SubObjectRef::Point(_))),
        "every element must be a point of the cloud"
    );
}

#[test]
fn cpu_pick_hits_point_cloud() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    cloud.point_size = 20.0;
    cloud.settings.pick_id = PickId(445);
    frame.scene.point_clouds.push(cloud);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Cpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::CLOUD_POINT,
    );
    let hit = hit.expect("centre point should be hit");
    assert_eq!(hit.id, 445);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
}

/// An `OBJECT`-only query answers with the object and no sub-object, on both
/// backends.
#[test]
fn an_object_query_drops_the_point_sub_object() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]];
    cloud.point_size = 20.0;
    cloud.settings.pick_id = PickId(446);
    frame.scene.point_clouds.push(cloud);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    for backend in [PickBackend::Gpu, PickBackend::Cpu] {
        let hit = renderer
            .pick_object(
                backend,
                glam::Vec2::new(32.0, 32.0),
                &frame,
                &device,
                &queue,
                PickMask::OBJECT,
            )
            .expect("point should be hit");
        assert_eq!(hit.id, 446);
        assert_eq!(hit.sub_object, None, "{backend:?} must not report a point");
    }
}

/// A cloud uploaded once and drawn through `PointCloudRefItem` picks the same
/// as an inline item.
#[test]
fn a_reference_item_picks_like_an_inline_one() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    cloud.point_size = 20.0;
    let source = renderer.upload_point_cloud(&device, &queue, &cloud);

    let mut item = viewport_lib::PointCloudRefItem::new(source);
    item.settings.pick_id = PickId(447);
    frame.scene.point_cloud_refs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::CLOUD_POINT,
    );
    let hit = hit.expect("centre point of the referenced cloud should be hit");
    assert_eq!(hit.id, 447);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
}

/// A hidden reference item draws nothing and picks nothing.
#[test]
fn a_hidden_reference_item_is_skipped() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]];
    cloud.point_size = 20.0;
    let source = renderer.upload_point_cloud(&device, &queue, &cloud);

    let mut item = viewport_lib::PointCloudRefItem::new(source);
    item.settings.pick_id = PickId(448);
    item.settings.hidden = true;
    frame.scene.point_cloud_refs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), None);
}

// ---------------------------------------------------------------------------
// The clouds the item type holds
// ---------------------------------------------------------------------------

fn sample_point_cloud() -> viewport_lib::renderer::PointCloudItem {
    let mut item = viewport_lib::renderer::PointCloudItem::default();
    item.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    item.point_size = 6.0;
    item
}

#[test]
fn an_uploaded_cloud_resolves_until_it_is_dropped() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer.upload_point_cloud(&device, &queue, &sample_point_cloud());
    assert!(
        renderer.resident_bytes().plugin_bytes > baseline,
        "an uploaded cloud counts toward the plugin working set"
    );
    assert!(renderer.replace_point_cloud(&device, &queue, id, &sample_point_cloud()));

    assert!(renderer.drop_point_cloud(id));
    assert!(!renderer.drop_point_cloud(id), "a handle drops once");
    assert!(
        !renderer.replace_point_cloud(&device, &queue, id, &sample_point_cloud()),
        "a dropped handle must not resolve"
    );
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_point_cloud_drains_to_a_handle() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let job = renderer.begin_upload_point_cloud(&device, &queue, sample_point_cloud());
    for _ in 0..200 {
        renderer.resources_mut().process_uploads(&device, &queue);
        match renderer.upload_status(job) {
            viewport_lib::resources::UploadStatus::Ready => break,
            viewport_lib::resources::UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
            viewport_lib::resources::UploadStatus::Pending { .. } => {
                std::thread::sleep(std::time::Duration::from_millis(5))
            }
            viewport_lib::resources::UploadStatus::Unknown => panic!("job id disappeared"),
        }
    }

    let id = renderer
        .upload_result_point_cloud(job)
        .expect("the finished job yields a handle");
    assert!(renderer.drop_point_cloud(id));
    assert!(matches!(
        renderer.upload_result_point_cloud(job),
        Err(viewport_lib::error::ViewportError::JobResultMissing { .. })
    ));
}

/// A stored upload resolves its colourmap once and keeps what it resolved, so
/// the built-in LUTs have to be resident by then. They are otherwise uploaded
/// on the first `prepare`, and pre-uploading at startup is the whole point of
/// the reference form: an upload that ran first would bind the neutral
/// fallback and stay grey for the life of the handle.
#[test]
fn an_upload_before_the_first_frame_still_gets_a_real_colourmap() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    assert!(
        renderer
            .resources()
            .builtin_colourmap_id_checked(viewport_lib::resources::BuiltinColourmap::Viridis)
            .is_none(),
        "nothing has uploaded the built-in set yet"
    );

    let id = renderer.upload_point_cloud(&device, &queue, &sample_point_cloud());

    assert!(
        renderer
            .resources()
            .builtin_colourmap_id_checked(viewport_lib::resources::BuiltinColourmap::Viridis)
            .is_some(),
        "the upload made the built-in LUTs resident before resolving one"
    );
    assert!(renderer.drop_point_cloud(id));
}
