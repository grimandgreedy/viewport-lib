//! The point cloud item type: the clouds it holds on the consumer's behalf,
//! plus picking (GPU pick-id, per-point sub-objects, CPU proximity picking,
//! rect select, and the pre-uploaded reference form).
//!
//! One file per item type, so a type's coverage travels with it.

use viewport_lib::gpu;
use viewport_lib::plugin_api::Uploads;

mod common;
use common::*;
use viewport_lib::plugin_api::{Handles, Sourced, Span, Writes};
use viewport_lib::{ColourSource, SizeSource};
use viewport_lib_item_types::channels::point_cloud as pc;
use viewport_lib_item_types::*;

#[test]
fn gpu_pick_point_cloud_resolves_point() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    // Three points spread along X. The centre point (index 1) sits at world
    // origin, under the cursor. CLOUD_POINT picking reads the forwarded
    // instance_index, which needs no device feature.
    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    cloud.size = SizeSource::Uniform(20.0);
    cloud.settings.pick_id = PickId(444);
    frame.scene.items_mut::<PointCloudItem>().push(cloud);

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
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    // Three fat points spread across the centre of the view.
    let mut pc = PointCloudItem::default();
    pc.positions = vec![[-1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]];
    pc.size = SizeSource::Uniform(24.0);
    pc.settings.pick_id = PickId(500);
    frame.scene.items_mut::<PointCloudItem>().push(pc);

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
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    cloud.size = SizeSource::Uniform(20.0);
    cloud.settings.pick_id = PickId(445);
    frame.scene.items_mut::<PointCloudItem>().push(cloud);

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
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]];
    cloud.size = SizeSource::Uniform(20.0);
    cloud.settings.pick_id = PickId(446);
    frame.scene.items_mut::<PointCloudItem>().push(cloud);

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
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    cloud.size = SizeSource::Uniform(20.0);
    let source = renderer.upload(&device, &queue, &cloud).unwrap();

    let mut item = PointCloudRefItem::new(source);
    item.settings.pick_id = PickId(447);
    frame.scene.items_mut::<PointCloudRefItem>().push(item);

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
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]];
    cloud.size = SizeSource::Uniform(20.0);
    let source = renderer.upload(&device, &queue, &cloud).unwrap();

    let mut item = PointCloudRefItem::new(source);
    item.settings.pick_id = PickId(448);
    item.settings.hidden = true;
    frame.scene.items_mut::<PointCloudRefItem>().push(item);

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

fn sample_point_cloud() -> PointCloudItem {
    let mut item = PointCloudItem::default();
    item.positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    item.size = SizeSource::Uniform(6.0);
    item
}

#[test]
fn an_uploaded_cloud_resolves_until_it_is_dropped() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer
        .upload(&device, &queue, &sample_point_cloud())
        .unwrap();
    assert!(
        renderer.resident_bytes().plugin_bytes > baseline,
        "an uploaded cloud counts toward the plugin working set"
    );
    assert!(
        renderer
            .replace(&device, &queue, id, &sample_point_cloud())
            .is_ok()
    );

    assert!(renderer.release(id));
    assert!(!renderer.release(id), "a handle drops once");
    assert!(
        renderer
            .replace(&device, &queue, id, &sample_point_cloud())
            .is_err(),
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
    let mut renderer = renderer_with_item_types(&device);

    let job = renderer
        .begin_upload(&device, &queue, sample_point_cloud())
        .unwrap();
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

    let id: PointCloudId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(renderer.release(id));
    assert!(matches!(
        Handles::<PointCloudId>::upload_result(&mut renderer, job),
        Err(viewport_lib::error::ViewportError::JobResultMissing { .. })
    ));
}

/// A stored upload resolves its colourmap once and keeps what it resolved, so
/// the built-in LUTs have to be resolvable by then. Pre-uploading at startup is
/// the whole point of the reference form, and an upload that ran before the
/// first frame used to bind the neutral fallback and stay grey for the life of
/// the handle. The ids and views now exist from construction.
#[test]
fn an_upload_before_the_first_frame_still_gets_a_real_colourmap() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let viridis = renderer
        .resources()
        .builtin_colourmap_id(viewport_lib::resources::BuiltinColourmap::Viridis);
    assert!(
        renderer.resources().colourmap_view(viridis).is_some(),
        "the built-in LUT views are resident before any frame has run"
    );

    let id = renderer
        .upload(&device, &queue, &sample_point_cloud())
        .unwrap();

    assert_eq!(
        viridis,
        renderer
            .resources()
            .builtin_colourmap_id(viewport_lib::resources::BuiltinColourmap::Viridis),
        "the upload resolved the same id a later frame would"
    );
    assert!(renderer.release(id));
}

/// A hidden item produces no draw data, so hiding one costs nothing beyond the
/// submission itself.
#[test]
fn hidden_items_produce_no_draw_data() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let mut visible = PointCloudItem::default();
    visible.positions = vec![[0.0, 0.0, 0.0]];
    let mut hidden = PointCloudItem::default();
    hidden.positions = vec![[1.0, 0.0, 0.0]];
    hidden.settings.hidden = true;
    frame
        .scene
        .items_mut::<PointCloudItem>()
        .extend([visible, hidden]);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let plugin = renderer
        .item_type_plugin::<PointCloudPlugin>(POINT_CLOUD_TYPE_NAME)
        .expect("registered by install()");
    assert_eq!(
        plugin.drawn_count(),
        1,
        "hidden item must not produce gpu data"
    );
}

// ---------------------------------------------------------------------------
// Ranged writes
// ---------------------------------------------------------------------------

/// The streaming shape the write surface exists for: upload once with headroom,
/// then write sectors into it and let the draw count follow, with no
/// reallocation and no replace.
#[test]
fn a_reserved_cloud_takes_ranged_writes_without_reallocating() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]; 1_000];
    cloud.colour = ColourSource::Scalar {
        values: vec![0.0; 1_000],
        range: Some((0.0, 1.0)),
        colourmap: None,
    };

    let id = renderer
        .upload(&device, &queue, &cloud)
        .expect("upload a cloud");
    renderer
        .reserve(pc::Positions, &device, &queue, id, 10_000)
        .expect("reserve room for the feed");

    let extent = renderer
        .extent(pc::Positions, id)
        .expect("a live handle has an extent");
    assert!(extent.capacity >= 10_000);
    assert_eq!(extent.len, 1_000, "the reserve adds headroom, not points");

    // Fill the headroom one sector at a time. Each write raises the live count.
    let sector: Vec<[f32; 3]> = (0..1_000).map(|i| [i as f32, 1.0, 2.0]).collect();
    let scalars: Vec<f32> = (0..1_000).map(|i| i as f32 / 1_000.0).collect();
    for s in 1..10 {
        let first = s * 1_000;
        renderer
            .write_range(pc::Positions, &queue, id, first, &sector)
            .expect("the window fits the reserve");
        renderer
            .write_range(pc::Scalars, &queue, id, first, &scalars)
            .expect("the scalar channel was uploaded populated");
    }
    assert_eq!(renderer.extent(pc::Positions, id).unwrap().len, 10_000);

    // And the cloud still draws: one reference item, prepared.
    let mut frame = sub_object_pick_frame();
    frame
        .scene
        .items_mut::<PointCloudRefItem>()
        .push(PointCloudRefItem::new(id));
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    assert!(
        renderer.release(id),
        "the handle survived every write and still releases"
    );
}

/// Several disjoint sectors in one call, which is the shape a feed with more
/// than one dirty region has.
#[test]
fn write_spans_covers_disjoint_sectors() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]; 64];
    let id = renderer
        .upload(&device, &queue, &cloud)
        .expect("upload a cloud");

    let a = [[1.0f32, 1.0, 1.0]; 4];
    let b = [[2.0f32, 2.0, 2.0]; 8];
    renderer
        .write_spans(
            pc::Positions,
            &queue,
            id,
            &[Span::new(0, &a), Span::new(40, &b)],
        )
        .expect("two runs inside the allocation");

    // A run past the end fails, and the trait makes no promise about the runs
    // before it: what it promises is that the call reports the failure.
    let err = renderer
        .write_spans(
            pc::Positions,
            &queue,
            id,
            &[Span::new(0, &a), Span::new(60, &b)],
        )
        .expect_err("[60..68) does not fit 64 points");
    assert!(
        matches!(
            err,
            viewport_lib::error::ViewportError::ContentBufferWriteOutOfRange { .. }
        ),
        "{err}"
    );
}

/// A released handle stops taking writes, the same way `replace` refuses one.
#[test]
fn a_stale_handle_is_refused_by_every_write_call() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]; 8];
    let id = renderer
        .upload(&device, &queue, &cloud)
        .expect("upload a cloud");
    assert!(renderer.release(id));

    assert!(renderer.extent(pc::Positions, id).is_none());
    assert!(
        renderer
            .write_range(pc::Positions, &queue, id, 0, &[[1.0, 2.0, 3.0]])
            .is_err()
    );
    assert!(
        renderer
            .reserve(pc::Positions, &device, &queue, id, 64)
            .is_err()
    );
    assert!(renderer.set_len(pc::Positions, id, 4).is_err());
}

/// Lowering the live count hides points without giving the allocation back,
/// which is what lets a shrinking feed grow again for free.
#[test]
fn set_len_hides_points_without_freeing_them() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]; 100];
    let id = renderer
        .upload(&device, &queue, &cloud)
        .expect("upload a cloud");

    renderer
        .set_len(pc::Positions, id, 10)
        .expect("lowering the live count");
    let extent = renderer.extent(pc::Positions, id).unwrap();
    assert_eq!(extent.len, 10);
    assert_eq!(extent.capacity, 100, "nothing was freed");

    assert!(
        renderer.set_len(pc::Positions, id, 101).is_err(),
        "past the capacity is an error rather than a grow"
    );
}

// ---------------------------------------------------------------------------
// Caller-owned channel sources
// ---------------------------------------------------------------------------

/// The other end of a partial update: a producer whose points are already on the
/// device hands its buffer over and no bytes move at all.
#[test]
fn a_channel_can_be_drawn_from_a_caller_owned_buffer() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]; 64];
    let id = renderer.upload(&device, &queue, &cloud).expect("upload");

    assert_eq!(renderer.has_source(pc::Positions, id), Some(false));

    let theirs = device.create_buffer(&gpu::BufferDescriptor {
        label: Some("consumer_positions"),
        size: 64 * 12,
        usage: gpu::BufferUsages::VERTEX | gpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    renderer
        .set_source(pc::Positions, &device, id, Some(theirs.clone()))
        .expect("a VERTEX buffer big enough for 64 points");
    assert_eq!(renderer.has_source(pc::Positions, id), Some(true));

    // It still draws, through their buffer.
    let mut frame = sub_object_pick_frame();
    frame
        .scene
        .items_mut::<PointCloudRefItem>()
        .push(PointCloudRefItem::new(id));
    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // And handing it back restores the cloud's own storage without an upload.
    renderer
        .set_source(pc::Positions, &device, id, None)
        .expect("handing it back");
    assert_eq!(renderer.has_source(pc::Positions, id), Some(false));
    let _ = renderer.pass().prepare(&device, &queue, &frame);
}

/// A source is checked when it is set, not at draw time: a wgpu validation
/// failure on a binding takes the device down and a returned error does not.
#[test]
fn a_source_is_refused_for_the_wrong_usage_or_size() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]; 64];
    let id = renderer.upload(&device, &queue, &cloud).expect("upload");

    let no_vertex_usage = device.create_buffer(&gpu::BufferDescriptor {
        label: Some("wrong_usage"),
        size: 64 * 12,
        usage: gpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let err = renderer
        .set_source(pc::Positions, &device, id, Some(no_vertex_usage))
        .expect_err("the positions are a vertex stream");
    assert!(
        matches!(
            err,
            viewport_lib::error::ViewportError::ExternalBufferUsageMissing { .. }
        ),
        "{err}"
    );

    let too_small = device.create_buffer(&gpu::BufferDescriptor {
        label: Some("too_small"),
        size: 16 * 12,
        usage: gpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    });
    let err = renderer
        .set_source(pc::Positions, &device, id, Some(too_small))
        .expect_err("16 points cannot feed a 64-point cloud");
    assert!(
        matches!(
            err,
            viewport_lib::error::ViewportError::ContentBufferWriteOutOfRange { .. }
        ),
        "{err}"
    );

    // A channel the upload never populated has nothing to re-point.
    let err = renderer
        .set_source(pc::Colours, &device, id, None)
        .expect_err("this cloud holds no colour channel");
    assert!(matches!(
        err,
        viewport_lib::error::ViewportError::ChannelNotPresent { .. }
    ));
}

/// Growing the cloud must not silently drop a caller-owned buffer back to the
/// cloud's own storage.
#[test]
fn a_source_survives_a_reserve() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let mut cloud = PointCloudItem::default();
    cloud.positions = vec![[0.0, 0.0, 0.0]; 8];
    cloud.transparencies = vec![0.5; 8];
    let id = renderer.upload(&device, &queue, &cloud).expect("upload");

    let theirs = device.create_buffer(&gpu::BufferDescriptor {
        label: Some("consumer_transparencies"),
        size: 8 * 4,
        usage: gpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    renderer
        .set_source(pc::Transparencies, &device, id, Some(theirs))
        .expect("set");
    renderer
        .reserve(pc::Positions, &device, &queue, id, 256)
        .expect("grow");
    assert_eq!(
        renderer.has_source(pc::Transparencies, id),
        Some(true),
        "a grow rebuilds the bind group and must rebuild it with the source"
    );
}
