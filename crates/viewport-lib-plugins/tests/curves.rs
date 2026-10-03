//! Picking and draw state for the three curve mesh item types: streamtube,
//! tube and ribbon. Covers GPU pick-id, the segment / strip / node
//! sub-objects, CPU proximity picking, rect select, and the pre-uploaded
//! reference form.
//!
//! One file per item type family, so a type's coverage travels with it.

#![cfg(feature = "item-types")]

mod common;
use common::*;
use viewport_lib::plugin_api::Handles;

use viewport_lib::SubObjectRef;
use viewport_lib::plugin_api::Uploads;
use viewport_lib_plugins::item_types::curves::{
    RIBBON_TYPE_NAME, RibbonItem, RibbonPlugin, RibbonRefItem, STREAMTUBE_TYPE_NAME,
    StreamtubeItem, StreamtubePlugin, StreamtubeRefItem, TUBE_TYPE_NAME, TubeItem, TubePlugin,
    TubeRefItem,
};

/// A 64x64 frame with the default orbit camera and no overlays, so the world
/// origin projects to the screen centre (32, 32).
fn curve_frame() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame
}

/// Three control points along X with the middle one at the origin, the shape
/// every test in this file sweeps.
fn spine() -> (Vec<[f32; 3]>, Vec<u32>) {
    (
        vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![3],
    )
}

/// One test renders at a time. The warm-up tests read the process-wide build
/// log, which the other tests' fresh renderers would write into.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn gpu_pick_hits_ribbon() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // A wide ribbon centred on the origin. Ribbons build an owned connected mesh
    // into ribbon_gpu_data during prepare(), so the pick pass only sees it after
    // the prepare path has run.
    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::RibbonItem>()
        .push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(4242)));
}

#[test]
fn gpu_pick_ribbon_resolves_segment_and_node() {
    let _serial = serial();
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    // Segment and node resolve from the pick shader variants; no CPU pick cache.
    let mut frame = sub_object_pick_frame();

    // A wide ribbon centred on the origin: 3 control points along X, so two
    // segments with the middle node under the centre pixel.
    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::RibbonItem>()
        .push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // SEGMENT: the hit resolves to one of the two segments.
    let seg = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::SEGMENT,
        )
        .expect("ribbon should be hit");
    assert_eq!(seg.id, 4242);
    match seg.sub_object {
        Some(viewport_lib::SubObjectRef::Segment(s)) => assert!(s < 2, "segment {s} out of range"),
        other => panic!("expected a Segment sub-object, got {other:?}"),
    }

    // POLY_NODE: a centre click resolves to the middle control point (index 1),
    // the nearer endpoint of whichever segment the ray landed on.
    let node = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::POLY_NODE,
        )
        .expect("ribbon should be hit");
    assert_eq!(node.id, 4242);
    assert_eq!(node.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
}

#[test]
fn gpu_pick_curve_node_fills_snap_world_pos() {
    let _serial = serial();
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::RibbonItem>()
        .push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::POLY_NODE,
        )
        .expect("ribbon should be hit");
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
    // Node 1 sits at the world origin; the snap position must be that node.
    let snap = hit
        .sub_object_world_pos
        .expect("node pick should fill the snap position");
    assert!(
        snap.length() < 1e-4,
        "node 1 should be at the origin, got {snap:?}"
    );
}

#[test]
fn gpu_pick_rect_resolves_curve_node() {
    let _serial = serial();
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    // No CPU pick cache: the POLY_NODE variant writes the nearest node index per
    // pixel, so a rect reads it straight from the primitive channel.
    let mut frame = sub_object_pick_frame();

    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::RibbonItem>()
        .push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::POLY_NODE,
    );
    assert!(
        result.objects.is_empty(),
        "POLY_NODE mask carries no OBJECT bit"
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect node sub-objects"
    );
    assert!(
        result.elements.iter().all(|(id, sub)| *id == 4242
            && matches!(sub, viewport_lib::SubObjectRef::Point(n) if *n < 3)),
        "every element must be a ribbon node, got {:?}",
        result.elements
    );
    // The middle node (index 1) sits under the centre of the rect, so it must be
    // among the collected nodes.
    assert!(
        result
            .elements
            .iter()
            .any(|(_, sub)| *sub == viewport_lib::SubObjectRef::Point(1)),
        "the middle node should be collected, got {:?}",
        result.elements
    );
}

// ---------------------------------------------------------------------------
// GPU pick: the other two curve types
// ---------------------------------------------------------------------------

#[test]
fn gpu_pick_hits_streamtube() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = StreamtubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(5151);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::StreamtubeItem>()
        .push(tube);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(5151)));
}

#[test]
fn gpu_pick_hits_tube() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = TubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(5252);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::TubeItem>()
        .push(tube);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(5252)));
}

/// STRIP folds a hit on any segment back to the strip that owns it, taking
/// priority over SEGMENT when both are asked for.
#[test]
fn gpu_pick_streamtube_resolves_strip_without_cpu_cache() {
    let _serial = serial();
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = StreamtubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(5151);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::StreamtubeItem>()
        .push(tube);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::STRIP | PickMask::SEGMENT,
        )
        .expect("streamtube should be hit");
    assert_eq!(hit.id, 5151);
    assert_eq!(hit.sub_object, Some(SubObjectRef::Strip(0)));
}

// ---------------------------------------------------------------------------
// CPU pick
// ---------------------------------------------------------------------------

/// The CPU path answers the same levels as the GPU one, from the item geometry
/// the plugin retained at prepare.
#[test]
fn cpu_pick_tube_resolves_segment() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = TubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(5252);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::TubeItem>()
        .push(tube);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Cpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::SEGMENT,
        )
        .expect("tube should be hit");
    assert_eq!(hit.id, 5252);
    match hit.sub_object {
        Some(SubObjectRef::Segment(s)) => assert!(s < 2, "segment {s} out of range"),
        other => panic!("expected a Segment sub-object, got {other:?}"),
    }
}

#[test]
fn rect_pick_collects_ribbon_segments() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut ribbon = RibbonItem::default();
    ribbon.positions = positions;
    ribbon.strip_lengths = strip_lengths;
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::RibbonItem>()
        .push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let result = renderer.pick_rect_objects(
        PickBackend::Cpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::SEGMENT,
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect ribbon segments"
    );
    assert!(
        result
            .elements
            .iter()
            .all(|(id, sub)| *id == 4242 && matches!(sub, SubObjectRef::Segment(s) if *s < 2)),
        "every element must be a ribbon segment, got {:?}",
        result.elements
    );
}

// ---------------------------------------------------------------------------
// The pre-uploaded reference form
// ---------------------------------------------------------------------------

/// A reference item picks under the id on the reference, not the one the
/// upload was made with: the same stored geometry can be drawn twice under two
/// ids.
#[test]
fn a_reference_streamtube_picks_under_its_own_id() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = StreamtubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(1);
    let source = renderer.upload(&device, &queue, &tube).unwrap();

    let mut reference = StreamtubeRefItem::new(source);
    reference.settings.pick_id = PickId(7171);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::StreamtubeRefItem>()
        .push(reference);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(7171)));
}

#[test]
fn a_hidden_reference_tube_is_skipped() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = TubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    let source = renderer.upload(&device, &queue, &tube).unwrap();

    let mut reference = TubeRefItem::new(source);
    reference.settings.pick_id = PickId(7272);
    reference.settings.hidden = true;
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::TubeRefItem>()
        .push(reference);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), None);
}

#[test]
fn a_reference_ribbon_draws_and_picks() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut ribbon = RibbonItem::default();
    ribbon.positions = positions;
    ribbon.strip_lengths = strip_lengths;
    ribbon.width = 2.0;
    let source = renderer.upload(&device, &queue, &ribbon).unwrap();

    let mut reference = RibbonRefItem::new(source);
    reference.settings.pick_id = PickId(7373);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::curves::RibbonRefItem>()
        .push(reference);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(7373)));
}

// ---------------------------------------------------------------------------
// The streamtubes the item type holds
// ---------------------------------------------------------------------------

#[test]
fn an_uploaded_streamtube_resolves_until_it_is_dropped() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let item = {
        let mut item = viewport_lib_plugins::item_types::curves::StreamtubeItem::default();
        item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        item.strip_lengths = vec![3];
        item.radius = 0.2;
        item
    };
    let id = renderer.upload(&device, &queue, &item).unwrap();
    assert!(
        renderer.resident_bytes().plugin_bytes > baseline,
        "an uploaded streamtube counts toward the plugin working set"
    );
    assert!(renderer.replace(&device, &queue, id, &item).is_ok());

    // A dropped handle stops resolving, and the freed slot comes back at a new
    // generation so it cannot alias its successor.
    assert!(renderer.release(id));
    assert!(!renderer.release(id), "a handle drops once");
    let reused = renderer.upload(&device, &queue, &item).unwrap();
    assert_ne!(id, reused, "the reused slot carries a new generation");
    assert!(renderer.replace(&device, &queue, id, &item).is_err());
    assert!(renderer.release(reused));
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_streamtube_drains_to_a_handle() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let job = renderer
        .begin_upload(&device, &queue, {
            let mut item = viewport_lib_plugins::item_types::curves::StreamtubeItem::default();
            item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
            item.strip_lengths = vec![3];
            item.radius = 0.2;
            item
        })
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
    let id: viewport_lib_plugins::item_types::curves::StreamtubeId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(renderer.release(id));
}

// ---------------------------------------------------------------------------
// The tubes the item type holds
// ---------------------------------------------------------------------------

#[test]
fn an_uploaded_tube_resolves_until_it_is_dropped() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let item = {
        let mut item = viewport_lib_plugins::item_types::curves::TubeItem::default();
        item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        item.strip_lengths = vec![3];
        item.radius = 0.2;
        item
    };
    let id = renderer.upload(&device, &queue, &item).unwrap();
    assert!(
        renderer.resident_bytes().plugin_bytes > baseline,
        "an uploaded tube counts toward the plugin working set"
    );
    assert!(renderer.replace(&device, &queue, id, &item).is_ok());

    // A dropped handle stops resolving, and the freed slot comes back at a new
    // generation so it cannot alias its successor.
    assert!(renderer.release(id));
    assert!(!renderer.release(id), "a handle drops once");
    let reused = renderer.upload(&device, &queue, &item).unwrap();
    assert_ne!(id, reused, "the reused slot carries a new generation");
    assert!(renderer.replace(&device, &queue, id, &item).is_err());
    assert!(renderer.release(reused));
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_tube_drains_to_a_handle() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let job = renderer
        .begin_upload(&device, &queue, {
            let mut item = viewport_lib_plugins::item_types::curves::TubeItem::default();
            item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
            item.strip_lengths = vec![3];
            item.radius = 0.2;
            item
        })
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
    let id: viewport_lib_plugins::item_types::curves::TubeId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(renderer.release(id));
}

// ---------------------------------------------------------------------------
// The ribbons the item type holds
// ---------------------------------------------------------------------------

#[test]
fn an_uploaded_ribbon_resolves_until_it_is_dropped() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let item = {
        let mut item = viewport_lib_plugins::item_types::curves::RibbonItem::default();
        item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        item.strip_lengths = vec![3];
        item.width = 0.3;
        item
    };
    let id = renderer.upload(&device, &queue, &item).unwrap();
    assert!(
        renderer.resident_bytes().plugin_bytes > baseline,
        "an uploaded ribbon counts toward the plugin working set"
    );
    assert!(renderer.replace(&device, &queue, id, &item).is_ok());

    // A dropped handle stops resolving, and the freed slot comes back at a new
    // generation so it cannot alias its successor.
    assert!(renderer.release(id));
    assert!(!renderer.release(id), "a handle drops once");
    let reused = renderer.upload(&device, &queue, &item).unwrap();
    assert_ne!(id, reused, "the reused slot carries a new generation");
    assert!(renderer.replace(&device, &queue, id, &item).is_err());
    assert!(renderer.release(reused));
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_ribbon_drains_to_a_handle() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let job = renderer
        .begin_upload(&device, &queue, {
            let mut item = viewport_lib_plugins::item_types::curves::RibbonItem::default();
            item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
            item.strip_lengths = vec![3];
            item.width = 0.3;
            item
        })
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
    let id: viewport_lib_plugins::item_types::curves::RibbonId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(renderer.release(id));
}

/// A hidden item produces no draw data, for each of the three curve types.
#[test]
fn hidden_items_produce_no_draw_data() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    viewport_lib_plugins::item_types::install(&mut renderer, &device);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();

    let mut vis_tube = TubeItem::default();
    vis_tube.positions = positions.clone();
    vis_tube.strip_lengths = strip_lengths.clone();
    let mut hid_tube = vis_tube.clone();
    hid_tube.settings.hidden = true;
    frame
        .scene
        .items_mut::<TubeItem>()
        .extend([vis_tube, hid_tube]);

    let mut vis_st = StreamtubeItem::default();
    vis_st.positions = positions.clone();
    vis_st.strip_lengths = strip_lengths.clone();
    let mut hid_st = vis_st.clone();
    hid_st.settings.hidden = true;
    frame
        .scene
        .items_mut::<StreamtubeItem>()
        .extend([vis_st, hid_st]);

    let mut vis_rib = RibbonItem::default();
    vis_rib.positions = positions;
    vis_rib.strip_lengths = strip_lengths;
    let mut hid_rib = vis_rib.clone();
    hid_rib.settings.hidden = true;
    frame
        .scene
        .items_mut::<RibbonItem>()
        .extend([vis_rib, hid_rib]);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    for (name, count) in [
        (
            "tube",
            renderer
                .item_type_plugin::<TubePlugin>(TUBE_TYPE_NAME)
                .expect("registered by install()")
                .drawn_count(),
        ),
        (
            "streamtube",
            renderer
                .item_type_plugin::<StreamtubePlugin>(STREAMTUBE_TYPE_NAME)
                .expect("registered by install()")
                .drawn_count(),
        ),
        (
            "ribbon",
            renderer
                .item_type_plugin::<RibbonPlugin>(RIBBON_TYPE_NAME)
                .expect("registered by install()")
                .drawn_count(),
        ),
    ] {
        assert_eq!(count, 1, "{name}: hidden item must not produce gpu data");
    }
}

// ---------------------------------------------------------------------------
// Warm-up
// ---------------------------------------------------------------------------

/// Warm `set`, then draw the items `push` adds in HDR and in LDR, outlined,
/// and pick them at object and node level. Returns every build those frames
/// made whose label starts with `prefix`; a warmed type should make none.
fn builds_after_warm(
    set: viewport_lib::PipelineSet,
    prefix: &str,
    push: impl Fn(&mut FrameData),
) -> Option<Vec<String>> {
    // The node pick variant needs primitive index; take it where the adapter
    // has it, so the warm-up is checked against every pick member.
    let (device, queue) = headless_device_with_primitive_index().or_else(headless_device)?;
    let mut renderer = renderer_with_item_types(&device);
    renderer.warm_pipelines(&device, &queue, &set);
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for hdr in [true, false] {
        let mut frame = curve_frame();
        frame.interaction.outline_selected = true;
        if !hdr {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        push(&mut frame);
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        for mask in [PickMask::OBJECT, PickMask::POLY_NODE] {
            let _ = renderer.pick_object(
                PickBackend::Gpu,
                glam::Vec2::new(32.0, 32.0),
                &frame,
                &device,
                &queue,
                mask,
            );
        }
    }
    let module = format!("module {prefix}");
    Some(
        viewport_lib::resources::build_log::drain()
            .into_iter()
            .map(|(label, _)| label)
            .filter(|l| l.starts_with(prefix) || l.starts_with(&module))
            .collect(),
    )
}

#[test]
fn a_warmed_streamtube_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some(builds) = builds_after_warm(
        viewport_lib::PipelineSet::default().with_item_type::<StreamtubePlugin>(),
        "streamtube",
        |frame| {
            let (positions, strip_lengths) = spine();
            let mut tube = StreamtubeItem::default();
            tube.positions = positions;
            tube.strip_lengths = strip_lengths;
            tube.radius = 0.5;
            tube.settings.pick_id = PickId(1);
            tube.settings.selected = true;
            frame.scene.items_mut::<StreamtubeItem>().push(tube);
        },
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(
        builds.is_empty(),
        "the first streamtube frames built pipelines after the warm-up: {builds:?}"
    );
}

#[test]
fn a_warmed_tube_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some(builds) = builds_after_warm(
        viewport_lib::PipelineSet::default().with_item_type::<TubePlugin>(),
        "tube",
        |frame| {
            let (positions, strip_lengths) = spine();
            let mut tube = TubeItem::default();
            tube.positions = positions;
            tube.strip_lengths = strip_lengths;
            tube.radius = 0.5;
            tube.settings.pick_id = PickId(1);
            tube.settings.selected = true;
            frame.scene.items_mut::<TubeItem>().push(tube);
        },
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(
        builds.is_empty(),
        "the first tube frames built pipelines after the warm-up: {builds:?}"
    );
}

/// An opaque ribbon, one that routes through OIT in HDR and an additive one,
/// so the warm-up is checked against the variant, OIT and shadow members.
#[test]
fn a_warmed_ribbon_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some(builds) = builds_after_warm(
        viewport_lib::PipelineSet::default().with_item_type::<RibbonPlugin>(),
        "ribbon",
        |frame| {
            let (positions, strip_lengths) = spine();
            let mut opaque = RibbonItem::default();
            opaque.positions = positions;
            opaque.strip_lengths = strip_lengths;
            opaque.width = 1.0;
            opaque.settings.pick_id = PickId(1);
            opaque.settings.selected = true;
            opaque.settings.cast_shadows = true;
            let mut transparent = opaque.clone();
            transparent.depth_write = false;
            transparent.settings.pick_id = PickId(2);
            let mut additive = opaque.clone();
            additive.blend = viewport_lib::SpriteBlend::Additive;
            additive.settings.pick_id = PickId(3);
            frame
                .scene
                .items_mut::<RibbonItem>()
                .extend([opaque, transparent, additive]);
        },
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(
        builds.is_empty(),
        "the first ribbon frames built pipelines after the warm-up: {builds:?}"
    );
}
