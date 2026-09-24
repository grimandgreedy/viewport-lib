//! Picking for the tensor glyph item type: GPU pick-id, per-instance
//! sub-objects, CPU proximity picking, rect select, and the pre-uploaded
//! reference form.
//!
//! One file per item type, so a type's coverage travels with it.

mod common;
use common::*;
use viewport_lib::plugin_api::Handles;

use viewport_lib::plugin_api::Uploads;
use viewport_lib_item_types::{
    TENSOR_GLYPH_TYPE_NAME, TensorGlyphItem, TensorGlyphPlugin, TensorGlyphSetRefItem,
};

/// Three unit-sphere tensors spread along X; the centre one sits at the origin,
/// under the cursor of a `sub_object_pick_frame` click at (32, 32).
fn three_tensors() -> TensorGlyphItem {
    let mut item = TensorGlyphItem::default();
    item.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    item.eigenvalues = vec![[1.0, 1.0, 1.0]; 3];
    item.eigenvectors = vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; 3];
    item.scale = 0.6;
    item
}

#[test]
fn gpu_pick_tensor_glyph_resolves_instance() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_tensors();
    item.settings.pick_id = PickId(700);
    frame
        .scene
        .items_mut::<viewport_lib_item_types::TensorGlyphItem>()
        .push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::INSTANCE,
        )
        .expect("centre ellipsoid should be hit");
    assert_eq!(hit.id, 700);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
}

#[test]
fn cpu_pick_tensor_glyph_resolves_instance() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut item = three_tensors();
    item.settings.pick_id = PickId(701);
    frame
        .scene
        .items_mut::<viewport_lib_item_types::TensorGlyphItem>()
        .push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Cpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::INSTANCE,
        )
        .expect("centre ellipsoid should be hit");
    assert_eq!(hit.id, 701);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
}

/// An `OBJECT`-only query answers with the set and no instance, on both
/// backends.
#[test]
fn an_object_query_drops_the_instance_sub_object() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut item = three_tensors();
    item.settings.pick_id = PickId(702);
    frame
        .scene
        .items_mut::<viewport_lib_item_types::TensorGlyphItem>()
        .push(item);

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
            .expect("centre ellipsoid should be hit");
        assert_eq!(hit.id, 702);
        assert_eq!(
            hit.sub_object, None,
            "{backend:?} must not report an instance"
        );
    }
}

#[test]
fn rect_pick_collects_tensor_glyph_instances() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut item = three_tensors();
    item.settings.pick_id = PickId(703);
    frame
        .scene
        .items_mut::<viewport_lib_item_types::TensorGlyphItem>()
        .push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let result = renderer.pick_rect_objects(
        PickBackend::Cpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::INSTANCE,
    );
    assert!(
        result.objects.is_empty(),
        "INSTANCE mask carries no OBJECT bit"
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect instance sub-objects"
    );
    assert!(
        result
            .elements
            .iter()
            .all(|(id, sub)| *id == 703 && matches!(sub, viewport_lib::SubObjectRef::Instance(_))),
        "every element must be an instance of the set"
    );
}

/// A set uploaded once and drawn through `TensorGlyphSetRefItem` picks the same
/// as an inline item, under the reference's own pick id.
#[test]
fn a_reference_item_picks_like_an_inline_one() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let source = renderer.upload(&device, &queue, &three_tensors()).unwrap();

    let mut item = TensorGlyphSetRefItem::new(source);
    item.settings.pick_id = PickId(704);
    frame
        .scene
        .items_mut::<viewport_lib_item_types::TensorGlyphSetRefItem>()
        .push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::INSTANCE,
        )
        .expect("centre ellipsoid of the referenced set should be hit");
    assert_eq!(hit.id, 704);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
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

    let source = renderer.upload(&device, &queue, &three_tensors()).unwrap();

    let mut item = TensorGlyphSetRefItem::new(source);
    item.settings.pick_id = PickId(705);
    item.settings.hidden = true;
    frame
        .scene
        .items_mut::<viewport_lib_item_types::TensorGlyphSetRefItem>()
        .push(item);

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
// The sets the item type holds
// ---------------------------------------------------------------------------

fn sample_tensor_glyph_set() -> viewport_lib_item_types::TensorGlyphItem {
    let mut item = viewport_lib_item_types::TensorGlyphItem::default();
    item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    item.eigenvalues = vec![[1.0, 0.5, 0.25], [0.5, 0.5, 0.5]];
    item.eigenvectors = vec![
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ];
    item
}

#[test]
fn an_uploaded_tensor_glyph_set_resolves_until_it_is_dropped() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer
        .upload(&device, &queue, &sample_tensor_glyph_set())
        .unwrap();
    assert!(renderer.resident_bytes().plugin_bytes > baseline);
    assert!(
        renderer
            .replace(&device, &queue, id, &sample_tensor_glyph_set())
            .is_ok()
    );

    assert!(renderer.release(id));
    assert!(!renderer.release(id), "a handle drops once");
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_tensor_glyph_set_drains_to_a_handle() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);

    let job = renderer
        .begin_upload(&device, &queue, sample_tensor_glyph_set())
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
    let id: viewport_lib_item_types::TensorGlyphSetId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(renderer.release(id));
}

/// A hidden item produces no draw data.
#[test]
fn hidden_items_produce_no_draw_data() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    viewport_lib_item_types::install(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let visible = three_tensors();
    let mut hidden = three_tensors();
    hidden.settings.hidden = true;
    frame
        .scene
        .items_mut::<TensorGlyphItem>()
        .extend([visible, hidden]);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let plugin = renderer
        .item_type_plugin::<TensorGlyphPlugin>(TENSOR_GLYPH_TYPE_NAME)
        .expect("registered by install()");
    assert_eq!(
        plugin.drawn_count(),
        1,
        "hidden item must not produce gpu data"
    );
}
