//! Picking for the tensor field item type: GPU pick-id, per-sample
//! sub-objects, CPU proximity picking, rect select, and the pre-uploaded
//! reference form, plus the component-to-eigen path.
//!
//! One file per item type, so a type's coverage travels with it.

mod common;
use common::*;
use viewport_lib::plugin_api::Handles;

use viewport_lib::MeshId;
use viewport_lib::plugin_api::{Uploads, Writes};
use viewport_lib_plugins::item_types::tensor_field::channels as tf;
use viewport_lib_plugins::item_types::tensor_field::{
    TYPE_NAME as TENSOR_FIELD_TYPE_NAME, TensorFieldItem, TensorFieldPlugin, TensorFieldRefItem,
    TensorSource,
};

/// The build log is process-wide, so a test that reads it must not overlap
/// another test building this type's pipelines.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A unit sphere for the field to instance.
fn sphere_mesh(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device) -> MeshId {
    renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::icosphere(1.0, 2))
        .expect("sphere mesh uploads")
}

/// Three isotropic tensors spread along X; the centre one sits at the origin,
/// under the cursor of a `sub_object_pick_frame` click at (32, 32).
///
/// The components are a pure hydrostatic state, so the decomposition gives three
/// equal eigenvalues and each instance stays a sphere.
fn three_tensors(shape: MeshId) -> TensorFieldItem {
    let mut item = TensorFieldItem::new(shape);
    item.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    item.tensors = TensorSource::Components(vec![[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]; 3]);
    item.scale = 0.6;
    item
}

#[test]
fn gpu_pick_tensor_field_resolves_instance() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = sphere_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_tensors(shape);
    item.settings.pick_id = PickId(700);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::tensor_field::TensorFieldItem>()
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
fn cpu_pick_tensor_field_resolves_instance() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let shape = sphere_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_tensors(shape);
    item.settings.pick_id = PickId(701);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::tensor_field::TensorFieldItem>()
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
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let shape = sphere_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_tensors(shape);
    item.settings.pick_id = PickId(702);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::tensor_field::TensorFieldItem>()
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
fn rect_pick_collects_tensor_field_instances() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let shape = sphere_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_tensors(shape);
    item.settings.pick_id = PickId(703);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::tensor_field::TensorFieldItem>()
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

/// A set uploaded once and drawn through `TensorFieldRefItem` picks the same
/// as an inline item, under the reference's own pick id.
#[test]
fn a_reference_item_picks_like_an_inline_one() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = sphere_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let source = renderer
        .upload(&device, &queue, &three_tensors(shape))
        .unwrap();

    let mut item = TensorFieldRefItem::new(source);
    item.settings.pick_id = PickId(704);
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::tensor_field::TensorFieldRefItem>()
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
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = sphere_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let source = renderer
        .upload(&device, &queue, &three_tensors(shape))
        .unwrap();

    let mut item = TensorFieldRefItem::new(source);
    item.settings.pick_id = PickId(705);
    item.settings.hidden = true;
    frame
        .scene
        .items_mut::<viewport_lib_plugins::item_types::tensor_field::TensorFieldRefItem>()
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

fn sample_tensor_field(shape: MeshId) -> TensorFieldItem {
    let mut item = TensorFieldItem::new(shape);
    item.positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    item.tensors = TensorSource::Eigen {
        values: vec![[1.0, 0.5, 0.25], [0.5, 0.5, 0.5]],
        vectors: vec![
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
    };
    item
}

#[test]
fn an_uploaded_tensor_field_set_resolves_until_it_is_dropped() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = sphere_mesh(&mut renderer, &device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer
        .upload(&device, &queue, &sample_tensor_field(shape))
        .unwrap();
    assert!(renderer.resident_bytes().plugin_bytes > baseline);
    assert!(
        renderer
            .replace(&device, &queue, id, &sample_tensor_field(shape))
            .is_ok()
    );

    assert!(renderer.release(id));
    assert!(!renderer.release(id), "a handle drops once");
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_tensor_field_set_drains_to_a_handle() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = sphere_mesh(&mut renderer, &device);

    let job = renderer
        .begin_upload(&device, &queue, sample_tensor_field(shape))
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
    let id: viewport_lib_plugins::item_types::tensor_field::TensorFieldId = renderer
        .upload_result(job)
        .expect("the finished job yields a handle");
    assert!(renderer.release(id));
}

/// A hidden item produces no draw data.
#[test]
fn hidden_items_produce_no_draw_data() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = sphere_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let visible = three_tensors(shape);
    let mut hidden = three_tensors(shape);
    hidden.settings.hidden = true;
    frame
        .scene
        .items_mut::<TensorFieldItem>()
        .extend([visible, hidden]);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let plugin = renderer
        .item_type_plugin::<TensorFieldPlugin>(TENSOR_FIELD_TYPE_NAME)
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

/// The tensor field is the case where the encode earns its place: the caller
/// supplies an eigendecomposition and the store bakes the matrices.
#[test]
fn a_reserved_tensor_field_takes_ranged_sample_writes() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = sphere_mesh(&mut renderer, &device);

    let mut field = TensorFieldItem::new(shape);
    field.positions = vec![[0.0, 0.0, 0.0]; 16];
    field.tensors = TensorSource::Components(vec![[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]; 16]);
    let id = renderer.upload(&device, &queue, &field).expect("upload");

    renderer
        .reserve(tf::Samples, &device, &queue, id, 128)
        .expect("reserve");
    assert!(renderer.extent(tf::Samples, id).unwrap().capacity >= 128);

    let batch: Vec<tf::Sample> = (0..16)
        .map(|i| tf::Sample {
            position: [i as f32, 0.0, 0.0],
            axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            extents: [1.0, 0.5, 0.25],
            scalar: 0.5,
            colour: viewport_lib::Colour::WHITE,
        })
        .collect();
    renderer
        .write_range(tf::Samples, &queue, id, 16, &batch)
        .expect("a window inside the reserve");
    assert_eq!(renderer.extent(tf::Samples, id).unwrap().len, 32);

    // A zero extent would make the normal matrix a division by zero, so the
    // encode clamps it the way the upload path does rather than sending an
    // infinity to the GPU.
    let degenerate = [tf::Sample {
        extents: [0.0, 0.0, 0.0],
        ..batch[0]
    }];
    renderer
        .write_range(tf::Samples, &queue, id, 0, &degenerate)
        .expect("a collapsed sample is clamped, not refused");

    let mut frame = sub_object_pick_frame();
    frame
        .scene
        .items_mut::<TensorFieldRefItem>()
        .push(TensorFieldRefItem::new(id));
    let _ = renderer.pass().prepare(&device, &queue, &frame);
    assert!(renderer.release(id));
}

/// Naming the tensor field type in a warm-up builds its pipelines, so the first
/// frame that draws, outlines and picks a field, in either format, compiles
/// none of them.
#[test]
fn a_warmed_tensor_field_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = sphere_mesh(&mut renderer, &device);
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default().with_item_type::<TensorFieldPlugin>(),
    );
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for hdr in [true, false] {
        let mut frame = sub_object_pick_frame();
        if !hdr {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        frame.interaction.outline_selected = true;
        let mut item = three_tensors(shape);
        item.settings.pick_id = PickId(700);
        item.settings.selected = true;
        frame.scene.items_mut::<TensorFieldItem>().push(item);
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        let _ = renderer.pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::INSTANCE,
        );
    }
    let builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| l.starts_with("tensor_field") || l.starts_with("module tensor_field"))
        .collect();
    assert!(
        builds.is_empty(),
        "the first tensor field frames built pipelines after the warm-up: {builds:?}"
    );
}
