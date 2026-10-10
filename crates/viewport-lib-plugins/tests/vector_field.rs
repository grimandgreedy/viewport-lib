//! Picking and selection for the vector field item type: GPU pick-id,
//! per-sample sub-objects, CPU proximity picking, rect select, the
//! pre-uploaded reference form, and the outline mask.
//!
//! One file per item type, so a type's coverage travels with it.

#![cfg(feature = "item-types")]

use viewport_lib::Colour;
mod common;
use common::*;
use viewport_lib::plugin_api::{Handles, Span, Uploads, Writes};
use viewport_lib::{ColourSource, MeshId, SizeSource};
use viewport_lib_plugins::item_types::vector_field::channels as vf;
use viewport_lib_plugins::item_types::vector_field::{
    TYPE_NAME as VECTOR_FIELD_TYPE_NAME, VectorFieldItem, VectorFieldPlugin, VectorFieldRefItem,
};

/// The build log is process-wide, so a test that reads it must not overlap
/// another test building this type's pipelines.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A unit-ish arrow along +Z, uploaded so a field has something to instance.
fn arrow_mesh(renderer: &mut ViewportRenderer, device: &viewport_lib::wgpu::Device) -> MeshId {
    renderer
        .resources_mut()
        .upload_mesh_data(device, &viewport_lib::primitives::arrow(0.1, 0.25, 0.4, 12))
        .expect("arrow mesh uploads")
}

/// Three samples spread along X; the centre one sits at the origin, under the
/// cursor of a `sub_object_pick_frame` click at (32, 32).
///
/// The vectors point back at `eye` so each arrow projects onto a tight cluster
/// around its own base. A sample drawn side-on has its body, and so the CPU
/// pick's hit circle, centred half a length away from the position it reports,
/// which is correct but makes a fixed click point a poor test of anything else.
fn three_samples_facing(shape: MeshId, eye: [f32; 3]) -> VectorFieldItem {
    let mut item = VectorFieldItem::new(shape);
    item.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    item.vectors = item
        .positions
        .iter()
        .map(|p| {
            (glam::Vec3::from(eye) - glam::Vec3::from(*p))
                .normalize()
                .to_array()
        })
        .collect();
    item.scale = 1.5;
    item
}

/// The same field with the vectors along +Z, for the cases that never pick.
fn three_samples(shape: MeshId) -> VectorFieldItem {
    let mut item = VectorFieldItem::new(shape);
    item.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    item.vectors = vec![[0.0, 0.0, 1.0]; 3];
    item.scale = 1.5;
    item
}

#[test]
fn gpu_pick_vector_field_resolves_sample() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = arrow_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_samples(shape);
    item.settings.pick_id = PickId(800);
    frame.scene.items_mut::<VectorFieldItem>().push(item);

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
        .expect("centre sample should be hit");
    assert_eq!(hit.id, 800);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
}

#[test]
fn cpu_pick_vector_field_resolves_sample() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let shape = arrow_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_samples_facing(shape, frame.camera.render_camera.eye_position);
    item.settings.pick_id = PickId(801);
    frame.scene.items_mut::<VectorFieldItem>().push(item);

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
        .expect("centre sample should be hit");
    assert_eq!(hit.id, 801);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
}

/// An `OBJECT`-only query answers with the field and no sample, on both
/// backends.
#[test]
fn an_object_query_drops_the_sample_sub_object() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let shape = arrow_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_samples_facing(shape, frame.camera.render_camera.eye_position);
    item.settings.pick_id = PickId(802);
    frame.scene.items_mut::<VectorFieldItem>().push(item);

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
            .expect("centre sample should be hit");
        assert_eq!(hit.id, 802);
        assert_eq!(hit.sub_object, None, "{backend:?} must not report a sample");
    }
}

#[test]
fn rect_pick_collects_vector_field_samples() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let shape = arrow_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_samples(shape);
    item.settings.pick_id = PickId(803);
    frame.scene.items_mut::<VectorFieldItem>().push(item);

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
        "rect should collect sample sub-objects"
    );
    assert!(
        result
            .elements
            .iter()
            .all(|(id, sub)| *id == 803 && matches!(sub, viewport_lib::SubObjectRef::Instance(_))),
        "every element must be a sample of the field"
    );
}

/// A field uploaded once and drawn through `VectorFieldRefItem` picks the same
/// as an inline item, under the reference's own pick id.
#[test]
fn a_reference_item_picks_like_an_inline_one() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = arrow_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let source = renderer
        .upload(&device, &queue, &three_samples(shape))
        .unwrap();

    let mut item = VectorFieldRefItem::new(source);
    item.settings.pick_id = PickId(804);
    frame.scene.items_mut::<VectorFieldRefItem>().push(item);

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
        .expect("centre sample of the referenced field should be hit");
    assert_eq!(hit.id, 804);
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
    let shape = arrow_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let source = renderer
        .upload(&device, &queue, &three_samples(shape))
        .unwrap();

    let mut item = VectorFieldRefItem::new(source);
    item.settings.pick_id = PickId(805);
    item.settings.hidden = true;
    frame.scene.items_mut::<VectorFieldRefItem>().push(item);

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
// The fields the item type holds
// ---------------------------------------------------------------------------

#[test]
fn an_uploaded_vector_field_resolves_until_it_is_dropped() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = arrow_mesh(&mut renderer, &device);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer
        .upload(&device, &queue, &three_samples(shape))
        .unwrap();
    assert!(renderer.resident_bytes().plugin_bytes > baseline);
    assert!(
        renderer
            .replace(&device, &queue, id, &three_samples(shape))
            .is_ok()
    );

    assert!(renderer.release(id));
    assert!(!renderer.release(id), "a handle drops once");
    assert_eq!(renderer.resident_bytes().plugin_bytes, baseline);
}

#[test]
fn begin_upload_vector_field_drains_to_a_handle() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = arrow_mesh(&mut renderer, &device);

    let job = renderer
        .begin_upload(&device, &queue, three_samples(shape))
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
    let id: viewport_lib_plugins::item_types::vector_field::VectorFieldId = renderer
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
    let shape = arrow_mesh(&mut renderer, &device);
    let mut frame = sub_object_pick_frame();

    let visible = three_samples(shape);
    let mut hidden = three_samples(shape);
    hidden.settings.hidden = true;
    frame
        .scene
        .items_mut::<VectorFieldItem>()
        .extend([visible, hidden]);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let plugin = renderer
        .item_type_plugin::<VectorFieldPlugin>(VECTOR_FIELD_TYPE_NAME)
        .expect("registered by install()");
    assert_eq!(
        plugin.drawn_count(),
        1,
        "hidden item must not produce gpu data"
    );
}

/// A field whose shape was never uploaded draws nothing rather than drawing
/// the wrong geometry.
#[test]
fn an_unset_shape_draws_nothing() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let mut item = three_samples(MeshId::INVALID);
    item.settings.pick_id = PickId(806);
    frame.scene.items_mut::<VectorFieldItem>().push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), None, "no shape, no pixels, no pick");
}

// ---------------------------------------------------------------------------
// Encoding and selection
// ---------------------------------------------------------------------------

/// `SizeSource` drives the drawn size: the same field at a larger output range
/// covers more pixels. This is the knob that replaced the hardcoded magnitude
/// floor, so it has to do something visible.
#[test]
fn the_size_source_changes_how_much_is_drawn() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = arrow_mesh(&mut renderer, &device);

    let mut drawn_pixels = |size: SizeSource| {
        let mut frame = sub_object_pick_frame();
        let mut item = three_samples(shape);
        item.size = size;
        item.colour = ColourSource::Solid(Colour::linear(1.0, 0.0, 0.0, 1.0));
        item.settings.unlit = true;
        frame.scene.items_mut::<VectorFieldItem>().push(item);
        let img = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        img.chunks_exact(4)
            .filter(|px| px[0] > 120 && px[1] < 90 && px[2] < 90)
            .count()
    };

    let small = drawn_pixels(SizeSource::Uniform(0.2));
    let large = drawn_pixels(SizeSource::Uniform(1.0));
    assert!(small > 0, "the small case must draw something");
    assert!(
        large > small,
        "a larger uniform size drew {large} pixels against {small} for the smaller"
    );
}

/// Every sample in a selected field goes into the outline mask, not just the
/// first. The mask shader reads the instance buffer as a storage array, so a
/// struct that does not match the record the store writes silently outlines the
/// wrong subset.
///
/// Measured as outlined area rather than by locating each sample: adjacent
/// outlines merge, so counting rings is unreliable, while the total area scales
/// with how many were drawn.
#[test]
fn a_selected_field_outlines_every_sample() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = arrow_mesh(&mut renderer, &device);

    let mut outlined_pixels = |count: usize| {
        let mut frame = sub_object_pick_frame();
        frame.camera.render_camera.aspect = 4.0;
        frame.camera.viewport_size = [512.0, 128.0];
        frame.camera.pixels_per_point = 1.0;
        frame.interaction.outline_selected = true;
        frame.interaction.outline_colour = Colour::linear(1.0, 0.0, 0.0, 1.0);
        frame.interaction.outline_width_px = 3.0;

        let mut item = VectorFieldItem::new(shape);
        item.positions = (0..count)
            .map(|i| [(i as f32 - 3.5) * 0.9, 0.0, 0.0])
            .collect();
        item.vectors = vec![[0.0, 0.0, 1.0]; count];
        item.scale = 1.0;
        item.size = SizeSource::Uniform(1.0);
        item.colour = ColourSource::Solid(Colour::linear(0.0, 0.0, 1.0, 1.0));
        item.settings.pick_id = PickId(1);
        item.settings.selected = true;
        frame.scene.items_mut::<VectorFieldItem>().push(item);

        let img = renderer.render_offscreen(&device, &queue, &frame, 512, 128);
        img.chunks_exact(4)
            .filter(|px| px[0] > 120 && px[1] < 90 && px[2] < 90)
            .count()
    };

    let one = outlined_pixels(1);
    let eight = outlined_pixels(8);
    assert!(one > 0, "the single-sample case must outline something");
    assert!(
        eight >= one * 6,
        "eight selected samples outlined {eight} pixels against {one} for one; \
         every sample in the field must be outlined"
    );
}

// ---------------------------------------------------------------------------
// Ranged writes
// ---------------------------------------------------------------------------

/// A field's storage is one interleaved record per sample, so the sample is the
/// channel and a write supplies whole samples.
#[test]
fn a_reserved_field_takes_ranged_sample_writes() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = arrow_mesh(&mut renderer, &device);

    let mut field = VectorFieldItem::new(shape);
    field.positions = vec![[0.0, 0.0, 0.0]; 64];
    field.vectors = vec![[0.0, 0.0, 1.0]; 64];
    let id = renderer.upload(&device, &queue, &field).expect("upload");

    renderer
        .reserve(vf::Samples, &device, &queue, id, 1_024)
        .expect("reserve");
    let extent = renderer.extent(vf::Samples, id).expect("live handle");
    assert!(extent.capacity >= 1_024);
    assert_eq!(extent.len, 64, "the reserve adds headroom, not samples");

    let batch: Vec<vf::Sample> = (0..64)
        .map(|i| {
            vf::Sample::new(
                [i as f32, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                1.0,
                viewport_lib::Colour::WHITE,
            )
        })
        .collect();
    renderer
        .write_range(vf::Samples, &queue, id, 64, &batch)
        .expect("a window inside the reserve");
    assert_eq!(renderer.extent(vf::Samples, id).unwrap().len, 128);

    // Two disjoint runs in one call, then one that does not fit.
    renderer
        .write_spans(
            vf::Samples,
            &queue,
            id,
            &[Span::new(0, &batch[..8]), Span::new(900, &batch[..8])],
        )
        .expect("two runs inside the reserve");
    assert!(
        renderer
            .write_range(vf::Samples, &queue, id, 1_020, &batch)
            .is_err(),
        "a window past the reserve is refused rather than grown into"
    );

    // And it still draws.
    let mut frame = sub_object_pick_frame();
    frame
        .scene
        .items_mut::<VectorFieldRefItem>()
        .push(VectorFieldRefItem::new(id));
    let _ = renderer.pass().prepare(&device, &queue, &frame);
    assert!(renderer.release(id));
}

/// Naming the vector field type in a warm-up builds its pipelines, so the
/// first frame that draws, outlines and picks a field, in either format,
/// compiles none of them.
#[test]
fn a_warmed_vector_field_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let shape = arrow_mesh(&mut renderer, &device);
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default().with_item_type::<VectorFieldPlugin>(),
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
        let mut item = three_samples(shape);
        item.settings.pick_id = PickId(700);
        item.settings.selected = true;
        frame.scene.items_mut::<VectorFieldItem>().push(item);
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
        .filter(|l| l.starts_with("vector_field") || l.starts_with("module vector_field"))
        .collect();
    assert!(
        builds.is_empty(),
        "the first vector field frames built pipelines after the warm-up: {builds:?}"
    );
}
