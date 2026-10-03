//! External instance sets: the caller-owned positions buffer an item draws
//! from, and what happens to a handle when the set behind it goes away.
//!
//! One file per item type, so a type's coverage travels with it.

use viewport_lib::gpu;
use viewport_lib_plugins::item_types::external_instances::{
    ExternalInstanceSetConfig, ExternalInstanceUploads, ExternalInstancesItem,
};

mod common;
use common::*;

fn positions_buffer(device: &gpu::Device, elements: u64, usage: gpu::BufferUsages) -> gpu::Buffer {
    device.create_buffer(&gpu::BufferDescriptor {
        label: Some("test_positions"),
        size: elements * 12,
        usage,
        mapped_at_creation: false,
    })
}

/// A handle resolves until the set is dropped, the slot is reused, and the
/// stale handle does not follow the slot to its new occupant.
#[test]
fn create_and_drop_roundtrip() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let buf = positions_buffer(&device, 8, gpu::BufferUsages::STORAGE);
    let id = renderer
        .create_external_instance_set(
            &device,
            &ExternalInstanceSetConfig::new(mesh_id, buf.clone()),
        )
        .unwrap();

    // Re-point works.
    let bigger = positions_buffer(&device, 16, gpu::BufferUsages::STORAGE);
    renderer
        .set_external_instance_set_buffer(id, bigger)
        .unwrap();

    renderer.drop_external_instance_set(id);

    // The dropped slot is reused by the next create, but the old handle must
    // not resolve to the slot's new occupant. Before the handle carried a
    // generation, it did.
    let id2 = renderer
        .create_external_instance_set(&device, &ExternalInstanceSetConfig::new(mesh_id, buf))
        .unwrap();
    assert_ne!(id, id2, "the reused slot must carry a new generation");

    let other = positions_buffer(&device, 4, gpu::BufferUsages::STORAGE);
    assert!(
        matches!(
            renderer.set_external_instance_set_buffer(id, other),
            Err(viewport_lib::error::ViewportError::StaleHandle { .. })
        ),
        "re-pointing a dropped handle must fail rather than write through to \
         whatever now occupies the slot"
    );
}

/// The buffer has to be a storage buffer: the shader binds it as one, so a
/// wrong usage is refused at creation rather than at pipeline build.
#[test]
fn create_rejects_non_storage_buffer() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let buf = positions_buffer(&device, 8, gpu::BufferUsages::VERTEX);
    assert!(matches!(
        renderer
            .create_external_instance_set(&device, &ExternalInstanceSetConfig::new(mesh_id, buf)),
        Err(viewport_lib::error::ViewportError::ExternalBufferUsageMissing { .. })
    ));
}

/// Warming the type builds every pipeline its draws use, so the first frames
/// in either format build nothing.
#[test]
fn a_warmed_external_instances_type_builds_nothing_on_its_first_frame() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let positions: Vec<f32> = vec![-0.6, 0.0, 0.0, 0.6, 0.0, 0.0];
    let buf = positions_buffer(
        &device,
        2,
        gpu::BufferUsages::STORAGE | gpu::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&positions));
    let set_id = renderer
        .create_external_instance_set(&device, &ExternalInstanceSetConfig::new(mesh_id, buf))
        .unwrap();

    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default()
            .with_item_type::<viewport_lib_plugins::item_types::external_instances::ExternalInstancesPlugin>(),
    );
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for hdr in [true, false] {
        let mut frame = sub_object_pick_frame();
        if !hdr {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        let mut item = ExternalInstancesItem::new(set_id, 2);
        item.scale = 0.4;
        frame.scene.items_mut::<ExternalInstancesItem>().push(item);
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    }
    let builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| {
            l.starts_with("external_instances") || l.starts_with("module external_instances")
        })
        .collect();
    assert!(
        builds.is_empty(),
        "the first external instance frames built pipelines after the warm-up: {builds:?}"
    );
}
