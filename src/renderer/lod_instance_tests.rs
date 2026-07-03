//! Per-instance LOD for `MeshInstanceItem`.
//!
//! Drives `prepare` with one mesh-instance batch whose instances sit at
//! different distances, and checks that the renderer splits the batch into one
//! GPU sub-batch per occupied level. Colocated with the renderer so the test can
//! read the private `mesh_instance_gpu_data` field. Skips when no wgpu adapter
//! is available.

use super::types::FrameData;
use super::{
    CameraFrame, MeshInstanceItem, RenderCamera, SceneFrame, SurfaceSubmission, ViewportRenderer,
};
use crate::geometry::primitives;
use crate::renderer::PickId;
use crate::resources::{LodGroupId, MeshData, ViewportGpuResources};

/// Upload each level mesh, then register the group.
fn register(
    res: &mut ViewportGpuResources,
    device: &wgpu::Device,
    levels: &[(MeshData, f32)],
) -> LodGroupId {
    let mut ids = Vec::with_capacity(levels.len());
    let mut sizes = Vec::with_capacity(levels.len());
    for (data, size) in levels {
        ids.push(res.upload_mesh_data(device, data).unwrap());
        sizes.push(*size);
    }
    res.register_lod_group(&ids, &sizes).unwrap()
}

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("lod_instance_tests"),
        ..Default::default()
    }))
    .ok()
}

fn frame_looking_down_z(item: MeshInstanceItem) -> FrameData {
    let mut cam = RenderCamera::default();
    cam.eye_position = [0.0, 0.0, 0.0];
    cam.forward = [0.0, 0.0, -1.0];
    let cf = CameraFrame::new(cam, [256.0, 256.0]);
    let mut sf = SceneFrame::new(SurfaceSubmission::Flat(std::sync::Arc::from(Vec::new())));
    sf.mesh_instances.push(item);
    FrameData::new(cf, sf)
}

fn translate(z: f32) -> [[f32; 4]; 4] {
    glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, z)).to_cols_array_2d()
}

#[test]
fn instances_split_into_one_batch_per_level() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping instances_split_into_one_batch_per_level: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

    let group = register(
        renderer.resources_mut(),
        &device,
        &[
            (primitives::icosphere(1.0, 3), 0.5),
            (primitives::icosphere(1.0, 1), 0.2),
            (primitives::icosphere(1.0, 0), 0.0),
        ],
    );

    // Two near instances (full detail) and one far (crudest). Level 1 stays
    // empty, so we expect two sub-batches, not three.
    let mut item = MeshInstanceItem::default();
    item.lod_group = Some(group);
    item.settings.pick_id = PickId(1);
    item.transforms = vec![translate(-3.0), translate(-3.0), translate(-300.0)];

    let fd = frame_looking_down_z(item);
    let _ = renderer.prepare_callback(&device, &queue, &fd);

    assert_eq!(
        renderer.mesh_instance_gpu_data.len(),
        2,
        "two occupied levels should produce two sub-batches"
    );
    let counts: Vec<u32> = renderer
        .mesh_instance_gpu_data
        .iter()
        .map(|b| b.instance_count)
        .collect();
    assert_eq!(
        counts.iter().sum::<u32>(),
        3,
        "every instance is drawn once"
    );
    assert!(
        counts.contains(&2),
        "two near instances share the full-detail batch"
    );
    assert!(
        counts.contains(&1),
        "one far instance is its own crude batch"
    );

    let stats = renderer.last_frame_stats();
    assert_eq!(stats.lod_items_resolved, 3, "three instances resolved");
}

#[test]
fn instances_below_cull_size_are_dropped() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping instances_below_cull_size_are_dropped: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);

    let group = register(
        renderer.resources_mut(),
        &device,
        &[
            (primitives::icosphere(1.0, 3), 0.5),
            (primitives::icosphere(1.0, 0), 0.0),
        ],
    );
    renderer
        .resources_mut()
        .set_lod_cull_below(group, Some(0.05))
        .unwrap();

    // Two near instances draw; two very far instances fall below the cull size.
    let mut item = MeshInstanceItem::default();
    item.lod_group = Some(group);
    item.transforms = vec![
        translate(-3.0),
        translate(-3.0),
        translate(-600.0),
        translate(-600.0),
    ];

    let fd = frame_looking_down_z(item);
    let _ = renderer.prepare_callback(&device, &queue, &fd);

    let drawn: u32 = renderer
        .mesh_instance_gpu_data
        .iter()
        .map(|b| b.instance_count)
        .sum();
    assert_eq!(drawn, 2, "only the two near instances are drawn");
    assert_eq!(renderer.last_frame_stats().lod_culled, 2);
}

#[test]
fn instanced_item_without_group_makes_one_batch() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping instanced_item_without_group_makes_one_batch: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mesh = renderer
        .resources_mut()
        .upload_mesh_data(&device, &primitives::cube(1.0))
        .unwrap();

    let mut item = MeshInstanceItem::default();
    item.mesh_id = mesh;
    item.transforms = vec![translate(-3.0), translate(-300.0)];

    let fd = frame_looking_down_z(item);
    let _ = renderer.prepare_callback(&device, &queue, &fd);

    assert_eq!(
        renderer.mesh_instance_gpu_data.len(),
        1,
        "no LOD group means one batch for the whole item"
    );
    assert_eq!(renderer.mesh_instance_gpu_data[0].instance_count, 2);
    assert_eq!(renderer.last_frame_stats().lod_items_resolved, 0);
}
