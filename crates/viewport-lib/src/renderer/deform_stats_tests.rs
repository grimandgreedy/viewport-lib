//! Frame-stats coverage for deform-slot writes.
//!
//! Two things are checked here. First, that a per-mesh slot write reports what
//! it cost: the write reallocates its buffer and rebuilds every bind group
//! pointing at it, and `FrameStats` has to show that rather than leave it
//! invisible. Second, that an item whose deform data the chosen draw path
//! cannot apply is counted, because the result is a wrong picture with no other
//! signal. Colocated with the renderer so the tests can drive `prepare`
//! directly. Skips when no wgpu adapter is available.

use super::types::FrameData;
use super::{
    CameraFrame, RenderCamera, SceneFrame, SceneRenderItem, SurfaceSubmission, ViewportRenderer,
};
use crate::geometry::primitives;

fn headless_device() -> Option<(crate::gpu::Device, crate::gpu::Queue)> {
    let instance = crate::gpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(
        &crate::gpu::RequestAdapterOptions {
            power_preference: crate::gpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            #[cfg(wgpu30)]
            apply_limit_buckets: false,
        },
    ))
    .ok()?;
    pollster::block_on(adapter.request_device(&crate::gpu::DeviceDescriptor {
        label: Some("deform_stats_tests"),
        required_limits: crate::ViewportRenderer::recommended_device_limits(&adapter),
        ..Default::default()
    }))
    .ok()
}

fn translate(x: f32) -> [[f32; 4]; 4] {
    glam::Mat4::from_translation(glam::Vec3::new(x, 0.0, -6.0)).to_cols_array_2d()
}

/// `count` copies of one mesh, all visible, looking down -Z.
fn frame_with(mesh_id: crate::MeshId, count: usize) -> FrameData {
    let mut cam = RenderCamera::default();
    cam.eye_position = [0.0, 0.0, 0.0];
    cam.forward = [0.0, 0.0, -1.0];
    let cf = CameraFrame::new(cam, [256.0, 256.0]);
    let items: Vec<SceneRenderItem> = (0..count)
        .map(|i| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = translate(i as f32 * 2.5);
            item
        })
        .collect();
    FrameData::new(
        cf,
        SceneFrame::new(SurfaceSubmission::Flat(std::sync::Arc::from(items))),
    )
}

/// A same-size per-mesh slot write still reallocates the buffer and rebuilds
/// every bind group on the mesh, and the counters say so. This is the
/// regression guard for making that write in place: when the per-mesh path
/// keeps a capacity the way `attach_slot_instance` does, the second write here
/// stops reallocating and these expectations change to zero.
#[test]
fn per_mesh_slot_write_reports_its_reallocation() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping per_mesh_slot_write_reports_its_reallocation: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
    let mesh = primitives::grid_plane(1.0, 1.0, 4, 4);
    let vertex_count = mesh.positions.len();
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();

    // One f32 per vertex on slot 0.
    let payload = vec![0u8; vertex_count * 4];
    let fd = frame_with(mesh_id, 1);

    // Clear anything the first prepare accumulated, then write the slot twice
    // with identical sizes and read what the second frame reports.
    let _ = renderer.prepare_callback(&device, &queue, &fd);
    renderer
        .resources_mut()
        .attach_deform_slot(&device, mesh_id, 0, 4, &payload);
    let _ = renderer.prepare_callback(&device, &queue, &fd);
    let first = renderer.last_frame_stats();
    assert_eq!(
        first.deform_buffer_reallocations, 1,
        "the first attach allocates the slot's storage"
    );
    assert_eq!(
        first.deform_mesh_bind_groups_rebuilt, 1,
        "the mesh bind group points at the buffer that was replaced"
    );

    renderer
        .resources_mut()
        .attach_deform_slot(&device, mesh_id, 0, 4, &payload);
    let _ = renderer.prepare_callback(&device, &queue, &fd);
    let second = renderer.last_frame_stats();
    assert_eq!(
        second.deform_buffer_reallocations, 1,
        "a same-size rewrite reallocates too: it is the per-frame cost this \
         counter exists to make visible"
    );
    assert_eq!(second.deform_mesh_bind_groups_rebuilt, 1);

    // And the counters are per-frame, not cumulative.
    let _ = renderer.prepare_callback(&device, &queue, &fd);
    let idle = renderer.last_frame_stats();
    assert_eq!(
        idle.deform_buffer_reallocations, 0,
        "a frame with no slot write reports no reallocation"
    );
    assert_eq!(idle.deform_mesh_bind_groups_rebuilt, 0);
}

/// One item with per-mesh deform data draws per-object and applies it. Two
/// items cross the instancing threshold, and the instanced draws bind the empty
/// deform group, so the data does not reach the shader and the items render
/// undeformed. That is a wrong picture with no other signal, so it is counted.
#[test]
fn instanced_items_with_per_mesh_deform_data_are_counted_as_ignored() {
    let Some((device, queue)) = headless_device() else {
        eprintln!(
            "skipping instanced_items_with_per_mesh_deform_data_are_counted_as_ignored: \
             no GPU adapter"
        );
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
    let mesh = primitives::grid_plane(1.0, 1.0, 4, 4);
    let vertex_count = mesh.positions.len();
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();
    renderer.resources_mut().attach_deform_slot(
        &device,
        mesh_id,
        0,
        4,
        &vec![0u8; vertex_count * 4],
    );

    let single = frame_with(mesh_id, 1);
    let _ = renderer.prepare_callback(&device, &queue, &single);
    assert_eq!(
        renderer.last_frame_stats().deform_slots_ignored,
        0,
        "a single item draws per-object, which binds the mesh's deform group"
    );

    let many = frame_with(mesh_id, 4);
    let _ = renderer.prepare_callback(&device, &queue, &many);
    let stats = renderer.last_frame_stats();
    assert!(
        stats.instanced_batches > 0,
        "four items of one mesh should batch; the counter means nothing otherwise"
    );
    assert_eq!(
        stats.deform_slots_ignored, 4,
        "every instanced item carries slot data the instanced path drops"
    );
}
