//! Frame-stats coverage for deform-slot writes.
//!
//! Two things are checked here. First, that a per-mesh slot write reports what
//! it cost: the write reallocates its buffer and rebuilds every bind group
//! pointing at it, and `FrameStats` has to show that rather than leave it
//! invisible. Second, that items carrying per-mesh slot data stay off the
//! instanced path, which is what keeps `deform_slots_ignored` at zero: the
//! instanced draws bind the empty deform group, so an item that reached them
//! would render undeformed with no other signal. Colocated with the renderer so
//! the tests can drive `prepare` directly. Skips when no wgpu adapter is
//! available.

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

/// Attaching a per-mesh slot reallocates the buffer and rebuilds every bind
/// group on the mesh, even when the new data is the same size, and the counters
/// say so.
///
/// That is correct for an attach, which is a structural change: a slot's length
/// may differ and its neighbours move with it inside the packed buffer. It is
/// the wrong call to make every frame, and
/// `write_deform_slot_range` is the one to use instead: see
/// `per_mesh_slot_range_write_reallocates_nothing` below, which is the same
/// scenario through that call.
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

/// The point of `write_deform_slot_range`: updating an attached slot every
/// frame costs no allocation and no bind group, where re-attaching costs both.
///
/// This is the counterpart to the test above, and the two together are the
/// argument for the call existing. A deformer driving a slot per frame was
/// paying a buffer allocation plus one bind group per instance on the mesh,
/// every frame, to move bytes that fit exactly where they already were.
#[test]
fn per_mesh_slot_range_write_reallocates_nothing() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping per_mesh_slot_range_write_reallocates_nothing: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
    let mesh = primitives::grid_plane(1.0, 1.0, 4, 4);
    let vertex_count = mesh.positions.len();
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();

    let payload = vec![0u8; vertex_count * 4];
    let fd = frame_with(mesh_id, 1);

    // Attach once, which is the structural call and is expected to allocate.
    let _ = renderer.prepare_callback(&device, &queue, &fd);
    renderer
        .resources_mut()
        .attach_deform_slot(&device, mesh_id, 0, 4, &payload);
    let _ = renderer.prepare_callback(&device, &queue, &fd);
    assert_eq!(renderer.last_frame_stats().deform_buffer_reallocations, 1);

    // Then update it the way a per-frame deformer should: whole slot first,
    // then a window inside it. Neither touches the buffer or a bind group.
    let next = vec![7u8; vertex_count * 4];
    renderer
        .resources_mut()
        .write_deform_slot_range(&queue, mesh_id, 0, 0, &next)
        .expect("whole-slot write");
    let _ = renderer.prepare_callback(&device, &queue, &fd);
    let whole = renderer.last_frame_stats();
    assert_eq!(
        whole.deform_buffer_reallocations, 0,
        "a whole-slot range write must not reallocate"
    );
    assert_eq!(whole.deform_mesh_bind_groups_rebuilt, 0);
    assert_eq!(whole.deform_instance_bind_groups_rebuilt, 0);

    let window = vec![3u8; 4 * 4];
    renderer
        .resources_mut()
        .write_deform_slot_range(&queue, mesh_id, 0, 2, &window)
        .expect("windowed write");
    let _ = renderer.prepare_callback(&device, &queue, &fd);
    let ranged = renderer.last_frame_stats();
    assert_eq!(ranged.deform_buffer_reallocations, 0);
    assert_eq!(ranged.deform_mesh_bind_groups_rebuilt, 0);
    assert_eq!(ranged.deform_instance_bind_groups_rebuilt, 0);
}

/// The range write updates the retained host copy as well as the GPU buffer, so
/// a later attach re-packs from current bytes instead of resurrecting the ones
/// the slot was established with.
///
/// Without this the two copies diverge silently: the picture stays right until
/// something else attaches a slot on the same mesh, and then the deformation
/// jumps back to whatever was attached first.
#[test]
fn a_range_write_is_visible_to_a_later_repack() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping a_range_write_is_visible_to_a_later_repack: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
    let mesh = primitives::grid_plane(1.0, 1.0, 4, 4);
    let vertex_count = mesh.positions.len();
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();

    let res = renderer.resources_mut();
    res.attach_deform_slot(&device, mesh_id, 0, 4, &vec![0u8; vertex_count * 4]);
    res.write_deform_slot_range(&queue, mesh_id, 0, 0, &vec![9u8; vertex_count * 4])
        .expect("range write");
    // Attaching a second slot re-packs the first one from its retained bytes.
    res.attach_deform_slot(&device, mesh_id, 1, 4, &vec![1u8; vertex_count * 4]);
    assert_eq!(
        res.deform_slot_bytes(mesh_id, 0),
        Some(vec![9u8; vertex_count * 4]),
        "the re-pack must see what the range write left, not what the attach established"
    );
}

/// The range write refuses what it cannot do, rather than writing somewhere
/// unexpected: a slot with nothing attached, and a window past the end.
#[test]
fn a_range_write_refuses_an_unattached_slot_or_an_overrun() {
    let Some((device, queue)) = headless_device() else {
        eprintln!(
            "skipping a_range_write_refuses_an_unattached_slot_or_an_overrun: no GPU adapter"
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
    let res = renderer.resources_mut();

    assert!(
        res.write_deform_slot_range(&queue, mesh_id, 0, 0, &[0u8; 4])
            .is_err(),
        "a slot with nothing attached has no length to write into"
    );

    res.attach_deform_slot(&device, mesh_id, 0, 4, &vec![0u8; vertex_count * 4]);
    assert!(
        res.write_deform_slot_range(&queue, mesh_id, 0, vertex_count as u32, &[0u8; 4])
            .is_err(),
        "a window starting at the end is past it"
    );
    assert!(
        res.write_deform_slot_range(&queue, mesh_id, 0, 0, &[0u8; 6])
            .is_err(),
        "six bytes is not a whole number of four-byte elements"
    );
    assert!(
        res.write_deform_slot_range(&queue, mesh_id, 0, 0, &[])
            .is_ok(),
        "an empty write is a no-op, not an error"
    );
}

/// Per-mesh deform data keeps its items off the instanced path however many of
/// them there are, so the deformation always reaches the draw and
/// `deform_slots_ignored` stays at zero. The control in the middle is the point
/// of the test: the same four items without slot data do batch, so the routing
/// change is the deform exclusion and not the scene shape.
#[test]
fn per_mesh_deform_items_stay_off_the_instanced_path() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping per_mesh_deform_items_stay_off_the_instanced_path: no GPU adapter");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, crate::gpu::TextureFormat::Bgra8UnormSrgb);
    let mesh = primitives::grid_plane(1.0, 1.0, 4, 4);
    let vertex_count = mesh.positions.len();
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();

    // Control: no slot data, four items, well past the instancing threshold.
    let many = frame_with(mesh_id, 4);
    let _ = renderer.prepare_callback(&device, &queue, &many);
    let control = renderer.last_frame_stats();
    assert!(
        control.instanced_batches > 0,
        "four plain items of one mesh should batch; without that the assertions \
         below would pass for the wrong reason"
    );
    assert_eq!(control.deform_slots_ignored, 0, "nothing is attached yet");

    renderer.resources_mut().attach_deform_slot(
        &device,
        mesh_id,
        0,
        4,
        &vec![0u8; vertex_count * 4],
    );

    // Same four items, now carrying per-mesh slot data.
    let _ = renderer.prepare_callback(&device, &queue, &many);
    let deformed = renderer.last_frame_stats();
    assert_eq!(
        deformed.instanced_batches, 0,
        "the deform exclusion takes every item off the instanced path"
    );
    assert_eq!(
        deformed.per_object_items, 4,
        "and they draw per-object, which binds the mesh's real deform group"
    );
    assert_eq!(
        deformed.deform_slots_ignored, 0,
        "so no item's slot data is dropped. A non-zero value here is a wrong \
         picture: the deformation the consumer attached is silently absent"
    );

    // A single item was never at risk, and still is not.
    let single = frame_with(mesh_id, 1);
    let _ = renderer.prepare_callback(&device, &queue, &single);
    assert_eq!(renderer.last_frame_stats().deform_slots_ignored, 0);
}
