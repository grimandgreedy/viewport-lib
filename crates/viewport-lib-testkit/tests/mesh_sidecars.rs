//! Upload-path checks: meshes near the device buffer limit upload and
//! render, oversized meshes are refused with a clean error instead of a
//! device loss, and the normals view still renders now that its line
//! buffer is built on first use rather than at upload time.

use viewport_lib::{CameraFrame, FrameData, Material, SceneFrame, SceneRenderItem, ViewportError};
use viewport_lib_testkit::{Harness, meshes, orbit_camera};

fn frame_for_item(item: SceneRenderItem, distance: f32, size: [f32; 2]) -> FrameData {
    let camera = orbit_camera(glam::Vec3::ZERO, distance, 0.6, 1.0);
    FrameData::new(
        CameraFrame::from_camera(&camera, size),
        SceneFrame::from_surface_items(vec![item]),
    )
}

/// A ~3.2M-vertex mesh fits the device's buffer limit and must upload and
/// render. This used to die on a wgpu validation error: the normal-line
/// sidecar (128 bytes per vertex, built eagerly for every upload) exceeded
/// `max_buffer_size` long before the mesh's own buffers did.
#[test]
fn big_mesh_uploads_within_device_limit() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let data = meshes::heightfield(1800, 1800, 100.0, 2.0);
    let expected_tris = (data.indices.len() / 3) as u64;
    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &data)
        .expect("a mesh within device limits must upload");

    let mut item = SceneRenderItem::default();
    item.mesh_id = id;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    let fd = frame_for_item(item, 150.0, [200.0, 150.0]);
    let stats = h.render_two_frames(&fd, 200, 150);
    assert_eq!(stats.triangles_submitted, expected_tris);
}

/// A mesh whose vertex buffer alone would exceed `max_buffer_size` is
/// refused with `MeshTooLarge`, and the device stays usable afterwards.
#[test]
fn oversized_mesh_is_refused_cleanly() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let max = h.device.limits().max_buffer_size;
    let data = meshes::heightfield(2100, 2100, 100.0, 2.0);
    let vertex_bytes = data.positions.len() as u64 * 64;
    assert!(
        vertex_bytes > max,
        "test premise: vertex buffer ({vertex_bytes}) must exceed the limit ({max})"
    );
    let err = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &data)
        .expect_err("an oversized mesh must be refused");
    assert!(matches!(err, ViewportError::MeshTooLarge { .. }), "{err:?}");

    // The refusal must leave the device alive: a normal upload and render
    // still work.
    let small = meshes::stress_sphere(1.0, 3);
    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &small)
        .expect("small upload after refusal");
    let mut item = SceneRenderItem::default();
    item.mesh_id = id;
    item.material = Material::from_colour([0.7, 0.7, 0.7]);
    let fd = frame_for_item(item, 4.0, [200.0, 150.0]);
    let stats = h.render_two_frames(&fd, 200, 150);
    assert_eq!(stats.triangles_submitted, (small.indices.len() / 3) as u64);
}

/// The normals view must still draw its lines: the buffer is built on the
/// first frame an item sets `show_normals`, so a frame with it must differ
/// from a frame without it.
#[test]
fn normals_view_renders_lazily() {
    let Some(mut h) = Harness::new() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let data = meshes::stress_sphere(1.0, 2);
    let id = h
        .renderer
        .resources_mut()
        .upload_mesh_data(&h.device, &data)
        .expect("upload");

    let mut plain = SceneRenderItem::default();
    plain.mesh_id = id;
    plain.material = Material::from_colour([0.7, 0.7, 0.7]);
    let mut with_normals = plain.clone();
    with_normals.show_normals = true;

    let base = h.render(&frame_for_item(plain, 4.0, [200.0, 150.0]), 200, 150);
    let lines = h.render(
        &frame_for_item(with_normals, 4.0, [200.0, 150.0]),
        200,
        150,
    );
    assert_ne!(
        base, lines,
        "show_normals produced no visible normal lines"
    );
}
