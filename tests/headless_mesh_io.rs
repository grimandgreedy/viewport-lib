//! Mesh upload, slot reuse, and offscreen render smoke tests.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;


#[test]
fn upload_mesh_data_valid() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let result = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh());
    assert!(result.is_ok());
}

#[test]
fn upload_mesh_data_empty() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let empty = MeshData::default();
    let result = renderer.resources_mut().upload_mesh_data(&device, &empty);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::EmptyMesh { .. }
    ));
}

#[test]
fn upload_mesh_data_length_mismatch() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut bad = MeshData::default();
    bad.positions = vec![[0.0; 3], [1.0; 3]];
    bad.normals = vec![[0.0; 3]]; // mismatched length
    bad.indices = vec![0, 1, 0];
    let result = renderer.resources_mut().upload_mesh_data(&device, &bad);
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::MeshLengthMismatch { .. }
    ));
}

#[test]
fn upload_mesh_data_invalid_index() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut bad = MeshData::default();
    bad.positions = vec![[0.0; 3], [1.0; 3], [2.0; 3]];
    bad.normals = vec![[0.0; 3]; 3];
    bad.indices = vec![0, 1, 99]; // 99 is out of bounds
    let result = renderer.resources_mut().upload_mesh_data(&device, &bad);
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::InvalidVertexIndex {
            vertex_index: 99,
            ..
        }
    ));
}

#[test]
fn replace_mesh_data_bad_index() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let result =
        renderer
            .resources_mut()
            .replace_mesh_data(&device, &queue, MeshId::INVALID, &box_mesh());
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::StaleHandle { .. }
    ));
}

#[test]
fn prepare_empty_scene_no_panic() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(&cam);
    frame.camera.viewport_size = [0.0, 0.0];
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    // Should not panic.
    let _ = renderer.pass().prepare(&device, &queue, &frame);
}

#[test]
fn test_remove_mesh_frees_slot() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    assert!(renderer.resources().mesh(idx).is_some());

    let removed = renderer.resources_mut().free_mesh(idx);
    assert!(removed);
    assert!(renderer.resources().mesh(idx).is_none());
}

#[test]
fn test_upload_reuses_freed_slot() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let idx1 = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    renderer.resources_mut().free_mesh(idx1);

    // Next upload should reuse the freed slot, but at a new generation so the
    // old handle no longer matches.
    let idx2 = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    assert_eq!(idx1.index(), idx2.index(), "freed slot should be reused");
    assert_ne!(
        idx1, idx2,
        "reused slot must carry a new generation so the old handle differs"
    );
}

#[test]
fn test_scene_collect_render_items_roundtrip() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    let mut scene = Scene::new();
    let node_id = scene.add(
        Some(mesh_idx),
        glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0)),
        Material::default(),
    );

    let mut sel = Selection::new();
    sel.select_one(node_id);

    let items = scene.collect_render_items(&sel);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].mesh_id, mesh_idx);
    assert!(items[0].settings.selected);
    // Verify position is in the model matrix.
    let pos_x = items[0].model[3][0];
    assert!((pos_x - 1.0).abs() < 1e-5, "model[3][0] = {pos_x}");
}

#[test]
fn render_offscreen_produces_rgba_pixels() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Use Rgba8UnormSrgb so no BGRA swizzle complicates assertions.
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Upload a mesh so the scene is non-trivial.
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

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
    // Add the box as a scene item.
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.settings.selected = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item].into());

    let width = 64u32;
    let height = 64u32;
    let pixels = renderer.render_offscreen(&device, &queue, &frame, width, height);

    // Must be exactly width * height * 4 RGBA bytes.
    assert_eq!(pixels.len(), (width * height * 4) as usize);

    // At least some pixels should be non-zero (the mesh or background).
    let has_nonzero = pixels.iter().any(|&b| b != 0);
    assert!(has_nonzero, "offscreen render produced all-zero image");
}

