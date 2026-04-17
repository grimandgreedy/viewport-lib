//! Headless integration tests for viewport-lib.
//!
//! These tests create a real wgpu device (headless) and exercise the GPU
//! resource APIs. Requires a GPU adapter (software or hardware).

use viewport_lib::{
    Camera, Material, MeshId, Scene, Selection,
    error::ViewportError,
    renderer::{FrameData, LightingSettings, SceneRenderItem, ViewportRenderer},
    resources::{CameraUniform, MeshData},
};

/// Create a headless wgpu device + queue for testing.
fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Simple unit box mesh data for testing.
fn box_mesh() -> MeshData {
    let positions = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let normals = vec![
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![
        0, 1, 2, 2, 3, 0, 4, 6, 5, 6, 4, 7, 0, 3, 7, 7, 4, 0, 1, 5, 6, 6, 2, 1, 3, 2, 6, 6, 7, 3,
        0, 4, 5, 5, 1, 0,
    ];
    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh
}

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
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let result = renderer
        .resources_mut()
        .replace_mesh_data(&device, 999, &box_mesh());
    assert!(matches!(
        result.unwrap_err(),
        ViewportError::MeshIndexOutOfBounds { index: 999, .. }
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
    frame.camera_uniform = viewport_lib::resources::CameraUniform {
        view_proj: cam.view_proj_matrix().to_cols_array_2d(),
        eye_pos: cam.eye_position().to_array(),
        _pad: 0.0,
        forward: [0.0, 0.0, -1.0],
        _pad1: 0.0,
    };
    frame.lighting = LightingSettings::default();
    frame.eye_pos = cam.eye_position().to_array();
    frame.scene_items = vec![];
    frame.show_grid = false;
    frame.show_axes_indicator = false;
    frame.viewport_size = [0.0, 0.0];
    frame.camera_orientation = cam.orientation;
    // Should not panic.
    renderer.prepare(&device, &queue, &frame);
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

    let removed = renderer.resources_mut().remove_mesh(idx);
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
    renderer.resources_mut().remove_mesh(idx1);

    // Next upload should reuse the freed slot.
    let idx2 = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    assert_eq!(idx1, idx2, "freed slot should be reused");
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
        Some(MeshId::from_index(mesh_idx)),
        glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0)),
        Material::default(),
    );

    let mut sel = Selection::new();
    sel.select_one(node_id);

    let items = scene.collect_render_items(&sel);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].mesh_index, mesh_idx);
    assert!(items[0].selected);
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
    frame.camera_uniform = CameraUniform {
        view_proj: cam.view_proj_matrix().to_cols_array_2d(),
        eye_pos: cam.eye_position().to_array(),
        _pad: 0.0,
        forward: [0.0, 0.0, -1.0],
        _pad1: 0.0,
    };
    frame.eye_pos = cam.eye_position().to_array();
    frame.camera_orientation = cam.orientation;
    frame.viewport_size = [64.0, 64.0];
    frame.camera_aspect = 1.0;
    frame.show_grid = false;
    frame.show_axes_indicator = false;
    // Add the box as a scene item.
    let mut item = SceneRenderItem::default();
    item.mesh_index = mesh_idx;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.selected = false;
    frame.scene_items.push(item);

    let width = 64u32;
    let height = 64u32;
    let pixels = renderer.render_offscreen(&device, &queue, &frame, width, height);

    // Must be exactly width * height * 4 RGBA bytes.
    assert_eq!(pixels.len(), (width * height * 4) as usize);

    // At least some pixels should be non-zero (the mesh or background).
    let has_nonzero = pixels.iter().any(|&b| b != 0);
    assert!(has_nonzero, "offscreen render produced all-zero image");
}
