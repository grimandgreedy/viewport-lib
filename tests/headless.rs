//! Headless integration tests for viewport-lib.
//!
//! These tests create a real wgpu device (headless) and exercise the GPU
//! resource APIs. Requires a GPU adapter (software or hardware).

use viewport_lib::{
    Camera, Material, MeshId, Scene, Selection,
    error::ViewportError,
    renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, ViewportRenderer},
    resources::MeshData,
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
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let result = renderer.resources_mut().replace_mesh_data(
        &device,
        &queue,
        MeshId::from_index(999),
        &box_mesh(),
    );
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
    frame.camera.render_camera = RenderCamera {
        view: cam.view_matrix(),
        projection: cam.proj_matrix(),
        eye_position: cam.eye_position().to_array(),
        forward: [0.0, 0.0, -1.0],
        orientation: cam.orientation,
        near: cam.effective_znear(),
        far: cam.zfar,
        distance: cam.distance,
        fov: cam.fov_y,
        aspect: cam.aspect,
    };
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
    frame.camera.render_camera = RenderCamera {
        view: cam.view_matrix(),
        projection: cam.proj_matrix(),
        eye_position: cam.eye_position().to_array(),
        forward: [0.0, 0.0, -1.0],
        orientation: cam.orientation,
        near: cam.effective_znear(),
        far: cam.zfar,
        distance: cam.distance,
        fov: cam.fov_y,
        aspect: 1.0,
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

/// Regression test for the silent-skip bug where a `set_position_override_buffer`
/// binding would render nothing when the item was routed through the instanced
/// pipeline. `mesh_instanced.wgsl` has no awareness of the override binding,
/// so the consumer's compute output is dropped on the floor. The fix is twofold:
///   1. Items with a bound override are excluded from the instanced batches
///      (forced through the per-object pipeline that does know about overrides).
///   2. The per-item `ObjectUniform` write loop is entered (and the override
///      item is not skipped within it) so `has_position_override = 1` reaches
///      the shader.
///
/// The test renders one red plane with an override displacing every vertex
/// far behind the camera. A second decoy item is added so the visible-item
/// count exceeds `INSTANCING_THRESHOLD = 1` and the instanced pipeline is
/// actually engaged. If the bug returns, the red plane remains visible
/// because the instanced shader ignores the override entirely.
#[test]
fn position_override_takes_effect_through_render_path() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Two simple plane meshes: a red "test" plane that will get the override,
    // and a blue "decoy" plane that exists only to push the visible-item
    // count past INSTANCING_THRESHOLD = 1 so the instanced pipeline engages.
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    let red_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();
    let blue_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();

    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera {
        view: cam.view_matrix(),
        projection: cam.proj_matrix(),
        eye_position: cam.eye_position().to_array(),
        forward: [0.0, 0.0, -1.0],
        orientation: cam.orientation,
        near: cam.effective_znear(),
        far: cam.zfar,
        distance: cam.distance,
        fov: cam.fov_y,
        aspect: 1.0,
    };
    frame.camera.viewport_size = [64.0, 64.0];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;

    // Red plane (target of the override) at the origin; blue decoy off to the
    // side so it doesn't overdraw the red region we measure.
    let mut red_item = SceneRenderItem::default();
    red_item.mesh_id = red_id;
    red_item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    red_item.material = Material::from_colour([1.0, 0.0, 0.0]);

    let mut blue_item = SceneRenderItem::default();
    blue_item.mesh_id = blue_id;
    blue_item.model = glam::Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0))
        .to_cols_array_2d();
    blue_item.material = Material::from_colour([0.0, 0.0, 1.0]);

    frame.scene.surfaces = SurfaceSubmission::Flat(
        vec![red_item.clone(), blue_item.clone()].into(),
    );

    // ---- Render 1: no override. The red plane should be visible. ----
    let baseline = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    let count_red = |pixels: &[u8]| -> usize {
        pixels
            .chunks_exact(4)
            .filter(|rgba| rgba[0] > 50 && rgba[1] < 30 && rgba[2] < 30)
            .count()
    };
    let baseline_red = count_red(&baseline);
    assert!(
        baseline_red > 0,
        "baseline render should show the red plane; got {baseline_red} red pixels",
    );

    // ---- Render 2: bind an override on the red plane that pushes every
    // vertex far behind the camera. If the fix is in place, the red plane
    // disappears regardless of whether instancing is active. If the bug
    // returns, the red plane stays put because the instanced shader ignores
    // the override.
    let displaced: Vec<f32> = (0..4)
        .flat_map(|_| [0.0_f32, 0.0, -1000.0])
        .collect();
    let override_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_position_override"),
        size: (displaced.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&override_buf, 0, bytemuck::cast_slice(&displaced));
    renderer
        .resources_mut()
        .set_position_override_buffer(red_id, override_buf)
        .unwrap();

    let overridden = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let overridden_red = count_red(&overridden);

    assert_eq!(
        overridden_red, 0,
        "with the position override pushing the red plane's vertices off-screen,\n\
         no red pixels should remain. Got {overridden_red} red (baseline had \
         {baseline_red}). If this regresses, the item was routed through the \
         instanced pipeline (`mesh_instanced.wgsl`) which has no awareness of \
         `has_position_override`, OR the per-item ObjectUniform write was \
         skipped so the shader flag stayed at 0.",
    );
}
