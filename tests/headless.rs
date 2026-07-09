//! Headless integration tests for viewport-lib.
//!
//! These tests create a real wgpu device (headless) and exercise the GPU
//! resource APIs. Requires a GPU adapter (software or hardware).

use viewport_lib::{
    BackfacePolicy, Camera, Material, MeshId, Scene, Selection,
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
    blue_item.model =
        glam::Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0)).to_cols_array_2d();
    blue_item.material = Material::from_colour([0.0, 0.0, 1.0]);

    frame.scene.surfaces =
        SurfaceSubmission::Flat(vec![red_item.clone(), blue_item.clone()].into());

    // ---- Render 1: no override. The red plane should be visible. ----
    let baseline = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    // The lit red plane tone-maps to roughly [241, 34, 34]; the Khronos
    // Neutral curve lifts the dark channels a little, so the green/blue
    // bounds sit above that rather than near zero.
    let count_red = |pixels: &[u8]| -> usize {
        pixels
            .chunks_exact(4)
            .filter(|rgba| rgba[0] > 100 && rgba[1] < 60 && rgba[2] < 60)
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
    let displaced: Vec<f32> = (0..4).flat_map(|_| [0.0_f32, 0.0, -1000.0]).collect();
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

/// Exercise the HiZ occlusion-cull path end to end: enabling occlusion culling
/// builds the HiZ pyramid from the depth written by the HDR scene pass (which
/// validates `hiz_copy.wgsl` / `hiz_reduce.wgsl` compile and run), and the next
/// frame's cull samples it (validating the extended `cull.wgsl`). Several boxes
/// are submitted so the instanced cull path engages, and a near box sits
/// directly in front of farther ones along the view direction so there is real
/// front-to-back occlusion to find.
///
/// The test asserts the render does not panic and produces pixels. When the
/// device supports GPU-driven culling, it also checks the cull breakdown is
/// monotonic: total >= frustum-survivors >= drawn.
#[test]
fn occlusion_culling_render_path_runs() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    renderer.set_occlusion_culling(true);
    assert!(renderer.occlusion_culling_enabled());

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
    // HiZ is built in the HDR scene pass, so post-processing must be on.
    frame.effects.post_process.enabled = true;

    // A column of boxes along the view direction (Z-up world, camera looks down
    // -Z by default here): a big near box and several smaller ones behind it.
    let mut items = Vec::new();
    for i in 0..8 {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_idx;
        let z = -(i as f32) * 1.5;
        let s = if i == 0 { 4.0 } else { 1.0 };
        item.model = (glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, z))
            * glam::Mat4::from_scale(glam::Vec3::splat(s)))
        .to_cols_array_2d();
        items.push(item);
    }
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    // Render several frames, nudging the camera each frame so the reprojection
    // uses a non-identity previous-to-current transform (exercises the inverse
    // view-projection path, not just a static-camera no-op). Frame 0 stores the
    // first depth, later frames reproject it and cull; the stats readback lags a
    // couple of frames before it lands.
    let base_view = frame.camera.render_camera.view;
    let mut last_pixels = Vec::new();
    for i in 0..4 {
        let nudge = glam::Mat4::from_translation(glam::Vec3::new(0.05 * i as f32, 0.0, 0.0));
        frame.camera.render_camera.view = base_view * nudge;
        last_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    }
    assert_eq!(last_pixels.len(), 64 * 64 * 4);
    assert!(
        last_pixels.iter().any(|&b| b != 0),
        "occlusion render produced an all-zero image",
    );

    let stats = renderer.last_frame_stats();
    if stats.gpu_culling_active {
        let total = stats.gpu_culled_total.expect("total should be read back");
        let frustum = stats
            .gpu_frustum_visible
            .expect("frustum-survivors should be read back");
        let drawn = stats
            .gpu_visible_instances
            .expect("drawn count should be read back");
        assert!(
            total >= frustum && frustum >= drawn,
            "cull breakdown must be monotonic: total={total} frustum={frustum} drawn={drawn}",
        );
        assert_eq!(total, 8, "all 8 instances should enter the cull");
    }
}

/// Regression test: the HiZ reprojection passes must use 2D dispatches. At a
/// large depth resolution a 1D dispatch over `w * h / 64` exceeds the 65535
/// per-dimension workgroup limit and wgpu raises a validation error. 2048x2048
/// gives 65536 workgroups, one over the limit, so this would panic before the
/// fix. Renders two frames so the reprojection (which needs a stored prior
/// depth) actually runs.
#[test]
fn occlusion_large_viewport_no_dispatch_overflow() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    renderer.set_occlusion_culling(true);

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
    let dim = 2048u32;
    frame.camera.viewport_size = [dim as f32, dim as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.effects.post_process.enabled = true;

    let mut items = Vec::new();
    for i in 0..4 {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_idx;
        item.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -(i as f32) * 1.5))
            .to_cols_array_2d();
        items.push(item);
    }
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    // Frame 0 stores depth; frame 1 reprojects it (runs the init/scatter passes
    // that would overflow with a 1D dispatch). No panic == pass.
    for _ in 0..2 {
        let _ = renderer.render_offscreen(&device, &queue, &frame, dim, dim);
    }
}

/// Occlusion culling must run on the LDR path too, not just HDR. With
/// post-processing off, the scene renders through `render_frame_ldr`, which now
/// keeps the scene depth and copies it into the HiZ prev-depth target. This
/// exercises that store path (sampling the LDR depth target, which required
/// adding TEXTURE_BINDING + a depth-only view) across several moving-camera
/// frames; a no-panic run with non-zero output is the pass.
#[test]
fn occlusion_culling_ldr_path_runs() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_idx = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    renderer.set_occlusion_culling(true);

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
    // LDR path: post-processing OFF (this is the path the fix targets).
    frame.effects.post_process.enabled = false;

    let mut items = Vec::new();
    for i in 0..8 {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_idx;
        let z = -(i as f32) * 1.5;
        let s = if i == 0 { 4.0 } else { 1.0 };
        item.model = (glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, z))
            * glam::Mat4::from_scale(glam::Vec3::splat(s)))
        .to_cols_array_2d();
        items.push(item);
    }
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    let base_view = frame.camera.render_camera.view;
    let mut last_pixels = Vec::new();
    for i in 0..4 {
        let nudge = glam::Mat4::from_translation(glam::Vec3::new(0.05 * i as f32, 0.0, 0.0));
        frame.camera.render_camera.view = base_view * nudge;
        last_pixels = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    }
    assert_eq!(last_pixels.len(), 64 * 64 * 4);
    assert!(
        last_pixels.iter().any(|&b| b != 0),
        "LDR occlusion render produced an all-zero image",
    );
}

/// Regression test for the per-object LOD draw path: a LOD item that is culled
/// (below its group's `cull_below` size) must actually be skipped by the paint
/// pass, not just by the stats. The bug was that LOD resolve mutated a throwaway
/// copy while the per-object draw re-read the raw `frame.scene.surfaces`, so a
/// culled non-instanced item still drew at full detail. A `DifferentColour`
/// back-face policy forces the item onto the per-object path (the path the fix
/// targets). A culled item must render identically to an empty scene.
#[test]
fn lod_culled_per_object_item_is_not_drawn() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let full = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let crude = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();
    let group = renderer
        .resources_mut()
        .register_lod_group(&[full, crude], &[0.5, 0.0])
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
    frame.effects.post_process.enabled = true;

    // A single non-instanced (per-object) LOD item filling the view centre. The
    // DifferentColour back-face policy is what forces it onto the per-object path.
    let mut item = SceneRenderItem::default();
    item.mesh_id = full;
    item.lod_group = Some(group);
    item.material.backface_policy = BackfacePolicy::DifferentColour([1.0, 0.2, 0.2]);
    item.model = glam::Mat4::from_scale(glam::Vec3::splat(3.0)).to_cols_array_2d();

    // Empty scene baseline.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    let empty = renderer.render_offscreen(&device, &queue, &frame, 64, 64);

    // cull_below defaults to None, so the item draws.
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let drawn = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_ne!(
        drawn, empty,
        "control: the per-object LOD item should be drawn when not culled",
    );

    // Force the item below the cull size: it must now contribute nothing.
    renderer
        .resources_mut()
        .set_lod_cull_below(group, Some(100.0))
        .unwrap();
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![item.clone()].into());
    let culled = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert_eq!(
        culled, empty,
        "a culled per-object LOD item must be skipped in the paint pass",
    );
}
