//! Position override, external instances, occlusion culling, and LOD.
//!
//! Part of the headless integration suite (split from the former single
//! headless.rs). Shared device and mesh helpers live in tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

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
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
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

    // The lit red plane tone-maps to roughly [246, 60, 60]: with the lit
    // clamp open on the HDR path, the Khronos Neutral curve desaturates the
    // above-1.0 highlight, lifting the green/blue channels.
    let count_red = |pixels: &[u8]| -> usize {
        pixels
            .chunks_exact(4)
            .filter(|rgba| rgba[0] > 150 && rgba[1] < 100 && rgba[2] < 100)
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

/// An external scalar source must drive GPU marching cubes per frame: the
/// slab scalar buffers are refreshed from the consumer's buffer before every
/// dispatch. The uploaded CPU volume has no surface at the isovalue; writing
/// a sphere field into the external buffer makes one appear, and writing the
/// empty field again makes it vanish, proving both the copy and its
/// per-frame cadence.
#[test]
fn mc_external_scalar_drives_isosurface() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let dims = [16u32, 16, 16];
    let spacing = [0.3f32; 3];
    let origin = [-(15.0 * 0.3) / 2.0; 3];
    let node_count = (dims[0] * dims[1] * dims[2]) as usize;

    // Uploaded field: uniformly far above the isovalue, no surface anywhere.
    let vol = viewport_lib::VolumeData {
        data: vec![10.0; node_count],
        dims,
        origin,
        spacing,
    };
    let volume_id = renderer
        .resources_mut()
        .upload_volume_for_mc(&device, &queue, &vol)
        .expect("mc volume upload");

    // Sphere field: distance from the origin; isovalue 1.5 is a sphere well
    // inside the grid.
    let mut sphere = vec![0.0f32; node_count];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * spacing[0];
                let wy = origin[1] + y as f32 * spacing[1];
                let wz = origin[2] + z as f32 * spacing[2];
                let idx = (x + y * dims[0] + z * dims[0] * dims[1]) as usize;
                sphere[idx] = (wx * wx + wy * wy + wz * wz).sqrt();
            }
        }
    }

    let scalar_src = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_mc_scalar_src"),
        size: (node_count * 4) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let make_frame = || -> FrameData {
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
        frame
            .scene
            .gpu_mc_jobs
            .push(viewport_lib::GpuMarchingCubesJob {
                volume_id,
                isovalue: 1.5,
                material: Material::default(),
                settings: ItemSettings::default(),
                cpu_data: None,
            });
        frame
    };
    let empty_frame = || -> FrameData {
        let mut frame = make_frame();
        frame.scene.gpu_mc_jobs.clear();
        frame
    };
    let diff_count = |a: &[u8], b: &[u8]| -> usize {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(pa, pb)| pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8))
            .count()
    };

    let empty = renderer.render_offscreen(&device, &queue, &empty_frame(), 64, 64);

    // Baseline: uploaded field has no surface.
    let flat = renderer.render_offscreen(&device, &queue, &make_frame(), 64, 64);
    assert_eq!(
        diff_count(&flat, &empty),
        0,
        "the uploaded all-above-iso field must extract no surface"
    );

    // Attach the external source and write the sphere field into it.
    queue.write_buffer(&scalar_src, 0, bytemuck::cast_slice(&sphere));
    renderer
        .resources_mut()
        .set_mc_scalar_source_buffer(volume_id, scalar_src.clone(), 0)
        .unwrap();
    let with_sphere = renderer.render_offscreen(&device, &queue, &make_frame(), 64, 64);
    assert!(
        diff_count(&with_sphere, &empty) > 0,
        "after the external sphere field is copied in, the isosurface must \
         render"
    );

    // Rewrite the buffer with the empty field: the surface must vanish on
    // the next frame, proving the copy happens every dispatch.
    queue.write_buffer(
        &scalar_src,
        0,
        bytemuck::cast_slice(&vec![10.0f32; node_count]),
    );
    let flat_again = renderer.render_offscreen(&device, &queue, &make_frame(), 64, 64);
    assert_eq!(
        diff_count(&flat_again, &empty),
        0,
        "rewriting the external buffer with an all-above-iso field must \
         remove the surface on the next frame"
    );
}

/// An external instance set must draw exactly the window of the consumer's
/// positions buffer selected by the item's instance range. The buffer holds
/// four positions: elements 0..2 far behind the camera, elements 2..4 in
/// front of it. Drawing `first_instance = 2, instance_count = 2` must show
/// geometry; re-pointing the range at 0..2 must show none.
#[test]
fn external_instances_render_with_instance_range_slice() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .unwrap();

    // Elements 0 and 1 behind the camera, 2 and 3 visible near the origin.
    let positions: Vec<f32> = vec![
        0.0, 0.0, -1000.0, // 0
        0.0, 0.0, -1000.0, // 1
        -0.6, 0.0, 0.0, // 2
        0.6, 0.0, 0.0, // 3
    ];
    let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_external_positions"),
        size: (positions.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&positions));

    let set_id = renderer
        .resources_mut()
        .create_external_instance_set(
            &device,
            &viewport_lib::ExternalInstanceSetConfig::new(mesh_id, pos_buf),
        )
        .unwrap();

    let make_frame = |first: u32, count: u32| -> FrameData {
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
        let mut item = viewport_lib::ExternalInstancesItem::new(set_id, count);
        item.first_instance = first;
        item.scale = 0.4;
        item.colour = [1.0, 0.2, 0.2, 1.0];
        frame.scene.external_instances = vec![item];
        frame
    };

    let empty = renderer.render_offscreen(&device, &queue, &make_frame(0, 0), 64, 64);
    let visible = renderer.render_offscreen(&device, &queue, &make_frame(2, 2), 64, 64);
    let hidden = renderer.render_offscreen(&device, &queue, &make_frame(0, 2), 64, 64);

    let diff_count = |a: &[u8], b: &[u8]| -> usize {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(pa, pb)| pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8))
            .count()
    };

    assert!(
        diff_count(&visible, &empty) > 0,
        "instance range 2..4 selects the visible elements; the boxes must \
         render. If nothing shows, instance_index is not honouring the draw \
         call's first_instance or the storage window is wrong.",
    );
    assert_eq!(
        diff_count(&hidden, &empty),
        0,
        "instance range 0..2 selects only the behind-camera elements; the \
         image must match an empty scene.",
    );
}

/// The selection outline mask must rasterise override-driven geometry, not
/// the bind pose. A selected plane's override pushes every vertex far behind
/// the camera: the mesh vanishes, and the halo must vanish with it. If the
/// mask still reads the vertex buffer, a ghost outline of the bind-pose
/// silhouette remains in an otherwise geometry-free image.
#[test]
fn outline_mask_follows_position_override() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3];
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &mesh)
        .unwrap();

    let make_frame = |items: Vec<SceneRenderItem>| -> FrameData {
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
        frame.interaction.outline_selected = true;
        frame.scene.surfaces = SurfaceSubmission::Flat(items.into());
        frame
    };

    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    item.material = Material::from_colour([1.0, 0.0, 0.0]);
    item.settings.selected = true;
    let frame = make_frame(vec![item]);

    // Empty reference: what the frame looks like with nothing drawn at all.
    let empty_frame = make_frame(Vec::new());
    let empty = renderer.render_offscreen(&device, &queue, &empty_frame, 64, 64);

    let diff_count = |a: &[u8], b: &[u8]| -> usize {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(pa, pb)| pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8))
            .count()
    };

    // Sanity: selected plane with no override differs from empty (mesh +
    // outline are drawn).
    let baseline = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    assert!(
        diff_count(&baseline, &empty) > 0,
        "baseline render should draw the selected plane and its outline"
    );

    // Override pushes the whole plane far behind the camera.
    let displaced: Vec<f32> = (0..4).flat_map(|_| [0.0_f32, 0.0, -1000.0]).collect();
    let override_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_outline_override"),
        size: (displaced.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&override_buf, 0, bytemuck::cast_slice(&displaced));
    renderer
        .resources_mut()
        .set_position_override_buffer(mesh_id, override_buf)
        .unwrap();

    let overridden = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let ghost = diff_count(&overridden, &empty);
    if ghost != 0 {
        for (i, (pa, pb)) in overridden
            .chunks_exact(4)
            .zip(empty.chunks_exact(4))
            .enumerate()
        {
            if pa.iter().zip(pb.iter()).any(|(&x, &y)| x.abs_diff(y) > 8) {
                eprintln!(
                    "diff at ({}, {}): got {:?} empty {:?}",
                    i % 64,
                    i / 64,
                    pa,
                    pb
                );
            }
        }
    }
    assert_eq!(
        ghost, 0,
        "with the override pushing the selected plane off-screen the image \
         must match an empty scene; {ghost} differing pixels indicate the \
         outline mask (or mesh) still rasterises the bind pose.",
    );
}

/// A sliced override binding must read its window out of a pooled buffer, not
/// element 0. The pool holds two "bodies": elements 0..5 push vertices far
/// behind the camera, elements 5..9 hold the plane's real (visible)
/// positions. `base_element = 5` is deliberately not 256-byte aligned
/// (5 * 12 = 60 bytes): the window is applied by the shader via
/// `position_override_base`, not a buffer binding offset.
#[test]
fn position_override_slice_reads_correct_pool_window() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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

    let mut red_item = SceneRenderItem::default();
    red_item.mesh_id = red_id;
    red_item.model = glam::Mat4::IDENTITY.to_cols_array_2d();
    red_item.material = Material::from_colour([1.0, 0.0, 0.0]);
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![red_item].into());

    // Pool: body A (elements 0..5) far behind the camera, body B (elements
    // 5..9) the plane's own corner positions.
    let mut pool_data: Vec<f32> = Vec::new();
    for _ in 0..5 {
        pool_data.extend_from_slice(&[0.0, 0.0, -1000.0]);
    }
    for p in &mesh.positions {
        pool_data.extend_from_slice(p);
    }
    let pool = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_override_pool"),
        size: (pool_data.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&pool, 0, bytemuck::cast_slice(&pool_data));

    let count_red = |pixels: &[u8]| -> usize {
        pixels
            .chunks_exact(4)
            .filter(|rgba| rgba[0] > 150 && rgba[1] < 100 && rgba[2] < 100)
            .count()
    };

    // Window over body B: the plane renders from the pool's tail.
    renderer
        .resources_mut()
        .set_position_override_buffer_sliced(red_id, pool.clone(), OverrideBufferSlice::new(5, 4))
        .unwrap();
    let body_b = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let body_b_red = count_red(&body_b);
    assert!(
        body_b_red > 0,
        "slice base_element = 5 should read the visible body B positions; \
         got {body_b_red} red pixels. If this regresses, the shader is \
         ignoring position_override_base and reading body A (off-screen) \
         from element 0.",
    );

    // Re-point the window at body A: everything moves behind the camera.
    renderer
        .resources_mut()
        .set_position_override_buffer_sliced(red_id, pool, OverrideBufferSlice::new(0, 4))
        .unwrap();
    let body_a = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
    let body_a_red = count_red(&body_a);
    assert_eq!(
        body_a_red, 0,
        "slice base_element = 0 covers body A (all vertices at z = -1000); \
         no red pixels should remain (body B window drew {body_b_red}).",
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
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
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
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
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
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
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
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
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

