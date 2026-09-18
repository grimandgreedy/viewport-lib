//! The GPU marching cubes item type: the volumes it holds on the consumer's
//! behalf, plus picking (GPU pick-id against the compute-generated isosurface,
//! the CPU ray-march, and rect select).
//!
//! One file per item type, so a type's coverage travels with it.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

#[test]
fn gpu_pick_hits_marching_cubes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // A radial scalar field centred at the origin: the isosurface at value 1.5 is
    // a sphere of radius 1.5. Grid spans roughly [-2.3, 2.3]^3.
    let dims = [24u32, 24, 24];
    let spacing = [0.2f32; 3];
    let origin = [-(23.0 * 0.2) / 2.0; 3];
    let mut data = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * spacing[0];
                let wy = origin[1] + y as f32 * spacing[1];
                let wz = origin[2] + z as f32 * spacing[2];
                let idx = (x + y * dims[0] + z * dims[0] * dims[1]) as usize;
                data[idx] = (wx * wx + wy * wy + wz * wz).sqrt();
            }
        }
    }
    let vol = viewport_lib::VolumeData {
        data,
        dims,
        origin,
        spacing,
    };
    let volume_id = renderer
        .upload_volume_for_mc(&device, &queue, &vol)
        .expect("mc volume upload");

    let mut job = viewport_lib::GpuMarchingCubesItem {
        volume_id,
        isovalue: 1.5,
        material: Material::default(),
        settings: ItemSettings::default(),
        cpu_data: None,
    };
    job.settings.pick_id = PickId(717);
    frame.scene.gpu_mc_items.push(job);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(717)));
}

/// The CPU pick marches `cpu_data`, not the compute output, so an item without
/// it is GPU-pickable but not CPU-pickable.
#[test]
fn cpu_pick_hits_marching_cubes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let vol = std::sync::Arc::new(radial_field());
    let volume_id = renderer
        .upload_volume_for_mc(&device, &queue, &vol)
        .expect("mc volume upload");

    let mut job = viewport_lib::GpuMarchingCubesItem {
        volume_id,
        isovalue: 1.5,
        material: Material::default(),
        settings: ItemSettings::default(),
        cpu_data: Some(vol),
    };
    job.settings.pick_id = PickId(718);
    frame.scene.gpu_mc_items.push(job);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let vp = glam::Vec2::new(64.0, 64.0);
    let view_proj = frame.camera.render_camera.view_proj();

    let hit = renderer.pick(glam::Vec2::new(32.0, 32.0), vp, view_proj, PickMask::OBJECT);
    assert_eq!(hit.map(|h| h.id), Some(718));

    let miss = renderer.pick(glam::Vec2::new(1.0, 1.0), vp, view_proj, PickMask::OBJECT);
    assert!(miss.is_none(), "corner ray should miss the isosurface");
}

#[test]
fn rect_pick_hits_marching_cubes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let vol = std::sync::Arc::new(radial_field());
    let volume_id = renderer
        .upload_volume_for_mc(&device, &queue, &vol)
        .expect("mc volume upload");

    let mut job = viewport_lib::GpuMarchingCubesItem {
        volume_id,
        isovalue: 1.5,
        material: Material::default(),
        settings: ItemSettings::default(),
        cpu_data: Some(vol),
    };
    job.settings.pick_id = PickId(719);
    frame.scene.gpu_mc_items.push(job);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let vp = glam::Vec2::new(64.0, 64.0);
    let view_proj = frame.camera.render_camera.view_proj();

    let result = renderer.pick_rect(
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        vp,
        view_proj,
        PickMask::OBJECT,
    );
    assert!(
        result.objects.contains(&719),
        "full-viewport rect should select the isosurface, got {:?}",
        result.objects
    );

    let away = renderer.pick_rect(
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(2.0, 2.0),
        vp,
        view_proj,
        PickMask::OBJECT,
    );
    assert!(!away.objects.contains(&719));
}

/// A radial scalar field centred at the origin: the isosurface at 1.5 is a
/// sphere of radius 1.5, spanning roughly [-2.3, 2.3]^3.
fn radial_field() -> viewport_lib::VolumeData {
    let dims = [24u32, 24, 24];
    let spacing = [0.2f32; 3];
    let origin = [-(23.0 * 0.2) / 2.0; 3];
    let mut data = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let wx = origin[0] + x as f32 * spacing[0];
                let wy = origin[1] + y as f32 * spacing[1];
                let wz = origin[2] + z as f32 * spacing[2];
                let idx = (x + y * dims[0] + z * dims[0] * dims[1]) as usize;
                data[idx] = (wx * wx + wy * wy + wz * wz).sqrt();
            }
        }
    }
    viewport_lib::VolumeData {
        data,
        dims,
        origin,
        spacing,
    }
}

// ---------------------------------------------------------------------------
// The volumes the item type holds
// ---------------------------------------------------------------------------

/// A small field with an isosurface somewhere in the middle of it.
fn sample_volume() -> viewport_lib::VolumeData {
    let dims = [8u32, 8, 8];
    let data = (0..(dims[0] * dims[1] * dims[2]))
        .map(|i| (i % 2) as f32)
        .collect();
    viewport_lib::VolumeData {
        data,
        dims,
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0, 1.0, 1.0],
    }
}

/// A buffer large enough to hold `sample_volume`'s scalars past `offset`.
fn scalar_buffer(device: &wgpu::Device, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_scalar_src"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    })
}

#[test]
fn a_stale_mc_volume_handle_does_not_alias_after_slot_reuse() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let id1 = renderer
        .upload_volume_for_mc(&device, &queue, &sample_volume())
        .expect("upload a volume");
    renderer.free_mc_volume(id1);

    // The next upload reuses the freed slot at a new generation.
    let id2 = renderer
        .upload_volume_for_mc(&device, &queue, &sample_volume())
        .expect("upload a second volume");
    assert_ne!(id1, id2, "the reused slot must carry a new generation");

    // The live handle resolves; the stale one does not, so it cannot reach the
    // volume now occupying its slot.
    renderer
        .clear_mc_scalar_source(id2)
        .expect("a live handle resolves");
    assert!(matches!(
        renderer.clear_mc_scalar_source(id1),
        Err(viewport_lib::error::ViewportError::StaleHandle { .. })
    ));
}

#[test]
fn mc_volume_bytes_are_reported_and_reclaimed() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let baseline = renderer.resident_bytes().plugin_bytes;

    let id = renderer
        .upload_volume_for_mc(&device, &queue, &sample_volume())
        .expect("upload a volume");
    assert!(
        renderer.resident_bytes().plugin_bytes > baseline,
        "an uploaded volume must count toward the plugin working set"
    );

    renderer.free_mc_volume(id);
    assert_eq!(
        renderer.resident_bytes().plugin_bytes,
        baseline,
        "freeing a volume must drop its slab buffers out of the resident total"
    );
}

#[test]
fn the_mc_scalar_source_round_trips_and_rejects_bad_inputs() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let id = renderer
        .upload_volume_for_mc(&device, &queue, &sample_volume())
        .expect("upload a volume");

    // 8x8x8 volume = 512 nodes = 2048 bytes; the source sits at offset 64.
    let buf = scalar_buffer(
        &device,
        2048 + 64,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    renderer
        .set_mc_scalar_source_buffer(id, buf, 64)
        .expect("a large enough COPY_SRC buffer is accepted");
    renderer.clear_mc_scalar_source(id).expect("and detaches");

    // Too small.
    let small = scalar_buffer(&device, 128, wgpu::BufferUsages::COPY_SRC);
    assert!(matches!(
        renderer.set_mc_scalar_source_buffer(id, small, 0),
        Err(viewport_lib::error::ViewportError::McScalarSourceMismatch {
            needed_bytes: 2048,
            available_bytes: 128,
            ..
        })
    ));

    // Misaligned offset.
    let buf = scalar_buffer(&device, 4096, wgpu::BufferUsages::COPY_SRC);
    assert!(matches!(
        renderer.set_mc_scalar_source_buffer(id, buf, 2),
        Err(viewport_lib::error::ViewportError::McScalarSourceMismatch { .. })
    ));

    // Missing COPY_SRC usage.
    let storage_only = scalar_buffer(&device, 4096, wgpu::BufferUsages::STORAGE);
    assert!(matches!(
        renderer.set_mc_scalar_source_buffer(id, storage_only, 0),
        Err(
            viewport_lib::error::ViewportError::ExternalBufferUsageMissing {
                missing: "COPY_SRC"
            }
        )
    ));
}

#[test]
fn begin_upload_volume_for_mc_drains_to_a_handle() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let job = renderer.begin_upload_volume_for_mc(&device, &queue, sample_volume());
    for _ in 0..200 {
        renderer.resources_mut().process_uploads(&device, &queue);
        match renderer.resources().upload_status(job) {
            viewport_lib::resources::UploadStatus::Ready => break,
            viewport_lib::resources::UploadStatus::Failed(e) => panic!("upload failed: {e:?}"),
            viewport_lib::resources::UploadStatus::Pending { .. } => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            viewport_lib::resources::UploadStatus::Unknown => panic!("job id disappeared"),
        }
    }

    let id = renderer
        .upload_result_volume_mc(job)
        .expect("the finished job yields a handle");
    assert!(
        renderer.resident_bytes().plugin_bytes > 0,
        "the volume is in the store once its handle is taken"
    );

    // The result is taken once; a second take has nothing to hand back.
    assert!(matches!(
        renderer.upload_result_volume_mc(job),
        Err(viewport_lib::error::ViewportError::JobResultMissing { .. })
    ));
    renderer.free_mc_volume(id);
}
