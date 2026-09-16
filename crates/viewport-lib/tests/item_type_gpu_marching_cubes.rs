//! Picking for the GPU marching cubes item type: GPU pick-id against the
//! compute-generated isosurface, plus the CPU ray-march and rect select.
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
        .resources_mut()
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
        .resources_mut()
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
        .resources_mut()
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
