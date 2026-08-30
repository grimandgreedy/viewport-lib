//! The BVH-accelerated CPU point pick must keep agreeing with the GPU pick across a
//! scene of many overlapping objects: the broad-phase BVH has to select the same
//! nearest object the rasterizer does.
//!
//! CPU and GPU differ by sub-pixel quantization at object silhouettes (the CPU casts
//! a continuous ray; the GPU reads back a rasterized pixel), so this compares over a
//! coarse cursor grid and requires high agreement at the unambiguous interior pixels
//! rather than an exact match at every cursor.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

fn pick_frame() -> FrameData {
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
}

#[test]
fn bvh_cpu_pick_matches_gpu_pick_many_objects() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = pick_frame();

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");

    // A handful of large boxes spread across the view with some depth overlap: big
    // enough that each covers many pixels (so interior pixels exist) and arranged so
    // the broad phase must pick the nearest where they stack.
    let placements: [(glam::Vec3, f32, u64); 6] = [
        (glam::Vec3::new(0.0, 0.0, 0.0), 1.0, 1),
        (glam::Vec3::new(1.7, 0.0, 0.3), 0.8, 2),
        (glam::Vec3::new(-1.7, 0.0, -0.3), 0.8, 3),
        (glam::Vec3::new(0.0, 1.7, 0.0), 0.8, 4),
        (glam::Vec3::new(0.0, -1.7, 0.0), 0.8, 5),
        (glam::Vec3::new(0.7, 0.7, 1.2), 0.5, 6),
    ];
    let mut items = Vec::new();
    for (t, scale, id) in placements {
        let mut item = SceneRenderItem::default();
        item.mesh_id = mesh_id;
        item.model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(scale),
            glam::Quat::IDENTITY,
            t,
        )
        .to_cols_array_2d();
        item.settings.pick_id = PickId(id);
        items.push(item);
    }
    frame.scene.surfaces = SurfaceSubmission::Flat(items.into());

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // Sample GPU and CPU object ids over a coarse grid.
    const N: usize = 16;
    const STEP: usize = 4;
    let mut gpu = vec![vec![None; N]; N];
    let mut cpu = vec![vec![None; N]; N];
    for iy in 0..N {
        for ix in 0..N {
            let cursor =
                glam::Vec2::new((2 + ix * STEP) as f32 + 0.5, (2 + iy * STEP) as f32 + 0.5);
            gpu[iy][ix] = renderer
                .pick_object(
                    PickBackend::Gpu,
                    cursor,
                    &frame,
                    &device,
                    &queue,
                    PickMask::OBJECT,
                )
                .map(|h| h.id);
            cpu[iy][ix] = renderer
                .pick_object(
                    PickBackend::Cpu,
                    cursor,
                    &frame,
                    &device,
                    &queue,
                    PickMask::OBJECT,
                )
                .map(|h| h.id);
        }
    }

    // At interior cells (the GPU id is the same across the 4-neighbourhood, so the
    // cursor is well inside one object, not on a silhouette) the CPU must resolve the
    // same object exactly.
    let mut interior = 0usize;
    let mut disagree = 0usize;
    for iy in 1..N - 1 {
        for ix in 1..N - 1 {
            let g = gpu[iy][ix];
            let interior_cell = g.is_some()
                && gpu[iy - 1][ix] == g
                && gpu[iy + 1][ix] == g
                && gpu[iy][ix - 1] == g
                && gpu[iy][ix + 1] == g;
            if interior_cell {
                interior += 1;
                if cpu[iy][ix] != g {
                    disagree += 1;
                }
            }
        }
    }

    eprintln!("cpu-vs-gpu interior pixels: {interior} tested, {disagree} disagreed");
    assert!(
        interior > 10,
        "too few interior pixels ({interior}) - scene not exercised"
    );
    assert_eq!(
        disagree, 0,
        "CPU pick disagreed with GPU at {disagree} interior pixels"
    );
}
