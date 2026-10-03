//! Picking for the volume surface slice item type: GPU pick-id, CPU ray/mesh
//! intersection, and rect select.
//!
//! One file per item type, so a type's coverage travels with it.

use viewport_lib_item_types::*;

mod common;
use common::*;

/// One test renders at a time. The warm-up test reads the process-wide build
/// log, which the other tests' fresh renderers would write into.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn gpu_pick_hits_volume_surface_slice() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &[0.5; 8], [2, 2, 2]);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");

    let mut slice = VolumeSurfaceSliceItem::default();
    slice.volume_id = volume_id;
    slice.mesh_id = mesh_id;
    slice.bbox_min = [-1.0, -1.0, -1.0];
    slice.bbox_max = [1.0, 1.0, 1.0];
    slice.settings.pick_id = PickId(333);
    frame
        .scene
        .items_mut::<VolumeSurfaceSliceItem>()
        .push(slice);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(333));
}

#[test]
fn cpu_pick_hits_volume_surface_slice() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &[0.5; 8], [2, 2, 2]);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");

    let mut slice = VolumeSurfaceSliceItem::default();
    slice.volume_id = volume_id;
    slice.mesh_id = mesh_id;
    slice.bbox_min = [-1.0, -1.0, -1.0];
    slice.bbox_max = [1.0, 1.0, 1.0];
    slice.settings.pick_id = PickId(334);
    frame
        .scene
        .items_mut::<VolumeSurfaceSliceItem>()
        .push(slice);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let vp = glam::Vec2::new(64.0, 64.0);
    let view_proj = frame.camera.render_camera.view_proj();

    // The CPU path ray-casts the mesh's retained triangles.
    let hit = renderer.pick(glam::Vec2::new(32.0, 32.0), vp, view_proj, PickMask::OBJECT);
    assert_eq!(hit.map(|h| h.id), Some(334));

    let miss = renderer.pick(glam::Vec2::new(1.0, 1.0), vp, view_proj, PickMask::OBJECT);
    assert!(miss.is_none(), "corner ray should miss the slice mesh");
}

#[test]
fn rect_pick_hits_volume_surface_slice() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &[0.5; 8], [2, 2, 2]);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");

    let mut slice = VolumeSurfaceSliceItem::default();
    slice.volume_id = volume_id;
    slice.mesh_id = mesh_id;
    slice.bbox_min = [-1.0, -1.0, -1.0];
    slice.bbox_max = [1.0, 1.0, 1.0];
    slice.settings.pick_id = PickId(335);
    frame
        .scene
        .items_mut::<VolumeSurfaceSliceItem>()
        .push(slice);

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
        result.objects.contains(&335),
        "full-viewport rect should select the slice, got {:?}",
        result.objects
    );

    let away = renderer.pick_rect(
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(2.0, 2.0),
        vp,
        view_proj,
        PickMask::OBJECT,
    );
    assert!(!away.objects.contains(&335));
}

/// Naming the type in a warm-up builds its pipelines, so the first frames that
/// draw, outline and pick a slice, in either format, compile none of them.
#[test]
fn a_warmed_slice_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let volume_id = renderer
        .resources_mut()
        .upload_volume(&device, &queue, &[0.5; 8], [2, 2, 2]);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &box_mesh())
        .expect("upload box mesh");
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default().with_item_type::<VolumeSurfaceSlicePlugin>(),
    );
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for hdr in [true, false] {
        let mut frame = sub_object_pick_frame();
        if !hdr {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        frame.interaction.outline_selected = true;
        let mut slice = VolumeSurfaceSliceItem::default();
        slice.volume_id = volume_id;
        slice.mesh_id = mesh_id;
        slice.bbox_min = [-1.0, -1.0, -1.0];
        slice.bbox_max = [1.0, 1.0, 1.0];
        slice.settings.pick_id = PickId(333);
        slice.settings.selected = true;
        frame
            .scene
            .items_mut::<VolumeSurfaceSliceItem>()
            .push(slice);
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        let hit = renderer.pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::OBJECT,
        );
        assert_eq!(hit.map(|h| h.id), Some(333));
    }
    let builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| {
            l.starts_with("volume_surface_slice") || l.starts_with("module volume_surface_slice")
        })
        .collect();
    assert!(
        builds.is_empty(),
        "the first slice frames built pipelines after the warm-up: {builds:?}"
    );
}
