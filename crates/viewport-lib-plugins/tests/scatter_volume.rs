//! Picking for the scatter-volume item type: the id pass rasterises the
//! volume's own shape, so a box volume answers through the cube proxy and a
//! sphere volume through the icosphere.
//!
//! One file per item type, so a type's coverage travels with it.

#![cfg(feature = "item-types")]

mod common;
use common::*;
use viewport_lib::{Aabb, SurfaceSubmission};
use viewport_lib_plugins::item_types::scatter_volume::{
    RefractionParams, ScatterVolume, ScatterVolumeItem,
};

/// One test renders at a time. The warm-up test reads the process-wide build
/// log, which the other tests' fresh renderers would write into.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn scatter_pick_frame() -> FrameData {
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
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

#[test]
fn gpu_pick_hits_box_scatter_volume() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = scatter_pick_frame();

    // A box scatter volume centred on the origin. The pick rasterises the actual
    // box (cube proxy) and reads back its id.
    let aabb = Aabb {
        min: glam::Vec3::splat(-0.5),
        max: glam::Vec3::splat(0.5),
    };
    let mut item = ScatterVolumeItem::new(ScatterVolume::box_uniform(aabb, 1.0, [1.0, 1.0, 1.0]));
    item.settings.pick_id = PickId(41);
    *frame.scene.items_mut::<ScatterVolumeItem>() = vec![item];

    // The volume's pick binding is built during prepare, like every other item
    // type that answers the id pass with geometry of its own.
    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(41)));
}

#[test]
fn gpu_pick_hits_sphere_scatter_volume() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = scatter_pick_frame();

    // A sphere scatter volume centred on the origin. The pick rasterises the
    // icosphere proxy and reads back its id at the viewport centre.
    let mut item = ScatterVolumeItem::new(ScatterVolume::sphere_uniform(
        [0.0, 0.0, 0.0],
        0.5,
        1.0,
        [1.0, 1.0, 1.0],
    ));
    item.settings.pick_id = PickId(42);
    *frame.scene.items_mut::<ScatterVolumeItem>() = vec![item];

    // The volume's pick binding is built during prepare, like every other item
    // type that answers the id pass with geometry of its own.
    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(42)));
}

/// Naming the type in a warm-up builds its pipelines, so the first frames that
/// march, resolve, refract, composite and pick a volume compile none of them.
#[test]
fn a_warmed_scatter_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default()
            .with_item_type::<viewport_lib_plugins::item_types::scatter_volume::ScatterVolumePlugin>(),
    );
    renderer.wait_for_pipelines(&device);

    viewport_lib::resources::build_log::enable();
    let _ = viewport_lib::resources::build_log::drain();
    for hdr in [true, false] {
        let mut frame = scatter_pick_frame();
        if !hdr {
            frame.effects.display.mode = viewport_lib::PipelineMode::Direct;
        }
        frame.effects.scatter.temporal = true;
        let aabb = Aabb {
            min: glam::Vec3::splat(-0.5),
            max: glam::Vec3::splat(0.5),
        };
        let mut volume = ScatterVolume::box_uniform(aabb, 1.0, [1.0, 1.0, 1.0]);
        volume.refraction = Some(RefractionParams::default());
        let mut item = ScatterVolumeItem::new(volume);
        item.settings.pick_id = PickId(41);
        *frame.scene.items_mut::<ScatterVolumeItem>() = vec![item];
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
        assert_eq!(hit.map(|h| h.object_id), Some(PickId(41)));
    }
    let builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| l.starts_with("scatter") || l.starts_with("module scatter"))
        .collect();
    assert!(
        builds.is_empty(),
        "the first scatter frames built pipelines after the warm-up: {builds:?}"
    );
}
