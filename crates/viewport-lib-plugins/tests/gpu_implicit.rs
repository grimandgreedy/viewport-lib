//! Picking for the GPU implicit surface item type: GPU pick-id, CPU ray-march,
//! and rect select.
//!
//! One file per item type, so a type's coverage travels with it.

#![cfg(feature = "item-types")]

use viewport_lib_plugins::item_types::gpu_implicit::{
    GpuImplicitItem, GpuImplicitPlugin, ImplicitPrimitive, TYPE_NAME as GPU_IMPLICIT_TYPE_NAME,
};

mod common;
use common::*;

/// One test renders at a time. The warm-up test reads the process-wide build
/// log, which the other tests' fresh renderers would write into.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn gpu_pick_hits_implicit_surface() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    // One SDF sphere of radius 1.5 at the origin. The pick pass raymarches the
    // isosurface on a full-screen quad and writes the item's pick id at the hit.
    let prim = ImplicitPrimitive {
        kind: 1, // sphere
        blend: 0.0,
        _pad: [0.0; 2],
        params: [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        colour: [1.0, 1.0, 1.0, 1.0].into(),
    };
    let mut item = GpuImplicitItem::default();
    item.primitives.push(prim);
    item.settings.pick_id = PickId(909);
    frame.scene.items_mut::<GpuImplicitItem>().push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(909)));
}

#[test]
fn cpu_pick_hits_implicit_surface() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    // Same sphere as the GPU case: the CPU path marches the SDF along the
    // cursor ray, so a click at the viewport centre lands on it and a click in
    // the corner misses.
    let prim = ImplicitPrimitive {
        kind: 1, // sphere
        blend: 0.0,
        _pad: [0.0; 2],
        params: [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        colour: [1.0, 1.0, 1.0, 1.0].into(),
    };
    let mut item = GpuImplicitItem::default();
    item.primitives.push(prim);
    item.settings.pick_id = PickId(910);
    frame.scene.items_mut::<GpuImplicitItem>().push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let vp = glam::Vec2::new(64.0, 64.0);
    let view_proj = frame.camera.render_camera.view_proj();

    let hit = renderer.pick(glam::Vec2::new(32.0, 32.0), vp, view_proj, PickMask::OBJECT);
    assert_eq!(hit.map(|h| h.id), Some(910));

    let miss = renderer.pick(glam::Vec2::new(1.0, 1.0), vp, view_proj, PickMask::OBJECT);
    assert!(miss.is_none(), "corner ray should miss the sphere");
}

#[test]
fn rect_pick_hits_implicit_surface() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let prim = ImplicitPrimitive {
        kind: 1, // sphere
        blend: 0.0,
        _pad: [0.0; 2],
        params: [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
        colour: [1.0, 1.0, 1.0, 1.0].into(),
    };
    let mut item = GpuImplicitItem::default();
    item.primitives.push(prim);
    item.settings.pick_id = PickId(911);
    frame.scene.items_mut::<GpuImplicitItem>().push(item);

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
        result.objects.contains(&911),
        "full-viewport rect should select the implicit item, got {:?}",
        result.objects
    );

    // A rect in the far corner covers none of the primitive's projected bound.
    let away = renderer.pick_rect(
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(2.0, 2.0),
        vp,
        view_proj,
        PickMask::OBJECT,
    );
    assert!(
        !away.objects.contains(&911),
        "corner rect should not select the implicit item"
    );
}

/// A hidden item produces no draw data, so hiding one costs nothing beyond the
/// submission itself.
#[test]
fn hidden_items_produce_no_draw_data() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    let mut frame = sub_object_pick_frame();

    let mut prim = ImplicitPrimitive::zeroed();
    prim.kind = 1; // sphere
    prim.params[3] = 1.0;

    let mut visible = GpuImplicitItem::default();
    visible.primitives.push(prim);
    let mut hidden = GpuImplicitItem::default();
    hidden.primitives.push(prim);
    hidden.settings.hidden = true;
    frame
        .scene
        .items_mut::<GpuImplicitItem>()
        .extend([visible, hidden]);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let plugin = renderer
        .item_type_plugin::<GpuImplicitPlugin>(GPU_IMPLICIT_TYPE_NAME)
        .expect("registered by install()");
    assert_eq!(
        plugin.drawn_count(),
        1,
        "hidden item must not produce gpu data"
    );
}

/// Warming the type builds every pipeline its draws, outline and pick use, so
/// the first frames in either format build nothing.
#[test]
fn a_warmed_implicit_type_builds_nothing_on_its_first_frame() {
    let _serial = serial();
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = renderer_with_item_types(&device);
    renderer.warm_pipelines(
        &device,
        &queue,
        &viewport_lib::PipelineSet::default().with_item_type::<GpuImplicitPlugin>(),
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
        let mut prim = ImplicitPrimitive::zeroed();
        prim.kind = 1; // sphere
        prim.params[3] = 1.5;
        let mut item = GpuImplicitItem::default();
        item.primitives.push(prim);
        item.settings.pick_id = PickId(912);
        item.settings.selected = true;
        frame.scene.items_mut::<GpuImplicitItem>().push(item);
        let _ = renderer.render_offscreen(&device, &queue, &frame, 64, 64);
        let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
        assert_eq!(hit.map(|h| h.object_id), Some(PickId(912)));
    }
    let builds: Vec<String> = viewport_lib::resources::build_log::drain()
        .into_iter()
        .map(|(label, _)| label)
        .filter(|l| l.starts_with("implicit") || l.starts_with("module implicit"))
        .collect();
    assert!(
        builds.is_empty(),
        "the first implicit frames built pipelines after the warm-up: {builds:?}"
    );
}
