//! Picking for the GPU implicit surface item type: GPU pick-id, CPU ray-march,
//! and rect select.
//!
//! One file per item type, so a type's coverage travels with it.

use bytemuck::Zeroable as _;
use viewport_lib::gpu;
use viewport_lib_item_types::*;

mod common;
use common::*;

/// A renderer with this crate's item types registered, which is what a
/// consumer of the crate builds.
fn renderer_with_item_types(device: &gpu::Device) -> ViewportRenderer {
    let mut renderer = ViewportRenderer::new(device, gpu::TextureFormat::Rgba8UnormSrgb);
    install(&mut renderer, device);
    renderer
}

#[test]
fn gpu_pick_hits_implicit_surface() {
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
