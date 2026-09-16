//! Picking for the glyph item type: GPU pick-id, per-instance
//! sub-objects, CPU proximity picking, rect select, and the pre-uploaded
//! reference form.
//!
//! One file per item type, so a type's coverage travels with it.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{GlyphItem, GlyphSetRefItem};

/// Three arrows spread along X, each pointing straight at the camera; the
/// centre one sits at the origin, under the cursor of a `sub_object_pick_frame`
/// click at (32, 32).
///
/// The arrows point at the eye so their midpoints project onto their base
/// positions: the CPU pick tests against midpoints, the GPU pick rasterises the
/// geometry, and only this orientation puts both under the same pixel.
fn three_arrows(frame: &FrameData) -> GlyphItem {
    let eye = glam::Vec3::from(frame.camera.render_camera.eye_position);
    let mut item = GlyphItem::default();
    item.positions = vec![[-3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    item.vectors = item
        .positions
        .iter()
        .map(|p| (eye - glam::Vec3::from(*p)).normalize().to_array())
        .collect();
    item.scale = 1.5;
    item
}

#[test]
fn gpu_pick_glyph_resolves_instance() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut item = three_arrows(&frame);
    item.settings.pick_id = PickId(710);
    frame.scene.glyphs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::INSTANCE,
        )
        .expect("centre arrow should be hit");
    assert_eq!(hit.id, 710);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
}

#[test]
fn cpu_pick_glyph_resolves_instance() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut item = three_arrows(&frame);
    item.settings.pick_id = PickId(711);
    frame.scene.glyphs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Cpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::INSTANCE,
        )
        .expect("centre arrow should be hit");
    assert_eq!(hit.id, 711);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
}

/// An `OBJECT`-only query answers with the set and no instance, on both
/// backends.
#[test]
fn an_object_query_drops_the_instance_sub_object() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut item = three_arrows(&frame);
    item.settings.pick_id = PickId(712);
    frame.scene.glyphs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    for backend in [PickBackend::Gpu, PickBackend::Cpu] {
        let hit = renderer
            .pick_object(
                backend,
                glam::Vec2::new(32.0, 32.0),
                &frame,
                &device,
                &queue,
                PickMask::OBJECT,
            )
            .expect("centre arrow should be hit");
        assert_eq!(hit.id, 712);
        assert_eq!(
            hit.sub_object, None,
            "{backend:?} must not report an instance"
        );
    }
}

#[test]
fn rect_pick_collects_glyph_instances() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut item = three_arrows(&frame);
    item.settings.pick_id = PickId(713);
    frame.scene.glyphs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let result = renderer.pick_rect_objects(
        PickBackend::Cpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::INSTANCE,
    );
    assert!(
        result.objects.is_empty(),
        "INSTANCE mask carries no OBJECT bit"
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect instance sub-objects"
    );
    assert!(
        result
            .elements
            .iter()
            .all(|(id, sub)| *id == 713 && matches!(sub, viewport_lib::SubObjectRef::Instance(_))),
        "every element must be an instance of the set"
    );
}

/// A set uploaded once and drawn through `GlyphSetRefItem` picks the same
/// as an inline item, under the reference's own pick id.
#[test]
fn a_reference_item_picks_like_an_inline_one() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let source = renderer
        .resources_mut()
        .upload_glyph_set(&device, &queue, &three_arrows(&frame));

    let mut item = GlyphSetRefItem::new(source);
    item.settings.pick_id = PickId(714);
    frame.scene.glyph_set_refs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::INSTANCE,
        )
        .expect("centre arrow of the referenced set should be hit");
    assert_eq!(hit.id, 714);
    assert_eq!(
        hit.sub_object,
        Some(viewport_lib::SubObjectRef::Instance(1))
    );
}

/// A hidden reference item draws nothing and picks nothing.
#[test]
fn a_hidden_reference_item_is_skipped() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let source = renderer
        .resources_mut()
        .upload_glyph_set(&device, &queue, &three_arrows(&frame));

    let mut item = GlyphSetRefItem::new(source);
    item.settings.pick_id = PickId(715);
    item.settings.hidden = true;
    frame.scene.glyph_set_refs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), None);
}
