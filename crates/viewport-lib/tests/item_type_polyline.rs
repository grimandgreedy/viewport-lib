//! Picking for the polyline item type: GPU pick-id, node / segment / strip
//! sub-objects, CPU proximity picking, rect select, and the pre-uploaded
//! reference form.
//!
//! One file per item type, so a type's coverage travels with it.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::PolylineRefItem;

#[test]
fn gpu_pick_hits_polyline() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

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

    // A thick polyline through the origin. Polylines expand to screen-space
    // ribbons in the render vertex stage, which the pick pipeline reuses, so
    // prepare must run first to build the segment buffer.
    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![2];
    polyline.line_width = 20.0;
    polyline.settings.pick_id = PickId(888);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(888)));
}

// ---------------------------------------------------------------------------
// GPU pick: sub-object identity (G8)
// ---------------------------------------------------------------------------

/// A 64x64 frame with the default orbit camera and no overlays, so world origin
/// projects to the screen centre (32, 32).
#[test]
fn gpu_pick_polyline_resolves_segment() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    // Three-node strip: segment 0 spans x in [-2, -1], segment 1 spans [-1, 2].
    // World origin (screen centre) lies on segment 1. Segment picking reads the
    // per-segment instance_index channel, so no device feature is needed.
    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![3];
    polyline.line_width = 20.0;
    polyline.settings.pick_id = PickId(888);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::SEGMENT,
    );
    let hit = hit.expect("polyline should be hit at the centre");
    assert_eq!(hit.id, 888);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Segment(1)));
}

#[test]
fn gpu_pick_polyline_resolves_strip_without_cpu_cache() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Deliberately leave the CPU pick cache OFF: strip resolution must come from
    // the persistent PolylineGpuData::strip_lengths, not pick_polyline_items.
    let mut frame = sub_object_pick_frame();

    // Two strips. Strip 0 is a single off-centre segment (global segment 0).
    // Strip 1 is three nodes whose middle segment (global segment 2) crosses the
    // world origin, which the camera centre projects onto.
    let mut polyline = PolylineItem::default();
    polyline.positions = vec![
        [-5.0, 3.0, 0.0],
        [-4.0, 3.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ];
    polyline.strip_lengths = vec![2, 3];
    polyline.line_width = 20.0;
    polyline.settings.pick_id = PickId(889);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::STRIP,
        )
        .expect("polyline should be hit at the centre");
    assert_eq!(hit.id, 889);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Strip(1)));
}

#[test]
fn cpu_pick_polyline_resolves_segment() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![3];
    polyline.line_width = 20.0;
    polyline.settings.pick_id = PickId(890);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Cpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::SEGMENT,
        )
        .expect("polyline should be hit at the centre");
    assert_eq!(hit.id, 890);
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Segment(1)));
}

#[test]
fn rect_pick_collects_polyline_segments() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = sub_object_pick_frame();

    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![3];
    polyline.line_width = 10.0;
    polyline.settings.pick_id = PickId(891);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let result = renderer.pick_rect_objects(
        PickBackend::Cpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::SEGMENT,
    );
    assert!(
        result.objects.is_empty(),
        "SEGMENT mask carries no OBJECT bit"
    );
    assert!(
        result
            .elements
            .iter()
            .all(|(id, sub)| *id == 891 && matches!(sub, viewport_lib::SubObjectRef::Segment(_))),
        "every element must be a segment of the polyline"
    );
    assert_eq!(
        result.elements.len(),
        2,
        "both segments are inside the rect"
    );
}

/// A polyline uploaded once and drawn through `PolylineRefItem` picks the same
/// as an inline item, under the reference's own pick id.
#[test]
fn a_reference_item_picks_like_an_inline_one() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![2];
    polyline.line_width = 20.0;
    let source = renderer
        .resources_mut()
        .upload_polyline(&device, &queue, &polyline);

    let mut item = PolylineRefItem::new(source);
    item.settings.pick_id = PickId(892);
    frame.scene.polyline_refs.push(item);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(892));
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

    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![2];
    polyline.line_width = 20.0;
    let source = renderer
        .resources_mut()
        .upload_polyline(&device, &queue, &polyline);

    let mut item = PolylineRefItem::new(source);
    item.settings.pick_id = PickId(893);
    item.settings.hidden = true;
    frame.scene.polyline_refs.push(item);

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

/// A polyline carrying node vectors draws its arrow decoration without
/// disturbing the pick of the line itself.
#[test]
fn a_decorated_polyline_still_picks() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut polyline = PolylineItem::default();
    polyline.positions = vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    polyline.strip_lengths = vec![2];
    polyline.line_width = 20.0;
    polyline.node_vectors = vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
    polyline.vector_scale = 0.5;
    polyline.settings.pick_id = PickId(894);
    frame.scene.polylines.push(polyline);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(32.0, 32.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );
    assert_eq!(hit.map(|h| h.id), Some(894));
}
