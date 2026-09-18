//! Picking and draw state for the three curve mesh item types: streamtube,
//! tube and ribbon. Covers GPU pick-id, the segment / strip / node
//! sub-objects, CPU proximity picking, rect select, and the pre-uploaded
//! reference form.
//!
//! One file per item type family, so a type's coverage travels with it.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    RibbonRefItem, StreamtubeItem, StreamtubeRefItem, SubObjectRef, TubeItem, TubeRefItem,
};

/// A 64x64 frame with the default orbit camera and no overlays, so the world
/// origin projects to the screen centre (32, 32).
fn curve_frame() -> FrameData {
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

/// Three control points along X with the middle one at the origin, the shape
/// every test in this file sweeps.
fn spine() -> (Vec<[f32; 3]>, Vec<u32>) {
    (
        vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        vec![3],
    )
}

#[test]
fn gpu_pick_hits_ribbon() {
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

    // A wide ribbon centred on the origin. Ribbons build an owned connected mesh
    // into ribbon_gpu_data during prepare(), so the pick pass only sees it after
    // the prepare path has run.
    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(4242)));
}

#[test]
fn gpu_pick_ribbon_resolves_segment_and_node() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Segment and node resolve from the pick shader variants; no CPU pick cache.
    let mut frame = sub_object_pick_frame();

    // A wide ribbon centred on the origin: 3 control points along X, so two
    // segments with the middle node under the centre pixel.
    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    // SEGMENT: the hit resolves to one of the two segments.
    let seg = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::SEGMENT,
        )
        .expect("ribbon should be hit");
    assert_eq!(seg.id, 4242);
    match seg.sub_object {
        Some(viewport_lib::SubObjectRef::Segment(s)) => assert!(s < 2, "segment {s} out of range"),
        other => panic!("expected a Segment sub-object, got {other:?}"),
    }

    // POLY_NODE: a centre click resolves to the middle control point (index 1),
    // the nearer endpoint of whichever segment the ray landed on.
    let node = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::POLY_NODE,
        )
        .expect("ribbon should be hit");
    assert_eq!(node.id, 4242);
    assert_eq!(node.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
}

#[test]
fn gpu_pick_curve_node_fills_snap_world_pos() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::POLY_NODE,
        )
        .expect("ribbon should be hit");
    assert_eq!(hit.sub_object, Some(viewport_lib::SubObjectRef::Point(1)));
    // Node 1 sits at the world origin; the snap position must be that node.
    let snap = hit
        .sub_object_world_pos
        .expect("node pick should fill the snap position");
    assert!(
        snap.length() < 1e-4,
        "node 1 should be at the origin, got {snap:?}"
    );
}

#[test]
fn gpu_pick_rect_resolves_curve_node() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    // No CPU pick cache: the POLY_NODE variant writes the nearest node index per
    // pixel, so a rect reads it straight from the primitive channel.
    let mut frame = sub_object_pick_frame();

    let mut ribbon = RibbonItem::default();
    ribbon.positions = vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    ribbon.strip_lengths = vec![3];
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

    let _ = renderer.pass().prepare(&device, &queue, &frame);

    let result = renderer.pick_rect_objects(
        PickBackend::Gpu,
        glam::Vec2::new(0.0, 0.0),
        glam::Vec2::new(64.0, 64.0),
        &frame,
        &device,
        &queue,
        PickMask::POLY_NODE,
    );
    assert!(
        result.objects.is_empty(),
        "POLY_NODE mask carries no OBJECT bit"
    );
    assert!(
        !result.elements.is_empty(),
        "rect should collect node sub-objects"
    );
    assert!(
        result.elements.iter().all(|(id, sub)| *id == 4242
            && matches!(sub, viewport_lib::SubObjectRef::Point(n) if *n < 3)),
        "every element must be a ribbon node, got {:?}",
        result.elements
    );
    // The middle node (index 1) sits under the centre of the rect, so it must be
    // among the collected nodes.
    assert!(
        result
            .elements
            .iter()
            .any(|(_, sub)| *sub == viewport_lib::SubObjectRef::Point(1)),
        "the middle node should be collected, got {:?}",
        result.elements
    );
}

// ---------------------------------------------------------------------------
// GPU pick: the other two curve types
// ---------------------------------------------------------------------------

#[test]
fn gpu_pick_hits_streamtube() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = StreamtubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(5151);
    frame.scene.streamtube_items.push(tube);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(5151)));
}

#[test]
fn gpu_pick_hits_tube() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = TubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(5252);
    frame.scene.tube_items.push(tube);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(5252)));
}

/// STRIP folds a hit on any segment back to the strip that owns it, taking
/// priority over SEGMENT when both are asked for.
#[test]
fn gpu_pick_streamtube_resolves_strip_without_cpu_cache() {
    let Some((device, queue)) = headless_device_with_primitive_index() else {
        eprintln!("skipping: no adapter with SHADER_PRIMITIVE_INDEX");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = sub_object_pick_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = StreamtubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(5151);
    frame.scene.streamtube_items.push(tube);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(32.0, 32.0),
            &frame,
            &device,
            &queue,
            PickMask::STRIP | PickMask::SEGMENT,
        )
        .expect("streamtube should be hit");
    assert_eq!(hit.id, 5151);
    assert_eq!(hit.sub_object, Some(SubObjectRef::Strip(0)));
}

// ---------------------------------------------------------------------------
// CPU pick
// ---------------------------------------------------------------------------

/// The CPU path answers the same levels as the GPU one, from the item geometry
/// the plugin retained at prepare.
#[test]
fn cpu_pick_tube_resolves_segment() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = TubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(5252);
    frame.scene.tube_items.push(tube);

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
        .expect("tube should be hit");
    assert_eq!(hit.id, 5252);
    match hit.sub_object {
        Some(SubObjectRef::Segment(s)) => assert!(s < 2, "segment {s} out of range"),
        other => panic!("expected a Segment sub-object, got {other:?}"),
    }
}

#[test]
fn rect_pick_collects_ribbon_segments() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_cpu_pick_cache(true);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut ribbon = RibbonItem::default();
    ribbon.positions = positions;
    ribbon.strip_lengths = strip_lengths;
    ribbon.width = 2.0;
    ribbon.settings.pick_id = PickId(4242);
    frame.scene.ribbon_items.push(ribbon);

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
        !result.elements.is_empty(),
        "rect should collect ribbon segments"
    );
    assert!(
        result
            .elements
            .iter()
            .all(|(id, sub)| *id == 4242 && matches!(sub, SubObjectRef::Segment(s) if *s < 2)),
        "every element must be a ribbon segment, got {:?}",
        result.elements
    );
}

// ---------------------------------------------------------------------------
// The pre-uploaded reference form
// ---------------------------------------------------------------------------

/// A reference item picks under the id on the reference, not the one the
/// upload was made with: the same stored geometry can be drawn twice under two
/// ids.
#[test]
fn a_reference_streamtube_picks_under_its_own_id() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = StreamtubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    tube.settings.pick_id = PickId(1);
    let source = renderer.upload_streamtube(&device, &queue, &tube);

    let mut reference = StreamtubeRefItem::new(source);
    reference.settings.pick_id = PickId(7171);
    frame.scene.streamtube_refs.push(reference);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(7171)));
}

#[test]
fn a_hidden_reference_tube_is_skipped() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut tube = TubeItem::default();
    tube.positions = positions;
    tube.strip_lengths = strip_lengths;
    tube.radius = 0.5;
    let source = renderer.upload_tube(&device, &queue, &tube);

    let mut reference = TubeRefItem::new(source);
    reference.settings.pick_id = PickId(7272);
    reference.settings.hidden = true;
    frame.scene.tube_refs.push(reference);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), None);
}

#[test]
fn a_reference_ribbon_draws_and_picks() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut frame = curve_frame();

    let (positions, strip_lengths) = spine();
    let mut ribbon = RibbonItem::default();
    ribbon.positions = positions;
    ribbon.strip_lengths = strip_lengths;
    ribbon.width = 2.0;
    let source = renderer.upload_ribbon(&device, &queue, &ribbon);

    let mut reference = RibbonRefItem::new(source);
    reference.settings.pick_id = PickId(7373);
    frame.scene.ribbon_refs.push(reference);

    let _ = renderer.pass().prepare(&device, &queue, &frame);
    let hit = renderer.pick_scene_gpu(&device, &queue, glam::Vec2::new(32.0, 32.0), &frame);
    assert_eq!(hit.map(|h| h.object_id), Some(PickId(7373)));
}
