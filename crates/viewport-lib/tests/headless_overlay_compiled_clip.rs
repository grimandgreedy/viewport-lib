//! An item's own clip on the retained path.
//!
//! `OverlayClip` is one field on every overlay item, and both halves of it have
//! to behave the same whether the item is submitted each frame or compiled into
//! a retained group. `clip.rect` is resolved at compile and clips the same
//! pixels either way. `clip.mask` cannot be: masks are registered per frame in
//! screen space and a compiled group has no frame to resolve one against, so an
//! item naming one is skipped rather than quietly drawn unclipped.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    OverlayFill, OverlayPolylineItem, OverlayShape, OverlayShapeItem, RetainedOverlay,
};

const SIZE: u32 = 64;

/// A 64x64 frame looking at nothing, flat grey background, chrome off.
fn overlay_frame() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.3, 0.3, 0.3, 1.0].into());
    frame
}

/// A closed, red-filled square over screen pixels 16..48, with no outline.
fn red_square() -> OverlayPolylineItem {
    let mut p =
        OverlayPolylineItem::new(vec![[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]]);
    p.closed = true;
    p.stroke = None;
    p.style.fill = OverlayFill::Solid([1.0, 0.0, 0.0, 1.0].into());
    p
}

/// Count the pixels a red fill reached.
fn red_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[0] > 150 && p[1] < 100 && p[2] < 100)
        .count()
}

/// A compiled item's `clip.rect` clips the same pixels as the same item drawn
/// immediately. Before this was baked at compile, the rect was silently dropped
/// and the compiled copy drew the whole square.
#[test]
fn a_compiled_clip_rect_clips_what_the_immediate_one_does() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Keep the left half of the square.
    let mut clipped = red_square();
    clipped.clip.rect = Some([0.0, 0.0, 32.0, 64.0]);

    let mut immediate = overlay_frame();
    immediate.overlays.polylines = vec![clipped.clone()];
    let immediate_px = renderer.render_offscreen(&device, &queue, &immediate, SIZE, SIZE);

    let id = renderer.compile_overlay_geometry(&device, &queue, &[clipped], &[], &[], &[], 1.0);
    let mut retained = overlay_frame();
    retained.overlays.retained = vec![RetainedOverlay::new(id)];
    let retained_px = renderer.render_offscreen(&device, &queue, &retained, SIZE, SIZE);

    let (immediate_red, retained_red) = (red_pixels(&immediate_px), red_pixels(&retained_px));
    assert!(
        immediate_red > 0,
        "the clipped square should still draw its left half"
    );
    assert!(
        (immediate_red as i64 - retained_red as i64).abs() <= 2,
        "compiled clip covered {retained_red} px, immediate {immediate_red}"
    );

    // And the clip is doing work: unclipped is about twice the area.
    let unclipped_id =
        renderer.compile_overlay_geometry(&device, &queue, &[red_square()], &[], &[], &[], 1.0);
    let mut unclipped = overlay_frame();
    unclipped.overlays.retained = vec![RetainedOverlay::new(unclipped_id)];
    let unclipped_px = renderer.render_offscreen(&device, &queue, &unclipped, SIZE, SIZE);
    assert!(
        red_pixels(&unclipped_px) > retained_red + 100,
        "an unclipped compiled square should cover much more than a clipped one"
    );
}

/// A compiled item naming a clip mask is skipped. Drawing it unclipped would be
/// the worse failure: content escaping the region it was meant to stay inside.
#[test]
fn a_compiled_clip_mask_is_rejected_rather_than_ignored() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let mut masked = red_square();
    masked.clip.mask = Some(7);
    let id = renderer.compile_overlay_geometry(&device, &queue, &[masked], &[], &[], &[], 1.0);

    let mut frame = overlay_frame();
    // The mask is registered as an immediate shape, the way the immediate path
    // wants it: it still cannot reach the compiled group.
    frame.overlays.shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [0.0, 0.0],
            [32.0, 64.0],
        )
        .with_clip_mask(7),
    ];
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, SIZE, SIZE);

    assert_eq!(
        red_pixels(&px),
        0,
        "an item whose clip mask cannot be resolved must be skipped, not drawn unclipped"
    );
}
