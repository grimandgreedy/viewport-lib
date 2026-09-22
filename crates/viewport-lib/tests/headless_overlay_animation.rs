//! Overlay animation tracks: what they drive, and that a retained group can
//! carry them.
//!
//! The channel list is exactly the state that rides the per-draw instance, so
//! an animated group re-draws from its compiled buffers. The only thing that
//! re-emits a compiled group is a glyph-atlas grow or a `pixels_per_point`
//! change (`reemit_overlay_geometry_if_stale`), neither of which a track
//! touches; what these tests pin is the visible half, that the same compiled
//! handle draws differently at different times.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    AnimTrack, Colour, OverlayAnimations, OverlayEasing, OverlayFill, OverlayPolylineItem,
    RepeatMode, RetainedOverlay,
};

const SIZE: u32 = 64;

fn frame_at(time: f64) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.0, 0.0, 0.0, 1.0].into());
    frame.overlays.time = time;
    frame
}

fn red_square() -> OverlayPolylineItem {
    OverlayPolylineItem::new(vec![[16.0, 16.0], [32.0, 16.0], [32.0, 32.0], [16.0, 32.0]])
        .with_closed(true)
        .with_thickness(0.0)
        .with_fill(OverlayFill::Solid(Colour::linear(1.0, 0.0, 0.0, 1.0)))
}

fn rgb_at(px: &[u8], size: u32, x: u32, y: u32) -> (u8, u8, u8) {
    let i = ((y * size + x) * 4) as usize;
    (px[i], px[i + 1], px[i + 2])
}

/// One compiled group, four frames, four different pictures: the translate,
/// scale, opacity, and tint tracks all resolve onto the instance.
#[test]
fn a_retained_group_animates_from_its_compiled_buffers() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let id =
        renderer.compile_overlay_geometry(&device, &queue, &[red_square()], &[], &[], &[], 1.0);

    let group = RetainedOverlay::new(id).with_animations(
        OverlayAnimations::default()
            .with_translate(AnimTrack::new(0.0, 1.0, [0.0, 0.0], [16.0, 0.0]))
            .with_tint(AnimTrack::new(
                0.0,
                1.0,
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            )),
    );

    let mut at_start = frame_at(0.0);
    at_start.overlays.retained = vec![group.clone()];
    let start = renderer.render_offscreen(&device, &queue, &at_start, SIZE, SIZE);

    let mut at_end = frame_at(1.0);
    at_end.overlays.retained = vec![group];
    let end = renderer.render_offscreen(&device, &queue, &at_end, SIZE, SIZE);

    // At t = 0 the square sits over 16..32 and is red.
    let (r, g, _) = rgb_at(&start, SIZE, 24, 24);
    assert!(r > 200 && g < 60, "expected red at t=0, got ({r}, {g})");

    // At t = 1 it has moved 16 px right and the tint has taken the red out, so
    // the old centre is background and the new one is dark.
    let (r2, g2, b2) = rgb_at(&end, SIZE, 24, 24);
    assert!(
        r2 < 30 && g2 < 30 && b2 < 30,
        "expected background at the old centre, got ({r2}, {g2}, {b2})"
    );
    let (r3, g3, _) = rgb_at(&end, SIZE, 40, 24);
    assert!(
        r3 < 60 && g3 < 60,
        "the tint should have removed the red channel, got ({r3}, {g3})"
    );
}

/// The same tracks on an immediate item, so the two paths agree. This is the
/// property the shared vocabulary exists for: moving content between the
/// immediate and retained paths must not change how it animates.
#[test]
fn an_immediate_item_animates_the_same_way() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let anims = OverlayAnimations::default().with_translate(AnimTrack::new(
        0.0,
        1.0,
        [0.0, 0.0],
        [16.0, 0.0],
    ));
    let item = red_square().with_animations(anims);

    let mut at_start = frame_at(0.0);
    at_start.overlays.polylines = vec![item.clone()];
    let start = renderer.render_offscreen(&device, &queue, &at_start, SIZE, SIZE);

    let mut at_end = frame_at(1.0);
    at_end.overlays.polylines = vec![item];
    let end = renderer.render_offscreen(&device, &queue, &at_end, SIZE, SIZE);

    assert!(rgb_at(&start, SIZE, 24, 24).0 > 200);
    assert!(rgb_at(&end, SIZE, 24, 24).0 < 30);
    assert!(rgb_at(&end, SIZE, 40, 24).0 > 200);
}

/// `epoch` makes a track's `start_time` a delay, which is what lets authored
/// content carry animation at all: a screen is authored long before the
/// process that plays it starts.
#[test]
fn the_epoch_shifts_a_whole_track() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Authored: "start a second in, run for a second".
    let authored = OverlayAnimations::default().with_translate(
        AnimTrack::new(1.0, 1.0, [0.0, 0.0], [16.0, 0.0])
            .with_easing(OverlayEasing::Linear)
            .with_repeat(RepeatMode::Once),
    );

    // Played from a clock that happens to be at 1000.
    let item = red_square().with_animations(authored.with_epoch(1000.0));
    let mut early = frame_at(1000.5);
    early.overlays.polylines = vec![item.clone()];
    let px = renderer.render_offscreen(&device, &queue, &early, SIZE, SIZE);
    assert!(
        rgb_at(&px, SIZE, 24, 24).0 > 200,
        "before the delay elapses the item should not have moved"
    );

    let mut late = frame_at(1002.0);
    late.overlays.polylines = vec![item];
    let px = renderer.render_offscreen(&device, &queue, &late, SIZE, SIZE);
    assert!(
        rgb_at(&px, SIZE, 40, 24).0 > 200,
        "after the delay plus the duration the item should be at its end value"
    );
}

/// Labels and glyph runs animate too. Wiring each family separately is exactly
/// how the paths drifted apart before, so each one gets a check rather than an
/// assumption that it was hooked up the same way.
#[test]
fn labels_and_glyph_runs_animate() {
    use viewport_lib::{GlyphRunItem, LabelItem, PositionedGlyph};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let slide = || {
        OverlayAnimations::default().with_translate(AnimTrack::new(
            0.0,
            1.0,
            [0.0, 0.0],
            [24.0, 0.0],
        ))
    };

    let mut at_start = frame_at(0.0);
    let mut at_end = frame_at(1.0);
    let label = LabelItem::new("Mg")
        .with_position([4.0, 20.0])
        .with_font_size(28.0)
        .with_colour(Colour::linear(1.0, 1.0, 1.0, 1.0))
        .with_animations(slide());
    at_start.overlays.labels = vec![label.clone()];
    at_end.overlays.labels = vec![label];
    let a = renderer.render_offscreen(&device, &queue, &at_start, SIZE, SIZE);
    let b = renderer.render_offscreen(&device, &queue, &at_end, SIZE, SIZE);
    assert!(
        a != b,
        "a label with a translate track should draw differently at t=0 and t=1"
    );

    let mut at_start = frame_at(0.0);
    let mut at_end = frame_at(1.0);
    let mut run = GlyphRunItem::new(vec![PositionedGlyph::new(55, 0.0, 0.0)]);
    run.text_style.size = 28.0;
    run.transform.translate = [4.0, 40.0];
    run.style.fill = OverlayFill::Solid(Colour::linear(1.0, 1.0, 1.0, 1.0));
    run.animations = Some(Box::new(slide()));
    at_start.overlays.glyph_runs = vec![run.clone()];
    at_end.overlays.glyph_runs = vec![run];
    let a = renderer.render_offscreen(&device, &queue, &at_start, SIZE, SIZE);
    let b = renderer.render_offscreen(&device, &queue, &at_end, SIZE, SIZE);
    assert!(
        a != b,
        "a glyph run with a translate track should draw differently at t=0 and t=1"
    );
}
