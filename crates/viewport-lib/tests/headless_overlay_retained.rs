//! Retained overlay geometry (`OverlayGeometryId` / `RetainedOverlay`), end to end.
//!
//! Compiles a filled polyline once with `compile_overlay_geometry`, then submits
//! it through `OverlayFrame::retained` and reads back pixels to confirm: it draws
//! from the cached buffer, the per-frame `translate` moves it without
//! re-compiling, `opacity` fades it, and a freed handle draws nothing.
//!
//! Part of the headless integration suite; shared device helpers live in
//! tests/common/mod.rs.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    AnchorX, AnchorY, GlyphRunItem, LabelItem, OverlayAnchor, OverlayFill, OverlayPolylineItem,
    OverlayShape, OverlayShapeItem, PositionedGlyph, RetainedOverlay,
};

/// A 64x64 frame looking at nothing, flat grey background, chrome off.
fn overlay_frame(size: u32) -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = {
        let mut rc = RenderCamera::from_camera(&cam);
        rc.aspect = 1.0;
        rc
    };
    frame.camera.viewport_size = [size as f32, size as f32];
    frame.camera.pixels_per_point = 1.0;
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.viewport.background_colour = Some([0.3, 0.3, 0.3, 1.0]);
    frame
}

/// A closed, red-filled square polyline over screen pixels 16..48.
fn red_square() -> OverlayPolylineItem {
    let mut p = OverlayPolylineItem::default();
    p.points = vec![[16.0, 16.0], [48.0, 16.0], [48.0, 48.0], [16.0, 48.0]];
    p.closed = true;
    p.thickness = 0.0;
    p.fill = Some(OverlayFill::Solid([1.0, 0.0, 0.0, 1.0]));
    p.opacity = 1.0;
    p
}

fn rgb_at(px: &[u8], size: u32, x: u32, y: u32) -> (u8, u8, u8) {
    let i = ((y * size + x) * 4) as usize;
    (px[i], px[i + 1], px[i + 2])
}

fn is_red(c: (u8, u8, u8)) -> bool {
    c.0 > 200 && c.1 < 100 && c.2 < 100
}

fn is_background(c: (u8, u8, u8)) -> bool {
    // 0.3 linear grey encodes to ~149 in sRGB; the key point is it is not red.
    c.0 < 200 && (c.0 as i32 - c.1 as i32).abs() < 30 && (c.1 as i32 - c.2 as i32).abs() < 30
}

/// A red SDF rect over pixels 16..48 (analytic shape pipeline).
fn red_sdf_rect() -> OverlayShapeItem {
    OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 0.0 },
        [16.0, 16.0],
        [32.0, 32.0],
    )
    .with_fill(OverlayFill::Solid([1.0, 0.0, 0.0, 1.0]))
}

/// A retained group carrying an analytic SDF shape draws through the shape
/// pipeline from its cached buffer and moves with the per-frame translate.
#[test]
fn retained_sdf_shape_draws_and_translates() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    // Pass the SDF rect in the shapes slice; it routes to the shape stream.
    let id =
        renderer.compile_overlay_geometry(&device, &queue, &[], &[red_sdf_rect()], &[], &[], 1.0);

    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_red(rgb_at(&px, size, 32, 32)),
        "SDF rect centre should be red"
    );
    assert!(
        is_background(rgb_at(&px, size, 8, 32)),
        "left of the rect should be background"
    );

    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id).with_translate([16.0, 0.0])];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_red(rgb_at(&px, size, 56, 32)),
        "translated SDF rect should cover x=56"
    );
    assert!(
        is_background(rgb_at(&px, size, 24, 32)),
        "translated SDF rect should have vacated x=24"
    );
}

/// A retained SDF shape is clipped to the group's per-frame outer `clip_rect`,
/// and that clip is fixed in screen space while the shape scrolls under it: the
/// scroll-container case. Without the per-frame shape clip, a retained shape
/// ignored the clip and drew everywhere.
#[test]
fn retained_shape_per_frame_clip() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    // red_sdf_rect covers pixels 16..48. Clip to the right half (x >= 32).
    let id =
        renderer.compile_overlay_geometry(&device, &queue, &[], &[red_sdf_rect()], &[], &[], 1.0);
    let clip = [32.0, 0.0, 64.0, 64.0];

    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id).with_clip_rect(clip)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_red(rgb_at(&px, size, 40, 32)),
        "shape right of the clip edge should draw"
    );
    assert!(
        is_background(rgb_at(&px, size, 24, 32)),
        "shape left of the clip edge should be clipped (P9)"
    );

    // The clip is fixed screen-space; the shape scrolls under it. Translate the
    // shape left by 20 (rect -> -4..28) with the same right-half clip. Pixel x=20
    // is inside the translated rect but left of the clip edge, so it is removed;
    // without a fixed per-frame shape clip it would read red.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![
        RetainedOverlay::new(id)
            .with_translate([-20.0, 0.0])
            .with_clip_rect(clip),
    ];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_background(rgb_at(&px, size, 20, 32)),
        "scrolled shape left of the fixed clip edge should be clipped"
    );
}

/// A single row of built-in-font glyphs (ids 4.. are letters/digits in Inter),
/// drawn white.
fn glyph_run() -> GlyphRunItem {
    let mut glyphs = Vec::new();
    let mut id: u16 = 4;
    for col in 0..8 {
        glyphs.push(PositionedGlyph::new(id, 6.0 + col as f32 * 12.0, 60.0));
        id += 1;
    }
    GlyphRunItem::new(glyphs)
        .with_font_size(22.0)
        .with_colour([1.0, 1.0, 1.0, 1.0])
}

/// Count near-white pixels (drawn glyph coverage) over the grey background.
fn bright_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[0] > 200 && p[1] > 200 && p[2] > 200)
        .count()
}

/// A retained group of glyph runs draws from the cache, and survives a DPI change:
/// when `pixels_per_point` changes the group's baked glyph UVs go stale, so the
/// renderer re-emits at the new physical size and the glyphs still draw.
#[test]
fn retained_glyphs_survive_dpi_change() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 128u32;

    let id = renderer.compile_overlay_geometry(&device, &queue, &[], &[], &[glyph_run()], &[], 1.0);

    // ppp 1.0 (matches the compile): glyphs draw straight from the cache.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    let n1 = bright_pixels(&px);
    assert!(
        n1 > 10,
        "glyphs should draw at ppp 1.0, got {n1} bright pixels"
    );

    // ppp 2.0: baked ppp no longer matches, so the group re-emits at the new
    // physical size; the glyphs still draw. The offscreen target is sized to the
    // physical resolution (logical size * ppp) so the colour and depth attachments
    // match.
    let mut frame = overlay_frame(size);
    frame.camera.pixels_per_point = 2.0;
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size * 2, size * 2);
    let n2 = bright_pixels(&px);
    assert!(
        n2 > 10,
        "glyphs should still draw after a DPI change (re-emit), got {n2} bright pixels"
    );
}

/// Count near-white pixels inside a rectangular region `[x0, y0)` .. `[x1, y1)`.
fn bright_in(px: &[u8], size: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> usize {
    let mut n = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let (r, g, b) = rgb_at(px, size, x, y);
            if r > 200 && g > 200 && b > 200 {
                n += 1;
            }
        }
    }
    n
}

/// A retained `LabelItem` lays its text out once and draws from the cache; a
/// freed handle draws nothing.
#[test]
fn retained_label_draws_and_free() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 128u32;

    let label = LabelItem::new("ABCD")
        .with_font_size(22.0)
        .with_colour([1.0, 1.0, 1.0, 1.0])
        .with_align_y(AnchorY::Top)
        .with_position([8.0, 8.0]);
    let id = renderer.compile_overlay_label(&device, &queue, &label, 1.0);

    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(bright_pixels(&px) > 10, "retained label should draw glyphs");

    assert!(
        renderer.free_overlay_geometry(id),
        "free should report success"
    );
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert_eq!(bright_pixels(&px), 0, "a freed label should draw nothing");
}

/// A retained label resolves its own anchor per frame: the same compiled geometry
/// lands in the top-left when anchored to the viewport top-left, and in the
/// bottom-right when anchored to the viewport bottom-right.
#[test]
fn retained_label_anchor_resolves_to_corner() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 128u32;
    let half = size / 2;

    let top_left = LabelItem::new("ABCD")
        .with_font_size(22.0)
        .with_colour([1.0, 1.0, 1.0, 1.0])
        .with_anchor(OverlayAnchor::Viewport {
            x: AnchorX::Left,
            y: AnchorY::Top,
        })
        .with_align_x(AnchorX::Left)
        .with_align_y(AnchorY::Top)
        .with_position([6.0, 6.0]);
    let tl_id = renderer.compile_overlay_label(&device, &queue, &top_left, 1.0);

    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(tl_id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        bright_in(&px, size, 0, 0, half, half) > 10,
        "top-left-anchored label should draw in the top-left quadrant"
    );
    assert_eq!(
        bright_in(&px, size, half, half, size, size),
        0,
        "top-left-anchored label should not reach the bottom-right quadrant"
    );

    let bottom_right = LabelItem::new("ABCD")
        .with_font_size(22.0)
        .with_colour([1.0, 1.0, 1.0, 1.0])
        .with_anchor(OverlayAnchor::Viewport {
            x: AnchorX::Right,
            y: AnchorY::Bottom,
        })
        .with_align_x(AnchorX::Right)
        .with_align_y(AnchorY::Bottom)
        .with_position([-6.0, -6.0]);
    let br_id = renderer.compile_overlay_label(&device, &queue, &bottom_right, 1.0);

    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(br_id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        bright_in(&px, size, half, half, size, size) > 10,
        "bottom-right-anchored label should draw in the bottom-right quadrant"
    );
    assert_eq!(
        bright_in(&px, size, 0, 0, half, half),
        0,
        "bottom-right-anchored label should not reach the top-left quadrant"
    );
}

/// A string-in label mixes into a multi-family group: one handle carries a
/// filled square (polyline stream) and an auto-laid-out label (text stream), both
/// draw, and the per-frame translate moves the whole group as one unit. The
/// label's anchor is ignored on this path (it is placed fixed-local by position).
#[test]
fn retained_mixed_group_shape_and_label() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    // A label laid out below the red square (glyphs land around y=40..).
    let label = LabelItem::new("ABCD")
        .with_font_size(18.0)
        .with_colour([1.0, 1.0, 1.0, 1.0])
        .with_align_y(AnchorY::Top)
        .with_position([2.0, 40.0]);
    let id = renderer.compile_overlay_geometry(
        &device,
        &queue,
        &[red_square()],
        &[],
        &[],
        &[label],
        1.0,
    );

    // Both families draw from the one handle: red square centre, white glyphs below.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_red(rgb_at(&px, size, 32, 32)),
        "mixed group should draw the square"
    );
    assert!(
        bright_pixels(&px) > 10,
        "mixed group should draw the label glyphs"
    );

    // One translate moves the whole group: the square vacates its old centre.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id).with_translate([16.0, 0.0])];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_red(rgb_at(&px, size, 56, 32)),
        "translated mixed group should cover x=56"
    );
    assert!(
        is_background(rgb_at(&px, size, 24, 32)),
        "translated mixed group should have vacated x=24"
    );
}

/// A compiled group draws from its cached buffer, and the per-frame `translate`
/// moves it without re-compiling.
#[test]
fn retained_draws_and_translates() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    let id =
        renderer.compile_overlay_geometry(&device, &queue, &[red_square()], &[], &[], &[], 1.0);

    // No translate: the square covers pixels 16..48, so the centre is red and a
    // point to its left is background.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(is_red(rgb_at(&px, size, 32, 32)), "centre should be red");
    assert!(
        is_background(rgb_at(&px, size, 8, 32)),
        "left of the square should be background"
    );

    // Translate +16 in x with the SAME compiled id: the square now covers 32..64,
    // so the old centre reads background and a point at x=56 reads red.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id).with_translate([16.0, 0.0])];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_background(rgb_at(&px, size, 24, 32)),
        "translated square should have vacated x=24"
    );
    assert!(
        is_red(rgb_at(&px, size, 56, 32)),
        "translated square should now cover x=56"
    );
}

/// `opacity` 0 fades the group out; a freed handle draws nothing.
#[test]
fn retained_opacity_and_free() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let size = 64u32;

    let id =
        renderer.compile_overlay_geometry(&device, &queue, &[red_square()], &[], &[], &[], 1.0);

    // Fully transparent: the centre reads the background.
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id).with_opacity(0.0)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_background(rgb_at(&px, size, 32, 32)),
        "opacity 0 should show the background"
    );

    // Free the group; a submission referencing the freed id is skipped.
    assert!(
        renderer.free_overlay_geometry(id),
        "free should report success"
    );
    let mut frame = overlay_frame(size);
    frame.overlays.retained = vec![RetainedOverlay::new(id)];
    let px = renderer.render_offscreen(&device, &queue, &frame, size, size);
    assert!(
        is_background(rgb_at(&px, size, 32, 32)),
        "a freed group should draw nothing"
    );
}
