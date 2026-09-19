//! Overlay shadow layers across every overlay family.
//!
//! `ShadowLayer` is shared by shapes, polylines, vector paths, labels, and
//! glyph runs. The analytic shape path evaluates it from the SDF; the other
//! families are tessellated or drawn from the glyph atlas and bake it into
//! geometry or into the atlas cell instead. These tests pin two properties on
//! every family: an empty or transparent layer list changes nothing, and a
//! contour actually puts shadow-coloured pixels around the item.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

const SIZE: u32 = 96;

fn base_frame() -> FrameData {
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
    // A bright background is the case the contour exists for: white text on it
    // is invisible without one.
    frame.viewport.background_colour = Some([0.95, 0.95, 0.95, 1.0].into());
    frame
}

/// Count pixels that are close to black, which is what a dark contour adds over
/// a bright background and white geometry.
fn dark_pixels(px: &[u8]) -> usize {
    px.chunks_exact(4)
        .filter(|p| p[0] < 90 && p[1] < 90 && p[2] < 90)
        .count()
}

/// Every family: an item with no shadow layers renders exactly as it did before
/// the field existed, and a layer whose colour is fully transparent is inert.
/// This is the property that keeps the feature free for consumers not using it.
#[test]
fn absent_and_transparent_layers_are_inert() {
    use viewport_lib::{
        GlyphRunItem, LabelItem, OverlayPolylineItem, PositionedGlyph, ShadowLayer,
    };

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let invisible = ShadowLayer::outline([0.0, 0.0, 0.0, 0.0], 2.0);

    let mut plain = base_frame();
    plain.overlays.labels = vec![LabelItem::new("Wg").with_position([8.0, 8.0])];
    plain.overlays.glyph_runs = vec![GlyphRunItem::new(vec![PositionedGlyph::new(40, 8.0, 40.0)])];
    plain.overlays.polylines =
        vec![OverlayPolylineItem::new(vec![[8.0, 70.0], [88.0, 70.0]]).with_thickness(2.0)];

    let mut transparent = base_frame();
    transparent.overlays.labels = plain.overlays.labels.clone();
    transparent.overlays.glyph_runs = plain.overlays.glyph_runs.clone();
    transparent.overlays.polylines = plain.overlays.polylines.clone();
    transparent.overlays.labels[0].shadows = vec![invisible];
    transparent.overlays.glyph_runs[0].shadows = vec![invisible];
    transparent.overlays.polylines[0].shadows = vec![invisible];

    let a = renderer.render_offscreen(&device, &queue, &plain, SIZE, SIZE);
    let b = renderer.render_offscreen(&device, &queue, &transparent, SIZE, SIZE);
    assert_eq!(
        a, b,
        "a fully transparent shadow layer must not change any pixel"
    );
}

/// A label contour puts dark pixels around white glyphs on a white background,
/// where there were effectively none before.
#[test]
fn label_outline_adds_contour_pixels() {
    use viewport_lib::{LabelItem, ShadowLayer};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let label = LabelItem::new("Hg")
        .with_position([10.0, 10.0])
        .with_font_size(40.0)
        .with_colour([1.0, 1.0, 1.0, 1.0]);

    let mut without = base_frame();
    without.overlays.labels = vec![label.clone()];
    let plain_px = renderer.render_offscreen(&device, &queue, &without, SIZE, SIZE);

    let mut with = base_frame();
    with.overlays.labels = vec![label.with_shadow(ShadowLayer::outline([0.0, 0.0, 0.0, 1.0], 2.0))];
    let outlined_px = renderer.render_offscreen(&device, &queue, &with, SIZE, SIZE);

    let (plain_dark, outlined_dark) = (dark_pixels(&plain_px), dark_pixels(&outlined_px));
    assert!(
        outlined_dark > plain_dark + 40,
        "outline should add dark contour pixels: {plain_dark} without, {outlined_dark} with"
    );
}

/// The same contour on the low-level glyph-run path, which lays out glyphs from
/// caller-supplied positions and takes a separate emit path from labels.
#[test]
fn glyph_run_outline_adds_contour_pixels() {
    use viewport_lib::{GlyphRunItem, PositionedGlyph, ShadowLayer};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Glyph ids are font-specific; a spread of ids keeps the test independent of
    // which one maps to a particular character in the built-in font.
    let glyphs: Vec<PositionedGlyph> = (0..6)
        .map(|i| PositionedGlyph::new(40 + i as u16, 6.0 + i as f32 * 14.0, 60.0))
        .collect();
    let run = GlyphRunItem::new(glyphs)
        .with_font_size(36.0)
        .with_colour([1.0, 1.0, 1.0, 1.0]);

    let mut without = base_frame();
    without.overlays.glyph_runs = vec![run.clone()];
    let plain_px = renderer.render_offscreen(&device, &queue, &without, SIZE, SIZE);

    let mut with = base_frame();
    with.overlays.glyph_runs =
        vec![run.with_shadow(ShadowLayer::outline([0.0, 0.0, 0.0, 1.0], 2.0))];
    let outlined_px = renderer.render_offscreen(&device, &queue, &with, SIZE, SIZE);

    let (plain_dark, outlined_dark) = (dark_pixels(&plain_px), dark_pixels(&outlined_px));
    assert!(
        outlined_dark > plain_dark + 40,
        "glyph-run outline should add dark contour pixels: {plain_dark} without, {outlined_dark} with"
    );
}

/// A polyline contour: the tessellated path, where the shadow is the stroke
/// re-emitted wider rather than an SDF falloff or an atlas cell.
#[test]
fn polyline_outline_adds_contour_pixels() {
    use viewport_lib::{OverlayPolylineItem, ShadowLayer};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let poly = OverlayPolylineItem::new(vec![[10.0, 20.0], [50.0, 70.0], [86.0, 20.0]])
        .with_thickness(3.0)
        .with_colour([1.0, 1.0, 1.0, 1.0]);

    let mut without = base_frame();
    without.overlays.polylines = vec![poly.clone()];
    let plain_px = renderer.render_offscreen(&device, &queue, &without, SIZE, SIZE);

    let mut with = base_frame();
    with.overlays.polylines =
        vec![poly.with_shadow(ShadowLayer::outline([0.0, 0.0, 0.0, 1.0], 2.5))];
    let outlined_px = renderer.render_offscreen(&device, &queue, &with, SIZE, SIZE);

    let (plain_dark, outlined_dark) = (dark_pixels(&plain_px), dark_pixels(&outlined_px));
    assert!(
        outlined_dark > plain_dark + 40,
        "polyline outline should add dark contour pixels: {plain_dark} without, {outlined_dark} with"
    );
}

/// A vector path contour. These draw on the text pipeline rather than the SDF
/// shape pipeline, and used to ignore `shadows` entirely.
#[test]
fn vector_shape_outline_adds_contour_pixels() {
    use viewport_lib::{FillRule, OverlayFill, OverlayShapeItem, ShadowLayer, SubPath};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let triangle = SubPath::new([0.0, 0.0])
        .line_to([40.0, 0.0])
        .line_to([20.0, 36.0])
        .close();
    let shape = OverlayShapeItem::vector(
        vec![triangle],
        FillRule::NonZero,
        [28.0, 30.0],
        [40.0, 36.0],
    )
    .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0].into()));

    let mut without = base_frame();
    without.overlays.shapes = vec![shape.clone()];
    let plain_px = renderer.render_offscreen(&device, &queue, &without, SIZE, SIZE);

    let mut with = base_frame();
    with.overlays.shapes =
        vec![shape.with_shadows(vec![ShadowLayer::outline([0.0, 0.0, 0.0, 1.0], 3.0)])];
    let outlined_px = renderer.render_offscreen(&device, &queue, &with, SIZE, SIZE);

    let (plain_dark, outlined_dark) = (dark_pixels(&plain_px), dark_pixels(&outlined_px));
    assert!(
        outlined_dark > plain_dark + 40,
        "vector outline should add dark contour pixels: {plain_dark} without, {outlined_dark} with"
    );
}

/// Spread is what makes a shadow usable on thin geometry: with no spread an
/// offset-free blurred layer hides under the stroke it is meant to back.
#[test]
fn spread_widens_a_shadow_beyond_the_stroke() {
    use viewport_lib::{OverlayPolylineItem, ShadowLayer};

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let poly = OverlayPolylineItem::new(vec![[10.0, 48.0], [86.0, 48.0]])
        .with_thickness(1.0)
        .with_colour([1.0, 1.0, 1.0, 1.0]);

    let mut narrow = base_frame();
    narrow.overlays.polylines = vec![
        poly.clone()
            .with_shadow(ShadowLayer::outline([0.0, 0.0, 0.0, 1.0], 1.0)),
    ];
    let narrow_px = renderer.render_offscreen(&device, &queue, &narrow, SIZE, SIZE);

    let mut wide = base_frame();
    wide.overlays.polylines =
        vec![poly.with_shadow(ShadowLayer::outline([0.0, 0.0, 0.0, 1.0], 4.0))];
    let wide_px = renderer.render_offscreen(&device, &queue, &wide, SIZE, SIZE);

    assert!(
        dark_pixels(&wide_px) > dark_pixels(&narrow_px),
        "a larger spread must cover more pixels"
    );
}

/// `falloff` must reach all three shadow backends, not just the SDF one: the
/// analytic shape evaluates it in the fragment shader, the polyline solves it
/// into concentric band alphas, and the label bakes it into the atlas cell.
/// Each is checked by rendering the same blurred layer at two exponents.
#[test]
fn falloff_changes_output_on_every_backend() {
    use viewport_lib::{
        LabelItem, OverlayFill, OverlayPolylineItem, OverlayShape, OverlayShapeItem, ShadowLayer,
    };

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let layer = |falloff: f32| {
        ShadowLayer::new([0.0, 0.0, 0.0, 0.9], 8.0, [0.0, 0.0])
            .with_spread(2.0)
            .with_falloff(falloff)
    };

    // Analytic shape: the SDF path in overlay_shape.wgsl.
    let shape_at = |f: f32, r: &mut ViewportRenderer| {
        let mut frame = base_frame();
        frame.overlays.shapes = vec![
            OverlayShapeItem::new(OverlayShape::Circle, [32.0, 32.0], [32.0, 32.0])
                .with_fill(OverlayFill::Solid([1.0, 1.0, 1.0, 1.0].into()))
                .with_shadows(vec![layer(f)]),
        ];
        r.render_offscreen(&device, &queue, &frame, SIZE, SIZE)
    };
    assert_ne!(
        shape_at(1.0, &mut renderer),
        shape_at(4.0, &mut renderer),
        "falloff must change the SDF shape shadow"
    );

    // Polyline: the banded tessellated path.
    let poly_at = |f: f32, r: &mut ViewportRenderer| {
        let mut frame = base_frame();
        frame.overlays.polylines = vec![
            OverlayPolylineItem::new(vec![[10.0, 48.0], [86.0, 48.0]])
                .with_thickness(2.0)
                .with_colour([1.0, 1.0, 1.0, 1.0])
                .with_shadow(layer(f)),
        ];
        r.render_offscreen(&device, &queue, &frame, SIZE, SIZE)
    };
    assert_ne!(
        poly_at(1.0, &mut renderer),
        poly_at(4.0, &mut renderer),
        "falloff must change the tessellated polyline shadow"
    );

    // Label: baked into the glyph atlas cell.
    let label_at = |f: f32, r: &mut ViewportRenderer| {
        let mut frame = base_frame();
        frame.overlays.labels = vec![
            LabelItem::new("Hg")
                .with_position([16.0, 16.0])
                .with_font_size(40.0)
                .with_colour([1.0, 1.0, 1.0, 1.0])
                .with_shadow(layer(f)),
        ];
        r.render_offscreen(&device, &queue, &frame, SIZE, SIZE)
    };
    assert_ne!(
        label_at(1.0, &mut renderer),
        label_at(4.0, &mut renderer),
        "falloff must change the atlas-baked label shadow"
    );
}
