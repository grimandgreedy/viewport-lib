//! `OverlayStyleSupport` against what the renderer actually draws.
//!
//! The support table is the machine-readable answer to "does setting this do
//! anything on this family", which a lowering layer branches on. A table that
//! drifts from the renderer is worse than no table, so every cell is checked
//! the only way that cannot drift: set the field, render, and see whether the
//! pixels moved.
//!
//! `shadows` and `inner_shadows` are not cells in that table: a shadow layer
//! means the same thing on every family, so the check for them is that setting
//! one always moves the pixels, whichever family it lands on.

#[cfg(feature = "wgpu29")]
use viewport_lib::wgpu;

mod common;
use common::*;

use viewport_lib::{
    BackdropEffects, Colour, GlyphRunItem, GradientStop, LabelItem, OverlayFill, OverlayFrame,
    OverlayPolylineItem, OverlayShape, OverlayShapeItem, OverlayStyle, OverlayStyleSupport,
    PositionedGlyph, ShadowLayer, SubPath,
};

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
    frame.viewport.background_colour = Some([0.5, 0.5, 0.5, 1.0].into());
    frame
}

/// The style fields the table reports on, each as "the non-default value to
/// try". `texture_transform` rides `texture` and is not separately queryable.
fn probes() -> Vec<(&'static str, fn(&mut OverlayStyle))> {
    vec![("fill", |s: &mut OverlayStyle| {
        s.fill = OverlayFill::LinearGradient {
            start_colour: Colour::srgb(1.0, 0.0, 0.0, 1.0),
            end_colour: Colour::srgb(0.0, 0.0, 1.0, 1.0),
            angle: 0.0,
        };
    })]
}

/// The two shadow lists, which every family draws. An inner layer needs a
/// spread: without one the band starts at the edge and the whole interior is
/// outside it, so a blur-only inset layer is invisible by construction.
fn shadow_probes() -> Vec<(&'static str, fn(&mut OverlayStyle))> {
    vec![
        ("shadows", |s: &mut OverlayStyle| {
            s.shadows = vec![ShadowLayer::outline(Colour::srgb(0.0, 0.0, 0.0, 1.0), 3.0)];
        }),
        ("inner_shadows", |s: &mut OverlayStyle| {
            s.inner_shadows = vec![
                ShadowLayer::new(Colour::srgb(0.0, 0.0, 0.0, 1.0), 6.0, [0.0, 0.0])
                    .with_spread(8.0),
            ];
        }),
    ]
}

fn supported(support: &OverlayStyleSupport, field: &str, style: &OverlayStyle) -> bool {
    match field {
        "fill" => support.draws_fill(&style.fill),
        other => panic!("unknown probe field {other}"),
    }
}

/// Render `build(style)` and return the pixels.
fn render(
    renderer: &mut ViewportRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    overlays: OverlayFrame,
) -> Vec<u8> {
    let mut frame = base_frame();
    frame.overlays = overlays;
    renderer.render_offscreen(device, queue, &frame, SIZE, SIZE)
}

fn differs(a: &[u8], b: &[u8]) -> bool {
    a.iter().zip(b).any(|(x, y)| x.abs_diff(*y) > 2)
}

/// For every family and every probe field: the pixels change if and only if the
/// support table says the field is drawn.
#[test]
fn reported_support_matches_what_the_renderer_draws() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // A flat two-by-two image, enough to make the textured shape pipeline the
    // one that draws: it reads the shadow layers through its own binding.
    let texture =
        renderer
            .resources_mut()
            .upload_overlay_texture(&device, &queue, 2, 2, &[255u8; 16]);

    let analytic = OverlayShape::Circle;
    let vector = OverlayShape::Vector {
        subpaths: vec![SubPath::polygon(&[
            [-20.0, -20.0],
            [20.0, -20.0],
            [20.0, 20.0],
            [-20.0, 20.0],
        ])],
        fill_rule: viewport_lib::FillRule::NonZero,
    };

    // (family name, support, builder taking a style and producing a frame).
    type Build = Box<dyn Fn(OverlayStyle) -> OverlayFrame>;
    let families: Vec<(&str, OverlayStyleSupport, Build)> = vec![
        (
            "analytic shape",
            OverlayStyleSupport::for_shape(&analytic),
            Box::new(|style| {
                let mut ovl = OverlayFrame::default();
                let mut item =
                    OverlayShapeItem::new(OverlayShape::Circle, [20.0, 20.0], [56.0, 56.0]);
                item.style = style;
                ovl.shapes = vec![item];
                ovl
            }),
        ),
        (
            "vector shape",
            OverlayStyleSupport::for_shape(&vector),
            Box::new(move |style| {
                let mut ovl = OverlayFrame::default();
                let mut item = OverlayShapeItem::new(
                    OverlayShape::Vector {
                        subpaths: vec![SubPath::polygon(&[
                            [0.0, 0.0],
                            [40.0, 0.0],
                            [40.0, 40.0],
                            [0.0, 40.0],
                        ])],
                        fill_rule: viewport_lib::FillRule::NonZero,
                    },
                    [28.0, 28.0],
                    [40.0, 40.0],
                );
                item.style = style;
                ovl.shapes = vec![item];
                ovl
            }),
        ),
        (
            "textured shape",
            OverlayStyleSupport::for_shape(&analytic),
            Box::new(move |style| {
                let mut ovl = OverlayFrame::default();
                let mut item =
                    OverlayShapeItem::new(OverlayShape::Circle, [20.0, 20.0], [56.0, 56.0]);
                item.style = style;
                ovl.shapes = vec![item];
                ovl
            }),
        ),
        (
            "polyline",
            OverlayStyleSupport::for_polyline(true),
            Box::new(|style| {
                let mut ovl = OverlayFrame::default();
                let mut item = OverlayPolylineItem::new(vec![
                    [20.0, 20.0],
                    [70.0, 20.0],
                    [70.0, 70.0],
                    [20.0, 70.0],
                ])
                .with_closed(true)
                .with_thickness(4.0)
                .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0));
                item.style.shadows = style.shadows.clone();
                item.style.inner_shadows = style.inner_shadows.clone();
                item.style.fill = style.fill.clone();
                ovl.polylines = vec![item];
                ovl
            }),
        ),
        (
            "label",
            OverlayStyleSupport::for_glyphs(),
            Box::new(|style| {
                let mut ovl = OverlayFrame::default();
                let mut item = LabelItem::new("Mg")
                    .with_position([20.0, 20.0])
                    .with_font_size(36.0);
                item.style = style;
                ovl.labels = vec![item];
                ovl
            }),
        ),
        (
            "glyph run",
            OverlayStyleSupport::for_glyphs(),
            Box::new(|style| {
                let mut ovl = OverlayFrame::default();
                let mut item = GlyphRunItem::new(vec![
                    PositionedGlyph::new(55, 0.0, 0.0),
                    PositionedGlyph::new(82, 22.0, 0.0),
                ]);
                item.text_style.size = 36.0;
                item.transform.translate = [20.0, 40.0];
                item.style = style;
                ovl.glyph_runs = vec![item];
                ovl
            }),
        ),
    ];

    for (name, support, build) in &families {
        let plain = render(
            &mut renderer,
            &device,
            &queue,
            build(base_style(name, texture)),
        );
        for (field, apply) in probes() {
            let mut style = base_style(name, texture);
            apply(&mut style);
            let drawn = supported(support, field, &style);
            let with = render(&mut renderer, &device, &queue, build(style));
            let changed = differs(&plain, &with);
            assert_eq!(
                changed,
                drawn,
                "{name}: style.{field} reported as {} but the render {} \
                 (the support table and the renderer disagree)",
                if drawn { "drawn" } else { "inert" },
                if changed { "changed" } else { "did not change" },
            );
        }
        // Shadow parity: both lists draw on every family, so there is no cell
        // to consult. A backend that quietly stops drawing one fails here.
        for (field, apply) in shadow_probes() {
            let mut style = base_style(name, texture);
            apply(&mut style);
            let with = render(&mut renderer, &device, &queue, build(style));
            assert!(
                differs(&plain, &with),
                "{name}: setting style.{field} changed nothing. A shadow layer \
                 means the same thing on every family, so every one of them \
                 has to draw it.",
            );
        }
    }
}

/// Each family needs a visible baseline, or "the pixels did not change" would
/// be true for every probe.
fn base_style(family: &str, texture: viewport_lib::OverlayTextureId) -> OverlayStyle {
    match family {
        // A text item's colour is its fill, so the glyph families need a
        // visible one to start from like everything else; the probe then swaps
        // it for a gradient.
        "label" | "glyph run" => OverlayStyle::solid(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
        // The textured pipeline is reached by the fill being a texture, so
        // that is this family's baseline: the fill probe then replaces it with
        // a gradient, which is exactly the swap the table has to predict.
        "textured shape" => OverlayStyle::default().with_fill(OverlayFill::texture(texture)),
        _ => OverlayStyle::solid(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
    }
}

/// Only a closed polyline has an interior, so a texture fill on an open one is
/// reported as inert instead of being silently dropped at draw time.
#[test]
fn an_open_polyline_reports_a_texture_fill_as_inert() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let tex = renderer
        .resources_mut()
        .upload_overlay_texture(&device, &queue, 1, 1, &[255u8; 4]);
    let style = OverlayStyle::default().with_fill(OverlayFill::texture(tex));

    assert_eq!(
        OverlayStyleSupport::for_polyline(false).inert_fields(&style),
        ["fill"],
        "an open polyline has no interior to fill"
    );
    assert!(
        OverlayStyleSupport::for_polyline(true)
            .inert_fields(&style)
            .is_empty(),
        "a closed polyline samples a texture across its interior"
    );
    // The glyph families are the other family that cannot sample an image.
    assert_eq!(
        OverlayStyleSupport::for_glyphs().inert_fields(&style),
        ["fill"]
    );
}

/// A gradient fill on a shape is not the same pixels as a solid one, so the
/// probe above is only meaningful if the baseline is genuinely visible.
#[test]
fn the_probe_baseline_actually_draws_something() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    let empty = render(&mut renderer, &device, &queue, OverlayFrame::default());
    let mut ovl = OverlayFrame::default();
    ovl.shapes = vec![
        OverlayShapeItem::new(OverlayShape::Circle, [20.0, 20.0], [56.0, 56.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(1.0, 1.0, 1.0, 1.0))),
    ];
    let drawn = render(&mut renderer, &device, &queue, ovl);
    assert!(differs(&empty, &drawn));
    let _ = GradientStop::new(0.0, Colour::srgb(0.0, 0.0, 0.0, 1.0));
    let _ = BackdropEffects::default();
}
