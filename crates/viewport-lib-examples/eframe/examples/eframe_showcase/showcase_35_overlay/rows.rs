//! The overlay page's rows, built into one `OverlayFrame`.
//!
//! Everything here depends on `viewport_lib` alone: [`build_page`] takes the
//! page clock, the control values and the uploaded assets and returns the
//! frame's overlays, so the page can be rendered without a window.

use viewport_lib as vpl;
use vpl::{
    Colour, LabelAnchor, LabelItem, LineCap, OutlineMode, OverlayBlend, OverlayFill, OverlayFrame,
    OverlayLayerStyle, OverlayPolylineItem, OverlayShape, OverlayShapeItem, PositionedGlyph,
    ShadowLayer, SubPath, TriangleDirection,
};

use std::f32::consts::{FRAC_PI_2, PI, TAU};

/// Left edge of every row.
const LEFT: f32 = 20.0;
/// Horizontal gap between the items of a row.
const GAP: f32 = 16.0;
/// Vertical gap between rows.
const ROW_GAP: f32 = 22.0;

/// Which rows the page draws. The full page is taller than a default window,
/// so it is split into two sheets that can also be stacked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sheet {
    /// Shapes, textures, gradients, shadows and strokes, transforms and
    /// animation.
    Shapes,
    /// Masks and polylines, blend modes, layers, text and meshes.
    Compositing,
    /// Both sheets, one above the other.
    Both,
}

/// Which halves of the heatmap row draw, so the overlay draw-call readout can
/// be compared between one mesh item and the same cells as shapes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HeatmapMode {
    Both,
    Mesh,
    Shapes,
}

/// What the page uploads once: textures, a compiled hairline grid, a custom
/// fill, and an optional colour-emoji font.
#[derive(Default)]
pub(crate) struct PageAssets {
    pub colour_wheel: Option<vpl::OverlayTextureId>,
    pub portrait: Option<vpl::OverlayTextureId>,
    pub nine_slice: Option<vpl::OverlayTextureId>,
    pub hairline_grid: Option<vpl::OverlayGeometryId>,
    pub hatch: Option<vpl::CustomFillId>,
    pub emoji_font: Option<vpl::FontHandle>,
    pub emoji_glyphs: Vec<u16>,
}

/// Everything one frame of the page depends on.
pub(crate) struct PageInput<'a> {
    /// Seconds on the page's animation clock.
    pub t: f32,
    pub sheet: Sheet,
    pub corner_radius: f32,
    pub border_width: f32,
    pub backdrop_blur: f32,
    pub heatmap: HeatmapMode,
    pub show_labels: bool,
    pub show_emoji: bool,
    pub assets: &'a PageAssets,
    /// The previous frame's statistics, shown in the stats strip.
    pub stats: vpl::FrameStats,
}

/// Build the page's overlays for one frame.
pub(crate) fn build_page(input: &PageInput<'_>) -> OverlayFrame {
    let mut ovl = OverlayFrame::default();
    ovl.time = input.t as f64;
    if input.show_labels {
        callout_labels(&mut ovl);
    }
    let mut y = 20.0;
    let mut row = |h: f32| {
        let top = y;
        y += h + ROW_GAP;
        top
    };
    if input.sheet != Sheet::Compositing {
        row_solid(&mut ovl, input, row(64.0));
        row_shapes(&mut ovl, input, row(64.0));
        row_textured(&mut ovl, input, row(80.0));
        row_gradients(&mut ovl, input, row(64.0));
        row_shadows_and_strokes(&mut ovl, input, row(72.0));
        row_transforms(&mut ovl, input, row(72.0));
        row_animation(&mut ovl, input, row(64.0));
    }
    if input.sheet != Sheet::Shapes {
        row_masks_and_polylines(&mut ovl, input, row(80.0));
        row_blend_modes(&mut ovl, input, row(86.0));
        row_layers(&mut ovl, input, row(76.0));
        row_text(&mut ovl, input, row(124.0));
        row_meshes(&mut ovl, input, row(70.0));
    }
    stats_strip(&mut ovl, input, row(22.0));
    ovl
}

/// Upload the page's textures, compile its hairline grid and register its
/// custom fill. The emoji font is the caller's: it needs a font parser.
pub(crate) fn build_assets(
    renderer: &mut vpl::ViewportRenderer,
    device: &vpl::wgpu::Device,
    queue: &vpl::wgpu::Queue,
) -> PageAssets {
    let mut upload = |w: u32, h: u32, rgba: Vec<u8>| {
        renderer
            .resources_mut()
            .upload_overlay_texture(device, queue, vpl::TextureData::srgb(w, h, rgba))
            .expect("the demo textures are well-formed sRGB images")
    };
    let (tw, th, wheel) = build_colour_wheel();
    let colour_wheel = Some(upload(tw, th, wheel));
    let portrait = Some(upload(
        PORTRAIT_WIDTH,
        PORTRAIT_HEIGHT,
        PORTRAIT_RGBA.to_vec(),
    ));
    let (nw, nh, button) = build_nine_slice_texture();
    let nine_slice = Some(upload(nw, nh, button));

    // Lines every 6 units over a 24-unit square, each a zero-width stroke.
    let grid: Vec<OverlayPolylineItem> = (0..=4)
        .flat_map(|i| {
            let t = 6.0 * i as f32;
            [vec![[t, 0.0], [t, 24.0]], vec![[0.0, t], [24.0, t]]]
        })
        .map(|points| {
            OverlayPolylineItem::new(points)
                .with_stroke(vpl::OverlayStroke::hairline(srgb(0.9, 0.95, 1.0, 0.9)))
        })
        .collect();
    let hairline_grid =
        Some(renderer.compile_overlay_geometry(device, queue, &grid, &[], &[], &[], 1.0));

    let hatch = renderer.resources().overlay_fill_id("hatch").or_else(|| {
        renderer
            .resources_mut()
            .register_overlay_fill(
                device,
                vpl::plugin_api::OverlayFillDesc::new("hatch", HATCH_FILL_WGSL),
            )
            .ok()
    });

    PageAssets {
        colour_wheel,
        portrait,
        nine_slice,
        hairline_grid,
        hatch,
        emoji_font: None,
        emoji_glyphs: Vec::new(),
    }
}

/// Diagonal hatching: lines `p0.x` pixels apart and `p0.y` wide, shifted
/// along their normal by `p0.z`, in `p1` over `p2`.
const HATCH_FILL_WGSL: &str = "
fn fill(local: vec2<f32>, params: OverlayFillParams) -> vec4<f32> {
    let spacing = max(params.p0.x, 1.0);
    let half_width = params.p0.y * 0.5;
    let d = (local.x + local.y) * 0.70710678 + params.p0.z;
    let m = abs(fract(d / spacing + 0.5) - 0.5) * spacing;
    let line = 1.0 - smoothstep(half_width - 0.5, half_width + 0.5, m);
    return mix(params.p2, params.p1, line);
}
";

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

/// A left-to-right cursor along one row.
struct Row {
    x: f32,
    top: f32,
    h: f32,
}

impl Row {
    fn new(top: f32, h: f32) -> Self {
        Self { x: LEFT, top, h }
    }

    /// Vertical centre of the row.
    fn mid(&self) -> f32 {
        self.top + self.h * 0.5
    }

    /// Reserve `w` pixels and return their left edge.
    fn take(&mut self, w: f32) -> f32 {
        let x = self.x;
        self.x += w + GAP;
        x
    }

    /// Reserve a `w` by `h` box centred on the row and return its top-left.
    fn place(&mut self, w: f32, h: f32) -> [f32; 2] {
        [self.take(w), self.mid() - h * 0.5]
    }
}

fn srgb(r: f32, g: f32, b: f32, a: f32) -> Colour {
    Colour::srgb(r, g, b, a)
}

fn shape(kind: OverlayShape, at: [f32; 2], size: [f32; 2]) -> OverlayShapeItem {
    OverlayShapeItem::new(kind, at, size)
}

fn rect(r: f32) -> OverlayShape {
    OverlayShape::Rect { corner_radius: r }
}

/// A small caption under an item, centred on `x`.
fn caption(ovl: &mut OverlayFrame, text: &str, x: f32, y: f32) {
    ovl.labels.push(
        LabelItem::new(text)
            .with_position([x, y])
            .with_align_x(LabelAnchor::Middle)
            .with_align_y(vpl::AnchorY::Top)
            .with_font_size(10.0)
            .with_colour(srgb(0.85, 0.88, 0.95, 1.0)),
    );
}

/// World-anchored callouts on the point cloud. A label draws text only, so a
/// contour keeps it readable over the scene.
fn callout_labels(ovl: &mut OverlayFrame) {
    for (text, pos, colour) in [
        (
            "Peak +1.5 m",
            [1.57_f32, 0.0, 1.5],
            srgb(0.3, 1.0, 0.5, 1.0),
        ),
        (
            "Trough -1.5 m",
            [-1.57, 3.14, -1.5],
            srgb(1.0, 0.4, 0.3, 1.0),
        ),
        ("Origin", [0.0, 0.0, 0.0], srgb(0.9, 0.9, 0.9, 1.0)),
    ] {
        ovl.labels.push(
            LabelItem::new(text)
                .with_world_anchor(pos)
                .with_colour(colour)
                .with_outline(srgb(0.0, 0.0, 0.0, 0.8), 2.0, OutlineMode::Outer)
                .with_align_x(LabelAnchor::Left),
        );
    }
}

// ---------------------------------------------------------------------------
// Sheet 1: shapes, fills and strokes
// ---------------------------------------------------------------------------

/// Solid shapes: every closed analytic shape with a solid fill and an inset
/// border (the corner radius and border width sliders drive them), then the
/// three border modes, inset, outer and centred, on the same rect.
fn row_solid(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let cr = input.corner_radius;
    let bw = input.border_width;
    let mut row = Row::new(top, 64.0);
    let items: [(f32, f32, OverlayShape, [f32; 3], [f32; 3]); 8] = [
        (100.0, 64.0, rect(cr), [0.12, 0.12, 0.18], [0.8, 0.8, 0.8]),
        (
            100.0,
            64.0,
            OverlayShape::RoundedRect {
                radii: [cr, 0.0, cr, 0.0],
            },
            [0.05, 0.15, 0.25],
            [0.3, 0.7, 1.0],
        ),
        (
            64.0,
            64.0,
            OverlayShape::Circle,
            [0.2, 0.5, 0.15],
            [0.4, 1.0, 0.3],
        ),
        (
            100.0,
            52.0,
            OverlayShape::Ellipse,
            [0.3, 0.1, 0.3],
            [0.8, 0.4, 1.0],
        ),
        (
            100.0,
            36.0,
            OverlayShape::Capsule,
            [0.3, 0.2, 0.05],
            [1.0, 0.8, 0.3],
        ),
        (
            64.0,
            64.0,
            OverlayShape::Ring {
                inner_radius_frac: 0.65,
            },
            [0.15, 0.35, 0.5],
            [0.3, 0.8, 1.0],
        ),
        (
            64.0,
            64.0,
            OverlayShape::Arc {
                inner_radius_frac: 0.6,
                start_angle: 0.0,
                end_angle: PI * 1.5,
            },
            [0.5, 0.2, 0.1],
            [1.0, 0.5, 0.2],
        ),
        (
            56.0,
            56.0,
            OverlayShape::Triangle {
                direction: TriangleDirection::Up,
            },
            [0.4, 0.4, 0.1],
            [1.0, 1.0, 0.3],
        ),
    ];
    for (w, h, kind, fill, border) in items {
        let at = row.place(w, h);
        ovl.shapes.push(
            shape(kind, at, [w, h])
                .with_fill(OverlayFill::Solid(srgb(fill[0], fill[1], fill[2], 0.85)))
                .with_outline(
                    srgb(border[0], border[1], border[2], 0.9),
                    bw,
                    OutlineMode::Inset,
                ),
        );
    }
    for (mode, border) in [
        (OutlineMode::Inset, [0.9, 0.9, 0.3]),
        (OutlineMode::Outer, [0.3, 0.9, 0.5]),
        (OutlineMode::Centre, [0.5, 0.5, 1.0]),
    ] {
        let at = row.place(56.0, 56.0);
        ovl.shapes.push(
            shape(rect(cr), at, [56.0, 56.0])
                .with_fill(OverlayFill::Solid(srgb(0.15, 0.15, 0.2, 0.9)))
                .with_outline(srgb(border[0], border[1], border[2], 1.0), 3.0, mode),
        );
    }
}

/// More analytic shapes: lines with each `LineCap` (butt ends flush with the
/// end points marked by the hairlines, square and round ends half the width
/// past them), stars, regular polygons and crosses.
fn row_shapes(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let bw = input.border_width;
    let mut row = Row::new(top, 64.0);

    // A diagonal line, corner to corner of its box.
    let at = row.place(90.0, 64.0);
    ovl.shapes.push(
        shape(
            OverlayShape::Line {
                thickness: 6.0,
                cap: LineCap::Round,
            },
            at,
            [90.0, 64.0],
        )
        .with_fill(OverlayFill::Solid(srgb(0.2, 0.7, 1.0, 0.9))),
    );

    // The three caps on the same horizontal segment.
    let x = row.take(130.0) + 10.0;
    let len = 110.0;
    for (k, cap) in [LineCap::Butt, LineCap::Square, LineCap::Round]
        .into_iter()
        .enumerate()
    {
        let y = top + 12.0 + k as f32 * 20.0;
        ovl.shapes.push(
            shape(
                OverlayShape::Line {
                    thickness: 10.0,
                    cap,
                },
                [x, y - 0.05],
                [len, 0.1],
            )
            .with_fill(OverlayFill::Solid(srgb(1.0, 0.6, 0.2, 0.9))),
        );
    }
    for end in [x, x + len] {
        ovl.polylines.push(
            OverlayPolylineItem::new(vec![[end, top], [end, top + 64.0]])
                .with_stroke(vpl::OverlayStroke::hairline(srgb(1.0, 1.0, 1.0, 0.8)))
                .with_z_order(1),
        );
    }

    let polygons: [(OverlayShape, [f32; 3], [f32; 3]); 6] = [
        (
            OverlayShape::Star {
                points: 5,
                inner_radius_frac: 0.45,
            },
            [1.0, 0.85, 0.1],
            [1.0, 1.0, 0.5],
        ),
        (
            OverlayShape::Star {
                points: 6,
                inner_radius_frac: 0.5,
            },
            [0.9, 0.3, 0.9],
            [1.0, 0.6, 1.0],
        ),
        (
            OverlayShape::RegularPolygon { sides: 5 },
            [0.2, 0.8, 0.4],
            [0.4, 1.0, 0.6],
        ),
        (
            OverlayShape::RegularPolygon { sides: 6 },
            [0.1, 0.5, 0.9],
            [0.3, 0.7, 1.0],
        ),
        (
            OverlayShape::Cross {
                arm_width_frac: 0.35,
            },
            [0.9, 0.2, 0.2],
            [1.0, 0.5, 0.5],
        ),
        (
            OverlayShape::Cross {
                arm_width_frac: 0.2,
            },
            [0.3, 0.8, 0.5],
            [0.5, 1.0, 0.7],
        ),
    ];
    for (kind, fill, border) in polygons {
        let at = row.place(64.0, 64.0);
        ovl.shapes.push(
            shape(kind, at, [64.0, 64.0])
                .with_fill(OverlayFill::Solid(srgb(fill[0], fill[1], fill[2], 0.9)))
                .with_outline(
                    srgb(border[0], border[1], border[2], 0.9),
                    bw,
                    OutlineMode::Inset,
                ),
        );
    }
}

/// Textured fills: an uploaded image on analytic shapes, a texture transform
/// spinning the image inside a still circle, the image tiled 3x3, and the
/// same images filling a vector path, a closed polyline with its own UVs and
/// a closed path sampled from a closure with the path bounds as UVs.
fn row_textured(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let (Some(wheel), Some(portrait)) = (input.assets.colour_wheel, input.assets.portrait) else {
        return;
    };
    let cr = input.corner_radius;
    let bw = input.border_width;
    let border = srgb(1.0, 1.0, 1.0, 0.9);
    let mut row = Row::new(top, 80.0);

    let at = row.place(80.0, 80.0);
    ovl.shapes.push(
        shape(OverlayShape::Circle, at, [80.0, 80.0])
            .with_fill(OverlayFill::texture(wheel))
            .with_outline(border, bw, OutlineMode::Inset),
    );

    let at = row.place(120.0, 80.0);
    ovl.shapes.push(
        shape(rect(cr), at, [120.0, 80.0])
            .with_fill(OverlayFill::texture(portrait))
            .with_outline(srgb(0.8, 0.8, 0.8, 0.9), bw, OutlineMode::Inset),
    );

    // The shape stays still; the texture turns inside it.
    let at = row.place(80.0, 80.0);
    ovl.shapes.push(
        shape(OverlayShape::Circle, at, [80.0, 80.0])
            .with_fill(OverlayFill::texture(wheel).with_texture_transform(
                vpl::TextureTransform::default().with_rotation(input.t * 0.5),
            ))
            .with_outline(border, bw, OutlineMode::Inset),
    );

    let at = row.place(140.0, 80.0);
    ovl.shapes.push(
        shape(rect(cr), at, [140.0, 80.0])
            .with_fill(
                OverlayFill::texture(wheel).with_texture_transform(
                    vpl::TextureTransform::default()
                        .with_scale([3.0, 3.0])
                        .with_tile_mode(vpl::TileMode::Tile),
                ),
            )
            .with_outline(border, bw, OutlineMode::Inset),
    );

    // A five-pointed star path, the image spanning the path's bounds.
    let sz = 80.0;
    let star: Vec<[f32; 2]> = (0..10)
        .map(|i| {
            let a = i as f32 * PI / 5.0 - FRAC_PI_2;
            let r = if i % 2 == 0 { sz * 0.5 } else { sz * 0.2 };
            [sz * 0.5 + a.cos() * r, sz * 0.5 + a.sin() * r]
        })
        .collect();
    let at = row.place(sz, sz);
    ovl.shapes.push(
        OverlayShapeItem::vector(
            vec![SubPath::polygon(&star)],
            vpl::FillRule::NonZero,
            at,
            [sz, sz],
        )
        .with_fill(OverlayFill::texture(wheel)),
    );

    // A closed polyline heart with explicit UVs into the portrait.
    let cx = row.take(76.0) + 38.0;
    let cy = row.mid() + 2.0;
    let heart: Vec<[f32; 2]> = (0..40)
        .map(|i| {
            let t = i as f32 / 40.0 * TAU;
            let s = t.sin();
            let c = t.cos();
            [
                cx + 2.1 * 16.0 * s.powi(3),
                cy - 2.1
                    * (13.0 * c - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos()),
            ]
        })
        .collect();
    let uvs = heart
        .iter()
        .map(|p| [(p[0] - (cx - 36.0)) / 72.0, (p[1] - (cy - 36.0)) / 72.0])
        .collect();
    ovl.polylines.push(
        OverlayPolylineItem::new(heart)
            .with_thickness(2.0)
            .with_colour(srgb(1.0, 0.78, 0.9, 0.9))
            .with_closed(true)
            .with_fill(OverlayFill::texture(portrait))
            .with_uvs(uvs),
    );

    // A rounded pentagon sampled from a closure; with no UVs the path bounds
    // map to [0, 1].
    let cx = row.take(72.0) + 36.0;
    let cy = row.mid();
    ovl.polylines.push(OverlayPolylineItem::closed_from_path(
        |u| {
            let a = u * TAU;
            let r = 30.0 + (a * 5.0).cos() * 3.0;
            [cx + a.cos() * r, cy + a.sin() * r]
        },
        60,
        OverlayFill::texture(portrait),
        Colour::linear(1.0, 0.9, 0.7, 0.9),
        2.0,
    ));
}

/// Gradients, evaluated per pixel: two-stop linear, radial and conical,
/// four- and eight-stop ramps (a stop list has no cap), a sweep over half a
/// turn like a gauge, a linear ramp on a ring, and a registered WGSL fill
/// function drawing drifting hatching.
fn row_gradients(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let cr = input.corner_radius;
    let bw = input.border_width;
    let edge = srgb(0.9, 0.9, 0.9, 0.8);
    let mut row = Row::new(top, 64.0);
    let stops = |list: &[(f32, [f32; 3])]| -> Vec<vpl::GradientStop> {
        list.iter()
            .map(|(p, c)| vpl::GradientStop::new(*p, srgb(c[0], c[1], c[2], 1.0)))
            .collect()
    };

    let mut push = |row: &mut Row, kind: OverlayShape, w: f32, fill: OverlayFill| {
        let at = row.place(w, 64.0);
        ovl.shapes
            .push(shape(kind, at, [w, 64.0]).with_fill(fill).with_outline(
                edge,
                bw,
                OutlineMode::Inset,
            ));
    };

    push(
        &mut row,
        rect(cr),
        100.0,
        OverlayFill::linear_gradient(
            srgb(0.05, 0.15, 0.55, 0.95),
            srgb(0.05, 0.65, 0.65, 0.95),
            0.0,
        ),
    );
    push(
        &mut row,
        OverlayShape::Circle,
        64.0,
        OverlayFill::radial_gradient(srgb(1.0, 0.95, 0.7, 1.0), srgb(0.2, 0.05, 0.0, 0.9)),
    );
    push(
        &mut row,
        OverlayShape::Circle,
        64.0,
        OverlayFill::conical_gradient(srgb(0.95, 0.2, 0.4, 1.0), srgb(0.2, 0.6, 1.0, 1.0), 0.0),
    );
    push(
        &mut row,
        rect(cr),
        100.0,
        OverlayFill::LinearGradient {
            stops: stops(&[
                (0.0, [0.05, 0.05, 0.20]),
                (0.4, [0.55, 0.10, 0.45]),
                (0.75, [0.95, 0.45, 0.20]),
                (1.0, [1.0, 0.95, 0.55]),
            ]),
            angle: 0.0,
        },
    );
    push(
        &mut row,
        OverlayShape::Circle,
        64.0,
        OverlayFill::ConicalGradient {
            stops: stops(&[
                (0.0, [0.95, 0.2, 0.2]),
                (0.25, [1.0, 0.85, 0.2]),
                (0.6, [0.2, 0.85, 0.45]),
                (1.0, [0.3, 0.5, 1.0]),
            ]),
            offset_angle: 0.0,
        },
    );
    let spectrum = [
        [0.90, 0.10, 0.10],
        [0.95, 0.50, 0.10],
        [0.95, 0.90, 0.15],
        [0.30, 0.85, 0.25],
        [0.10, 0.75, 0.75],
        [0.15, 0.35, 0.95],
        [0.45, 0.20, 0.85],
        [0.85, 0.20, 0.70],
    ];
    let eight: Vec<(f32, [f32; 3])> = spectrum
        .iter()
        .enumerate()
        .map(|(i, c)| (i as f32 / 7.0, *c))
        .collect();
    push(
        &mut row,
        rect(cr),
        110.0,
        OverlayFill::LinearGradient {
            stops: stops(&eight),
            angle: PI / 4.0,
        },
    );
    push(
        &mut row,
        rect(cr),
        110.0,
        OverlayFill::SweepGradient {
            stops: stops(&[
                (0.0, [0.2, 0.85, 0.45]),
                (0.6, [1.0, 0.85, 0.2]),
                (1.0, [0.95, 0.2, 0.2]),
            ]),
            centre: [0.5, 1.0],
            start_angle: -PI,
            end_angle: 0.0,
        },
    );
    push(
        &mut row,
        OverlayShape::Ring {
            inner_radius_frac: 0.6,
        },
        64.0,
        OverlayFill::linear_gradient(
            srgb(0.1, 0.3, 0.6, 1.0),
            srgb(0.7, 0.9, 1.0, 1.0),
            -PI / 4.0,
        ),
    );
    if let Some(hatch) = input.assets.hatch {
        let mut params = [0.0f32; 16];
        params[0] = 9.0;
        params[1] = 3.0;
        params[2] = input.t * 6.0;
        params[4..8].copy_from_slice(&[0.95, 0.95, 1.0, 1.0]);
        params[8..12].copy_from_slice(&[0.15, 0.2, 0.45, 0.85]);
        push(
            &mut row,
            rect(cr),
            100.0,
            OverlayFill::custom(hatch, params),
        );
    }
}

/// Shadows and strokes: a drop shadow, a coloured glow and an inner shadow;
/// a vector path stroked with marching dashes, round joins and caps; a
/// hairline grid drawn as a retained group at three times its size, its
/// lines still one device pixel wide; a brush stroke whose width is set per
/// point; an inverse-filled ring; and polyline stroke patterns (a marching
/// marquee, a round-capped dashed wave, a dotted circle).
fn row_shadows_and_strokes(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let cr = input.corner_radius;
    let bw = input.border_width;
    let t = input.t;
    let mut row = Row::new(top, 72.0);

    let at = row.place(100.0, 64.0);
    ovl.shapes.push(
        shape(rect(cr), at, [100.0, 64.0])
            .with_fill(OverlayFill::Solid(srgb(0.15, 0.15, 0.2, 0.95)))
            .with_outline(srgb(0.5, 0.5, 0.6, 0.8), bw, OutlineMode::Inset)
            .with_shadow(ShadowLayer::new(
                Colour::linear(0.0, 0.0, 0.0, 0.5),
                12.0,
                [4.0, 4.0],
            )),
    );
    row.x += 12.0;
    let at = row.place(64.0, 64.0);
    ovl.shapes.push(
        shape(OverlayShape::Circle, at, [64.0, 64.0])
            .with_fill(OverlayFill::Solid(srgb(0.1, 0.15, 0.35, 0.95)))
            .with_outline(srgb(0.3, 0.5, 1.0, 0.9), bw, OutlineMode::Inset)
            .with_shadow(ShadowLayer::new(
                Colour::linear(0.2, 0.4, 1.0, 0.6),
                16.0,
                [0.0, 0.0],
            )),
    );
    row.x += 12.0;
    // A pressed button: a dark inner shadow offset down reads as recessed.
    let at = row.place(100.0, 64.0);
    ovl.shapes.push(
        shape(rect(cr), at, [100.0, 64.0])
            .with_fill(OverlayFill::Solid(srgb(0.22, 0.24, 0.30, 1.0)))
            .with_outline(srgb(0.05, 0.07, 0.12, 0.9), 1.0, OutlineMode::Inset)
            .with_inner_shadow(ShadowLayer::new(
                Colour::linear(0.0, 0.0, 0.0, 0.7),
                14.0,
                [0.0, 4.0],
            )),
    );

    // An open cubic with a zigzag tail. The fill closes the path, the stroke
    // does not; dashes are cut before stroking so each gets its own caps.
    let path = SubPath::new([0.0, 50.0])
        .cubic_to([15.0, -10.0], [50.0, 70.0], [70.0, 15.0])
        .line_to([83.0, 45.0])
        .line_to([96.0, 10.0])
        .line_to([110.0, 50.0]);
    let at = row.place(110.0, 60.0);
    ovl.shapes.push(
        OverlayShapeItem::vector(vec![path], vpl::FillRule::NonZero, at, [110.0, 60.0])
            .with_fill(OverlayFill::Solid(srgb(0.2, 0.5, 0.9, 0.25)))
            .with_stroke(
                vpl::OverlayStroke::new(4.0, srgb(0.98, 0.85, 0.3, 1.0))
                    .with_pattern(vpl::StrokePattern::Dashed {
                        dash_length: 14.0,
                        gap_length: 7.0,
                        offset: -t * 20.0,
                    })
                    .with_join(vpl::LineJoin::Round)
                    .with_cap(LineCap::Round),
            ),
    );

    // The half-pixel offset puts the grid's line centres on pixel centres.
    if let Some(id) = input.assets.hairline_grid {
        let at = row.place(72.0, 72.0);
        ovl.retained.push(
            vpl::RetainedOverlay::new(id)
                .with_position([at[0].round() + 0.5, at[1].round() + 0.5])
                .with_scale(3.0),
        );
    }

    // A brush stroke that swells and tapers point by point.
    let x = row.take(100.0);
    let points: Vec<[f32; 2]> = (0..7)
        .map(|i| {
            [
                x + 8.0 + 14.0 * i as f32,
                if i % 2 == 0 { top + 60.0 } else { top + 12.0 },
            ]
        })
        .collect();
    ovl.polylines.push(
        OverlayPolylineItem::new(points)
            .with_colour(srgb(0.85, 0.45, 0.95, 1.0))
            .with_join(vpl::LineJoin::Round)
            .with_cap(LineCap::Round)
            .with_widths(vec![1.0, 5.0, 10.0, 14.0, 10.0, 5.0, 1.0]),
    );

    // An even-odd ring, inverse filled: the box less the ring is its four
    // corners and the central disc.
    let ring = vec![
        circle_path([32.0, 32.0], 28.0),
        circle_path([32.0, 32.0], 16.0),
    ];
    let at = row.place(64.0, 64.0);
    ovl.shapes.push(
        OverlayShapeItem::vector(ring, vpl::FillRule::EvenOdd, at, [64.0, 64.0])
            .with_inverse_fill(true)
            .with_fill(OverlayFill::Solid(srgb(0.35, 0.85, 0.5, 1.0))),
    );

    // Polyline stroke patterns.
    let mid = row.mid();
    let x = row.take(56.0);
    ovl.polylines.push(
        OverlayPolylineItem::new(vec![
            [x, mid - 26.0],
            [x + 56.0, mid - 26.0],
            [x + 56.0, mid + 26.0],
            [x, mid + 26.0],
        ])
        .with_thickness(2.0)
        .with_colour(srgb(1.0, 1.0, 1.0, 0.85))
        .with_closed(true)
        .with_stroke_pattern(vpl::StrokePattern::Dashed {
            dash_length: 8.0,
            gap_length: 6.0,
            offset: t * 20.0,
        }),
    );
    let x = row.take(70.0);
    ovl.polylines.push(
        OverlayPolylineItem::new(
            (0..=40)
                .map(|i| {
                    let f = i as f32 / 40.0;
                    [x + f * 70.0, mid + (f * TAU).sin() * 22.0]
                })
                .collect(),
        )
        .with_thickness(5.0)
        .with_colour(srgb(0.3, 0.9, 0.6, 0.9))
        .with_cap(LineCap::Round)
        .with_stroke_pattern(vpl::StrokePattern::Dashed {
            dash_length: 16.0,
            gap_length: 9.0,
            offset: 0.0,
        }),
    );
    let cx = row.take(56.0) + 28.0;
    ovl.polylines.push(
        OverlayPolylineItem::new(
            (0..48)
                .map(|i| {
                    let a = i as f32 / 48.0 * TAU;
                    [cx + a.cos() * 26.0, mid + a.sin() * 26.0]
                })
                .collect(),
        )
        .with_thickness(4.0)
        .with_colour(srgb(1.0, 0.75, 0.3, 0.9))
        .with_closed(true)
        .with_stroke_pattern(vpl::StrokePattern::Dotted {
            spacing: 10.0,
            offset: 0.0,
        }),
    );
}

/// A circle of radius `r` about `c` from four cubic quarters.
fn circle_path(c: [f32; 2], r: f32) -> SubPath {
    let k = r * 0.552_284_8;
    let [x, y] = c;
    SubPath::new([x + r, y])
        .cubic_to([x + r, y + k], [x + k, y + r], [x, y + r])
        .cubic_to([x - k, y + r], [x - r, y + k], [x - r, y])
        .cubic_to([x - r, y - k], [x - k, y - r], [x, y - r])
        .cubic_to([x + k, y - r], [x + r, y - k], [x + r, y])
        .close()
}

/// Transforms: rotation (a conical gradient turns with its circle), a card
/// sheared by an animated skew, a card stretched along each axis in turn, a
/// card placed by a whole 2x2 matrix, a metronome turning about its foot
/// rather than its centre, the portrait flipped by its texture transform, and
/// a label slanted by a skew.
fn row_transforms(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let cr = input.corner_radius;
    let bw = input.border_width;
    let t = input.t;
    let edge = srgb(1.0, 1.0, 1.0, 0.7);
    let mut row = Row::new(top, 72.0);

    let at = row.place(64.0, 64.0);
    ovl.shapes.push(
        shape(OverlayShape::Circle, at, [64.0, 64.0])
            .with_fill(OverlayFill::ConicalGradient {
                stops: [
                    (0.0, [0.27, 0.0, 0.33]),
                    (0.25, [0.13, 0.32, 0.55]),
                    (0.5, [0.12, 0.62, 0.50]),
                    (0.75, [0.78, 0.79, 0.21]),
                    (1.0, [0.99, 0.91, 0.14]),
                ]
                .iter()
                .map(|(p, c)| vpl::GradientStop::new(*p, srgb(c[0], c[1], c[2], 1.0)))
                .collect(),
                offset_angle: 0.0,
            })
            .with_rotation(t * 0.8)
            .with_outline(edge, bw, OutlineMode::Inset),
    );

    let card = |at: [f32; 2]| {
        shape(rect(cr), at, [84.0, 52.0])
            .with_fill(OverlayFill::linear_gradient(
                srgb(0.2, 0.45, 0.95, 1.0),
                srgb(0.6, 0.2, 0.8, 1.0),
                FRAC_PI_2,
            ))
            .with_outline(edge, bw, OutlineMode::Inset)
    };

    row.x += 8.0;
    let mut skewed = card(row.place(84.0, 52.0));
    skewed.transform.skew = 0.6 * (t * 1.1).sin();
    ovl.shapes.push(skewed);
    row.x += 8.0;

    let mut stretched = card(row.place(84.0, 52.0));
    let s = (t * 1.3).sin();
    stretched.transform.stretch = [1.0 + 0.25 * s.max(0.0), 1.0 + 0.3 * (-s).max(0.0)];
    ovl.shapes.push(stretched);
    row.x += 8.0;

    // A rotation times a shear, handed over as one matrix.
    let (sn, cs) = (t * 0.6).sin_cos();
    let shear = 0.3;
    row.x += 8.0;
    let at = row.place(84.0, 52.0);
    let mut matrix = card(at);
    matrix.transform =
        vpl::OverlayTransform::from_linear([[cs, cs * shear - sn], [sn, sn * shear + cs]])
            .with_translate(at);
    ovl.shapes.push(matrix);
    row.x += 16.0;

    // The metronome's arm pivots about its bottom end.
    let arm = 60.0;
    let x = row.take(80.0) + 32.0;
    ovl.shapes.push(
        shape(
            OverlayShape::Capsule,
            [x, row.mid() - arm * 0.5],
            [16.0, arm],
        )
        .with_fill(OverlayFill::Solid(srgb(0.95, 0.75, 0.2, 0.95)))
        .with_outline(srgb(1.0, 0.9, 0.5, 0.9), bw, OutlineMode::Inset)
        .with_rotation((t * 2.0).sin() * 0.7)
        .with_rotation_pivot([0.0, arm * 0.5]),
    );

    if let Some(portrait) = input.assets.portrait {
        let at = row.place(100.0, 64.0);
        ovl.shapes.push(
            shape(rect(cr), at, [100.0, 64.0])
                .with_fill(OverlayFill::texture(portrait).with_texture_transform(
                    vpl::TextureTransform::default().with_flip(false, true),
                ))
                .with_outline(srgb(0.8, 0.8, 0.8, 0.9), bw, OutlineMode::Inset),
        );
    }

    let x = row.take(150.0);
    let mut slanted = LabelItem::new("Slanted by skew")
        .with_position([x, row.mid()])
        .with_align_y(vpl::AnchorY::Middle)
        .with_font_size(20.0)
        .with_colour(srgb(0.95, 0.95, 1.0, 1.0));
    slanted.transform.skew = -0.3;
    ovl.labels.push(slanted);
}

/// Animation tracks, evaluated by the renderer from the frame's clock: one
/// shape per channel (translate, scale, tint, rotation, opacity), a fade that
/// restarts every four seconds, and a dot the page moves along a Bezier
/// figure-eight traced underneath by a polyline sampled from the same curve.
fn row_animation(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let cr = input.corner_radius;
    let bw = input.border_width;
    let t = input.t;
    let mut row = Row::new(top, 64.0);
    let anim = vpl::OverlayAnimations::default;
    let looped = |track: vpl::AnimTrack<f32>, easing| {
        track.with_easing(easing).with_repeat(vpl::RepeatMode::Loop)
    };

    let x = row.take(100.0);
    let at = [x, row.mid() - 14.0];
    ovl.shapes.push(
        shape(rect(4.0), at, [44.0, 28.0])
            .with_fill(OverlayFill::Solid(srgb(0.95, 0.65, 0.25, 0.95)))
            .with_outline(srgb(1.0, 0.85, 0.4, 0.9), bw, OutlineMode::Inset)
            .with_animations(
                anim().with_translate(
                    vpl::AnimTrack::new(0.0, 1.8, at, [x + 56.0, at[1]])
                        .with_easing(vpl::OverlayEasing::EaseInOut)
                        .with_repeat(vpl::RepeatMode::PingPong),
                ),
            ),
    );

    // Scale rides the per-draw instance and pulses about the pivot.
    let at = row.place(64.0, 44.0);
    ovl.shapes.push(
        shape(OverlayShape::Circle, [at[0] + 10.0, at[1]], [44.0, 44.0])
            .with_fill(OverlayFill::Solid(srgb(0.45, 0.85, 1.0, 0.95)))
            .with_outline(srgb(0.7, 0.95, 1.0, 0.9), bw, OutlineMode::Inset)
            .with_animations(anim().with_scale(looped(
                vpl::AnimTrack::new(0.0, 1.4, 1.0, 1.4),
                vpl::OverlayEasing::Pulse,
            ))),
    );

    // A tint multiplies a white fill, so the tint reads as the colour.
    let at = row.place(64.0, 64.0);
    ovl.shapes.push(
        shape(rect(cr), at, [64.0, 64.0])
            .with_fill(OverlayFill::Solid(srgb(1.0, 1.0, 1.0, 0.95)))
            .with_outline(srgb(1.0, 1.0, 1.0, 0.7), bw, OutlineMode::Inset)
            .with_animations(
                anim().with_tint(
                    vpl::AnimTrack::new(0.0, 1.6, [1.0, 0.45, 0.65, 1.0], [0.45, 0.75, 1.0, 1.0])
                        .with_easing(vpl::OverlayEasing::EaseInOut)
                        .with_repeat(vpl::RepeatMode::PingPong),
                ),
            ),
    );

    let at = row.place(64.0, 64.0);
    ovl.shapes.push(
        shape(
            OverlayShape::Star {
                points: 5,
                inner_radius_frac: 0.45,
            },
            at,
            [64.0, 64.0],
        )
        .with_fill(OverlayFill::Solid(srgb(0.95, 0.9, 0.3, 0.95)))
        .with_outline(srgb(1.0, 0.95, 0.5, 0.9), bw, OutlineMode::Inset)
        .with_animations(anim().with_rotation(looped(
            vpl::AnimTrack::new(0.0, 4.0, 0.0, TAU),
            vpl::OverlayEasing::Linear,
        ))),
    );

    let at = row.place(64.0, 64.0);
    ovl.shapes.push(
        shape(OverlayShape::Circle, at, [64.0, 64.0])
            .with_fill(OverlayFill::Solid(srgb(0.2, 0.5, 1.0, 0.9)))
            .with_outline(srgb(0.4, 0.7, 1.0, 0.9), bw, OutlineMode::Inset)
            .with_animations(anim().with_opacity(looped(
                vpl::AnimTrack::new(0.0, 2.0, 0.0, 0.9),
                vpl::OverlayEasing::Pulse,
            ))),
    );

    let cycle = 4.0;
    let fade_start = (input.t as f64 / cycle).floor() * cycle;
    let at = row.place(100.0, 36.0);
    ovl.shapes.push(
        shape(OverlayShape::Capsule, at, [100.0, 36.0])
            .with_fill(OverlayFill::Solid(srgb(0.6, 0.2, 0.1, 0.9)))
            .with_outline(srgb(1.0, 0.5, 0.3, 0.9), bw, OutlineMode::Inset)
            .with_animations(anim().with_opacity(vpl::AnimTrack::new(fade_start, 3.0, 0.0, 0.9))),
    );

    // A motion path is the consumer's: sample the curve at this frame's time
    // and set the position.
    let cx = row.take(200.0) + 100.0;
    let cy = row.mid();
    let dot_size = 18.0;
    let dot = figure_eight((t / 4.5).fract(), cx, cy);
    ovl.polylines.push(
        OverlayPolylineItem::from_path(
            |u| figure_eight(u, cx, cy),
            160,
            2.0,
            Colour::linear(1.0, 1.0, 1.0, 0.45),
        )
        .with_closed(true)
        .with_z_order(-1),
    );
    ovl.shapes.push(
        shape(
            OverlayShape::Circle,
            [dot[0] - dot_size * 0.5, dot[1] - dot_size * 0.5],
            [dot_size, dot_size],
        )
        .with_fill(OverlayFill::Solid(srgb(0.95, 0.45, 0.85, 1.0)))
        .with_outline(srgb(1.0, 0.7, 0.95, 0.9), bw, OutlineMode::Inset),
    );
}

/// A closed figure-eight of four cubic Bezier segments centred on
/// `(cx, cy)`, at parameter `t` in `[0, 1]`. It passes through the centre at
/// the start of the first and third segments.
fn figure_eight(t: f32, cx: f32, cy: f32) -> [f32; 2] {
    let raw = (t.clamp(0.0, 1.0) * 4.0).min(4.0 - 1e-6);
    let seg = raw.floor() as usize;
    let u = raw - seg as f32;
    let (w, h) = (80.0_f32, 36.0_f32);
    let (p0, p1, p2, p3) = match seg {
        0 => (
            [cx, cy],
            [cx + w * 0.4, cy - h * 1.4],
            [cx + w * 1.2, cy - h * 0.9],
            [cx + w * 1.2, cy],
        ),
        1 => (
            [cx + w * 1.2, cy],
            [cx + w * 1.2, cy + h * 0.9],
            [cx + w * 0.4, cy + h * 1.4],
            [cx, cy],
        ),
        2 => (
            [cx, cy],
            [cx - w * 0.4, cy + h * 1.4],
            [cx - w * 1.2, cy + h * 0.9],
            [cx - w * 1.2, cy],
        ),
        _ => (
            [cx - w * 1.2, cy],
            [cx - w * 1.2, cy - h * 0.9],
            [cx - w * 0.4, cy - h * 1.4],
            [cx, cy],
        ),
    };
    let v = 1.0 - u;
    let (w0, w1, w2, w3) = (v * v * v, 3.0 * v * v * u, 3.0 * v * u * u, u * u * u);
    [
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
    ]
}

// ---------------------------------------------------------------------------
// Sheet 2: compositing, text and meshes
// ---------------------------------------------------------------------------

/// Masks and polylines: a sweep-gradient disc clipped by a turning rounded
/// rect mask, its rotated edge anti-aliased; a button texture stretched
/// plainly and as a nine-slice; a closed polyline with a gradient fill; a
/// blob resampled from a closure every frame; and two thin zigzags (1 and 1.5
/// px) each beside a quarter-size copy drawn at four times the scale, the
/// edge fringe one device pixel either way.
fn row_masks_and_polylines(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let t = input.t;
    let mut row = Row::new(top, 80.0);

    const MASK: u32 = 7;
    let at = row.place(80.0, 80.0);
    ovl.shapes.push(
        shape(rect(14.0), [at[0] + 10.0, at[1] + 10.0], [60.0, 60.0])
            .with_rotation(t * 0.6)
            .provides_mask(MASK),
    );
    ovl.shapes.push(
        shape(OverlayShape::Circle, at, [80.0, 80.0])
            .with_fill(OverlayFill::sweep_gradient(
                srgb(0.95, 0.35, 0.2, 1.0),
                srgb(0.2, 0.6, 1.0, 1.0),
                0.0,
                TAU,
            ))
            .with_clip(MASK),
    );

    if let Some(button) = input.assets.nine_slice {
        let at = row.place(130.0, 64.0);
        ovl.shapes
            .push(shape(rect(0.0), at, [130.0, 64.0]).with_fill(OverlayFill::texture(button)));
        let at = row.place(130.0, 64.0);
        ovl.shapes.push(
            shape(rect(0.0), at, [130.0, 64.0]).with_fill(
                OverlayFill::texture(button)
                    .with_nine_slice(vpl::NineSlice::new([10.0, 10.0, 10.0, 10.0])),
            ),
        );
    }

    let x = row.take(90.0) + 40.0;
    let y = row.mid();
    ovl.polylines.push(
        OverlayPolylineItem::new(vec![
            [x, y - 30.0],
            [x + 46.0, y - 12.0],
            [x + 40.0, y + 28.0],
            [x - 18.0, y + 22.0],
            [x - 38.0, y - 10.0],
        ])
        .with_thickness(2.0)
        .with_colour(srgb(1.0, 1.0, 1.0, 0.8))
        .with_closed(true)
        .with_fill(OverlayFill::linear_gradient(
            srgb(0.12, 0.65, 0.95, 0.78),
            srgb(0.95, 0.35, 0.7, 0.82),
            PI * 0.25,
        )),
    );

    // Resampled every frame, so it is tessellated every frame.
    let cx = row.take(70.0) + 35.0;
    ovl.polylines.push(OverlayPolylineItem::closed_from_path(
        |u| {
            let a = u * TAU;
            let r = 28.0 + 6.0 * (a * 3.0 + t * 2.0).sin();
            [cx + a.cos() * r, y + a.sin() * r]
        },
        64,
        OverlayFill::radial_gradient(srgb(0.9, 0.7, 0.2, 0.85), srgb(0.7, 0.2, 0.5, 0.85)),
        Colour::linear(1.0, 1.0, 1.0, 0.85),
        2.0,
    ));

    // Compare at pixels-per-point 1, where a logical pixel is a device pixel.
    let zigzag = |x: f32, y: f32, w: f32, h: f32| -> Vec<[f32; 2]> {
        (0..5)
            .map(|i| [x + w * i as f32 / 4.0, if i % 2 == 0 { y } else { y + h }])
            .collect()
    };
    let ink = srgb(0.9, 0.95, 1.0, 1.0);
    for width in [1.0_f32, 1.5] {
        let x = row.take(112.0);
        ovl.polylines.push(
            OverlayPolylineItem::new(zigzag(x, y - 20.0, 48.0, 40.0))
                .with_thickness(width)
                .with_colour(ink),
        );
        // Scaled about its own box, so it lands centred on `x + 88`.
        ovl.polylines.push(
            OverlayPolylineItem::new(zigzag(x + 82.0, y - 5.0, 12.0, 10.0))
                .with_thickness(width)
                .with_colour(ink)
                .with_scale(4.0),
        );
    }
}

/// Blend modes, each a grey or lilac disc over the same two-colour plate:
/// the fixed-function additive, multiply and screen; difference, overlay and
/// hue, which read the destination and composite by formula; and a card in a
/// shadowed layer with a disc punched out of it by destination-out, the hole
/// showing the plate through the card and its shadow.
fn row_blend_modes(ovl: &mut OverlayFrame, _input: &PageInput<'_>, top: f32) {
    let mut row = Row::new(top, 70.0);
    let grey = srgb(0.5, 0.5, 0.5, 1.0);
    let lilac = srgb(0.7, 0.6, 1.0, 1.0);
    let cells = [
        ("Additive", OverlayBlend::Additive, grey),
        ("Multiply", OverlayBlend::Multiply, lilac),
        ("Screen", OverlayBlend::Screen, grey),
        ("Difference", OverlayBlend::Difference, lilac),
        ("Overlay", OverlayBlend::Overlay, grey),
        ("Hue", OverlayBlend::Hue, lilac),
        (
            "Destination-out",
            OverlayBlend::DestinationOut,
            Colour::WHITE,
        ),
    ];
    for (name, blend, colour) in cells {
        let x = row.take(84.0);
        let plate_top = top + 2.0;
        for (dx, plate) in [
            (0.0, srgb(0.95, 0.55, 0.15, 1.0)),
            (36.0, srgb(0.1, 0.55, 0.7, 1.0)),
        ] {
            ovl.shapes.push(
                shape(rect(0.0), [x + 6.0 + dx, plate_top], [36.0, 56.0])
                    .with_fill(OverlayFill::Solid(plate)),
            );
        }
        if blend == OverlayBlend::DestinationOut {
            // The layer's scratch target keeps the hole inside the card, and
            // its shadow falls from the punched card.
            let mark = ovl.layer_mark();
            ovl.shapes.push(
                shape(rect(8.0), [x, plate_top + 6.0], [84.0, 44.0])
                    .with_fill(OverlayFill::Solid(srgb(0.85, 0.87, 0.92, 1.0))),
            );
            ovl.shapes.push(
                shape(
                    OverlayShape::Circle,
                    [x + 26.0, plate_top + 12.0],
                    [32.0, 32.0],
                )
                .with_fill(OverlayFill::Solid(colour))
                .with_blend(blend),
            );
            ovl.push_layer(
                mark,
                OverlayLayerStyle::default().with_shadow(ShadowLayer::new(
                    srgb(0.0, 0.0, 0.0, 0.6),
                    6.0,
                    [3.0, 4.0],
                )),
            );
        } else {
            ovl.shapes.push(
                shape(
                    OverlayShape::Circle,
                    [x + 20.0, plate_top + 6.0],
                    [44.0, 44.0],
                )
                .with_fill(OverlayFill::Solid(colour))
                .with_blend(blend),
            );
        }
        caption(ovl, name, x + 42.0, top + 62.0);
    }
}

/// Layers: a card with two overlapping discs faded as one beside the same
/// items faded one by one, where the panel shows through the discs; a card
/// and badge casting one shadow from their union; a frosted card drifting
/// over bright bars, a layer with a backdrop blur clipped to a rounded mask,
/// beside a shape whose own backdrop is blurred, desaturated and hue-shifted;
/// a group blurred as a whole; and a gradient beside its greyscale twin, the
/// same item with a colour matrix.
fn row_layers(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let cr = input.corner_radius;
    let t = input.t;
    let h = 70.0;
    let mut row = Row::new(top, 76.0);

    let fade = 0.6 + 0.25 * (t * 1.5).sin();
    let faded_card = |x: f32| {
        let panel = shape(rect(cr), [x, top], [120.0, h])
            .with_fill(OverlayFill::Solid(srgb(0.85, 0.87, 0.92, 1.0)));
        let discs = [
            (16.0, srgb(0.9, 0.3, 0.25, 1.0)),
            (50.0, srgb(0.25, 0.5, 0.95, 1.0)),
        ]
        .map(|(dx, colour)| {
            shape(OverlayShape::Circle, [x + dx, top + 8.0], [54.0, 54.0])
                .with_fill(OverlayFill::Solid(colour))
        });
        [panel, discs[0].clone(), discs[1].clone()]
    };
    let x = row.take(120.0);
    let mark = ovl.layer_mark();
    ovl.shapes.extend(faded_card(x));
    ovl.push_layer(mark, OverlayLayerStyle::default().with_opacity(fade));
    let x = row.take(120.0);
    ovl.shapes
        .extend(faded_card(x).map(|item| item.with_opacity(fade)));
    caption(ovl, "one layer", x - GAP - 60.0, top + h + 2.0);
    caption(ovl, "item by item", x + 60.0, top + h + 2.0);
    row.x += 8.0;

    let x = row.take(120.0);
    let mark = ovl.layer_mark();
    ovl.shapes.push(
        shape(rect(cr), [x, top + 8.0], [96.0, 56.0])
            .with_fill(OverlayFill::Solid(srgb(0.95, 0.95, 0.97, 1.0))),
    );
    ovl.shapes.push(
        shape(OverlayShape::Circle, [x + 74.0, top - 2.0], [38.0, 38.0])
            .with_fill(OverlayFill::Solid(srgb(0.95, 0.75, 0.2, 1.0))),
    );
    ovl.push_layer(
        mark,
        OverlayLayerStyle::default().with_shadow(ShadowLayer::new(
            srgb(0.0, 0.0, 0.0, 0.6),
            10.0,
            [6.0, 8.0],
        )),
    );

    // Bright bars, then two kinds of frosted glass over them.
    let bars_x = row.take(220.0);
    for (i, colour) in [
        srgb(0.9, 0.2, 0.2, 1.0),
        srgb(0.95, 0.85, 0.2, 1.0),
        srgb(0.2, 0.75, 0.3, 1.0),
        srgb(0.2, 0.4, 0.95, 1.0),
    ]
    .into_iter()
    .cycle()
    .take(8)
    .enumerate()
    {
        ovl.shapes.push(
            shape(rect(0.0), [bars_x + i as f32 * 28.0, top], [16.0, h])
                .with_fill(OverlayFill::Solid(colour)),
        );
    }
    if input.backdrop_blur > 0.0 {
        let card = [bars_x + 50.0 + 50.0 * (t * 0.5).sin(), top + 8.0];
        let size = [100.0, 34.0];
        const MASK: u32 = 0x4c41_5952;
        ovl.shapes
            .push(shape(rect(12.0), card, size).provides_mask(MASK));
        let mark = ovl.layer_mark();
        ovl.shapes.push(
            shape(rect(12.0), card, size)
                .with_fill(OverlayFill::Solid(srgb(1.0, 1.0, 1.0, 0.18)))
                .with_outline(srgb(1.0, 1.0, 1.0, 0.35), 1.0, OutlineMode::Inset),
        );
        ovl.shapes.push(
            shape(
                rect(5.0),
                [card[0] + 12.0, card[1] + 12.0],
                [size[0] - 24.0, 10.0],
            )
            .with_fill(OverlayFill::Solid(srgb(1.0, 1.0, 1.0, 0.55))),
        );
        ovl.push_layer(
            mark,
            OverlayLayerStyle::default()
                .with_backdrop(vpl::BackdropEffects::default().with_blur(input.backdrop_blur)),
        )
        .clip = vpl::OverlayClip::mask(MASK);

        ovl.shapes.push(
            shape(rect(10.0), [bars_x + 20.0, top + 48.0], [180.0, 20.0])
                .with_fill(OverlayFill::Solid(srgb(0.4, 0.5, 0.7, 0.14)))
                .with_outline(srgb(1.0, 1.0, 1.0, 0.25), 1.0, OutlineMode::Inset)
                .with_backdrop_blur(input.backdrop_blur * 0.5)
                .with_backdrop_filters(0.35, 0.9, 0.6 * t.sin()),
        );
    }

    // Blurred as one: the blur radius breathes.
    let x = row.take(80.0);
    let mark = ovl.layer_mark();
    for (dx, dy, colour) in [
        (0.0, 6.0, srgb(0.95, 0.4, 0.3, 1.0)),
        (24.0, 20.0, srgb(0.3, 0.8, 0.5, 1.0)),
        (8.0, 34.0, srgb(0.35, 0.5, 1.0, 1.0)),
    ] {
        ovl.shapes.push(
            shape(OverlayShape::Circle, [x + dx, top + dy], [44.0, 30.0])
                .with_fill(OverlayFill::Solid(colour)),
        );
    }
    ovl.push_layer(
        mark,
        OverlayLayerStyle::default().with_blur(4.0 + 3.0 * (t * 1.2).sin()),
    );

    let gradient = OverlayFill::linear_gradient(
        srgb(0.9, 0.2, 0.2, 1.0),
        srgb(0.2, 0.4, 0.95, 1.0),
        FRAC_PI_2,
    );
    let x = row.take(44.0);
    let coloured = shape(rect(6.0), [x, top], [44.0, h]).with_fill(gradient);
    let mut grey = coloured.clone();
    grey.transform.translate = [row.take(44.0), top];
    grey.style.colour_matrix = Some(vpl::ColourMatrix::greyscale());
    ovl.shapes.push(coloured);
    ovl.shapes.push(grey);
}

/// Text: one small label at a whole-pixel pen position and at three quarter
/// pixels from it (pens land on whole device pixels with the glyph
/// rasterised at the quarter it moved from, so the copies differ only in
/// their edges); a label whose scale animates, rasterised at its drawn size
/// rather than magnified; a label filled from the portrait; a run of
/// caller-positioned, per-glyph tinted glyphs from the built-in font; and
/// colour emoji from a system font when one is found.
fn row_text(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let t = input.t;
    let ink = srgb(0.92, 0.94, 1.0, 1.0);
    for (i, fraction) in [0.0_f32, 0.25, 0.5, 0.75].into_iter().enumerate() {
        ovl.labels.push(
            LabelItem::new(format!("Pen at x + {fraction:.2} px: Hamburgefonstiv 0123"))
                .with_position([LEFT + fraction, top + i as f32 * 18.0])
                .with_align_y(vpl::AnchorY::Top)
                .with_font_size(11.0)
                .with_colour(ink),
        );
    }
    ovl.labels.push(
        LabelItem::new("Scaled, not magnified")
            .with_position([320.0, top + 20.0])
            .with_align_y(vpl::AnchorY::Top)
            .with_font_size(11.0)
            .with_colour(srgb(1.0, 0.85, 0.5, 1.0))
            .with_scale(1.5 + 0.5 * (t * 1.3).sin()),
    );
    if let Some(portrait) = input.assets.portrait {
        ovl.labels.push(
            LabelItem::new("Gauss")
                .with_position([620.0, top - 4.0])
                .with_align_y(vpl::AnchorY::Top)
                .with_font_size(64.0)
                .with_fill(OverlayFill::texture(portrait))
                .with_outline(srgb(0.95, 0.9, 0.8, 0.9), 1.5, OutlineMode::Outer),
        );
    }

    let runs_y = top + 96.0;
    let count = 16usize;
    let mut glyphs = Vec::with_capacity(count);
    let mut tints = Vec::with_capacity(count);
    for i in 0..count {
        let y = (i as f32 * 0.5 + t * 2.0).sin() * 8.0;
        glyphs.push(PositionedGlyph::new(20 + i as u16, i as f32 * 24.0, y));
        tints.push(hue_rgba(i as f32 / count as f32));
    }
    ovl.glyph_runs.push(
        vpl::GlyphRunItem::new(glyphs)
            .with_position([LEFT, runs_y])
            .with_font_size(26.0)
            .with_glyph_tints(tints),
    );

    // Colour glyphs draw their own bitmap and take only the run's opacity.
    if let Some(font) = input.assets.emoji_font
        && input.show_emoji
    {
        let size = 34.0;
        let glyphs: Vec<PositionedGlyph> = input
            .assets
            .emoji_glyphs
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let y = (i as f32 * 0.6 + t * 2.0).sin() * 5.0;
                PositionedGlyph::new(id, i as f32 * size * 1.15, y)
            })
            .collect();
        ovl.glyph_runs.push(
            vpl::GlyphRunItem::new(glyphs)
                .with_font(font)
                .with_position([440.0, runs_y])
                .with_font_size(size),
        );
    }
}

/// A saturated rainbow colour for hue `h` in 0..1.
fn hue_rgba(h: f32) -> [f32; 4] {
    let f = |n: f32| {
        let k = (n + h * 6.0) % 6.0;
        1.0 - k.min(4.0 - k).clamp(0.0, 1.0)
    };
    [f(5.0), f(3.0), f(1.0), 1.0]
}

/// The colour of heatmap cell `(i, j)` at time `t`: a drifting field through
/// a blue to red ramp.
fn heat(i: usize, j: usize, t: f32) -> Colour {
    let v = ((i as f32 * 0.5 + t).sin() * 0.5 + (j as f32 * 0.8 - t * 0.7).cos() * 0.5) * 0.5 + 0.5;
    srgb(
        0.1 + 0.85 * v,
        0.25 + 0.4 * (1.0 - (v - 0.5).abs() * 2.0),
        0.9 - 0.8 * v,
        1.0,
    )
}

/// Meshes: a heatmap as one `OverlayMeshItem` (four vertices per cell, one
/// flat colour each, no edge fringe so cells meet without seams) beside the
/// same cells as one rect shape each. Each cell shape takes its own
/// `z_order`; the cells do not overlap, so they still draw as one batch. A
/// ribbon mesh with UVs carries the portrait.
fn row_meshes(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    const COLS: usize = 24;
    const ROWS: usize = 5;
    const CELL: f32 = 14.0;
    let t = input.t;
    let mut row = Row::new(top, 70.0);
    let mesh_x = row.take(COLS as f32 * CELL);
    let shapes_x = row.take(COLS as f32 * CELL);
    if input.heatmap != HeatmapMode::Shapes {
        let mut positions = Vec::with_capacity(COLS * ROWS * 4);
        let mut colours = Vec::with_capacity(COLS * ROWS * 4);
        let mut indices = Vec::with_capacity(COLS * ROWS * 6);
        for j in 0..ROWS {
            for i in 0..COLS {
                let (x, y) = (i as f32 * CELL, j as f32 * CELL);
                let base = positions.len() as u32;
                positions.extend_from_slice(&[
                    [x, y],
                    [x + CELL, y],
                    [x + CELL, y + CELL],
                    [x, y + CELL],
                ]);
                colours.extend_from_slice(&[heat(i, j, t); 4]);
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
        ovl.meshes.push(
            vpl::OverlayMeshItem::new(positions, indices)
                .with_colours(colours)
                .with_position([mesh_x, top]),
        );
    }
    if input.heatmap != HeatmapMode::Mesh {
        for j in 0..ROWS {
            for i in 0..COLS {
                ovl.shapes.push(
                    shape(
                        rect(0.0),
                        [shapes_x + i as f32 * CELL, top + j as f32 * CELL],
                        [CELL, CELL],
                    )
                    .with_fill(OverlayFill::Solid(heat(i, j, t)))
                    .with_z_order((j * COLS + i) as i32),
                );
            }
        }
    }

    // A wavy ribbon: two rows of vertices, UVs across its length.
    if let Some(portrait) = input.assets.portrait {
        let x = row.take(180.0);
        let n = 24;
        let mut positions = Vec::with_capacity(2 * (n + 1));
        let mut uvs = Vec::with_capacity(2 * (n + 1));
        for k in 0..=n {
            let u = k as f32 / n as f32;
            let wave = (u * TAU + t * 1.5).sin() * 8.0;
            positions.push([u * 180.0, 8.0 + wave]);
            positions.push([u * 180.0, 62.0 + wave]);
            uvs.push([u, 0.0]);
            uvs.push([u, 1.0]);
        }
        let indices: Vec<u32> = (0..n as u32)
            .flat_map(|k| {
                let a = 2 * k;
                [a, a + 1, a + 2, a + 1, a + 3, a + 2]
            })
            .collect();
        ovl.meshes.push(
            vpl::OverlayMeshItem::new(positions, indices)
                .with_uvs(uvs)
                .with_fill(OverlayFill::texture(portrait))
                .with_position([x, top]),
        );
    }
}

/// The renderer's overlay counters for the previous frame: draw calls,
/// tessellations (vector shapes and polylines whose geometry changed, such as
/// the blob and the marching dashes; static and merely moving items are
/// cached), glyph and vector-shape coverage rasters (zero for static content
/// once cached), and layers composited through a scratch target.
fn stats_strip(ovl: &mut OverlayFrame, input: &PageInput<'_>, top: f32) {
    let s = &input.stats;
    let text = format!(
        "draw calls {}   tessellations {}   glyph rasters {}   vector rasters {}   layers composited {}",
        s.overlay_draw_calls,
        s.overlay_tessellations,
        s.overlay_glyph_rasters,
        s.overlay_vector_rasters,
        s.overlay_layers_composited,
    );
    ovl.shapes.push(
        shape(rect(6.0), [LEFT - 6.0, top - 4.0], [600.0, 22.0])
            .with_fill(OverlayFill::Solid(srgb(0.05, 0.06, 0.09, 0.8))),
    );
    ovl.labels.push(
        LabelItem::new(text)
            .with_position([LEFT, top])
            .with_align_y(vpl::AnchorY::Top)
            .with_font_size(12.0)
            .with_colour(srgb(0.85, 0.95, 0.85, 1.0))
            .with_z_order(1),
    );
}

// ---------------------------------------------------------------------------
// Demo textures
// ---------------------------------------------------------------------------

/// Carl Gauss portrait: raw RGBA, 1500 x 1000.
const PORTRAIT_WIDTH: u32 = 1500;
const PORTRAIT_HEIGHT: u32 = 1000;
const PORTRAIT_RGBA: &[u8] = include_bytes!("../assets/carlgauss.rgba");

/// A 128 x 128 colour wheel: hue turns with angle and saturation rises from
/// the centre.
fn build_colour_wheel() -> (u32, u32, Vec<u8>) {
    let size: u32 = 128;
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let nx = (x as f32 / (size - 1) as f32) * 2.0 - 1.0;
            let ny = (y as f32 / (size - 1) as f32) * 2.0 - 1.0;
            let hue = (ny.atan2(nx) / TAU + 0.5).fract();
            let sat = (nx * nx + ny * ny).sqrt().min(1.0);
            let [r, g, b] = hsv_to_rgb(hue, sat, 1.0);
            pixels.extend_from_slice(&[
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (b * 255.0) as u8,
                255,
            ]);
        }
    }
    (size, size, pixels)
}

/// A 64 x 32 button for the nine-slice demo: 10 px rounded corners, a
/// vertical gradient body, a dark rim and a bright bevel along the top.
/// Stretched plainly its corners elongate into ovals; as a nine-slice with
/// 10 px insets they keep their size and only the centre stretches.
fn build_nine_slice_texture() -> (u32, u32, Vec<u8>) {
    let w: u32 = 64;
    let h: u32 = 32;
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    let radius = 10.0_f32;
    let rim = 2.0_f32;
    let body_top = [0.30_f32, 0.62, 0.90];
    let body_bottom = [0.10_f32, 0.25, 0.55];
    let rim_col = [0.04_f32, 0.08, 0.18];
    let bevel = [0.80_f32, 0.92, 1.0];
    // Signed distance to the rounded rect, negative inside.
    let sd_rounded = |fx: f32, fy: f32| -> f32 {
        let (half_w, half_h) = (w as f32 * 0.5, h as f32 * 0.5);
        let qx = (fx - half_w).abs() - half_w + radius;
        let qy = (fy - half_h).abs() - half_h + radius;
        let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt();
        qx.max(qy).min(0.0) + outside - radius
    };
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let d = sd_rounded(fx, fy);
            if d > 0.5 {
                continue;
            }
            let v = (y as f32 / (h - 1) as f32).clamp(0.0, 1.0);
            let body = [0, 1, 2].map(|k| body_top[k] * (1.0 - v) + body_bottom[k] * v);
            let c = if d > -rim {
                rim_col
            } else if d > -(rim + 1.0) && fy < h as f32 * 0.5 {
                bevel
            } else {
                body
            };
            let a = (1.0 - (d + 0.5).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            let i = ((y * w + x) * 4) as usize;
            pixels[i..i + 4].copy_from_slice(&[
                (c[0] * 255.0) as u8,
                (c[1] * 255.0) as u8,
                (c[2] * 255.0) as u8,
                (a * 255.0) as u8,
            ]);
        }
    }
    (w, h, pixels)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let i = (h * 6.0).floor() as u32 % 6;
    let f = h * 6.0 - (h * 6.0).floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}
