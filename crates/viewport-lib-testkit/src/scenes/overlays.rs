//! Catalogue scenes for the screen-space overlay families.
//!
//! One scene per coverage backend and per cross-cutting behaviour, so a golden
//! mismatch names what moved: analytic SDF shapes, tessellated vector shapes
//! and polylines, and glyph-atlas labels and runs. The scenes also pin the
//! behaviours that are easy to break while refactoring the overlay item types:
//! shadow layers, rotation about a non-centre pivot, clipping (rect, SDF mask,
//! nested mask), and retained groups.
//!
//! Every scene draws over the same small mesh backdrop so an overlay that
//! wrongly ends up behind the scene, or wrongly tone-mapped, shows up as a
//! difference rather than as a black frame.

use glam::Vec3;
use viewport_lib::{
    Alignment, AnchorX, AnchorY, BorderMode, Colour, FillRule, GlyphRunItem, LabelItem, LineCap,
    LineJoin, Material, OverlayFill, OverlayFrame, OverlayOrigin, OverlayPolylineItem,
    OverlayShape, OverlayShapeItem, PolylineCap, PositionedGlyph, RetainedOverlay, ShadowLayer,
    SubPath, TriangleDirection, primitives,
};

use super::{BuildCtx, BuiltScene, NamedCamera, NamedScene, orbit_camera, rigs};

/// The overlay scenes appended to the main catalogue.
pub fn scenes() -> Vec<NamedScene> {
    vec![
        scene("overlay_shapes", build_shapes),
        scene("overlay_vector", build_vector),
        scene("overlay_polylines", build_polylines),
        scene("overlay_labels", build_labels),
        scene("overlay_glyph_runs", build_glyph_runs),
        scene("overlay_shadows", build_shadows),
        scene("overlay_rotation", build_rotation),
        scene("overlay_clipping", build_clipping),
        scene("overlay_retained", build_retained),
        scene("overlay_composition", build_composition),
        // Appended rather than slotted next to the other text scenes: the
        // harness renders the catalogue through one renderer, so a scene that
        // rasterises new glyph sizes grows the shared atlas and shifts every
        // later scene's atlas UVs by a fraction of a texel. Adding a
        // glyph-bearing scene in the middle of the list re-blesses everything
        // after it for no behavioural reason.
        scene("overlay_text_fill", build_text_fill),
        scene("overlay_shadow_parity", build_shadow_parity),
        NamedScene {
            name: "overlay_group_anchor",
            // A second viewpoint for the catalogue viewer: a world anchor
            // moves with the camera. The snapshot gate renders the first
            // camera only, so the culling case is a headless test instead
            // (`headless_overlay_retained`).
            cameras: vec![
                NamedCamera {
                    name: "iso",
                    camera: orbit_camera(Vec3::ZERO, 6.0, 0.7, 1.0),
                },
                NamedCamera {
                    name: "behind",
                    camera: orbit_camera(Vec3::ZERO, 6.0, 3.6, 1.0),
                },
            ],
            build: build_group_anchor,
        },
    ]
}

/// Overlay scenes all use one viewpoint: the overlays are screen-space, so a
/// second camera would re-render the same pixels over a different backdrop.
fn scene(name: &'static str, build: fn(&mut BuildCtx<'_>) -> BuiltScene) -> NamedScene {
    NamedScene {
        name,
        cameras: vec![NamedCamera {
            name: "iso",
            camera: orbit_camera(Vec3::ZERO, 6.0, 0.7, 1.0),
        }],
        build,
    }
}

/// The same outer and inner layer on every family at once.
///
/// A shadow layer is meant to mean one thing everywhere: an outer layer dilates
/// what the item covers and is clipped to outside it, an inner layer erodes it
/// inward from the boundary. The families reach that through three different
/// coverage backends, so the gate is one image with all of them side by side:
/// a backend that drifts shows up as a mismatch here rather than as a support
/// flag quietly flipping.
///
/// The fills are translucent on purpose. That is what makes the clipping
/// visible: an unclipped outer layer paints its dilated silhouette behind the
/// fill and tints the interior.
fn build_shadow_parity(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let red = Colour::srgb(1.0, 0.25, 0.2, 1.0);
    let outer = ShadowLayer::outline(red, 4.0);
    // Narrower than the outer layer so the glyph families keep a visible core:
    // an inset band as wide as a stem swallows the letterform whole, which
    // draws but pins very little.
    let inner = ShadowLayer::new(red, 0.0, [0.0, 0.0]).with_spread(2.0);
    let fill = OverlayFill::Solid(Colour::srgb(0.95, 0.95, 0.98, 0.6));
    let cell = [86.0, 54.0];

    let mut style = viewport_lib::OverlayStyle::default();
    style.shadows = vec![outer.clone()];
    style.inner_shadows = vec![inner.clone()];

    let mut analytic = OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 8.0 },
        [24.0, 24.0],
        cell,
    );
    analytic.style = style.clone();
    analytic.style.fill = fill.clone();

    let mut vector = OverlayShapeItem::new(
        OverlayShape::Vector {
            subpaths: vec![SubPath::polygon(&[
                [0.0, 0.0],
                [cell[0], 0.0],
                [cell[0], cell[1]],
                [0.0, cell[1]],
            ])],
            fill_rule: FillRule::NonZero,
        },
        [140.0, 24.0],
        cell,
    );
    vector.style = style.clone();
    vector.style.fill = fill.clone();

    // A flat image, so what the cell pins is the shadow rather than the texel
    // filtering: this path reads the shadow layers through its own binding.
    let texture = ctx.renderer.resources_mut().upload_overlay_texture(
        ctx.device,
        ctx.queue,
        2,
        2,
        &[200u8, 205, 240, 153].repeat(4),
    );
    let mut textured = OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 8.0 },
        [256.0, 24.0],
        cell,
    );
    textured.style = style.clone();
    textured.style.fill = OverlayFill::texture(texture).with_tint(Colour::srgb(1.0, 1.0, 1.0, 1.0));

    let mut filled = OverlayPolylineItem::new(vec![
        [24.0, 110.0],
        [24.0 + cell[0], 110.0],
        [24.0 + cell[0], 110.0 + cell[1]],
        [24.0, 110.0 + cell[1]],
    ])
    .with_closed(true)
    .without_stroke();
    filled.style = style.clone();
    filled.style.fill = fill.clone();

    let mut stroke = OverlayPolylineItem::new(vec![
        [150.0, 118.0],
        [200.0, 150.0],
        [250.0, 118.0],
        [310.0, 152.0],
    ])
    .with_thickness(10.0)
    .with_colour(Colour::srgb(0.95, 0.95, 0.98, 0.6));
    stroke.style = style.clone();

    let mut label = LabelItem::new("Parity")
        .with_position([24.0, 196.0])
        .with_font_size(34.0);
    label.style = style.clone();
    label.style.fill = OverlayFill::Solid(Colour::srgb(0.95, 0.95, 0.98, 0.6));

    let mut run = GlyphRunItem::new(run_glyphs(16.0));
    run.text_style.size = 26.0;
    run.transform.translate = [190.0, 210.0];
    run.style = style.clone();
    run.style.fill = OverlayFill::Solid(Colour::srgb(0.95, 0.95, 0.98, 0.6));

    let mut retained_label = LabelItem::new("Group")
        .with_position([24.0, 246.0])
        .with_font_size(26.0);
    retained_label.style = style;
    retained_label.style.fill = OverlayFill::Solid(Colour::srgb(0.95, 0.95, 0.98, 0.6));
    let id = ctx.renderer.compile_overlay_geometry(
        ctx.device,
        ctx.queue,
        &[],
        &[],
        &[],
        std::slice::from_ref(&retained_label),
        1.0,
    );

    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = vec![analytic, vector, textured];
        ovl.polylines = vec![filled, stroke];
        ovl.labels = vec![label];
        ovl.glyph_runs = vec![run];
        ovl.retained = vec![RetainedOverlay::new(id)];
        ovl
    })
}

/// A plain lit backdrop, so overlays are composited over geometry rather than
/// over the clear colour alone.
fn backdrop(ctx: &mut BuildCtx<'_>, overlays: OverlayFrame) -> BuiltScene {
    let mesh = ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::sphere(1.6, 32, 16))
        .expect("mesh upload");
    let mut item = viewport_lib::SceneRenderItem::default();
    item.mesh_id = mesh;
    item.material = Material::pbr([0.35, 0.38, 0.45], 0.1, 0.6);
    BuiltScene {
        items: vec![item],
        overlays,
        lighting: rigs::from_above(),
        ..Default::default()
    }
}

// --- glyph runs ------------------------------------------------------------

/// Glyph ids into the built-in font's glyph table, not codepoints. Fixed
/// literals rather than a lookup: the run path takes ids as given and never
/// sees text, so what matters for the reference image is that the same ids
/// rasterise to the same bitmaps every run.
const RUN_IDS: [u16; 7] = [55, 82, 89, 87, 80, 74, 81];

fn run_glyphs(advance: f32) -> Vec<PositionedGlyph> {
    RUN_IDS
        .iter()
        .enumerate()
        .map(|(i, id)| PositionedGlyph::new(*id, i as f32 * advance, 0.0))
        .collect()
}

// --- scenes ----------------------------------------------------------------

fn build_shapes(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [20.0, 20.0],
            [80.0, 50.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.85, 0.35, 0.25, 0.95)))
        .with_border(Colour::srgb(1.0, 0.8, 0.6, 1.0), 2.0, BorderMode::Inset),
        OverlayShapeItem::new(
            OverlayShape::Rect {
                corner_radius: 12.0,
            },
            [115.0, 20.0],
            [80.0, 50.0],
        )
        .with_fill(OverlayFill::LinearGradient {
            start_colour: Colour::srgb(0.1, 0.4, 0.9, 1.0),
            end_colour: Colour::srgb(0.9, 0.2, 0.6, 1.0),
            angle: 0.6,
        })
        .with_border(Colour::srgb(0.95, 0.95, 1.0, 0.9), 2.0, BorderMode::Outer),
        OverlayShapeItem::new(OverlayShape::Circle, [215.0, 20.0], [50.0, 50.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.8, 0.45, 0.95))),
        OverlayShapeItem::new(OverlayShape::Ellipse, [285.0, 20.0], [90.0, 50.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.75, 0.2, 0.9)))
            .with_border(Colour::srgb(0.3, 0.2, 0.0, 1.0), 3.0, BorderMode::Inset),
        OverlayShapeItem::new(OverlayShape::Capsule, [20.0, 95.0], [110.0, 36.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.55, 0.3, 0.85, 0.95))),
        OverlayShapeItem::new(
            OverlayShape::Triangle {
                direction: TriangleDirection::Up,
            },
            [150.0, 90.0],
            [50.0, 46.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.25, 0.7, 0.95, 0.95))),
        OverlayShapeItem::new(
            OverlayShape::Line {
                thickness: 5.0,
                cap: LineCap::Round,
            },
            [220.0, 95.0],
            [140.0, 36.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(1.0, 1.0, 1.0, 0.9))),
    ];
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = shapes;
        ovl
    })
}

fn build_vector(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // A star and a rounded blob: one polygon path, one with curve segments, so
    // the flattening tolerance is part of the reference too.
    let star = {
        let mut pts = Vec::new();
        for i in 0..10 {
            let a = i as f32 * std::f32::consts::PI / 5.0 - std::f32::consts::FRAC_PI_2;
            let r = if i % 2 == 0 { 45.0 } else { 18.0 };
            pts.push([a.cos() * r, a.sin() * r]);
        }
        SubPath::polygon(&pts)
    };
    let blob = SubPath::new([0.0, -35.0])
        .cubic_to([45.0, -35.0], [45.0, 35.0], [0.0, 35.0])
        .cubic_to([-45.0, 35.0], [-45.0, -35.0], [0.0, -35.0])
        .close();
    let shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Vector {
                subpaths: vec![star],
                fill_rule: FillRule::NonZero,
            },
            [60.0, 60.0],
            [90.0, 90.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.95, 0.8, 0.2, 1.0))),
        OverlayShapeItem::new(
            OverlayShape::Vector {
                subpaths: vec![blob],
                fill_rule: FillRule::NonZero,
            },
            [220.0, 60.0],
            [90.0, 70.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.3, 0.75, 0.9, 1.0)))
        .with_border(Colour::srgb(0.05, 0.2, 0.3, 1.0), 2.5, BorderMode::Inset),
    ];
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = shapes;
        ovl
    })
}

fn build_polylines(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let zigzag: Vec<[f32; 2]> = (0..7)
        .map(|i| [20.0 + i as f32 * 25.0, if i % 2 == 0 { 30.0 } else { 80.0 }])
        .collect();
    let polylines = vec![
        OverlayPolylineItem::new(zigzag.clone())
            .with_thickness(6.0)
            .with_colour(Colour::srgb(1.0, 0.55, 0.2, 1.0))
            .with_join(LineJoin::Mitre)
            .with_cap(PolylineCap::Round),
        OverlayPolylineItem::new(
            zigzag
                .iter()
                .map(|p| [p[0], p[1] + 100.0])
                .collect::<Vec<_>>(),
        )
        .with_thickness(6.0)
        .with_colour(Colour::srgb(0.4, 0.85, 1.0, 1.0))
        .with_join(LineJoin::Bevel)
        .with_cap(PolylineCap::Square),
        // Closed and filled: the tessellated fill path, which is the same
        // machinery a vector shape's fill runs through.
        OverlayPolylineItem::new(vec![
            [230.0, 150.0],
            [310.0, 150.0],
            [340.0, 210.0],
            [270.0, 250.0],
            [210.0, 210.0],
        ])
        .with_closed(true)
        .with_fill(OverlayFill::Solid(Colour::srgb(0.5, 0.3, 0.8, 0.85)))
        .with_thickness(3.0)
        .with_colour(Colour::srgb(1.0, 1.0, 1.0, 0.9)),
    ];
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.polylines = polylines;
        ovl
    })
}

fn build_labels(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let labels = vec![
        LabelItem::new("Overlay label")
            .with_position([20.0, 20.0])
            .with_font_size(20.0)
            .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
        LabelItem::new("Backed and padded")
            .with_position([20.0, 55.0])
            .with_font_size(14.0)
            .with_background(true)
            .with_background_colour(Colour::srgb(0.05, 0.05, 0.1, 0.8))
            .with_padding(6.0)
            .with_border_radius(4.0),
        LabelItem::new("Wrapped text that runs past the maximum width it was given")
            .with_position([20.0, 100.0])
            .with_font_size(13.0)
            .with_max_width(160.0)
            .with_colour(Colour::srgb(0.85, 0.95, 0.7, 1.0)),
        LabelItem::new("Right, bottom")
            .with_anchor(OverlayOrigin::Viewport(Alignment::new(
                AnchorX::Right,
                AnchorY::Bottom,
            )))
            .with_position([-20.0, -20.0])
            .with_align_x(AnchorX::Right)
            .with_align_y(AnchorY::Bottom)
            .with_font_size(16.0)
            .with_colour(Colour::srgb(1.0, 0.8, 0.4, 1.0)),
    ];
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.labels = labels;
        ovl
    })
}

fn build_glyph_runs(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut per_glyph = GlyphRunItem::new(run_glyphs(14.0));
    per_glyph.text_style.size = 22.0;
    per_glyph.transform.translate = [20.0, 90.0];
    per_glyph.style.fill = OverlayFill::Solid(Colour::srgb(1.0, 1.0, 1.0, 1.0));
    per_glyph.glyph_tints = RUN_IDS
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let t = i as f32 / (RUN_IDS.len() - 1) as f32;
            Colour::srgb(1.0 - t * 0.7, 0.4 + t * 0.5, 0.3 + t * 0.6, 1.0).to_linear_rgba()
        })
        .collect();

    let mut plain = GlyphRunItem::new(run_glyphs(16.0));
    plain.text_style.size = 26.0;
    plain.transform.translate = [20.0, 30.0];
    plain.style.fill = OverlayFill::Solid(Colour::srgb(1.0, 1.0, 1.0, 1.0));

    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.glyph_runs = vec![plain, per_glyph];
        ovl
    })
}

fn build_group_anchor(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // One compiled group, submitted four ways: pinned to each viewport corner
    // with the matching alignment, and pinned to a world point. The corner
    // submissions are what a resize has to keep correct; the world one is what
    // a camera move has to keep correct, and it culls when the point goes
    // behind the camera.
    let id = compile_group(ctx);
    let retained = vec![
        RetainedOverlay::new(id)
            .with_anchor(OverlayOrigin::Viewport(Alignment::new(
                AnchorX::Left,
                AnchorY::Top,
            )))
            .with_translate([8.0, 8.0]),
        RetainedOverlay::new(id)
            .with_anchor(OverlayOrigin::Viewport(Alignment::new(
                AnchorX::Right,
                AnchorY::Bottom,
            )))
            .with_align(Alignment::new(AnchorX::Right, AnchorY::Bottom))
            .with_translate([-8.0, -8.0])
            .with_tint([0.6, 1.0, 0.7, 1.0]),
        RetainedOverlay::new(id)
            .with_anchor(OverlayOrigin::World([0.0, 0.0, -1.9]))
            .with_align(Alignment::new(AnchorX::Middle, AnchorY::Top))
            .with_scale(0.7),
    ];
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.retained = retained;
        ovl
    })
}

fn build_text_fill(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Gradient-filled text. The fill is evaluated per glyph vertex over the
    // laid-out text box, so a linear gradient is exact and a radial one is
    // piecewise-linear per glyph: both worth pinning, since the faceting is
    // the part that would show first if the sampling box drifted.
    let linear = OverlayFill::LinearGradient {
        start_colour: Colour::srgb(1.0, 0.85, 0.2, 1.0),
        end_colour: Colour::srgb(0.9, 0.2, 0.55, 1.0),
        angle: 0.0,
    };
    let radial = OverlayFill::RadialGradient {
        centre_colour: Colour::srgb(0.6, 1.0, 0.9, 1.0),
        edge_colour: Colour::srgb(0.1, 0.25, 0.6, 1.0),
    };

    let mut wide = LabelItem::new("Gradient")
        .with_position([20.0, 20.0])
        .with_font_size(44.0);
    wide.style.fill = linear.clone();

    let mut wrapped = LabelItem::new("A gradient across wrapped lines")
        .with_position([20.0, 90.0])
        .with_font_size(20.0)
        .with_max_width(180.0);
    wrapped.style.fill = radial;

    // Per-glyph colours multiply into the fill, so the run shows both.
    let mut run = GlyphRunItem::new(run_glyphs(20.0));
    run.text_style.size = 30.0;
    run.transform.translate = [20.0, 200.0];
    run.glyph_tints = RUN_IDS
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let t = i as f32 / (RUN_IDS.len() - 1) as f32;
            Colour::srgb(1.0, 1.0 - t * 0.6, 1.0, 1.0).to_linear_rgba()
        })
        .collect();
    run.style.fill = linear;

    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.labels = vec![wide, wrapped];
        ovl.glyph_runs = vec![run];
        ovl
    })
}

fn build_shadows(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // The three coverage backends produce visibly different shadow curves for
    // the same layer values. That difference is a behaviour worth pinning, so
    // the same two-layer nameplate composition is applied on a shape, a label,
    // and a glyph run.
    let nameplate = vec![
        ShadowLayer::new(Colour::srgb(0.0, 0.0, 0.0, 0.75), 4.0, [0.0, 2.0]).with_spread(2.0),
        ShadowLayer::outline(Colour::srgb(0.0, 0.0, 0.0, 0.9), 1.5),
    ];
    let shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 8.0 },
            [20.0, 20.0],
            [120.0, 50.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.9, 0.95, 1.0)))
        .with_shadows(nameplate.clone()),
        OverlayShapeItem::new(OverlayShape::Circle, [170.0, 20.0], [50.0, 50.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.6, 1.0, 1.0)))
            .with_shadows(vec![ShadowLayer::new(
                Colour::srgb(0.2, 0.6, 1.0, 0.8),
                18.0,
                [0.0, 0.0],
            )]),
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 8.0 },
            [250.0, 20.0],
            [120.0, 50.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.25, 0.27, 0.33, 1.0)))
        .with_inner_shadows(vec![ShadowLayer::new(
            Colour::srgb(0.0, 0.0, 0.0, 0.8),
            10.0,
            [0.0, 3.0],
        )]),
    ];
    let mut label = LabelItem::new("Nameplate")
        .with_position([20.0, 100.0])
        .with_font_size(24.0)
        .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0));
    label.style.shadows = nameplate.clone();

    let mut contour = LabelItem::new("Contour only")
        .with_position([20.0, 140.0])
        .with_font_size(24.0)
        .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0));
    contour.style.shadows = vec![ShadowLayer::outline(Colour::srgb(0.0, 0.0, 0.0, 1.0), 2.0)];

    let mut run = GlyphRunItem::new(run_glyphs(18.0));
    run.text_style.size = 28.0;
    run.transform.translate = [20.0, 190.0];
    run.style.fill = OverlayFill::Solid(Colour::srgb(1.0, 1.0, 1.0, 1.0));
    run.style.shadows = nameplate;

    // A polyline shadow: the tessellated backend's banded falloff.
    let polylines = vec![
        OverlayPolylineItem::new(vec![
            [230.0, 120.0],
            [300.0, 170.0],
            [250.0, 220.0],
            [360.0, 250.0],
        ])
        .with_thickness(4.0)
        .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0))
        .with_shadows(vec![ShadowLayer::new(
            Colour::srgb(0.0, 0.0, 0.0, 0.8),
            6.0,
            [2.0, 2.0],
        )]),
    ];

    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = shapes;
        ovl.labels = vec![label, contour];
        ovl.glyph_runs = vec![run];
        ovl.polylines = polylines;
        ovl
    })
}

fn build_rotation(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Every rotating family turns about a pivot away from its centre, because
    // a centre pivot hides an incorrect pivot transform.
    let off_centre = [30.0, -15.0];
    let shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 6.0 },
            [40.0, 40.0],
            [100.0, 40.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.45, 0.2, 0.95)))
        .with_rotation(0.5)
        .with_rotation_pivot(off_centre),
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 6.0 },
            [40.0, 40.0],
            [100.0, 40.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.5, 0.9, 0.35))),
    ];
    let mut label = LabelItem::new("rotated label")
        .with_position([200.0, 60.0])
        .with_font_size(20.0)
        .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0));
    label.transform.rotation = -0.7;
    label.transform.pivot = off_centre;

    let mut run = GlyphRunItem::new(run_glyphs(16.0));
    run.text_style.size = 24.0;
    run.transform.translate = [60.0, 180.0];
    run.style.fill = OverlayFill::Solid(Colour::srgb(0.7, 1.0, 0.6, 1.0));
    run.transform.rotation = 0.9;
    run.transform.pivot = off_centre;

    // Polylines rotate too. A stroked path with joins and caps, and a closed
    // filled one: the join geometry is generated per segment and is the part a
    // transform applied in the wrong order tears apart.
    let polylines = vec![
        OverlayPolylineItem::new(vec![
            [210.0, 150.0],
            [250.0, 190.0],
            [290.0, 150.0],
            [330.0, 200.0],
        ])
        .with_thickness(8.0)
        .with_colour(Colour::srgb(1.0, 0.8, 0.3, 1.0))
        .with_join(LineJoin::Mitre)
        .with_cap(PolylineCap::Round)
        .with_rotation(0.6)
        .with_rotation_pivot(off_centre),
        OverlayPolylineItem::new(vec![
            [60.0, 215.0],
            [130.0, 215.0],
            [150.0, 265.0],
            [90.0, 285.0],
            [40.0, 250.0],
        ])
        .with_closed(true)
        .with_thickness(3.0)
        .with_colour(Colour::srgb(1.0, 1.0, 1.0, 0.9))
        .with_fill(OverlayFill::Solid(Colour::srgb(0.4, 0.35, 0.8, 0.9)))
        .with_rotation(-0.5),
    ];

    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = shapes;
        ovl.labels = vec![label];
        ovl.glyph_runs = vec![run];
        ovl.polylines = polylines;
        ovl
    })
}

fn build_clipping(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // Mask 1 is a rounded rect; mask 2 is a circle nested inside it, so the
    // content clipped to 2 is clipped by both.
    let shapes = vec![
        OverlayShapeItem::new(
            OverlayShape::Rect {
                corner_radius: 16.0,
            },
            [20.0, 20.0],
            [160.0, 120.0],
        )
        .with_clip_mask(1),
        OverlayShapeItem::new(OverlayShape::Circle, [60.0, 40.0], [90.0, 90.0])
            .with_clip_mask(2)
            .with_clip(1),
        // Content clipped to the rounded rect only.
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [0.0, 40.0],
            [400.0, 24.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.3, 0.3, 0.9)))
        .with_clip(1),
        // Content clipped to the nested circle.
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [0.0, 80.0],
            [400.0, 40.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.3, 0.9, 0.5, 0.9)))
        .with_clip(2),
    ];
    let labels = vec![
        LabelItem::new("clipped to the rounded rect mask")
            .with_position([30.0, 24.0])
            .with_font_size(14.0)
            .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0))
            .with_clip(1),
        // An unclipped control: the same label with no clip, so a clip that
        // starts applying to everything is visible as this one disappearing.
        LabelItem::new("unclipped control")
            .with_position([210.0, 180.0])
            .with_font_size(18.0)
            .with_colour(Colour::srgb(1.0, 0.9, 0.5, 1.0)),
    ];
    let polylines = vec![
        OverlayPolylineItem::new(vec![[0.0, 130.0], [400.0, 100.0]])
            .with_thickness(5.0)
            .with_colour(Colour::srgb(0.4, 0.7, 1.0, 1.0))
            .with_clip(1),
    ];
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = shapes;
        ovl.labels = labels;
        ovl.polylines = polylines;
        ovl
    })
}

/// The items a retained group is compiled from, shared by the retained and
/// composition scenes so the two references differ only by how the group is
/// submitted.
fn group_items() -> (Vec<OverlayPolylineItem>, Vec<LabelItem>, Vec<GlyphRunItem>) {
    let polylines = vec![
        OverlayPolylineItem::new(vec![[0.0, 0.0], [120.0, 0.0], [120.0, 60.0], [0.0, 60.0]])
            .with_closed(true)
            .with_thickness(3.0)
            .with_colour(Colour::srgb(0.9, 0.9, 1.0, 1.0))
            .with_fill(OverlayFill::Solid(Colour::srgb(0.15, 0.2, 0.35, 0.85))),
        OverlayPolylineItem::new(vec![
            [10.0, 48.0],
            [40.0, 20.0],
            [70.0, 38.0],
            [110.0, 12.0],
        ])
        .with_thickness(2.5)
        .with_colour(Colour::srgb(1.0, 0.6, 0.3, 1.0)),
    ];
    let labels = vec![
        LabelItem::new("retained")
            .with_position([8.0, 4.0])
            .with_font_size(14.0)
            .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
    ];
    let mut run = GlyphRunItem::new(run_glyphs(10.0));
    run.text_style.size = 14.0;
    run.transform.translate = [8.0, 62.0];
    run.style.fill = OverlayFill::Solid(Colour::srgb(0.6, 1.0, 0.8, 1.0));
    (polylines, labels, vec![run])
}

fn compile_group(ctx: &mut BuildCtx<'_>) -> viewport_lib::OverlayGeometryId {
    let (polylines, labels, glyph_runs) = group_items();
    ctx.renderer.compile_overlay_geometry(
        ctx.device,
        ctx.queue,
        &polylines,
        &[],
        &glyph_runs,
        &labels,
        1.0,
    )
}

fn build_retained(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let id = compile_group(ctx);
    // One compiled group, drawn four times with different per-frame state.
    // Nothing here recompiles, so a per-frame field that quietly stopped being
    // honoured shows as four identical copies.
    let retained = vec![
        RetainedOverlay::new(id).with_translate([20.0, 20.0]),
        RetainedOverlay::new(id)
            .with_translate([180.0, 20.0])
            .with_opacity(0.45),
        RetainedOverlay::new(id)
            .with_translate([20.0, 120.0])
            .with_tint([1.0, 0.5, 0.5, 1.0]),
        RetainedOverlay::new(id)
            .with_translate([180.0, 120.0])
            .with_scale(1.35),
    ];
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.retained = retained;
        ovl
    })
}

fn build_composition(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    // The scene that fixes composition order: a group with a non-identity
    // transform holding items that carry their own non-identity transforms.
    // `group_T . item_T` and a component-wise addition of the two agree in
    // every simpler arrangement and disagree here, so this is the reference
    // that makes the contract fail a build rather than only a doc comment. The
    // Both groups rotate, which is what makes the two rules disagree: an item
    // offset inside a rotated group must travel along the group's rotated axis.
    let id = compile_group(ctx);
    let retained = vec![
        RetainedOverlay::new(id)
            .with_translate([200.0, 150.0])
            .with_rotation(0.35)
            .with_rotation_pivot([60.0, 30.0])
            .with_scale(0.9),
        RetainedOverlay::new(id)
            .with_translate([30.0, 30.0])
            .with_rotation(-0.25)
            .with_opacity(0.6),
    ];
    // Immediate items carrying their own rotation, over the same backdrop, so
    // the two roles are visible in one image.
    let shapes = vec![
        // The mask itself is never drawn.
        OverlayShapeItem::new(
            OverlayShape::Rect {
                corner_radius: 12.0,
            },
            [200.0, 10.0],
            [160.0, 120.0],
        )
        .with_clip_mask(7),
        // A rotated item clipped by that mask: the item transform and the clip
        // have to agree about which space they are in.
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 4.0 },
            [200.0, 40.0],
            [180.0, 50.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.85, 0.7, 0.2, 0.95)))
        .with_rotation(0.35)
        .with_rotation_pivot([60.0, -10.0])
        .with_clip(7),
    ];
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = shapes;
        ovl.retained = retained;
        ovl
    })
}
