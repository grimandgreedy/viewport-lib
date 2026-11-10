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
    Alignment, AnchorX, AnchorY, BackdropEffects, Colour, ColourMatrix, FillRule, GlyphRunItem,
    LabelItem, LineCap, LineJoin, Material, OutlineMode, OverlayBlend, OverlayClip, OverlayFill,
    OverlayFrame, OverlayGroupItems, OverlayLayerStyle, OverlayMeshItem, OverlayOrigin,
    OverlayPolylineItem, OverlayShape, OverlayShapeItem, OverlayStroke, PositionedGlyph,
    RetainedOverlay, ShadowLayer, StrokePattern, SubPath, TriangleDirection, primitives,
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
        scene("overlay_blend_modes", build_blend_modes),
        scene("overlay_strokes", build_strokes),
        scene("overlay_text_subpixel", build_text_subpixel),
        scene("overlay_layers", build_layers),
        scene("overlay_mesh", build_mesh),
        scene("overlay_layers_nested", build_layers_nested),
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
            stroke: None,
            inverse: false,
        },
        [140.0, 24.0],
        cell,
    );
    vector.style = style.clone();
    vector.style.fill = fill.clone();

    // A flat image, so what the cell pins is the shadow rather than the texel
    // filtering: this path reads the shadow layers through its own binding.
    let texture = ctx
        .renderer
        .resources_mut()
        .upload_overlay_texture(
            ctx.device,
            ctx.queue,
            viewport_lib::TextureData::srgb(2, 2, [200u8, 205, 240, 153].repeat(4)),
        )
        .expect("a well-formed sRGB image");
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
///
/// An overlay-only renderer has no meshes, so there the overlays draw over the
/// clear colour.
fn backdrop(ctx: &mut BuildCtx<'_>, overlays: OverlayFrame) -> BuiltScene {
    let mesh = match ctx
        .renderer
        .resources_mut()
        .upload_mesh_data(ctx.device, &primitives::sphere(1.6, 32, 16))
    {
        Ok(mesh) => mesh,
        Err(viewport_lib::ViewportError::TierUnsupported { .. }) => {
            return BuiltScene {
                overlays,
                ..Default::default()
            };
        }
        Err(e) => panic!("mesh upload: {e}"),
    };
    let mut item = viewport_lib::SceneRenderItem::default();
    item.mesh_id = mesh;
    item.material = super::material(
        ctx,
        Material::pbr(Colour::linear_rgb(0.35, 0.38, 0.45), 0.1, 0.6),
    );
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
/// rasterise to the same bitmaps every run. They name letters with marks above,
/// below and stacked, so a misplaced glyph box shows up as a clipped accent:
/// Ǻ Ɔ Ș Ď Ç Ƈ Ȼ. Changing the built-in font changes what these ids name.
const RUN_IDS: [u16; 7] = [744, 153, 877, 807, 745, 154, 236];

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
        .with_outline(Colour::srgb(1.0, 0.8, 0.6, 1.0), 2.0, OutlineMode::Inset),
        OverlayShapeItem::new(
            OverlayShape::Rect {
                corner_radius: 12.0,
            },
            [115.0, 20.0],
            [80.0, 50.0],
        )
        .with_fill(OverlayFill::linear_gradient(
            Colour::srgb(0.1, 0.4, 0.9, 1.0),
            Colour::srgb(0.9, 0.2, 0.6, 1.0),
            0.6,
        ))
        .with_outline(Colour::srgb(0.95, 0.95, 1.0, 0.9), 2.0, OutlineMode::Outer),
        OverlayShapeItem::new(OverlayShape::Circle, [215.0, 20.0], [50.0, 50.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.8, 0.45, 0.95))),
        OverlayShapeItem::new(OverlayShape::Ellipse, [285.0, 20.0], [90.0, 50.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.75, 0.2, 0.9)))
            .with_outline(Colour::srgb(0.3, 0.2, 0.0, 1.0), 3.0, OutlineMode::Inset),
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
                stroke: None,
                inverse: false,
            },
            [60.0, 60.0],
            [90.0, 90.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.95, 0.8, 0.2, 1.0))),
        OverlayShapeItem::new(
            OverlayShape::Vector {
                subpaths: vec![blob],
                fill_rule: FillRule::NonZero,
                stroke: None,
                inverse: false,
            },
            [220.0, 60.0],
            [90.0, 70.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.3, 0.75, 0.9, 1.0)))
        .with_outline(Colour::srgb(0.05, 0.2, 0.3, 1.0), 2.5, OutlineMode::Inset),
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
            .with_cap(LineCap::Round),
        OverlayPolylineItem::new(
            zigzag
                .iter()
                .map(|p| [p[0], p[1] + 100.0])
                .collect::<Vec<_>>(),
        )
        .with_thickness(6.0)
        .with_colour(Colour::srgb(0.4, 0.85, 1.0, 1.0))
        .with_join(LineJoin::Bevel)
        .with_cap(LineCap::Square),
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
    // A label draws text and nothing else, so its backing is a shape. The
    // panel is the measured text plus padding on every side, placed on the
    // label's own resolved box rather than on its position: a label aligns its
    // box against the origin, so the two are not the same point.
    let backed = LabelItem::new("Backed and padded")
        .with_position([20.0, 55.0])
        .with_font_size(14.0)
        .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0));
    let backed_pad = 6.0;
    let backed_size = ctx.renderer.resources().measure_overlay_text(
        &backed.text,
        backed.text_style.size,
        backed.text_style.font,
    );
    // A viewport origin at the top-left resolves to [0, 0] whatever the
    // viewport size, so the scene does not need to know what it renders at.
    let backed_tl = backed
        .resolve_top_left(
            [backed_size.width, backed_size.height],
            backed_size.ascent,
            [0.0, 0.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
        )
        .expect("a viewport-anchored label always resolves");
    let backing = OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 4.0 },
        [backed_tl[0] - backed_pad, backed_tl[1] - backed_pad],
        [
            backed_size.width + backed_pad * 2.0,
            backed_size.height + backed_pad * 2.0,
        ],
    )
    .with_fill(OverlayFill::Solid(Colour::srgb(0.05, 0.05, 0.1, 0.8)));

    let labels = vec![
        LabelItem::new("Overlay label")
            .with_position([20.0, 20.0])
            .with_font_size(20.0)
            .with_colour(Colour::srgb(1.0, 1.0, 1.0, 1.0)),
        backed,
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
        ovl.shapes = vec![backing];
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
            .with_position([8.0, 8.0]),
        RetainedOverlay::new(id)
            .with_anchor(OverlayOrigin::Viewport(Alignment::new(
                AnchorX::Right,
                AnchorY::Bottom,
            )))
            .with_align(Alignment::new(AnchorX::Right, AnchorY::Bottom))
            .with_position([-8.0, -8.0])
            .with_tint([0.6, 1.0, 0.7, 1.0]),
        RetainedOverlay::new(id)
            .with_anchor(OverlayOrigin::World([0.0, 0.0, -1.9].into()))
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
    // Gradient-filled text. The fill is evaluated per fragment over the
    // laid-out text box, so a radial gradient is as smooth across a glyph as
    // a linear one; both are worth pinning, since the box the fill spans is
    // the part that would show first if it drifted.
    let linear = OverlayFill::linear_gradient(
        Colour::srgb(1.0, 0.85, 0.2, 1.0),
        Colour::srgb(0.9, 0.2, 0.55, 1.0),
        0.0,
    );
    let radial = OverlayFill::radial_gradient(
        Colour::srgb(0.6, 1.0, 0.9, 1.0),
        Colour::srgb(0.1, 0.25, 0.6, 1.0),
    );

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
        .with_cap(LineCap::Round)
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
        .provides_mask(1),
        OverlayShapeItem::new(OverlayShape::Circle, [60.0, 40.0], [90.0, 90.0])
            .provides_mask(2)
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
        RetainedOverlay::new(id).with_position([20.0, 20.0]),
        RetainedOverlay::new(id)
            .with_position([180.0, 20.0])
            .with_opacity(0.45),
        RetainedOverlay::new(id)
            .with_position([20.0, 120.0])
            .with_tint([1.0, 0.5, 0.5, 1.0]),
        RetainedOverlay::new(id)
            .with_position([180.0, 120.0])
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
            .with_position([200.0, 150.0])
            .with_rotation(0.35)
            .with_rotation_pivot([60.0, 30.0])
            .with_scale(0.9),
        RetainedOverlay::new(id)
            .with_position([30.0, 30.0])
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
        .provides_mask(7),
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

// --- blend modes -----------------------------------------------------------

/// One cell per blend mode, each a two-colour plate with a disc drawn over it
/// in that mode, the disc hanging off the plate's corner onto the backdrop.
///
/// Top row: additive brightens, multiply darkens, screen brightens without
/// clipping to white. Bottom row: destination-out punches a hole through the
/// plate and the scene; destination-over fills such a hole and nothing else;
/// source-atop paints a disc only where the plate is left after a hole.
fn build_blend_modes(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let plate = |shapes: &mut Vec<OverlayShapeItem>, at: [f32; 2]| {
        for (dx, colour) in [
            (0.0, Colour::srgb(0.95, 0.55, 0.15, 1.0)),
            (50.0, Colour::srgb(0.1, 0.55, 0.7, 1.0)),
        ] {
            shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [at[0] + dx, at[1]],
                    [50.0, 100.0],
                )
                .with_fill(OverlayFill::Solid(colour)),
            );
        }
    };
    let disc = |at: [f32; 2], colour: Colour, blend: OverlayBlend| {
        OverlayShapeItem::new(
            OverlayShape::Circle,
            [at[0] + 40.0, at[1] + 40.0],
            [76.0, 76.0],
        )
        .with_fill(OverlayFill::Solid(colour))
        .with_blend(blend)
    };
    let hole = |at: [f32; 2]| {
        OverlayShapeItem::new(
            OverlayShape::Circle,
            [at[0] + 15.0, at[1] + 15.0],
            [50.0, 50.0],
        )
        .with_fill(OverlayFill::Solid(Colour::WHITE))
        .with_blend(OverlayBlend::DestinationOut)
    };
    let mut shapes = Vec::new();

    let grey = Colour::srgb(0.5, 0.5, 0.5, 1.0);
    for (at, blend, colour) in [
        ([20.0, 20.0], OverlayBlend::Additive, grey),
        (
            [150.0, 20.0],
            OverlayBlend::Multiply,
            Colour::srgb(0.7, 0.6, 1.0, 1.0),
        ),
        ([280.0, 20.0], OverlayBlend::Screen, grey),
    ] {
        plate(&mut shapes, at);
        shapes.push(disc(at, colour, blend));
    }

    // Destination-out on its own.
    let at = [20.0, 160.0];
    plate(&mut shapes, at);
    shapes.push(disc(at, Colour::WHITE, OverlayBlend::DestinationOut));
    // A hole, then a disc that shows only through it.
    let at = [150.0, 160.0];
    plate(&mut shapes, at);
    shapes.push(hole(at));
    shapes.push(disc(
        [at[0] - 30.0, at[1] - 30.0],
        Colour::srgb(0.9, 0.85, 0.2, 1.0),
        OverlayBlend::DestinationOver,
    ));
    // A hole, then a disc kept only where the plate is left.
    let at = [280.0, 160.0];
    plate(&mut shapes, at);
    shapes.push(hole(at));
    shapes.push(disc(
        [at[0] - 30.0, at[1] - 30.0],
        Colour::srgb(0.35, 0.85, 0.35, 1.0),
        OverlayBlend::SourceAtop,
    ));

    // A rising z_order per item, the way a toolkit lowers a draw list: the
    // cells do not overlap, so they still batch.
    for (z, shape) in shapes.iter_mut().enumerate() {
        shape.z_order = z as i32;
    }
    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = shapes;
        ovl
    })
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

/// Strokes on any path: a dashed open curve with round joins, a closed star
/// with mitred tips, a hairline grid under a 3x group scale that must stay one
/// device pixel wide, a polyline whose width varies point by point, and an
/// inverse-filled ring at two sizes, one drawn from the coverage atlas and one
/// past its limit tessellated.
fn build_strokes(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let mut shapes = Vec::new();

    // An open cubic with a zigzag tail: the fill would close it, the stroke
    // does not. The dashes are cut before stroking, so each dash gets round
    // caps and round joins at the zigzag's corners.
    let curve = SubPath::new([0.0, 90.0])
        .cubic_to([20.0, -10.0], [80.0, 130.0], [110.0, 30.0])
        .line_to([130.0, 70.0])
        .line_to([150.0, 20.0])
        .line_to([170.0, 80.0]);
    shapes.push(
        OverlayShapeItem::vector(vec![curve], FillRule::NonZero, [15.0, 20.0], [170.0, 110.0])
            .with_fill(OverlayFill::Solid(Colour::srgb(0.2, 0.5, 0.9, 0.25)))
            .with_stroke(
                OverlayStroke::new(5.0, Colour::srgb(0.98, 0.85, 0.3, 1.0))
                    .with_pattern(StrokePattern::Dashed {
                        dash_length: 18.0,
                        gap_length: 9.0,
                        offset: 4.0,
                    })
                    .with_join(LineJoin::Round)
                    .with_cap(LineCap::Round),
            ),
    );

    // A closed star, mitred at every tip and notch.
    let star: Vec<[f32; 2]> = (0..10)
        .map(|i| {
            let a = std::f32::consts::PI * i as f32 / 5.0 - std::f32::consts::FRAC_PI_2;
            let r = if i % 2 == 0 { 48.0 } else { 20.0 };
            [50.0 + r * a.cos(), 52.0 + r * a.sin()]
        })
        .collect();
    shapes.push(
        OverlayShapeItem::vector(
            vec![SubPath::polygon(&star)],
            FillRule::NonZero,
            [205.0, 20.0],
            [100.0, 100.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.9, 0.3, 0.35, 0.6)))
        .with_stroke(
            OverlayStroke::new(4.0, Colour::srgb(0.98, 0.95, 0.9, 1.0))
                .with_join(LineJoin::Mitre)
                .with_mitre_limit(8.0),
        ),
    );

    // An inverse-filled ring: the box less a ring is its corners and the
    // central disc. Small enough to come from the coverage atlas.
    let ring = |c: f32, outer: f32, inner: f32| {
        vec![circle_path([c, c], outer), circle_path([c, c], inner)]
    };
    shapes.push(
        OverlayShapeItem::vector(
            ring(35.0, 30.0, 18.0),
            FillRule::EvenOdd,
            [315.0, 30.0],
            [70.0, 70.0],
        )
        .with_inverse_fill(true)
        .with_fill(OverlayFill::Solid(Colour::srgb(0.35, 0.85, 0.5, 1.0))),
    );
    // The same past the atlas limit, through the tessellated fallback.
    shapes.push(
        OverlayShapeItem::vector(
            ring(68.0, 58.0, 34.0),
            FillRule::EvenOdd,
            [262.0, 155.0],
            [136.0, 136.0],
        )
        .with_inverse_fill(true)
        .with_fill(OverlayFill::Solid(Colour::srgb(0.35, 0.85, 0.5, 1.0))),
    );

    // A brush stroke: round joins, round caps, and a width that swells and
    // tapers point by point.
    let brush_points: Vec<[f32; 2]> = (0..7)
        .map(|i| {
            [
                140.0 + 18.0 * i as f32,
                if i % 2 == 0 { 250.0 } else { 175.0 },
            ]
        })
        .collect();
    let brush = OverlayPolylineItem::new(brush_points)
        .with_colour(Colour::srgb(0.85, 0.45, 0.95, 1.0))
        .with_join(LineJoin::Round)
        .with_cap(LineCap::Round)
        .with_widths(vec![1.0, 5.0, 10.0, 14.0, 10.0, 5.0, 1.0]);

    // A hairline grid compiled once and drawn at three times its size: the
    // lines stay one device pixel wide. Line centres land on pixel centres.
    let mut grid = Vec::new();
    for i in 0..=6 {
        let t = 6.0 * i as f32;
        for points in [vec![[t, 0.0], [t, 36.0]], vec![[0.0, t], [36.0, t]]] {
            grid.push(
                OverlayPolylineItem::new(points)
                    .with_stroke(OverlayStroke::hairline(Colour::srgb(0.95, 0.95, 0.95, 1.0))),
            );
        }
    }
    let id =
        ctx.renderer
            .compile_overlay_geometry(ctx.device, ctx.queue, &grid, &[], &[], &[], 1.0);

    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.shapes = shapes;
        ovl.polylines = vec![brush];
        ovl.retained = vec![
            RetainedOverlay::new(id)
                .with_position([15.5, 160.5])
                .with_scale(3.0),
        ];
        ovl
    })
}

// --- sub-pixel text ----------------------------------------------------------

/// Small text at fractional pen positions, and text whose transform scale
/// picks its raster size.
///
/// The left column is one label stepped across by a quarter pixel per row,
/// with whole-pixel row spacing: each row lands on whole device pixels and
/// differs from the one above only in the glyph edges rasterised at that
/// quarter. The right column is a small label scaled 1x, 1.5x, 2x and 3x
/// about its top-left, each rasterised at its drawn size, beside a 3x scaled
/// rotated copy that rasterises at 3x and resamples.
fn build_text_subpixel(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let ink = Colour::srgb(1.0, 1.0, 1.0, 1.0);
    let mut labels = Vec::new();
    for i in 0..8 {
        labels.push(
            LabelItem::new("Quarter pixel steps 0123")
                .with_position([20.0 + i as f32 * 0.25, 20.0 + i as f32 * 14.0])
                .with_align_x(AnchorX::Left)
                .with_align_y(AnchorY::Top)
                .with_font_size(11.0)
                .with_colour(ink),
        );
    }

    let small = |position: [f32; 2]| {
        LabelItem::new("Scaled")
            .with_position(position)
            .with_align_x(AnchorX::Left)
            .with_align_y(AnchorY::Top)
            .with_font_size(9.0)
            .with_colour(Colour::srgb(1.0, 0.85, 0.5, 1.0))
    };
    let box_size = ctx
        .renderer
        .resources()
        .measure_overlay_text("Scaled", 9.0, None);
    // About the top-left of the text box: the pivot is measured from its centre.
    let top_left = [-box_size.width * 0.5, -box_size.height * 0.5];
    let mut y = 20.0;
    for scale in [1.0, 1.5, 2.0, 3.0] {
        labels.push(
            small([220.0, y])
                .with_scale(scale)
                .with_rotation_pivot(top_left),
        );
        y += box_size.height * scale + 6.0;
    }
    labels.push(small([240.0, y + 10.0]).with_scale(3.0).with_rotation(0.2));

    backdrop(ctx, {
        let mut ovl = OverlayFrame::default();
        ovl.labels = labels;
        ovl
    })
}

/// Layers: overlay items drawn into a scratch target and composited as one.
///
/// Top left, a faded layer of overlapping discs beside the same discs faded
/// one by one: the layer shows only the top disc where they overlap, the twin
/// shows the lower one through it. Top right, a row of bars with a frosted
/// card over it, the card a layer with a backdrop blur clipped to a rounded
/// mask, so the bars blur under it, and a plain item with a backdrop blur over
/// the bars' feet. Bottom left, a group of overlapping shapes
/// casting one drop shadow from their union. Bottom middle, a layer composited
/// with `Difference` over coloured stripes. Bottom right, a gradient rect
/// beside its twin filtered to greyscale by a colour matrix.
fn build_layers(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let rect = |at: [f32; 2], size: [f32; 2], r: f32, colour: Colour| {
        OverlayShapeItem::new(OverlayShape::Rect { corner_radius: r }, at, size)
            .with_fill(OverlayFill::Solid(colour))
    };
    let disc = |at: [f32; 2], d: f32, colour: Colour| {
        OverlayShapeItem::new(OverlayShape::Circle, at, [d, d])
            .with_fill(OverlayFill::Solid(colour))
    };
    let red = Colour::srgb(0.9, 0.2, 0.2, 1.0);
    let green = Colour::srgb(0.2, 0.75, 0.3, 1.0);
    let blue = Colour::srgb(0.2, 0.4, 0.95, 1.0);
    let yellow = Colour::srgb(0.95, 0.85, 0.2, 1.0);
    let mut ovl = OverlayFrame::default();

    // A faded layer of overlapping discs, and the same discs faded one by one.
    let discs = |x: f32| {
        [
            disc([x, 20.0], 56.0, red),
            disc([x + 26.0, 34.0], 56.0, green),
            disc([x + 8.0, 60.0], 56.0, blue),
        ]
    };
    let mark = ovl.layer_mark();
    ovl.shapes.extend(discs(14.0));
    ovl.push_layer(mark, OverlayLayerStyle::default().with_opacity(0.5));
    for d in discs(100.0) {
        ovl.shapes.push(d.with_opacity(0.5));
    }

    // A row of bars, then a frosted card over them: a layer with a backdrop
    // blur clipped to the card's rounded mask.
    for (i, colour) in [red, yellow, green, blue, red, yellow]
        .into_iter()
        .enumerate()
    {
        ovl.shapes.push(rect(
            [200.0 + i as f32 * 30.0, 20.0],
            [18.0, 100.0],
            0.0,
            colour,
        ));
    }
    ovl.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect {
                corner_radius: 14.0,
            },
            [230.0, 40.0],
            [120.0, 60.0],
        )
        .provides_mask(41),
    );
    let mark = ovl.layer_mark();
    ovl.shapes.push(rect(
        [230.0, 40.0],
        [120.0, 60.0],
        14.0,
        Colour::srgb(1.0, 1.0, 1.0, 0.25),
    ));
    ovl.shapes.push(rect(
        [246.0, 62.0],
        [88.0, 16.0],
        8.0,
        Colour::srgb(1.0, 1.0, 1.0, 0.6),
    ));
    let card = ovl.push_layer(
        mark,
        OverlayLayerStyle::default().with_backdrop(BackdropEffects::default().with_blur(6.0)),
    );
    card.clip = OverlayClip::mask(41);

    // A plain item with a backdrop blur over the bars' feet: the bars blur
    // under it too.
    ovl.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 7.0 },
            [196.0, 104.0],
            [52.0, 14.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.1, 0.1, 0.15, 0.3)))
        .with_backdrop_blur(4.0),
    );

    // A group casting one drop shadow from the union of its members.
    let mark = ovl.layer_mark();
    ovl.shapes.push(rect(
        [24.0, 170.0],
        [70.0, 50.0],
        10.0,
        Colour::srgb(0.95, 0.95, 0.98, 1.0),
    ));
    ovl.shapes.push(disc([70.0, 196.0], 54.0, yellow));
    ovl.push_layer(
        mark,
        OverlayLayerStyle::default().with_shadow(ShadowLayer::new(
            Colour::srgb(0.0, 0.0, 0.0, 0.7),
            6.0,
            [8.0, 8.0],
        )),
    );

    // Coloured stripes under a layer composited by difference.
    for (i, colour) in [red, green, blue, yellow].into_iter().enumerate() {
        ovl.shapes.push(rect(
            [160.0 + i as f32 * 25.0, 160.0],
            [25.0, 110.0],
            0.0,
            colour,
        ));
    }
    let mark = ovl.layer_mark();
    ovl.shapes.push(disc([170.0, 175.0], 80.0, Colour::WHITE));
    ovl.shapes.push(rect(
        [150.0, 240.0],
        [120.0, 20.0],
        0.0,
        Colour::srgb(0.5, 0.5, 0.5, 1.0),
    ));
    ovl.push_layer(
        mark,
        OverlayLayerStyle::default().with_blend(OverlayBlend::Difference),
    );

    // A gradient rect and its greyscale twin.
    let gradient = OverlayFill::linear_gradient(red, blue, 0.0);
    let coloured = OverlayShapeItem::new(
        OverlayShape::Rect { corner_radius: 6.0 },
        [290.0, 160.0],
        [44.0, 110.0],
    )
    .with_fill(gradient.clone());
    let mut grey = coloured.clone();
    grey.transform.translate = [342.0, 160.0];
    grey.style.colour_matrix = Some(ColourMatrix::greyscale());
    ovl.shapes.push(coloured);
    ovl.shapes.push(grey);

    backdrop(ctx, ovl)
}

/// Nested layers: layers inside layers, each composited into the target of
/// the layer around it.
///
/// Top left, a blurred group holding a panel, two discs and a faded card of
/// two overlapping rects: the card is faded as one inside the group, then
/// the whole group blurs. Top right, a faded layer of coloured bars with a
/// white disc composited by `Difference` inside it: the disc inverts the
/// bars it covers, not the scene under the layer. Along the bottom, three
/// levels: a shadowed group around a faded layer of bars, inside which a
/// frosted card (a layer with a backdrop blur clipped to a rounded mask)
/// blurs the bars the faded layer has drawn under it, and a plain item with
/// a backdrop blur does the same over the bars' feet.
fn build_layers_nested(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    let rect = |at: [f32; 2], size: [f32; 2], r: f32, colour: Colour| {
        OverlayShapeItem::new(OverlayShape::Rect { corner_radius: r }, at, size)
            .with_fill(OverlayFill::Solid(colour))
    };
    let disc = |at: [f32; 2], d: f32, colour: Colour| {
        OverlayShapeItem::new(OverlayShape::Circle, at, [d, d])
            .with_fill(OverlayFill::Solid(colour))
    };
    let red = Colour::srgb(0.9, 0.2, 0.2, 1.0);
    let green = Colour::srgb(0.2, 0.75, 0.3, 1.0);
    let blue = Colour::srgb(0.2, 0.4, 0.95, 1.0);
    let yellow = Colour::srgb(0.95, 0.85, 0.2, 1.0);
    let mut ovl = OverlayFrame::default();

    // A faded card inside a blurred group.
    let group = ovl.layer_mark();
    ovl.shapes.push(rect(
        [14.0, 14.0],
        [170.0, 120.0],
        12.0,
        Colour::srgb(0.15, 0.17, 0.22, 1.0),
    ));
    ovl.shapes.push(disc([28.0, 30.0], 60.0, red));
    ovl.shapes.push(disc([62.0, 50.0], 60.0, blue));
    let card = ovl.layer_mark();
    ovl.shapes
        .push(rect([96.0, 36.0], [76.0, 60.0], 8.0, Colour::WHITE));
    ovl.shapes
        .push(rect([84.0, 70.0], [76.0, 46.0], 8.0, yellow));
    ovl.push_layer(card, OverlayLayerStyle::default().with_opacity(0.5));
    ovl.push_layer(group, OverlayLayerStyle::default().with_blur(2.0));

    // A difference disc inside a faded layer of bars.
    let faded = ovl.layer_mark();
    for (i, colour) in [red, green, blue, yellow].into_iter().enumerate() {
        ovl.shapes.push(rect(
            [212.0 + i as f32 * 44.0, 14.0],
            [44.0, 120.0],
            0.0,
            colour,
        ));
    }
    ovl.shapes
        .push(disc([248.0, 30.0], 90.0, Colour::WHITE).with_blend(OverlayBlend::Difference));
    ovl.push_layer(faded, OverlayLayerStyle::default().with_opacity(0.7));

    // Three levels: a shadowed group, a faded layer of bars, a frosted card.
    ovl.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect {
                corner_radius: 14.0,
            },
            [120.0, 180.0],
            [160.0, 70.0],
        )
        .provides_mask(51),
    );
    let shadowed = ovl.layer_mark();
    let bars = ovl.layer_mark();
    for (i, colour) in [red, yellow, green, blue, red, yellow, green, blue]
        .into_iter()
        .enumerate()
    {
        ovl.shapes.push(rect(
            [40.0 + i as f32 * 42.0, 166.0],
            [24.0, 110.0],
            0.0,
            colour,
        ));
    }
    let frosted = ovl.layer_mark();
    ovl.shapes.push(rect(
        [120.0, 180.0],
        [160.0, 70.0],
        14.0,
        Colour::srgb(1.0, 1.0, 1.0, 0.25),
    ));
    ovl.shapes.push(rect(
        [140.0, 206.0],
        [120.0, 18.0],
        9.0,
        Colour::srgb(1.0, 1.0, 1.0, 0.6),
    ));
    let card = ovl.push_layer(
        frosted,
        OverlayLayerStyle::default().with_backdrop(BackdropEffects::default().with_blur(6.0)),
    );
    card.clip = OverlayClip::mask(51);
    ovl.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 7.0 },
            [36.0, 258.0],
            [120.0, 14.0],
        )
        .with_fill(OverlayFill::Solid(Colour::srgb(0.1, 0.1, 0.15, 0.3)))
        .with_backdrop_blur(4.0),
    );
    ovl.push_layer(bars, OverlayLayerStyle::default().with_opacity(0.9));
    ovl.push_layer(
        shadowed,
        OverlayLayerStyle::default().with_shadow(ShadowLayer::new(
            Colour::srgb(0.0, 0.0, 0.0, 0.7),
            6.0,
            [6.0, 6.0],
        )),
    );

    backdrop(ctx, ovl)
}

/// The hatching fill the mesh scene registers: diagonal lines `p0.x` pixels
/// apart and `p0.y` wide, in `p1` over `p2`.
pub const HATCH_FILL_WGSL: &str = "
fn fill(local: vec2<f32>, params: OverlayFillParams) -> vec4<f32> {
    let spacing = max(params.p0.x, 1.0);
    let half_width = params.p0.y * 0.5;
    let d = (local.x + local.y) * 0.70710678;
    let m = abs(fract(d / spacing + 0.5) - 0.5) * spacing;
    let line = 1.0 - smoothstep(half_width - 0.5, half_width + 0.5, m);
    return mix(params.p2, params.p1, line);
}
";

/// The hatching fill, registered on first use.
pub fn hatch_fill(
    renderer: &mut viewport_lib::ViewportRenderer,
    device: &viewport_lib::wgpu::Device,
) -> viewport_lib::CustomFillId {
    if let Some(id) = renderer.resources().overlay_fill_id("hatch") {
        return id;
    }
    renderer
        .resources_mut()
        .register_overlay_fill(
            device,
            viewport_lib::plugin_api::OverlayFillDesc::new("hatch", HATCH_FILL_WGSL),
        )
        .expect("the hatch fill composes")
}

/// The colour of heatmap cell `(i, j)`: a smooth field through a blue to red
/// ramp.
fn heat(i: usize, j: usize) -> Colour {
    let t = ((i as f32 * 0.7).sin() * 0.5 + (j as f32 * 0.9).cos() * 0.5) * 0.5 + 0.5;
    Colour::srgb(
        0.1 + 0.85 * t,
        0.25 + 0.4 * (1.0 - (t - 0.5).abs() * 2.0),
        0.9 - 0.8 * t,
        1.0,
    )
}

/// Meshes: a heatmap as one mesh beside the same cells as crisp-edged shapes
/// (the two match wherever cells abut, with no seam between shape cells), a textured mesh over UVs, a rect with a
/// registered hatching fill, and a mesh compiled into a retained group.
fn build_mesh(ctx: &mut BuildCtx<'_>) -> BuiltScene {
    const COLS: usize = 8;
    const ROWS: usize = 5;
    const CELL: f32 = 14.0;
    let mut ovl = OverlayFrame::default();

    // One mesh, four vertices per cell so each cell is one flat colour.
    let mut positions = Vec::new();
    let mut colours = Vec::new();
    let mut indices = Vec::new();
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
            colours.extend_from_slice(&[heat(i, j); 4]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    ovl.meshes.push(
        OverlayMeshItem::new(positions, indices)
            .with_colours(colours)
            .with_position([20.0, 20.0]),
    );
    for j in 0..ROWS {
        for i in 0..COLS {
            ovl.shapes.push(
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [150.0 + i as f32 * CELL, 20.0 + j as f32 * CELL],
                    [CELL, CELL],
                )
                .with_style(
                    viewport_lib::OverlayStyle::solid(heat(i, j))
                        .with_edge(viewport_lib::OverlayEdge::Crisp),
                ),
            );
        }
    }

    // A hexagon fan with a texture fill sampled at its vertex UVs.
    let texture = ctx
        .renderer
        .resources_mut()
        .upload_overlay_texture(
            ctx.device,
            ctx.queue,
            viewport_lib::TextureData::srgb(
                2,
                2,
                [
                    [240u8, 90, 60, 255],
                    [250, 220, 80, 255],
                    [70, 140, 240, 255],
                    [90, 210, 120, 255],
                ]
                .concat(),
            ),
        )
        .expect("a well-formed sRGB image");
    let mut hex = vec![[40.0f32, 40.0]];
    let mut uvs = vec![[0.5f32, 0.5]];
    for k in 0..6 {
        let a = k as f32 * std::f32::consts::TAU / 6.0;
        hex.push([40.0 + 38.0 * a.cos(), 40.0 + 38.0 * a.sin()]);
        uvs.push([0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
    }
    let fan: Vec<u32> = (0..6).flat_map(|k| [0, 1 + k, 1 + (k + 1) % 6]).collect();
    ovl.meshes.push(
        OverlayMeshItem::new(hex, fan)
            .with_uvs(uvs)
            .with_fill(OverlayFill::texture(texture))
            .with_position([20.0, 110.0]),
    );

    // A rect filled by the registered hatching function.
    let hatch = hatch_fill(ctx.renderer, ctx.device);
    let mut params = [0.0f32; 16];
    params[0] = 9.0;
    params[1] = 3.0;
    params[4..8].copy_from_slice(&[0.95, 0.95, 1.0, 1.0]);
    params[8..12].copy_from_slice(&[0.15, 0.2, 0.45, 0.85]);
    ovl.shapes.push(
        OverlayShapeItem::new(
            OverlayShape::Rect {
                corner_radius: 10.0,
            },
            [120.0, 112.0],
            [100.0, 70.0],
        )
        .with_fill(OverlayFill::custom(hatch, params)),
    );

    // A per-vertex coloured strip, compiled into a retained group.
    let strip = OverlayMeshItem::new(
        vec![
            [0.0, 0.0],
            [60.0, 0.0],
            [0.0, 60.0],
            [60.0, 60.0],
            [0.0, 30.0],
            [60.0, 30.0],
        ],
        vec![0, 1, 4, 1, 5, 4, 4, 5, 2, 5, 3, 2],
    )
    .with_colours(vec![
        Colour::srgb(1.0, 0.3, 0.3, 1.0),
        Colour::srgb(1.0, 0.8, 0.2, 1.0),
        Colour::srgb(0.3, 0.4, 1.0, 1.0),
        Colour::srgb(0.6, 0.2, 0.9, 1.0),
        Colour::srgb(0.3, 0.9, 0.4, 1.0),
        Colour::srgb(0.2, 0.9, 0.9, 1.0),
    ]);
    let id = ctx.renderer.compile_overlay_group(
        ctx.device,
        ctx.queue,
        &OverlayGroupItems::default().with_meshes(vec![strip]),
        1.0,
    );
    ovl.retained
        .push(RetainedOverlay::new(id).with_position([250.0, 115.0]));

    backdrop(ctx, ovl)
}
