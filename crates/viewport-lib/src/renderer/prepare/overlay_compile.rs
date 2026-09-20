//! Compile overlay items into a retained [`OverlayGeometryId`] once, so the
//! renderer re-draws them from a cached buffer each frame instead of
//! re-tessellating them.

use super::*;
use crate::resources::overlay::font::GlyphStyle;

/// Emit a group's polyline and vector-shape fills into `verts` (local logical
/// pixels). These are viewport- and DPI-independent, so they are emitted once and
/// never need re-emission.
fn emit_base(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    polylines: &[crate::renderer::types::OverlayPolylineItem],
    vector_shapes: &[crate::renderer::types::OverlayShapeItem],
) {
    for poly in polylines {
        if poly.points.len() < 2 || poly.opacity <= 0.0 {
            continue;
        }
        let item_start = verts.len();
        for layer in poly
            .style
            .shadows
            .iter()
            .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
            .filter(|l| l.is_visible())
        {
            overlay_geometry::emit_polyline_shadow(verts, poly, layer, poly.opacity, 0.0, 0.0);
        }
        let content_start = verts.len();
        let filled = poly.closed && poly.style.fill.is_set();
        if filled && poly.style.fill.texture_id().is_none() {
            overlay_geometry::emit_filled_polyline(
                verts,
                &poly.points,
                &poly.style.fill,
                poly.opacity,
                0.0,
                0.0,
            );
        }
        viewport_overlays::tint_vertices_from(verts, content_start, poly.tint);
        // An inset layer goes over what it erodes and under the edge of it:
        // over the fill and under the stroke for a filled path, over the stroke
        // when the stroke is all the item covers.
        let mut inner = |verts: &mut Vec<crate::resources::OverlayTextVertex>| {
            for layer in poly
                .style
                .inner_shadows
                .iter()
                .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
                .filter(|l| l.is_visible())
            {
                overlay_geometry::emit_polyline_inner_shadow(
                    verts,
                    poly,
                    layer,
                    poly.opacity,
                    0.0,
                    0.0,
                );
            }
        };
        if filled {
            inner(verts);
        }
        let stroke_start = verts.len();
        if let Some(stroke) = poly.stroke.as_ref().filter(|s| s.width > 0.0) {
            let mut colour = stroke.colour.to_linear_rgba();
            colour[3] *= poly.opacity;
            overlay_geometry::emit_polyline_stroke(verts, poly, stroke, colour, 0.0, 0.0);
        }
        viewport_overlays::tint_vertices_from(verts, stroke_start, poly.tint);
        if !filled {
            inner(verts);
        }
        // The item transform is baked into the compiled geometry; the group
        // transform rides the instance each frame. A compiled group ignores
        // `anchor`, so `translate` is a plain offset in group-local pixels.
        if let Some((pmin, pmax)) = super::projection::polyline_bounds(&poly.points) {
            let rot = overlay_geometry::OverlayRotation::new(
                poly.transform.rotation,
                poly.transform.scale,
                overlay_geometry::OverlayRotation::pivot_point(
                    pmin,
                    [pmax[0] - pmin[0], pmax[1] - pmin[1]],
                    poly.transform.pivot,
                ),
            );
            overlay_geometry::rotate_vertices_from(verts, item_start, rot);
        }
        let t = poly.transform.translate;
        if t != [0.0, 0.0] {
            for v in &mut verts[item_start..] {
                v.position = [v.position[0] + t[0], v.position[1] + t[1]];
            }
        }
    }
    for shape in vector_shapes {
        if shape.opacity <= 0.0 {
            continue;
        }
        if let crate::renderer::types::OverlayShape::Vector {
            subpaths,
            fill_rule,
        } = &shape.shape
        {
            viewport_overlays::emit_vector_shape(verts, shape, subpaths, *fill_rule, 0.0, 0.0);
        }
    }
}

/// Emit one glyph run's quads into `verts` in local logical pixels (no anchor
/// resolution: a retained group is positioned per frame by its translate). May
/// rasterize glyphs into the atlas (and grow it).
fn emit_glyph_run(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    atlas: &mut crate::resources::overlay::font::GlyphAtlas,
    device: &crate::gpu::Device,
    run: &crate::renderer::types::GlyphRunItem,
    ppp: f32,
) {
    if run.glyphs.is_empty() || run.opacity <= 0.0 {
        return;
    }
    let Some(([min_x, min_y], [ext_w, ext_h])) = run.extent() else {
        return;
    };
    let run_x = run.transform.translate[0] + run.align_x.align_shift(ext_w);
    let run_y = run.transform.translate[1] + run.align_y.align_shift(ext_h);
    let opacity = run.opacity.clamp(0.0, 1.0);
    let rot_start = verts.len();
    let rot = overlay_geometry::OverlayRotation::new(
        run.transform.rotation,
        run.transform.scale,
        overlay_geometry::OverlayRotation::pivot_point(
            [run_x + min_x, run_y + min_y],
            [ext_w, ext_h],
            run.transform.pivot,
        ),
    );
    let quads = atlas.layout_glyph_run(
        run.glyphs.iter().enumerate().map(|(i, g)| {
            let colour = run
                .colours
                .get(i)
                .copied()
                .unwrap_or(run.colour)
                .to_linear_rgba();
            (
                g.glyph_id,
                g.x,
                g.y,
                overlay_geometry::apply_opacity(colour, opacity),
            )
        }),
        run.font_size,
        run.font,
        ppp,
        device,
        GlyphStyle::PLAIN,
    );
    // Shadow layers bake into the retained buffer like everything else: they are
    // fixed geometry once compiled, so a retained run carries its contour without
    // re-laying it out per frame.
    for layer in run
        .style
        .shadows
        .iter()
        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
        .filter(|l| l.is_visible())
    {
        let style = GlyphStyle::from_shadow(
            layer.spread * ppp,
            layer.blur * ppp,
            layer.falloff,
            [layer.offset[0] * ppp, layer.offset[1] * ppp],
        );
        let col = overlay_geometry::apply_opacity(layer.colour.to_linear_rgba(), opacity);
        let sq = atlas.layout_glyph_run(
            run.glyphs.iter().map(|g| (g.glyph_id, g.x, g.y, col)),
            run.font_size,
            run.font,
            ppp,
            device,
            style,
        );
        overlay_geometry::emit_glyph_quads_colored(
            verts,
            &sq,
            run_x + layer.offset[0],
            run_y + layer.offset[1],
            0.0,
            0.0,
        );
    }
    let glyph_start = verts.len();
    overlay_geometry::emit_glyph_quads_colored(verts, &quads, run_x, run_y, 0.0, 0.0);
    if run.style.fill.is_set() {
        viewport_overlays::fill_vertices_from(
            verts,
            glyph_start,
            &run.style.fill,
            [run_x + min_x, run_y + min_y],
            [ext_w, ext_h],
        );
    }
    viewport_overlays::tint_vertices_from(verts, glyph_start, run.tint);
    for layer in run
        .style
        .inner_shadows
        .iter()
        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
        .filter(|l| l.is_visible())
    {
        let style =
            GlyphStyle::from_inner_shadow(layer.spread * ppp, layer.blur * ppp, layer.falloff);
        if style == GlyphStyle::PLAIN {
            continue;
        }
        let col = overlay_geometry::apply_opacity(layer.colour.to_linear_rgba(), opacity);
        let sq = atlas.layout_glyph_run(
            run.glyphs.iter().map(|g| (g.glyph_id, g.x, g.y, col)),
            run.font_size,
            run.font,
            ppp,
            device,
            style,
        );
        overlay_geometry::emit_glyph_quads_colored(verts, &sq, run_x, run_y, 0.0, 0.0);
    }
    overlay_geometry::rotate_vertices_from(verts, rot_start, rot);
}

/// Emit one label's text-stream geometry (background box, leader line, glyph
/// quads) into `verts`, laid out as if its anchor origin were `[0, 0]`. This
/// mirrors the immediate label draw (`prepare_overlay_labels`) with the anchor
/// held at the local origin: the renderer resolves the real anchor per frame and
/// folds it into the group's translate. A world-anchor leader line runs from the
/// projected point to the text, and that point coincides with the resolved
/// origin, so the leader is a constant segment from `[0, 0]` here and rides the
/// same translate. May rasterize glyphs into the atlas (and grow it).
///
/// `emit_leader` gates the world-anchor leader line: it is drawn for a
/// self-anchoring label (`compile_overlay_label`, whose anchor is resolved per
/// frame so the leader points at the projected anchor), and suppressed for a
/// fixed-local label mixed into a geometry group (`compile_overlay_geometry`,
/// whose anchor is ignored, so a leader would point at nothing).
fn emit_label(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    atlas: &mut crate::resources::overlay::font::GlyphAtlas,
    device: &crate::gpu::Device,
    label: &crate::renderer::types::LabelItem,
    emit_leader: bool,
    ppp: f32,
) {
    use crate::renderer::types::{AnchorX, AnchorY, OverlayAnchor};
    if label.text.is_empty() || label.opacity <= 0.0 {
        return;
    }
    let opacity = label.opacity.clamp(0.0, 1.0);
    let layout = if let Some(max_w) = label.max_width {
        atlas.layout_text_wrapped(
            &label.text,
            label.font_size,
            label.font,
            max_w,
            ppp,
            device,
            GlyphStyle::PLAIN,
        )
    } else {
        atlas.layout_text(
            &label.text,
            label.font_size,
            label.font,
            ppp,
            device,
            GlyphStyle::PLAIN,
        )
    };
    let font_index = label.font.map_or(0, |h| h.0);
    let ascent = atlas.font_ascent(font_index, label.font_size);

    // Alignment folds in anchor_padding on X (Left pushes right, Right pulls
    // left, Middle unaffected); position nudges last. The anchor origin is [0, 0].
    let align_offset = match label.align_x {
        AnchorX::Left => label.anchor_padding,
        AnchorX::Middle => -layout.total_width * 0.5,
        AnchorX::Right => -layout.total_width - label.anchor_padding,
    };
    let align_offset_y = match label.align_y {
        AnchorY::Top => 0.0,
        AnchorY::Middle => -layout.height * 0.5,
        AnchorY::Bottom => -layout.height,
    };
    let text_x = align_offset + label.transform.translate[0];
    let text_y = align_offset_y + label.transform.translate[1];

    let rot = overlay_geometry::OverlayRotation::new(
        label.transform.rotation,
        label.transform.scale,
        overlay_geometry::OverlayRotation::pivot_point(
            [text_x, text_y],
            [layout.total_width, layout.height],
            label.transform.pivot,
        ),
    );
    let bg_start = verts.len();

    if label.background {
        let pad = label.padding;
        let (bx0, by0) = (text_x - pad, text_y - pad);
        let (bx1, by1) = (
            text_x + layout.total_width + pad,
            text_y + layout.height + pad,
        );
        let bg = overlay_geometry::apply_opacity(label.background_colour.to_linear_rgba(), opacity);
        if label.border_radius > 0.0 {
            overlay_geometry::emit_rounded_quad(
                verts,
                bx0,
                by0,
                bx1,
                by1,
                label.border_radius,
                bg,
                0.0,
                0.0,
            );
        } else {
            overlay_geometry::emit_solid_quad(verts, bx0, by0, bx1, by1, bg, 0.0, 0.0);
        }
    }

    viewport_overlays::tint_vertices_from(verts, bg_start, label.tint);
    overlay_geometry::rotate_vertices_from(verts, bg_start, rot);

    if emit_leader && label.leader_line && matches!(label.anchor, OverlayAnchor::World(_)) {
        overlay_geometry::emit_line_quad(
            verts,
            0.0,
            0.0,
            text_x,
            text_y + layout.height * 0.5,
            1.5,
            overlay_geometry::apply_opacity(label.leader_colour.to_linear_rgba(), opacity),
            0.0,
            0.0,
        );
    }

    let text_start = verts.len();

    for layer in label
        .style
        .shadows
        .iter()
        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
        .filter(|l| l.is_visible())
    {
        let style = GlyphStyle::from_shadow(
            layer.spread * ppp,
            layer.blur * ppp,
            layer.falloff,
            [layer.offset[0] * ppp, layer.offset[1] * ppp],
        );
        let sl = if let Some(max_w) = label.max_width {
            atlas.layout_text_wrapped(
                &label.text,
                label.font_size,
                label.font,
                max_w,
                ppp,
                device,
                style,
            )
        } else {
            atlas.layout_text(&label.text, label.font_size, label.font, ppp, device, style)
        };
        let col = overlay_geometry::apply_opacity(layer.colour.to_linear_rgba(), opacity);
        overlay_geometry::emit_glyph_quads(
            verts,
            &sl.quads,
            text_x + layer.offset[0],
            text_y + ascent + layer.offset[1],
            col,
            0.0,
            0.0,
        );
    }

    let text_colour = overlay_geometry::apply_opacity(label.colour.to_linear_rgba(), opacity);
    // The label origin is the text-box top-left; add the ascent to reach the
    // first baseline the quads are relative to.
    let glyph_start = verts.len();
    overlay_geometry::emit_glyph_quads(
        verts,
        &layout.quads,
        text_x,
        text_y + ascent,
        text_colour,
        0.0,
        0.0,
    );
    if label.style.fill.is_set() {
        viewport_overlays::fill_vertices_from(
            verts,
            glyph_start,
            &label.style.fill,
            [text_x, text_y],
            [layout.total_width, layout.height],
        );
    }
    viewport_overlays::tint_vertices_from(verts, glyph_start, label.tint);
    for layer in label
        .style
        .inner_shadows
        .iter()
        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
        .filter(|l| l.is_visible())
    {
        let style =
            GlyphStyle::from_inner_shadow(layer.spread * ppp, layer.blur * ppp, layer.falloff);
        if style == GlyphStyle::PLAIN {
            continue;
        }
        let sl = if let Some(max_w) = label.max_width {
            atlas.layout_text_wrapped(
                &label.text,
                label.font_size,
                label.font,
                max_w,
                ppp,
                device,
                style,
            )
        } else {
            atlas.layout_text(&label.text, label.font_size, label.font, ppp, device, style)
        };
        let col = overlay_geometry::apply_opacity(layer.colour.to_linear_rgba(), opacity);
        overlay_geometry::emit_glyph_quads(
            verts,
            &sl.quads,
            text_x,
            text_y + ascent,
            col,
            0.0,
            0.0,
        );
    }
    overlay_geometry::rotate_vertices_from(verts, text_start, rot);
}

/// Emit a whole group (polylines, vector shapes, glyph runs, labels) into a fresh
/// vertex list, returning it with the atlas version it baked glyph UVs against.
///
/// Glyph UVs divide by the atlas size, so if rasterizing this group's glyphs grows
/// the atlas, glyphs emitted before the grow carry a stale divisor. The loop
/// re-emits the glyph portion once the atlas has stopped growing (all this group's
/// glyphs are then resident), so the returned geometry always uses a single,
/// final atlas size. The base (polyline/vector) portion is emitted once.
///
/// `label_leaders` is passed through to each label's `emit_label`: true for a
/// self-anchoring label group, false for fixed-local labels in a mixed group.
pub(super) fn emit_group_verts(
    atlas: &mut crate::resources::overlay::font::GlyphAtlas,
    device: &crate::gpu::Device,
    polylines: &[crate::renderer::types::OverlayPolylineItem],
    vector_shapes: &[crate::renderer::types::OverlayShapeItem],
    glyph_runs: &[crate::renderer::types::GlyphRunItem],
    labels: &[crate::renderer::types::LabelItem],
    label_leaders: bool,
    ppp: f32,
) -> (Vec<crate::resources::OverlayTextVertex>, u64) {
    let mut base = Vec::new();
    emit_base(&mut base, polylines, vector_shapes);

    // Fast path: no glyph-bearing content (runs or labels), nothing grows the atlas.
    let has_glyphs = glyph_runs
        .iter()
        .any(|r| !r.glyphs.is_empty() && r.opacity > 0.0)
        || labels.iter().any(|l| !l.text.is_empty() && l.opacity > 0.0);
    if !has_glyphs {
        return (base, atlas.version());
    }

    // A doubling atlas stabilises in a few iterations; the guard bounds it.
    let mut guard = 0;
    loop {
        let v0 = atlas.version();
        let mut verts = base.clone();
        for run in glyph_runs {
            emit_glyph_run(&mut verts, atlas, device, run, ppp);
        }
        for label in labels {
            emit_label(&mut verts, atlas, device, label, label_leaders, ppp);
        }
        guard += 1;
        if atlas.version() == v0 || guard >= 8 {
            let v1 = atlas.version();
            return (verts, v1);
        }
    }
}

/// Emit one analytic SDF shape (Rect, RoundedRect, Circle, Ring, ...) as solid
/// `OverlayShapeVertex` geometry plus its stacked shadow layers, for the retained
/// path. Mirrors the solid branch of the immediate shape emission
/// (`prepare_viewport_internal`) for the static case: no animation sampling, no
/// anchor resolution, and no internal clip mask (a retained group's outer clip is
/// carried by its text stream). Textured and backdrop-blur shapes use a separate
/// pipeline and are skipped.
fn emit_sdf_shape(
    shape: &crate::renderer::types::OverlayShapeItem,
    out_verts: &mut Vec<crate::resources::OverlayShapeVertex>,
    out_shadows: &mut Vec<crate::resources::OverlayShadowLayerGpu>,
) {
    use crate::renderer::types::{LineCap, OverlayFill, OverlayShape, TriangleDirection};
    if matches!(shape.shape, OverlayShape::Vector { .. })
        || shape.clip_mask_id.is_some()
        || shape.style.fill.texture_id().is_some()
        || shape.style.backdrop.blur > 0.0
        || shape.opacity <= 0.0
    {
        return;
    }
    let op = shape.opacity;
    let hw = shape.size[0] * 0.5;
    let hh = shape.size[1] * 0.5;
    let cx = shape.transform.translate[0] + hw;
    let cy = shape.transform.translate[1] + hh;

    let mut shadow_pad = 0.0f32;
    for l in &shape.style.shadows {
        shadow_pad = shadow_pad.max(l.extent());
    }
    let extra_expand = match &shape.shape {
        OverlayShape::Line { thickness, .. } => thickness * 0.5,
        _ => 0.0,
    };
    let bx = hw + extra_expand;
    let by = hh + extra_expand;
    let (rx, ry) = if shape.transform.rotation != 0.0 {
        let c = shape.transform.rotation.cos();
        let s = shape.transform.rotation.sin();
        let piv = shape.transform.pivot;
        let (mut mx, mut my) = (0.0f32, 0.0f32);
        for cxp in [-bx, bx] {
            for cyp in [-by, by] {
                let dx = cxp - piv[0];
                let dy = cyp - piv[1];
                mx = mx.max((c * dx - s * dy + piv[0]).abs());
                my = my.max((s * dx + c * dy + piv[1]).abs());
            }
        }
        (mx, my)
    } else {
        (bx, by)
    };
    let ex = rx + shadow_pad + 1.0;
    let ey = ry + shadow_pad + 1.0;

    let (shape_type, radii) = match &shape.shape {
        OverlayShape::Rect { corner_radius } => {
            let r = corner_radius.min(hw).min(hh).max(0.0);
            (0.0, [r, r, r, r])
        }
        OverlayShape::RoundedRect { radii: r } => (
            0.0,
            [
                r[1].min(hw).min(hh).max(0.0),
                r[2].min(hw).min(hh).max(0.0),
                r[3].min(hw).min(hh).max(0.0),
                r[0].min(hw).min(hh).max(0.0),
            ],
        ),
        OverlayShape::Circle => (1.0, [0.0; 4]),
        OverlayShape::Ellipse => (2.0, [0.0; 4]),
        OverlayShape::Capsule => (3.0, [0.0; 4]),
        OverlayShape::Ring { inner_radius_frac } => {
            (4.0, [inner_radius_frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0])
        }
        OverlayShape::Arc {
            inner_radius_frac,
            start_angle,
            end_angle,
        } => (
            5.0,
            [
                inner_radius_frac.clamp(0.0, 1.0),
                *start_angle,
                *end_angle,
                0.0,
            ],
        ),
        OverlayShape::Triangle { direction } => {
            let d = match direction {
                TriangleDirection::Up => 0.0,
                TriangleDirection::Down => 1.0,
                TriangleDirection::Left => 2.0,
                TriangleDirection::Right => 3.0,
            };
            (6.0, [d, 0.0, 0.0, 0.0])
        }
        OverlayShape::Line { thickness, cap } => {
            let cap_f = match cap {
                LineCap::Round => 0.0,
                LineCap::Square => 1.0,
            };
            (7.0, [thickness * 0.5, cap_f, 0.0, 0.0])
        }
        OverlayShape::Star {
            points,
            inner_radius_frac,
        } => (
            8.0,
            [
                (*points).max(3) as f32,
                inner_radius_frac.clamp(0.0, 1.0),
                0.0,
                0.0,
            ],
        ),
        OverlayShape::RegularPolygon { sides } => (9.0, [(*sides).max(3) as f32, 0.0, 0.0, 0.0]),
        OverlayShape::Cross { arm_width_frac } => {
            (10.0, [arm_width_frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0])
        }
        _ => (0.0, [0.0; 4]),
    };

    let mut stop_colours = [[0.0f32; 4]; 4];
    let mut stop_positions = [0.0f32, 1.0, 1.0, 1.0];
    let stop_count: f32;
    let gradient_params = match &shape.style.fill {
        OverlayFill::Solid(c) => {
            stop_colours[0] = c.to_linear_rgba();
            stop_colours[1] = c.to_linear_rgba();
            stop_count = 0.0;
            [0.0f32, 0.0]
        }
        OverlayFill::LinearGradient {
            start_colour,
            end_colour,
            angle,
        } => {
            stop_colours[0] = start_colour.to_linear_rgba();
            stop_colours[1] = end_colour.to_linear_rgba();
            stop_count = 2.0;
            [1.0f32, *angle]
        }
        OverlayFill::RadialGradient {
            centre_colour,
            edge_colour,
        } => {
            stop_colours[0] = centre_colour.to_linear_rgba();
            stop_colours[1] = edge_colour.to_linear_rgba();
            stop_count = 2.0;
            [2.0f32, 0.0]
        }
        OverlayFill::ConicalGradient {
            start_colour,
            end_colour,
            offset_angle,
        } => {
            stop_colours[0] = start_colour.to_linear_rgba();
            stop_colours[1] = end_colour.to_linear_rgba();
            stop_count = 2.0;
            [3.0f32, *offset_angle]
        }
        OverlayFill::LinearGradientMulti { stops, angle } => {
            stop_count =
                overlay_geometry::pack_stops(stops, &mut stop_colours, &mut stop_positions);
            [1.0f32, *angle]
        }
        OverlayFill::RadialGradientMulti { stops } => {
            stop_count =
                overlay_geometry::pack_stops(stops, &mut stop_colours, &mut stop_positions);
            [2.0f32, 0.0]
        }
        OverlayFill::ConicalGradientMulti {
            stops,
            offset_angle,
        } => {
            stop_count =
                overlay_geometry::pack_stops(stops, &mut stop_colours, &mut stop_positions);
            [3.0f32, *offset_angle]
        }
        // A texture fill has no gradient: its tint is the colour the sample is
        // multiplied by.
        OverlayFill::Texture { tint, .. } => {
            stop_colours[0] = tint.to_linear_rgba();
            stop_colours[1] = stop_colours[0];
            stop_count = 0.0;
            [0.0f32, 0.0]
        }
        _ => {
            stop_count = 0.0;
            [0.0f32, 0.0]
        }
    };
    for colour in &mut stop_colours {
        colour[3] *= op;
    }
    for colour in &mut stop_colours {
        for (c, t) in colour.iter_mut().zip(shape.tint) {
            *c *= t;
        }
    }
    let fc = stop_colours[0];
    let fc2 = stop_colours[1];
    let gp4 = [gradient_params[0], gradient_params[1], stop_count, 0.0];

    let base_index = out_shadows.len();
    let (mut outer_count, mut inner_count) = (0usize, 0usize);
    let max_layers = crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS;
    if !shape.style.shadows.is_empty() {
        for l in shape.style.shadows.iter().take(max_layers) {
            let mut col = l.colour.to_linear_rgba();
            col[3] *= op;
            out_shadows.push(crate::resources::OverlayShadowLayerGpu {
                colour: col,
                params: [l.blur, l.offset[0], l.offset[1], 0.0],
                params2: [l.spread, l.falloff, 0.0, 0.0],
            });
            outer_count += 1;
        }
    }
    if !shape.style.inner_shadows.is_empty() {
        for l in shape.style.inner_shadows.iter().take(max_layers) {
            let mut col = l.colour.to_linear_rgba();
            col[3] *= op;
            out_shadows.push(crate::resources::OverlayShadowLayerGpu {
                colour: col,
                params: [l.blur, l.offset[0], l.offset[1], 1.0],
                params2: [l.spread, l.falloff, 0.0, 0.0],
            });
            inner_count += 1;
        }
    }
    let shadow_index = [base_index as f32, outer_count as f32, inner_count as f32];
    let rotation_pivot = [
        shape.transform.rotation,
        shape.transform.pivot[0],
        shape.transform.pivot[1],
        0.0,
    ];
    let half_size = [hw, hh];
    // Scale the quad about the pivot and leave `local_pos` in the shape's own
    // frame, the same split the shader uses for a group transform.
    let item_scale = shape.transform.scale;
    let pivot_px = [cx + shape.transform.pivot[0], cy + shape.transform.pivot[1]];
    let corner = |x: f32, y: f32, lx: f32, ly: f32| {
        if item_scale == 1.0 {
            (x, y, lx, ly)
        } else {
            (
                pivot_px[0] + (x - pivot_px[0]) * item_scale,
                pivot_px[1] + (y - pivot_px[1]) * item_scale,
                lx,
                ly,
            )
        }
    };
    let corners = [
        corner(cx - ex, cy - ey, -ex, -ey),
        corner(cx + ex, cy - ey, ex, -ey),
        corner(cx + ex, cy + ey, ex, ey),
        corner(cx - ex, cy - ey, -ex, -ey),
        corner(cx + ex, cy + ey, ex, ey),
        corner(cx - ex, cy + ey, -ex, ey),
    ];
    for (px, py, lx, ly) in corners {
        out_verts.push(crate::resources::OverlayShapeVertex {
            position: [px, py],
            local_pos: [lx, ly],
            fill_colour: fc,
            half_size,
            radii,
            shape_type,
            fill_colour2: fc2,
            gradient_params: gp4,
            shadow_index,
            rotation_pivot,
            clip_rect: [0.0; 4],
            clip_index: -1.0,
            stop_colour_c: stop_colours[2],
            stop_colour_d: stop_colours[3],
            stop_positions,
        });
    }
}

/// The extent of a compiled group in its own local logical pixels, over both
/// vertex streams. `None` when the group compiled nothing.
///
/// This is the box `align_x` / `align_y` shift against when a group is
/// anchored, so it plays the role an item's own extent box plays. Computed once
/// at compile rather than per frame: the geometry is fixed by definition.
fn group_bounds(
    text: &[crate::resources::OverlayTextVertex],
    shapes: &[crate::resources::OverlayShapeVertex],
) -> Option<([f32; 2], [f32; 2])> {
    let mut min = [f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN];
    let mut any = false;
    for p in text
        .iter()
        .map(|v| v.position)
        .chain(shapes.iter().map(|v| v.position))
    {
        any = true;
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
    }
    any.then_some((min, max))
}

impl ViewportRenderer {
    /// Compile a group of polylines, vector shapes, glyph runs, and labels into a
    /// retained overlay-geometry handle.
    ///
    /// The items are tessellated once (polyline fills and strokes, vector-path
    /// fills, SDF shapes, glyph quads, and each label's laid-out text,
    /// background box, and glyph quads) into local logical-pixel geometry and
    /// uploaded to a buffer that lives until the group is freed. Each frame, submit
    /// the returned id through `OverlayFrame::retained` as a [`RetainedOverlay`]
    /// carrying a per-frame translate, opacity, z-order, and clip, instead of
    /// pushing the items into `OverlayFrame`. Release it with
    /// [`free_overlay_geometry`](Self::free_overlay_geometry).
    ///
    /// `pixels_per_point` must match the frame's; glyphs rasterize at
    /// `font_size * pixels_per_point`. A group that carries glyphs (in a run or a
    /// label) is re-emitted automatically when the atlas grows or the frame's
    /// `pixels_per_point` changes, so callers only re-compile when their own text,
    /// style, or width changes.
    ///
    /// All slices are optional (pass `&[]`). `vector_shapes` entries that are not
    /// `OverlayShape::Vector` are drawn as analytic SDF shapes. Geometry is taken in
    /// its own screen-pixel space; position the group each frame with
    /// `RetainedOverlay::translate`. Anchors are not resolved: every item, labels
    /// included, is fixed-local geometry placed by its own `position` / alignment
    /// and moved by the group's translate. A label's `anchor` and `leader_line` are
    /// therefore ignored here (they only make sense for a single anchor-tracking
    /// label); use [`compile_overlay_label`](Self::compile_overlay_label) for a
    /// label that tracks a viewport corner or a world point.
    ///
    /// **Draw order inside a group is submission order, not `z_order`.** The
    /// families use different vertex streams that have to stay batched, so a
    /// compiled group draws polylines and vector shapes first, then glyph runs,
    /// then labels, each in the order given. An item's `z_order` is ignored
    /// here; it still orders the whole group against other overlay content
    /// through `RetainedOverlay::z_order`. To control order within a group,
    /// submit the items in the order you want, or compile several groups.
    pub fn compile_overlay_geometry(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        polylines: &[crate::renderer::types::OverlayPolylineItem],
        vector_shapes: &[crate::renderer::types::OverlayShapeItem],
        glyph_runs: &[crate::renderer::types::GlyphRunItem],
        labels: &[crate::renderer::types::LabelItem],
        pixels_per_point: f32,
    ) -> crate::renderer::OverlayGeometryId {
        let has_glyphs = glyph_runs
            .iter()
            .any(|r| !r.glyphs.is_empty() && r.opacity > 0.0)
            || labels.iter().any(|l| !l.text.is_empty() && l.opacity > 0.0);

        let (verts, baked_version) = emit_group_verts(
            &mut self.resources.content.glyph_atlas,
            device,
            polylines,
            vector_shapes,
            glyph_runs,
            labels,
            // Fixed-local labels in a mixed group: anchor ignored, so no leader.
            false,
            pixels_per_point,
        );
        // Upload any newly rasterized glyphs so the atlas texture has them.
        self.resources.content.glyph_atlas.upload_if_dirty(queue);

        let bytes = std::mem::size_of_val(&verts[..]) as u64;
        let vertex_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("compiled_overlay_vbuf"),
            size: bytes.max(4),
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !verts.is_empty() {
            queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&verts));
        }

        // Analytic SDF shapes (the non-Vector entries of `vector_shapes`) go to the
        // shape-pipeline stream. Vector entries were already handled above.
        let mut shape_verts: Vec<crate::resources::OverlayShapeVertex> = Vec::new();
        let mut shadow_layers: Vec<crate::resources::OverlayShadowLayerGpu> = Vec::new();
        for shape in vector_shapes {
            if !matches!(
                shape.shape,
                crate::renderer::types::OverlayShape::Vector { .. }
            ) {
                emit_sdf_shape(shape, &mut shape_verts, &mut shadow_layers);
            }
        }
        let (shape_vertex_buf, shadow_buf, shape_bytes) = if shape_verts.is_empty() {
            (None, None, 0)
        } else {
            // The shape pipeline layout always expects a shadow buffer; provide a
            // dummy entry when no shape has shadows.
            if shadow_layers.is_empty() {
                shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                    colour: [0.0; 4],
                    params: [0.0; 4],
                    params2: [0.0, 1.0, 0.0, 0.0],
                });
            }
            let sv_bytes = std::mem::size_of_val(&shape_verts[..]) as u64;
            let sh_bytes = std::mem::size_of_val(&shadow_layers[..]) as u64;
            let sv = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("compiled_overlay_shape_vbuf"),
                size: sv_bytes,
                usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&sv, 0, bytemuck::cast_slice(&shape_verts));
            let sh = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("compiled_overlay_shadow_buf"),
                size: sh_bytes,
                usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&sh, 0, bytemuck::cast_slice(&shadow_layers));
            (Some(sv), Some(sh), sv_bytes + sh_bytes)
        };

        // Retain the source only when the group has glyphs (the only geometry that
        // can go stale). Polyline/vector/shape-only groups never re-emit.
        let source = has_glyphs.then(|| crate::resources::CompiledSource {
            polylines: polylines.to_vec(),
            vector_shapes: vector_shapes.to_vec(),
            glyph_runs: glyph_runs.to_vec(),
            labels: labels.to_vec(),
            baked_atlas_version: baked_version,
            baked_ppp: pixels_per_point,
        });

        let total_bytes = bytes + shape_bytes;
        self.resources.content.overlay_geometry.insert(
            crate::resources::CompiledOverlay {
                vertex_buf,
                vertex_count: verts.len() as u32,
                bytes: total_bytes,
                shape_vertex_buf,
                shape_vertex_count: shape_verts.len() as u32,
                shadow_buf,
                source,
                anchor: None,
                bounds: group_bounds(&verts, &shape_verts),
            },
            total_bytes,
        )
    }

    /// Compile a single [`LabelItem`](crate::renderer::types::LabelItem) into a
    /// retained overlay-geometry handle.
    ///
    /// The text is laid out once, the way the immediate label draw lays it out,
    /// into local logical-pixel geometry on the text stream: the background box,
    /// the leader line (for a world anchor), and the glyph quads. Each frame the
    /// renderer resolves the label's own anchor (a viewport corner, or a world
    /// point projected through the camera) to a screen origin and folds it into the
    /// group's translate, so the compiled label tracks its anchor across a resize
    /// or camera move without re-laying-out. A world-anchored label is skipped for
    /// any frame its point is behind the camera or off screen, like the immediate
    /// path.
    ///
    /// `pixels_per_point` must match the frame's; the group re-emits automatically
    /// when the atlas grows or `pixels_per_point` changes, like any glyph-bearing
    /// group. Submit the returned id through `OverlayFrame::retained`; a
    /// [`RetainedOverlay::translate`](crate::renderer::RetainedOverlay) composes on
    /// top of the resolved anchor (to scroll or nudge), and the per-frame opacity
    /// and outer `clip_rect` apply as for any retained group. The label's own
    /// `clip_id` mask is not applied to a retained label; use the group's
    /// `clip_rect`. Release it with
    /// [`free_overlay_geometry`](Self::free_overlay_geometry).
    pub fn compile_overlay_label(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        label: &crate::renderer::types::LabelItem,
        pixels_per_point: f32,
    ) -> crate::renderer::OverlayGeometryId {
        let (verts, baked_version) = emit_group_verts(
            &mut self.resources.content.glyph_atlas,
            device,
            &[],
            &[],
            &[],
            std::slice::from_ref(label),
            // A self-anchoring label: its anchor is resolved per frame, so its
            // world-anchor leader line is meaningful.
            true,
            pixels_per_point,
        );
        self.resources.content.glyph_atlas.upload_if_dirty(queue);

        let bytes = std::mem::size_of_val(&verts[..]) as u64;
        let vertex_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("compiled_overlay_label_vbuf"),
            size: bytes.max(4),
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !verts.is_empty() {
            queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&verts));
        }

        let source = crate::resources::CompiledSource {
            polylines: Vec::new(),
            vector_shapes: Vec::new(),
            glyph_runs: Vec::new(),
            labels: vec![label.clone()],
            baked_atlas_version: baked_version,
            baked_ppp: pixels_per_point,
        };

        self.resources.content.overlay_geometry.insert(
            crate::resources::CompiledOverlay {
                vertex_buf,
                vertex_count: verts.len() as u32,
                bytes,
                shape_vertex_buf: None,
                shape_vertex_count: 0,
                shadow_buf: None,
                source: Some(source),
                anchor: Some(label.anchor),
                bounds: group_bounds(&verts, &[]),
            },
            bytes,
        )
    }

    /// Free a compiled overlay group, dropping its GPU buffer. Returns `true` if a
    /// group was freed, `false` if the id was already freed or never valid. A
    /// `RetainedOverlay` referencing a freed id is skipped for the frame.
    pub fn free_overlay_geometry(&mut self, id: crate::renderer::OverlayGeometryId) -> bool {
        self.resources.content.overlay_geometry.remove(id).is_some()
    }

    /// Re-emit a retained group's geometry if its baked glyph UVs are stale (the
    /// atlas grew or `pixels_per_point` changed since it was compiled). Cheap when
    /// the group has no glyphs or nothing changed: a version/ppp compare and return.
    pub(super) fn reemit_overlay_geometry_if_stale(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::renderer::OverlayGeometryId,
        ppp: f32,
    ) {
        let current_version = self.resources.content.glyph_atlas.version();
        let stale = match self.resources.content.overlay_geometry.get(id) {
            Some(c) => match &c.source {
                Some(src) => src.baked_atlas_version != current_version || src.baked_ppp != ppp,
                None => false,
            },
            None => false,
        };
        if !stale {
            return;
        }
        // Clone the source so the atlas and store borrows do not overlap. A group
        // that resolves an anchor per frame (a single self-anchoring label) keeps
        // its leader line on re-emit; a fixed-local mixed group does not.
        let (src, label_leaders) = self
            .resources
            .content
            .overlay_geometry
            .get(id)
            .and_then(|c| c.source.clone().map(|s| (s, c.anchor.is_some())))
            .unwrap();
        let (verts, baked_version) = emit_group_verts(
            &mut self.resources.content.glyph_atlas,
            device,
            &src.polylines,
            &src.vector_shapes,
            &src.glyph_runs,
            &src.labels,
            label_leaders,
            ppp,
        );
        self.resources.content.glyph_atlas.upload_if_dirty(queue);

        let bytes = std::mem::size_of_val(&verts[..]) as u64;
        let vertex_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("compiled_overlay_vbuf"),
            size: bytes.max(4),
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !verts.is_empty() {
            queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&verts));
        }

        let bounds = group_bounds(&verts, &[]);
        if let Some(c) = self.resources.content.overlay_geometry.get_mut(id) {
            c.vertex_buf = vertex_buf;
            c.vertex_count = verts.len() as u32;
            c.bytes = bytes;
            // A re-emit re-lays out the glyphs, so the extent can move.
            if c.shape_vertex_count == 0 {
                c.bounds = bounds;
            }
            if let Some(s) = &mut c.source {
                s.baked_atlas_version = baked_version;
                s.baked_ppp = ppp;
            }
        }
    }
}
