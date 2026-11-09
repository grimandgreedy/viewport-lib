//! Compile overlay items into a retained [`OverlayGeometryId`] once, so the
//! renderer re-draws them from a cached buffer each frame instead of
//! re-tessellating them.

use super::*;
use crate::resources::overlay::glyph_style::GlyphStyle;

/// Bake an item's own `clip.rect` onto the vertices it just emitted, from
/// `start` to the end of `verts`.
///
/// The immediate path stamps the same box onto an item's vertices every frame.
/// A compiled item has no frame to be stamped in, so the box is resolved once
/// here instead, in framebuffer pixels, and rides the compiled vertices. It is
/// a screen box, so it stays where it was authored when the group translates,
/// which is the rule the field already follows for rotation.
fn bake_clip_rect(
    verts: &mut [crate::resources::OverlayTextVertex],
    clip: &crate::renderer::types::OverlayClip,
    start: usize,
    ppp: f32,
) {
    let Some(r) = clip.rect else {
        return;
    };
    let cr = super::viewport_overlays::clip_rect_to_framebuffer(r, ppp);
    for v in &mut verts[start..] {
        v.clip_rect = cr;
    }
}

/// `bake_clip_rect` for the shape stream.
fn bake_shape_clip_rect(
    verts: &mut [crate::resources::OverlayShapeInstance],
    clip: &crate::renderer::types::OverlayClip,
    start: usize,
    ppp: f32,
) {
    let Some(r) = clip.rect else {
        return;
    };
    let cr = super::viewport_overlays::clip_rect_to_framebuffer(r, ppp);
    for v in &mut verts[start..] {
        v.clip_rect = cr;
    }
}

/// Whether `clip` names a mask, which a compiled item cannot be clipped to.
///
/// Masks are registered per frame in screen space and a compiled group has no
/// frame to resolve one against. An item naming one is skipped rather than
/// drawn unclipped, so the mistake shows up as missing content instead of as
/// content escaping the region it was meant to stay inside.
fn names_unusable_mask(clip: &crate::renderer::types::OverlayClip, item: &str) -> bool {
    if clip.mask.is_none() {
        return false;
    }
    #[cfg(debug_assertions)]
    tracing::warn!(
        "overlay: a compiled {item} names a clip mask, which only resolves on the immediate \
         path, so it was skipped. Clip the retained group instead, or submit the item each \
         frame."
    );
    let _ = item;
    true
}

/// Emit a group's polylines into `verts` (local logical pixels), recording
/// their gradient fills in `fills`. These are viewport- and DPI-independent
/// (their fringes are sized in the vertex shader), so they are emitted once and
/// never need re-emission.
///
/// A compiled group draws with no image bound, so an image fill on a polyline
/// or a vector path is not drawn.
fn emit_base(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    polylines: &[crate::renderer::types::OverlayPolylineItem],
    ppp: f32,
    fills: &mut super::overlay_fill::FillTable,
) {
    for poly in polylines {
        if poly.points.len() < 2
            || poly.style.opacity <= 0.0
            || names_unusable_mask(&poly.clip, "polyline")
        {
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
            overlay_geometry::emit_polyline_shadow(
                verts,
                poly,
                layer,
                poly.style.opacity,
                0.0,
                0.0,
            );
        }
        let content_start = verts.len();
        let filled = poly.closed && poly.style.fill.is_set();
        if filled && poly.style.fill.texture_id().is_none() {
            overlay_geometry::emit_filled_polyline(
                verts,
                &poly.points,
                &poly.style.fill,
                poly.style.opacity,
                0.0,
                0.0,
                fills,
                None,
                [1, 1],
            );
        }
        viewport_overlays::tint_vertices_from(verts, content_start, poly.style.tint);
        // An inset layer goes over what it erodes and under the edge of it:
        // over the fill and under the stroke for a filled path, over the stroke
        // when the stroke is all the item covers.
        let inner = |verts: &mut Vec<crate::resources::OverlayTextVertex>| {
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
                    poly.style.opacity,
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
            colour[3] *= poly.style.opacity;
            overlay_geometry::emit_polyline_stroke(verts, poly, stroke, colour, 0.0, 0.0);
        }
        viewport_overlays::tint_vertices_from(verts, stroke_start, poly.style.tint);
        if !filled {
            inner(verts);
        }
        // The item transform is baked into the compiled geometry; the group
        // transform rides the instance each frame. A compiled group ignores
        // `anchor`, so `translate` is a plain offset in group-local pixels.
        if let Some((pmin, pmax)) = super::projection::polyline_bounds(&poly.points) {
            let rot = overlay_geometry::OverlayRotation::new(
                &poly.transform,
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
        bake_clip_rect(verts, &poly.clip, item_start, ppp);
    }
}

/// Emit a group's vector shapes into `verts`, after its polylines. With an
/// atlas, a shape small enough draws its fill from the coverage atlas (and the
/// group then re-emits with the atlas, as glyphs do); without one every fill is
/// tessellated.
fn emit_vector_shapes(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    vector_shapes: &[crate::renderer::types::OverlayShapeItem],
    ppp: f32,
    fills: &mut super::overlay_fill::FillTable,
    mut atlas: Option<(
        &mut crate::resources::overlay::coverage_atlas::CoverageAtlas,
        &crate::gpu::Device,
    )>,
) {
    for shape in vector_shapes {
        if shape.style.opacity <= 0.0 || names_unusable_mask(&shape.clip, "vector shape") {
            continue;
        }
        if let crate::renderer::types::OverlayShape::Vector {
            subpaths,
            fill_rule,
        } = &shape.shape
        {
            let item_start = verts.len();
            let va = atlas
                .as_mut()
                .map(|(atlas, device)| viewport_overlays::VectorAtlas {
                    atlas,
                    device,
                    ppp,
                    max_px: viewport_overlays::VECTOR_ATLAS_MAX_RETAINED_PX,
                });
            viewport_overlays::emit_vector_shape(
                verts, shape, subpaths, *fill_rule, 0.0, 0.0, fills, None, va,
            );
            bake_clip_rect(verts, &shape.clip, item_start, ppp);
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
    fills: &mut super::overlay_fill::FillTable,
) {
    if run.glyphs.is_empty()
        || run.style.opacity <= 0.0
        || names_unusable_mask(&run.clip, "glyph run")
    {
        return;
    }
    let Some(([min_x, min_y], [ext_w, ext_h])) = run.extent() else {
        return;
    };
    let item_start = verts.len();
    let run_x = run.transform.translate[0] + run.anchoring.align.x.align_shift(ext_w);
    let run_y = run.transform.translate[1] + run.anchoring.align.y.align_shift(ext_h, Some(min_y));
    let opacity = run.style.opacity.clamp(0.0, 1.0);
    let rot_start = verts.len();
    let rot = overlay_geometry::OverlayRotation::new(
        &run.transform,
        overlay_geometry::OverlayRotation::pivot_point(
            [run_x + min_x, run_y + min_y],
            [ext_w, ext_h],
            run.transform.pivot,
        ),
    );
    let quads = atlas.layout_glyph_run(
        run.glyphs.iter().enumerate().map(|(i, g)| {
            // White where the run has no per-glyph multiplier: the fill goes on
            // afterwards and multiplies, so white is "the fill unmodified".
            let tint = run
                .glyph_tints
                .get(i)
                .copied()
                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
            (
                g.glyph_id,
                g.x,
                g.y,
                overlay_geometry::apply_opacity(tint, opacity),
            )
        }),
        run.text_style.size,
        run.text_style.font,
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
            run.text_style.size,
            run.text_style.font,
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
            fills,
            [1, 1],
        );
    }
    viewport_overlays::tint_vertices_from(verts, glyph_start, run.style.tint);
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
            run.text_style.size,
            run.text_style.font,
            ppp,
            device,
            style,
        );
        overlay_geometry::emit_glyph_quads_colored(verts, &sq, run_x, run_y, 0.0, 0.0);
    }
    overlay_geometry::rotate_vertices_from(verts, rot_start, rot);
    bake_clip_rect(verts, &run.clip, item_start, ppp);
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
    _emit_leader: bool,
    ppp: f32,
    fills: &mut super::overlay_fill::FillTable,
) {
    if label.text.is_empty()
        || label.style.opacity <= 0.0
        || names_unusable_mask(&label.clip, "label")
    {
        return;
    }
    let item_start = verts.len();
    let opacity = label.style.opacity.clamp(0.0, 1.0);
    let layout = if let Some(max_w) = label.max_width {
        atlas.layout_text_wrapped(
            &label.text,
            label.text_style.size,
            label.text_style.font,
            max_w,
            ppp,
            device,
            GlyphStyle::PLAIN,
        )
    } else {
        atlas.layout_text(
            &label.text,
            label.text_style.size,
            label.text_style.font,
            ppp,
            device,
            GlyphStyle::PLAIN,
        )
    };
    let font_index = label.text_style.font.map_or(0, |h| h.0);
    let ascent = atlas.font_ascent(font_index, label.text_style.size);

    // Alignment places the laid-out box on the origin, which is [0, 0] here;
    // position nudges last.
    let align_offset = label.anchoring.align.x.align_shift(layout.total_width);
    let align_offset_y = label
        .anchoring
        .align
        .y
        .align_shift(layout.height, Some(ascent));
    let text_x = align_offset + label.transform.translate[0];
    let text_y = align_offset_y + label.transform.translate[1];

    let rot = overlay_geometry::OverlayRotation::new(
        &label.transform,
        overlay_geometry::OverlayRotation::pivot_point(
            [text_x, text_y],
            [layout.total_width, layout.height],
            label.transform.pivot,
        ),
    );

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
                label.text_style.size,
                label.text_style.font,
                max_w,
                ppp,
                device,
                style,
            )
        } else {
            atlas.layout_text(
                &label.text,
                label.text_style.size,
                label.text_style.font,
                ppp,
                device,
                style,
            )
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

    // The glyphs start white and `style.fill` multiplies into them below, so a
    // label has one colour source.
    let text_colour = overlay_geometry::apply_opacity([1.0, 1.0, 1.0, 1.0], opacity);
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
            fills,
            [1, 1],
        );
    }
    viewport_overlays::tint_vertices_from(verts, glyph_start, label.style.tint);
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
                label.text_style.size,
                label.text_style.font,
                max_w,
                ppp,
                device,
                style,
            )
        } else {
            atlas.layout_text(
                &label.text,
                label.text_style.size,
                label.text_style.font,
                ppp,
                device,
                style,
            )
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
    bake_clip_rect(verts, &label.clip, item_start, ppp);
}

/// Emit a whole group (polylines, vector shapes, glyph runs, labels) into a fresh
/// vertex list, returning it with its fill records and the versions of the
/// glyph and coverage atlases it baked UVs against.
///
/// Glyph and coverage-cell UVs divide by the atlas size, so if rasterizing this
/// group's glyphs or cells grows an atlas, what was emitted before the grow
/// carries a stale divisor. The loop re-emits the atlas-backed portion once
/// both atlases have stopped growing (all this group's glyphs and cells are
/// then resident), so the returned geometry always uses a single, final atlas
/// size. The polylines are emitted once, and so are the vector shapes when
/// none of them draws from the coverage atlas.
///
/// `label_leaders` is passed through to each label's `emit_label`: true for a
/// self-anchoring label group, false for fixed-local labels in a mixed group.
pub(super) fn emit_group_verts(
    atlas: &mut crate::resources::overlay::font::GlyphAtlas,
    coverage: &mut crate::resources::overlay::coverage_atlas::CoverageAtlas,
    device: &crate::gpu::Device,
    polylines: &[crate::renderer::types::OverlayPolylineItem],
    vector_shapes: &[crate::renderer::types::OverlayShapeItem],
    glyph_runs: &[crate::renderer::types::GlyphRunItem],
    labels: &[crate::renderer::types::LabelItem],
    label_leaders: bool,
    ppp: f32,
) -> (
    Vec<crate::resources::OverlayTextVertex>,
    super::overlay_fill::FillTable,
    (u64, u64),
) {
    let mut base = Vec::new();
    let mut base_fills = super::overlay_fill::FillTable::without_textures();
    emit_base(&mut base, polylines, ppp, &mut base_fills);

    // Fast path: nothing draws from the atlas, so nothing grows it and the
    // vector shapes are as fixed as the polylines.
    if !group_uses_atlas(vector_shapes, glyph_runs, labels, ppp) {
        emit_vector_shapes(&mut base, vector_shapes, ppp, &mut base_fills, None);
        return (base, base_fills, (atlas.version(), coverage.version()));
    }

    // A doubling atlas stabilises in a few iterations; the guard bounds it.
    let mut guard = 0;
    loop {
        let v0 = (atlas.version(), coverage.version());
        let mut verts = base.clone();
        let mut fills = base_fills.clone();
        emit_vector_shapes(
            &mut verts,
            vector_shapes,
            ppp,
            &mut fills,
            Some((&mut *coverage, device)),
        );
        for run in glyph_runs {
            emit_glyph_run(&mut verts, atlas, device, run, ppp, &mut fills);
        }
        for label in labels {
            emit_label(
                &mut verts,
                atlas,
                device,
                label,
                label_leaders,
                ppp,
                &mut fills,
            );
        }
        guard += 1;
        let v1 = (atlas.version(), coverage.version());
        if v1 == v0 || guard >= 8 {
            return (verts, fills, v1);
        }
    }
}

/// Emit one analytic SDF shape (Rect, RoundedRect, Circle, Ring, ...) as one
/// `OverlayShapeInstance` plus its stacked shadow layers, for the retained
/// path. Mirrors the solid branch of the immediate shape emission
/// (`prepare_viewport_internal`) for the static case: no animation sampling, no
/// anchor resolution, and no internal clip mask (a retained group's outer clip is
/// carried by its text stream). Textured and backdrop-blur shapes are skipped: a
/// compiled group binds no image and runs no backdrop pass.
fn emit_sdf_shape(
    shape: &crate::renderer::types::OverlayShapeItem,
    out_verts: &mut Vec<crate::resources::OverlayShapeInstance>,
    out_shadows: &mut Vec<crate::resources::OverlayShadowLayerGpu>,
    out_fills: &mut super::overlay_fill::FillTable,
    ppp: f32,
) {
    use crate::renderer::types::{LineCap, OverlayFill, OverlayShape, TriangleDirection};
    if matches!(shape.shape, OverlayShape::Vector { .. })
        || shape.provides_mask.is_some()
        || shape.style.fill.texture_id().is_some()
        || shape.style.backdrop.blur > 0.0
        || shape.style.opacity <= 0.0
        || names_unusable_mask(&shape.clip, "shape")
    {
        return;
    }
    let item_start = out_verts.len();
    let op = shape.style.opacity;
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
    let Some(frame) = overlay_geometry::shape_frame(
        &shape.transform,
        [cx, cy],
        [hw + extra_expand, hh + extra_expand],
        shadow_pad + 1.0,
    ) else {
        return;
    };

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

    // The flat colour (a solid fill, or a gradient's first stop) with the
    // item's opacity and tint folded in; a gradient is a fill record, folded
    // the same way.
    let mut fc = super::overlay_fill::leading_colour(&shape.style.fill);
    fc[3] *= op;
    for (c, t) in fc.iter_mut().zip(shape.style.tint) {
        *c *= t;
    }
    let fill_index = match &shape.style.fill {
        OverlayFill::Solid(_) => super::overlay_fill::NO_FILL,
        fill => out_fills.push(fill, [hw, hh], op, shape.style.tint, [1, 1]),
    };

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
    let pivot_local = shape.transform.pivot;
    out_verts.push(crate::resources::OverlayShapeInstance {
        quad: frame.quad,
        pivot_half: [frame.pivot_px[0], frame.pivot_px[1], hw, hh],
        linv: frame.linv,
        fill_colour: fc,
        radii,
        shadow_meta: [
            base_index as f32,
            outer_count as f32,
            inner_count as f32,
            shape_type,
        ],
        clip_rect: [0.0; 4],
        misc: [-1.0, pivot_local[0], pivot_local[1], frame.item_scale],
        fill: [fill_index, 0.0, 0.0, 0.0],
    });
    bake_shape_clip_rect(out_verts, &shape.clip, item_start, ppp);
}

/// A compiled group's text-stream fill records as a texture, or nothing (and
/// no bytes) when the group has no gradient fill: its draws then bind the
/// frame's records, which no vertex of the group reads.
fn compiled_fill_records(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    fills: &super::overlay_fill::FillTable,
) -> (Option<crate::gpu::TextureView>, u64) {
    if fills.is_empty() {
        return (None, 0);
    }
    let (view, bytes) = crate::resources::overlay::overlay_shape::fill_record_texture(
        device,
        queue,
        fills.texels(),
    );
    (Some(view), bytes)
}

/// The extent of a compiled group in its own local logical pixels, over both
/// vertex streams. `None` when the group compiled nothing.
///
/// This is the box the group's alignment shifts against when it is
/// anchored, so it plays the role an item's own extent box plays. Computed once
/// at compile rather than per frame: the geometry is fixed by definition.
fn group_bounds(
    text: &[crate::resources::OverlayTextVertex],
    shapes: &[crate::resources::OverlayShapeInstance],
) -> Option<([f32; 2], [f32; 2])> {
    let mut min = [f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN];
    let mut any = false;
    // A shape's quad contributes its two extreme corners.
    let shape_corners = shapes.iter().flat_map(|s| {
        let [cx, cy, hx, hy] = s.quad;
        [[cx - hx, cy - hy], [cx + hx, cy + hy]]
    });
    for p in text.iter().map(|v| v.position).chain(shape_corners) {
        any = true;
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
    }
    any.then_some((min, max))
}

/// Whether a group draws anything from the glyph atlas: glyphs, or a vector
/// shape small enough to draw its fill from a coverage cell. Such a group bakes
/// atlas UVs at `ppp`, so it keeps its source and re-emits when the atlas grows
/// or is cleared, or the pixels per point change.
fn group_uses_atlas(
    vector_shapes: &[crate::renderer::types::OverlayShapeItem],
    glyph_runs: &[crate::renderer::types::GlyphRunItem],
    labels: &[crate::renderer::types::LabelItem],
    ppp: f32,
) -> bool {
    glyph_runs
        .iter()
        .any(|r| !r.glyphs.is_empty() && r.style.opacity > 0.0)
        || labels
            .iter()
            .any(|l| !l.text.is_empty() && l.style.opacity > 0.0)
        || vector_shapes
            .iter()
            .any(|s| viewport_overlays::retained_vector_uses_atlas(s, ppp))
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
    /// `font_size * pixels_per_point`, and so does the coverage of a vector
    /// shape small enough (512 device pixels a side) to draw its fill from a
    /// coverage cell rather than as triangles. A group that carries either is
    /// re-emitted automatically when an atlas grows or the frame's
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
    ///
    /// **An item's own clip is resolved here, not per frame.** `clip.rect` is
    /// baked into the compiled geometry as a fixed screen box, so it clips
    /// where it was authored rather than following the group's translate. A
    /// `clip.mask` cannot be honoured at all, because masks are registered per
    /// frame and a compiled group has no frame to resolve one against: an item
    /// naming one is skipped rather than drawn unclipped. Clip the group
    /// instead, through `RetainedOverlay`.
    ///
    /// **A compiled group draws every item source-over.** An item's
    /// `style.blend` is not kept in the compiled geometry.
    ///
    /// **A compiled group draws no image fill.** An analytic shape with a
    /// texture fill (or a backdrop blur) is skipped, a textured polyline or
    /// vector path draws without its fill, and textured text draws in the
    /// fill's tint. Gradients and solid fills draw as they do immediately.
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
        let has_glyphs = group_uses_atlas(vector_shapes, glyph_runs, labels, pixels_per_point);

        let (verts, text_fills, baked_version) = emit_group_verts(
            &mut self.resources.content.glyph_atlas,
            &mut self.resources.content.coverage_atlas,
            device,
            polylines,
            vector_shapes,
            glyph_runs,
            labels,
            // Fixed-local labels in a mixed group: anchor ignored, so no leader.
            false,
            pixels_per_point,
        );
        let (text_fill_tex, text_fill_bytes) = compiled_fill_records(device, queue, &text_fills);
        // Upload any newly rasterized glyphs so the atlas texture has them.
        self.resources.content.glyph_atlas.upload_if_dirty(queue);
        self.resources.content.coverage_atlas.upload_if_dirty(queue);

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
        let mut shape_verts: Vec<crate::resources::OverlayShapeInstance> = Vec::new();
        let mut shadow_layers: Vec<crate::resources::OverlayShadowLayerGpu> = Vec::new();
        let mut shape_fills = super::overlay_fill::FillTable::without_textures();
        for shape in vector_shapes {
            if !matches!(
                shape.shape,
                crate::renderer::types::OverlayShape::Vector { .. }
            ) {
                emit_sdf_shape(
                    shape,
                    &mut shape_verts,
                    &mut shadow_layers,
                    &mut shape_fills,
                    pixels_per_point,
                );
            }
        }
        let (shape_instance_buf, shadow_tex, shape_fill_tex, shape_bytes) =
            if shape_verts.is_empty() {
                (None, None, None, 0)
            } else {
                // The shape pipeline layout always expects a shadow-layer texture;
                // provide a dummy entry when no shape has shadows.
                if shadow_layers.is_empty() {
                    shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                        colour: [0.0; 4],
                        params: [0.0; 4],
                        params2: [0.0, 1.0, 0.0, 0.0],
                    });
                }
                let sv_bytes = std::mem::size_of_val(&shape_verts[..]) as u64;
                let sv = device.create_buffer(&crate::gpu::BufferDescriptor {
                    label: Some("compiled_overlay_shape_ibuf"),
                    size: sv_bytes,
                    usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&sv, 0, bytemuck::cast_slice(&shape_verts));
                let (sh, sh_bytes) = crate::resources::overlay::overlay_shape::shadow_layer_texture(
                    device,
                    queue,
                    &shadow_layers,
                );
                let (fl, fl_bytes) = crate::resources::overlay::overlay_shape::fill_record_texture(
                    device,
                    queue,
                    shape_fills.texels(),
                );
                (Some(sv), Some(sh), Some(fl), sv_bytes + sh_bytes + fl_bytes)
            };

        // Retain the source only when the group has glyphs (the only geometry that
        // can go stale). Polyline/vector/shape-only groups never re-emit.
        let source = has_glyphs.then(|| crate::resources::CompiledSource {
            polylines: polylines.to_vec(),
            vector_shapes: vector_shapes.to_vec(),
            glyph_runs: glyph_runs.to_vec(),
            labels: labels.to_vec(),
            baked_atlas_version: baked_version.0,
            baked_coverage_version: baked_version.1,
            baked_ppp: pixels_per_point,
        });

        let total_bytes = bytes + text_fill_bytes + shape_bytes;
        self.resources.content.overlay_geometry.insert(
            crate::resources::CompiledOverlay {
                vertex_buf,
                vertex_count: verts.len() as u32,
                text_fill_tex,
                text_fill_bytes,
                shape_bytes,
                shape_instance_buf,
                shape_instance_count: shape_verts.len() as u32,
                shadow_tex,
                shape_fill_tex,
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
    /// top of the resolved anchor (to scroll or nudge), and the group's own
    /// per-frame opacity and clip apply as for any retained group.
    ///
    /// The label's own `clip.rect` is baked into the compiled geometry as a
    /// fixed screen box, so it clips where it was authored rather than
    /// following the group's translate. A `clip.mask` cannot be honoured at all,
    /// because masks are registered per frame and a compiled group has no frame
    /// to resolve one against: a label naming one is skipped rather than drawn
    /// unclipped. Clip the group instead.
    ///
    /// Release it with [`free_overlay_geometry`](Self::free_overlay_geometry).
    pub fn compile_overlay_label(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        label: &crate::renderer::types::LabelItem,
        pixels_per_point: f32,
    ) -> crate::renderer::OverlayGeometryId {
        let (verts, text_fills, baked_version) = emit_group_verts(
            &mut self.resources.content.glyph_atlas,
            &mut self.resources.content.coverage_atlas,
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
        self.resources.content.coverage_atlas.upload_if_dirty(queue);
        let (text_fill_tex, text_fill_bytes) = compiled_fill_records(device, queue, &text_fills);

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
            baked_atlas_version: baked_version.0,
            baked_coverage_version: baked_version.1,
            baked_ppp: pixels_per_point,
        };

        self.resources.content.overlay_geometry.insert(
            crate::resources::CompiledOverlay {
                vertex_buf,
                vertex_count: verts.len() as u32,
                text_fill_tex,
                text_fill_bytes,
                shape_bytes: 0,
                shape_instance_buf: None,
                shape_instance_count: 0,
                shadow_tex: None,
                shape_fill_tex: None,
                source: Some(source),
                anchor: Some(label.anchoring.origin),
                bounds: group_bounds(&verts, &[]),
            },
            bytes + text_fill_bytes,
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
    ///
    /// Returns `true` when the group was rebuilt, which is what
    /// `FrameStats::overlay_retained_reemitted` counts.
    pub(super) fn reemit_overlay_geometry_if_stale(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        id: crate::renderer::OverlayGeometryId,
        ppp: f32,
    ) -> bool {
        let current_version = self.resources.content.glyph_atlas.version();
        let current_coverage = self.resources.content.coverage_atlas.version();
        let stale = match self.resources.content.overlay_geometry.get(id) {
            Some(c) => match &c.source {
                Some(src) => {
                    src.baked_atlas_version != current_version
                        || src.baked_coverage_version != current_coverage
                        || src.baked_ppp != ppp
                }
                None => false,
            },
            None => false,
        };
        if !stale {
            return false;
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
        let (verts, text_fills, baked_version) = emit_group_verts(
            &mut self.resources.content.glyph_atlas,
            &mut self.resources.content.coverage_atlas,
            device,
            &src.polylines,
            &src.vector_shapes,
            &src.glyph_runs,
            &src.labels,
            label_leaders,
            ppp,
        );
        self.resources.content.glyph_atlas.upload_if_dirty(queue);
        self.resources.content.coverage_atlas.upload_if_dirty(queue);
        let (text_fill_tex, text_fill_bytes) = compiled_fill_records(device, queue, &text_fills);

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
        let Some(c) = self.resources.content.overlay_geometry.get_mut(id) else {
            return false;
        };
        c.vertex_buf = vertex_buf;
        c.vertex_count = verts.len() as u32;
        // The text boxes the fills span can move with the layout too.
        c.text_fill_tex = text_fill_tex;
        c.text_fill_bytes = text_fill_bytes;
        // A re-emit re-lays out the glyphs, so the extent can move.
        if c.shape_instance_count == 0 {
            c.bounds = bounds;
        }
        if let Some(s) = &mut c.source {
            s.baked_atlas_version = baked_version.0;
            s.baked_coverage_version = baked_version.1;
            s.baked_ppp = ppp;
        }
        // The new text buffer replaces the old one, so the slot's charge has to
        // move with it. The shape half is untouched by a re-emit and carries over.
        let total_bytes = bytes + text_fill_bytes + c.shape_bytes;
        self.resources
            .content
            .overlay_geometry
            .set_bytes(id, total_bytes);
        true
    }
}
