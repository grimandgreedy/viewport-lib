//! Compile overlay items into a retained [`OverlayGeometryId`] once, so the
//! renderer re-draws them from a cached buffer each frame instead of
//! re-tessellating them. See `docs/plans/retained-overlay-geometry-plan.md`.

use super::*;

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
        if poly.closed && poly.texture.is_none() {
            if let Some(fill) = &poly.fill {
                overlay_geometry::emit_filled_polyline(
                    verts,
                    &poly.points,
                    fill,
                    poly.opacity,
                    0.0,
                    0.0,
                );
            }
        }
        if poly.thickness > 0.0 {
            let mut colour = poly.colour;
            colour[3] *= poly.opacity;
            overlay_geometry::emit_polyline_stroke(verts, poly, colour, 0.0, 0.0);
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
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for g in &run.glyphs {
        min_x = min_x.min(g.x);
        min_y = min_y.min(g.y);
        max_x = max_x.max(g.x);
        max_y = max_y.max(g.y);
    }
    let run_x = run.position[0] + run.align_x.align_shift(max_x - min_x);
    let run_y = run.position[1] + run.align_y.align_shift(max_y - min_y);
    let opacity = run.opacity.clamp(0.0, 1.0);
    let quads = atlas.layout_glyph_run(
        run.glyphs.iter().enumerate().map(|(i, g)| {
            let colour = run.colours.get(i).copied().unwrap_or(run.colour);
            (g.glyph_id, g.x, g.y, overlay_geometry::apply_opacity(colour, opacity))
        }),
        run.font_size,
        run.font,
        ppp,
        device,
    );
    overlay_geometry::emit_glyph_quads_colored(verts, &quads, run_x, run_y, 0.0, 0.0);
}

/// Emit a whole group (polylines, vector shapes, glyph runs) into a fresh vertex
/// list, returning it with the atlas version it baked glyph UVs against.
///
/// Glyph UVs divide by the atlas size, so if rasterizing this group's glyphs grows
/// the atlas, glyphs emitted before the grow carry a stale divisor. The loop
/// re-emits the glyph portion once the atlas has stopped growing (all this group's
/// glyphs are then resident), so the returned geometry always uses a single,
/// final atlas size. The base (polyline/vector) portion is emitted once.
pub(super) fn emit_group_verts(
    atlas: &mut crate::resources::overlay::font::GlyphAtlas,
    device: &crate::gpu::Device,
    polylines: &[crate::renderer::types::OverlayPolylineItem],
    vector_shapes: &[crate::renderer::types::OverlayShapeItem],
    glyph_runs: &[crate::renderer::types::GlyphRunItem],
    ppp: f32,
) -> (Vec<crate::resources::OverlayTextVertex>, u64) {
    let mut base = Vec::new();
    emit_base(&mut base, polylines, vector_shapes);

    // Fast path: no glyphs, nothing can grow the atlas.
    if glyph_runs.iter().all(|r| r.glyphs.is_empty() || r.opacity <= 0.0) {
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
    use crate::renderer::types::{
        BorderMode, LineCap, OverlayFill, OverlayShape, TriangleDirection,
    };
    if matches!(shape.shape, OverlayShape::Vector { .. })
        || shape.clip_mask_id.is_some()
        || shape.texture.is_some()
        || shape.backdrop_blur > 0.0
        || shape.opacity <= 0.0
    {
        return;
    }
    let op = shape.opacity;
    let hw = shape.size[0] * 0.5;
    let hh = shape.size[1] * 0.5;
    let cx = shape.position[0] + hw;
    let cy = shape.position[1] + hh;

    let mut shadow_pad = if shape.shadow_radius > 0.0 {
        shape.shadow_radius + shape.shadow_offset[0].abs().max(shape.shadow_offset[1].abs())
    } else {
        0.0
    };
    for l in &shape.shadows {
        shadow_pad = shadow_pad.max(l.radius + l.offset[0].abs().max(l.offset[1].abs()));
    }
    let extra_expand = match &shape.shape {
        OverlayShape::Line { thickness, .. } => thickness * 0.5,
        _ => 0.0,
    };
    let bx = hw + shape.border_width + extra_expand;
    let by = hh + shape.border_width + extra_expand;
    let (rx, ry) = if shape.rotation != 0.0 {
        let c = shape.rotation.cos();
        let s = shape.rotation.sin();
        let piv = shape.rotation_pivot;
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
            [inner_radius_frac.clamp(0.0, 1.0), *start_angle, *end_angle, 0.0],
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
            [(*points).max(3) as f32, inner_radius_frac.clamp(0.0, 1.0), 0.0, 0.0],
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
    let gradient_params = match &shape.fill {
        OverlayFill::Solid(c) => {
            stop_colours[0] = *c;
            stop_colours[1] = *c;
            stop_count = 0.0;
            [0.0f32, 0.0]
        }
        OverlayFill::LinearGradient {
            start_colour,
            end_colour,
            angle,
        } => {
            stop_colours[0] = *start_colour;
            stop_colours[1] = *end_colour;
            stop_count = 2.0;
            [1.0f32, *angle]
        }
        OverlayFill::RadialGradient {
            centre_colour,
            edge_colour,
        } => {
            stop_colours[0] = *centre_colour;
            stop_colours[1] = *edge_colour;
            stop_count = 2.0;
            [2.0f32, 0.0]
        }
        OverlayFill::ConicalGradient {
            start_colour,
            end_colour,
            offset_angle,
        } => {
            stop_colours[0] = *start_colour;
            stop_colours[1] = *end_colour;
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
        _ => {
            stop_count = 0.0;
            [0.0f32, 0.0]
        }
    };
    for colour in &mut stop_colours {
        colour[3] *= op;
    }
    let fc = stop_colours[0];
    let fc2 = stop_colours[1];
    let mut bc = shape.border_colour;
    bc[3] *= op;
    let mut sc = shape.shadow_colour;
    sc[3] *= op;
    let border_mode_f = match shape.border_mode {
        BorderMode::Inset => 0.0,
        BorderMode::Outer => 1.0,
        BorderMode::Center => 2.0,
    };
    let gp4 = [gradient_params[0], gradient_params[1], stop_count, 0.0];

    let base_index = out_shadows.len();
    let (mut outer_count, mut inner_count) = (0usize, 0usize);
    let max_layers = crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS;
    if !shape.shadows.is_empty() {
        for l in shape.shadows.iter().take(max_layers) {
            let mut col = l.colour;
            col[3] *= op;
            out_shadows.push(crate::resources::OverlayShadowLayerGpu {
                colour: col,
                params: [l.radius, l.offset[0], l.offset[1], 0.0],
            });
            outer_count += 1;
        }
    } else if shape.shadow_radius > 0.0 && !shape.shadow_inset {
        out_shadows.push(crate::resources::OverlayShadowLayerGpu {
            colour: sc,
            params: [shape.shadow_radius, shape.shadow_offset[0], shape.shadow_offset[1], 0.0],
        });
        outer_count += 1;
    }
    if !shape.inner_shadows.is_empty() {
        for l in shape.inner_shadows.iter().take(max_layers) {
            let mut col = l.colour;
            col[3] *= op;
            out_shadows.push(crate::resources::OverlayShadowLayerGpu {
                colour: col,
                params: [l.radius, l.offset[0], l.offset[1], 1.0],
            });
            inner_count += 1;
        }
    } else if shape.shadow_radius > 0.0 && shape.shadow_inset {
        out_shadows.push(crate::resources::OverlayShadowLayerGpu {
            colour: sc,
            params: [shape.shadow_radius, shape.shadow_offset[0], shape.shadow_offset[1], 1.0],
        });
        inner_count += 1;
    }
    let shadow_index = [
        base_index as f32,
        outer_count as f32,
        inner_count as f32,
        border_mode_f,
    ];
    let rotation_pivot = [
        shape.rotation,
        shape.rotation_pivot[0],
        shape.rotation_pivot[1],
        0.0,
    ];
    let half_size = [hw, hh];
    let corners = [
        (cx - ex, cy - ey, -ex, -ey),
        (cx + ex, cy - ey, ex, -ey),
        (cx + ex, cy + ey, ex, ey),
        (cx - ex, cy - ey, -ex, -ey),
        (cx + ex, cy + ey, ex, ey),
        (cx - ex, cy + ey, -ex, ey),
    ];
    for (px, py, lx, ly) in corners {
        out_verts.push(crate::resources::OverlayShapeVertex {
            position: [px, py],
            local_pos: [lx, ly],
            fill_colour: fc,
            border_colour: bc,
            half_size,
            radii,
            border_width: shape.border_width,
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

impl ViewportRenderer {
    /// Compile a group of polylines, vector shapes, and glyph runs into a retained
    /// overlay-geometry handle.
    ///
    /// The items are tessellated once (polyline fills and strokes, vector-path
    /// fills and borders, glyph quads) into local logical-pixel geometry and
    /// uploaded to a buffer that lives until the group is freed. Each frame, submit
    /// the returned id through `OverlayFrame::retained` as a [`RetainedOverlay`]
    /// carrying a per-frame translate, opacity, z-order, and clip, instead of
    /// pushing the items into `OverlayFrame`. Release it with
    /// [`free_overlay_geometry`](Self::free_overlay_geometry).
    ///
    /// `pixels_per_point` must match the frame's; glyphs rasterize at
    /// `font_size * pixels_per_point`. A group that carries glyphs is re-emitted
    /// automatically when the atlas grows or the frame's `pixels_per_point` changes,
    /// so callers only re-compile when their own text, style, or width changes.
    ///
    /// All slices are optional (pass `&[]`). `vector_shapes` entries that are not
    /// `OverlayShape::Vector` are ignored (SDF shapes are a later phase). Geometry is
    /// taken in its own screen-pixel space; position the group each frame with
    /// `RetainedOverlay::translate`. Anchors are not resolved (a retained group is
    /// static geometry moved by its per-frame translate).
    pub fn compile_overlay_geometry(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        polylines: &[crate::renderer::types::OverlayPolylineItem],
        vector_shapes: &[crate::renderer::types::OverlayShapeItem],
        glyph_runs: &[crate::renderer::types::GlyphRunItem],
        pixels_per_point: f32,
    ) -> crate::renderer::OverlayGeometryId {
        let has_glyphs = glyph_runs
            .iter()
            .any(|r| !r.glyphs.is_empty() && r.opacity > 0.0);

        let (verts, baked_version) = emit_group_verts(
            &mut self.resources.content.glyph_atlas,
            device,
            polylines,
            vector_shapes,
            glyph_runs,
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
            if !matches!(shape.shape, crate::renderer::types::OverlayShape::Vector { .. }) {
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
            },
            total_bytes,
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
        // Clone the source so the atlas and store borrows do not overlap.
        let src = self
            .resources
            .content
            .overlay_geometry
            .get(id)
            .and_then(|c| c.source.clone())
            .unwrap();
        let (verts, baked_version) = emit_group_verts(
            &mut self.resources.content.glyph_atlas,
            device,
            &src.polylines,
            &src.vector_shapes,
            &src.glyph_runs,
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

        if let Some(c) = self.resources.content.overlay_geometry.get_mut(id) {
            c.vertex_buf = vertex_buf;
            c.vertex_count = verts.len() as u32;
            c.bytes = bytes;
            if let Some(s) = &mut c.source {
                s.baked_atlas_version = baked_version;
                s.baked_ppp = ppp;
            }
        }
    }
}
