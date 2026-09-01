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

        // Retain the source only when the group has glyphs (the only geometry that
        // can go stale). Polyline/vector-only groups never re-emit.
        let source = has_glyphs.then(|| crate::resources::CompiledSource {
            polylines: polylines.to_vec(),
            vector_shapes: vector_shapes.to_vec(),
            glyph_runs: glyph_runs.to_vec(),
            baked_atlas_version: baked_version,
            baked_ppp: pixels_per_point,
        });

        self.resources.content.overlay_geometry.insert(
            crate::resources::CompiledOverlay {
                vertex_buf,
                vertex_count: verts.len() as u32,
                bytes,
                source,
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
