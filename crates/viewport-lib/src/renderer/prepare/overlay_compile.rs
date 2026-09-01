//! Compile overlay items into a retained [`OverlayGeometryId`] once, so the
//! renderer re-draws them from a cached buffer each frame instead of
//! re-tessellating them. See `docs/plans/retained-overlay-geometry-plan.md`.

use super::*;

impl ViewportRenderer {
    /// Compile a group of polylines and vector shapes into a retained
    /// overlay-geometry handle.
    ///
    /// The items are tessellated once (polyline fills and strokes, vector-path
    /// fills and outline borders) into local logical-pixel geometry and uploaded to
    /// a buffer that lives until the group is freed. Each frame, submit the returned
    /// id through `OverlayFrame::retained` as a [`RetainedOverlay`] carrying a
    /// per-frame translate, opacity, z-order, and clip, instead of pushing the items
    /// into `OverlayFrame`. Release it with
    /// [`free_overlay_geometry`](Self::free_overlay_geometry).
    ///
    /// Both slices are optional (pass `&[]`). `vector_shapes` entries that are not
    /// `OverlayShape::Vector` are ignored (SDF shapes are a later phase). Geometry is
    /// taken in its own screen-pixel space; position the group each frame with
    /// `RetainedOverlay::translate`. Anchors are not resolved here (a retained group
    /// is static geometry moved by its per-frame translate), so world-anchored items
    /// are not supported on this path.
    pub fn compile_overlay_geometry(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        polylines: &[crate::renderer::types::OverlayPolylineItem],
        vector_shapes: &[crate::renderer::types::OverlayShapeItem],
    ) -> crate::renderer::OverlayGeometryId {
        let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();
        for poly in polylines {
            if poly.points.len() < 2 || poly.opacity <= 0.0 {
                continue;
            }
            // Closed polylines with a fill draw a filled interior first, matching
            // the immediate path.
            if poly.closed && poly.texture.is_none() {
                if let Some(fill) = &poly.fill {
                    overlay_geometry::emit_filled_polyline(
                        &mut verts,
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
                overlay_geometry::emit_polyline_stroke(&mut verts, poly, colour, 0.0, 0.0);
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
                viewport_overlays::emit_vector_shape(
                    &mut verts, shape, subpaths, *fill_rule, 0.0, 0.0,
                );
            }
        }

        let bytes = std::mem::size_of_val(&verts[..]) as u64;
        // Always allocate a non-empty buffer so the handle is drawable even for an
        // empty group (it simply draws nothing).
        let vertex_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("compiled_overlay_vbuf"),
            size: bytes.max(4),
            usage: crate::gpu::BufferUsages::VERTEX | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !verts.is_empty() {
            queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&verts));
        }

        self.resources.content.overlay_geometry.insert(
            crate::resources::CompiledOverlay {
                vertex_buf,
                vertex_count: verts.len() as u32,
                bytes,
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
}
