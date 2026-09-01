//! Overlay prepare passes: labels, glyph runs, polylines, and SDF overlay
//! shapes.

use super::*;

/// Encode an overlay shape into `(shape_type, radii)` for the SDF, matching the
/// draw path's packing. `hw`/`hh` are the half-extents used to clamp corner radii.
/// This is shared by the shape draw path and the clip-shape registry so the mask
/// SDF is evaluated exactly as the shape would be drawn.
fn encode_overlay_shape(
    shape: &crate::renderer::types::OverlayShape,
    hw: f32,
    hh: f32,
) -> (f32, [f32; 4]) {
    use crate::renderer::types::{LineCap, OverlayShape, TriangleDirection};
    match shape {
        OverlayShape::Rect { corner_radius } => {
            let r = corner_radius.min(hw).min(hh).max(0.0);
            (0.0, [r, r, r, r])
        }
        OverlayShape::RoundedRect { radii: r } => {
            let clamped = [
                r[1].min(hw).min(hh).max(0.0),
                r[2].min(hw).min(hh).max(0.0),
                r[3].min(hw).min(hh).max(0.0),
                r[0].min(hw).min(hh).max(0.0),
            ];
            (0.0, clamped)
        }
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
            let dir_f = match direction {
                TriangleDirection::Up => 0.0,
                TriangleDirection::Down => 1.0,
                TriangleDirection::Left => 2.0,
                TriangleDirection::Right => 3.0,
            };
            (6.0, [dir_f, 0.0, 0.0, 0.0])
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
        // A vector shape has no analytic SDF; it is not drawn through this
        // encoder. Callers skip Vector shapes before reaching here. Used as a
        // clip mask it degrades to its bounding box.
        OverlayShape::Vector { .. } => (0.0, [0.0; 4]),
        // OverlayShape is non_exhaustive; no special edge/radius for shapes not handled here.
        _ => (0.0, [0.0; 4]),
    }
}

/// Curve-flattening tolerance for vector-shape fills, in logical pixels.
const VECTOR_FILL_TOLERANCE: f32 = 0.2;

/// Mitre limit for a vector shape's outline stroke (it has no per-item value,
/// unlike `OverlayPolylineItem`).
const VECTOR_BORDER_MITRE_LIMIT: f32 = 4.0;

/// Map a vector shape's path-local points into screen-space logical pixels,
/// applying the item's position and rotation about its centre plus pivot. This
/// is the inverse of the frame `OverlayShapeItem::distance` evaluates in, so
/// the drawn shape and the hit-test agree.
fn transform_vector_positions(
    local: &[[f32; 2]],
    shape: &crate::renderer::types::OverlayShapeItem,
) -> Vec<[f32; 2]> {
    let hw = shape.size[0] * 0.5;
    let hh = shape.size[1] * 0.5;
    let cx = shape.position[0] + hw;
    let cy = shape.position[1] + hh;
    let piv = shape.rotation_pivot;
    let c = shape.rotation.cos();
    let s = shape.rotation.sin();
    local
        .iter()
        .map(|lp| {
            // Path-local -> centred (the unrotated `p` frame in `distance`).
            let cpx = lp[0] - hw;
            let cpy = lp[1] - hh;
            if shape.rotation == 0.0 {
                return [cx + cpx, cy + cpy];
            }
            let ax = cpx - piv[0];
            let ay = cpy - piv[1];
            let rx = c * ax - s * ay + piv[0];
            let ry = s * ax + c * ay + piv[1];
            [cx + rx, cy + ry]
        })
        .collect()
}

/// Tessellate and emit a vector shape's fill, plus its outline border when set,
/// as `OverlayTextVertex`s into `batch`.
pub(super) fn emit_vector_shape(
    batch: &mut Vec<crate::resources::OverlayTextVertex>,
    shape: &crate::renderer::types::OverlayShapeItem,
    subpaths: &[crate::renderer::types::SubPath],
    fill_rule: crate::renderer::types::FillRule,
    vp_w: f32,
    vp_h: f32,
) {
    let mesh = super::overlay_vector::tessellate(subpaths, fill_rule, VECTOR_FILL_TOLERANCE);
    if !mesh.indices.is_empty() {
        let positions = transform_vector_positions(&mesh.positions, shape);
        emit_vector_fill(
            batch,
            &positions,
            &mesh.indices,
            &shape.fill,
            shape.opacity,
            vp_w,
            vp_h,
        );
    }

    // Border: a vector outline stroke, not an SDF band. Stroke each flattened
    // contour through the same tessellator polylines use.
    if shape.border_width > 0.0 && shape.border_colour[3] > 0.0 {
        let mut colour = shape.border_colour;
        colour[3] *= shape.opacity;
        for contour in crate::renderer::types::flatten_contours(subpaths) {
            if contour.len() < 2 {
                continue;
            }
            let pts = transform_vector_positions(&contour, shape);
            batch.extend(tessellate_polyline(
                &pts,
                shape.border_width,
                true,
                crate::renderer::types::LineJoin::Mitre,
                VECTOR_BORDER_MITRE_LIMIT,
                crate::renderer::types::PolylineCap::Butt,
                colour,
                vp_w,
                vp_h,
            ));
        }
    }
}

/// Build the per-frame clip-shape registry from the overlay shapes that are clip
/// masks (`clip_mask_id` set). Returns the GPU array (framebuffer-pixel geometry),
/// a map from `clip_mask_id` to its index in that array, and the axis-aligned
/// bounding box (framebuffer pixels) of each mask, parallel to the GPU array. A
/// mask's own `clip_id` links it to a parent, so nested masks compose by
/// intersection.
///
/// The bounding boxes drive the shader's cheap pre-reject and share this array's
/// indexing, so a drawn item's `clip_rect` and `clip_index` always describe the
/// same mask. A `clip_mask_id` is expected to be unique within a frame; if one is
/// reused, the first mask with that id wins (both for the index and the bbox),
/// which is why the bbox comes from here rather than a separate id-keyed map.
fn build_clip_shapes(
    shapes: &[crate::renderer::types::OverlayShapeItem],
    ppp: f32,
    viewport: [f32; 2],
    view: &glam::Mat4,
    proj: &glam::Mat4,
) -> (
    Vec<crate::resources::ClipShapeGpu>,
    std::collections::HashMap<u32, i32>,
    Vec<[f32; 4]>,
) {
    // Collect masks in stable order, recording each id's index. Each mask
    // resolves its anchor to an effective top-left, the same way the draw loop
    // does, so corner- and world-anchored masks clip where they draw. A mask
    // whose world anchor is culled is dropped: items referencing it fall through
    // to no clipping, matching how an absent mask id behaves.
    let mut index_of: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
    let mut masks: Vec<(&crate::renderer::types::OverlayShapeItem, [f32; 2])> = Vec::new();
    for s in shapes {
        if let Some(id) = s.clip_mask_id {
            if let std::collections::hash_map::Entry::Vacant(e) = index_of.entry(id) {
                if let Some(tl) = s.resolve_top_left(viewport, view, proj) {
                    e.insert(masks.len() as i32);
                    masks.push((s, tl));
                }
            }
        }
    }
    let mut out = Vec::with_capacity(masks.len());
    let mut bboxes = Vec::with_capacity(masks.len());
    for (s, tl) in &masks {
        let hw = s.size[0] * 0.5;
        let hh = s.size[1] * 0.5;
        let (shape_type, radii_l) = encode_overlay_shape(&s.shape, hw, hh);
        // Scale to framebuffer pixels. Length-type radii (rounded-rect corners at
        // shape_type 0, line thickness at 7) scale by ppp; fractions and angles do not.
        let mut radii = radii_l;
        let st = shape_type as i32;
        if st == 0 {
            for r in radii.iter_mut() {
                *r *= ppp;
            }
        } else if st == 7 {
            radii[0] *= ppp;
        }
        let center = [(tl[0] + hw) * ppp, (tl[1] + hh) * ppp];
        let half = [hw * ppp, hh * ppp];
        let parent = s
            .clip_id
            .and_then(|pid| index_of.get(&pid).copied())
            .unwrap_or(-1);
        out.push(crate::resources::ClipShapeGpu {
            center,
            half_size: half,
            radii,
            params: [shape_type, s.rotation, parent as f32, 0.0],
            pivot: [s.rotation_pivot[0] * ppp, s.rotation_pivot[1] * ppp],
            _pad: [0.0, 0.0],
        });
        bboxes.push([
            tl[0] * ppp,
            tl[1] * ppp,
            (tl[0] + s.size[0]) * ppp,
            (tl[1] + s.size[1]) * ppp,
        ]);
    }
    (out, index_of, bboxes)
}

/// Create the clip-shape storage buffer for a frame, always non-empty (a single
/// zero entry when there are no masks) so the pipeline layout is always satisfied.
fn upload_clip_buffer(
    device: &crate::gpu::Device,
    queue: &crate::gpu::Queue,
    clips: &[crate::resources::ClipShapeGpu],
) -> crate::gpu::Buffer {
    let dummy = [crate::resources::ClipShapeGpu {
        center: [0.0, 0.0],
        half_size: [0.0, 0.0],
        radii: [0.0; 4],
        params: [0.0, 0.0, -1.0, 0.0],
        pivot: [0.0, 0.0],
        _pad: [0.0, 0.0],
    }];
    let data: &[crate::resources::ClipShapeGpu] = if clips.is_empty() { &dummy } else { clips };
    let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
        label: Some("overlay_clip_buf"),
        size: std::mem::size_of_val(data) as u64,
        usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
    buf
}

impl ViewportRenderer {
    /// Build the transform-gizmo overlay items for this frame, if a gizmo is
    /// active. Returns `(shapes, polylines)` in the overlay coordinate space,
    /// ready to merge into the viewport's overlay batches. Empty when
    /// `frame.interaction.gizmo_model` is `None`.
    ///
    /// The gizmo used to draw through its own always-on-top 3D pipeline; it now
    /// projects to 2D overlay primitives here so no dedicated GPU resources are
    /// needed. Centre and screen scale come straight out of the gizmo model
    /// matrix the host already supplies (translation and uniform scale).
    fn gizmo_overlay_items(
        &self,
        frame: &FrameData,
    ) -> (
        Vec<crate::renderer::types::OverlayShapeItem>,
        Vec<crate::renderer::types::OverlayPolylineItem>,
    ) {
        let mut shapes = Vec::new();
        let mut polylines = Vec::new();
        if let Some(model) = frame.interaction.gizmo_model {
            let (scale, _rot, translation) = model.to_scale_rotation_translation();
            let cam = &frame.camera.render_camera;
            crate::interaction::manipulation::gizmo_overlay::build_gizmo_overlays(
                frame.interaction.gizmo_mode,
                cam.view_proj(),
                glam::Vec3::from(cam.forward),
                frame.camera.viewport_size,
                translation,
                scale.x,
                frame.interaction.gizmo_hovered,
                frame.interaction.gizmo_space_orientation,
                &mut shapes,
                &mut polylines,
            );
        }
        (shapes, polylines)
    }

    /// The bottom-left XYZ orientation indicator, emitted as screen-space overlay
    /// shapes (axis lines, circles, rings, letters) into the shared shape pass.
    /// Empty when `show_axes_indicator` is off or the viewport has no area.
    fn axes_overlay_items(
        &self,
        frame: &FrameData,
    ) -> Vec<crate::renderer::types::OverlayShapeItem> {
        let mut shapes = Vec::new();
        let [w, h] = frame.camera.viewport_size;
        if frame.viewport.show_axes_indicator && w > 0.0 && h > 0.0 {
            crate::interaction::widgets::axes_indicator::build_axes_overlays(
                w,
                h,
                frame.camera.render_camera.orientation,
                &mut shapes,
            );
        }
        shapes
    }

    /// Ensure the shared overlay viewport-size uniform exists and holds this
    /// frame's logical size `[w, h, 0, 0]`. Overlay vertices are stored in local
    /// logical pixels; the overlay shaders map them to NDC using this uniform, so
    /// overlay geometry does not depend on the viewport size.
    fn ensure_overlay_viewport_buf(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vp_w: f32,
        vp_h: f32,
    ) {
        let data = [vp_w, vp_h, 0.0, 0.0];
        if let Some(buf) = &self.overlay_viewport_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&data));
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("overlay_viewport_uniform"),
                size: std::mem::size_of_val(&data) as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
            self.overlay_viewport_buf = Some(buf);
        }
    }

    /// Ensure `overlay_instances_buf` holds at least the identity instance. The
    /// label prepare fills it with the identity plus one slot per retained group;
    /// this fallback covers a frame with immediate shapes but no text or retained
    /// groups, so the shape pipeline's binding is always satisfied.
    fn ensure_overlay_instances_default(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        if self.overlay_instances_ready {
            return;
        }
        let data = [crate::resources::OverlayInstance::IDENTITY];
        let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
            label: Some("overlay_instances_buf"),
            size: std::mem::size_of_val(&data) as u64,
            usage: crate::gpu::BufferUsages::STORAGE | crate::gpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
        self.overlay_instances_buf = Some(buf);
        self.overlay_instances_ready = true;
    }

    pub(super) fn prepare_overlay_labels(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // ---------------------------------------------------------------
        // Overlay labels
        // ---------------------------------------------------------------
        // Labels, glyph runs, and polylines are merged into a single
        // z_order-sorted vertex buffer so that items at the same z_order
        // interleave correctly (e.g. a polyline underline at i32::MAX sits with
        // console text at i32::MAX but above HUD labels at 0).
        self.label_gpu_data = None;
        let (_, gizmo_polylines) = self.gizmo_overlay_items(frame);
        let has_vector_shape = frame.overlays.shapes.iter().any(|s| {
            matches!(
                &s.shape,
                crate::renderer::types::OverlayShape::Vector { .. }
            )
        });
        let has_overlay = !frame.overlays.labels.is_empty()
            || !frame.overlays.glyph_runs.is_empty()
            || !frame.overlays.polylines.is_empty()
            || !gizmo_polylines.is_empty()
            || has_vector_shape
            || !frame.overlays.retained.is_empty();
        if has_overlay {
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            let ppp = frame.camera.pixels_per_point;
            if vp_w > 0.0 && vp_h > 0.0 {
                let view = &frame.camera.render_camera.view;
                let proj = &frame.camera.render_camera.projection;

                // Each item (label, glyph run, or polyline) contributes a
                // (z_order, verts) batch. A stable sort keeps submission order
                // within an equal z_order.
                let mut batches: Vec<(i32, Vec<crate::resources::OverlayTextVertex>)> = Vec::new();

                // Clip masks: an overlay shape with `clip_mask_id` set contributes a
                // clipping bounding box (framebuffer pixels) plus an SDF entry. Labels
                // with a matching `clip_id` are clipped to it, the same way the shape
                // pipeline clips shapes. (Shape-accurate SDF clipping is applied in the
                // shape pass; text uses the bounding box, exact for the rectangular
                // masks scroll containers use.) The bbox and the SDF index come from the
                // same registry, so a clipped item's `clip_rect` and `clip_index` always
                // describe the same mask.
                let (clip_shapes, clip_index_of, clip_bboxes) =
                    build_clip_shapes(&frame.overlays.shapes, ppp, [vp_w, vp_h], view, proj);
                let stamp_clip = |batch: &mut [crate::resources::OverlayTextVertex],
                                  clip_id: Option<u32>| {
                    if let Some(id) = clip_id {
                        if let Some(&ci) = clip_index_of.get(&id) {
                            let cr = clip_bboxes[ci as usize];
                            for v in batch.iter_mut() {
                                v.clip_rect = cr;
                                v.clip_index = ci as f32;
                            }
                        }
                    }
                };

                // --- Polylines (between shapes and labels in z order) ---
                // The gizmo's rings, plane quads, cube faces, and centre handle
                // ride along here as generated polylines.
                for poly in frame
                    .overlays
                    .polylines
                    .iter()
                    .chain(gizmo_polylines.iter())
                {
                    if poly.points.len() < 2 || poly.opacity <= 0.0 {
                        continue;
                    }
                    // Resolve the anchor to a screen-pixel offset and draw the
                    // path there; a culled world anchor skips it. The offset is
                    // baked into a translated copy so the stroke and fill paths
                    // (which read `poly.points`) both see anchored coordinates.
                    let Some(offset) = poly.resolve_offset([vp_w, vp_h], view, proj) else {
                        continue;
                    };
                    let translated_storage;
                    let poly: &crate::renderer::types::OverlayPolylineItem = if offset == [0.0, 0.0]
                    {
                        poly
                    } else {
                        let mut ts = poly.clone();
                        ts.points = poly
                            .points
                            .iter()
                            .map(|p| [p[0] + offset[0], p[1] + offset[1]])
                            .collect();
                        translated_storage = ts;
                        &translated_storage
                    };
                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                    if poly.closed && poly.texture.is_none() {
                        if let Some(fill) = &poly.fill {
                            emit_filled_polyline(
                                &mut batch,
                                &poly.points,
                                fill,
                                poly.opacity,
                                vp_w,
                                vp_h,
                            );
                        }
                    }
                    if poly.thickness > 0.0 {
                        let mut colour = poly.colour;
                        colour[3] *= poly.opacity;
                        emit_polyline_stroke(&mut batch, poly, colour, vp_w, vp_h);
                    }
                    if !batch.is_empty() {
                        batches.push((poly.z_order, batch));
                    }
                }

                // --- Vector shapes (tessellated fills + outline strokes) ---
                // A vector shape has no SDF, so it draws here through the same
                // triangle-fill pipeline as filled polylines rather than the
                // SDF shape pass. Fill, gradient, opacity, clip, and the border
                // outline carry over; SDF-derived effects (soft shadows, the
                // distance-band border) do not apply.
                for shape in &frame.overlays.shapes {
                    let crate::renderer::types::OverlayShape::Vector {
                        subpaths,
                        fill_rule,
                    } = &shape.shape
                    else {
                        continue;
                    };
                    // Mask-only shapes contribute a clip rect, not a draw; skip
                    // fully transparent ones. A `texture` set on a vector shape
                    // is ignored (documented on the variant): the fill still
                    // draws.
                    if shape.clip_mask_id.is_some() || shape.opacity <= 0.0 {
                        continue;
                    }
                    // Resolve the anchor to an absolute top-left so anchored
                    // vector shapes tessellate where they draw; a culled world
                    // anchor skips the shape. Vector shapes take no animation
                    // tracks, so the raw item resolves directly.
                    let Some(tl) = shape.resolve_top_left([vp_w, vp_h], view, proj) else {
                        continue;
                    };
                    let mut owned = shape.clone();
                    owned.position = tl;
                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                    emit_vector_shape(&mut batch, &owned, subpaths, *fill_rule, vp_w, vp_h);
                    stamp_clip(&mut batch, owned.clip_id);
                    if !batch.is_empty() {
                        batches.push((shape.z_order, batch));
                    }
                }

                // --- Labels ---
                for label in &frame.overlays.labels {
                    if label.text.is_empty() || label.opacity <= 0.0 {
                        continue;
                    }

                    let Some(anchor_px) = crate::renderer::types::resolve_anchor_origin(
                        &label.anchor,
                        [vp_w, vp_h],
                        view,
                        proj,
                    ) else {
                        continue;
                    };

                    let opacity = label.opacity.clamp(0.0, 1.0);

                    let layout = if let Some(max_w) = label.max_width {
                        self.resources.content.glyph_atlas.layout_text_wrapped(
                            &label.text,
                            label.font_size,
                            label.font,
                            max_w,
                            ppp,
                            device,
                        )
                    } else {
                        self.resources.content.glyph_atlas.layout_text(
                            &label.text,
                            label.font_size,
                            label.font,
                            ppp,
                            device,
                        )
                    };

                    let font_index = label.font.map_or(0, |h| h.0);
                    let ascent = self
                        .resources
                        .content
                        .glyph_atlas
                        .font_ascent(font_index, label.font_size);

                    let align_offset = match label.align_x {
                        crate::renderer::types::AnchorX::Left => label.anchor_padding,
                        crate::renderer::types::AnchorX::Middle => -layout.total_width * 0.5,
                        crate::renderer::types::AnchorX::Right => {
                            -layout.total_width - label.anchor_padding
                        }
                    };

                    let align_offset_y = match label.align_y {
                        crate::renderer::types::AnchorY::Top => 0.0,
                        crate::renderer::types::AnchorY::Middle => -layout.height * 0.5,
                        crate::renderer::types::AnchorY::Bottom => -layout.height,
                    };

                    let text_x = anchor_px[0] + align_offset + label.position[0];
                    let text_y = anchor_px[1] + align_offset_y + label.position[1];

                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                    if label.background {
                        let pad = label.padding;
                        let bx0 = text_x - pad;
                        let by0 = text_y - pad;
                        let bx1 = text_x + layout.total_width + pad;
                        let by1 = text_y + layout.height + pad;
                        let bg_colour = apply_opacity(label.background_colour, opacity);
                        if label.border_radius > 0.0 {
                            emit_rounded_quad(
                                &mut batch,
                                bx0,
                                by0,
                                bx1,
                                by1,
                                label.border_radius,
                                bg_colour,
                                vp_w,
                                vp_h,
                            );
                        } else {
                            emit_solid_quad(&mut batch, bx0, by0, bx1, by1, bg_colour, vp_w, vp_h);
                        }
                    }

                    if label.leader_line {
                        if let crate::renderer::types::OverlayAnchor::World(wa) = label.anchor {
                            let world_px = project_to_screen(wa, view, proj, vp_w, vp_h);
                            if let Some(wp) = world_px {
                                emit_line_quad(
                                    &mut batch,
                                    wp[0],
                                    wp[1],
                                    text_x,
                                    text_y + layout.height * 0.5,
                                    1.5,
                                    apply_opacity(label.leader_colour, opacity),
                                    vp_w,
                                    vp_h,
                                );
                            }
                        }
                    }

                    let text_colour = apply_opacity(label.colour, opacity);
                    // The label origin is the top-left of the text box; add the
                    // ascent to reach the first baseline the quads are relative to.
                    emit_glyph_quads(
                        &mut batch,
                        &layout.quads,
                        text_x,
                        text_y + ascent,
                        text_colour,
                        vp_w,
                        vp_h,
                    );

                    stamp_clip(&mut batch, label.clip_id);
                    batches.push((label.z_order, batch));
                }

                // --- Glyph runs (pre-positioned glyphs, drawn as given) ---
                for run in &frame.overlays.glyph_runs {
                    if run.glyphs.is_empty() || run.opacity <= 0.0 {
                        continue;
                    }

                    // Resolve the anchor origin; skip the run when a world anchor
                    // is culled. Alignment shifts the whole run by its glyph-extent
                    // box, so default Left/Top leaves the authored positions.
                    let Some(origin) = crate::renderer::types::resolve_anchor_origin(
                        &run.anchor,
                        [vp_w, vp_h],
                        view,
                        proj,
                    ) else {
                        continue;
                    };
                    let (mut min_x, mut min_y, mut max_x, mut max_y) =
                        (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
                    for g in &run.glyphs {
                        min_x = min_x.min(g.x);
                        min_y = min_y.min(g.y);
                        max_x = max_x.max(g.x);
                        max_y = max_y.max(g.y);
                    }
                    let run_x =
                        origin[0] + run.position[0] + run.align_x.align_shift(max_x - min_x);
                    let run_y =
                        origin[1] + run.position[1] + run.align_y.align_shift(max_y - min_y);

                    let opacity = run.opacity.clamp(0.0, 1.0);
                    // Each glyph carries its tint through layout so per-glyph
                    // colours stay aligned with the quads after zero-area skips.
                    // Glyphs without a per-glyph entry fall back to the run colour.
                    let quads = self.resources.content.glyph_atlas.layout_glyph_run(
                        run.glyphs.iter().enumerate().map(|(i, g)| {
                            let colour = run.colours.get(i).copied().unwrap_or(run.colour);
                            (g.glyph_id, g.x, g.y, apply_opacity(colour, opacity))
                        }),
                        run.font_size,
                        run.font,
                        ppp,
                        device,
                    );
                    if quads.is_empty() {
                        continue;
                    }

                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                    // Positions are already relative to the run origin, so the
                    // resolved origin is the only offset added; no ascent, unlike
                    // labels.
                    emit_glyph_quads_colored(&mut batch, &quads, run_x, run_y, vp_w, vp_h);

                    stamp_clip(&mut batch, run.clip_id);
                    batches.push((run.z_order, batch));
                }

                // Upload atlas if new glyphs were rasterized.
                self.resources.content.glyph_atlas.upload_if_dirty(queue);

                // Stable sort preserves submission order for equal z_order values.
                batches.sort_by_key(|(z, _)| *z);
                // Flatten into one buffer, recording a draw segment per distinct
                // z_order run so this batch can interleave with other families.
                use crate::renderer::overlay_draw_order::OverlayDrawSegment;
                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                for (z, v) in batches {
                    let start = verts.len() as u32;
                    verts.extend(v);
                    if self.overlay_uses_zorder {
                        OverlayDrawSegment::push_text(
                            &mut self.overlay_draw_segments,
                            z,
                            start,
                            verts.len() as u32 - start,
                        );
                    }
                }

                // Retained overlay groups: resolve each submitted group to its
                // compiled buffer, build its per-frame instance (translate,
                // opacity, and outer clip in framebuffer pixels), and record a
                // draw. Slot 0 of the instance buffer is the identity that every
                // immediate draw reads.
                let mut instances = vec![crate::resources::OverlayInstance::IDENTITY];
                // Collected shape-stream draws, finished after the instance buffer
                // exists: (shape_vbuf, count, shadow_buf, instance_index, z_order).
                let mut pending_shapes: Vec<(
                    crate::gpu::Buffer,
                    u32,
                    crate::gpu::Buffer,
                    u32,
                    i32,
                )> = Vec::new();
                for r in &frame.overlays.retained {
                    // Re-emit the group first if its baked glyph UVs went stale
                    // (atlas grew or pixels_per_point changed); cheap no-op otherwise.
                    self.reemit_overlay_geometry_if_stale(device, queue, r.id, ppp);
                    let (text, shape, anchor) =
                        match self.resources.content.overlay_geometry.get(r.id) {
                            Some(c) => {
                                let text = (c.vertex_count > 0)
                                    .then(|| (c.vertex_buf.clone(), c.vertex_count));
                                let shape = match (&c.shape_vertex_buf, &c.shadow_buf) {
                                    (Some(sv), Some(sh)) if c.shape_vertex_count > 0 => {
                                        Some((sv.clone(), c.shape_vertex_count, sh.clone()))
                                    }
                                    _ => None,
                                };
                                (text, shape, c.anchor)
                            }
                            None => continue,
                        };
                    if text.is_none() && shape.is_none() {
                        continue;
                    }
                    // A compiled label carries its own anchor: resolve it to a
                    // screen origin (a viewport corner, or a world point projected
                    // through the camera) and fold it into the translate. The
                    // submission's translate composes on top (scroll/nudge). A world
                    // anchor that is culled skips the group for this frame.
                    let translate = match anchor {
                        Some(a) => {
                            let Some(origin) = crate::renderer::types::resolve_anchor_origin(
                                &a,
                                [vp_w, vp_h],
                                view,
                                proj,
                            ) else {
                                continue;
                            };
                            [origin[0] + r.translate[0], origin[1] + r.translate[1]]
                        }
                        None => r.translate,
                    };
                    let clip_rect = if r.clip_rect == [0.0, 0.0, 0.0, 0.0] {
                        [0.0, 0.0, 0.0, 0.0]
                    } else {
                        [
                            r.clip_rect[0] * ppp,
                            r.clip_rect[1] * ppp,
                            r.clip_rect[2] * ppp,
                            r.clip_rect[3] * ppp,
                        ]
                    };
                    let instance_index = instances.len() as u32;
                    instances.push(crate::resources::OverlayInstance {
                        translate,
                        opacity: r.opacity,
                        _pad: 0.0,
                        clip_rect,
                    });
                    if let Some((vbuf, vcount)) = text {
                        let draw_index = self.overlay_retained_draws.len() as u32;
                        self.overlay_retained_draws.push(
                            crate::renderer::overlay_buffers::RetainedDraw {
                                vertex_buf: vbuf,
                                vertex_count: vcount,
                                instance_index,
                            },
                        );
                        crate::renderer::overlay_draw_order::OverlayDrawSegment::push_retained(
                            &mut self.overlay_draw_segments,
                            r.z_order,
                            draw_index,
                        );
                    }
                    if let Some((svbuf, svcount, shbuf)) = shape {
                        pending_shapes.push((svbuf, svcount, shbuf, instance_index, r.z_order));
                    }
                }

                if !verts.is_empty()
                    || !self.overlay_retained_draws.is_empty()
                    || !pending_shapes.is_empty()
                {
                    let vertex_buf = self.overlay_text_vbuf.write(device, queue, &verts);
                    self.ensure_overlay_viewport_buf(device, queue, vp_w, vp_h);
                    let clip_buf = upload_clip_buffer(device, queue, &clip_shapes);
                    let instances_buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some("overlay_instances_buf"),
                        size: std::mem::size_of_val(&instances[..]) as u64,
                        usage: crate::gpu::BufferUsages::STORAGE
                            | crate::gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    queue.write_buffer(&instances_buf, 0, bytemuck::cast_slice(&instances));
                    // Share the instance buffer with the shape pass (immediate shapes
                    // read the identity at slot 0; retained shape draws use their slot).
                    self.overlay_instances_buf = Some(instances_buf.clone());
                    self.overlay_instances_ready = true;

                    // Finish retained shape-stream draws now that the instance
                    // buffer exists: build each group's shape bind group (its shadow
                    // buffer plus the shared clip / viewport / instances) and record
                    // its draw + segment.
                    if !pending_shapes.is_empty() {
                        self.resources.ensure_overlay_shape_pipeline(device);
                        let vp_buf = self.overlay_viewport_buf.as_ref().unwrap();
                        if let Some(sh_bgl) = self.resources.overlay_shape.shadow_bgl.as_ref() {
                            for (svbuf, svcount, shbuf, instance_index, z) in
                                pending_shapes.drain(..)
                            {
                                let bind_group =
                                    device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                                        label: Some("overlay_retained_shape_bg"),
                                        layout: sh_bgl,
                                        entries: &[
                                            crate::gpu::BindGroupEntry {
                                                binding: 0,
                                                resource: shbuf.as_entire_binding(),
                                            },
                                            crate::gpu::BindGroupEntry {
                                                binding: 1,
                                                resource: clip_buf.as_entire_binding(),
                                            },
                                            crate::gpu::BindGroupEntry {
                                                binding: 2,
                                                resource: vp_buf.as_entire_binding(),
                                            },
                                            crate::gpu::BindGroupEntry {
                                                binding: 3,
                                                resource: instances_buf.as_entire_binding(),
                                            },
                                        ],
                                    });
                                let draw_index = self.overlay_retained_shape_draws.len() as u32;
                                self.overlay_retained_shape_draws.push(
                                    crate::renderer::overlay_buffers::RetainedShapeDraw {
                                        vertex_buf: svbuf,
                                        vertex_count: svcount,
                                        bind_group,
                                        instance_index,
                                    },
                                );
                                crate::renderer::overlay_draw_order::OverlayDrawSegment::push_retained_shape(
                                    &mut self.overlay_draw_segments,
                                    z,
                                    draw_index,
                                );
                            }
                        }
                    }
                    let bgl = self.resources.overlay_text.bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text.sampler.as_ref().unwrap();
                    let vp_buf = self.overlay_viewport_buf.as_ref().unwrap();
                    let bind_group = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("overlay_label_bg"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::gpu::BindingResource::TextureView(
                                    &self.resources.content.glyph_atlas.view,
                                ),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: crate::gpu::BindingResource::Sampler(sampler),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 2,
                                resource: clip_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 3,
                                resource: vp_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 4,
                                resource: instances_buf.as_entire_binding(),
                            },
                        ],
                    });
                    self.label_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        vertex_count: verts.len() as u32,
                        bind_group,
                        _clip_buf: clip_buf,
                        _instances_buf: instances_buf,
                    });
                }
            }
        }
    }

    pub(super) fn prepare_overlay_shapes(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
    ) {
        // ------------------------------------------------------------------
        // SDF overlay shapes
        // ------------------------------------------------------------------
        self.overlay_shape_gpu_data = None;
        // The gizmo's arrow shafts and cone heads are generated as overlay
        // shapes; they merge into the shape batch here (see `gizmo_overlay`).
        let (gizmo_shapes, _) = self.gizmo_overlay_items(frame);
        // The bottom-left orientation indicator draws as overlay shapes too.
        let axes_shapes = self.axes_overlay_items(frame);
        let has_textured_polyline_fill = frame
            .overlays
            .polylines
            .iter()
            .any(|p| p.closed && p.texture.is_some() && p.opacity > 0.0 && p.points.len() >= 3);
        if !frame.overlays.shapes.is_empty()
            || !gizmo_shapes.is_empty()
            || !axes_shapes.is_empty()
            || has_textured_polyline_fill
        {
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            if vp_w > 0.0 && vp_h > 0.0 {
                self.ensure_overlay_viewport_buf(device, queue, vp_w, vp_h);
                self.ensure_overlay_instances_default(device, queue);
                let mut sorted: Vec<&crate::renderer::types::OverlayShapeItem> = frame
                    .overlays
                    .shapes
                    .iter()
                    .chain(gizmo_shapes.iter())
                    .chain(axes_shapes.iter())
                    .collect();
                sorted.sort_by_key(|s| s.z_order);

                let has_solid = sorted
                    .iter()
                    .any(|s| s.texture.is_none() && s.backdrop_blur <= 0.0);
                let has_tex =
                    sorted.iter().any(|s| s.texture.is_some()) || has_textured_polyline_fill;
                let has_blur = sorted
                    .iter()
                    .any(|s| s.backdrop_blur > 0.0 && s.texture.is_none());
                if has_solid {
                    self.resources.ensure_overlay_shape_pipeline(device);
                }
                if has_tex || has_blur {
                    self.resources.ensure_overlay_shape_tex_pipeline(device);
                }
                if has_blur {
                    self.resources.ensure_backdrop_blur_pipeline(device);
                    self.resources.ensure_dyn_res_pipeline(device);
                }

                let mut solid_verts: Vec<crate::resources::OverlayShapeVertex> = Vec::new();
                // Stacked shadow layers for solid shapes, shared across the
                // whole frame. Each solid shape references a contiguous run
                // (outer layers first, then inner) via its `shadow_index`.
                let mut shadow_layers: Vec<crate::resources::OverlayShadowLayerGpu> = Vec::new();
                // One vertex list per unique texture ID, in order of first appearance.
                let mut tex_groups: Vec<(
                    crate::renderer::types::OverlayTextureId,
                    Vec<crate::resources::OverlayShapeTexVertex>,
                )> = Vec::new();
                // Lowest z_order contributing to each tex group, parallel to
                // `tex_groups`. A textured batch draws as one unit, so it orders
                // against other families at its frontmost (lowest-z) shape.
                let mut tex_group_z: Vec<i32> = Vec::new();
                // One entry per drawn textured shape (and textured polyline
                // fill), in z-sorted emission order: `(z_order, group_idx,
                // vertex_start, vertex_count)` where the range indexes into that
                // group's vertex buffer. Each becomes its own draw segment so a
                // solid shape can layer between two shapes sharing a texture,
                // instead of the whole texture group collapsing to its lowest z.
                let mut tex_shape_segs: Vec<(i32, usize, u32, u32)> = Vec::new();
                // Blur backdrop vertices (share the tex vertex layout with screen UVs).
                let mut blur_verts: Vec<crate::resources::OverlayShapeTexVertex> = Vec::new();
                let mut max_blur_radius: f32 = 0.0;

                let overlay_time = frame.overlays.time;

                // The SDF clip registry, plus each mask's bounding box (framebuffer
                // pixels) for the shader's cheap pre-reject. Built from
                // `frame.overlays.shapes` (not `sorted`) so the mask indices match
                // exactly what the label pass assigned, since text and shapes reference
                // the same masks by index. The bbox is looked up by that same index, so
                // `clip_rect` and `clip_index` always describe the same mask.
                let ppp = frame.camera.pixels_per_point;
                let view = &frame.camera.render_camera.view;
                let proj = &frame.camera.render_camera.projection;
                let (clip_shapes, clip_index_of, clip_bboxes) =
                    build_clip_shapes(&frame.overlays.shapes, ppp, [vp_w, vp_h], view, proj);

                for shape_orig in &sorted {
                    // Mask-only shapes contribute a clip rectangle but are
                    // not drawn themselves.
                    if shape_orig.clip_mask_id.is_some() {
                        continue;
                    }
                    // Vector shapes have no analytic SDF; the solid/textured
                    // paths below cannot draw them. They are tessellated to a
                    // triangle fill on a separate path (not yet wired), so skip
                    // them here rather than encoding them as a rect quad.
                    if matches!(
                        &shape_orig.shape,
                        crate::renderer::types::OverlayShape::Vector { .. }
                    ) {
                        continue;
                    }
                    // Clone so per-frame animation overrides are local and
                    // the input frame data stays untouched.
                    let mut owned: crate::renderer::types::OverlayShapeItem = (*shape_orig).clone();
                    if let Some(track) = owned.animations.opacity {
                        // Multi-channel opacity track takes precedence over
                        // the legacy `animation` field.
                        owned.opacity = track.sample(overlay_time);
                        owned.animation = crate::renderer::types::OverlayAnimation::None;
                    }
                    if let Some(track) = owned.animations.position {
                        owned.position = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.size {
                        owned.size = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.fill {
                        if let crate::renderer::types::OverlayFill::Solid(_) = owned.fill {
                            owned.fill = crate::renderer::types::OverlayFill::Solid(
                                track.sample(overlay_time),
                            );
                        }
                    }
                    if let Some(track) = owned.animations.border {
                        owned.border_colour = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.rotation {
                        owned.rotation = track.sample(overlay_time);
                    }
                    // Path tracks override the matching linear track when
                    // both are set. `opacity_path` also takes precedence
                    // over the legacy `animation` field.
                    if let Some(track) = owned.animations.opacity_path.clone() {
                        owned.opacity = track.sample(overlay_time);
                        owned.animation = crate::renderer::types::OverlayAnimation::None;
                    }
                    if let Some(track) = owned.animations.position_path.clone() {
                        owned.position = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.size_path.clone() {
                        owned.size = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.fill_path.clone() {
                        if let crate::renderer::types::OverlayFill::Solid(_) = owned.fill {
                            owned.fill = crate::renderer::types::OverlayFill::Solid(
                                track.sample(overlay_time),
                            );
                        }
                    }
                    if let Some(track) = owned.animations.border_path.clone() {
                        owned.border_colour = track.sample(overlay_time);
                    }
                    if let Some(track) = owned.animations.rotation_path.clone() {
                        owned.rotation = track.sample(overlay_time);
                    }
                    // Resolve the anchor origin + animated position + alignment
                    // to an absolute top-left, then draw the shape as if it were
                    // positioned there. A culled world anchor skips the shape for
                    // this frame. Everything below (fill, texture, clip, shadow,
                    // rotation pivot) reads `position`, so this one resolution
                    // covers all of them. The `animations.position` track has
                    // already been applied above, so it drives the nudge that is
                    // layered on the resolved origin here.
                    let Some(resolved_tl) = owned.resolve_top_left([vp_w, vp_h], view, proj) else {
                        continue;
                    };
                    owned.position = resolved_tl;
                    let shape = &owned;
                    // Resolve animation to final opacity.
                    let resolved_opacity = match shape.animation {
                        crate::renderer::types::OverlayAnimation::None => shape.opacity,
                        crate::renderer::types::OverlayAnimation::FadeIn {
                            start_time,
                            duration,
                        } => {
                            let t = ((overlay_time - start_time) as f32 / duration.max(1e-6))
                                .clamp(0.0, 1.0);
                            shape.opacity * t
                        }
                        crate::renderer::types::OverlayAnimation::FadeOut {
                            start_time,
                            duration,
                        } => {
                            let t = ((overlay_time - start_time) as f32 / duration.max(1e-6))
                                .clamp(0.0, 1.0);
                            shape.opacity * (1.0 - t)
                        }
                        crate::renderer::types::OverlayAnimation::Pulse { start_time, period } => {
                            let t = ((overlay_time - start_time) as f32) / period.max(1e-6);
                            let wave = (t * std::f32::consts::TAU).sin() * 0.5 + 0.5;
                            shape.opacity * wave
                        }
                    };

                    if resolved_opacity <= 0.0 {
                        continue;
                    }

                    let hw = shape.size[0] * 0.5;
                    let hh = shape.size[1] * 0.5;
                    let cx = shape.position[0] + hw;
                    let cy = shape.position[1] + hh;

                    // Outer-shadow extent that reaches past the shape edge.
                    // Inner shadows stay inside the shape, so they need no quad
                    // padding. Both the legacy single shadow and the stacked
                    // `shadows` layers contribute.
                    let mut shadow_pad = if shape.shadow_radius > 0.0 {
                        shape.shadow_radius
                            + shape.shadow_offset[0]
                                .abs()
                                .max(shape.shadow_offset[1].abs())
                    } else {
                        0.0
                    };
                    for l in &shape.shadows {
                        let e = l.radius + l.offset[0].abs().max(l.offset[1].abs());
                        shadow_pad = shadow_pad.max(e);
                    }

                    // Extra quad expansion for shapes whose stroke extends
                    // beyond the item's position/size bounding box.
                    let extra_expand = match &shape.shape {
                        crate::renderer::types::OverlayShape::Line { thickness, .. } => {
                            thickness * 0.5
                        }
                        _ => 0.0,
                    };

                    // Base half-extents including the border, before rotation.
                    let bx = hw + shape.border_width + extra_expand;
                    let by = hh + shape.border_width + extra_expand;

                    // Rotation about the pivot moves the shape's corners out of
                    // the axis-aligned box, so grow the quad to the AABB of the
                    // rotated border box (matching how the fragment shader maps
                    // the rotated shape into the quad). Without this a rotated
                    // rect, capsule, or off-centre pivot clips against its own
                    // quad. A zero rotation with a zero pivot leaves bx/by
                    // unchanged.
                    let (rx, ry) = if shape.rotation != 0.0 {
                        let c = shape.rotation.cos();
                        let s = shape.rotation.sin();
                        let piv = shape.rotation_pivot;
                        let mut mx = 0.0_f32;
                        let mut my = 0.0_f32;
                        for cxp in [-bx, bx] {
                            for cyp in [-by, by] {
                                let dx = cxp - piv[0];
                                let dy = cyp - piv[1];
                                let rxp = c * dx - s * dy + piv[0];
                                let ryp = s * dx + c * dy + piv[1];
                                mx = mx.max(rxp.abs());
                                my = my.max(ryp.abs());
                            }
                        }
                        (mx, my)
                    } else {
                        (bx, by)
                    };

                    let ex = rx + shadow_pad + 1.0; // +1 for AA
                    let ey = ry + shadow_pad + 1.0;

                    // Encode shape type and radii.
                    let (shape_type, radii) = match &shape.shape {
                        crate::renderer::types::OverlayShape::Rect { corner_radius } => {
                            let r = corner_radius.min(hw).min(hh).max(0.0);
                            (0.0, [r, r, r, r])
                        }
                        crate::renderer::types::OverlayShape::RoundedRect { radii: r } => {
                            // Input: [tl, tr, br, bl].
                            // Shader expects iq convention: [tr, br, bl, tl].
                            let clamped = [
                                r[1].min(hw).min(hh).max(0.0), // top-right
                                r[2].min(hw).min(hh).max(0.0), // bottom-right
                                r[3].min(hw).min(hh).max(0.0), // bottom-left
                                r[0].min(hw).min(hh).max(0.0), // top-left
                            ];
                            (0.0, clamped)
                        }
                        crate::renderer::types::OverlayShape::Circle => (1.0, [0.0; 4]),
                        crate::renderer::types::OverlayShape::Ellipse => (2.0, [0.0; 4]),
                        crate::renderer::types::OverlayShape::Capsule => (3.0, [0.0; 4]),
                        crate::renderer::types::OverlayShape::Ring { inner_radius_frac } => {
                            (4.0, [inner_radius_frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::Arc {
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
                        crate::renderer::types::OverlayShape::Triangle { direction } => {
                            let dir_f = match direction {
                                crate::renderer::types::TriangleDirection::Up => 0.0,
                                crate::renderer::types::TriangleDirection::Down => 1.0,
                                crate::renderer::types::TriangleDirection::Left => 2.0,
                                crate::renderer::types::TriangleDirection::Right => 3.0,
                            };
                            (6.0, [dir_f, 0.0, 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::Line { thickness, cap } => {
                            let cap_f = match cap {
                                crate::renderer::types::LineCap::Round => 0.0,
                                crate::renderer::types::LineCap::Square => 1.0,
                            };
                            (7.0, [thickness * 0.5, cap_f, 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::Star {
                            points,
                            inner_radius_frac,
                        } => {
                            let n = (*points).max(3) as f32;
                            (8.0, [n, inner_radius_frac.clamp(0.0, 1.0), 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::RegularPolygon { sides } => {
                            let n = (*sides).max(3) as f32;
                            (9.0, [n, 0.0, 0.0, 0.0])
                        }
                        crate::renderer::types::OverlayShape::Cross { arm_width_frac } => {
                            (10.0, [arm_width_frac.clamp(0.0, 1.0), 0.0, 0.0, 0.0])
                        }
                        // Skipped earlier in the loop; this keeps the match
                        // exhaustive without drawing the vector shape as a rect.
                        crate::renderer::types::OverlayShape::Vector { .. } => (0.0, [0.0; 4]),
                        // OverlayShape is non_exhaustive; encode unknown shapes as a plain rect.
                        _ => (0.0, [0.0; 4]),
                    };

                    // Resolve the fill into four colour stops + positions +
                    // a gradient type. Solid and 2-stop variants set
                    // count = 2 and use only stops[0..2]; multi-stop fills
                    // pack up to OVERLAY_MAX_GRADIENT_STOPS stops.
                    let mut stop_colours = [[0.0f32; 4]; 4];
                    let mut stop_positions = [0.0_f32, 1.0, 1.0, 1.0];
                    let stop_count: f32;
                    let gradient_params = match &shape.fill {
                        crate::renderer::types::OverlayFill::Solid(c) => {
                            stop_colours[0] = *c;
                            stop_colours[1] = *c;
                            stop_count = 0.0;
                            [0.0_f32, 0.0]
                        }
                        crate::renderer::types::OverlayFill::LinearGradient {
                            start_colour,
                            end_colour,
                            angle,
                        } => {
                            stop_colours[0] = *start_colour;
                            stop_colours[1] = *end_colour;
                            stop_count = 2.0;
                            [1.0_f32, *angle]
                        }
                        crate::renderer::types::OverlayFill::RadialGradient {
                            centre_colour,
                            edge_colour,
                        } => {
                            stop_colours[0] = *centre_colour;
                            stop_colours[1] = *edge_colour;
                            stop_count = 2.0;
                            [2.0_f32, 0.0]
                        }
                        crate::renderer::types::OverlayFill::ConicalGradient {
                            start_colour,
                            end_colour,
                            offset_angle,
                        } => {
                            stop_colours[0] = *start_colour;
                            stop_colours[1] = *end_colour;
                            stop_count = 2.0;
                            [3.0_f32, *offset_angle]
                        }
                        crate::renderer::types::OverlayFill::LinearGradientMulti {
                            stops,
                            angle,
                        } => {
                            stop_count = pack_stops(stops, &mut stop_colours, &mut stop_positions);
                            [1.0_f32, *angle]
                        }
                        crate::renderer::types::OverlayFill::RadialGradientMulti { stops } => {
                            stop_count = pack_stops(stops, &mut stop_colours, &mut stop_positions);
                            [2.0_f32, 0.0]
                        }
                        crate::renderer::types::OverlayFill::ConicalGradientMulti {
                            stops,
                            offset_angle,
                        } => {
                            stop_count = pack_stops(stops, &mut stop_colours, &mut stop_positions);
                            [3.0_f32, *offset_angle]
                        }
                        // OverlayFill is non_exhaustive; render unknown fills as a flat no-op gradient.
                        _ => {
                            stop_count = 0.0;
                            [0.0_f32, 0.0]
                        }
                    };
                    let start_colour = stop_colours[0];
                    let end_colour = stop_colours[1];
                    // Apply opacity to every stop (stops[0..4]) so multi-stop
                    // gradients fade as a whole when the item's opacity changes.
                    for colour in &mut stop_colours {
                        colour[3] *= resolved_opacity;
                    }
                    let fc = stop_colours[0];
                    let fc2 = stop_colours[1];
                    let _ = (start_colour, end_colour);
                    let mut bc = shape.border_colour;
                    bc[3] *= resolved_opacity;

                    let half_size = [hw, hh];

                    let mut sc = shape.shadow_colour;
                    sc[3] *= resolved_opacity;
                    let border_mode_f = match shape.border_mode {
                        crate::renderer::types::BorderMode::Inset => 0.0,
                        crate::renderer::types::BorderMode::Outer => 1.0,
                        crate::renderer::types::BorderMode::Center => 2.0,
                    };
                    // Pack the inset-shadow flag alongside border_mode in
                    // shadow_params.w. The shader decodes via (combined % 3)
                    // for border_mode and (combined >= 3) for inset.
                    let inset_flag = if shape.shadow_inset { 3.0 } else { 0.0 };
                    let shadow_params = [
                        shape.shadow_radius,
                        shape.shadow_offset[0],
                        shape.shadow_offset[1],
                        border_mode_f + inset_flag,
                    ];

                    // Emit 6 vertices (two triangles) for the bounding quad.
                    let corners_px = [
                        (cx - ex, cy - ey, -ex, -ey),
                        (cx + ex, cy - ey, ex, -ey),
                        (cx + ex, cy + ey, ex, ey),
                        (cx - ex, cy - ey, -ex, -ey),
                        (cx + ex, cy + ey, ex, ey),
                        (cx - ex, cy + ey, -ex, ey),
                    ];

                    // Resolve the clip mask for this shape (shared by the solid,
                    // textured, and blur branches). Missing or unmatched ids fall
                    // through to a -1 index and an all-zero rect, which the shaders
                    // treat as no clipping. The bbox and index come from the same
                    // registry entry, so they never disagree.
                    let clip_index_i = shape
                        .clip_id
                        .and_then(|id| clip_index_of.get(&id).copied())
                        .unwrap_or(-1);
                    let clip_rect = if clip_index_i >= 0 {
                        clip_bboxes[clip_index_i as usize]
                    } else {
                        [0.0, 0.0, 0.0, 0.0]
                    };
                    let clip_index = clip_index_i as f32;

                    if let Some(tex_id) = shape.texture {
                        // Find or create a group for this texture ID.
                        let group_idx = tex_groups
                            .iter()
                            .position(|(id, _)| *id == tex_id)
                            .unwrap_or_else(|| {
                                tex_groups.push((tex_id, Vec::new()));
                                tex_group_z.push(i32::MAX);
                                tex_groups.len() - 1
                            });
                        tex_group_z[group_idx] = tex_group_z[group_idx].min(shape.z_order);
                        let group_verts = &mut tex_groups[group_idx].1;
                        let tex_seg_start = group_verts.len() as u32;

                        // UV maps local_pos to [0,1] over the shape content area.
                        // hw/hh safe-guarded so we never divide by zero.
                        let hw_s = if hw > 0.0 { hw } else { 1.0 };
                        let hh_s = if hh > 0.0 { hh } else { 1.0 };

                        // 9-slice: convert pixel insets to texture UV ratios
                        // (using the bound texture's size) and to shape-fraction
                        // ratios for the shader's piecewise UV remap.
                        let (nine_uv, nine_frac, nine_extras_yzw) =
                            if let Some(ns) = shape.nine_slice {
                                let tex_size = self
                                    .resources
                                    .content
                                    .overlay_textures
                                    .get(tex_id)
                                    .map(|t| t.texture.size())
                                    .map(|s| (s.width as f32, s.height as f32))
                                    .unwrap_or((1.0, 1.0));
                                let tw = tex_size.0.max(1.0);
                                let th = tex_size.1.max(1.0);
                                let shape_w = (shape.size[0]).max(1.0);
                                let shape_h = (shape.size[1]).max(1.0);
                                let inset_uv = [
                                    (ns.insets_px[0] / th).clamp(0.0, 0.5),
                                    (ns.insets_px[1] / tw).clamp(0.0, 0.5),
                                    (ns.insets_px[2] / th).clamp(0.0, 0.5),
                                    (ns.insets_px[3] / tw).clamp(0.0, 0.5),
                                ];
                                let inset_frac = [
                                    (ns.insets_px[0] / shape_h).clamp(0.0, 0.5),
                                    (ns.insets_px[1] / shape_w).clamp(0.0, 0.5),
                                    (ns.insets_px[2] / shape_h).clamp(0.0, 0.5),
                                    (ns.insets_px[3] / shape_w).clamp(0.0, 0.5),
                                ];
                                let centre = tile_mode_to_f(ns.centre_mode);
                                let edge = tile_mode_to_f(ns.edge_mode);
                                (inset_uv, inset_frac, [centre, edge, 1.0])
                            } else {
                                ([0.0; 4], [0.0; 4], [0.0, 0.0, 0.0])
                            };

                        let tt = shape.texture_transform;
                        let tt_a = [tt.offset[0], tt.offset[1], tt.scale[0], tt.scale[1]];
                        let tt_b = [
                            tt.rotation,
                            tile_mode_to_f(tt.tile_mode),
                            if tt.flip_x { 1.0 } else { 0.0 },
                            if tt.flip_y { 1.0 } else { 0.0 },
                        ];
                        for (px, py, lx, ly) in corners_px {
                            group_verts.push(crate::resources::OverlayShapeTexVertex {
                                position: overlay_local_px(px, py, vp_w, vp_h),
                                local_pos: [lx, ly],
                                fill_colour: fc,
                                border_colour: bc,
                                half_size,
                                radii,
                                border_width: shape.border_width,
                                shape_type,
                                uv: [(lx + hw_s) / (2.0 * hw_s), (ly + hh_s) / (2.0 * hh_s)],
                                shadow_colour: sc,
                                shadow_params,
                                extras: [
                                    0.0,
                                    nine_extras_yzw[0],
                                    nine_extras_yzw[1],
                                    nine_extras_yzw[2],
                                ],
                                nine_slice_uv: nine_uv,
                                nine_slice_frac: nine_frac,
                                texture_transform_a: tt_a,
                                texture_transform_b: tt_b,
                                clip_index,
                                clip_rect,
                            });
                        }
                        tex_shape_segs.push((
                            shape.z_order,
                            group_idx,
                            tex_seg_start,
                            group_verts.len() as u32 - tex_seg_start,
                        ));
                    } else if shape.backdrop_blur > 0.0 {
                        max_blur_radius = max_blur_radius.max(shape.backdrop_blur);
                        // Blur backdrop: same tex vertex layout but UV is screen-space.
                        for (px, py, lx, ly) in corners_px {
                            blur_verts.push(crate::resources::OverlayShapeTexVertex {
                                position: overlay_local_px(px, py, vp_w, vp_h),
                                local_pos: [lx, ly],
                                fill_colour: fc,
                                border_colour: bc,
                                half_size,
                                radii,
                                border_width: shape.border_width,
                                shape_type,
                                uv: [px / vp_w, py / vp_h],
                                shadow_colour: sc,
                                shadow_params,
                                // extras.x = 1.0 flags the blur path; yzw carry
                                // the backdrop colour filters (saturation,
                                // brightness, hue-shift radians).
                                extras: [
                                    1.0,
                                    shape.backdrop_saturation,
                                    shape.backdrop_brightness,
                                    shape.backdrop_hue_shift,
                                ],
                                nine_slice_uv: [0.0; 4],
                                nine_slice_frac: [0.0; 4],
                                texture_transform_a: [0.0, 0.0, 1.0, 1.0],
                                texture_transform_b: [0.0, 0.0, 0.0, 0.0],
                                // Backdrop blur is composited separately and is
                                // not clipped by a mask.
                                clip_index: -1.0,
                                clip_rect: [0.0; 4],
                            });
                        }
                    } else {
                        // gradient_params is now vec4: [type, angle, stop_count, _pad]
                        let gp4 = [gradient_params[0], gradient_params[1], stop_count, 0.0];

                        // Build the stacked shadow layers for this shape:
                        // outer layers first, then inner, appended to the
                        // shared frame buffer. Prefer the Vec-based lists; fall
                        // back to the single legacy `shadow_*` fields so old
                        // call sites keep working.
                        let base_index = shadow_layers.len();
                        let mut outer_count = 0usize;
                        let mut inner_count = 0usize;
                        let max_layers = crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS;
                        if !shape.shadows.is_empty() {
                            for l in shape.shadows.iter().take(max_layers) {
                                let mut col = l.colour;
                                col[3] *= resolved_opacity;
                                shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                                    colour: col,
                                    params: [l.radius, l.offset[0], l.offset[1], 0.0],
                                });
                                outer_count += 1;
                            }
                        } else if shape.shadow_radius > 0.0 && !shape.shadow_inset {
                            shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                                colour: sc,
                                params: [
                                    shape.shadow_radius,
                                    shape.shadow_offset[0],
                                    shape.shadow_offset[1],
                                    0.0,
                                ],
                            });
                            outer_count += 1;
                        }
                        if !shape.inner_shadows.is_empty() {
                            for l in shape.inner_shadows.iter().take(max_layers) {
                                let mut col = l.colour;
                                col[3] *= resolved_opacity;
                                shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                                    colour: col,
                                    params: [l.radius, l.offset[0], l.offset[1], 1.0],
                                });
                                inner_count += 1;
                            }
                        } else if shape.shadow_radius > 0.0 && shape.shadow_inset {
                            shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                                colour: sc,
                                params: [
                                    shape.shadow_radius,
                                    shape.shadow_offset[0],
                                    shape.shadow_offset[1],
                                    1.0,
                                ],
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
                        let solid_seg_start = solid_verts.len() as u32;
                        for (px, py, lx, ly) in corners_px {
                            solid_verts.push(crate::resources::OverlayShapeVertex {
                                position: overlay_local_px(px, py, vp_w, vp_h),
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
                                clip_rect,
                                clip_index,
                                stop_colour_c: stop_colours[2],
                                stop_colour_d: stop_colours[3],
                                stop_positions,
                            });
                        }
                        if self.overlay_uses_zorder {
                            crate::renderer::overlay_draw_order::OverlayDrawSegment::push_shape(
                                &mut self.overlay_draw_segments,
                                shape.z_order,
                                solid_seg_start,
                                solid_verts.len() as u32 - solid_seg_start,
                            );
                        }
                    }
                }

                for poly in &frame.overlays.polylines {
                    let Some(tex_id) = poly.texture else {
                        continue;
                    };
                    if !poly.closed || poly.opacity <= 0.0 || poly.points.len() < 3 {
                        continue;
                    }
                    // Resolve the anchor to a screen offset and translate the
                    // points, matching the stroke/fill path in the label pass, so
                    // an anchored textured polygon fills where it draws.
                    let Some(offset) = poly.resolve_offset([vp_w, vp_h], view, proj) else {
                        continue;
                    };
                    let translated_storage;
                    let poly: &crate::renderer::types::OverlayPolylineItem = if offset == [0.0, 0.0]
                    {
                        poly
                    } else {
                        let mut ts = poly.clone();
                        ts.points = poly
                            .points
                            .iter()
                            .map(|p| [p[0] + offset[0], p[1] + offset[1]])
                            .collect();
                        translated_storage = ts;
                        &translated_storage
                    };
                    let Some((min, max)) = polyline_bounds(&poly.points) else {
                        continue;
                    };
                    let tris = triangulate_polygon(&poly.points);
                    if tris.is_empty() {
                        continue;
                    }
                    let group_idx = tex_groups
                        .iter()
                        .position(|(id, _)| *id == tex_id)
                        .unwrap_or_else(|| {
                            tex_groups.push((tex_id, Vec::new()));
                            tex_group_z.push(i32::MAX);
                            tex_groups.len() - 1
                        });
                    tex_group_z[group_idx] = tex_group_z[group_idx].min(poly.z_order);
                    let group_verts = &mut tex_groups[group_idx].1;
                    let poly_seg_start = group_verts.len() as u32;
                    let size = [(max[0] - min[0]).max(1e-6), (max[1] - min[1]).max(1e-6)];
                    let centre = [min[0] + size[0] * 0.5, min[1] + size[1] * 0.5];
                    let half_size = [size[0] * 0.5, size[1] * 0.5];
                    let mut tint = match &poly.fill {
                        Some(crate::renderer::types::OverlayFill::Solid(c)) => *c,
                        _ => [1.0, 1.0, 1.0, 1.0],
                    };
                    tint[3] *= poly.opacity;
                    let explicit_uvs = poly
                        .uvs
                        .as_ref()
                        .filter(|uvs| uvs.len() == poly.points.len());
                    let tt = poly.texture_transform;
                    let tt_a = [tt.offset[0], tt.offset[1], tt.scale[0], tt.scale[1]];
                    let tt_b = [
                        tt.rotation,
                        tile_mode_to_f(tt.tile_mode),
                        if tt.flip_x { 1.0 } else { 0.0 },
                        if tt.flip_y { 1.0 } else { 0.0 },
                    ];
                    for tri in tris {
                        for idx in tri {
                            let p = poly.points[idx];
                            let local = [p[0] - centre[0], p[1] - centre[1]];
                            let uv = explicit_uvs.map_or(
                                [(p[0] - min[0]) / size[0], (p[1] - min[1]) / size[1]],
                                |uvs| uvs[idx],
                            );
                            group_verts.push(crate::resources::OverlayShapeTexVertex {
                                position: overlay_local_px(p[0], p[1], vp_w, vp_h),
                                local_pos: local,
                                fill_colour: tint,
                                border_colour: [0.0; 4],
                                half_size,
                                radii: [0.0; 4],
                                border_width: 0.0,
                                shape_type: 0.0,
                                uv,
                                shadow_colour: [0.0; 4],
                                shadow_params: [0.0; 4],
                                extras: [0.0; 4],
                                nine_slice_uv: [0.0; 4],
                                nine_slice_frac: [0.0; 4],
                                texture_transform_a: tt_a,
                                texture_transform_b: tt_b,
                                // Textured polyline fills carry no clip mask.
                                clip_index: -1.0,
                                clip_rect: [0.0; 4],
                            });
                        }
                    }
                    tex_shape_segs.push((
                        poly.z_order,
                        group_idx,
                        poly_seg_start,
                        group_verts.len() as u32 - poly_seg_start,
                    ));
                }

                let solid_vbuf = if !solid_verts.is_empty() {
                    Some(self.overlay_shape_vbuf.write(device, queue, &solid_verts))
                } else {
                    None
                };

                // Build the shadow-layer storage buffer and its bind group.
                // The solid pipeline layout always expects group 0, so we
                // provide at least one (dummy) element even when no shape has
                // shadows.
                let (shadow_bind_group, shadow_buf, shape_clip_buf) = if solid_vbuf.is_some() {
                    if shadow_layers.is_empty() {
                        shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                            colour: [0.0; 4],
                            params: [0.0; 4],
                        });
                    }
                    let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                        label: Some("overlay_shape_shadow_buf"),
                        size: std::mem::size_of_val(&shadow_layers[..]) as u64,
                        usage: crate::gpu::BufferUsages::STORAGE
                            | crate::gpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&shadow_layers));
                    let clip_buf = upload_clip_buffer(device, queue, &clip_shapes);
                    let vp_buf = self.overlay_viewport_buf.as_ref().unwrap();
                    let inst_buf = self.overlay_instances_buf.as_ref().unwrap();
                    let bg = self.resources.overlay_shape.shadow_bgl.as_ref().map(|bgl| {
                        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                            label: Some("overlay_shape_shadow_bg"),
                            layout: bgl,
                            entries: &[
                                crate::gpu::BindGroupEntry {
                                    binding: 0,
                                    resource: buf.as_entire_binding(),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 1,
                                    resource: clip_buf.as_entire_binding(),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 2,
                                    resource: vp_buf.as_entire_binding(),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 3,
                                    resource: inst_buf.as_entire_binding(),
                                },
                            ],
                        })
                    });
                    (bg, Some(buf), Some(clip_buf))
                } else {
                    (None, None, None)
                };

                let mut tex_batches = Vec::new();
                // Maps a texture group index to its batch index in `tex_batches`
                // (groups with no drawable verts or a missing texture have none),
                // so per-shape segments can reference the right batch.
                let mut group_to_batch: Vec<Option<u32>> = vec![None; tex_groups.len()];
                if has_tex {
                    if let (Some(bgl), Some(sampler)) = (
                        self.resources.overlay_shape.tex_bgl.as_ref(),
                        self.resources.overlay_shape.tex_sampler.as_ref(),
                    ) {
                        for (group_idx, (tex_id, verts)) in tex_groups.iter().enumerate() {
                            if verts.is_empty() {
                                continue;
                            }
                            let Some(entry) = self.resources.content.overlay_textures.get(*tex_id)
                            else {
                                continue;
                            };
                            let view = &entry.view;
                            let bind_group =
                                device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                                    label: Some("overlay_shape_tex_bg"),
                                    layout: bgl,
                                    entries: &[
                                        crate::gpu::BindGroupEntry {
                                            binding: 0,
                                            resource: crate::gpu::BindingResource::TextureView(
                                                view,
                                            ),
                                        },
                                        crate::gpu::BindGroupEntry {
                                            binding: 1,
                                            resource: crate::gpu::BindingResource::Sampler(sampler),
                                        },
                                    ],
                                });
                            let batch_index = tex_batches.len();
                            if batch_index >= self.overlay_shape_tex_vbufs.len() {
                                self.overlay_shape_tex_vbufs.push(
                                    crate::renderer::overlay_buffers::GrowBuffer::vertex(
                                        "overlay_shape_tex_vbuf",
                                    ),
                                );
                            }
                            let vertex_buf = self.overlay_shape_tex_vbufs[batch_index]
                                .write(device, queue, verts);
                            group_to_batch[group_idx] = Some(batch_index as u32);
                            tex_batches.push(crate::resources::OverlayShapeTexBatch {
                                vertex_buf,
                                vertex_count: verts.len() as u32,
                                bind_group,
                            });
                        }
                    }

                    // Emit one draw segment per textured shape (coalescing
                    // contiguous same-texture, same-z runs) at the shape's own
                    // z_order, so a solid shape can layer between two shapes that
                    // share a texture. The whole batch no longer collapses to its
                    // lowest z.
                    if self.overlay_uses_zorder {
                        for &(z, group_idx, vstart, vcount) in &tex_shape_segs {
                            if let Some(batch_index) =
                                group_to_batch.get(group_idx).copied().flatten()
                            {
                                crate::renderer::overlay_draw_order::OverlayDrawSegment::push_shape_tex(
                                    &mut self.overlay_draw_segments,
                                    z,
                                    batch_index,
                                    vstart,
                                    vcount,
                                );
                            }
                        }
                    }
                }

                // Clip-mask bind group (group 1) for the texture pipeline, used by
                // both textured shape batches and blur shapes. Built whenever the
                // texture pipeline runs this frame so `with_clip` is honoured on a
                // textured shape the same way it is on a solid one.
                let (tex_clip_bind_group, tex_clip_buf) = if has_tex || has_blur {
                    if let Some(bgl) = self.resources.overlay_shape.tex_clip_bgl.as_ref() {
                        let clip_buf = upload_clip_buffer(device, queue, &clip_shapes);
                        let vp_buf = self.overlay_viewport_buf.as_ref().unwrap();
                        let bg = device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                            label: Some("overlay_shape_tex_clip_bg"),
                            layout: bgl,
                            entries: &[
                                crate::gpu::BindGroupEntry {
                                    binding: 0,
                                    resource: clip_buf.as_entire_binding(),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 1,
                                    resource: vp_buf.as_entire_binding(),
                                },
                            ],
                        });
                        (Some(bg), Some(clip_buf))
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

                let blur_vbuf = if !blur_verts.is_empty() {
                    Some(
                        self.overlay_shape_blur_vbuf
                            .write(device, queue, &blur_verts),
                    )
                } else {
                    None
                };

                if solid_vbuf.is_some() || !tex_batches.is_empty() || blur_vbuf.is_some() {
                    self.overlay_shape_gpu_data = Some(crate::resources::OverlayShapeGpuData {
                        vertex_buf: solid_vbuf,
                        vertex_count: solid_verts.len() as u32,
                        shadow_bind_group,
                        _shadow_buf: shadow_buf,
                        _clip_buf: shape_clip_buf,
                        tex_batches,
                        tex_clip_bind_group,
                        _tex_clip_buf: tex_clip_buf,
                        blur_vertex_buf: blur_vbuf,
                        blur_vertex_count: blur_verts.len() as u32,
                        max_blur_radius,
                    });
                }
            }
        }
    }

    /// Sort the overlay draw list into paint order. The family passes have
    /// already recorded their own segments; this orders them by
    /// `(z_order, family_rank)` for the ordered emit path.
    pub(super) fn finalize_overlay_draw_order(&mut self, _frame: &FrameData) {
        // When no overlay uses z_order, the family passes recorded nothing and
        // the emit path keeps its fixed order; there is nothing to finalize.
        if !self.overlay_uses_zorder {
            return;
        }
        use crate::renderer::overlay_draw_order::sort_overlay_segments;
        sort_overlay_segments(&mut self.overlay_draw_segments);
    }
}

#[cfg(test)]
mod clip_registry_tests {
    use super::build_clip_shapes;
    use crate::renderer::types::{OverlayShape, OverlayShapeItem};

    #[test]
    fn indexes_chains_and_scales_masks() {
        // A rounded-rect parent mask (id 10) and a circle child mask (id 20)
        // clipped by the parent, at 2x device scale.
        let parent = OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 8.0 },
            [100.0, 50.0],
            [200.0, 100.0],
        )
        .with_clip_mask(10);
        let child = OverlayShapeItem::new(OverlayShape::Circle, [110.0, 60.0], [80.0, 80.0])
            .with_clip_mask(20)
            .with_clip(10);
        let (gpu, map, bboxes) = build_clip_shapes(
            &[parent, child],
            2.0,
            [1000.0, 1000.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
        );

        assert_eq!(gpu.len(), 2);
        assert_eq!(map[&10], 0);
        assert_eq!(map[&20], 1);

        // Parent: centre and half-size scaled by ppp, corner radius scaled, no parent.
        assert_eq!(gpu[0].center, [400.0, 200.0]);
        assert_eq!(gpu[0].half_size, [200.0, 100.0]);
        assert_eq!(gpu[0].radii, [16.0, 16.0, 16.0, 16.0]);
        assert_eq!(gpu[0].params[0], 0.0); // rect/rounded shape type
        assert_eq!(gpu[0].params[2], -1.0); // no parent

        // Child: circle shape type, parent index points at the parent (0).
        assert_eq!(gpu[1].params[0], 1.0);
        assert_eq!(gpu[1].params[2], 0.0);

        // Bounding boxes are parallel to the GPU array and scaled by ppp:
        // [pos.x, pos.y, pos.x + size.x, pos.y + size.y] * ppp.
        assert_eq!(bboxes.len(), 2);
        assert_eq!(bboxes[0], [200.0, 100.0, 600.0, 300.0]);
        assert_eq!(bboxes[1], [220.0, 120.0, 380.0, 280.0]);
    }

    #[test]
    fn duplicate_mask_ids_resolve_to_the_first_consistently() {
        // Two masks share id 5. The first wins for both the index and the bbox,
        // so a clipped item can never get a bbox from one and an SDF index from
        // another.
        let first = OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [0.0, 0.0],
            [100.0, 100.0],
        )
        .with_clip_mask(5);
        let second = OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [500.0, 500.0],
            [50.0, 50.0],
        )
        .with_clip_mask(5);
        let (gpu, map, bboxes) = build_clip_shapes(
            &[first, second],
            1.0,
            [1000.0, 1000.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
        );

        assert_eq!(gpu.len(), 1);
        assert_eq!(map[&5], 0);
        // The bbox is the first mask's, matching the index the first mask claimed.
        assert_eq!(bboxes[0], [0.0, 0.0, 100.0, 100.0]);
    }

    #[test]
    fn non_mask_shapes_are_ignored() {
        let drawn = OverlayShapeItem::new(OverlayShape::Circle, [0.0, 0.0], [10.0, 10.0]);
        let (gpu, map, bboxes) = build_clip_shapes(
            &[drawn],
            1.0,
            [1000.0, 1000.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
        );
        assert!(gpu.is_empty());
        assert!(map.is_empty());
        assert!(bboxes.is_empty());
    }
}
