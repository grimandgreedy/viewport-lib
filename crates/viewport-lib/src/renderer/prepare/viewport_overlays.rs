//! Overlay prepare passes: labels, glyph runs, polylines, and SDF overlay
//! shapes.

use super::*;
use crate::resources::overlay::glyph_style::GlyphStyle;

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
        // encoder, and `build_clip_shapes` rejects it as a mask, so callers skip
        // Vector shapes before reaching here.
        OverlayShape::Vector { .. } => (0.0, [0.0; 4]),
        // OverlayShape is non_exhaustive; no special edge/radius for shapes not handled here.
        _ => (0.0, [0.0; 4]),
    }
}

/// Curve-flattening tolerance for vector-shape fills, in logical pixels.
const VECTOR_FILL_TOLERANCE: f32 = 0.2;

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
    let cx = shape.transform.translate[0] + hw;
    let cy = shape.transform.translate[1] + hh;
    let piv = shape.transform.pivot;
    let identity = shape.transform.linear_is_identity();
    let m = shape.transform.linear();
    local
        .iter()
        .map(|lp| {
            // Path-local -> centred (the unrotated `p` frame in `distance`).
            let cpx = lp[0] - hw;
            let cpy = lp[1] - hh;
            if identity {
                return [cx + cpx, cy + cpy];
            }
            // The item's linear map about its pivot: the inner half of the
            // composition contract, baked in because an immediate item has no
            // instance slot of its own.
            let ax = cpx - piv[0];
            let ay = cpy - piv[1];
            [
                cx + m[0][0] * ax + m[0][1] * ay + piv[0],
                cy + m[1][0] * ax + m[1][1] * ay + piv[1],
            ]
        })
        .collect()
}

/// Tessellate and emit a vector shape's fill and its shadow layers as
/// `OverlayTextVertex`s into `batch`, recording the fill in `fills`.
///
/// `image_size` is the pixel size of the image a texture fill samples, `None`
/// when the image is missing, which skips the fill the way a missing image
/// skips an analytic shape. A table without textures (a compiled group) skips
/// a texture fill too.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_vector_shape(
    batch: &mut Vec<crate::resources::OverlayTextVertex>,
    shape: &crate::renderer::types::OverlayShapeItem,
    subpaths: &[crate::renderer::types::SubPath],
    fill_rule: crate::renderer::types::FillRule,
    vp_w: f32,
    vp_h: f32,
    fills: &mut super::overlay_fill::FillTable,
    image_size: Option<[u32; 2]>,
) {
    let mesh = super::overlay_vector::tessellate(subpaths, fill_rule, VECTOR_FILL_TOLERANCE);

    // Shadow layers, behind the fill. A vector path has no distance field, so a
    // layer is drawn as the filled silhouette plus its contours re-stroked
    // wider, once per band: the stroke is what supplies the dilation the SDF
    // path gets from `spread`.
    for layer in shape
        .style
        .shadows
        .iter()
        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
        .filter(|l| l.is_visible())
    {
        emit_vector_shadow(batch, shape, subpaths, layer, vp_w, vp_h);
    }

    let content_start = batch.len();
    let textured = shape.style.fill.texture_id().is_some();
    let draws_fill = !textured || (fills.draws_textures() && image_size.is_some());
    if !mesh.indices.is_empty() && draws_fill {
        let positions = transform_vector_positions(&mesh.positions, shape);
        emit_vector_fill(
            batch,
            &positions,
            &mesh.positions,
            &mesh.indices,
            &shape.style.fill,
            shape.style.opacity,
            vp_w,
            vp_h,
            fills,
            image_size.unwrap_or([1, 1]),
        );
    }

    // Tint the fill now rather than at the end: the inner shadows that follow
    // are shadow geometry, and a tint never reaches a shadow layer.
    tint_vertices_from(batch, content_start, shape.style.tint);

    // Inner shadow layers, over the fill, matching the order the SDF path
    // composites in. A band runs inward from each contour, which is the
    // erosion an inset layer describes.
    for layer in shape
        .style
        .inner_shadows
        .iter()
        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
        .filter(|l| l.is_visible())
    {
        emit_vector_inner_shadow(batch, shape, subpaths, layer, vp_w, vp_h);
    }
}

/// Push a tessellated vector path as a triangle list for the backdrop
/// pipeline. `positions` are screen logical pixels.
///
/// The triangles are the coverage, so each vertex is encoded as a sharp rect
/// grown a pixel past the path's bounds: the shader's distance test then reads
/// fully inside everywhere, instead of fading the edges that touch the bounds.
fn push_tessellated_backdrop_verts(
    out: &mut Vec<crate::resources::OverlayBackdropVertex>,
    positions: &[[f32; 2]],
    indices: &[u32],
    filters: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    let Some((min, max)) = polyline_bounds(positions) else {
        return;
    };
    let centre = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
    let half_size = [(max[0] - min[0]) * 0.5 + 1.0, (max[1] - min[1]) * 0.5 + 1.0];
    for &i in indices {
        let p = positions[i as usize];
        out.push(crate::resources::OverlayBackdropVertex {
            position: overlay_local_px(p[0], p[1], vp_w, vp_h),
            local_pos: [p[0] - centre[0], p[1] - centre[1]],
            // A transparent tint leaves the filtered backdrop alone: the
            // shape's own fill draws over it in the text stream.
            tint: [0.0; 4],
            half_size,
            radii: [0.0; 4],
            shape_meta: [0.0, -1.0, 0.0, 0.0],
            clip_rect: [0.0; 4],
            shadow_index: [0.0; 3],
            filters,
            linv: [1.0, 0.0, 0.0, 1.0],
            item_params: [1.0, 0.0, 0.0, 0.0],
        });
    }
}

/// A clip box that lies wholly off screen, so every fragment tested against it
/// is discarded. It sits above and left of the framebuffer, where no fragment
/// position falls, and keeps a real area: a box far off to the right would need
/// large coordinates, where `x + 1.0` rounds back to `x` in f32 and the box
/// collapses into the "no clip" sentinel.
const OFFSCREEN_CLIP_RECT: [f32; 4] = [-2.0, -2.0, -1.0, -1.0];

/// Scale a public `OverlayClip::rect` from logical to framebuffer pixels.
///
/// The shaders and `combine_clip_rects` read any box without area as "no
/// clip", which is the internal sentinel. A public box without area is an
/// empty intersection and must clip everything, so it becomes the off-screen
/// box instead.
pub(super) fn clip_rect_to_framebuffer(r: [f32; 4], ppp: f32) -> [f32; 4] {
    if r[2] > r[0] && r[3] > r[1] {
        [r[0] * ppp, r[1] * ppp, r[2] * ppp, r[3] * ppp]
    } else {
        OFFSCREEN_CLIP_RECT
    }
}

/// Intersect two clip boxes in framebuffer pixels. A box without area means
/// "no clip", so the other wins; two real boxes intersect, and a
/// non-overlapping pair returns an off-screen box so nothing survives. This
/// mirrors `combine_clip` in the overlay shaders, which does the same for the
/// instance half of the clip. Public boxes reach here through
/// `clip_rect_to_framebuffer`, so an empty one is never read as no clip.
fn combine_clip_rects(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let a_valid = a[2] > a[0] && a[3] > a[1];
    let b_valid = b[2] > b[0] && b[3] > b[1];
    if !a_valid {
        return b;
    }
    if !b_valid {
        return a;
    }
    let r = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    if r[2] <= r[0] || r[3] <= r[1] {
        return OFFSCREEN_CLIP_RECT;
    }
    r
}

/// Give the vertices `batch[from..]` their item's fill over the box at
/// `box_min` of size `box_size`, in the same logical pixels as the vertex
/// positions (before any item rotation is applied to them).
///
/// A solid fill multiplies into the vertex colours, which is all a flat colour
/// needs. Any other fill becomes a record in `fills` that the text pipeline
/// evaluates per fragment through the shared fill module, and each vertex
/// carries its offset from the box centre and the record's index. Either way
/// the fill multiplies the vertex colour, so a per-glyph `colours` entry
/// composes with it the way `tint` does.
///
/// `image_size` is the pixel size of a texture fill's image, which a
/// nine-slice fill needs.
pub(super) fn fill_vertices_from(
    batch: &mut [crate::resources::OverlayTextVertex],
    from: usize,
    fill: &crate::renderer::types::OverlayFill,
    box_min: [f32; 2],
    box_size: [f32; 2],
    fills: &mut super::overlay_fill::FillTable,
    image_size: [u32; 2],
) {
    let half = [box_size[0] * 0.5, box_size[1] * 0.5];
    let centre = [box_min[0] + half[0], box_min[1] + half[1]];
    if let crate::renderer::types::OverlayFill::Solid(c) = fill {
        let c = c.to_linear_rgba();
        for v in &mut batch[from..] {
            v.colour = [
                v.colour[0] * c[0],
                v.colour[1] * c[1],
                v.colour[2] * c[2],
                v.colour[3] * c[3],
            ];
        }
        return;
    }
    let index = fills.push(fill, half, 1.0, [1.0; 4], image_size);
    for v in &mut batch[from..] {
        // Text-stream vertex positions are logical pixels.
        v.fill = [
            v.position[0] - centre[0],
            v.position[1] - centre[1],
            index,
            0.0,
        ];
    }
}

/// Multiply an item's `tint` into the colours of `batch[from..]`, and mark that
/// range as the item's content.
///
/// Called per emitted range rather than over the whole batch so shadow ranges
/// are skipped: a tint never reaches a shadow layer, on any path. A compiled
/// group's shadow colours are baked, so retention could only ever honour that
/// rule, and content that changed appearance when it moved between the
/// immediate and retained paths is the defect this vocabulary exists to remove.
///
/// The same range is what a retained group's own per-frame tint applies to, so
/// the mark is set whether or not the item has a tint of its own.
pub(super) fn tint_vertices_from(
    batch: &mut [crate::resources::OverlayTextVertex],
    from: usize,
    tint: [f32; 4],
) {
    let identity = tint == [1.0, 1.0, 1.0, 1.0];
    for v in &mut batch[from..] {
        v.group_tint = 1.0;
        if !identity {
            v.colour = [
                v.colour[0] * tint[0],
                v.colour[1] * tint[1],
                v.colour[2] * tint[2],
                v.colour[3] * tint[3],
            ];
        }
    }
}

/// Emit one inner shadow layer of a vector path: a band running inward from
/// each contour, `spread + d` wide for each blur band, in the layer colour.
///
/// The band is stacked from the contour rather than drawn as an eroded copy of
/// the fill, so it stays inside the silhouette whatever the layer offset does:
/// the offset widens the band where it points and narrows it where it points
/// away, which is the region an offset erosion covers.
fn emit_vector_inner_shadow(
    batch: &mut Vec<crate::resources::OverlayTextVertex>,
    shape: &crate::renderer::types::OverlayShapeItem,
    subpaths: &[crate::renderer::types::SubPath],
    layer: &crate::renderer::types::ShadowLayer,
    vp_w: f32,
    vp_h: f32,
) {
    let base = layer.colour.to_linear_rgba();
    let contours = crate::renderer::types::flatten_contours(subpaths);
    for (d, band_alpha) in overlay_geometry::shadow_bands(layer) {
        let mut colour = base;
        colour[3] *= shape.style.opacity * band_alpha;
        let width = layer.spread + d;
        if colour[3] <= 0.0 || width <= 0.0 {
            continue;
        }
        for (contour, _closed) in &contours {
            // A contour bounds the fill whether or not its subpath was closed,
            // so the band follows it as a loop either way.
            let pts = transform_vector_positions(contour, shape);
            let side = -overlay_geometry::contour_outward_sign(&pts);
            overlay_geometry::emit_contour_band(
                batch,
                &pts,
                side,
                width,
                layer.offset,
                colour,
                vp_w,
                vp_h,
            );
        }
    }
}

/// Emit one outer shadow layer of a vector path: a band running outward from
/// each contour, `spread + d` wide for each blur band, in the layer colour.
///
/// The band stops at the contour instead of painting the dilated silhouette
/// behind the fill, which is how the distance-field path clips an outer layer
/// and is what keeps a translucent path from tinting itself through its own
/// shadow. The layer offset widens the band where it points and narrows it
/// where it points away, which is the region an offset dilation covers.
fn emit_vector_shadow(
    batch: &mut Vec<crate::resources::OverlayTextVertex>,
    shape: &crate::renderer::types::OverlayShapeItem,
    subpaths: &[crate::renderer::types::SubPath],
    layer: &crate::renderer::types::ShadowLayer,
    vp_w: f32,
    vp_h: f32,
) {
    let base = layer.colour.to_linear_rgba();
    let contours = crate::renderer::types::flatten_contours(subpaths);
    for (d, band_alpha) in overlay_geometry::shadow_bands(layer) {
        let mut colour = base;
        colour[3] *= shape.style.opacity * band_alpha;
        let width = layer.spread + d;
        if colour[3] <= 0.0 || width <= 0.0 {
            continue;
        }
        for (contour, _closed) in &contours {
            // A contour bounds the fill whether or not its subpath was closed,
            // so the band follows it as a loop either way.
            let pts = transform_vector_positions(contour, shape);
            let side = overlay_geometry::contour_outward_sign(&pts);
            overlay_geometry::emit_contour_band(
                batch,
                &pts,
                side,
                width,
                layer.offset,
                colour,
                vp_w,
                vp_h,
            );
        }
    }
}

/// Build the per-frame clip-shape registry from the overlay shapes that are clip
/// masks (`provides_mask` set). Returns the GPU array (framebuffer-pixel geometry),
/// a map from mask id to its index in that array, and the axis-aligned
/// bounding box (framebuffer pixels) of each mask, parallel to the GPU array. A
/// mask's own `clip.mask` links it to a parent, so nested masks compose by
/// intersection.
///
/// The bounding boxes drive the shader's cheap pre-reject and share this array's
/// indexing, so a drawn item's `clip_rect` and `clip_index` always describe the
/// same mask. A mask id is expected to be unique within a frame; if one is
/// reused, the first mask with that id wins (both for the index and the bbox),
/// which is why the bbox comes from here rather than a separate id-keyed map.
///
/// At most `MAX_OVERLAY_CLIP_SHAPES` masks are kept, the length of the shaders'
/// uniform array. A later mask is dropped, as a culled one is, and counted in
/// the last element of the result.
fn build_clip_shapes(
    shapes: &[crate::renderer::types::OverlayShapeItem],
    ppp: f32,
    viewport: [f32; 2],
    view: &glam::Mat4,
    proj: &glam::Mat4,
    world_origin: glam::DVec3,
) -> (
    Vec<crate::resources::ClipShapeGpu>,
    std::collections::HashMap<u32, i32>,
    Vec<[f32; 4]>,
    u32,
) {
    let mut dropped = 0;
    // Collect masks in stable order, recording each id's index. Each mask
    // resolves its anchor to an effective top-left, the same way the draw loop
    // does, so corner- and world-anchored masks clip where they draw. A mask
    // whose world anchor is culled is dropped: items referencing it fall through
    // to no clipping, matching how an absent mask id behaves.
    let mut index_of: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
    let mut masks: Vec<(&crate::renderer::types::OverlayShapeItem, [f32; 2])> = Vec::new();
    for (i, s) in shapes.iter().enumerate() {
        if let Some(id) = s.provides_mask {
            // A vector shape is tessellated triangles with no distance field, so
            // there is nothing for the shader to evaluate the mask against. It is
            // rejected rather than encoded as its bounding box: a wrong clip is
            // harder to find than no clip, and a documented box fallback is a
            // behaviour a consumer could come to depend on.
            if matches!(s.shape, crate::renderer::types::OverlayShape::Vector { .. }) {
                super::overlay_style_check::warn_vector_mask(i, id);
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = index_of.entry(id) {
                if let Some(tl) = s.resolve_top_left_at(viewport, view, proj, world_origin) {
                    if masks.len()
                        == crate::resources::overlay::overlay_shape::MAX_OVERLAY_CLIP_SHAPES
                    {
                        dropped += 1;
                        continue;
                    }
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
        let centre = [(tl[0] + hw) * ppp, (tl[1] + hh) * ppp];
        let half = [hw * ppp, hh * ppp];
        let parent = s
            .clip
            .mask
            .and_then(|pid| index_of.get(&pid).copied())
            .unwrap_or(-1);
        out.push(crate::resources::ClipShapeGpu {
            centre,
            half_size: half,
            radii,
            params: [shape_type, s.transform.rotation, parent as f32, 0.0],
            pivot: [s.transform.pivot[0] * ppp, s.transform.pivot[1] * ppp],
            _pad: [0.0, 0.0],
        });
        bboxes.push([
            tl[0] * ppp,
            tl[1] * ppp,
            (tl[0] + s.size[0]) * ppp,
            (tl[1] + s.size[1]) * ppp,
        ]);
    }
    (out, index_of, bboxes, dropped)
}

/// Written in place of an empty clip list, so the clip binding always has an
/// element to point at.
static NO_CLIP: [crate::resources::ClipShapeGpu; 1] = [crate::resources::ClipShapeGpu {
    centre: [0.0, 0.0],
    half_size: [0.0, 0.0],
    radii: [0.0; 4],
    params: [0.0, 0.0, -1.0, 0.0],
    pivot: [0.0, 0.0],
    _pad: [0.0, 0.0],
}];

/// The clip-shape data to upload for a frame: never empty.
fn clip_data(clips: &[crate::resources::ClipShapeGpu]) -> &[crate::resources::ClipShapeGpu] {
    if clips.is_empty() { &NO_CLIP } else { clips }
}

impl ViewportRenderer {
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
                frame.camera.render_camera.orientation(),
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
        ov: &mut crate::renderer::ViewportOverlay,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        vp_w: f32,
        vp_h: f32,
    ) {
        let data = [vp_w, vp_h, 0.0, 0.0];
        if let Some(buf) = &ov.overlay_viewport_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&data));
        } else {
            let buf = device.create_buffer(&crate::gpu::BufferDescriptor {
                label: Some("overlay_viewport_uniform"),
                size: std::mem::size_of_val(&data) as u64,
                usage: crate::gpu::BufferUsages::UNIFORM | crate::gpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
            ov.overlay_viewport_buf = Some(buf);
        }
    }

    /// Ensure the instance stream holds at least the identity instance. The
    /// label prepare fills it with the identity plus one slot per retained group;
    /// this fallback covers a frame with immediate shapes but no text or retained
    /// groups, so the shape pipeline always has an instance stream to read.
    fn ensure_overlay_instances_default(
        ov: &mut crate::renderer::ViewportOverlay,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
    ) {
        if ov.overlay_instances_ready {
            return;
        }
        let data = [crate::resources::OverlayInstance::IDENTITY];
        ov.overlay_bindings
            .instances
            .write_changed(device, queue, &data);
        ov.overlay_bindings.instances_uniform.write_changed(
            device,
            queue,
            &crate::resources::OverlayInstance::padded_uniform_bytes(&data),
        );
        ov.overlay_instances_ready = true;
    }

    pub(super) fn prepare_overlay_labels(
        &mut self,
        device: &crate::gpu::Device,
        queue: &crate::gpu::Queue,
        frame: &FrameData,
        ov: &mut crate::renderer::ViewportOverlay,
    ) {
        // ---------------------------------------------------------------
        // Overlay labels
        // ---------------------------------------------------------------
        // Labels, glyph runs, and polylines are merged into a single
        // z_order-sorted vertex buffer so that items at the same z_order
        // interleave correctly (e.g. a polyline underline at i32::MAX sits with
        // console text at i32::MAX but above HUD labels at 0).
        ov.label_gpu_data = None;
        // Counted before any of the paths below can skip a group, so the
        // submitted/drawn pair carries the skips.
        self.overlay_retained_counters.submitted = frame.overlays.retained.len() as u32;
        let has_vector_shape = frame.overlays.shapes.iter().any(|s| {
            matches!(
                &s.shape,
                crate::renderer::types::OverlayShape::Vector { .. }
            )
        });
        let has_overlay = !frame.overlays.labels.is_empty()
            || !frame.overlays.glyph_runs.is_empty()
            || !frame.overlays.polylines.is_empty()
            || has_vector_shape
            || !frame.overlays.retained.is_empty();
        if has_overlay {
            self.resources.ensure_overlay_fill_resources(device, queue);
            self.resources.ensure_overlay_text_pipeline(device);
            let vp_w = frame.camera.viewport_size[0];
            let vp_h = frame.camera.viewport_size[1];
            let ppp = frame.camera.pixels_per_point;
            if vp_w > 0.0 && vp_h > 0.0 {
                // World anchors project through the camera relative to the
                // world origin, with the origin subtracted in f64.
                let world_origin = self.origin.origin();
                let view = &self.origin.camera(&frame.camera.render_camera).view();
                let proj = &frame.camera.render_camera.projection();

                // Each item (label, glyph run, or polyline) contributes a
                // (z_order, fill image, verts) batch. A stable sort keeps
                // submission order within an equal z_order.
                let mut batches: Vec<(
                    i32,
                    Option<crate::renderer::types::OverlayTextureId>,
                    Vec<crate::resources::OverlayTextVertex>,
                )> = Vec::new();
                // The fill records the text stream's gradient and texture
                // fills read, uploaded once for the frame.
                let mut text_fills = super::overlay_fill::FillTable::new();
                // The pixel size of a texture fill's image: `None` for a fill
                // that is not a texture, `Some(None)` when the image is gone,
                // which skips the item as it skips an analytic shape.
                let fill_image = |fill: &crate::renderer::types::OverlayFill,
                                  textures: &crate::resources::handle::SlotStore<
                    crate::resources::overlay::overlay_shape::OverlayShapeTextureEntry,
                    crate::renderer::OverlayTextureId,
                >| {
                    fill.texture_id().map(|id| textures.get(id).map(|t| t.size))
                };

                // Clip masks: an overlay shape with `clip_mask_id` set contributes a
                // clipping bounding box (framebuffer pixels) plus an SDF entry. Labels
                // with a matching `clip_id` are clipped to it, the same way the shape
                // pipeline clips shapes. (Shape-accurate SDF clipping is applied in the
                // shape pass; text uses the bounding box, exact for the rectangular
                // masks scroll containers use.) The bbox and the SDF index come from the
                // same registry, so a clipped item's `clip_rect` and `clip_index` always
                // describe the same mask.
                let overlay_time = frame.overlays.time;
                let (clip_shapes, clip_index_of, clip_bboxes, clips_dropped) = build_clip_shapes(
                    &frame.overlays.shapes,
                    ppp,
                    [vp_w, vp_h],
                    view,
                    proj,
                    world_origin,
                );
                self.overlay_retained_counters.clip_masks_dropped = clips_dropped;
                // Stamp an item's clip onto every vertex it emitted: the mask
                // bbox and index when it names a mask, intersected with the
                // item's own `clip_rect`. Both are framebuffer-pixel boxes, and
                // they compose by intersection, so naming both clips to the
                // overlap.
                let stamp_clip = |batch: &mut [crate::resources::OverlayTextVertex],
                                  clip_id: Option<u32>,
                                  clip_rect: Option<[f32; 4]>| {
                    let mask = clip_id.and_then(|id| clip_index_of.get(&id).copied());
                    let rect = clip_rect.map(|r| clip_rect_to_framebuffer(r, ppp));
                    if mask.is_none() && rect.is_none() {
                        return;
                    }
                    let mask_rect = mask.map_or([0.0; 4], |ci| clip_bboxes[ci as usize]);
                    let cr = combine_clip_rects(mask_rect, rect.unwrap_or([0.0; 4]));
                    let ci = mask.map_or(-1.0, |ci| ci as f32);
                    for v in batch.iter_mut() {
                        v.clip_rect = cr;
                        if ci >= 0.0 {
                            v.clip_index = ci;
                        }
                    }
                };

                // --- Polylines (between shapes and labels in z order) ---
                for poly in frame.overlays.polylines.iter() {
                    if poly.points.len() < 2 || poly.style.opacity <= 0.0 {
                        continue;
                    }
                    // Resolve animation tracks first: a `translate` track drives
                    // the same channel the anchor offset is measured from, so
                    // resolving the offset before the track would use last
                    // frame's translate.
                    let animated_storage;
                    let poly: &crate::renderer::types::OverlayPolylineItem =
                        if poly.animations.is_none() {
                            poly
                        } else {
                            let mut owned = poly.clone();
                            if let Some(anims) = owned.animations.take() {
                                anims.apply(
                                    overlay_time,
                                    &mut owned.transform,
                                    &mut owned.style.opacity,
                                    &mut owned.style.tint,
                                );
                            }
                            animated_storage = owned;
                            &animated_storage
                        };
                    // Resolve the anchor to a screen-pixel offset and draw the
                    // path there; a culled world anchor skips it. The offset is
                    // baked into a translated copy so the stroke and fill paths
                    // (which read `poly.points`) both see anchored coordinates.
                    let Some(offset) =
                        poly.resolve_offset_at([vp_w, vp_h], view, proj, world_origin)
                    else {
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
                    super::overlay_style_check::warn_inert_style(
                        "OverlayPolylineItem",
                        crate::renderer::types::OverlayStyleSupport::for_polyline(poly.closed),
                        &poly.style,
                    );
                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                    // Shadow layers first, behind both the fill and the stroke.
                    for layer in poly
                        .style
                        .shadows
                        .iter()
                        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
                        .filter(|l| l.is_visible())
                    {
                        emit_polyline_shadow(
                            &mut batch,
                            poly,
                            layer,
                            poly.style.opacity,
                            vp_w,
                            vp_h,
                        );
                    }
                    let content_start = batch.len();
                    let image =
                        fill_image(&poly.style.fill, &self.resources.content.overlay_textures);
                    let fill_texture = match image {
                        Some(Some(_)) if poly.closed => poly.style.fill.texture_id(),
                        _ => None,
                    };
                    if poly.closed && image != Some(None) {
                        emit_filled_polyline(
                            &mut batch,
                            &poly.points,
                            &poly.style.fill,
                            poly.style.opacity,
                            vp_w,
                            vp_h,
                            &mut text_fills,
                            poly.uvs.as_deref(),
                            image.flatten().unwrap_or([1, 1]),
                        );
                    }
                    // Tint the fill before the inner shadows go over it: a tint
                    // never reaches a shadow layer.
                    tint_vertices_from(&mut batch, content_start, poly.style.tint);
                    // An inset layer goes over what it erodes and under the
                    // edge of it: over the fill and under the stroke for a
                    // filled path, over the stroke when the stroke is all the
                    // item covers.
                    let filled = poly.closed && poly.style.fill.is_set();
                    let inner = |batch: &mut Vec<crate::resources::OverlayTextVertex>| {
                        for layer in poly
                            .style
                            .inner_shadows
                            .iter()
                            .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
                            .filter(|l| l.is_visible())
                        {
                            emit_polyline_inner_shadow(
                                batch,
                                poly,
                                layer,
                                poly.style.opacity,
                                vp_w,
                                vp_h,
                            );
                        }
                    };
                    if filled {
                        inner(&mut batch);
                    }
                    let stroke_start = batch.len();
                    if let Some(stroke) = poly.stroke.as_ref().filter(|s| s.width > 0.0) {
                        let mut colour = stroke.colour.to_linear_rgba();
                        colour[3] *= poly.style.opacity;
                        emit_polyline_stroke(&mut batch, poly, stroke, colour, vp_w, vp_h);
                    }
                    tint_vertices_from(&mut batch, stroke_start, poly.style.tint);
                    if !filled {
                        inner(&mut batch);
                    }
                    // The path turns and scales about its own bounding box,
                    // shadows included, after every part of it has been emitted.
                    if let Some((pmin, pmax)) = polyline_bounds(&poly.points) {
                        let rot = OverlayRotation::new(
                            &poly.transform,
                            OverlayRotation::pivot_point(
                                pmin,
                                [pmax[0] - pmin[0], pmax[1] - pmin[1]],
                                poly.transform.pivot,
                            ),
                        );
                        rotate_vertices_from(&mut batch, 0, rot);
                    }
                    stamp_clip(&mut batch, poly.clip.mask, poly.clip.rect);
                    if !batch.is_empty() {
                        batches.push((poly.z_order, fill_texture, batch));
                    }
                }

                // --- Vector shapes (tessellated fills + outline strokes) ---
                // A vector shape has no SDF, so it draws here through the same
                // triangle-fill pipeline as filled polylines rather than the
                // SDF shape pass, image fills included. Only a backdrop blur
                // goes through the shape pass's backdrop pipeline (see
                // `prepare_overlay_shapes`).
                for shape in &frame.overlays.shapes {
                    let crate::renderer::types::OverlayShape::Vector {
                        subpaths,
                        fill_rule,
                    } = &shape.shape
                    else {
                        continue;
                    };
                    // Mask-only shapes contribute a clip rect, not a draw; skip
                    // fully transparent ones.
                    if shape.provides_mask.is_some() || shape.style.opacity <= 0.0 {
                        continue;
                    }
                    super::overlay_style_check::warn_inert_style(
                        "OverlayShape::Vector",
                        crate::renderer::types::OverlayStyleSupport::for_shape(&shape.shape),
                        &shape.style,
                    );
                    let mut owned = shape.clone();
                    if let Some(anims) = owned.animations.take() {
                        anims.apply(
                            overlay_time,
                            &mut owned.transform,
                            &mut owned.style.opacity,
                            &mut owned.style.tint,
                        );
                    }
                    // Resolve the anchor to an absolute top-left so anchored
                    // vector shapes tessellate where they draw; a culled world
                    // anchor skips the shape. Tracks resolve first, because a
                    // `translate` track drives the channel the top-left is
                    // measured from.
                    let Some(tl) =
                        owned.resolve_top_left_at([vp_w, vp_h], view, proj, world_origin)
                    else {
                        continue;
                    };
                    owned.transform.translate = tl;
                    let image =
                        fill_image(&owned.style.fill, &self.resources.content.overlay_textures);
                    let fill_texture = match image {
                        Some(Some(_)) => owned.style.fill.texture_id(),
                        _ => None,
                    };
                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                    emit_vector_shape(
                        &mut batch,
                        &owned,
                        subpaths,
                        *fill_rule,
                        vp_w,
                        vp_h,
                        &mut text_fills,
                        image.flatten(),
                    );
                    stamp_clip(&mut batch, owned.clip.mask, owned.clip.rect);
                    if !batch.is_empty() {
                        batches.push((shape.z_order, fill_texture, batch));
                    }
                }

                // --- Labels ---
                for label in &frame.overlays.labels {
                    if label.text.is_empty() || label.style.opacity <= 0.0 {
                        continue;
                    }

                    let Some(anchor_px) = crate::renderer::types::resolve_anchor_origin_at(
                        &label.anchoring.origin,
                        [vp_w, vp_h],
                        view,
                        proj,
                        world_origin,
                    ) else {
                        continue;
                    };

                    // Resolve animation tracks onto a local copy so the input
                    // frame stays untouched; a static label skips the clone.
                    let animated_storage;
                    let label: &crate::renderer::types::LabelItem = if label.animations.is_none() {
                        label
                    } else {
                        let mut owned = label.clone();
                        if let Some(anims) = owned.animations.take() {
                            anims.apply(
                                overlay_time,
                                &mut owned.transform,
                                &mut owned.style.opacity,
                                &mut owned.style.tint,
                            );
                        }
                        animated_storage = owned;
                        &animated_storage
                    };
                    let opacity = label.style.opacity.clamp(0.0, 1.0);
                    super::overlay_style_check::warn_inert_style(
                        "LabelItem",
                        crate::renderer::types::OverlayStyleSupport::for_glyphs(),
                        &label.style,
                    );
                    let image =
                        fill_image(&label.style.fill, &self.resources.content.overlay_textures);
                    if image == Some(None) {
                        continue;
                    }

                    let layout = if let Some(max_w) = label.max_width {
                        self.resources.content.glyph_atlas.layout_text_wrapped(
                            &label.text,
                            label.text_style.size,
                            label.text_style.font,
                            max_w,
                            ppp,
                            device,
                            GlyphStyle::PLAIN,
                        )
                    } else {
                        self.resources.content.glyph_atlas.layout_text(
                            &label.text,
                            label.text_style.size,
                            label.text_style.font,
                            ppp,
                            device,
                            GlyphStyle::PLAIN,
                        )
                    };

                    let font_index = label.text_style.font.map_or(0, |h| h.0);
                    let ascent = self
                        .resources
                        .content
                        .glyph_atlas
                        .font_ascent(font_index, label.text_style.size);

                    let align_offset = label.anchoring.align.x.align_shift(layout.total_width);
                    let align_offset_y = label
                        .anchoring
                        .align
                        .y
                        .align_shift(layout.height, Some(ascent));

                    let text_x = anchor_px[0] + align_offset + label.transform.translate[0];
                    let text_y = anchor_px[1] + align_offset_y + label.transform.translate[1];

                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();

                    // A label is one range: its glyphs and their shadow layers,
                    // turned together about the laid-out box.
                    let rot = OverlayRotation::new(
                        &label.transform,
                        OverlayRotation::pivot_point(
                            [text_x, text_y],
                            [layout.total_width, layout.height],
                            label.transform.pivot,
                        ),
                    );

                    let text_start = batch.len();

                    // Shadow layers, back to front, behind the glyphs. Each is a
                    // re-layout against a styled atlas cell, so the dilation and
                    // fade are baked per glyph rather than faked with offset
                    // copies. Layout is identical to the plain pass, so the
                    // shadow stays registered with the text it backs.
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
                            self.resources.content.glyph_atlas.layout_text_wrapped(
                                &label.text,
                                label.text_style.size,
                                label.text_style.font,
                                max_w,
                                ppp,
                                device,
                                style,
                            )
                        } else {
                            self.resources.content.glyph_atlas.layout_text(
                                &label.text,
                                label.text_style.size,
                                label.text_style.font,
                                ppp,
                                device,
                                style,
                            )
                        };
                        let col = apply_opacity(layer.colour.to_linear_rgba(), opacity);
                        emit_glyph_quads(
                            &mut batch,
                            &sl.quads,
                            text_x + layer.offset[0],
                            text_y + ascent + layer.offset[1],
                            col,
                            vp_w,
                            vp_h,
                        );
                    }

                    // The glyphs start white and `style.fill` multiplies into
                    // them below, so a label has one colour source.
                    let text_colour = apply_opacity([1.0, 1.0, 1.0, 1.0], opacity);
                    // The label origin is the top-left of the text box; add the
                    // ascent to reach the first baseline the quads are relative to.
                    let glyph_start = batch.len();
                    emit_glyph_quads(
                        &mut batch,
                        &layout.quads,
                        text_x,
                        text_y + ascent,
                        text_colour,
                        vp_w,
                        vp_h,
                    );
                    if label.style.fill.is_set() {
                        fill_vertices_from(
                            &mut batch,
                            glyph_start,
                            &label.style.fill,
                            [text_x, text_y],
                            [layout.total_width, layout.height],
                            &mut text_fills,
                            image.flatten().unwrap_or([1, 1]),
                        );
                    }
                    tint_vertices_from(&mut batch, glyph_start, label.style.tint);

                    // Inner shadow layers, over the glyphs. The cell is the
                    // letterform with a band eaten in from its edge, so it lands
                    // on the same quads the plain pass used.
                    for layer in label
                        .style
                        .inner_shadows
                        .iter()
                        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
                        .filter(|l| l.is_visible())
                    {
                        let style = GlyphStyle::from_inner_shadow(
                            layer.spread * ppp,
                            layer.blur * ppp,
                            layer.falloff,
                        );
                        if style == GlyphStyle::PLAIN {
                            continue;
                        }
                        let sl = if let Some(max_w) = label.max_width {
                            self.resources.content.glyph_atlas.layout_text_wrapped(
                                &label.text,
                                label.text_style.size,
                                label.text_style.font,
                                max_w,
                                ppp,
                                device,
                                style,
                            )
                        } else {
                            self.resources.content.glyph_atlas.layout_text(
                                &label.text,
                                label.text_style.size,
                                label.text_style.font,
                                ppp,
                                device,
                                style,
                            )
                        };
                        let col = apply_opacity(layer.colour.to_linear_rgba(), opacity);
                        emit_glyph_quads(
                            &mut batch,
                            &sl.quads,
                            text_x,
                            text_y + ascent,
                            col,
                            vp_w,
                            vp_h,
                        );
                    }

                    rotate_vertices_from(&mut batch, text_start, rot);

                    stamp_clip(&mut batch, label.clip.mask, label.clip.rect);
                    batches.push((label.z_order, label.style.fill.texture_id(), batch));
                }

                // --- Glyph runs (pre-positioned glyphs, drawn as given) ---
                for run in &frame.overlays.glyph_runs {
                    if run.glyphs.is_empty() || run.style.opacity <= 0.0 {
                        continue;
                    }
                    let animated_storage;
                    let run: &crate::renderer::types::GlyphRunItem = if run.animations.is_none() {
                        run
                    } else {
                        let mut owned = run.clone();
                        if let Some(anims) = owned.animations.take() {
                            anims.apply(
                                overlay_time,
                                &mut owned.transform,
                                &mut owned.style.opacity,
                                &mut owned.style.tint,
                            );
                        }
                        animated_storage = owned;
                        &animated_storage
                    };
                    super::overlay_style_check::warn_inert_style(
                        "GlyphRunItem",
                        crate::renderer::types::OverlayStyleSupport::for_glyphs(),
                        &run.style,
                    );
                    let image =
                        fill_image(&run.style.fill, &self.resources.content.overlay_textures);
                    if image == Some(None) {
                        continue;
                    }

                    // Resolve the anchor origin; skip the run when a world anchor
                    // is culled. Alignment shifts the whole run by its glyph-extent
                    // box, so default Left/Top leaves the authored positions.
                    let Some(origin) = crate::renderer::types::resolve_anchor_origin_at(
                        &run.anchoring.origin,
                        [vp_w, vp_h],
                        view,
                        proj,
                        world_origin,
                    ) else {
                        continue;
                    };
                    let Some(([min_x, min_y], [ext_w, ext_h])) = run.extent() else {
                        continue;
                    };
                    let run_x = origin[0]
                        + run.transform.translate[0]
                        + run.anchoring.align.x.align_shift(ext_w);
                    let run_y = origin[1]
                        + run.transform.translate[1]
                        + run.anchoring.align.y.align_shift(ext_h, Some(min_y));

                    let opacity = run.style.opacity.clamp(0.0, 1.0);
                    // Each glyph carries its multiplier through layout so the
                    // per-glyph entries stay aligned with the quads after
                    // zero-area skips. White is "the fill unmodified": the fill
                    // multiplies into the quads below.
                    let quads = self.resources.content.glyph_atlas.layout_glyph_run(
                        run.glyphs.iter().enumerate().map(|(i, g)| {
                            let tint = run
                                .glyph_tints
                                .get(i)
                                .copied()
                                .unwrap_or([1.0, 1.0, 1.0, 1.0]);
                            (g.glyph_id, g.x, g.y, apply_opacity(tint, opacity))
                        }),
                        run.text_style.size,
                        run.text_style.font,
                        ppp,
                        device,
                        GlyphStyle::PLAIN,
                    );
                    if quads.is_empty() {
                        continue;
                    }

                    let mut batch: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                    // The run turns about its glyph-extent box, shadows included.
                    let rot = OverlayRotation::new(
                        &run.transform,
                        OverlayRotation::pivot_point(
                            [run_x + min_x, run_y + min_y],
                            [ext_w, ext_h],
                            run.transform.pivot,
                        ),
                    );

                    // Shadow layers, back to front, behind the run. Per-glyph
                    // colours do not carry into a shadow: the whole layer draws
                    // in the layer colour, which is what makes it read as one
                    // silhouette rather than a blurred copy of the text.
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
                        let col = apply_opacity(layer.colour.to_linear_rgba(), opacity);
                        let sq = self.resources.content.glyph_atlas.layout_glyph_run(
                            run.glyphs.iter().map(|g| (g.glyph_id, g.x, g.y, col)),
                            run.text_style.size,
                            run.text_style.font,
                            ppp,
                            device,
                            style,
                        );
                        emit_glyph_quads_colored(
                            &mut batch,
                            &sq,
                            run_x + layer.offset[0],
                            run_y + layer.offset[1],
                            vp_w,
                            vp_h,
                        );
                    }

                    // Positions are already relative to the run origin, so the
                    // resolved origin is the only offset added; no ascent, unlike
                    // labels.
                    let glyph_start = batch.len();
                    emit_glyph_quads_colored(&mut batch, &quads, run_x, run_y, vp_w, vp_h);
                    if run.style.fill.is_set() {
                        fill_vertices_from(
                            &mut batch,
                            glyph_start,
                            &run.style.fill,
                            [run_x + min_x, run_y + min_y],
                            [ext_w, ext_h],
                            &mut text_fills,
                            image.flatten().unwrap_or([1, 1]),
                        );
                    }
                    tint_vertices_from(&mut batch, glyph_start, run.style.tint);

                    // Inner shadow layers, over the run, on the same quads the
                    // plain pass used.
                    for layer in run
                        .style
                        .inner_shadows
                        .iter()
                        .take(crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS)
                        .filter(|l| l.is_visible())
                    {
                        let style = GlyphStyle::from_inner_shadow(
                            layer.spread * ppp,
                            layer.blur * ppp,
                            layer.falloff,
                        );
                        if style == GlyphStyle::PLAIN {
                            continue;
                        }
                        let col = apply_opacity(layer.colour.to_linear_rgba(), opacity);
                        let sq = self.resources.content.glyph_atlas.layout_glyph_run(
                            run.glyphs.iter().map(|g| (g.glyph_id, g.x, g.y, col)),
                            run.text_style.size,
                            run.text_style.font,
                            ppp,
                            device,
                            style,
                        );
                        emit_glyph_quads_colored(&mut batch, &sq, run_x, run_y, vp_w, vp_h);
                    }

                    rotate_vertices_from(&mut batch, 0, rot);

                    stamp_clip(&mut batch, run.clip.mask, run.clip.rect);
                    batches.push((run.z_order, run.style.fill.texture_id(), batch));
                }

                // Upload atlas if new glyphs were rasterized.
                self.resources.content.glyph_atlas.upload_if_dirty(queue);

                // Stable sort preserves submission order for equal z_order values.
                batches.sort_by_key(|(z, _, _)| *z);
                // Flatten into one buffer, recording a draw segment per distinct
                // z_order run so this batch can interleave with other families,
                // and a run per stretch that samples the same fill image: the
                // image is bound per draw, as the shape pass binds it.
                use crate::renderer::overlay_draw_order::OverlayDrawSegment;
                let mut verts: Vec<crate::resources::OverlayTextVertex> = Vec::new();
                // The images text-stream items are filled from; an item's fill
                // bind group is its image's index plus one, zero being the
                // blank image.
                let mut text_images: Vec<crate::renderer::types::OverlayTextureId> = Vec::new();
                let mut text_runs: Vec<crate::resources::overlay::overlay_text::OverlayTextRun> =
                    Vec::new();
                for (z, texture, v) in batches {
                    let fill =
                        match texture {
                            None => 0,
                            Some(id) => {
                                let i = text_images.iter().position(|t| *t == id).unwrap_or_else(
                                    || {
                                        text_images.push(id);
                                        text_images.len() - 1
                                    },
                                );
                                i as u32 + 1
                            }
                        };
                    let start = verts.len() as u32;
                    verts.extend(v);
                    let count = verts.len() as u32 - start;
                    if count == 0 {
                        continue;
                    }
                    match text_runs.last_mut() {
                        Some(r) if r.fill == fill && r.vertex_start + r.vertex_count == start => {
                            r.vertex_count += count;
                        }
                        _ => text_runs.push(
                            crate::resources::overlay::overlay_text::OverlayTextRun {
                                vertex_start: start,
                                vertex_count: count,
                                fill,
                            },
                        ),
                    }
                    if ov.overlay_uses_zorder {
                        OverlayDrawSegment::push_text(
                            &mut ov.overlay_draw_segments,
                            z,
                            start,
                            count,
                            fill,
                        );
                    }
                }

                // Retained overlay groups: resolve each submitted group to its
                // compiled buffer, build its per-frame instance (translate,
                // opacity, and outer clip in framebuffer pixels), and record a
                // draw. Slot 0 of the instance buffer is the identity that every
                // immediate draw reads.
                let mut instances = vec![crate::resources::OverlayInstance::IDENTITY];
                // Collected text-stream draws, finished once the frame's fill
                // records exist: (vbuf, count, the group's fill records,
                // instance_index, z_order).
                let mut pending_text: Vec<(
                    crate::gpu::Buffer,
                    u32,
                    Option<crate::gpu::TextureView>,
                    u32,
                    i32,
                )> = Vec::new();
                // Collected shape-stream draws, finished after the instance buffer
                // exists: (shape_vbuf, count, shadow_tex, fill_tex,
                // instance_index, z_order).
                let mut pending_shapes: Vec<(
                    crate::gpu::Buffer,
                    u32,
                    crate::gpu::TextureView,
                    crate::gpu::TextureView,
                    u32,
                    i32,
                )> = Vec::new();
                for r in &frame.overlays.retained {
                    // Re-emit the group first if its baked glyph UVs went stale
                    // (atlas grew or pixels_per_point changed); cheap no-op otherwise.
                    if self.reemit_overlay_geometry_if_stale(device, queue, r.id, ppp) {
                        self.overlay_retained_counters.reemitted += 1;
                    }
                    // Resolve the group's tracks. Every channel they drive rides
                    // the instance, so an animated group never re-compiles; a
                    // static group skips the clone.
                    let animated_storage;
                    let r: &crate::renderer::types::RetainedOverlay = if r.animations.is_none() {
                        r
                    } else {
                        let mut owned = r.clone();
                        if let Some(anims) = owned.animations.take() {
                            anims.apply(
                                frame.overlays.time,
                                &mut owned.transform,
                                &mut owned.opacity,
                                &mut owned.tint,
                            );
                        }
                        animated_storage = owned;
                        &animated_storage
                    };
                    let (text, shape, (compiled_anchor, bounds)) =
                        match self.resources.content.overlay_geometry.get(r.id) {
                            Some(c) => {
                                let text = (c.vertex_count > 0).then(|| {
                                    (
                                        c.vertex_buf.clone(),
                                        c.vertex_count,
                                        c.text_fill_tex.clone(),
                                    )
                                });
                                let shape =
                                    match (&c.shape_instance_buf, &c.shadow_tex, &c.shape_fill_tex)
                                    {
                                        (Some(sv), Some(sh), Some(fl))
                                            if c.shape_instance_count > 0 =>
                                        {
                                            Some((
                                                sv.clone(),
                                                c.shape_instance_count,
                                                sh.clone(),
                                                fl.clone(),
                                            ))
                                        }
                                        _ => None,
                                    };
                                (text, shape, (c.anchor, c.bounds))
                            }
                            None => continue,
                        };
                    if text.is_none() && shape.is_none() {
                        continue;
                    }
                    // Resolve the group's anchor to a screen origin (a viewport
                    // corner, or a world point projected through the camera) and
                    // fold it into the translate, then shift by the alignment
                    // against the group's own compiled extent. The submission's
                    // translate composes on top (scroll/nudge). A world anchor
                    // that is culled skips the group for this frame.
                    //
                    // The submission's anchor wins over the one a group compiled
                    // from a single label carries, so a caller can re-anchor a
                    // compiled label without re-compiling it.
                    let submitted_origin = r.anchoring.map(|a| a.origin);
                    let translate = match submitted_origin.or(compiled_anchor) {
                        Some(a) => {
                            let Some(origin) = crate::renderer::types::resolve_anchor_origin_at(
                                &a,
                                [vp_w, vp_h],
                                view,
                                proj,
                                world_origin,
                            ) else {
                                continue;
                            };
                            // Alignment only means something against an extent,
                            // and a group compiled from one label already placed
                            // its own alignment at compile time, so it is applied
                            // only for an anchoring the submission set.
                            let align = match (r.anchoring, bounds) {
                                (Some(anchoring), Some((min, max))) => [
                                    anchoring.align.x.align_shift(max[0] - min[0]) - min[0],
                                    anchoring.align.y.align_shift(max[1] - min[1], None) - min[1],
                                ],
                                _ => [0.0, 0.0],
                            };
                            [
                                origin[0] + align[0] + r.transform.translate[0],
                                origin[1] + align[1] + r.transform.translate[1],
                            ]
                        }
                        None => r.transform.translate,
                    };
                    // Resolve the group's clip mask (if any) to this frame's
                    // clip-shape index and bbox, mirroring the immediate
                    // `stamp_clip`. The mask is registered as an immediate shape
                    // each frame; if it is absent this frame the group is unclipped.
                    let (clip_index, mask_bbox) =
                        match r.clip.mask.and_then(|id| clip_index_of.get(&id).copied()) {
                            Some(ci) => (ci as f32, Some(clip_bboxes[ci as usize])),
                            None => (-1.0, None),
                        };
                    // Outer bbox in framebuffer pixels: the group's explicit
                    // `clip_rect` if set, else the mask's own bbox (a cheap reject
                    // matching the shaped mask), else none.
                    let clip_rect = if let Some(rect) = r.clip.rect {
                        clip_rect_to_framebuffer(rect, ppp)
                    } else if let Some(bb) = mask_bbox {
                        bb
                    } else {
                        [0.0, 0.0, 0.0, 0.0]
                    };
                    let instance_index = instances.len() as u32;
                    let mut instance = crate::resources::OverlayInstance {
                        translate,
                        opacity: r.opacity,
                        clip_index,
                        clip_rect,
                        tint: r.tint,
                        ..crate::resources::OverlayInstance::IDENTITY
                    };
                    // Logical pixels, like `translate`: the instance's transform
                    // is applied to vertex positions, which are logical. Only
                    // `clip_rect` is in framebuffer pixels, because it is
                    // compared against @builtin(position).
                    instance.set_transform(&r.transform);
                    instances.push(instance);
                    // Past every skip: the group has an instance slot and at
                    // least one of the two streams below records a draw.
                    self.overlay_retained_counters.drawn += 1;
                    if let Some((vbuf, vcount, fill_tex)) = text {
                        pending_text.push((vbuf, vcount, fill_tex, instance_index, r.z_order));
                    }
                    if let Some((svbuf, svcount, shbuf, fill_tex)) = shape {
                        pending_shapes.push((
                            svbuf,
                            svcount,
                            shbuf,
                            fill_tex,
                            instance_index,
                            r.z_order,
                        ));
                    }
                }

                if !verts.is_empty() || !pending_text.is_empty() || !pending_shapes.is_empty() {
                    let vertex_buf = ov.overlay_text_vbuf.write(device, queue, &verts);
                    Self::ensure_overlay_viewport_buf(ov, device, queue, vp_w, vp_h);
                    let clip_buf = ov.overlay_bindings.clip.write_changed(
                        device,
                        queue,
                        clip_data(&clip_shapes),
                    );
                    // The instance stream, shared with the shape pass (immediate
                    // draws read the identity at slot 0; retained draws their own).
                    ov.overlay_bindings
                        .instances
                        .write_changed(device, queue, &instances);
                    ov.overlay_bindings.instances_uniform.write_changed(
                        device,
                        queue,
                        &crate::resources::OverlayInstance::padded_uniform_bytes(&instances),
                    );
                    ov.overlay_instances_ready = true;

                    // The frame's fill records and the fill bind groups over
                    // them: the blank image first, then one per image a
                    // text-stream item is filled from. Last frame's groups are
                    // reused where the pair is the same, and dropped otherwise.
                    let text_records = ov.overlay_bindings.text_fills.write_changed(
                        device,
                        queue,
                        text_fills.texels(),
                    );
                    let mut fill_cache = std::mem::take(&mut ov.overlay_bindings.text_fill_bgs);
                    let mut fill_used = Default::default();
                    let mut fill_bind_groups = vec![self.resources.overlay_fill_bind_group(
                        device,
                        &mut fill_cache,
                        &mut fill_used,
                        &text_records,
                        None,
                    )];
                    for id in &text_images {
                        let view = self
                            .resources
                            .content
                            .overlay_textures
                            .get(*id)
                            .map(|t| t.view.clone());
                        fill_bind_groups.push(self.resources.overlay_fill_bind_group(
                            device,
                            &mut fill_cache,
                            &mut fill_used,
                            &text_records,
                            view.as_ref(),
                        ));
                    }

                    // Finish retained text-stream draws: each reads its own
                    // group's fill records, or the frame's when it has none.
                    for (vbuf, vcount, fill_tex, instance_index, z) in pending_text.drain(..) {
                        let records = fill_tex.as_ref().unwrap_or(&text_records);
                        let fill_bind_group = self.resources.overlay_fill_bind_group(
                            device,
                            &mut fill_cache,
                            &mut fill_used,
                            records,
                            None,
                        );
                        let draw_index = ov.overlay_retained_draws.len() as u32;
                        ov.overlay_retained_draws.push(
                            crate::renderer::overlay_buffers::RetainedDraw {
                                vertex_buf: vbuf,
                                vertex_count: vcount,
                                instance_index,
                                fill_bind_group,
                            },
                        );
                        crate::renderer::overlay_draw_order::OverlayDrawSegment::push_retained(
                            &mut ov.overlay_draw_segments,
                            z,
                            draw_index,
                        );
                    }

                    // Finish retained shape-stream draws: build each group's shape
                    // bind group (its shadow-layer texture plus the shared clip and
                    // viewport) and record its draw + segment.
                    if pending_shapes.is_empty() {
                        ov.overlay_bindings.retained_shape_bgs.clear();
                    } else {
                        self.resources.ensure_overlay_shape_pipeline(device);
                        let vp_buf = ov.overlay_viewport_buf.as_ref().unwrap();
                        let inst_buf = ov
                            .overlay_bindings
                            .instances_uniform
                            .buffer()
                            .expect("written just above")
                            .clone();
                        if let Some(sh_bgl) = self.resources.overlay_shape.shadow_bgl.as_ref() {
                            let mut previous =
                                std::mem::take(&mut ov.overlay_bindings.retained_shape_bgs);
                            for (svbuf, svcount, shbuf, fill_tex, instance_index, z) in
                                pending_shapes.drain(..)
                            {
                                let fill_bind_group = self.resources.overlay_fill_bind_group(
                                    device,
                                    &mut fill_cache,
                                    &mut fill_used,
                                    &fill_tex,
                                    None,
                                );
                                let mut cached = previous.remove(&shbuf).unwrap_or_else(
                                    crate::resources::cached_bind_group::CachedBindGroup::new,
                                );
                                let key = [clip_buf.clone(), vp_buf.clone(), inst_buf.clone()];
                                let bind_group = cached.get(key, || {
                                    device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                                        label: Some("overlay_retained_shape_bg"),
                                        layout: sh_bgl,
                                        entries: &[
                                            crate::gpu::BindGroupEntry {
                                                binding: 0,
                                                resource: crate::gpu::BindingResource::TextureView(
                                                    &shbuf,
                                                ),
                                            },
                                            crate::gpu::BindGroupEntry {
                                                binding: 1,
                                                resource: crate::resources::overlay::overlay_shape::clip_binding(
                                                    &clip_buf,
                                                ),
                                            },
                                            crate::gpu::BindGroupEntry {
                                                binding: 2,
                                                resource: vp_buf.as_entire_binding(),
                                            },
                                            crate::gpu::BindGroupEntry {
                                                binding: 3,
                                                resource: crate::resources::overlay::overlay_shape::group_instance_binding(
                                                    &inst_buf,
                                                ),
                                            },
                                        ],
                                    })
                                });
                                ov.overlay_bindings.retained_shape_bgs.insert(shbuf, cached);
                                let draw_index = ov.overlay_retained_shape_draws.len() as u32;
                                ov.overlay_retained_shape_draws.push(
                                    crate::renderer::overlay_buffers::RetainedShapeDraw {
                                        instance_buf: svbuf,
                                        instance_count: svcount,
                                        bind_group,
                                        instance_index,
                                        fill_bind_group,
                                    },
                                );
                                crate::renderer::overlay_draw_order::OverlayDrawSegment::push_retained_shape(
                                    &mut ov.overlay_draw_segments,
                                    z,
                                    draw_index,
                                );
                            }
                        }
                    }
                    let bgl = self.resources.overlay_text.bgl.as_ref().unwrap();
                    let sampler = self.resources.overlay_text.sampler.as_ref().unwrap();
                    let vp_buf = ov.overlay_viewport_buf.as_ref().unwrap();
                    let atlas_view = &self.resources.content.glyph_atlas.view;
                    let key = (atlas_view.clone(), clip_buf.clone(), vp_buf.clone());
                    let bind_group = ov.overlay_bindings.label_bg.get(key, || {
                        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                            label: Some("overlay_label_bg"),
                            layout: bgl,
                            entries: &[
                                crate::gpu::BindGroupEntry {
                                    binding: 0,
                                    resource: crate::gpu::BindingResource::TextureView(atlas_view),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 1,
                                    resource: crate::gpu::BindingResource::Sampler(sampler),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 2,
                                    resource:
                                        crate::resources::overlay::overlay_shape::clip_binding(
                                            &clip_buf,
                                        ),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 3,
                                    resource: vp_buf.as_entire_binding(),
                                },
                            ],
                        })
                    });
                    ov.overlay_bindings.text_fill_bgs = fill_used;
                    ov.label_gpu_data = Some(crate::resources::LabelGpuData {
                        vertex_buf,
                        runs: text_runs,
                        fill_bind_groups,
                        bind_group,
                        _clip_buf: clip_buf,
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
        ov: &mut crate::renderer::ViewportOverlay,
    ) {
        // ------------------------------------------------------------------
        // SDF overlay shapes
        // ------------------------------------------------------------------
        ov.overlay_shape_gpu_data = None;
        // The bottom-left orientation indicator draws as overlay shapes too.
        let axes_shapes = self.axes_overlay_items(frame);
        if frame.overlays.shapes.is_empty() && axes_shapes.is_empty() {
            return;
        }
        let vp_w = frame.camera.viewport_size[0];
        let vp_h = frame.camera.viewport_size[1];
        if vp_w <= 0.0 || vp_h <= 0.0 {
            return;
        }
        Self::ensure_overlay_viewport_buf(ov, device, queue, vp_w, vp_h);
        Self::ensure_overlay_instances_default(ov, device, queue);
        let mut sorted: Vec<&crate::renderer::types::OverlayShapeItem> = frame
            .overlays
            .shapes
            .iter()
            .chain(axes_shapes.iter())
            .collect();
        sorted.sort_by_key(|s| s.z_order);

        let is_vector = |s: &crate::renderer::types::OverlayShapeItem| {
            matches!(s.shape, crate::renderer::types::OverlayShape::Vector { .. })
        };
        let has_instances = sorted.iter().any(|s| {
            !is_vector(s) && (s.style.fill.texture_id().is_some() || s.style.backdrop.blur <= 0.0)
        });
        let has_blur = sorted
            .iter()
            .any(|s| s.style.backdrop.blur > 0.0 && s.style.fill.texture_id().is_none());
        if has_instances {
            self.resources.ensure_overlay_fill_resources(device, queue);
            self.resources.ensure_overlay_shape_pipeline(device);
        }
        if has_blur {
            self.resources.ensure_overlay_shape_tex_pipeline(device);
            self.resources.ensure_backdrop_blur_pipeline(device);
            self.resources.ensure_dyn_res_pipeline(device);
        }

        // Untextured shapes, in z-sorted order, then one list per image in
        // order of first appearance; they share one instance buffer.
        let mut solid_instances: Vec<crate::resources::OverlayShapeInstance> = Vec::new();
        let mut tex_groups: Vec<(
            crate::renderer::types::OverlayTextureId,
            Vec<crate::resources::OverlayShapeInstance>,
        )> = Vec::new();
        // One entry per drawn textured shape, in z-sorted emission order:
        // `(z_order, group_idx, instance_in_group)`. Each becomes its own draw
        // segment so a solid shape can layer between two shapes sharing an
        // image, instead of the whole image group collapsing to its lowest z.
        let mut tex_shape_segs: Vec<(i32, usize, u32)> = Vec::new();
        // Stacked shadow layers for every shape, shared across the whole
        // frame. Each shape references a contiguous run (outer layers first,
        // then inner) via its `shadow_meta`.
        let mut shadow_layers: Vec<crate::resources::OverlayShadowLayerGpu> = Vec::new();
        // The fill records the shapes' gradients and images read.
        let mut fills = super::overlay_fill::FillTable::new();
        let mut blur_verts: Vec<crate::resources::OverlayBackdropVertex> = Vec::new();
        let mut max_blur_radius: f32 = 0.0;

        let overlay_time = frame.overlays.time;

        // The SDF clip registry, plus each mask's bounding box (framebuffer
        // pixels) for the shader's cheap pre-reject. Built from
        // `frame.overlays.shapes` (not `sorted`) so the mask indices match
        // exactly what the label pass assigned, since text and shapes reference
        // the same masks by index. The bbox is looked up by that same index, so
        // `clip_rect` and `clip_index` always describe the same mask.
        let ppp = frame.camera.pixels_per_point;
        // World anchors project through the camera relative to the
        // world origin, with the origin subtracted in f64.
        let world_origin = self.origin.origin();
        let view = &self.origin.camera(&frame.camera.render_camera).view();
        let proj = &frame.camera.render_camera.projection();
        let (clip_shapes, clip_index_of, clip_bboxes, clips_dropped) = build_clip_shapes(
            &frame.overlays.shapes,
            ppp,
            [vp_w, vp_h],
            view,
            proj,
            world_origin,
        );
        self.overlay_retained_counters.clip_masks_dropped = clips_dropped;

        for shape_orig in &sorted {
            // Mask-only shapes contribute a clip rectangle but are
            // not drawn themselves.
            if shape_orig.provides_mask.is_some() {
                continue;
            }
            // A vector shape has no distance field, so its fill (an image
            // fill included) and shadows draw as triangles in the label pass.
            // Only a backdrop blur comes here, as the tessellated triangles
            // through the backdrop pipeline. As on an analytic shape, an image
            // fill takes the place of the backdrop.
            if let crate::renderer::types::OverlayShape::Vector {
                subpaths,
                fill_rule,
            } = &shape_orig.shape
            {
                if shape_orig.style.fill.texture_id().is_some()
                    || shape_orig.style.backdrop.blur <= 0.0
                {
                    continue;
                }
                let mut owned = (*shape_orig).clone();
                if let Some(anims) = owned.animations.take() {
                    anims.apply(
                        overlay_time,
                        &mut owned.transform,
                        &mut owned.style.opacity,
                        &mut owned.style.tint,
                    );
                }
                let Some(tl) = owned.resolve_top_left_at([vp_w, vp_h], view, proj, world_origin)
                else {
                    continue;
                };
                owned.transform.translate = tl;
                if owned.style.opacity <= 0.0 {
                    continue;
                }
                let mesh =
                    super::overlay_vector::tessellate(subpaths, *fill_rule, VECTOR_FILL_TOLERANCE);
                if mesh.indices.is_empty() {
                    continue;
                }
                let positions = transform_vector_positions(&mesh.positions, &owned);
                let backdrop = &owned.style.backdrop;
                max_blur_radius = max_blur_radius.max(backdrop.blur);
                // Like an analytic backdrop, it is not clipped by a mask.
                push_tessellated_backdrop_verts(
                    &mut blur_verts,
                    &positions,
                    &mesh.indices,
                    [
                        0.0,
                        backdrop.saturation,
                        backdrop.brightness,
                        backdrop.hue_shift,
                    ],
                    vp_w,
                    vp_h,
                );
                continue;
            }
            super::overlay_style_check::warn_inert_style(
                "OverlayShapeItem",
                crate::renderer::types::OverlayStyleSupport::for_shape(&shape_orig.shape),
                &shape_orig.style,
            );
            // Clone so per-frame animation overrides are local and
            // the input frame data stays untouched.
            let mut owned: crate::renderer::types::OverlayShapeItem = (*shape_orig).clone();
            if let Some(anims) = owned.animations.take() {
                anims.apply(
                    overlay_time,
                    &mut owned.transform,
                    &mut owned.style.opacity,
                    &mut owned.style.tint,
                );
            }
            // Resolve the anchor origin + animated position + alignment
            // to an absolute top-left, then draw the shape as if it were
            // positioned there. A culled world anchor skips the shape for
            // this frame. Everything below (fill, texture, clip, shadow,
            // rotation pivot) reads `position`, so this one resolution
            // covers all of them. The `animations.position` track has
            // already been applied above, so it drives the nudge that is
            // layered on the resolved origin here.
            let Some(resolved_tl) =
                owned.resolve_top_left_at([vp_w, vp_h], view, proj, world_origin)
            else {
                continue;
            };
            owned.transform.translate = resolved_tl;
            let shape = &owned;
            let resolved_opacity = shape.style.opacity;

            if resolved_opacity <= 0.0 {
                continue;
            }
            // An image fill whose texture is gone skips the shape.
            let image = match shape.style.fill.texture_id() {
                Some(id) => match self.resources.content.overlay_textures.get(id) {
                    Some(t) => Some((id, t.size)),
                    None => continue,
                },
                None => None,
            };

            let hw = shape.size[0] * 0.5;
            let hh = shape.size[1] * 0.5;
            let cx = shape.transform.translate[0] + hw;
            let cy = shape.transform.translate[1] + hh;

            // Outer-shadow extent that reaches past the shape edge.
            // Inner shadows stay inside the shape, so they need no quad
            // padding.
            let mut shadow_pad = 0.0f32;
            for l in &shape.style.shadows {
                shadow_pad = shadow_pad.max(l.extent());
            }

            // Extra quad expansion for shapes whose stroke extends
            // beyond the item's position/size bounding box.
            let extra_expand = match &shape.shape {
                crate::renderer::types::OverlayShape::Line { thickness, .. } => thickness * 0.5,
                _ => 0.0,
            };

            // Where the shape lands under its transform, with room for
            // its outer shadows and a pixel of anti-aliasing. A map with
            // no inverse (a zero stretch) draws nothing.
            let Some(shape_frame) = overlay_geometry::shape_frame(
                &shape.transform,
                [cx, cy],
                [hw + extra_expand, hh + extra_expand],
                shadow_pad + 1.0,
            ) else {
                continue;
            };

            let (shape_type, radii) = encode_overlay_shape(&shape.shape, hw, hh);

            // The flat colour: a solid fill, a gradient's first stop, or a
            // texture's tint, with the item's opacity on every alpha and then
            // its tint on every component. It stops there: a tint never
            // reaches a shadow layer, matching the group tint the shaders
            // apply.
            let tint = shape.style.tint;
            let mut fc = super::overlay_fill::leading_colour(&shape.style.fill);
            fc[3] *= resolved_opacity;
            for (c, t) in fc.iter_mut().zip(tint) {
                *c *= t;
            }

            // Build the stacked shadow layers for this shape: outer
            // layers first, then inner, appended to the shared frame
            // texture. The shape and backdrop pipelines both read it,
            // so a shape takes the same layers whichever one draws it.
            let base_index = shadow_layers.len();
            let mut outer_count = 0usize;
            let mut inner_count = 0usize;
            let max_layers = crate::renderer::types::OVERLAY_MAX_SHADOW_LAYERS;
            for l in shape.style.shadows.iter().take(max_layers) {
                let mut col = l.colour.to_linear_rgba();
                col[3] *= resolved_opacity;
                shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                    colour: col,
                    params: [l.blur, l.offset[0], l.offset[1], 0.0],
                    params2: [l.spread, l.falloff, 0.0, 0.0],
                });
                outer_count += 1;
            }
            for l in shape.style.inner_shadows.iter().take(max_layers) {
                let mut col = l.colour.to_linear_rgba();
                col[3] *= resolved_opacity;
                shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                    colour: col,
                    params: [l.blur, l.offset[0], l.offset[1], 1.0],
                    params2: [l.spread, l.falloff, 0.0, 0.0],
                });
                inner_count += 1;
            }
            let shadow_index = [base_index as f32, outer_count as f32, inner_count as f32];
            let pivot_local = shape.transform.pivot;

            // Resolve the clip mask for this shape. Missing or unmatched ids
            // fall through to a -1 index and an all-zero rect, which the
            // shaders treat as no clipping. The bbox and index come from the
            // same registry entry, so they never disagree.
            let clip_index_i = shape
                .clip
                .mask
                .and_then(|id| clip_index_of.get(&id).copied())
                .unwrap_or(-1);
            let mask_rect = if clip_index_i >= 0 {
                clip_bboxes[clip_index_i as usize]
            } else {
                [0.0, 0.0, 0.0, 0.0]
            };
            let clip_rect = match shape.clip.rect {
                Some(r) => combine_clip_rects(mask_rect, clip_rect_to_framebuffer(r, ppp)),
                None => mask_rect,
            };
            let clip_index = clip_index_i as f32;

            if image.is_none() && shape.style.backdrop.blur > 0.0 {
                max_blur_radius = max_blur_radius.max(shape.style.backdrop.blur);
                // A backdrop blur draws through the backdrop pipeline, as the
                // shape's quad in six vertices; it is composited separately
                // and is not clipped by a mask.
                for (pos, local) in shape_frame.corners() {
                    blur_verts.push(crate::resources::OverlayBackdropVertex {
                        position: overlay_local_px(pos[0], pos[1], vp_w, vp_h),
                        local_pos: local,
                        tint: fc,
                        half_size: [hw, hh],
                        radii,
                        shape_meta: [shape_type, -1.0, pivot_local[0], pivot_local[1]],
                        clip_rect: [0.0; 4],
                        shadow_index,
                        filters: [
                            0.0,
                            shape.style.backdrop.saturation,
                            shape.style.backdrop.brightness,
                            shape.style.backdrop.hue_shift,
                        ],
                        linv: shape_frame.linv,
                        item_params: [shape_frame.item_scale, 0.0, 0.0, 0.0],
                    });
                }
                continue;
            }

            // A gradient or an image is a fill record, evaluated in the
            // shape's own frame; a solid fill is the flat colour alone.
            let fill_index = match &shape.style.fill {
                crate::renderer::types::OverlayFill::Solid(_) => super::overlay_fill::NO_FILL,
                fill => fills.push(
                    fill,
                    [hw, hh],
                    resolved_opacity,
                    tint,
                    image.map_or([1, 1], |(_, size)| size),
                ),
            };
            let instance = crate::resources::OverlayShapeInstance {
                quad: shape_frame.quad,
                pivot_half: [shape_frame.pivot_px[0], shape_frame.pivot_px[1], hw, hh],
                linv: shape_frame.linv,
                fill_colour: fc,
                radii,
                shadow_meta: [
                    shadow_index[0],
                    shadow_index[1],
                    shadow_index[2],
                    shape_type,
                ],
                clip_rect,
                misc: [
                    clip_index,
                    pivot_local[0],
                    pivot_local[1],
                    shape_frame.item_scale,
                ],
                fill: [fill_index, 0.0, 0.0, 0.0],
            };
            match image {
                Some((tex_id, _)) => {
                    let group_idx = tex_groups
                        .iter()
                        .position(|(id, _)| *id == tex_id)
                        .unwrap_or_else(|| {
                            tex_groups.push((tex_id, Vec::new()));
                            tex_groups.len() - 1
                        });
                    let group = &mut tex_groups[group_idx].1;
                    tex_shape_segs.push((shape.z_order, group_idx, group.len() as u32));
                    group.push(instance);
                }
                None => {
                    let instance_start = solid_instances.len() as u32;
                    solid_instances.push(instance);
                    if ov.overlay_uses_zorder {
                        crate::renderer::overlay_draw_order::OverlayDrawSegment::push_shape(
                            &mut ov.overlay_draw_segments,
                            shape.z_order,
                            instance_start,
                            1,
                            0,
                        );
                    }
                }
            }
        }

        // One instance buffer: the untextured run, then each image's run.
        let solid_count = solid_instances.len() as u32;
        let mut batches: Vec<crate::resources::OverlayShapeBatch> = Vec::new();
        if solid_count > 0 {
            batches.push(crate::resources::OverlayShapeBatch {
                instance_start: 0,
                instance_count: solid_count,
                fill: 0,
            });
        }
        let mut group_start = Vec::with_capacity(tex_groups.len());
        let mut instances = solid_instances;
        for (i, (_, group)) in tex_groups.iter().enumerate() {
            group_start.push(instances.len() as u32);
            batches.push(crate::resources::OverlayShapeBatch {
                instance_start: instances.len() as u32,
                instance_count: group.len() as u32,
                fill: i as u32 + 1,
            });
            instances.extend_from_slice(group);
        }
        if ov.overlay_uses_zorder {
            // One draw segment per textured shape at its own z_order
            // (coalescing contiguous same-image, same-z runs), so a solid
            // shape can layer between two shapes that share an image.
            for &(z, group_idx, k) in &tex_shape_segs {
                crate::renderer::overlay_draw_order::OverlayDrawSegment::push_shape(
                    &mut ov.overlay_draw_segments,
                    z,
                    group_start[group_idx] + k,
                    1,
                    group_idx as u32 + 1,
                );
            }
        }

        let instance_buf =
            (!instances.is_empty()).then(|| ov.overlay_shape_vbuf.write(device, queue, &instances));

        // One shadow-layer texture per frame, read by the shape and backdrop
        // pipelines. A layer is always present, even when no shape has
        // shadows, so the binding has something to point at.
        let shadow_tex = if instance_buf.is_some() || has_blur {
            if shadow_layers.is_empty() {
                shadow_layers.push(crate::resources::OverlayShadowLayerGpu {
                    colour: [0.0; 4],
                    params: [0.0; 4],
                    params2: [0.0, 1.0, 0.0, 0.0],
                });
            }
            Some(
                ov.overlay_bindings
                    .shape_shadow
                    .write_changed(device, queue, &shadow_layers),
            )
        } else {
            None
        };

        let (shadow_bind_group, shape_clip_buf, fill_bind_groups) = if let (true, Some(shadow)) =
            (instance_buf.is_some(), shadow_tex.as_ref())
        {
            let clip_buf =
                ov.overlay_bindings
                    .clip
                    .write_changed(device, queue, clip_data(&clip_shapes));
            let vp_buf = ov.overlay_viewport_buf.as_ref().unwrap();
            let inst_buf = ov
                .overlay_bindings
                .instances_uniform
                .buffer()
                .expect("the identity instance is written before the shapes")
                .clone();
            let key = (
                shadow.clone(),
                clip_buf.clone(),
                vp_buf.clone(),
                inst_buf.clone(),
            );
            let cache = &mut ov.overlay_bindings.shape_shadow_bg;
            let bg = self.resources.overlay_shape.shadow_bgl.as_ref().map(|bgl| {
                    cache.get(key, || {
                        device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                            label: Some("overlay_shape_shadow_bg"),
                            layout: bgl,
                            entries: &[
                                crate::gpu::BindGroupEntry {
                                    binding: 0,
                                    resource: crate::gpu::BindingResource::TextureView(shadow),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 1,
                                    resource:
                                        crate::resources::overlay::overlay_shape::clip_binding(
                                            &clip_buf,
                                        ),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 2,
                                    resource: vp_buf.as_entire_binding(),
                                },
                                crate::gpu::BindGroupEntry {
                                    binding: 3,
                                    resource:
                                        crate::resources::overlay::overlay_shape::group_instance_binding(
                                            &inst_buf,
                                        ),
                                },
                            ],
                        })
                    })
                });
            // The fill bind groups: the frame's records with the blank
            // image, then with each image a textured shape samples.
            let records =
                ov.overlay_bindings
                    .shape_fills
                    .write_changed(device, queue, fills.texels());
            let mut cache = std::mem::take(&mut ov.overlay_bindings.shape_fill_bgs);
            let mut used = Default::default();
            let mut groups = vec![
                self.resources
                    .overlay_fill_bind_group(device, &mut cache, &mut used, &records, None),
            ];
            for (tex_id, _) in &tex_groups {
                let view = self
                    .resources
                    .content
                    .overlay_textures
                    .get(*tex_id)
                    .map(|t| t.view.clone());
                groups.push(self.resources.overlay_fill_bind_group(
                    device,
                    &mut cache,
                    &mut used,
                    &records,
                    view.as_ref(),
                ));
            }
            ov.overlay_bindings.shape_fill_bgs = used;
            (bg, Some(clip_buf), groups)
        } else {
            (None, None, Vec::new())
        };

        // Clip-mask bind group (group 1) for the backdrop pipeline.
        let (tex_clip_bind_group, tex_clip_buf) = if has_blur {
            if let Some(bgl) = self.resources.overlay_shape.tex_clip_bgl.as_ref() {
                let clip_buf =
                    ov.overlay_bindings
                        .clip
                        .write_changed(device, queue, clip_data(&clip_shapes));
                let vp_buf = ov.overlay_viewport_buf.as_ref().unwrap();
                let shadow = shadow_tex.as_ref().unwrap();
                let key = (clip_buf.clone(), vp_buf.clone(), shadow.clone());
                let bg = ov.overlay_bindings.tex_clip_bg.get(key, || {
                    device.create_bind_group(&crate::gpu::BindGroupDescriptor {
                        label: Some("overlay_shape_tex_clip_bg"),
                        layout: bgl,
                        entries: &[
                            crate::gpu::BindGroupEntry {
                                binding: 0,
                                resource: crate::resources::overlay::overlay_shape::clip_binding(
                                    &clip_buf,
                                ),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 1,
                                resource: vp_buf.as_entire_binding(),
                            },
                            crate::gpu::BindGroupEntry {
                                binding: 2,
                                resource: crate::gpu::BindingResource::TextureView(shadow),
                            },
                        ],
                    })
                });
                (Some(bg), Some(clip_buf))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let blur_vbuf = (!blur_verts.is_empty())
            .then(|| ov.overlay_shape_blur_vbuf.write(device, queue, &blur_verts));

        if instance_buf.is_some() || blur_vbuf.is_some() {
            ov.overlay_shape_gpu_data = Some(crate::resources::OverlayShapeGpuData {
                instance_buf,
                batches,
                fill_bind_groups,
                shadow_bind_group,
                _shadow_tex: shadow_tex,
                _clip_buf: shape_clip_buf,
                tex_clip_bind_group,
                _tex_clip_buf: tex_clip_buf,
                blur_vertex_buf: blur_vbuf,
                blur_vertex_count: blur_verts.len() as u32,
                max_blur_radius,
            });
        }
    }

    /// Sort the overlay draw list into paint order. The family passes have
    /// already recorded their own segments; this orders them by
    /// `(z_order, family_rank)` for the ordered emit path.
    pub(super) fn finalize_overlay_draw_order(ov: &mut crate::renderer::ViewportOverlay) {
        // When no overlay uses z_order, the family passes recorded nothing and
        // the emit path keeps its fixed order; there is nothing to finalize.
        if !ov.overlay_uses_zorder {
            return;
        }
        use crate::renderer::overlay_draw_order::sort_overlay_segments;
        sort_overlay_segments(&mut ov.overlay_draw_segments);
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
        .provides_mask(10);
        let child = OverlayShapeItem::new(OverlayShape::Circle, [110.0, 60.0], [80.0, 80.0])
            .provides_mask(20)
            .with_clip(10);
        let (gpu, map, bboxes, _) = build_clip_shapes(
            &[parent, child],
            2.0,
            [1000.0, 1000.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
            glam::DVec3::ZERO,
        );

        assert_eq!(gpu.len(), 2);
        assert_eq!(map[&10], 0);
        assert_eq!(map[&20], 1);

        // Parent: centre and half-size scaled by ppp, corner radius scaled, no parent.
        assert_eq!(gpu[0].centre, [400.0, 200.0]);
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
        .provides_mask(5);
        let second = OverlayShapeItem::new(
            OverlayShape::Rect { corner_radius: 0.0 },
            [500.0, 500.0],
            [50.0, 50.0],
        )
        .provides_mask(5);
        let (gpu, map, bboxes, _) = build_clip_shapes(
            &[first, second],
            1.0,
            [1000.0, 1000.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
            glam::DVec3::ZERO,
        );

        assert_eq!(gpu.len(), 1);
        assert_eq!(map[&5], 0);
        // The bbox is the first mask's, matching the index the first mask claimed.
        assert_eq!(bboxes[0], [0.0, 0.0, 100.0, 100.0]);
    }

    #[test]
    fn masks_past_the_limit_are_dropped_and_counted() {
        let max = crate::resources::overlay::overlay_shape::MAX_OVERLAY_CLIP_SHAPES;
        let masks: Vec<_> = (0..max as u32 + 3)
            .map(|id| {
                OverlayShapeItem::new(
                    OverlayShape::Rect { corner_radius: 0.0 },
                    [0.0, 0.0],
                    [10.0, 10.0],
                )
                .provides_mask(id)
            })
            .collect();
        let (gpu, map, bboxes, dropped) = build_clip_shapes(
            &masks,
            1.0,
            [1000.0, 1000.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
            glam::DVec3::ZERO,
        );
        assert_eq!(gpu.len(), max);
        assert_eq!(bboxes.len(), max);
        assert_eq!(dropped, 3);
        // A dropped mask has no index, so an item naming it draws unclipped.
        assert!(!map.contains_key(&(max as u32)));
        assert_eq!(map[&(max as u32 - 1)], max as i32 - 1);
    }

    #[test]
    fn non_mask_shapes_are_ignored() {
        let drawn = OverlayShapeItem::new(OverlayShape::Circle, [0.0, 0.0], [10.0, 10.0]);
        let (gpu, map, bboxes, _) = build_clip_shapes(
            &[drawn],
            1.0,
            [1000.0, 1000.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
            glam::DVec3::ZERO,
        );
        assert!(gpu.is_empty());
        assert!(map.is_empty());
        assert!(bboxes.is_empty());
    }

    #[test]
    fn a_vector_shape_registers_no_mask() {
        // A vector path has no distance field, so it cannot be a mask. It must
        // register nothing rather than encode as its bounding box, which would
        // clip the wrong region silently. An analytic mask in the same frame is
        // unaffected and keeps index 0.
        use crate::renderer::types::{FillRule, SubPath};
        let vector = OverlayShapeItem::new(
            OverlayShape::Vector {
                subpaths: vec![SubPath::default()],
                fill_rule: FillRule::NonZero,
            },
            [0.0, 0.0],
            [100.0, 100.0],
        )
        .provides_mask(7);
        let analytic = OverlayShapeItem::new(OverlayShape::Circle, [10.0, 10.0], [40.0, 40.0])
            .provides_mask(8);
        let (gpu, map, bboxes, _) = build_clip_shapes(
            &[vector, analytic],
            1.0,
            [1000.0, 1000.0],
            &glam::Mat4::IDENTITY,
            &glam::Mat4::IDENTITY,
            glam::DVec3::ZERO,
        );

        assert_eq!(gpu.len(), 1, "the vector shape must not be encoded");
        assert!(!map.contains_key(&7), "mask 7 must be absent, not wrong");
        assert_eq!(map[&8], 0);
        assert_eq!(bboxes.len(), 1);
    }
}
