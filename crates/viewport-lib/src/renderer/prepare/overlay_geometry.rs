//! CPU tessellation and fill helpers for overlay rects, shapes, and polylines.

use super::*;
use crate::Colour;

pub(super) fn polygon_area(points: &[[f32; 2]]) -> f32 {
    let mut area = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area * 0.5
}

/// Emit a closed polyline's fill as a triangle list with an anti-aliased
/// edge, recording a gradient or texture fill in `fills`.
///
/// The fill spans the points' bounding box. `uvs`, when given (one per input
/// point), place a texture fill's image instead: each vertex samples the UV its
/// outline point or edge carries. `image_size` is the pixel size of a texture
/// fill's image.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_filled_polyline(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    points: &[[f32; 2]],
    fill: &crate::renderer::types::OverlayFill,
    opacity: f32,
    vp_w: f32,
    vp_h: f32,
    fills: &mut super::overlay_fill::FillTable,
    uvs: Option<&[[f32; 2]]>,
    image_size: [u32; 2],
) {
    let Some((min, max)) = polyline_bounds(points) else {
        return;
    };
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return;
    }
    let mesh = super::overlay_vector::tessellate_fringed(
        std::slice::from_ref(&points.to_vec()),
        crate::renderer::types::FillRule::NonZero,
    );
    let half = [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5];
    let centre = [min[0] + half[0], min[1] + half[1]];
    let (base_colour, index) = fill_record(fill, opacity, half, fills, image_size);
    // The local coordinate that lands an explicit UV where the fill module's
    // box mapping reads it back.
    let uv_half = [half[0].max(0.5), half[1].max(0.5)];
    let uvs = uvs.filter(|u| u.len() == points.len());
    for &i in &mesh.indices {
        let i = i as usize;
        let p = mesh.positions[i];
        let local = match uvs {
            Some(uvs) => {
                let (from, to, t) = mesh.sources[i];
                let (a, b) = (uvs[from], uvs[to]);
                let uv = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
                [
                    uv[0] * 2.0 * uv_half[0] - half[0],
                    uv[1] * 2.0 * uv_half[1] - half[1],
                ]
            }
            None => [p[0] - centre[0], p[1] - centre[1]],
        };
        let mut v = solid_vertex(
            overlay_local_px(p[0], p[1], vp_w, vp_h),
            base_colour,
            mesh.aa[i],
        );
        v.fill = [local[0], local[1], index, 0.0];
        verts.push(v);
    }
}

/// Emit an overlay mesh as a triangle list, with no anti-aliasing fringe: the
/// consumer's triangles are drawn as given.
///
/// `offset` is added to every position (the resolved anchor plus translate on
/// the immediate path, the item's translate in a compiled group), and the
/// item's linear map then turns the mesh about its bounding-box centre plus
/// pivot. The fill spans the bounding box and multiplies the vertex colours;
/// with UVs each vertex reads the fill at the point of the box its UV names.
/// The item's tint is applied and its vertices marked as content.
pub(super) fn emit_mesh(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    mesh: &crate::renderer::types::OverlayMeshItem,
    offset: [f32; 2],
    fills: &mut super::overlay_fill::FillTable,
    image_size: [u32; 2],
) {
    let Some((min, max)) = mesh.bounds() else {
        return;
    };
    if mesh.positions.iter().flatten().any(|v| !v.is_finite()) {
        return;
    }
    let start = verts.len();
    let n = mesh.positions.len();
    let half = [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5];
    let centre = [min[0] + half[0], min[1] + half[1]];
    let (base_colour, index) = fill_record(
        &mesh.style.fill,
        mesh.style.opacity,
        half,
        fills,
        image_size,
    );
    let uv_half = [half[0].max(0.5), half[1].max(0.5)];
    let uvs = mesh.uvs.as_deref().filter(|u| u.len() == n);
    // Resolve every vertex once; triangles index into it.
    let resolved: Vec<crate::resources::OverlayTextVertex> = (0..n)
        .map(|i| {
            let p = mesh.positions[i];
            let c = mesh.colours.at(i, n).to_linear_rgba();
            let colour = [
                c[0] * base_colour[0],
                c[1] * base_colour[1],
                c[2] * base_colour[2],
                c[3] * base_colour[3],
            ];
            let local = match uvs {
                Some(uvs) => [
                    uvs[i][0] * 2.0 * uv_half[0] - half[0],
                    uvs[i][1] * 2.0 * uv_half[1] - half[1],
                ],
                None => [p[0] - centre[0], p[1] - centre[1]],
            };
            let mut v = solid_vertex(
                [p[0] + offset[0], p[1] + offset[1]],
                colour,
                crate::resources::OverlayTextVertex::NO_AA,
            );
            v.fill = [local[0], local[1], index, 0.0];
            v
        })
        .collect();
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if a >= n || b >= n || c >= n {
            continue;
        }
        verts.extend_from_slice(&[resolved[a], resolved[b], resolved[c]]);
    }
    super::viewport_overlays::tint_vertices_from(verts, start, mesh.style.tint);
    let rot = OverlayRotation::new(
        &mesh.transform,
        OverlayRotation::pivot_point(
            [min[0] + offset[0], min[1] + offset[1]],
            [max[0] - min[0], max[1] - min[1]],
            mesh.transform.pivot,
        ),
    );
    rotate_vertices_from(verts, start, rot);
}

/// The vertex colour and fill-record index a tessellated fill draws with: a
/// solid fill is the vertex colour alone, any other fill a record in `fills`
/// over a box of half-extents `half` with the opacity on the vertex.
pub(super) fn fill_record(
    fill: &crate::renderer::types::OverlayFill,
    opacity: f32,
    half: [f32; 2],
    fills: &mut super::overlay_fill::FillTable,
    image_size: [u32; 2],
) -> ([f32; 4], f32) {
    if let crate::renderer::types::OverlayFill::Solid(c) = fill {
        let mut colour = c.to_linear_rgba();
        colour[3] *= opacity;
        return (colour, super::overlay_fill::NO_FILL);
    }
    let index = fills.push(fill, half, 1.0, [1.0; 4], image_size);
    ([1.0, 1.0, 1.0, opacity], index)
}

/// Emit a fringed fill (see [`tessellate_fringed`]) as a vector shape's fill,
/// for an [`OverlayShape::Vector`](crate::renderer::types::OverlayShape).
/// `mesh` is in screen-space logical pixels (already transformed for the
/// shape's position and rotation) and `local_points` are its input contours'
/// points in the path's own frame, which `mesh.sources` index. The fill spans
/// the path's own bounding box, so it turns and scales with the shape.
///
/// [`tessellate_fringed`]: super::overlay_vector::tessellate_fringed
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_vector_fill(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    mesh: &super::overlay_vector::FringedFill,
    local_points: &[[f32; 2]],
    fill: &crate::renderer::types::OverlayFill,
    opacity: f32,
    vp_w: f32,
    vp_h: f32,
    fills: &mut super::overlay_fill::FillTable,
    image_size: [u32; 2],
) {
    let Some((min, max)) = polyline_bounds(local_points) else {
        return;
    };
    let half = [(max[0] - min[0]) * 0.5, (max[1] - min[1]) * 0.5];
    let centre = [min[0] + half[0], min[1] + half[1]];
    let (base_colour, index) = fill_record(fill, opacity, half, fills, image_size);
    for &idx in &mesh.indices {
        let idx = idx as usize;
        let p = mesh.positions[idx];
        let (from, to, t) = mesh.sources[idx];
        let (a, b) = (local_points[from], local_points[to]);
        let l = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let mut v = solid_vertex(
            overlay_local_px(p[0], p[1], vp_w, vp_h),
            base_colour,
            mesh.aa[idx],
        );
        v.fill = [l[0] - centre[0], l[1] - centre[1], index, 0.0];
        verts.push(v);
    }
}

/// How one edge of a ribbon is anti-aliased.
///
/// A fringe is a one-device-pixel band whose coverage ramps from 1 to 0. The
/// vertex shader places it in device pixels, so it stays one pixel wide
/// whatever transform or `pixels_per_point` the ribbon is drawn with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeAa {
    /// A hard edge.
    None,
    /// A fringe straddling the edge, half a pixel either side, so the visual
    /// width is the geometric one. For an edge with nothing drawn against it.
    Centred,
    /// A fringe running one pixel out from the edge. For a shadow band's edge
    /// on the item it belongs to: the item's own fringe straddles that line,
    /// and a band whose coverage stays full up to it lays no seam of
    /// background or fill colour between the two.
    Outset,
}

impl EdgeAa {
    fn is_some(self) -> bool {
        self != EdgeAa::None
    }
}

/// Which parts of a ribbon get an anti-aliasing fringe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Fringe {
    /// The edge at `d_lo`.
    pub lo: EdgeAa,
    /// The edge at `d_hi`.
    pub hi: EdgeAa,
    /// Across each end of an open path (butt and square caps) and around a
    /// round cap. Off for a band whose ends abut other geometry, which would
    /// otherwise overlap it there.
    pub ends: bool,
}

impl Fringe {
    /// Both edges and both ends: a stroke.
    pub(super) const ALL: Self = Self {
        lo: EdgeAa::Centred,
        hi: EdgeAa::Centred,
        ends: true,
    };

    /// A shadow band running outward from the item, or inward from a
    /// contour: the edge nearer the path lies on the item and is outset, the
    /// far edge is centred.
    pub(super) fn band(d_lo: f32, d_hi: f32, ends: bool) -> Self {
        let lo_near = d_lo.abs() <= d_hi.abs();
        Self {
            lo: if lo_near {
                EdgeAa::Outset
            } else {
                EdgeAa::Centred
            },
            hi: if lo_near {
                EdgeAa::Centred
            } else {
                EdgeAa::Outset
            },
            ends,
        }
    }
}

/// A solid text-stream vertex at `p` (logical pixels) in `colour`, with its
/// place in a fringe (see [`OverlayTextVertex::aa`]).
///
/// [`OverlayTextVertex::aa`]: crate::resources::OverlayTextVertex
pub(super) fn solid_vertex(
    p: [f32; 2],
    colour: [f32; 4],
    aa: [f32; 4],
) -> crate::resources::OverlayTextVertex {
    crate::resources::OverlayTextVertex {
        position: p,
        uv: [0.0, 0.0],
        colour,
        use_texture: 0.0,
        clip_index: -1.0,
        clip_rect: [0.0; 4],
        group_tint: 0.0,
        fill: crate::resources::OverlayTextVertex::NO_FILL,
        aa,
    }
}

/// The `aa` of a fringe's inner row: full coverage, half a device pixel in
/// from the edge against `dir`, never more than `max_inset`.
#[inline]
pub(super) fn aa_inner(dir: [f32; 2], max_inset: f32) -> [f32; 4] {
    [dir[0], dir[1], -1.0, max_inset]
}

/// The `aa` of a fringe's outer row: zero coverage, half a device pixel out
/// along `dir`.
#[inline]
pub(super) fn aa_outer(dir: [f32; 2], max_inset: f32) -> [f32; 4] {
    [dir[0], dir[1], 1.0, max_inset]
}

/// The `max_inset` of a fill's edge, which has no opposite edge to meet.
pub(super) const FILL_MAX_INSET: f32 = 1.0e6;

/// Push the quad `a0 a1 b1 b0` (two rows, `a` and `b`, each from `0` to `1`)
/// as two triangles.
#[inline]
fn push_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    a0: crate::resources::OverlayTextVertex,
    a1: crate::resources::OverlayTextVertex,
    b0: crate::resources::OverlayTextVertex,
    b1: crate::resources::OverlayTextVertex,
) {
    verts.extend_from_slice(&[a0, a1, b0, a1, b1, b0]);
}

/// Tessellate a polyline into a triangle list of `OverlayTextVertex`s, with
/// an anti-aliasing fringe along both edges and across both ends.
///
/// At each interior point the stroke is extruded along the bisector of the
/// two adjacent segment normals, scaled by `1 / cos(half_angle)` so the
/// outer edge stays at constant `thickness / 2` from the centreline. When
/// that scale exceeds `mitre_limit`, or when `join` is `Bevel`, the outside
/// of the turn is cut by a bevel and the inside keeps the mitred point. A
/// `Round` join sweeps an arc round the outside instead.
///
/// Open polylines end with the requested cap style (butt, square, or round).
/// Closed polylines wrap the last point back to the first and ignore `cap`. A
/// `thickness` of zero is a hairline, one device pixel wide.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tessellate_polyline(
    points: &[[f32; 2]],
    thickness: f32,
    closed: bool,
    join: crate::renderer::types::LineJoin,
    mitre_limit: f32,
    cap: crate::renderer::types::LineCap,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) -> Vec<crate::resources::OverlayTextVertex> {
    let half_t = thickness * 0.5;
    tessellate_ribbon(
        points,
        -half_t,
        half_t,
        [0.0, 0.0],
        closed,
        join,
        mitre_limit,
        cap,
        Fringe::ALL,
        colour,
        vp_w,
        vp_h,
    )
}

/// One edge of a rib: the point on it, and the fringe directions its inner
/// and outer rows move along (zero for an edge without a fringe).
#[derive(Clone, Copy)]
struct RibEdge {
    p: [f32; 2],
    inner: [f32; 2],
    outer: [f32; 2],
}

/// A cross-section of a ribbon: its two edges, and how far a fringe may
/// inset either of them before the two meet.
#[derive(Clone, Copy)]
struct Rib {
    hi: RibEdge,
    lo: RibEdge,
    inset: f32,
}

/// The `max_inset` that marks a hairline's fringe rows (see
/// [`OverlayTextVertex::aa`]): any negative value.
///
/// [`OverlayTextVertex::aa`]: crate::resources::OverlayTextVertex
pub(super) const HAIRLINE_INSET: f32 = -1.0;

/// Tessellate a band running alongside a path, between two signed offsets from
/// its centreline. `d_lo` and `d_hi` are distances along the left-hand normal,
/// with `d_lo <= d_hi`; a centred stroke of thickness `t` is `(-t/2, t/2)`.
///
/// A band that sits wholly on one side of the path is how a shadow layer is
/// clipped to one side of a silhouette: an outer layer runs from the contour
/// outward, an inner layer from the contour inward, and neither crosses into
/// the other's territory the way a centred stroke does.
///
/// `shift` moves a one-sided band's far edge by the component of `shift` along
/// the band, leaving the near edge on the path. That is what a shadow layer's
/// offset becomes once the layer is clipped to one side of the contour: moving
/// the whole band would push it across the contour, while moving only the far
/// edge widens the band where the offset points and narrows it where it points
/// away, which is the same region. A straddling band ignores `shift`.
///
/// Caps are a half-disc about the centreline, which only reads correctly for a
/// band that straddles it. A one-sided band is meant for closed paths, where
/// there are no ends to cap.
///
/// `fringe` says which edges and ends carry an anti-aliasing fringe. Each
/// segment is a core quad between the edges' inner rows plus one quad per
/// fringed edge, and consecutive segments share their ribs, so no two
/// triangles of the band cover the same point.
///
/// A band with both offsets zero and both edges fringed is a hairline: the
/// fringe alone draws it, one device pixel wide.
#[allow(clippy::too_many_arguments)]
pub(super) fn tessellate_ribbon(
    points: &[[f32; 2]],
    d_lo: f32,
    d_hi: f32,
    shift: [f32; 2],
    closed: bool,
    join: crate::renderer::types::LineJoin,
    mitre_limit: f32,
    cap: crate::renderer::types::LineCap,
    fringe: Fringe,
    colour: [f32; 4],
    _vp_w: f32,
    _vp_h: f32,
) -> Vec<crate::resources::OverlayTextVertex> {
    tessellate_ribbon_scaled(
        points,
        None,
        d_lo,
        d_hi,
        shift,
        closed,
        join,
        mitre_limit,
        cap,
        fringe,
        colour,
    )
}

/// [`tessellate_ribbon`] with the offsets scaled per point: at point `i` the
/// band runs from `d_lo * scales[i]` to `d_hi * scales[i]`. A stroke whose
/// width varies along it is `(-0.5, 0.5)` scaled by the widths. `scales` must
/// have one entry per point; any other length is ignored.
#[allow(clippy::too_many_arguments)]
pub(super) fn tessellate_ribbon_scaled(
    points: &[[f32; 2]],
    scales: Option<&[f32]>,
    d_lo: f32,
    d_hi: f32,
    shift: [f32; 2],
    closed: bool,
    join: crate::renderer::types::LineJoin,
    mitre_limit: f32,
    cap: crate::renderer::types::LineCap,
    fringe: Fringe,
    colour: [f32; 4],
) -> Vec<crate::resources::OverlayTextVertex> {
    use crate::renderer::types::{LineCap, LineJoin};
    let scales = scales.filter(|k| k.len() == points.len());
    // A point that repeats the one before it is a segment of no length, and its
    // normal is zero. At a corner that turns the joint into two square ends with
    // nothing between them, so the outer corner is missing. It says nothing about
    // the path, so it goes. A closed path whose last point repeats its first is the
    // same case at the seam.
    let same = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4;
    let repeats = points.windows(2).any(|w| same(w[0], w[1]))
        || (closed && points.len() > 1 && same(points[0], points[points.len() - 1]));
    let deduped: Vec<[f32; 2]>;
    let deduped_scales: Vec<f32>;
    let (points, scales) = if repeats {
        let mut kept: Vec<[f32; 2]> = Vec::with_capacity(points.len());
        let mut kept_scales: Vec<f32> = Vec::with_capacity(points.len());
        for (i, &p) in points.iter().enumerate() {
            if kept.last().is_none_or(|&q| !same(p, q)) {
                kept.push(p);
                kept_scales.push(scales.map_or(1.0, |k| k[i]));
            }
        }
        if closed && kept.len() > 1 && same(kept[0], kept[kept.len() - 1]) {
            kept.pop();
            kept_scales.pop();
        }
        deduped = kept;
        deduped_scales = kept_scales;
        (&deduped[..], scales.map(|_| &deduped_scales[..]))
    } else {
        (points, scales)
    };
    let n = points.len();
    let hairline = scales.is_none()
        && d_lo == 0.0
        && d_hi == 0.0
        && fringe.lo == EdgeAa::Centred
        && fringe.hi == EdgeAa::Centred;
    if n < 2 || !(d_hi > d_lo || hairline) {
        return Vec::new();
    }
    // A point's offset scale, which is never negative.
    let k_at = |i: usize| scales.map_or(1.0, |k| k[i].max(0.0));
    // Reach of the band from the centreline at a point, used to place caps.
    let half_t_at = |i: usize| d_lo.abs().max(d_hi.abs()) * k_at(i);

    let normalize = |v: [f32; 2]| -> [f32; 2] {
        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
        if len < 1e-6 {
            [0.0, 0.0]
        } else {
            [v[0] / len, v[1] / len]
        }
    };
    let sub = |a: [f32; 2], b: [f32; 2]| -> [f32; 2] { [a[0] - b[0], a[1] - b[1]] };
    let add = |a: [f32; 2], b: [f32; 2]| -> [f32; 2] { [a[0] + b[0], a[1] + b[1]] };
    let scale = |a: [f32; 2], k: f32| -> [f32; 2] { [a[0] * k, a[1] * k] };
    let perp_left = |t: [f32; 2]| -> [f32; 2] { [-t[1], t[0]] };
    let dot = |a: [f32; 2], b: [f32; 2]| -> f32 { a[0] * b[0] + a[1] * b[1] };
    let length = |v: [f32; 2]| -> f32 { (v[0] * v[0] + v[1] * v[1]).sqrt() };

    // The band's two edges at point `i` on a rib whose outward direction is
    // `nrm`. A one-sided band keeps its near edge on the path and moves the
    // far one by the shift's component along `nrm`; a straddling band is a
    // plain stroke.
    let one_sided = !hairline && (d_lo >= 0.0 || d_hi <= 0.0);
    let edges = |nrm: [f32; 2], i: usize| -> (f32, f32) {
        let (d_hi, d_lo) = (d_hi * k_at(i), d_lo * k_at(i));
        if !one_sided || shift == [0.0, 0.0] {
            return (d_hi, d_lo);
        }
        let m = dot(shift, nrm);
        if d_lo >= 0.0 {
            ((d_hi + m).max(d_lo), d_lo)
        } else {
            (d_hi, (d_lo + m).min(d_hi))
        }
    };
    // How far an edge's fringe may move in before it meets the other edge (or
    // the other edge's fringe), in the same units as the offsets.
    let inset_of = |hi: f32, lo: f32| -> f32 {
        if hairline {
            return HAIRLINE_INSET;
        }
        let w = (hi - lo).max(0.0);
        if fringe.hi == EdgeAa::Centred && fringe.lo == EdgeAa::Centred {
            w * 0.5
        } else {
            w
        }
    };
    // An edge at `p` whose outward direction is `dir`, anti-aliased as `aa`
    // says. An outset fringe's inner row stays on the edge and its outer row
    // goes a whole pixel out.
    let edge = |p: [f32; 2], dir: [f32; 2], aa: EdgeAa| match aa {
        EdgeAa::None => RibEdge {
            p,
            inner: [0.0, 0.0],
            outer: [0.0, 0.0],
        },
        EdgeAa::Centred => RibEdge {
            p,
            inner: dir,
            outer: dir,
        },
        EdgeAa::Outset => RibEdge {
            p,
            inner: [0.0, 0.0],
            outer: scale(dir, 2.0),
        },
    };
    // A rib across the band at `cur` along `dir` (a unit normal, or a bisector
    // scaled to the mitre length), with its edges `hi` and `lo` from the path.
    let rib_along = |cur: [f32; 2], dir: [f32; 2], hi: f32, lo: f32| Rib {
        hi: edge(add(cur, scale(dir, hi)), dir, fringe.hi),
        lo: edge(add(cur, scale(dir, lo)), scale(dir, -1.0), fringe.lo),
        inset: inset_of(hi, lo),
    };

    // Build "ribs": for each logical join, one or more cross-sections.
    let mut ribs: Vec<Rib> = Vec::with_capacity(n + 4);

    let iter_count = if closed { n + 1 } else { n };
    for i in 0..iter_count {
        let cur_idx = if closed { i % n } else { i };
        let cur = points[cur_idx];
        let prev = if closed {
            Some(points[(cur_idx + n - 1) % n])
        } else if cur_idx > 0 {
            Some(points[cur_idx - 1])
        } else {
            None
        };
        let next = if closed {
            Some(points[(cur_idx + 1) % n])
        } else if cur_idx + 1 < n {
            Some(points[cur_idx + 1])
        } else {
            None
        };

        let in_t = prev.map(|p| normalize(sub(cur, p)));
        let out_t = next.map(|q| normalize(sub(q, cur)));

        // On a closed path the joint at the first point is built twice, at the
        // start and again at the wrap. The wrap keeps only the rib the closing
        // segment arrives at, so a bevel or an arc there is drawn once.
        let wrap = closed && i == n;
        match (in_t, out_t) {
            (Some(t1), Some(t2)) => {
                let n1 = perp_left(t1);
                let n2 = perp_left(t2);
                let bisect = normalize(add(n1, n2));
                let cos_half = dot(n1, bisect).abs().max(1e-4);
                let mitre_scale = 1.0 / cos_half;
                // A turn too slight to show an arc is mitred whatever the join.
                let use_mitre = match join {
                    LineJoin::Mitre => mitre_scale <= mitre_limit,
                    LineJoin::Bevel => false,
                    LineJoin::Round => dot(n1, n2) > 0.9999,
                };
                if use_mitre {
                    let (hi, lo) = edges(bisect, cur_idx);
                    ribs.push(rib_along(cur, scale(bisect, mitre_scale), hi, lo));
                    continue;
                }
                // Bevel: the outside of the turn gets one edge point per
                // adjacent segment. The inside keeps the mitred point, where
                // the two inner edges cross, so the segments meet without
                // overlapping; two separate points there would lay one
                // segment's corner (and its fringe) over the other's. A turn
                // so sharp that the crossing falls beyond a segment's far end
                // keeps the two points. A round join adds ribs between the two
                // that sweep the outside edge round an arc.
                let (h1, l1) = edges(n1, cur_idx);
                let (h2, l2) = edges(n2, cur_idx);
                let hi_inside = dot(t2, n1) > 0.0;
                let (hm, lm) = edges(bisect, cur_idx);
                let d_in = if hi_inside { hm } else { lm };
                let reach = d_in.abs() * mitre_scale;
                let shortest = prev
                    .map(|p| length(sub(cur, p)))
                    .unwrap_or(0.0)
                    .min(next.map(|q| length(sub(q, cur))).unwrap_or(0.0));
                let rib1 = rib_along(cur, n1, h1, l1);
                let rib2 = rib_along(cur, n2, h2, l2);
                let mut arc: Vec<Rib> = Vec::new();
                if join == LineJoin::Round && !wrap {
                    let turn = (n1[0] * n2[1] - n1[1] * n2[0]).atan2(dot(n1, n2));
                    let radius = if hi_inside { l1.abs() } else { h1.abs() };
                    let steps = round_steps(turn.abs(), radius);
                    for k in 1..steps {
                        let a = turn * k as f32 / steps as f32;
                        let (sa, ca) = a.sin_cos();
                        let nk = [n1[0] * ca - n1[1] * sa, n1[0] * sa + n1[1] * ca];
                        let (hk, lk) = edges(nk, cur_idx);
                        arc.push(rib_along(cur, nk, hk, lk));
                    }
                }
                if bisect != [0.0, 0.0] && reach <= shortest {
                    let mdir = scale(bisect, mitre_scale);
                    let inner = rib_along(cur, mdir, hm, lm);
                    let joined = |mut r: Rib| {
                        if hi_inside {
                            r.hi = inner.hi;
                        } else {
                            r.lo = inner.lo;
                        }
                        r.inset = r.inset.min(inner.inset);
                        r
                    };
                    ribs.push(joined(rib1));
                    if !wrap {
                        ribs.extend(arc.into_iter().map(joined));
                        ribs.push(joined(rib2));
                    }
                } else {
                    ribs.push(rib1);
                    if !wrap {
                        ribs.extend(arc);
                        ribs.push(rib2);
                    }
                }
            }
            (Some(t1), None) => {
                let n1 = perp_left(t1);
                let (hi, lo) = edges(n1, cur_idx);
                ribs.push(rib_along(cur, n1, hi, lo));
            }
            (None, Some(t2)) => {
                let n2 = perp_left(t2);
                let (hi, lo) = edges(n2, cur_idx);
                ribs.push(rib_along(cur, n2, hi, lo));
            }
            (None, None) => {}
        }
    }

    if ribs.len() < 2 {
        return Vec::new();
    }

    // End tangents for cap placement (zero when the end segment is degenerate).
    let t_start = normalize(sub(points[1], points[0]));
    let t_end = normalize(sub(points[n - 1], points[n - 2]));
    let (half_start, half_end) = (half_t_at(0), half_t_at(n - 1));
    let round = cap == LineCap::Round;

    if !closed {
        // Square caps: push the endpoint ribs out along the tangent so the
        // stroke rectangle extends half its width beyond each endpoint.
        if cap == LineCap::Square {
            let first = &mut ribs[0];
            first.hi.p = sub(first.hi.p, scale(t_start, half_start));
            first.lo.p = sub(first.lo.p, scale(t_start, half_start));
            let last = ribs.last_mut().unwrap();
            last.hi.p = add(last.hi.p, scale(t_end, half_end));
            last.lo.p = add(last.lo.p, scale(t_end, half_end));
        }
        // A flat end's fringe: both end corners also move out along the end's
        // tangent, so the side fringes meet the one across the end on the
        // corner's diagonal.
        if fringe.ends && !round {
            for (rib, out) in [(0, scale(t_start, -1.0)), (ribs.len() - 1, t_end)] {
                let r = &mut ribs[rib];
                for e in [&mut r.hi, &mut r.lo] {
                    e.inner = add(e.inner, out);
                    e.outer = add(e.outer, out);
                }
            }
        }
    }

    let inner = |e: RibEdge, inset: f32| solid_vertex(e.p, colour, aa_inner(e.inner, inset));
    let outer = |e: RibEdge, inset: f32| solid_vertex(e.p, colour, aa_outer(e.outer, inset));

    // Each pair of consecutive ribs is a core quad between the inner rows,
    // plus a fringe quad on each fringed edge. A hairline's core has no area,
    // so it is left out.
    let per_seg = 6 * (1 + fringe.hi.is_some() as usize + fringe.lo.is_some() as usize);
    let mut verts: Vec<crate::resources::OverlayTextVertex> =
        Vec::with_capacity((ribs.len() - 1) * per_seg + 64);
    for w in ribs.windows(2) {
        let (a, b) = (w[0], w[1]);
        if !hairline {
            push_quad(
                &mut verts,
                inner(a.hi, a.inset),
                inner(a.lo, a.inset),
                inner(b.hi, b.inset),
                inner(b.lo, b.inset),
            );
        }
        if fringe.hi.is_some() {
            push_quad(
                &mut verts,
                inner(a.hi, a.inset),
                outer(a.hi, a.inset),
                inner(b.hi, b.inset),
                outer(b.hi, b.inset),
            );
        }
        if fringe.lo.is_some() {
            push_quad(
                &mut verts,
                inner(a.lo, a.inset),
                outer(a.lo, a.inset),
                inner(b.lo, b.inset),
                outer(b.lo, b.inset),
            );
        }
    }

    if closed {
        return verts;
    }
    let ends = [
        (ribs[0], points[0], scale(t_start, -1.0), half_start),
        (*ribs.last().unwrap(), points[n - 1], t_end, half_end),
    ];
    if round {
        // Round caps: a semicircle fan on the outside half of each endpoint,
        // from one edge of the end rib round to the other, so the fan's rim
        // starts and ends exactly on the rib.
        for (rib, centre, outward, half_t) in ends {
            if outward == [0.0, 0.0] {
                continue;
            }
            let segs = 8;
            let normal = perp_left(outward);
            // Which rib edge the sweep starts from: the one on `normal`'s side.
            let (first, last) = if dot(sub(rib.hi.p, centre), normal) >= 0.0 {
                (rib.hi, rib.lo)
            } else {
                (rib.lo, rib.hi)
            };
            let rim = |k: usize| -> RibEdge {
                if k == 0 {
                    return first;
                }
                if k == segs {
                    return last;
                }
                let a = std::f32::consts::PI * k as f32 / segs as f32;
                let d = add(scale(normal, a.cos()), scale(outward, a.sin()));
                let aa = if fringe.ends {
                    EdgeAa::Centred
                } else {
                    EdgeAa::None
                };
                edge(add(centre, scale(d, half_t)), d, aa)
            };
            let hub = inner(
                RibEdge {
                    p: centre,
                    inner: [0.0, 0.0],
                    outer: [0.0, 0.0],
                },
                rib.inset,
            );
            for k in 0..segs {
                let (r0, r1) = (rim(k), rim(k + 1));
                if !hairline {
                    verts.extend_from_slice(&[hub, inner(r0, rib.inset), inner(r1, rib.inset)]);
                }
                if fringe.ends {
                    push_quad(
                        &mut verts,
                        inner(r0, rib.inset),
                        outer(r0, rib.inset),
                        inner(r1, rib.inset),
                        outer(r1, rib.inset),
                    );
                }
            }
        }
    } else if fringe.ends {
        // The fringe across each flat end, between the end rib's inner and
        // outer rows.
        for (rib, _, _, _) in ends {
            push_quad(
                &mut verts,
                inner(rib.hi, rib.inset),
                outer(rib.hi, rib.inset),
                inner(rib.lo, rib.inset),
                outer(rib.lo, rib.inset),
            );
        }
    }
    verts
}

/// How many steps an arc of `angle` radians at `radius` logical pixels is
/// drawn in: few enough to stay cheap, enough that the chord stays within a
/// tenth of a pixel of the arc at unit scale.
fn round_steps(angle: f32, radius: f32) -> usize {
    let step = if radius > 0.1 {
        2.0 * (1.0 - 0.1 / radius).clamp(-1.0, 1.0).acos()
    } else {
        std::f32::consts::FRAC_PI_4
    };
    let step = step.clamp(0.05, std::f32::consts::FRAC_PI_4);
    ((angle / step).ceil() as usize).clamp(1, 64)
}

/// Working point list and cumulative arc-length table for a polyline. For
/// closed paths the first point is appended again so the table covers the
/// closing segment.
pub(super) fn polyline_arc_table(points: &[[f32; 2]], closed: bool) -> (Vec<[f32; 2]>, Vec<f32>) {
    let mut pts: Vec<[f32; 2]> = points.to_vec();
    if closed && points.len() >= 2 {
        pts.push(points[0]);
    }
    let mut cum = Vec::with_capacity(pts.len());
    cum.push(0.0);
    let mut s = 0.0f32;
    for w in pts.windows(2) {
        let dx = w[1][0] - w[0][0];
        let dy = w[1][1] - w[0][1];
        s += (dx * dx + dy * dy).sqrt();
        cum.push(s);
    }
    (pts, cum)
}

/// Interpolate the point at arc length `s` along the path (clamped to the
/// path range).
pub(super) fn point_at_arc(pts: &[[f32; 2]], cum: &[f32], s: f32) -> [f32; 2] {
    let total = *cum.last().unwrap_or(&0.0);
    let s = s.clamp(0.0, total);
    let i = cum
        .partition_point(|&c| c < s)
        .saturating_sub(1)
        .min(pts.len().saturating_sub(2));
    let seg = (cum[i + 1] - cum[i]).max(1e-6);
    let t = ((s - cum[i]) / seg).clamp(0.0, 1.0);
    [
        pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
        pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t,
    ]
}

/// Split a path into the visible sub-paths of a dash pattern. Each sub-path
/// is an open point list covering one dash, including the interpolated dash
/// endpoints and any original waypoints in between. For closed paths, a dash
/// spanning the final-to-first seam is merged into a single sub-path.
pub(super) fn dash_subpaths(
    pts: &[[f32; 2]],
    cum: &[f32],
    dash: f32,
    gap: f32,
    offset: f32,
    closed: bool,
) -> Vec<Vec<[f32; 2]>> {
    dash_pieces(pts, cum, dash, gap, offset, closed)
        .into_iter()
        .map(|(points, _)| points)
        .collect()
}

/// [`dash_subpaths`], with each sub-path's points paired with their arc
/// lengths along the whole path, so anything carried per point (a width) can
/// be read back at them.
pub(super) fn dash_pieces(
    pts: &[[f32; 2]],
    cum: &[f32],
    dash: f32,
    gap: f32,
    offset: f32,
    closed: bool,
) -> Vec<(Vec<[f32; 2]>, Vec<f32>)> {
    let total = *cum.last().unwrap_or(&0.0);
    if total <= 1e-4 {
        return Vec::new();
    }
    let period = dash + gap;

    // Visible intervals are [k * period - offset, k * period - offset + dash).
    let k_min = ((offset - dash) / period).floor() as i64;
    let k_max = ((total + offset) / period).ceil() as i64;
    let mut intervals: Vec<(f32, f32)> = Vec::new();
    for k in k_min..=k_max {
        let a = (k as f32 * period - offset).max(0.0);
        let b = (k as f32 * period - offset + dash).min(total);
        if b - a > 1e-4 {
            intervals.push((a, b));
        }
    }
    if intervals.is_empty() {
        return Vec::new();
    }

    // A dash crossing the closed-path seam shows up as one interval ending at
    // `total` and another starting at 0; stitch them into one sub-path so the
    // seam gets a joint instead of two caps.
    let merge_seam = closed
        && intervals.len() >= 2
        && intervals.first().unwrap().0 <= 1e-4
        && total - intervals.last().unwrap().1 <= 1e-4;

    let extract = |a: f32, b: f32| -> (Vec<[f32; 2]>, Vec<f32>) {
        let mut sub = vec![point_at_arc(pts, cum, a)];
        let mut arcs = vec![a];
        for (i, &c) in cum.iter().enumerate() {
            if c > a + 1e-4 && c < b - 1e-4 {
                sub.push(pts[i]);
                arcs.push(c);
            }
        }
        sub.push(point_at_arc(pts, cum, b));
        arcs.push(b);
        (sub, arcs)
    };

    let mut subs: Vec<(Vec<[f32; 2]>, Vec<f32>)> =
        intervals.iter().map(|&(a, b)| extract(a, b)).collect();
    if merge_seam {
        let (first, first_arcs) = subs.remove(0);
        let (last, last_arcs) = subs.last_mut().unwrap();
        // It ends at `total`, the same point as the first sub-path's start.
        last.pop();
        last_arcs.pop();
        last.extend(first);
        last_arcs.extend(first_arcs);
    }
    subs
}

/// The value of a per-point quantity at arc length `s`, interpolated between
/// the points either side; `values` runs parallel to `cum`.
pub(super) fn value_at_arc(values: &[f32], cum: &[f32], s: f32) -> f32 {
    if values.is_empty() || values.len() != cum.len() {
        return 0.0;
    }
    if values.len() == 1 {
        return values[0];
    }
    let total = *cum.last().unwrap_or(&0.0);
    let s = s.clamp(0.0, total);
    let i = cum
        .partition_point(|&c| c < s)
        .saturating_sub(1)
        .min(values.len() - 2);
    let seg = (cum[i + 1] - cum[i]).max(1e-6);
    let t = ((s - cum[i]) / seg).clamp(0.0, 1.0);
    values[i] + (values[i + 1] - values[i]) * t
}

/// Emit a filled disc in screen pixel coordinates, with a fringe round its
/// rim.
pub(super) fn emit_disc(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    centre: [f32; 2],
    radius: f32,
    colour: [f32; 4],
    _vp_w: f32,
    _vp_h: f32,
) {
    emit_arc_band(
        verts,
        centre,
        0.0,
        radius,
        0.0,
        std::f32::consts::TAU,
        10,
        [EdgeAa::None, EdgeAa::Centred],
        colour,
    );
}

/// Emit a band about `centre` from radius `r_in` to `r_out`, sweeping `sweep`
/// radians from angle `a0` in `segs` steps. `fringe` is `[inner rim, outer
/// rim]`: how each rim is anti-aliased. An `r_in` of zero is a fan
/// (a disc or a wedge), whose inner rim is its hub.
#[allow(clippy::too_many_arguments)]
fn emit_arc_band(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    centre: [f32; 2],
    r_in: f32,
    r_out: f32,
    a0: f32,
    sweep: f32,
    segs: usize,
    fringe: [EdgeAa; 2],
    colour: [f32; 4],
) {
    if r_out <= r_in || segs == 0 {
        return;
    }
    let width = r_out - r_in;
    let inset = if fringe == [EdgeAa::Centred; 2] {
        width * 0.5
    } else {
        width
    };
    // (point, inner-row direction, outer-row direction) on each rim.
    let rim = |p: [f32; 2], d: [f32; 2], aa: EdgeAa| match aa {
        EdgeAa::None => (p, [0.0, 0.0], [0.0, 0.0]),
        EdgeAa::Centred => (p, d, d),
        EdgeAa::Outset => (p, [0.0, 0.0], [d[0] * 2.0, d[1] * 2.0]),
    };
    let rims = |k: usize| {
        let a = a0 + sweep * k as f32 / segs as f32;
        let d = [a.cos(), a.sin()];
        let at = |r: f32| [centre[0] + d[0] * r, centre[1] + d[1] * r];
        (
            rim(at(r_in), [-d[0], -d[1]], fringe[0]),
            rim(at(r_out), d, fringe[1]),
        )
    };
    type Rim = ([f32; 2], [f32; 2], [f32; 2]);
    let inner = |(p, d, _): Rim| solid_vertex(p, colour, aa_inner(d, inset));
    let outer = |(p, _, d): Rim| solid_vertex(p, colour, aa_outer(d, inset));
    for k in 0..segs {
        let (i0, o0) = rims(k);
        let (i1, o1) = rims(k + 1);
        push_quad(verts, inner(o0), inner(i0), inner(o1), inner(i1));
        if fringe[1].is_some() {
            push_quad(verts, inner(o0), outer(o0), inner(o1), outer(o1));
        }
        if fringe[0].is_some() {
            push_quad(verts, inner(i0), outer(i0), inner(i1), outer(i1));
        }
    }
}

/// Emit a straight band from the edge `a b` out along `out` by `width`: a
/// shadow wrapped round a stroke's flat end, so the edge `a b` lies on the
/// stroke and its fringe is outset, the far edge's centred, and its ends are
/// hard.
#[allow(clippy::too_many_arguments)]
fn emit_straight_band(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    a: [f32; 2],
    b: [f32; 2],
    out: [f32; 2],
    width: f32,
    colour: [f32; 4],
) {
    if width <= 0.0 {
        return;
    }
    let inset = width;
    let push = |p: [f32; 2]| [p[0] + out[0] * width, p[1] + out[1] * width];
    let back = [-out[0] * 2.0, -out[1] * 2.0];
    let near = |p: [f32; 2]| solid_vertex(p, colour, aa_inner([0.0, 0.0], inset));
    let near_out = |p: [f32; 2]| solid_vertex(p, colour, aa_outer(back, inset));
    let far = |p: [f32; 2]| solid_vertex(push(p), colour, aa_inner(out, inset));
    let far_out = |p: [f32; 2]| solid_vertex(push(p), colour, aa_outer(out, inset));
    push_quad(verts, near(a), far(a), near(b), far(b));
    push_quad(verts, far(a), far_out(a), far(b), far_out(b));
    push_quad(verts, near(a), near_out(a), near(b), near_out(b));
}

/// The stroke a polyline draws with, standing an absent one in as a stroke of
/// no width. The bands a shadow layer puts around a path are measured from the
/// stroke edges, so a path with no line keeps the geometry it has for a line of
/// zero width rather than losing its shadow.
pub(super) fn resolved_stroke(
    poly: &crate::renderer::types::OverlayPolylineItem,
) -> crate::renderer::types::OverlayStroke {
    poly.stroke.clone().unwrap_or_else(|| {
        crate::renderer::types::OverlayStroke::new(0.0, Colour::linear(1.0, 1.0, 1.0, 1.0))
    })
}

/// Emit the stroke geometry for a polyline item, honouring its cap style,
/// stroke pattern, and per-point widths. `colour` must already have opacity
/// applied.
pub(super) fn emit_polyline_stroke(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    poly: &crate::renderer::types::OverlayPolylineItem,
    stroke: &crate::renderer::types::OverlayStroke,
    colour: [f32; 4],
    _vp_w: f32,
    _vp_h: f32,
) {
    emit_stroke_path(
        verts,
        &poly.points,
        poly.closed,
        poly.widths.as_deref(),
        stroke,
        colour,
    );
}

/// Emit `stroke` along one path, honouring its cap, join, and pattern.
/// `widths`, when given with one entry per point, replaces the stroke's width
/// point by point. A stroke width of zero is a one-device-pixel hairline, and
/// a negative one draws nothing. `colour` must already have opacity applied.
///
/// A dash pattern cuts the path into dashes first and strokes each dash, so
/// each dash gets the caps and the joins inside it its own.
pub(super) fn emit_stroke_path(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    points: &[[f32; 2]],
    closed: bool,
    widths: Option<&[f32]>,
    stroke: &crate::renderer::types::OverlayStroke,
    colour: [f32; 4],
) {
    use crate::renderer::types::StrokePattern;
    let widths = widths.filter(|w| w.len() == points.len());
    if widths.is_none() && !(stroke.width >= 0.0) {
        return;
    }
    let piece = |verts: &mut Vec<crate::resources::OverlayTextVertex>,
                 pts: &[[f32; 2]],
                 ws: Option<&[f32]>,
                 closed: bool| {
        let (d, scales) = match ws {
            Some(ws) => (0.5, Some(ws)),
            None => (stroke.width * 0.5, None),
        };
        verts.extend(tessellate_ribbon_scaled(
            pts,
            scales,
            -d,
            d,
            [0.0, 0.0],
            closed,
            stroke.join,
            stroke.mitre_limit,
            stroke.cap,
            Fringe::ALL,
            colour,
        ));
    };
    // Widths along the arc-length table, which repeats the first point at the
    // end of a closed path.
    let arc_widths = |widths: &[f32]| {
        let mut w = widths.to_vec();
        if closed && points.len() >= 2 {
            w.push(widths[0]);
        }
        w
    };
    match stroke.pattern {
        StrokePattern::Solid => piece(verts, points, widths, closed),
        StrokePattern::Dashed {
            dash_length,
            gap_length,
            offset,
        } => {
            // Degenerate patterns collapse to a solid stroke rather than
            // disappearing or looping forever. The 0.25 px floor bounds the
            // sub-path count on long paths.
            if dash_length <= 0.0 || gap_length <= 0.0 {
                piece(verts, points, widths, closed);
                return;
            }
            let dash = dash_length.max(0.25);
            let gap = gap_length.max(0.25);
            let (pts, cum) = polyline_arc_table(points, closed);
            let along = widths.map(arc_widths);
            for (sub, arcs) in dash_pieces(&pts, &cum, dash, gap, offset, closed) {
                let ws: Option<Vec<f32>> = along
                    .as_ref()
                    .map(|w| arcs.iter().map(|&s| value_at_arc(w, &cum, s)).collect());
                piece(verts, &sub, ws.as_deref(), false);
            }
        }
        StrokePattern::Dotted { spacing, offset } => {
            let (pts, cum) = polyline_arc_table(points, closed);
            if pts.len() < 2 {
                return;
            }
            let along = widths.map(arc_widths);
            let total = *cum.last().unwrap();
            let spacing = spacing.max(0.5);
            let mut s = offset.rem_euclid(spacing);
            // On closed paths the point at `total` is the point at 0; skip it
            // so the seam doesn't get a doubled dot.
            let limit = if closed { total - 1e-3 } else { total + 1e-3 };
            while s < limit {
                let c = point_at_arc(&pts, &cum, s);
                let width = along
                    .as_ref()
                    .map_or(stroke.width, |w| value_at_arc(w, &cum, s));
                emit_disc(verts, c, width * 0.5, colour, 0.0, 0.0);
                s += spacing;
            }
        }
    }
}

/// How far an analytic line's quad grows past its box: half the thickness,
/// which covers a round cap, or half the thickness times root two for a square
/// cap, whose corners reach that far along the box's axes.
pub(super) fn line_quad_reach(thickness: f32, cap: crate::renderer::types::LineCap) -> f32 {
    let half = thickness * 0.5;
    if cap == crate::renderer::types::LineCap::Square {
        half * std::f32::consts::SQRT_2
    } else {
        half
    }
}

/// Encode `TileMode` as the float flag the fill module consumes.
pub(super) fn tile_mode_to_f(m: crate::renderer::types::TileMode) -> f32 {
    match m {
        crate::renderer::types::TileMode::Stretch => 0.0,
        crate::renderer::types::TileMode::Tile => 1.0,
        crate::renderer::types::TileMode::Mirror => 2.0,
    }
}

/// Emit a textured quad (6 vertices) for a glyph in screen pixel coordinates.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_textured_quad(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    colour: [f32; 4],
    tex: f32,
    vp_w: f32,
    vp_h: f32,
) {
    let tl = overlay_local_px(x0, y0, vp_w, vp_h);
    let tr = overlay_local_px(x1, y0, vp_w, vp_h);
    let bl = overlay_local_px(x0, y1, vp_w, vp_h);
    let br = overlay_local_px(x1, y1, vp_w, vp_h);
    let v = |pos: [f32; 2], uv: [f32; 2]| crate::resources::OverlayTextVertex {
        position: pos,
        uv,
        colour,
        use_texture: tex,
        clip_index: -1.0,
        clip_rect: [0.0; 4],
        group_tint: 0.0,
        fill: crate::resources::OverlayTextVertex::NO_FILL,
        aa: crate::resources::OverlayTextVertex::NO_AA,
    };
    // UV layout: top-left = uv_min, bottom-right = uv_max.
    verts.extend_from_slice(&[
        v(tl, uv_min),
        v(bl, [uv_min[0], uv_max[1]]),
        v(tr, [uv_max[0], uv_min[1]]),
        v(tr, [uv_max[0], uv_min[1]]),
        v(bl, [uv_min[0], uv_max[1]]),
        v(br, uv_max),
    ]);
}

/// Emit one textured quad per glyph in a laid-out run.
///
/// Shared by the label and glyph-run overlay paths: both produce a slice of
/// [`GlyphQuad`]s (positions and atlas UVs in logical pixels) and place them at
/// `(origin_x, origin_y)` with a single tint `colour`.
pub(super) fn emit_glyph_quads(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    quads: &[crate::resources::overlay::font::GlyphQuad],
    origin_x: f32,
    origin_y: f32,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    for gq in quads {
        let gx = origin_x + gq.pos[0];
        let gy = origin_y + gq.pos[1];
        emit_textured_quad(
            verts,
            gx,
            gy,
            gx + gq.size[0],
            gy + gq.size[1],
            gq.uv_min,
            gq.uv_max,
            colour,
            glyph_tex_mode(gq.color, gq.nearest),
            vp_w,
            vp_h,
        );
    }
}

/// The vertex `use_texture` mode for a glyph: `2.0` samples the atlas as full
/// colour (emoji), `1.0` tints the atlas coverage by the run colour (text).
/// A bitmap face's cell adds three (`4.0`, `5.0`): read texel by texel.
fn glyph_tex_mode(color: bool, nearest: bool) -> f32 {
    let mode = if color { 2.0 } else { 1.0 };
    if nearest { mode + 3.0 } else { mode }
}

/// Emit one textured quad per glyph, each with its own tint colour.
///
/// The glyph-run path uses this so a single run can carry per-glyph colours. Each
/// entry pairs a laid-out [`GlyphQuad`] with the colour to draw it in.
pub(super) fn emit_glyph_quads_colored(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    quads: &[(crate::resources::overlay::font::GlyphQuad, [f32; 4])],
    origin_x: f32,
    origin_y: f32,
    vp_w: f32,
    vp_h: f32,
) {
    for (gq, colour) in quads {
        let gx = origin_x + gq.pos[0];
        let gy = origin_y + gq.pos[1];
        emit_textured_quad(
            verts,
            gx,
            gy,
            gx + gq.size[0],
            gy + gq.size[1],
            gq.uv_min,
            gq.uv_max,
            *colour,
            glyph_tex_mode(gq.color, gq.nearest),
            vp_w,
            vp_h,
        );
    }
}

/// Apply an opacity multiplier to a colour's alpha channel.
#[inline]
pub(super) fn apply_opacity(colour: [f32; 4], opacity: f32) -> [f32; 4] {
    [colour[0], colour[1], colour[2], colour[3] * opacity]
}

#[cfg(test)]
mod stroke_tests {
    use super::*;
    use crate::renderer::types::{
        LineCap, LineJoin, OverlayPolylineItem, OverlayStroke, StrokePattern,
    };

    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    fn line_item(pattern: StrokePattern, cap: LineCap) -> OverlayPolylineItem {
        let mut item = OverlayPolylineItem::default();
        item.points = vec![[0.0, 0.0], [100.0, 0.0]];
        item.stroke = Some(
            OverlayStroke::new(4.0, Colour::linear(1.0, 1.0, 1.0, 1.0))
                .with_pattern(pattern)
                .with_cap(cap),
        );
        item
    }

    #[test]
    fn point_at_arc_interpolates_across_segments() {
        let (pts, cum) = polyline_arc_table(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], false);
        assert_eq!(*cum.last().unwrap(), 20.0);
        assert_eq!(point_at_arc(&pts, &cum, 5.0), [5.0, 0.0]);
        assert_eq!(point_at_arc(&pts, &cum, 15.0), [10.0, 5.0]);
        // Clamped past the end.
        assert_eq!(point_at_arc(&pts, &cum, 99.0), [10.0, 10.0]);
    }

    #[test]
    fn dash_subpaths_open_line() {
        let (pts, cum) = polyline_arc_table(&[[0.0, 0.0], [100.0, 0.0]], false);
        let subs = dash_subpaths(&pts, &cum, 10.0, 10.0, 0.0, false);
        assert_eq!(subs.len(), 5);
        assert_eq!(subs[0], vec![[0.0, 0.0], [10.0, 0.0]]);
        assert_eq!(subs[4], vec![[80.0, 0.0], [90.0, 0.0]]);
    }

    #[test]
    fn dash_subpaths_merges_closed_seam() {
        // 10x10 square, perimeter 40. dash 15 / gap 5 / offset 5 gives
        // visible intervals [0, 10], [15, 30], [35, 40]: the last wraps into
        // the first across the seam.
        let square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let (pts, cum) = polyline_arc_table(&square, true);
        let subs = dash_subpaths(&pts, &cum, 15.0, 5.0, 5.0, true);
        assert_eq!(subs.len(), 2);
        let seam = subs.last().unwrap();
        assert_eq!(seam.first().unwrap(), &[0.0, 5.0]);
        assert!(seam.contains(&[0.0, 0.0]));
        assert_eq!(seam.last().unwrap(), &[10.0, 0.0]);
    }

    #[test]
    fn caps_extend_the_stroke() {
        let tess = |cap| {
            tessellate_polyline(
                &[[10.0, 50.0], [90.0, 50.0]],
                4.0,
                false,
                LineJoin::Mitre,
                4.0,
                cap,
                WHITE,
                100.0,
                100.0,
            )
        };
        let butt = tess(LineCap::Butt);
        let square = tess(LineCap::Square);
        let round = tess(LineCap::Round);
        // One segment: a core quad and a fringe quad per edge, plus a fringe
        // quad across each flat end.
        assert_eq!(butt.len(), 18 + 2 * 6);
        assert_eq!(square.len(), 18 + 2 * 6);
        // Round caps add two 8-segment fans, each a hub triangle and a fringe
        // quad per step.
        assert_eq!(round.len(), 18 + 2 * 8 * (3 + 6));
        let min_x = |v: &[crate::resources::OverlayTextVertex]| {
            v.iter().map(|v| v.position[0]).fold(f32::MAX, f32::min)
        };
        assert!(square.iter().all(|v| v.position[0].is_finite()));
        assert!(min_x(&square) < min_x(&butt));
        assert!(min_x(&round) < min_x(&butt));
    }

    #[test]
    fn a_repeated_point_at_a_corner_keeps_the_corner() {
        // Down, then the corner twice, then right: a stepped line reduced to pixel
        // columns repeats its corners like this. The outer corner of a four-pixel
        // stroke is at (48, 52), and a mitred joint reaches it.
        let corner = |points: &[[f32; 2]]| {
            tessellate_polyline(
                points,
                4.0,
                false,
                LineJoin::Mitre,
                4.0,
                LineCap::Butt,
                WHITE,
                100.0,
                100.0,
            )
            .iter()
            .any(|v| (v.position[0] - 48.0).abs() < 1e-3 && (v.position[1] - 52.0).abs() < 1e-3)
        };
        assert!(corner(&[[50.0, 10.0], [50.0, 50.0], [90.0, 50.0]]));
        assert!(
            corner(&[[50.0, 10.0], [50.0, 50.0], [50.0, 50.0], [90.0, 50.0]]),
            "a repeated corner point leaves the outer corner unfilled"
        );
    }

    /// Where the vertex shader puts a text-stream vertex and the coverage it
    /// gives it, at `s` device pixels per local unit (`place_fringe` in
    /// `overlay_text.wgsl`).
    fn place(v: &crate::resources::OverlayTextVertex, s: f32) -> ([f32; 2], f32) {
        let [dx, dy, side, max_inset] = v.aa;
        let p = v.position;
        if side < -0.5 {
            let m = max_inset * s;
            let inset = m.min(0.5);
            let c = if m < 0.5 { 2.0 * m / (m + 0.5) } else { 1.0 };
            ([p[0] - dx * inset / s, p[1] - dy * inset / s], c)
        } else if side > 0.5 {
            ([p[0] + dx * 0.5 / s, p[1] + dy * 0.5 / s], 0.0)
        } else {
            (p, 1.0)
        }
    }

    /// The integral of coverage across a horizontal stroke at its middle rib
    /// (`x = 50`), and the coverage at each row there, at `s` device pixels
    /// per unit.
    fn cross_section(width: f32, s: f32) -> (f32, Vec<(f32, f32)>) {
        let verts = tessellate_polyline(
            &[[10.0, 50.0], [50.0, 50.0], [90.0, 50.0]],
            width,
            false,
            LineJoin::Mitre,
            4.0,
            LineCap::Butt,
            WHITE,
            100.0,
            100.0,
        );
        let mut rows: Vec<(f32, f32)> = verts
            .iter()
            .filter(|v| (v.position[0] - 50.0).abs() < 1e-4)
            .map(|v| {
                let (p, c) = place(v, s);
                (p[1], c)
            })
            .collect();
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        rows.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5);
        let integral = rows
            .windows(2)
            .map(|w| (w[1].0 - w[0].0) * (w[0].1 + w[1].1) * 0.5)
            .sum::<f32>()
            * s;
        (integral, rows)
    }

    #[test]
    fn fringe_rows_ramp_from_full_to_none_and_keep_the_width() {
        for s in [1.0, 2.0] {
            // A 4-unit stroke: inner rows at full coverage half a device pixel
            // inside each edge, outer rows at none half a pixel outside, and
            // the coverage integrates to the stroke's width.
            let (integral, rows) = cross_section(4.0, s);
            let half_px = 0.5 / s;
            let expect = [
                (48.0 - half_px, 0.0),
                (48.0 + half_px, 1.0),
                (52.0 - half_px, 1.0),
                (52.0 + half_px, 0.0),
            ];
            assert_eq!(rows.len(), 4, "{rows:?}");
            for (r, e) in rows.iter().zip(expect) {
                assert!(
                    (r.0 - e.0).abs() < 1e-4 && (r.1 - e.1).abs() < 1e-6,
                    "{rows:?}"
                );
            }
            assert!((integral - 4.0 * s).abs() < 1e-3, "width {integral} at {s}");
        }
        // A hairline under a device pixel keeps its weight: the inner rows meet
        // at the centreline below full coverage, and the integral is still the
        // width.
        let (integral, rows) = cross_section(0.5, 1.0);
        assert!(rows.iter().all(|r| r.1 < 1.0), "{rows:?}");
        assert!((integral - 0.5).abs() < 1e-3, "hairline weight {integral}");
    }

    #[test]
    fn a_fringed_segment_triples_the_ribbon() {
        // An interior segment was one quad (6 vertices); it is now a core quad
        // and a fringe quad per edge (18). Two segments and two flat ends:
        let verts = tessellate_polyline(
            &[[10.0, 50.0], [50.0, 50.0], [90.0, 50.0]],
            4.0,
            false,
            LineJoin::Mitre,
            4.0,
            LineCap::Butt,
            WHITE,
            100.0,
            100.0,
        );
        assert_eq!(verts.len(), 2 * 18 + 2 * 6);
    }

    #[test]
    fn a_sharp_bevel_shares_its_inner_point() {
        // A tight V with a bevel join: the inside of the turn is one mitred
        // point, so the two segments' inner edges (and fringes) meet rather
        // than cross, and no core triangle is folded over.
        let verts = tessellate_polyline(
            &[[10.0, 10.0], [50.0, 80.0], [90.0, 10.0]],
            6.0,
            false,
            LineJoin::Bevel,
            4.0,
            LineCap::Butt,
            WHITE,
            100.0,
            100.0,
        );
        // Every core triangle (all three vertices inner rows) keeps one
        // orientation after placement, and together they cover the stroke's
        // area once: no overlap at the joint.
        let mut area = 0.0;
        let mut signs = (0, 0);
        for tri in verts.chunks_exact(3) {
            if tri.iter().any(|v| v.aa[2] > 0.5) {
                continue;
            }
            let p: Vec<[f32; 2]> = tri.iter().map(|v| place(v, 1.0).0).collect();
            let a = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
            if a.abs() < 1e-6 {
                continue;
            }
            if a > 0.0 {
                signs.0 += 1
            } else {
                signs.1 += 1
            }
            area += a.abs() * 0.5;
        }
        assert!(signs.0 == 0 || signs.1 == 0, "folded triangles {signs:?}");
        // Each arm is ~80.6 long and the core is 5 wide (one pixel of fringe
        // taken off): the shared inner point means the core is no more than
        // the two arms' rectangles plus the outer bevel wedge.
        let arm = (40.0f32 * 40.0 + 70.0 * 70.0).sqrt();
        assert!(area < 2.0 * arm * 5.0 + 20.0, "core area {area}");
    }

    #[test]
    fn dotted_stroke_emits_discs() {
        let mut verts = Vec::new();
        let item = line_item(
            StrokePattern::Dotted {
                spacing: 10.0,
                offset: 0.0,
            },
            LineCap::Butt,
        );
        let stroke = item.stroke.clone().unwrap();
        emit_polyline_stroke(&mut verts, &item, &stroke, WHITE, 100.0, 100.0);
        // Dots at 0, 10, .., 100 = 11 discs of 10 steps, each a core quad (one
        // of its triangles empty at the hub) and a rim fringe quad.
        assert_eq!(verts.len(), 11 * 10 * 12);
    }

    #[test]
    fn degenerate_dash_pattern_falls_back_to_solid() {
        let mut dashed = Vec::new();
        let dashed_item = line_item(
            StrokePattern::Dashed {
                dash_length: 10.0,
                gap_length: 0.0,
                offset: 0.0,
            },
            LineCap::Butt,
        );
        let dashed_stroke = dashed_item.stroke.clone().unwrap();
        emit_polyline_stroke(
            &mut dashed,
            &dashed_item,
            &dashed_stroke,
            WHITE,
            100.0,
            100.0,
        );
        let mut solid = Vec::new();
        let solid_item = line_item(StrokePattern::Solid, LineCap::Butt);
        let solid_stroke = solid_item.stroke.clone().unwrap();
        emit_polyline_stroke(&mut solid, &solid_item, &solid_stroke, WHITE, 100.0, 100.0);
        assert_eq!(dashed.len(), solid.len());
    }

    #[test]
    fn dashes_are_cut_at_the_pattern_lengths_with_the_offset() {
        // dash 10, gap 5, offset 3: the pattern starts 3 into its first dash,
        // so the dashes run over [0, 7), [12, 22), [27, 37), ...
        let (pts, cum) = polyline_arc_table(&[[0.0, 0.0], [100.0, 0.0]], false);
        let pieces = dash_pieces(&pts, &cum, 10.0, 5.0, 3.0, false);
        let spans: Vec<(f32, f32)> = pieces
            .iter()
            .map(|(p, arcs)| {
                assert!((p.first().unwrap()[0] - arcs[0]).abs() < 1e-4);
                assert!((p.last().unwrap()[0] - arcs.last().unwrap()).abs() < 1e-4);
                (arcs[0], *arcs.last().unwrap())
            })
            .collect();
        let expect = [
            (0.0, 7.0),
            (12.0, 22.0),
            (27.0, 37.0),
            (42.0, 52.0),
            (57.0, 67.0),
            (72.0, 82.0),
            (87.0, 97.0),
        ];
        assert_eq!(spans.len(), expect.len(), "{spans:?}");
        for (s, e) in spans.iter().zip(expect) {
            assert!(
                (s.0 - e.0).abs() < 1e-3 && (s.1 - e.1).abs() < 1e-3,
                "{spans:?}"
            );
        }
    }

    #[test]
    fn an_open_vector_subpath_strokes_open() {
        use crate::renderer::types::SubPath;
        // A "U" whose fill would close it along the top edge.
        let u = SubPath::new([0.0, 0.0])
            .line_to([0.0, 50.0])
            .line_to([50.0, 50.0])
            .line_to([50.0, 0.0]);
        let flat = super::super::overlay_vector::flatten_for_stroke(&[u.clone()], 0.2);
        assert_eq!(flat.len(), 1);
        assert!(!flat[0].1, "an open subpath must stay open");
        assert_eq!(flat[0].0.len(), 4);
        let closed = super::super::overlay_vector::flatten_for_stroke(&[u.close()], 0.2);
        assert!(closed[0].1);
        assert_eq!(closed[0].0.len(), 4);

        let stroke = OverlayStroke::new(4.0, Colour::WHITE);
        // Whether any triangle covers the middle of the closing edge.
        let along_top = |verts: &[crate::resources::OverlayTextVertex]| {
            let q = [25.0f32, 0.0f32];
            verts.chunks_exact(3).any(|t| {
                let [a, b, c] = [t[0].position, t[1].position, t[2].position];
                let side = |p: [f32; 2], r: [f32; 2]| {
                    (r[0] - p[0]) * (q[1] - p[1]) - (r[1] - p[1]) * (q[0] - p[0])
                };
                let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
                if area.abs() < 1e-6 {
                    return false;
                }
                let (d1, d2, d3) = (side(a, b), side(b, c), side(c, a));
                let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
                let pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
                !(neg && pos)
            })
        };
        let mut open_verts = Vec::new();
        emit_stroke_path(&mut open_verts, &flat[0].0, false, None, &stroke, WHITE);
        let mut closed_verts = Vec::new();
        emit_stroke_path(&mut closed_verts, &closed[0].0, true, None, &stroke, WHITE);
        assert!(
            !along_top(&open_verts),
            "the open stroke drew its closing edge"
        );
        assert!(along_top(&closed_verts));
        assert!(open_verts.len() < closed_verts.len());
    }

    /// `hairline_offset` in `overlay_text.wgsl`: the local offset of a
    /// hairline's outer row under the group linear map `m` (row-major).
    fn hairline_offset(d: [f32; 2], m: [f32; 4], ppp: f32) -> [f32; 2] {
        let det = m[0] * m[3] - m[1] * m[2];
        let nt = [
            (d[0] * m[3] - d[1] * m[2]) / det,
            (d[1] * m[0] - d[0] * m[1]) / det,
        ];
        let len = (nt[0] * nt[0] + nt[1] * nt[1]).sqrt();
        let k = (d[0] * d[0] + d[1] * d[1]).sqrt() / (len * ppp);
        let ns = [nt[0] * k, nt[1] * k];
        [
            (m[3] * ns[0] - m[1] * ns[1]) / det,
            (m[0] * ns[1] - m[2] * ns[0]) / det,
        ]
    }

    /// A hairline vertex's device position and coverage under the group map
    /// `m` at `ppp` pixels per point.
    fn place_hairline(
        v: &crate::resources::OverlayTextVertex,
        m: [f32; 4],
        ppp: f32,
    ) -> ([f32; 2], f32) {
        assert!(v.aa[3] < 0.0, "not a hairline vertex");
        let (p, c) = if v.aa[2] > 0.5 {
            let o = hairline_offset([v.aa[0], v.aa[1]], m, ppp);
            ([v.position[0] + o[0], v.position[1] + o[1]], 0.0)
        } else {
            (v.position, 1.0)
        };
        let g = [m[0] * p[0] + m[1] * p[1], m[2] * p[0] + m[3] * p[1]];
        ([g[0] * ppp, g[1] * ppp], c)
    }

    /// The device-pixel cross-section of the middle of a hairline from
    /// `(10, 50)` to `(90, 50)`, item-transformed by `item`, under the group
    /// map `m`: the rows' distances along the line's device normal, and the
    /// coverage integral across it.
    fn hairline_section(
        item: Option<OverlayRotation>,
        m: [f32; 4],
        ppp: f32,
    ) -> (Vec<(f32, f32)>, f32) {
        let mut verts = tessellate_polyline(
            &[[10.0, 50.0], [50.0, 50.0], [90.0, 50.0]],
            0.0,
            false,
            LineJoin::Mitre,
            4.0,
            LineCap::Butt,
            WHITE,
            100.0,
            100.0,
        );
        assert!(!verts.is_empty());
        let mid: Vec<usize> = (0..verts.len())
            .filter(|&i| (verts[i].position[0] - 50.0).abs() < 1e-4)
            .collect();
        rotate_vertices_from(&mut verts, 0, item);
        let centre = place_hairline(&verts[mid[0]], m, ppp).0;
        // The device normal of the line, which runs along x before any map.
        let dir = |p: [f32; 2]| {
            let q = item.map_or(p, |r| r.apply(p));
            [m[0] * q[0] + m[1] * q[1], m[2] * q[0] + m[3] * q[1]]
        };
        let (a, b) = (dir([10.0, 50.0]), dir([90.0, 50.0]));
        let t = [b[0] - a[0], b[1] - a[1]];
        let tl = (t[0] * t[0] + t[1] * t[1]).sqrt();
        let n = [-t[1] / tl, t[0] / tl];
        let mut rows: Vec<(f32, f32)> = mid
            .iter()
            .map(|&i| {
                let (p, c) = place_hairline(&verts[i], m, ppp);
                ((p[0] - centre[0]) * n[0] + (p[1] - centre[1]) * n[1], c)
            })
            .collect();
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        rows.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-4 && (a.1 - b.1).abs() < 1e-4);
        let integral = rows
            .windows(2)
            .map(|w| (w[1].0 - w[0].0) * (w[0].1 + w[1].1) * 0.5)
            .sum::<f32>();
        (rows, integral)
    }

    #[test]
    fn a_hairline_is_one_device_pixel_at_any_scale() {
        let check = |rows: Vec<(f32, f32)>, integral: f32, what: &str| {
            assert_eq!(rows.len(), 3, "{what}: {rows:?}");
            for (r, e) in rows.iter().zip([(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]) {
                assert!(
                    (r.0 - e.0).abs() < 1e-3 && (r.1 - e.1).abs() < 1e-6,
                    "{what}: {rows:?}"
                );
            }
            assert!((integral - 1.0).abs() < 1e-3, "{what}: weight {integral}");
        };
        // Group scale and pixels per point.
        for k in [1.0, 3.0, 0.5] {
            for ppp in [1.0, 2.0] {
                let (rows, w) = hairline_section(None, [k, 0.0, 0.0, k], ppp);
                check(rows, w, &format!("group scale {k} at ppp {ppp}"));
            }
        }
        // A group stretch and a skew.
        let (rows, w) = hairline_section(None, [1.0, 0.0, 0.0, 3.0], 1.0);
        check(rows, w, "stretch");
        let (rows, w) = hairline_section(None, [2.0, 0.5, 0.3, 1.5], 1.0);
        check(rows, w, "skew");
        // An item scale baked into the vertices.
        for k in [3.0, 0.5] {
            let mut t = crate::renderer::types::OverlayTransform::IDENTITY;
            t.scale = k;
            let (rows, w) = hairline_section(
                OverlayRotation::new(&t, [50.0, 50.0]),
                [1.0, 0.0, 0.0, 1.0],
                1.0,
            );
            check(rows, w, &format!("item scale {k}"));
        }
    }

    #[test]
    fn square_caps_extend_by_half_the_width() {
        let tess = |cap| {
            tessellate_polyline(
                &[[10.0, 50.0], [90.0, 50.0]],
                4.0,
                false,
                LineJoin::Mitre,
                4.0,
                cap,
                WHITE,
                100.0,
                100.0,
            )
        };
        let span = |v: &[crate::resources::OverlayTextVertex]| {
            v.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| {
                (lo.min(v.position[0]), hi.max(v.position[0]))
            })
        };
        assert_eq!(span(&tess(LineCap::Butt)), (10.0, 90.0));
        assert_eq!(span(&tess(LineCap::Square)), (8.0, 92.0));
    }

    #[test]
    fn widths_per_point_set_the_stroke_width_along_the_line() {
        let item = OverlayPolylineItem::new(vec![[10.0, 50.0], [50.0, 50.0], [90.0, 50.0]])
            .with_thickness(1.0)
            .with_widths(vec![2.0, 6.0, 10.0]);
        let mut verts = Vec::new();
        emit_polyline_stroke(
            &mut verts,
            &item,
            item.stroke.as_ref().unwrap(),
            WHITE,
            100.0,
            100.0,
        );
        let half_at = |x: f32| {
            verts
                .iter()
                .filter(|v| (v.position[0] - x).abs() < 1e-4)
                .map(|v| (v.position[1] - 50.0).abs())
                .fold(0.0f32, f32::max)
        };
        assert!((half_at(10.0) - 1.0).abs() < 1e-4);
        assert!((half_at(50.0) - 3.0).abs() < 1e-4);
        assert!((half_at(90.0) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn a_round_join_sweeps_the_outer_corner() {
        let corner = |join| {
            tessellate_polyline(
                &[[50.0, 10.0], [50.0, 50.0], [90.0, 50.0]],
                4.0,
                false,
                join,
                4.0,
                LineCap::Butt,
                WHITE,
                100.0,
                100.0,
            )
        };
        let near = |verts: &[crate::resources::OverlayTextVertex], p: [f32; 2]| {
            verts
                .iter()
                .any(|v| (v.position[0] - p[0]).abs() < 1e-3 && (v.position[1] - p[1]).abs() < 1e-3)
        };
        // The outer corner is round the point (50, 50) at the half-width, 2:
        // some vertex sits on that arc strictly between its two ends.
        let round = corner(LineJoin::Round);
        let on_arc = round.iter().any(|v| {
            let d = [v.position[0] - 50.0, v.position[1] - 50.0];
            ((d[0] * d[0] + d[1] * d[1]).sqrt() - 2.0).abs() < 1e-3 && d[0] < -0.1 && d[1] > 0.1
        });
        assert!(on_arc, "no arc at the corner");
        assert!(!near(&round, [48.0, 52.0]), "a round join drew the mitre");
        assert!(near(&corner(LineJoin::Mitre), [48.0, 52.0]));
    }
}

/// Number of concentric bands used to fake a blurred edge on tessellated
/// geometry (polyline strokes and vector fills). Glyphs do not use this: their
/// blur is baked exactly into the atlas cell. Five is enough that the banding
/// is not visible at the blur distances overlays actually use.
const SHADOW_BANDS: usize = 5;

/// Coverage of a shadow layer at `d` pixels outside its spread edge, matching
/// `shadow_coverage` in `overlay_shape.wgsl` so the same `ShadowLayer` reads the
/// same way on an SDF shape and on tessellated geometry.
fn shadow_coverage(d: f32, blur: f32, falloff: f32) -> f32 {
    if blur <= 0.0 {
        return if d <= 0.0 { 1.0 } else { 0.0 };
    }
    let t = (d / blur).clamp(0.0, 1.0);
    let base = 1.0 - (t * t * (3.0 - 2.0 * t));
    if (falloff - 1.0).abs() <= f32::EPSILON {
        base
    } else {
        base.powf(falloff)
    }
}

/// The (offset-from-edge, per-band alpha) pairs to draw a shadow layer as
/// concentric bands, widest first.
///
/// Bands are drawn outside-in and composited src-over, so each band's own alpha
/// is solved from the coverage the previous bands already laid down. That makes
/// the accumulated alpha follow the target curve instead of simply summing.
pub(super) fn shadow_bands(layer: &crate::renderer::types::ShadowLayer) -> Vec<(f32, f32)> {
    let blur = layer.blur.max(0.0);
    if blur <= 0.0 {
        return vec![(0.0, 1.0)];
    }
    let mut out = Vec::with_capacity(SHADOW_BANDS);
    let mut acc = 0.0f32;
    for i in 0..SHADOW_BANDS {
        // i = 0 is the outermost band, at the full blur distance.
        let d = blur * (1.0 - i as f32 / SHADOW_BANDS as f32);
        let c = shadow_coverage(d, blur, layer.falloff);
        if c <= 0.0 {
            continue;
        }
        let a = if acc >= 1.0 {
            0.0
        } else {
            ((c - acc) / (1.0 - acc)).clamp(0.0, 1.0)
        };
        if a > 0.0 {
            out.push((d, a));
            acc = c;
        }
    }
    out
}

/// Mitre limit used when a shadow band follows a tessellated contour. Flattened
/// curves have shallow joins, so the limit only ever bites on a genuine spike.
const CONTOUR_BAND_MITRE_LIMIT: f32 = 4.0;

/// Which side of a contour faces away from the area it encloses, as a sign on
/// the left-hand normal. A contour wound counter-clockwise encloses what is to
/// its left, so its outward side is the right one.
pub(super) fn contour_outward_sign(points: &[[f32; 2]]) -> f32 {
    if polygon_area(points) > 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// Emit one side of a shadow band along a closed contour: from the contour out
/// to `width` on the side `side` selects (a sign on the left-hand normal), with
/// `shift` moving the far edge the way a layer offset does.
///
/// This is how a shadow layer is clipped to one side of a tessellated
/// silhouette. An outer layer takes the outward side and so never paints over
/// the fill it belongs to; an inner layer takes the other and never leaves it.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_contour_band(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    points: &[[f32; 2]],
    side: f32,
    width: f32,
    shift: [f32; 2],
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    if width <= 0.0 || points.len() < 3 {
        return;
    }
    let (lo, hi) = if side > 0.0 {
        (0.0, width)
    } else {
        (-width, 0.0)
    };
    verts.extend(tessellate_ribbon(
        points,
        lo,
        hi,
        shift,
        true,
        crate::renderer::types::LineJoin::Mitre,
        CONTOUR_BAND_MITRE_LIMIT,
        crate::renderer::types::LineCap::Butt,
        Fringe::band(lo, hi, false),
        colour,
        vp_w,
        vp_h,
    ));
}

/// Emit one outer shadow layer of a polyline: a band running outward from the
/// edge of what the item covers, once per blur band.
///
/// The band stops at that edge rather than painting the dilated silhouette
/// behind the item, so a translucent path never tints itself through its own
/// shadow. What counts as the edge depends on the item: a closed path with a
/// fill is bounded by its contour, and anything else is bounded by both sides
/// of its stroke, ends included.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_polyline_shadow(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    poly: &crate::renderer::types::OverlayPolylineItem,
    layer: &crate::renderer::types::ShadowLayer,
    opacity: f32,
    vp_w: f32,
    vp_h: f32,
) {
    let base = layer.colour.to_linear_rgba();
    let filled = poly.closed && poly.style.fill.is_set();
    for (d, band_alpha) in shadow_bands(layer) {
        let colour = apply_opacity(base, opacity * band_alpha);
        let width = layer.spread + d;
        if colour[3] <= 0.0 || width <= 0.0 {
            continue;
        }
        if filled {
            let side = contour_outward_sign(&poly.points);
            emit_contour_band(
                verts,
                &poly.points,
                side,
                width,
                layer.offset,
                colour,
                vp_w,
                vp_h,
            );
        } else {
            emit_stroke_outer_shadow(verts, poly, width, layer.offset, colour, vp_w, vp_h);
        }
    }
}

/// Emit the band outside a stroke: one ribbon alongside each of its two edges,
/// plus the piece that wraps each open end.
///
/// A dash pattern is followed dash by dash, and a dotted pattern becomes a ring
/// around each dot, so the shadow keeps the shape of the stroke it belongs to.
#[allow(clippy::too_many_arguments)]
fn emit_stroke_outer_shadow(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    poly: &crate::renderer::types::OverlayPolylineItem,
    width: f32,
    shift: [f32; 2],
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    use crate::renderer::types::StrokePattern;
    let stroke = resolved_stroke(poly);
    let half = (stroke.width * 0.5).max(0.0);
    let mut sides = |points: &[[f32; 2]], closed: bool| {
        for (lo, hi) in [(half, half + width), (-half - width, -half)] {
            verts.extend(tessellate_ribbon(
                points,
                lo,
                hi,
                shift,
                closed,
                stroke.join,
                stroke.mitre_limit,
                crate::renderer::types::LineCap::Butt,
                // The ends meet the wrap `emit_stroke_end_shadow` draws.
                Fringe::band(lo, hi, false),
                colour,
                vp_w,
                vp_h,
            ));
        }
        if !closed {
            emit_stroke_end_shadow(verts, points, half, width, stroke.cap, colour, vp_w, vp_h);
        }
    };
    match stroke.pattern {
        StrokePattern::Solid => sides(&poly.points, poly.closed),
        StrokePattern::Dashed {
            dash_length,
            gap_length,
            offset,
        } => {
            if dash_length <= 0.0 || gap_length <= 0.0 {
                sides(&poly.points, poly.closed);
                return;
            }
            let (pts, cum) = polyline_arc_table(&poly.points, poly.closed);
            for sub in dash_subpaths(
                &pts,
                &cum,
                dash_length.max(0.25),
                gap_length.max(0.25),
                offset,
                poly.closed,
            ) {
                sides(&sub, false);
            }
        }
        StrokePattern::Dotted { spacing, offset } => {
            let (pts, cum) = polyline_arc_table(&poly.points, poly.closed);
            if pts.len() < 2 {
                return;
            }
            let total = *cum.last().unwrap();
            let spacing = spacing.max(0.5);
            let mut s = offset.rem_euclid(spacing);
            let limit = if poly.closed {
                total - 1e-3
            } else {
                total + 1e-3
            };
            while s < limit {
                let c = point_at_arc(&pts, &cum, s);
                emit_ring(verts, c, half, half + width, colour, vp_w, vp_h);
                s += spacing;
            }
        }
    }
}

/// Emit the shadow that wraps each open end of a stroke: the points outside the
/// stroke, within `width` of it, past where it stops.
///
/// A round cap ends in a half-disc, so its wrap is a half-annulus. A butt or
/// square cap ends flat, so its wrap is a rectangle across the end with a
/// quarter-disc at each corner.
#[allow(clippy::too_many_arguments)]
fn emit_stroke_end_shadow(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    points: &[[f32; 2]],
    half: f32,
    width: f32,
    cap: crate::renderer::types::LineCap,
    colour: [f32; 4],
    vp_w: f32,
    vp_h: f32,
) {
    use crate::renderer::types::LineCap;
    let n = points.len();
    if n < 2 {
        return;
    }
    let unit = |v: [f32; 2]| {
        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
        if len < 1e-6 {
            [0.0, 0.0]
        } else {
            [v[0] / len, v[1] / len]
        }
    };
    let ends = [
        (
            points[0],
            unit([points[0][0] - points[1][0], points[0][1] - points[1][1]]),
        ),
        (
            points[n - 1],
            unit([
                points[n - 1][0] - points[n - 2][0],
                points[n - 1][1] - points[n - 2][1],
            ]),
        ),
    ];
    for (p, out) in ends {
        if out == [0.0, 0.0] {
            continue;
        }
        let nrm = [-out[1], out[0]];
        match cap {
            LineCap::Round => {
                emit_half_ring(verts, p, out, half, half + width, colour, vp_w, vp_h);
            }
            LineCap::Butt | LineCap::Square => {
                // A square cap puts the flat end half a thickness further on.
                let e = if cap == LineCap::Square {
                    [p[0] + out[0] * half, p[1] + out[1] * half]
                } else {
                    p
                };
                let corner = |s: f32| [e[0] + nrm[0] * half * s, e[1] + nrm[1] * half * s];
                let (c0, c1) = (corner(1.0), corner(-1.0));
                emit_straight_band(verts, c0, c1, out, width, colour);
                for (c, from) in [(c0, nrm), (c1, [-nrm[0], -nrm[1]])] {
                    emit_fan(verts, c, from, out, width, colour, vp_w, vp_h);
                }
            }
        }
    }
}

/// Emit a fan sweeping from direction `from` to direction `to` (the short way),
/// at `radius` about `centre`, with a fringe round its rim.
#[allow(clippy::too_many_arguments)]
fn emit_fan(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    centre: [f32; 2],
    from: [f32; 2],
    to: [f32; 2],
    radius: f32,
    colour: [f32; 4],
    _vp_w: f32,
    _vp_h: f32,
) {
    let a0 = from[1].atan2(from[0]);
    let mut sweep = to[1].atan2(to[0]) - a0;
    while sweep > std::f32::consts::PI {
        sweep -= std::f32::consts::TAU;
    }
    while sweep < -std::f32::consts::PI {
        sweep += std::f32::consts::TAU;
    }
    emit_arc_band(
        verts,
        centre,
        0.0,
        radius,
        a0,
        sweep,
        4,
        [EdgeAa::None, EdgeAa::Centred],
        colour,
    );
}

/// Emit an annulus between two radii about `centre`, with a fringe on both
/// rims.
#[allow(clippy::too_many_arguments)]
fn emit_ring(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    centre: [f32; 2],
    r_inner: f32,
    r_outer: f32,
    colour: [f32; 4],
    _vp_w: f32,
    _vp_h: f32,
) {
    emit_arc_band(
        verts,
        centre,
        r_inner,
        r_outer,
        0.0,
        std::f32::consts::TAU,
        16,
        // The inner rim lies on the dot or cap the shadow belongs to.
        [EdgeAa::Outset, EdgeAa::Centred],
        colour,
    );
}

/// Emit half an annulus about `centre`, on the side `outward` points to, with
/// a fringe on both rims.
#[allow(clippy::too_many_arguments)]
fn emit_half_ring(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    centre: [f32; 2],
    outward: [f32; 2],
    r_inner: f32,
    r_outer: f32,
    colour: [f32; 4],
    _vp_w: f32,
    _vp_h: f32,
) {
    let base = outward[1].atan2(outward[0]) - std::f32::consts::FRAC_PI_2;
    emit_arc_band(
        verts,
        centre,
        r_inner,
        r_outer,
        base,
        std::f32::consts::PI,
        8,
        // The inner rim lies on the dot or cap the shadow belongs to.
        [EdgeAa::Outset, EdgeAa::Centred],
        colour,
    );
}

/// Emit one inner shadow layer of a polyline: a band running inward from the
/// edge of what the item covers, once per blur band.
///
/// A closed path with a fill covers an area, so the band runs inward from the
/// contour, as it does on any other filled silhouette. Anything else covers
/// only its stroke, so the band eats into the stroke from both of its edges,
/// which is what an inset layer means for a line. In that case the layer offset
/// is left out: a stroke has no interior for the band to slide within.
pub(super) fn emit_polyline_inner_shadow(
    verts: &mut Vec<crate::resources::OverlayTextVertex>,
    poly: &crate::renderer::types::OverlayPolylineItem,
    layer: &crate::renderer::types::ShadowLayer,
    opacity: f32,
    vp_w: f32,
    vp_h: f32,
) {
    let base = layer.colour.to_linear_rgba();
    let area = poly.closed && poly.style.fill.is_set();
    let stroke_width = resolved_stroke(poly).width;
    for (d, band_alpha) in shadow_bands(layer) {
        let colour = apply_opacity(base, opacity * band_alpha);
        let width = layer.spread + d;
        if colour[3] <= 0.0 || width <= 0.0 {
            continue;
        }
        if area {
            let side = -contour_outward_sign(&poly.points);
            emit_contour_band(
                verts,
                &poly.points,
                side,
                width,
                layer.offset,
                colour,
                vp_w,
                vp_h,
            );
        } else if stroke_width > 0.0 {
            let half = stroke_width * 0.5;
            let w = width.min(half);
            // Each band's edge on the stroke's edge is the outset one.
            let on_hi = Fringe {
                lo: EdgeAa::Centred,
                hi: EdgeAa::Outset,
                ends: true,
            };
            let on_lo = Fringe {
                lo: EdgeAa::Outset,
                hi: EdgeAa::Centred,
                ends: true,
            };
            for (lo, hi, fringe) in [(half - w, half, on_hi), (-half, w - half, on_lo)] {
                verts.extend(tessellate_ribbon(
                    &poly.points,
                    lo,
                    hi,
                    [0.0, 0.0],
                    poly.closed,
                    crate::renderer::types::LineJoin::Mitre,
                    CONTOUR_BAND_MITRE_LIMIT,
                    crate::renderer::types::LineCap::Butt,
                    fringe,
                    colour,
                    vp_w,
                    vp_h,
                ));
            }
        }
    }
}

/// An item transform baked into vertex positions: the linear part of an
/// `OverlayTransform` (rotation, scale, stretch and skew) about a centre in
/// the same pixel space as the vertices.
#[derive(Clone, Copy)]
pub(super) struct OverlayRotation {
    linear: [[f32; 2]; 2],
    centre: [f32; 2],
}

impl OverlayRotation {
    /// The linear part of `transform` about `centre`. `None` when it would
    /// move nothing, so callers can pass one through unconditionally.
    ///
    /// This is the inner half of the composition contract. The outer half (the
    /// group transform) rides the GPU instance; an immediate item has no
    /// instance slot of its own, so its half is folded in here.
    pub(super) fn new(
        transform: &crate::renderer::types::OverlayTransform,
        centre: [f32; 2],
    ) -> Option<Self> {
        if transform.linear_is_identity() {
            return None;
        }
        Some(Self {
            linear: transform.linear(),
            centre,
        })
    }

    /// The centre of a box, offset by a pivot measured from that centre. This is
    /// the `OverlayShapeItem::rotation_pivot` contract: `[0, 0]` is the centre.
    pub(super) fn pivot_point(box_min: [f32; 2], box_size: [f32; 2], pivot: [f32; 2]) -> [f32; 2] {
        [
            box_min[0] + box_size[0] * 0.5 + pivot[0],
            box_min[1] + box_size[1] * 0.5 + pivot[1],
        ]
    }

    fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let dx = p[0] - self.centre[0];
        let dy = p[1] - self.centre[1];
        let m = self.linear;
        [
            self.centre[0] + m[0][0] * dx + m[0][1] * dy,
            self.centre[1] + m[1][0] * dx + m[1][1] * dy,
        ]
    }
}

/// Where an analytic shape lands under its item transform, in the item's
/// pixel space before any group transform.
pub(super) struct ShapeFrame {
    /// The axis-aligned quad covering the transformed shape plus its padding:
    /// centre `xy`, half-extents `zw`.
    pub quad: [f32; 4],
    /// The pivot in the same pixel space.
    pub pivot_px: [f32; 2],
    /// The inverse of the item's linear map, row-major `[a, b, c, d]`.
    pub linv: [f32; 4],
    /// The item's uniform scale, which its shadow offsets are given in.
    pub item_scale: f32,
}

impl ShapeFrame {
    /// The quad's six corners (two triangles) as `(position, offset from the
    /// pivot)`, for a pipeline that still takes the quad as vertices.
    pub(super) fn corners(&self) -> [([f32; 2], [f32; 2]); 6] {
        let [cx, cy, hx, hy] = self.quad;
        let corner = |sx: f32, sy: f32| {
            let p = [cx + sx * hx, cy + sy * hy];
            (p, [p[0] - self.pivot_px[0], p[1] - self.pivot_px[1]])
        };
        [
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ]
    }
}

/// The frame of a shape whose box is `half` either side of `centre` (shape
/// units) under `transform`, with `pad` shape units of shadow and
/// anti-aliasing room around it, scaled to pixels by the map's largest
/// stretch so the padding still covers under a non-uniform scale. `None` when
/// the map has no inverse (a zero stretch), which draws nothing.
pub(super) fn shape_frame(
    transform: &crate::renderer::types::OverlayTransform,
    centre: [f32; 2],
    half: [f32; 2],
    pad: f32,
) -> Option<ShapeFrame> {
    let m = transform.linear();
    let inv = transform.inverse_linear()?;
    let piv = transform.pivot;
    let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            let dx = sx * half[0] - piv[0];
            let dy = sy * half[1] - piv[1];
            let x = m[0][0] * dx + m[0][1] * dy + piv[0];
            let y = m[1][0] * dx + m[1][1] * dy + piv[1];
            min = [min[0].min(x), min[1].min(y)];
            max = [max[0].max(x), max[1].max(y)];
        }
    }
    // The spectral norm of a 2x2: the largest factor by which it stretches any
    // direction, so a pad in shape units covers in pixels.
    let a = m[0][0] * m[0][0] + m[1][0] * m[1][0];
    let b = m[0][1] * m[0][1] + m[1][1] * m[1][1];
    let c = m[0][0] * m[0][1] + m[1][0] * m[1][1];
    let norm = ((a + b) * 0.5 + (((a - b) * 0.5).powi(2) + c * c).sqrt()).sqrt();
    let pad_px = pad * norm;
    Some(ShapeFrame {
        quad: [
            centre[0] + (min[0] + max[0]) * 0.5,
            centre[1] + (min[1] + max[1]) * 0.5,
            (max[0] - min[0]) * 0.5 + pad_px,
            (max[1] - min[1]) * 0.5 + pad_px,
        ],
        pivot_px: [centre[0] + piv[0], centre[1] + piv[1]],
        linv: [inv[0][0], inv[0][1], inv[1][0], inv[1][1]],
        item_scale: transform.scale,
    })
}

/// Apply an item transform to the positions of `verts[from..]` in place. A
/// `None` transform is a no-op, so callers can pass one through
/// unconditionally.
pub(super) fn rotate_vertices_from(
    verts: &mut [crate::resources::OverlayTextVertex],
    from: usize,
    rot: Option<OverlayRotation>,
) {
    let Some(r) = rot else {
        return;
    };
    let m = r.linear;
    // A fringe is measured in device pixels, so its direction turns with the
    // item but keeps its length, and the inset limit (in the item's units)
    // scales with it.
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let sigma = det.abs().sqrt();
    let inv_sigma = if sigma > 1e-6 { 1.0 / sigma } else { 0.0 };
    for v in &mut verts[from..] {
        v.position = r.apply(v.position);
        if v.aa[2] != 0.0 && v.aa[3] < 0.0 {
            // A hairline's direction is a normal, which maps by the inverse
            // transpose; it keeps its length so it still moves one pixel.
            let [dx, dy, side, inset] = v.aa;
            let len = (dx * dx + dy * dy).sqrt();
            let t = if det.abs() > 1e-12 {
                [
                    (m[1][1] * dx - m[1][0] * dy) / det,
                    (m[0][0] * dy - m[0][1] * dx) / det,
                ]
            } else {
                [0.0, 0.0]
            };
            let tl = (t[0] * t[0] + t[1] * t[1]).sqrt();
            let k = if tl > 1e-12 { len / tl } else { 0.0 };
            v.aa = [t[0] * k, t[1] * k, side, inset];
        } else if v.aa[2] != 0.0 {
            let [dx, dy, side, inset] = v.aa;
            v.aa = [
                (m[0][0] * dx + m[0][1] * dy) * inv_sigma,
                (m[1][0] * dx + m[1][1] * dy) * inv_sigma,
                side,
                inset * sigma,
            ];
        }
    }
}
