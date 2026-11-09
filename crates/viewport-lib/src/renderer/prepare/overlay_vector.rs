//! Curve flattening and fill-rule tessellation for
//! [`OverlayShape::Vector`](crate::renderer::types::OverlayShape). Turns a set
//! of subpaths (line and Bezier segments) plus a fill rule into a triangle
//! list the overlay fill path draws.

use crate::renderer::types::{FillRule, PathSegment, SubPath};

use lyon_path::Path;
use lyon_path::math::point;
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule as LyonFillRule, FillTessellator, FillVertex,
    VertexBuffers,
};

/// A tessellated vector shape: 2D positions in path-local logical pixels plus a
/// triangle index list. Empty when the shape has no fillable area or
/// tessellation fails.
#[derive(Debug, Clone, Default)]
pub(crate) struct VectorMesh {
    pub positions: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

fn build_path(subpaths: &[SubPath]) -> Path {
    let mut builder = Path::builder();
    for sp in subpaths {
        if sp.segments.is_empty() {
            continue;
        }
        builder.begin(point(sp.start[0], sp.start[1]));
        for seg in &sp.segments {
            match *seg {
                PathSegment::Line { to } => {
                    builder.line_to(point(to[0], to[1]));
                }
                PathSegment::Quad { ctrl, to } => {
                    builder.quadratic_bezier_to(point(ctrl[0], ctrl[1]), point(to[0], to[1]));
                }
                PathSegment::Cubic { ctrl1, ctrl2, to } => {
                    builder.cubic_bezier_to(
                        point(ctrl1[0], ctrl1[1]),
                        point(ctrl2[0], ctrl2[1]),
                        point(to[0], to[1]),
                    );
                }
            }
        }
        // Fills close every subpath. `end(true)` matches how SVG fills an open
        // subpath (as if a closing segment ran back to the start).
        builder.end(true);
    }
    builder.build()
}

/// Flatten and tessellate `subpaths` under `fill_rule`. `tolerance` is the
/// maximum deviation, in logical pixels, of a flattened curve from the true
/// curve: smaller is smoother and produces more triangles.
pub(crate) fn tessellate(subpaths: &[SubPath], fill_rule: FillRule, tolerance: f32) -> VectorMesh {
    let path = build_path(subpaths);

    let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let options = FillOptions::tolerance(tolerance.max(1e-3)).with_fill_rule(match fill_rule {
        FillRule::NonZero => LyonFillRule::NonZero,
        FillRule::EvenOdd => LyonFillRule::EvenOdd,
    });

    let mut tessellator = FillTessellator::new();
    let result = tessellator.tessellate_path(
        &path,
        &options,
        &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| {
            let p = v.position();
            [p.x, p.y]
        }),
    );

    if result.is_err() {
        return VectorMesh::default();
    }

    VectorMesh {
        positions: buffers.vertices,
        indices: buffers.indices,
    }
}

/// Flatten `subpaths` into closed polygons, in path-local logical pixels, no
/// further than `tolerance` from the true curves. Every subpath is closed, as a
/// fill closes it.
pub(crate) fn flatten_path(subpaths: &[SubPath], tolerance: f32) -> Vec<Vec<[f32; 2]>> {
    use lyon_path::PathEvent;
    use lyon_path::iterator::PathIterator;
    let path = build_path(subpaths);
    let mut contours: Vec<Vec<[f32; 2]>> = Vec::new();
    for event in path.iter().flattened(tolerance.max(1e-3)) {
        match event {
            PathEvent::Begin { at } => contours.push(vec![[at.x, at.y]]),
            PathEvent::Line { to, .. } => {
                if let Some(c) = contours.last_mut() {
                    c.push([to.x, to.y]);
                }
            }
            PathEvent::End { first, .. } => {
                if let Some(c) = contours.last_mut() {
                    if c.len() > 1 && c.last() == Some(&[first.x, first.y]) {
                        c.pop();
                    }
                }
            }
            _ => {}
        }
    }
    contours
}

/// A filled outline with an anti-aliasing fringe along its boundary.
///
/// `positions`, `aa` (an `OverlayTextVertex::aa` each) and `sources` are per
/// vertex, and `indices` a triangle list over them: the fill's triangles, then
/// the fringe's. A fill vertex on the boundary is also the fringe's inner row,
/// so the fill insets by half a device pixel exactly where the fringe starts,
/// and its copy after the fill vertices is the outer row. `sources` gives for
/// each vertex the outline point it came from, as `(from, to, t)` indices into
/// the concatenated input contours: `points[from]` moved toward `points[to]`
/// by `t`. A vertex that is an input point has `from == to`. Vertices do not
/// map one to one to the input (coincident points merge and crossings add
/// vertices), so anything carried per input point is looked up through it.
#[derive(Debug, Clone, Default)]
pub(crate) struct FringedFill {
    pub positions: Vec<[f32; 2]>,
    pub aa: Vec<[f32; 4]>,
    pub sources: Vec<(usize, usize, f32)>,
    pub indices: Vec<u32>,
}

/// Longest a fill's fringe direction grows at a sharp corner, as a multiple
/// of the edge offset. Past it the corner's fringe is shortened rather than
/// reaching far past the tip.
const FILL_MITRE_LIMIT: f32 = 4.0;

/// Tessellate the closed `contours` under `fill_rule`, with a fringe along the
/// boundary of what fills.
///
/// The boundary is read off the triangulation (the edges only one triangle
/// uses, facing away from it), not off the contours, so it is the edge of the
/// filled area under either fill rule: a hole's edge faces into the hole, an
/// edge inside the filled area (a self-crossing, or a contour another one
/// covers) gets no fringe, and a spur or a doubled-back stretch, which fills
/// nothing, gets none either. A point where two filled regions touch keeps its
/// place and the fringe narrows to nothing there. Contours with fewer than
/// three points, or with a point that is not finite, are skipped.
pub(crate) fn tessellate_fringed(contours: &[Vec<[f32; 2]>], fill_rule: FillRule) -> FringedFill {
    let mut builder = Path::builder();
    // Endpoint ids are assigned by the builder, so keep the map back to the
    // input index rather than assuming they count from zero.
    let mut input_of: Vec<usize> = Vec::new();
    let mut push = |id: lyon_path::EndpointId, i: usize| {
        let slot = id.to_usize();
        if input_of.len() <= slot {
            input_of.resize(slot + 1, 0);
        }
        input_of[slot] = i;
    };
    let mut offset = 0;
    let mut any = false;
    for c in contours {
        let start = offset;
        offset += c.len();
        if c.len() < 3 || c.iter().flatten().any(|v| !v.is_finite()) {
            continue;
        }
        any = true;
        push(builder.begin(point(c[0][0], c[0][1])), start);
        for (i, p) in c.iter().enumerate().skip(1) {
            push(builder.line_to(point(p[0], p[1])), start + i);
        }
        builder.end(true);
    }
    if !any {
        return FringedFill::default();
    }
    let path = builder.build();

    let mut buffers: VertexBuffers<([f32; 2], (usize, usize, f32)), u32> = VertexBuffers::new();
    let input = |id: lyon_path::EndpointId| input_of.get(id.to_usize()).copied().unwrap_or(0);
    let result = FillTessellator::new().tessellate_with_ids(
        path.id_iter(),
        &path,
        None,
        &FillOptions::default().with_fill_rule(match fill_rule {
            FillRule::NonZero => LyonFillRule::NonZero,
            FillRule::EvenOdd => LyonFillRule::EvenOdd,
        }),
        &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| {
            let p = v.position();
            let source = match v.sources().next() {
                Some(lyon_tessellation::VertexSource::Endpoint { id }) => {
                    let i = input(id);
                    (i, i, 0.0)
                }
                Some(lyon_tessellation::VertexSource::Edge { from, to, t }) => {
                    (input(from), input(to), t)
                }
                None => (0, 0, 0.0),
            };
            ([p.x, p.y], source)
        }),
    );
    if let Err(e) = result {
        tracing::debug!(?e, "fringed fill tessellation failed");
        return FringedFill::default();
    }
    let (mut positions, mut sources): (Vec<[f32; 2]>, Vec<(usize, usize, f32)>) =
        buffers.vertices.into_iter().unzip();
    let mut indices = buffers.indices;
    let n = positions.len();

    // Boundary edges: used by exactly one triangle. Each keeps the triangle's
    // third vertex, which says which side the fill is on. Degenerate
    // triangles cover nothing and say nothing about the boundary.
    let mut edge_use: std::collections::HashMap<(u32, u32), (u32, u32)> =
        std::collections::HashMap::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0], tri[1], tri[2]];
        let (pa, pb, pc) = (
            positions[a as usize],
            positions[b as usize],
            positions[c as usize],
        );
        let area = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1]);
        if area.abs() < 1e-9 {
            continue;
        }
        for (u, v, w) in [(a, b, c), (b, c, a), (c, a, b)] {
            let e = edge_use.entry((u.min(v), u.max(v))).or_insert((0, w));
            e.0 += 1;
        }
    }
    // Per vertex: the sum of its boundary edges' outward normals, the first
    // of them, and how many there are.
    let mut normals: Vec<([f32; 2], [f32; 2], u32)> = vec![([0.0; 2], [0.0; 2], 0); n];
    let mut boundary: Vec<(u32, u32)> = Vec::new();
    for (&(u, v), &(count, third)) in &edge_use {
        if count != 1 {
            continue;
        }
        let (pu, pv, pt) = (
            positions[u as usize],
            positions[v as usize],
            positions[third as usize],
        );
        let d = [pv[0] - pu[0], pv[1] - pu[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if len < 1e-9 {
            continue;
        }
        let mut nrm = [-d[1] / len, d[0] / len];
        if nrm[0] * (pt[0] - pu[0]) + nrm[1] * (pt[1] - pu[1]) > 0.0 {
            nrm = [-nrm[0], -nrm[1]];
        }
        for w in [u, v] {
            let e = &mut normals[w as usize];
            if e.2 == 0 {
                e.1 = nrm;
            }
            e.0 = [e.0[0] + nrm[0], e.0[1] + nrm[1]];
            e.2 += 1;
        }
        boundary.push((u, v));
    }
    // Hash-map order is not stable, and the fringe's triangles should come
    // out the same every time.
    boundary.sort_unstable();

    // Each boundary vertex's fringe direction: the mitre of its two edges'
    // normals. Anything else (a point two regions touch at) stays put.
    let mut aa = vec![[0.0f32; 4]; n];
    let mut outer_of = vec![u32::MAX; n];
    for (i, &(sum, first, count)) in normals.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let dir = if count == 2 {
            let len = (sum[0] * sum[0] + sum[1] * sum[1]).sqrt();
            if len < 1e-4 {
                [0.0, 0.0]
            } else {
                let bis = [sum[0] / len, sum[1] / len];
                let cos = (bis[0] * first[0] + bis[1] * first[1]).max(1e-4);
                let k = (1.0 / cos).min(FILL_MITRE_LIMIT);
                [bis[0] * k, bis[1] * k]
            }
        } else if count == 1 {
            first
        } else {
            [0.0, 0.0]
        };
        aa[i] = [
            dir[0],
            dir[1],
            -1.0,
            super::overlay_geometry::FILL_MAX_INSET,
        ];
        outer_of[i] = positions.len() as u32;
        positions.push(positions[i]);
        sources.push(sources[i]);
        aa.push([dir[0], dir[1], 1.0, super::overlay_geometry::FILL_MAX_INSET]);
    }
    for (u, v) in boundary {
        let (uo, vo) = (outer_of[u as usize], outer_of[v as usize]);
        indices.extend_from_slice(&[u, uo, v, uo, vo, v]);
    }
    FringedFill {
        positions,
        aa,
        sources,
        indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// True if `pt` falls inside any triangle of the mesh.
    fn covered(mesh: &VectorMesh, pt: [f32; 2]) -> bool {
        fn sign(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
            (a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])
        }
        mesh.indices.chunks_exact(3).any(|tri| {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            let d1 = sign(pt, a, b);
            let d2 = sign(pt, b, c);
            let d3 = sign(pt, c, a);
            let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
            let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
            !(has_neg && has_pos)
        })
    }

    fn square(min: f32, max: f32) -> SubPath {
        SubPath::polygon(&[[min, min], [max, min], [max, max], [min, max]])
    }

    #[test]
    fn square_tessellates_and_covers_centre() {
        let mesh = tessellate(&[square(0.0, 10.0)], FillRule::NonZero, 0.1);
        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
        assert!(covered(&mesh, [5.0, 5.0]));
        assert!(!covered(&mesh, [15.0, 5.0]));
    }

    #[test]
    fn even_odd_hole_is_not_covered() {
        let mesh = tessellate(
            &[square(0.0, 10.0), square(3.0, 7.0)],
            FillRule::EvenOdd,
            0.1,
        );
        // Centre of the hole is empty; a point in the ring is filled.
        assert!(!covered(&mesh, [5.0, 5.0]));
        assert!(covered(&mesh, [1.0, 5.0]));
    }

    #[test]
    fn nonzero_same_winding_fills_hole() {
        // Two squares wound the same direction under NonZero: the inner region
        // has winding 2, still non-zero, so it fills (no hole).
        let mesh = tessellate(
            &[square(0.0, 10.0), square(3.0, 7.0)],
            FillRule::NonZero,
            0.1,
        );
        assert!(covered(&mesh, [5.0, 5.0]));
    }

    #[test]
    fn finer_tolerance_yields_more_vertices() {
        // A single cubic-heavy contour: a rough circle from four cubic quarters.
        let k = 8.0 * 0.5522847498;
        let circle = SubPath::new([18.0, 10.0])
            .cubic_to([18.0, 10.0 + k], [10.0 + k, 18.0], [10.0, 18.0])
            .cubic_to([10.0 - k, 18.0], [2.0, 10.0 + k], [2.0, 10.0])
            .cubic_to([2.0, 10.0 - k], [10.0 - k, 2.0], [10.0, 2.0])
            .cubic_to([10.0 + k, 2.0], [18.0, 10.0 - k], [18.0, 10.0])
            .close();
        let coarse = tessellate(&[circle.clone()], FillRule::NonZero, 2.0);
        let fine = tessellate(&[circle], FillRule::NonZero, 0.05);
        assert!(fine.positions.len() >= coarse.positions.len());
        assert!(fine.positions.len() > coarse.positions.len());
    }

    fn signed_area(p: &[[f32; 2]]) -> f32 {
        (0..p.len())
            .map(|i| {
                let (a, b) = (p[i], p[(i + 1) % p.len()]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<f32>()
            * 0.5
    }

    fn tessellate_polygon(outline: &[[f32; 2]]) -> FringedFill {
        tessellate_fringed(&[outline.to_vec()], FillRule::NonZero)
    }

    /// Fraction of the outline's area the mesh's triangles cover. A fringe
    /// quad has no area until the vertex shader moves its rows apart.
    fn coverage(mesh: &FringedFill, outline: &[[f32; 2]]) -> f32 {
        let filled: f32 = mesh
            .indices
            .chunks_exact(3)
            .map(|t| {
                signed_area(&[
                    mesh.positions[t[0] as usize],
                    mesh.positions[t[1] as usize],
                    mesh.positions[t[2] as usize],
                ])
                .abs()
            })
            .sum();
        filled / signed_area(outline).abs()
    }

    #[test]
    fn outline_through_a_point_twice_fills_completely() {
        // A spur down to (0, 10) and back along the base to (0, 10) again.
        let spur = [
            [0.0, 6.0],
            [0.0, 10.0],
            [2.0, 2.0],
            [4.0, 9.9],
            [6.0, 3.0],
            [6.0, 10.0],
            [0.0, 10.0],
        ];
        // Two squares meeting at one corner, visited twice.
        let touch = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        for outline in [&spur[..], &touch[..]] {
            let c = coverage(&tessellate_polygon(outline), outline);
            assert!((c - 1.0).abs() < 1e-3, "covered {c} of {outline:?}");
        }
    }

    #[test]
    fn sources_point_back_at_the_outline() {
        let outline = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]];
        let mesh = tessellate_polygon(&outline);
        assert_eq!(mesh.sources.len(), mesh.positions.len());
        for (p, &(from, to, t)) in mesh.positions.iter().zip(&mesh.sources) {
            let (a, b) = (outline[from], outline[to]);
            let q = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            assert!((p[0] - q[0]).abs() < 1e-4 && (p[1] - q[1]).abs() < 1e-4);
        }
    }

    #[test]
    fn large_outline_fills_completely() {
        // An area-chart-like outline: a dense wavy top over a flat base.
        let n = 4000;
        let mut outline: Vec<[f32; 2]> = (0..n)
            .map(|i| {
                let x = i as f32;
                [x, 50.0 + 40.0 * (x * 0.05).sin()]
            })
            .collect();
        outline.push([(n - 1) as f32, 100.0]);
        outline.push([0.0, 100.0]);
        let c = coverage(&tessellate_polygon(&outline), &outline);
        assert!((c - 1.0).abs() < 1e-3, "covered {c}");
    }

    #[test]
    fn a_fill_fringes_its_boundary_outward() {
        // A square with an even-odd hole: every boundary vertex's fringe points
        // away from the filled ring (out of the square, or into the hole), and
        // each has an outer-row copy.
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let hole = vec![[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        let mesh = tessellate_fringed(&[outer, hole], FillRule::EvenOdd);
        let inner: Vec<usize> = (0..mesh.positions.len())
            .filter(|&i| mesh.aa[i][2] < -0.5)
            .collect();
        let outer_rows = mesh.aa.iter().filter(|a| a[2] > 0.5).count();
        assert_eq!(inner.len(), 8);
        assert_eq!(outer_rows, 8);
        for i in inner {
            let p = mesh.positions[i];
            let d = [mesh.aa[i][0], mesh.aa[i][1]];
            let from_centre = [p[0] - 5.0, p[1] - 5.0];
            let in_hole = p[0] > 1.0 && p[0] < 9.0;
            let dot = d[0] * from_centre[0] + d[1] * from_centre[1];
            // Out of the square at its corners, into the hole at the hole's.
            assert!(if in_hole { dot < 0.0 } else { dot > 0.0 }, "{p:?} {d:?}");
            // A right-angle corner's mitre is sqrt(2) long.
            let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
            assert!((len - std::f32::consts::SQRT_2).abs() < 1e-3, "{len}");
        }
    }

    #[test]
    fn degenerate_outlines_are_empty() {
        assert!(
            tessellate_polygon(&[[0.0, 0.0], [1.0, 1.0]])
                .indices
                .is_empty()
        );
        assert!(
            tessellate_polygon(&[[0.0, 0.0], [f32::NAN, 1.0], [1.0, 0.0]])
                .indices
                .is_empty()
        );
    }
}
