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

/// A filled closed polyline: positions, a triangle index list, and for each
/// position the outline point it came from, as `(from, to, t)` indices into the
/// input: `points[from]` moved toward `points[to]` by `t`. A position that is an
/// input point has `from == to`.
#[derive(Debug, Clone, Default)]
pub(crate) struct PolygonMesh {
    pub positions: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub sources: Vec<(usize, usize, f32)>,
}

/// Tessellate the closed outline through `points` under the non-zero fill rule.
///
/// Takes any outline: one that touches itself, doubles back along a spur or
/// crosses itself fills like the same path in SVG. Positions do not map one to
/// one to `points` (coincident points merge and crossings add vertices), so
/// anything carried per input point is looked up through `sources`. Empty when
/// there are fewer than three points, a point is not finite, or tessellation
/// fails.
pub(crate) fn tessellate_polygon(points: &[[f32; 2]]) -> PolygonMesh {
    if points.len() < 3 || points.iter().flatten().any(|v| !v.is_finite()) {
        return PolygonMesh::default();
    }
    let mut builder = Path::builder();
    // Endpoint ids are assigned by the builder, so keep the map back to the
    // input index rather than assuming they count from zero.
    let mut input_of = Vec::with_capacity(points.len());
    let mut push = |id: lyon_path::EndpointId, i: usize| {
        let slot = id.to_usize();
        if input_of.len() <= slot {
            input_of.resize(slot + 1, 0);
        }
        input_of[slot] = i;
    };
    push(builder.begin(point(points[0][0], points[0][1])), 0);
    for (i, p) in points.iter().enumerate().skip(1) {
        push(builder.line_to(point(p[0], p[1])), i);
    }
    builder.end(true);
    let path = builder.build();

    let mut buffers: VertexBuffers<([f32; 2], (usize, usize, f32)), u32> = VertexBuffers::new();
    // `tessellate_with_ids` rather than `tessellate_path`: only it reports which
    // endpoints each output vertex came from.
    let input = |id: lyon_path::EndpointId| input_of.get(id.to_usize()).copied().unwrap_or(0);
    let result = FillTessellator::new().tessellate_with_ids(
        path.id_iter(),
        &path,
        None,
        &FillOptions::default().with_fill_rule(LyonFillRule::NonZero),
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
        tracing::debug!(
            ?e,
            points = points.len(),
            "polygon fill tessellation failed"
        );
        return PolygonMesh::default();
    }
    let (positions, sources) = buffers.vertices.into_iter().unzip();
    PolygonMesh {
        positions,
        indices: buffers.indices,
        sources,
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

    /// Fraction of the outline's area the mesh's triangles cover.
    fn coverage(mesh: &PolygonMesh, outline: &[[f32; 2]]) -> f32 {
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
