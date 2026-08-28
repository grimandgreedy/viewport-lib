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
}
