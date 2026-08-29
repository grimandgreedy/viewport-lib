//! Geometry for [`OverlayShape::Vector`](crate::renderer::types::OverlayShape):
//! an arbitrary vector shape built from subpaths of line and Bezier segments,
//! combined by a fill rule. This is the authoring representation; the renderer
//! flattens and tessellates it to triangles when it draws the shape.

/// One segment of a [`SubPath`]. The start point is implicit: it is the
/// subpath's `start` for the first segment, and the previous segment's end
/// point after that. Coordinates are path-local logical pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PathSegment {
    /// Straight line to `to`.
    Line {
        /// End point.
        to: [f32; 2],
    },
    /// Quadratic Bezier through control point `ctrl` to `to`.
    Quad {
        /// Control point.
        ctrl: [f32; 2],
        /// End point.
        to: [f32; 2],
    },
    /// Cubic Bezier through control points `ctrl1`, `ctrl2` to `to`.
    Cubic {
        /// First control point.
        ctrl1: [f32; 2],
        /// Second control point.
        ctrl2: [f32; 2],
        /// End point.
        to: [f32; 2],
    },
}

/// A single contour: a start point, a run of segments, and whether it closes
/// back to the start. Multiple subpaths in one shape combine under the shape's
/// [`FillRule`], so the inner loop of a letter "O" becomes a hole.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubPath {
    /// Start point in path-local logical pixels.
    pub start: [f32; 2],
    /// Segments in order from `start`.
    pub segments: Vec<PathSegment>,
    /// When `true`, a closing segment runs from the last point back to
    /// `start`. Fills treat every subpath as closed regardless; the flag
    /// matters for stroking the outline.
    pub closed: bool,
}

impl SubPath {
    /// Start a new subpath at `start`.
    pub fn new(start: [f32; 2]) -> Self {
        Self {
            start,
            segments: Vec::new(),
            closed: false,
        }
    }

    /// Append a straight line to `to`.
    pub fn line_to(mut self, to: [f32; 2]) -> Self {
        self.segments.push(PathSegment::Line { to });
        self
    }

    /// Append a quadratic Bezier through `ctrl` to `to`.
    pub fn quad_to(mut self, ctrl: [f32; 2], to: [f32; 2]) -> Self {
        self.segments.push(PathSegment::Quad { ctrl, to });
        self
    }

    /// Append a cubic Bezier through `ctrl1`, `ctrl2` to `to`.
    pub fn cubic_to(mut self, ctrl1: [f32; 2], ctrl2: [f32; 2], to: [f32; 2]) -> Self {
        self.segments.push(PathSegment::Cubic { ctrl1, ctrl2, to });
        self
    }

    /// Mark the subpath closed (the last point connects back to `start`).
    pub fn close(mut self) -> Self {
        self.closed = true;
        self
    }

    /// Build a closed polygon from a point list. The first point is the start;
    /// the rest become line segments; the subpath is closed.
    pub fn polygon(points: &[[f32; 2]]) -> Self {
        let mut sp = match points.split_first() {
            Some((first, _)) => SubPath::new(*first),
            None => SubPath::new([0.0, 0.0]),
        };
        for p in points.iter().skip(1) {
            sp = sp.line_to(*p);
        }
        sp.close()
    }
}

/// How overlapping and nested subpaths combine into filled area. Matches the
/// SVG fill rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FillRule {
    /// A point is inside when the signed crossing count (winding number) is
    /// non-zero. The SVG default.
    #[default]
    NonZero,
    /// A point is inside when the crossing count is odd.
    EvenOdd,
}

/// Number of straight segments a curve is subdivided into for the CPU
/// point-in-path test. This is only used for hit-testing
/// ([`OverlayShapeItem::contains`](crate::renderer::types::OverlayShapeItem::contains));
/// the GPU fill uses a tolerance-driven tessellator instead.
const CONTAINS_CURVE_STEPS: usize = 24;

fn quad_point(p0: [f32; 2], c: [f32; 2], p1: [f32; 2], t: f32) -> [f32; 2] {
    let u = 1.0 - t;
    let a = u * u;
    let b = 2.0 * u * t;
    let d = t * t;
    [
        a * p0[0] + b * c[0] + d * p1[0],
        a * p0[1] + b * c[1] + d * p1[1],
    ]
}

fn cubic_point(p0: [f32; 2], c1: [f32; 2], c2: [f32; 2], p1: [f32; 2], t: f32) -> [f32; 2] {
    let u = 1.0 - t;
    let a = u * u * u;
    let b = 3.0 * u * u * t;
    let c = 3.0 * u * t * t;
    let d = t * t * t;
    [
        a * p0[0] + b * c1[0] + c * c2[0] + d * p1[0],
        a * p0[1] + b * c1[1] + c * c2[1] + d * p1[1],
    ]
}

/// Flatten a subpath to a polyline of points, subdividing curves at a fixed
/// step count. Adequate for hit-testing; not the GPU tessellation path.
fn flatten_subpath(sp: &SubPath) -> Vec<[f32; 2]> {
    let mut pts = vec![sp.start];
    let mut cur = sp.start;
    for seg in &sp.segments {
        match *seg {
            PathSegment::Line { to } => {
                pts.push(to);
                cur = to;
            }
            PathSegment::Quad { ctrl, to } => {
                for i in 1..=CONTAINS_CURVE_STEPS {
                    let t = i as f32 / CONTAINS_CURVE_STEPS as f32;
                    pts.push(quad_point(cur, ctrl, to, t));
                }
                cur = to;
            }
            PathSegment::Cubic { ctrl1, ctrl2, to } => {
                for i in 1..=CONTAINS_CURVE_STEPS {
                    let t = i as f32 / CONTAINS_CURVE_STEPS as f32;
                    pts.push(cubic_point(cur, ctrl1, ctrl2, to, t));
                }
                cur = to;
            }
        }
    }
    pts
}

/// Flatten every subpath to a polyline of points, subdividing curves at a
/// fixed step count. Used for stroking a vector shape's outline and for the
/// point-in-path test; the GPU fill uses a tolerance-driven tessellator.
pub(crate) fn flatten_contours(subpaths: &[SubPath]) -> Vec<Vec<[f32; 2]>> {
    subpaths.iter().map(flatten_subpath).collect()
}

/// Point-in-path test for a set of subpaths under a fill rule. `pt` is in the
/// same path-local space as the subpath coordinates. Curves are flattened at a
/// fixed resolution, so the boundary is approximate near tight curvature; this
/// backs `contains`, not the rendered fill.
pub(crate) fn path_contains(subpaths: &[SubPath], fill_rule: FillRule, pt: [f32; 2]) -> bool {
    let (px, py) = (pt[0], pt[1]);
    let mut crossings: u32 = 0;
    let mut winding: i32 = 0;

    for sp in subpaths {
        let poly = flatten_subpath(sp);
        if poly.len() < 2 {
            continue;
        }
        // Iterate every edge, including the closing edge back to the start.
        for i in 0..poly.len() {
            let a = poly[i];
            let b = poly[(i + 1) % poly.len()];
            let (y0, y1) = (a[1], b[1]);
            // Does a horizontal ray at py, going +x, cross this edge?
            if (y0 > py) != (y1 > py) {
                let t = (py - y0) / (y1 - y0);
                let x_cross = a[0] + t * (b[0] - a[0]);
                if x_cross > px {
                    crossings += 1;
                    winding += if y1 > y0 { 1 } else { -1 };
                }
            }
        }
    }

    match fill_rule {
        FillRule::EvenOdd => crossings % 2 == 1,
        FillRule::NonZero => winding != 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_contains_centre_excludes_outside() {
        let sq = SubPath::polygon(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        let paths = [sq];
        assert!(path_contains(&paths, FillRule::NonZero, [5.0, 5.0]));
        assert!(!path_contains(&paths, FillRule::NonZero, [15.0, 5.0]));
    }

    #[test]
    fn even_odd_hole_is_empty() {
        // Outer 0..10 with an inner 3..7 hole. Even-odd makes the inner region
        // a hole regardless of winding direction.
        let outer = SubPath::polygon(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        let inner = SubPath::polygon(&[[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]]);
        let paths = [outer, inner];
        // Centre of the hole is outside the filled area.
        assert!(!path_contains(&paths, FillRule::EvenOdd, [5.0, 5.0]));
        // A point in the ring between outer and inner is filled.
        assert!(path_contains(&paths, FillRule::EvenOdd, [1.0, 5.0]));
    }

    #[test]
    fn curved_subpath_contains_interior() {
        // A rough circle from four cubic quarters, centred at (10,10) r=8.
        let k = 8.0 * 0.5522847498;
        let sp = SubPath::new([18.0, 10.0])
            .cubic_to([18.0, 10.0 + k], [10.0 + k, 18.0], [10.0, 18.0])
            .cubic_to([10.0 - k, 18.0], [2.0, 10.0 + k], [2.0, 10.0])
            .cubic_to([2.0, 10.0 - k], [10.0 - k, 2.0], [10.0, 2.0])
            .cubic_to([10.0 + k, 2.0], [18.0, 10.0 - k], [18.0, 10.0])
            .close();
        let paths = [sp];
        assert!(path_contains(&paths, FillRule::NonZero, [10.0, 10.0]));
        assert!(!path_contains(&paths, FillRule::NonZero, [10.0, 25.0]));
    }
}
