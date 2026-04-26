//! CPU-side isoline (contour line) extraction from triangulated surfaces.
//!
//! An **isoline** is the set of points on a surface where a scalar field equals
//! a given value (the *isovalue*).  This module implements a per-triangle
//! edge-walk algorithm: for each triangle it checks which edges are crossed by
//! the isovalue, linearly interpolates the crossing position, and emits a
//! 2-point line segment.
//!
//! The extracted segments are returned in a form that can be passed directly to
//! the existing [`crate::renderer::PolylineItem`] / polyline pipeline : no new
//! GPU pipeline or shader is needed.
//!
//! # Usage
//!
//! ```ignore
//! let item = IsolineItem {
//!     positions: mesh_positions.clone(),
//!     indices:   mesh_indices.clone(),
//!     scalars:   per_vertex_pressure.clone(),
//!     isovalues: vec![100.0, 200.0, 300.0],
//!     color:     [1.0, 1.0, 0.0, 1.0],
//!     line_width: 1.5,
//!     ..IsolineItem::default()
//! };
//! let (positions, strip_lengths) = extract_isolines(&item);
//! // feed into PolylineItem …
//! ```

/// Describes a contour-line extraction request for one mesh / scalar field.
///
/// `IsolineItem` is `#[non_exhaustive]` so future fields can be added without
/// breaking existing callers that construct it via `Default::default()` +
/// field assignment.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct IsolineItem {
    /// Mesh vertex positions in local (object) space.  Must be the same length
    /// as `scalars`.
    pub positions: Vec<[f32; 3]>,

    /// Triangle indices into `positions`.  Length must be a multiple of 3.
    pub indices: Vec<u32>,

    /// Per-vertex scalar values for the field to contour.  Must be the same
    /// length as `positions`.
    pub scalars: Vec<f32>,

    /// Isovalues at which contour lines are extracted.  Each value produces an
    /// independent set of line segments.
    pub isovalues: Vec<f32>,

    /// RGBA colour applied to every line segment.  Defaults to opaque white.
    pub color: [f32; 4],

    /// Line width in pixels.  Defaults to `1.0`.
    pub line_width: f32,

    /// Model matrix that transforms local-space positions to world space.
    /// Defaults to [`glam::Mat4::IDENTITY`].
    pub model_matrix: glam::Mat4,

    /// Small offset along the surface face normal applied to extracted vertices
    /// to prevent z-fighting with the underlying mesh.  Defaults to `0.001`.
    pub depth_bias: f32,
}

impl Default for IsolineItem {
    fn default() -> Self {
        Self {
            positions: Vec::new(),
            indices: Vec::new(),
            scalars: Vec::new(),
            isovalues: Vec::new(),
            color: [1.0, 1.0, 1.0, 1.0],
            line_width: 1.0,
            model_matrix: glam::Mat4::IDENTITY,
            depth_bias: 0.001,
        }
    }
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// Extract isoline segments from a triangulated surface for all isovalues
/// specified in `item`.
///
/// Returns `(positions, strip_lengths)` suitable for constructing a
/// [`crate::renderer::PolylineItem`]:
/// - `positions` : world-space positions of every line-segment endpoint,
///   concatenated.
/// - `strip_lengths` : number of vertices per strip.  Every segment is
///   emitted as an independent 2-vertex strip, so each entry is `2`.
///
/// If `item` has empty `positions`, `indices`, or `scalars` the function
/// returns `(vec![], vec![])` immediately.
pub fn extract_isolines(item: &IsolineItem) -> (Vec<[f32; 3]>, Vec<u32>) {
    // Fast-exit guards.
    if item.positions.is_empty()
        || item.indices.is_empty()
        || item.scalars.is_empty()
        || item.isovalues.is_empty()
    {
        return (Vec::new(), Vec::new());
    }

    // Validate consistency : mismatched arrays are a caller bug; skip silently.
    if item.scalars.len() != item.positions.len() {
        return (Vec::new(), Vec::new());
    }

    // Pre-allocate conservatively.
    let tri_count = item.indices.len() / 3;
    let iso_count = item.isovalues.len();
    let mut out_positions: Vec<[f32; 3]> = Vec::with_capacity(tri_count * iso_count * 2);
    let mut strip_lengths: Vec<u32> = Vec::with_capacity(tri_count * iso_count);

    for &iso in &item.isovalues {
        extract_for_isovalue(item, iso, &mut out_positions, &mut strip_lengths);
    }

    (out_positions, strip_lengths)
}

// ---------------------------------------------------------------------------
// Per-isovalue inner loop
// ---------------------------------------------------------------------------

/// Edge-walk for a single isovalue.  Appends results to the caller's output
/// vecs to avoid per-isovalue allocations.
fn extract_for_isovalue(
    item: &IsolineItem,
    iso: f32,
    out_positions: &mut Vec<[f32; 3]>,
    strip_lengths: &mut Vec<u32>,
) {
    let positions = &item.positions;
    let scalars = &item.scalars;
    let model = item.model_matrix;
    let bias = item.depth_bias;

    let tri_count = item.indices.len() / 3;

    for tri_idx in 0..tri_count {
        let i0 = item.indices[tri_idx * 3] as usize;
        let i1 = item.indices[tri_idx * 3 + 1] as usize;
        let i2 = item.indices[tri_idx * 3 + 2] as usize;

        // Bounds check : skip malformed index data.
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }

        let p0 = glam::Vec3::from(positions[i0]);
        let p1 = glam::Vec3::from(positions[i1]);
        let p2 = glam::Vec3::from(positions[i2]);

        let s0 = scalars[i0];
        let s1 = scalars[i1];
        let s2 = scalars[i2];

        // Face normal for depth bias.  Skip degenerate (zero-area) triangles.
        let edge_a = p1 - p0;
        let edge_b = p2 - p0;
        let cross = edge_a.cross(edge_b);
        let len_sq = cross.length_squared();
        if len_sq < f32::EPSILON {
            continue; // degenerate triangle
        }
        let face_normal = cross / len_sq.sqrt();

        // Collect crossing points on the 3 edges.
        let edges = [(p0, p1, s0, s1), (p1, p2, s1, s2), (p2, p0, s2, s0)];
        let mut crossings: [Option<glam::Vec3>; 3] = [None; 3];

        for (edge_slot, &(ea, eb, sa, sb)) in edges.iter().enumerate() {
            if let Some(p) = edge_crossing(ea, eb, sa, sb, iso) {
                crossings[edge_slot] = Some(p);
            }
        }

        // Collect the valid crossing points.
        let pts: Vec<glam::Vec3> = crossings.iter().filter_map(|&c| c).collect();

        // We need exactly 2 points to form a line segment.
        if pts.len() < 2 {
            continue;
        }

        // If more than 2 (rare due to vertex exactly on isovalue), take first 2.
        let a = pts[0];
        let b = pts[1];

        // Apply depth bias along the face normal (local space), then transform
        // to world space using the model matrix.
        let bias_vec = face_normal * bias;
        let wa = transform_point(model, a + bias_vec);
        let wb = transform_point(model, b + bias_vec);

        out_positions.push(wa);
        out_positions.push(wb);
        strip_lengths.push(2);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the interpolated position where the isovalue `iso` crosses the
/// directed edge from `pa` (scalar `sa`) to `pb` (scalar `sb`), or `None`
/// if the edge does not cross `iso`.
///
/// The edge is considered crossing when one endpoint is strictly below `iso`
/// and the other is at or above `iso`.  If both endpoints are equal the edge
/// is skipped.
#[inline]
fn edge_crossing(pa: glam::Vec3, pb: glam::Vec3, sa: f32, sb: f32, iso: f32) -> Option<glam::Vec3> {
    // Skip if scalars are equal : no direction.
    if (sb - sa).abs() < f32::EPSILON {
        return None;
    }

    let crosses = (sa < iso && sb >= iso) || (sb < iso && sa >= iso);
    if !crosses {
        return None;
    }

    let t = (iso - sa) / (sb - sa);
    Some(pa + t * (pb - pa))
}

/// Apply a 4×4 model matrix to a 3-D point (homogeneous w=1, divide by w).
#[inline]
fn transform_point(m: glam::Mat4, p: glam::Vec3) -> [f32; 3] {
    m.transform_point3(p).to_array()
}
