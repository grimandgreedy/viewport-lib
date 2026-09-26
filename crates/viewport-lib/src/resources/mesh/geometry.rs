//! Geometry builders for glyph base meshes and primitive shapes.

use crate::resources::types::*;

/// The unique edges of a triangle index buffer, as a line-list index buffer.
///
/// Each edge appears once, with the smaller vertex index first. An item type
/// drawing its own wireframe from a triangle mesh builds its line indices with
/// this so its edges match the ones the renderer's own wireframes draw.
pub fn generate_edge_indices(triangle_indices: &[u32]) -> Vec<u32> {
    // Canonical form: smaller index first, packed into a u64 key so
    // (a,b) and (b,a) collapse to the same edge under sort+dedup. Sorting
    // is several times faster than hashing every edge, and this runs for
    // every uploaded mesh. The output is sorted rather than first-seen
    // order, which line rendering does not care about.
    let mut keys: Vec<u64> = Vec::with_capacity(triangle_indices.len());
    for tri in triangle_indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            keys.push((u64::from(lo) << 32) | u64::from(hi));
        }
    }
    keys.sort_unstable();
    keys.dedup();
    let mut result = Vec::with_capacity(keys.len() * 2);
    for k in keys {
        result.push((k >> 32) as u32);
        result.push(k as u32);
    }
    result
}

// ---------------------------------------------------------------------------
// Procedural unit cube mesh (24 vertices, 4 per face, 36 indices)
// ---------------------------------------------------------------------------

/// Generate a unit cube centered at the origin.
///
/// 24 vertices (4 per face with shared normals), 36 indices (2 triangles per face).
/// All vertices are white [1,1,1,1].
pub(crate) fn build_unit_cube() -> (Vec<Vertex>, Vec<u32>) {
    let white = [1.0f32, 1.0, 1.0, 1.0];
    let mut verts: Vec<Vertex> = Vec::with_capacity(24);
    let mut idx: Vec<u32> = Vec::with_capacity(36);

    // Helper: add a face quad (4 vertices in CCW order) and its 2 triangles.
    let mut add_face = |positions: [[f32; 3]; 4], normal: [f32; 3]| {
        let base = verts.len() as u32;
        for pos in &positions {
            verts.push(Vertex {
                position: *pos,
                normal,
                colour: white,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
        // Two triangles: (base, base+1, base+2) and (base, base+2, base+3)
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // +X face (right), normal [1, 0, 0]
    add_face(
        [
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
        ],
        [1.0, 0.0, 0.0],
    );

    // -X face (left), normal [-1, 0, 0]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5],
        ],
        [-1.0, 0.0, 0.0],
    );

    // +Y face (top), normal [0, 1, 0]
    add_face(
        [
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
        ],
        [0.0, 1.0, 0.0],
    );

    // -Y face (bottom), normal [0, -1, 0]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
        ],
        [0.0, -1.0, 0.0],
    );

    // +Z face (front), normal [0, 0, 1]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        [0.0, 0.0, 1.0],
    );

    // -Z face (back), normal [0, 0, -1]
    add_face(
        [
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ],
        [0.0, 0.0, -1.0],
    );

    (verts, idx)
}

// ---------------------------------------------------------------------------
// Procedural glyph arrow mesh (cone tip + cylinder shaft, local +Z axis)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Procedural icosphere (2 subdivisions, ~240 triangles)
// ---------------------------------------------------------------------------
