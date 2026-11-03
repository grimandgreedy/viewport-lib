//! Cell topology: face and edge tables per cell type, and the per-cell helpers
//! the extractors share.

use super::CELL_SENTINEL;

// ---------------------------------------------------------------------------
// Tet face table
// ---------------------------------------------------------------------------
//
// One face per vertex of the tet (face is opposite that vertex).
// The winding listed here may be inward or outward depending on the tet's
// signed volume; the geometric winding-correction step in
// `extract_boundary_faces` normalises every boundary face to outward after
// extraction, so the exact winding here does not matter for correctness.
// We just need a consistent convention so the sorted-key boundary detection
// works (both cells that share an interior face must produce the same key).

/// Canonical tetrahedron triangular-face table (one triangle opposite each
/// vertex). This is the single home for the tet face winding; `viewport-lib`'s
/// CPU picker references it rather than keeping its own copy.
#[doc(hidden)]
pub const TET_FACES: [[usize; 3]; 4] = [
    [1, 2, 3], // opposite v0
    [0, 3, 2], // opposite v1
    [0, 1, 3], // opposite v2
    [0, 2, 1], // opposite v3
];

// ---------------------------------------------------------------------------
// Hex face table
// ---------------------------------------------------------------------------
//
// VTK hex vertex numbering used throughout this module:
//
//     7 --- 6          top face
//    /|    /|
//   4 --- 5 |
//   | 3 --| 2          bottom face
//   |/    |/
//   0 --- 1
//
// Six quad faces.  Verified to produce outward normals (from-cell CCW):
//
//   bottom (-Y): [0,1,2,3]  : normal = (1,0,0)x(1,0,1) = (0,-1,0) ok
//   top    (+Y): [4,7,6,5]  : normal = (0,0,1)x(1,0,1) = (0,+1,0) ok
//   front  (-Z): [0,4,5,1]  : normal = (0,1,0)x(1,1,0) = (0,0,-1) ok
//   back   (+Z): [2,6,7,3]  : normal = (0,1,0)x(-1,1,0)= (0,0,+1) ok
//   left   (-X): [0,3,7,4]  : normal = (0,0,1)x(0,1,1) = (-1,0,0) ok
//   right  (+X): [1,5,6,2]  : normal = (0,1,0)x(0,1,1) = (+1,0,0) ok
//
// The geometric winding-correction step acts as a safety net in case any
// cell is degenerate or oriented unexpectedly.

pub(super) const HEX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3], // bottom (-Y)
    [4, 7, 6, 5], // top    (+Y)
    [0, 4, 5, 1], // front  (-Z)
    [2, 6, 7, 3], // back   (+Z)
    [0, 3, 7, 4], // left   (-X)
    [1, 5, 6, 2], // right  (+X)
];

// ---------------------------------------------------------------------------
// Pyramid face tables
// ---------------------------------------------------------------------------
//
// VTK pyramid vertex numbering:
//
//        4 (apex)
//       /|\
//      / | \
//     /  |  \
//    3---+---2
//    |       |
//    0-------1
//
// One quad base face and four triangular side faces.
// Winding correction in the extractor normalises outward direction.

/// Quad base face of a pyramid (vertices 0-3).
pub(super) const PYRAMID_QUAD_FACE: [[usize; 4]; 1] = [
    [0, 1, 2, 3], // base
];

/// Triangular side faces of a pyramid (apex = vertex 4).
pub(super) const PYRAMID_TRI_FACES: [[usize; 3]; 4] = [
    [0, 4, 1], // front
    [1, 4, 2], // right
    [2, 4, 3], // back
    [3, 4, 0], // left
];

/// Edges of a pyramid: 4 base + 4 lateral.
pub(super) const PYRAMID_EDGES: [[usize; 2]; 8] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0], // base ring
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4], // lateral
];

// ---------------------------------------------------------------------------
// Wedge (triangular prism) face tables
// ---------------------------------------------------------------------------
//
// VTK wedge vertex numbering: 0,1,2 = bottom tri, 3,4,5 = top tri
// (vertex 3 is directly above vertex 0, etc.)
//
//   3 --- 5
//   |  \  |
//   |   4 |
//   |     |
//   0 --- 2
//    \   /
//      1
//
// Two triangular end faces and three quad lateral faces.

/// Triangular end faces of a wedge.
pub(super) const WEDGE_TRI_FACES: [[usize; 3]; 2] = [
    [0, 2, 1], // bottom (outward = downward)
    [3, 4, 5], // top    (outward = upward)
];

/// Quad lateral faces of a wedge.
pub(super) const WEDGE_QUAD_FACES: [[usize; 4]; 3] = [
    [0, 1, 4, 3], // side 0
    [1, 2, 5, 4], // side 1
    [2, 0, 3, 5], // side 2
];

/// Edges of a wedge: 3 bottom + 3 top + 3 vertical.
pub(super) const WEDGE_EDGES: [[usize; 2]; 9] = [
    [0, 1],
    [1, 2],
    [2, 0], // bottom tri
    [3, 4],
    [4, 5],
    [5, 3], // top tri
    [0, 3],
    [1, 4],
    [2, 5], // vertical
];

/// Cell edges for tets: all 6 pairs from 4 vertices.
pub(super) const TET_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// Cell edges for hexes (VTK ordering).
///
/// ```text
///     7 --- 6
///    /|    /|
///   4 --- 5 |
///   | 3 --| 2
///   |/    |/
///   0 --- 1
/// ```
pub(super) const HEX_EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0], // bottom ring
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4], // top ring
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7], // vertical
];

/// Internal cell type, detected from sentinel slots.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CellType {
    Tet,
    Pyramid,
    Wedge,
    Hex,
}

impl CellType {
    pub(super) fn vertex_count(self) -> usize {
        match self {
            CellType::Tet => 4,
            CellType::Pyramid => 5,
            CellType::Wedge => 6,
            CellType::Hex => 8,
        }
    }

    pub(super) fn edges(self) -> &'static [[usize; 2]] {
        match self {
            CellType::Tet => &TET_EDGES,
            CellType::Pyramid => &PYRAMID_EDGES,
            CellType::Wedge => &WEDGE_EDGES,
            CellType::Hex => &HEX_EDGES,
        }
    }
}

/// Detect cell type from sentinel pattern in the 8-slot cell array.
#[inline]
pub(super) fn cell_type(cell: &[u32; 8]) -> CellType {
    if cell[4] == CELL_SENTINEL {
        CellType::Tet
    } else if cell[5] == CELL_SENTINEL {
        CellType::Pyramid
    } else if cell[6] == CELL_SENTINEL {
        CellType::Wedge
    } else {
        CellType::Hex
    }
}

/// Interior reference point for winding-correcting one boundary face, computed
/// from the owning cell (deferred from face generation so it is only paid for
/// boundary faces). For a tet this is the opposite vertex -- the vertex not on
/// the face -- which is a larger, more numerically robust reference than the
/// cell centroid for sliver tets; for the other cell types the cell centroid is
/// used (any strictly-interior point gives the correct outward sign).
#[inline]
pub(super) fn boundary_interior_ref(
    cell: &[u32; 8],
    tri: &[u32; 3],
    positions: &[[f32; 3]],
) -> [f32; 3] {
    let ct = cell_type(cell);
    if matches!(ct, CellType::Tet) {
        for k in 0..4 {
            let v = cell[k];
            if v != tri[0] && v != tri[1] && v != tri[2] {
                return positions[v as usize];
            }
        }
    }
    cell_centroid(cell, ct.vertex_count(), positions)
}

/// Centroid of the first `nv` vertices of `cell`.
pub(super) fn cell_centroid(cell: &[u32; 8], nv: usize, positions: &[[f32; 3]]) -> [f32; 3] {
    let mut c = [0.0f32; 3];
    for i in 0..nv {
        let p = positions[cell[i] as usize];
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    let n = nv as f32;
    [c[0] / n, c[1] / n, c[2] / n]
}
