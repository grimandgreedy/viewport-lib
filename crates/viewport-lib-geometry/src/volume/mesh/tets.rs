//! Tetrahedral decomposition of mixed cells, for transparent volume rendering.

use super::VolumeMeshData;
use super::cells::{CellType, cell_type};

/// Hex-to-tet decomposition using the Freudenthal 6-tet split.
///
/// All 6 tets share the main diagonal (vertex 0 <-> vertex 6 in VTK hex ordering).
pub(super) const HEX_TO_TETS: [[usize; 4]; 6] = [
    [0, 1, 5, 6],
    [0, 1, 2, 6],
    [0, 4, 5, 6],
    [0, 4, 7, 6],
    [0, 3, 2, 6],
    [0, 3, 7, 6],
];

/// Wedge-to-tet decomposition (3 tets from a triangular prism).
///
/// Vertices: 0,1,2 = bottom triangle; 3,4,5 = top triangle (3 above 0, etc.).
pub(super) const WEDGE_TO_TETS: [[usize; 4]; 3] = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]];

/// Pyramid-to-tet decomposition (2 tets from a square pyramid).
///
/// Vertices: 0-3 = base quad; 4 = apex.
pub(super) const PYRAMID_TO_TETS: [[usize; 4]; 2] = [[0, 1, 2, 4], [0, 2, 3, 4]];

/// Call `f` once per output tetrahedron across all cells in `data`.
///
/// `f` receives the four world-space vertices and the scalar value for that tet.
/// The scalar is taken from `data.cell_scalars[attribute]` at the parent cell index.
/// When the name is not a cell scalar but is in `data.node_scalars`, the tet takes
/// the mean of its four corner values. It is 0.0 when the attribute is absent or
/// an index is out of range.
///
/// Cell decomposition:
/// - Tet -> 1 tet
/// - Pyramid -> 2 tets
/// - Wedge -> 3 tets
/// - Hex -> 6 tets (Freudenthal split)
#[doc(hidden)]
pub fn for_each_tet<F>(data: &VolumeMeshData, attribute: &str, mut f: F)
where
    F: FnMut([[f32; 3]; 4], f32),
{
    let source = TetScalarSource::of(data, attribute);
    for (cell_idx, cell) in data.cells.iter().enumerate() {
        for local in cell_tets(cell) {
            let verts = [
                data.positions[cell[local[0]] as usize],
                data.positions[cell[local[1]] as usize],
                data.positions[cell[local[2]] as usize],
                data.positions[cell[local[3]] as usize],
            ];
            f(verts, source.value(cell_idx, cell, local));
        }
    }
}

/// Where the transparent mode's per-tet scalar comes from.
pub(super) enum TetScalarSource<'a> {
    Cell(&'a [f32]),
    Node(&'a [f32]),
    Absent,
}

impl<'a> TetScalarSource<'a> {
    fn of(data: &'a VolumeMeshData, attribute: &str) -> Self {
        if let Some(values) = data.cell_scalars.get(attribute) {
            Self::Cell(values)
        } else if let Some(values) = data.node_scalars.get(attribute) {
            Self::Node(values)
        } else {
            Self::Absent
        }
    }

    fn value(&self, cell_idx: usize, cell: &[u32; 8], tet: &[usize; 4]) -> f32 {
        match self {
            Self::Cell(values) => values.get(cell_idx).copied().unwrap_or(0.0),
            Self::Node(values) => {
                let sum: f32 = tet
                    .iter()
                    .map(|&slot| values.get(cell[slot] as usize).copied().unwrap_or(0.0))
                    .sum();
                sum / 4.0
            }
            Self::Absent => 0.0,
        }
    }
}

/// Per-output-tet scalar values, in the exact order [`for_each_tet`] emits tets.
///
/// The cheap counterpart of [`for_each_tet`] when only the scalars are needed
/// (a scalar-only projected-tet refresh): it walks the cells and repeats each
/// cell's scalar once per tet the cell decomposes into (tet -> 1, pyramid -> 2,
/// wedge -> 3, hex -> 6), skipping all vertex lookups. The result aligns
/// one-to-one with the tet geometry buffer, so it can be written straight into
/// the parallel scalar buffer. A node scalar gives each tet the mean of its
/// corners, as [`for_each_tet`] does.
#[doc(hidden)]
pub fn tet_scalars(data: &VolumeMeshData, attribute: &str) -> Vec<f32> {
    let source = TetScalarSource::of(data, attribute);
    let mut out = Vec::with_capacity(data.cells.len());
    for (cell_idx, cell) in data.cells.iter().enumerate() {
        for tet in cell_tets(cell) {
            out.push(source.value(cell_idx, cell, tet));
        }
    }
    out
}

/// Decompose all cells in `data` into tetrahedra and collect the results.
///
/// Returns `(positions, scalars)`:
/// - `positions`: flat list of `[[f32; 3]; 4]`, one entry per output tet (4 world-space vertices)
/// - `scalars`: one `f32` per output tet, taken from `data.cell_scalars[attribute]` at the
///   parent cell index (0.0 when the attribute is absent or the cell index is out of range)
///
/// Used in tests. Production upload paths use `for_each_tet` directly to avoid
/// materialising the full decomposed data before chunking.
#[cfg(test)]
pub(crate) fn decompose_to_tetrahedra(
    data: &VolumeMeshData,
    attribute: &str,
) -> (Vec<[[f32; 3]; 4]>, Vec<f32>) {
    let mut positions: Vec<[[f32; 3]; 4]> = Vec::new();
    let mut scalars: Vec<f32> = Vec::new();
    for_each_tet(data, attribute, |verts, scalar| {
        positions.push(verts);
        scalars.push(scalar);
    });
    (positions, scalars)
}

/// The tets a cell decomposes into, as local vertex slots.
pub(super) fn cell_tets(cell: &[u32; 8]) -> &'static [[usize; 4]] {
    match cell_type(cell) {
        CellType::Tet => &[[0, 1, 2, 3]],
        CellType::Pyramid => &PYRAMID_TO_TETS,
        CellType::Wedge => &WEDGE_TO_TETS,
        CellType::Hex => &HEX_TO_TETS,
    }
}
