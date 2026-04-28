//! Sparse voxel grid topology processing : boundary face extraction.
//!
//! Converts a sparse regular grid (a subset of occupied cells on an axis-aligned
//! lattice) into a standard [`MeshData`] by extracting boundary faces (quad faces
//! not shared between two active cells) and computing area-weighted vertex normals.
//!
//! Per-cell scalar and color attributes are remapped to per-face attributes so the
//! existing Phase 2 face-rendering path handles coloring without any new GPU
//! infrastructure.  Per-node scalars are averaged over the four quad corner nodes
//! to produce per-face scalars on the same path.
//!
//! # Node scalar indexing
//!
//! `node_scalars` is a dense `Vec<f32>` indexed as:
//!
//! ```text
//! index = nk * (W * H) + nj * W + ni
//! ```
//!
//! where `W = max_i + 2`, `H = max_j + 2`, and `max_i`/`max_j`/`max_k` are the
//! maximum cell indices found in `active_cells`.  Node indices range from 0 to
//! `max_cell_i + 1` on each axis, so the node grid is one cell larger than the
//! cell grid on every axis.  If the vec is shorter than `W * H * D` (where
//! `D = max_k + 2`), missing entries default to `0.0`.

use std::collections::{HashMap, HashSet};

use super::types::{AttributeData, MeshData};

// ---------------------------------------------------------------------------
// Face corner table
// ---------------------------------------------------------------------------
//
// Six axis-aligned cube faces.  Each entry lists the four corner offsets
// `[di, dj, dk]` from cell `[ci, cj, ck]` in counter-clockwise order when
// viewed from outside the cell (outward normal convention).
//
// Verified normal directions:
//   0  -X : [0,0,0],[0,0,1],[0,1,1],[0,1,0]  normal = cross([0,0,1],[0,1,1]) = [-1,0,0] ✓
//   1  +X : [1,0,0],[1,1,0],[1,1,1],[1,0,1]  normal = cross([0,1,0],[0,1,1]) = [+1,0,0] ✓
//   2  -Y : [0,0,0],[1,0,0],[1,0,1],[0,0,1]  normal = cross([1,0,0],[1,0,1]) = [0,-1,0] ✓
//   3  +Y : [0,1,0],[0,1,1],[1,1,1],[1,1,0]  normal = cross([0,0,1],[1,0,1]) = [0,+1,0] ✓
//   4  -Z : [0,0,0],[0,1,0],[1,1,0],[1,0,0]  normal = cross([0,1,0],[1,1,0]) = [0,0,-1] ✓
//   5  +Z : [0,0,1],[1,0,1],[1,1,1],[0,1,1]  normal = cross([1,0,0],[1,1,0]) = [0,0,+1] ✓

const FACE_CORNERS: [[[u32; 3]; 4]; 6] = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], // 0: -X
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]], // 1: +X
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], // 2: -Y
    [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]], // 3: +Y
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], // 4: -Z
    [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], // 5: +Z
];

// ---------------------------------------------------------------------------
// Public data type
// ---------------------------------------------------------------------------

/// Input data for a sparse regular voxel grid.
///
/// Only the cells listed in `active_cells` are considered occupied.  Boundary
/// faces (faces not shared between two active cells) are extracted and uploaded
/// as a standard surface mesh.
///
/// # Attribute conventions
///
/// - `cell_scalars` / `cell_colors`: one entry per element of `active_cells`
///   (parallel arrays).
/// - `node_scalars`: dense array indexed by
///   `nk * (W * H) + nj * W + ni` where `W = max_i + 2`, `H = max_j + 2`
///   and `max_i`, `max_j`, `max_k` are derived from `active_cells` (see module
///   docs).  Set via [`AttributeRef { kind: AttributeKind::Face, .. }`] after
///   upload — node scalars are averaged to face scalars during extraction.
///
/// # Upload
///
/// ```no_run
/// use viewport_lib::{SparseVolumeGridData, AttributeKind, AttributeRef};
/// use std::collections::HashMap;
///
/// let mut data = SparseVolumeGridData::default();
/// data.origin    = [-1.0, -1.0, -1.0];
/// data.cell_size = 1.0;
/// data.active_cells = vec![[0, 0, 0], [1, 0, 0]];
/// data.cell_scalars.insert("pressure".to_string(), vec![0.3, 0.8]);
/// // upload via resources.upload_sparse_volume_grid_data(device, &data)
/// ```
#[non_exhaustive]
#[derive(Default)]
pub struct SparseVolumeGridData {
    /// World-space position of the `[0, 0, 0]` corner of cell `[0, 0, 0]`.
    pub origin: [f32; 3],

    /// Side length of one cubic cell in world units.  Must be positive.
    pub cell_size: f32,

    /// Grid indices `[i, j, k]` of occupied cells.
    pub active_cells: Vec<[u32; 3]>,

    /// Named per-cell scalar attributes (one `f32` per active cell, parallel to
    /// `active_cells`).  Remapped to `AttributeKind::Face` during extraction.
    pub cell_scalars: HashMap<String, Vec<f32>>,

    /// Named per-node scalar attributes.  Dense array; see module-level
    /// indexing convention.  Averaged over 4 quad corners to produce
    /// `AttributeKind::Face` values.
    pub node_scalars: HashMap<String, Vec<f32>>,

    /// Named per-cell RGBA color attributes (one `[f32; 4]` per active cell,
    /// parallel to `active_cells`).  Remapped to `AttributeKind::FaceColor`
    /// during extraction.
    pub cell_colors: HashMap<String, Vec<[f32; 4]>>,
}

// ---------------------------------------------------------------------------
// Boundary extraction
// ---------------------------------------------------------------------------

/// Convert [`SparseVolumeGridData`] into a standard [`MeshData`] by extracting
/// the boundary surface and remapping per-cell / per-node attributes to
/// per-face attributes.
///
/// Returns an empty [`MeshData`] if `active_cells` is empty or `cell_size <=
/// 0.0`.  This causes [`upload_mesh_data`](super::ViewportGpuResources::upload_mesh_data)
/// to return a `ViewportError::EmptyMesh` — the upload layer handles it.
pub(crate) fn extract_sparse_boundary(data: &SparseVolumeGridData) -> MeshData {
    if data.active_cells.is_empty() || data.cell_size <= 0.0 {
        return MeshData::default();
    }

    let s = data.cell_size;
    let [ox, oy, oz] = data.origin;

    // Build the active-cell lookup set for O(1) neighbor queries.
    let active_set: HashSet<[u32; 3]> = data.active_cells.iter().copied().collect();

    // Compute the extent of the node grid (needed for node_scalars indexing).
    let (max_ci, max_cj, _max_ck) = data
        .active_cells
        .iter()
        .fold((0u32, 0u32, 0u32), |(mi, mj, mk), &[ci, cj, ck]| {
            (mi.max(ci), mj.max(cj), mk.max(ck))
        });
    // Node grid dimensions: one more than cell grid on each axis.
    let node_w = (max_ci + 2) as usize; // nodes along I axis
    let node_h = (max_cj + 2) as usize; // nodes along J axis

    // Deduplicate vertices: grid-node key -> vertex index in output mesh.
    let mut node_to_vi: HashMap<[u32; 3], u32> = HashMap::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();

    // Each boundary quad records: active_cell_index, 4 vertex indices, 4 node keys.
    struct BoundaryQuad {
        cell_idx: usize,
        vi: [u32; 4],
        node_keys: [[u32; 3]; 4],
    }
    let mut boundary_quads: Vec<BoundaryQuad> = Vec::new();

    for (cell_idx, &[ci, cj, ck]) in data.active_cells.iter().enumerate() {
        for face_dir in 0usize..6 {
            // Check whether the neighbor in this direction is active.
            let neighbor_active = match face_dir {
                0 => ci > 0 && active_set.contains(&[ci - 1, cj, ck]), // -X
                1 => active_set.contains(&[ci + 1, cj, ck]),           // +X
                2 => cj > 0 && active_set.contains(&[ci, cj - 1, ck]), // -Y
                3 => active_set.contains(&[ci, cj + 1, ck]),           // +Y
                4 => ck > 0 && active_set.contains(&[ci, cj, ck - 1]), // -Z
                5 => active_set.contains(&[ci, cj, ck + 1]),           // +Z
                _ => unreachable!(),
            };

            if neighbor_active {
                continue; // interior face — skip
            }

            // Boundary face: resolve the 4 node keys and deduplicate vertices.
            let corners = &FACE_CORNERS[face_dir];
            let mut vi = [0u32; 4];
            let mut node_keys = [[0u32; 3]; 4];
            for (k, &[di, dj, dk]) in corners.iter().enumerate() {
                let nk: [u32; 3] = [ci + di, cj + dj, ck + dk];
                node_keys[k] = nk;
                let next_idx = positions.len() as u32;
                let vi_k = *node_to_vi.entry(nk).or_insert_with(|| {
                    positions.push([
                        ox + nk[0] as f32 * s,
                        oy + nk[1] as f32 * s,
                        oz + nk[2] as f32 * s,
                    ]);
                    next_idx
                });
                vi[k] = vi_k;
            }

            boundary_quads.push(BoundaryQuad {
                cell_idx,
                vi,
                node_keys,
            });
        }
    }

    let n_verts = positions.len();
    let n_quads = boundary_quads.len();

    // Build index buffer (2 triangles per quad: [A,B,C] and [A,C,D]).
    let mut indices: Vec<u32> = Vec::with_capacity(n_quads * 6);
    // Normal accumulation in f64 for precision.
    let mut normal_accum: Vec<[f64; 3]> = vec![[0.0; 3]; n_verts];

    for quad in &boundary_quads {
        let [va, vb, vc, vd] = quad.vi;

        // Triangle 0: A, B, C
        indices.push(va);
        indices.push(vb);
        indices.push(vc);
        // Triangle 1: A, C, D
        indices.push(va);
        indices.push(vc);
        indices.push(vd);

        // Accumulate area-weighted face normal to each vertex in both triangles.
        for &[v0, v1, v2] in &[[va, vb, vc], [va, vc, vd]] {
            let pa = positions[v0 as usize];
            let pb = positions[v1 as usize];
            let pc = positions[v2 as usize];
            let ab = [
                (pb[0] - pa[0]) as f64,
                (pb[1] - pa[1]) as f64,
                (pb[2] - pa[2]) as f64,
            ];
            let ac = [
                (pc[0] - pa[0]) as f64,
                (pc[1] - pa[1]) as f64,
                (pc[2] - pa[2]) as f64,
            ];
            let n = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            for &vi in &[v0, v1, v2] {
                let acc = &mut normal_accum[vi as usize];
                acc[0] += n[0];
                acc[1] += n[1];
                acc[2] += n[2];
            }
        }
    }

    // Normalize accumulated normals.
    let normals: Vec<[f32; 3]> = normal_accum
        .iter()
        .map(|n| {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-12 {
                [
                    (n[0] / len) as f32,
                    (n[1] / len) as f32,
                    (n[2] / len) as f32,
                ]
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect();

    // ---------------------------------------------------------------------------
    // Attribute remapping
    // ---------------------------------------------------------------------------

    let mut attributes: HashMap<String, AttributeData> = HashMap::new();

    // Cell scalars: one value per quad -> one per triangle (two per quad).
    for (name, cell_vals) in &data.cell_scalars {
        let face_scalars: Vec<f32> = boundary_quads
            .iter()
            .flat_map(|q| {
                let v = cell_vals.get(q.cell_idx).copied().unwrap_or(0.0);
                [v, v] // two triangles per quad
            })
            .collect();
        attributes.insert(name.clone(), AttributeData::Face(face_scalars));
    }

    // Cell colors: one RGBA per quad -> one per triangle (two per quad).
    for (name, cell_vals) in &data.cell_colors {
        let face_colors: Vec<[f32; 4]> = boundary_quads
            .iter()
            .flat_map(|q| {
                let c = cell_vals.get(q.cell_idx).copied().unwrap_or([1.0; 4]);
                [c, c] // two triangles per quad
            })
            .collect();
        attributes.insert(name.clone(), AttributeData::FaceColor(face_colors));
    }

    // Node scalars: average the 4 quad corner values -> one per triangle.
    for (name, node_vals) in &data.node_scalars {
        let face_scalars: Vec<f32> = boundary_quads
            .iter()
            .flat_map(|q| {
                let avg = {
                    let mut sum = 0.0f32;
                    for &[ni, nj, nk] in &q.node_keys {
                        let idx =
                            nk as usize * node_w * node_h + nj as usize * node_w + ni as usize;
                        sum += node_vals.get(idx).copied().unwrap_or(0.0);
                    }
                    sum / 4.0
                };
                [avg, avg] // two triangles per quad
            })
            .collect();
        attributes.insert(name.clone(), AttributeData::Face(face_scalars));
    }

    MeshData {
        positions,
        normals,
        indices,
        uvs: None,
        tangents: None,
        attributes,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn single_cell() -> SparseVolumeGridData {
        SparseVolumeGridData {
            origin: [0.0; 3],
            cell_size: 1.0,
            active_cells: vec![[0, 0, 0]],
            ..Default::default()
        }
    }

    fn two_adjacent_cells() -> SparseVolumeGridData {
        SparseVolumeGridData {
            origin: [0.0; 3],
            cell_size: 1.0,
            active_cells: vec![[0, 0, 0], [1, 0, 0]],
            ..Default::default()
        }
    }

    fn three_cells_in_a_line() -> SparseVolumeGridData {
        SparseVolumeGridData {
            origin: [0.0; 3],
            cell_size: 1.0,
            active_cells: vec![[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            ..Default::default()
        }
    }

    #[test]
    fn single_active_cell_has_twelve_boundary_triangles() {
        let data = single_cell();
        let mesh = extract_sparse_boundary(&data);
        // 6 faces × 2 triangles = 12 triangles → 36 indices
        assert_eq!(
            mesh.indices.len(),
            36,
            "single cell -> 12 boundary triangles (36 indices)"
        );
    }

    #[test]
    fn two_adjacent_cells_share_one_face() {
        let data = two_adjacent_cells();
        let mesh = extract_sparse_boundary(&data);
        // 2 × 6 quads = 12 quads, minus 2 shared quads (one from each cell) = 10 quads
        // 10 quads × 2 triangles = 20 triangles → 60 indices
        assert_eq!(
            mesh.indices.len(),
            60,
            "two adjacent cells -> 20 boundary triangles (60 indices)"
        );
    }

    #[test]
    fn three_cells_in_a_line_share_two_faces() {
        let data = three_cells_in_a_line();
        let mesh = extract_sparse_boundary(&data);
        // 3 × 6 = 18 quads minus 4 shared (2 interior boundaries × 2 cells each) = 14 quads
        // 14 × 2 = 28 triangles → 84 indices
        assert_eq!(
            mesh.indices.len(),
            84,
            "three cells in a line -> 28 boundary triangles (84 indices)"
        );
    }

    #[test]
    fn positions_are_correct() {
        let data = SparseVolumeGridData {
            origin: [1.0, 2.0, 3.0],
            cell_size: 0.5,
            active_cells: vec![[0, 0, 0]],
            ..Default::default()
        };
        let mesh = extract_sparse_boundary(&data);
        // Corner (0,0,0) -> [1.0, 2.0, 3.0]
        assert!(
            mesh.positions.contains(&[1.0, 2.0, 3.0]),
            "corner [0,0,0] must be at origin"
        );
        // Corner (1,1,1) -> [1.5, 2.5, 3.5]
        assert!(
            mesh.positions.contains(&[1.5, 2.5, 3.5]),
            "corner [1,1,1] must be at origin + cell_size"
        );
        // All 8 corners, no duplicates
        assert_eq!(mesh.positions.len(), 8, "single cell -> 8 unique corners");
    }

    #[test]
    fn normals_have_correct_length() {
        let data = single_cell();
        let mesh = extract_sparse_boundary(&data);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        for n in &mesh.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "normal not unit length: {n:?} (len={len})"
            );
        }
    }

    #[test]
    fn cell_scalar_remaps_to_face_attribute() {
        let mut data = single_cell();
        data.cell_scalars.insert("pressure".to_string(), vec![42.0]);
        let mesh = extract_sparse_boundary(&data);
        match mesh.attributes.get("pressure") {
            Some(AttributeData::Face(vals)) => {
                // 6 quads × 2 triangles = 12 entries
                assert_eq!(vals.len(), 12, "one value per boundary triangle");
                for &v in vals {
                    assert_eq!(v, 42.0, "scalar must match cell value");
                }
            }
            other => panic!("expected Face attribute, got {other:?}"),
        }
    }

    #[test]
    fn cell_color_remaps_to_face_color_attribute() {
        let mut data = two_adjacent_cells();
        data.cell_colors.insert(
            "label".to_string(),
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        );
        let mesh = extract_sparse_boundary(&data);
        match mesh.attributes.get("label") {
            Some(AttributeData::FaceColor(colors)) => {
                // 20 boundary triangles
                assert_eq!(colors.len(), 20, "20 boundary triangles");
            }
            other => panic!("expected FaceColor attribute, got {other:?}"),
        }
    }

    #[test]
    fn node_scalar_remaps_to_face_attribute() {
        // Single cell [0,0,0] with cell_size=1.0.
        // Node grid: W=2, H=2. All 8 corners set to 1.0.
        // All 4-corner averages = 1.0.
        let mut data = single_cell();
        let node_vals = vec![1.0f32; 8]; // 2×2×2
        data.node_scalars.insert("dist".to_string(), node_vals);
        let mesh = extract_sparse_boundary(&data);
        match mesh.attributes.get("dist") {
            Some(AttributeData::Face(vals)) => {
                assert_eq!(vals.len(), 12, "12 boundary triangles");
                for &v in vals {
                    assert!(
                        (v - 1.0).abs() < 1e-6,
                        "averaged node scalar must equal 1.0, got {v}"
                    );
                }
            }
            other => panic!("expected Face attribute, got {other:?}"),
        }
    }

    #[test]
    fn empty_active_cells_returns_empty_mesh_data() {
        let data = SparseVolumeGridData {
            origin: [0.0; 3],
            cell_size: 1.0,
            ..Default::default()
        };
        let mesh = extract_sparse_boundary(&data);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn cell_size_zero_returns_empty_mesh_data() {
        let data = SparseVolumeGridData {
            origin: [0.0; 3],
            cell_size: 0.0,
            active_cells: vec![[0, 0, 0]],
            ..Default::default()
        };
        let mesh = extract_sparse_boundary(&data);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }
}
