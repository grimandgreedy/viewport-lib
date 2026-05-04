//! Unstructured volume mesh processing : tet and hex cell topologies.
//!
//! Converts volumetric cell connectivity into a standard [`MeshData`] by
//! extracting boundary faces (faces shared by exactly one cell) and computing
//! area-weighted vertex normals. Per-cell scalar and color attributes are
//! remapped to per-face attributes so the existing Phase 2 face-rendering path
//! handles coloring without any new GPU infrastructure.
//!
//! # Cell conventions
//!
//! Every cell is stored as exactly **8 vertex indices**:
//! - **Tet**: indices `[0..4]` are the 4 tet vertices; indices `[4..8]` are
//!   `u32::MAX` (sentinel).
//! - **Hex**: all 8 indices are valid vertex positions.
//! - **Mixed** meshes use the sentinel convention to distinguish per cell.
//!
//! Hex face winding follows the standard VTK unstructured-grid ordering so that
//! outward normals are consistent when all cells have positive volume.

use std::collections::HashMap;

use super::types::{AttributeData, MeshData};

/// Sentinel value that marks unused index slots in a tet cell stored as 8 indices.
pub const TET_SENTINEL: u32 = u32::MAX;

/// Input data for an unstructured volume mesh (tets, hexes, or mixed).
///
/// Each cell is represented as exactly 8 vertex indices.  For tetrahedral
/// cells, fill the last four indices with [`TET_SENTINEL`] (`u32::MAX`).
///
/// ```
/// use viewport_lib::{VolumeMeshData, TET_SENTINEL};
///
/// // Two tets sharing vertices 0-1-2
/// let data = VolumeMeshData {
///     positions: vec![
///         [0.0, 0.0, 0.0],
///         [1.0, 0.0, 0.0],
///         [0.5, 1.0, 0.0],
///         [0.5, 0.5, 1.0],
///         [0.5, 0.5, -1.0],
///     ],
///     cells: vec![
///         [0, 1, 2, 3, TET_SENTINEL, TET_SENTINEL, TET_SENTINEL, TET_SENTINEL],
///         [0, 2, 1, 4, TET_SENTINEL, TET_SENTINEL, TET_SENTINEL, TET_SENTINEL],
///     ],
///     ..Default::default()
/// };
/// ```
#[non_exhaustive]
#[derive(Default)]
pub struct VolumeMeshData {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,

    /// Cell connectivity : exactly 8 indices per cell.
    ///
    /// Tets: first 4 indices are the tet vertices; indices `[4..8]` must be
    /// [`TET_SENTINEL`].  Hexes: all 8 indices are valid.
    pub cells: Vec<[u32; 8]>,

    /// Named per-cell scalar attributes (one `f32` per cell).
    ///
    /// Automatically remapped to boundary face scalars during upload so they
    /// can be visualised via [`AttributeKind::Face`](super::types::AttributeKind::Face).
    pub cell_scalars: HashMap<String, Vec<f32>>,

    /// Named per-cell RGBA color attributes (one `[f32; 4]` per cell).
    ///
    /// Automatically remapped to boundary face colors during upload, rendered
    /// via [`AttributeKind::FaceColor`](super::types::AttributeKind::FaceColor).
    pub cell_colors: HashMap<String, Vec<[f32; 4]>>,
}

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

const TET_FACES: [[usize; 3]; 4] = [
    [1, 2, 3], // opposite v0
    [0, 3, 2], // opposite v1
    [0, 1, 3], // opposite v2
    [0, 2, 1], // opposite v3
];

// ---------------------------------------------------------------------------
// Hex face table
// ---------------------------------------------------------------------------
//
// VTK hex vertex numbering used in `upload_volume_mesh_data` docs:
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
//   bottom (-Y): [0,1,2,3]  : normal = (1,0,0)×(1,0,1) = (0,-1,0) ✓
//   top    (+Y): [4,7,6,5]  : normal = (0,0,1)×(1,0,1) = (0,+1,0) ✓
//   front  (-Z): [0,4,5,1]  : normal = (0,1,0)×(1,1,0) = (0,0,-1) ✓
//   back   (+Z): [2,6,7,3]  : normal = (0,1,0)×(-1,1,0)= (0,0,+1) ✓
//   left   (-X): [0,3,7,4]  : normal = (0,0,1)×(0,1,1) = (-1,0,0) ✓
//   right  (+X): [1,5,6,2]  : normal = (0,1,0)×(0,1,1) = (+1,0,0) ✓
//
// The geometric winding-correction step acts as a safety net in case any
// cell is degenerate or oriented unexpectedly.

const HEX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3], // bottom (-Y)
    [4, 7, 6, 5], // top    (+Y)
    [0, 4, 5, 1], // front  (-Z)
    [2, 6, 7, 3], // back   (+Z)
    [0, 3, 7, 4], // left   (-X)
    [1, 5, 6, 2], // right  (+X)
];

/// Opposite face index for each entry in [`HEX_FACES`].
const HEX_FACE_OPPOSITE: [usize; 6] = [1, 0, 3, 2, 5, 4];

// ---------------------------------------------------------------------------
// Boundary extraction
// ---------------------------------------------------------------------------

/// A canonical (sorted) face key used for boundary detection.
type FaceKey = (u32, u32, u32);

/// Canonical key for a quad face, sorted by vertex index.
type QuadFaceKey = (u32, u32, u32, u32);

/// Build a sorted key from three vertex indices.
#[inline]
fn face_key(a: u32, b: u32, c: u32) -> FaceKey {
    let mut arr = [a, b, c];
    arr.sort_unstable();
    (arr[0], arr[1], arr[2])
}

/// Build a sorted key from four vertex indices.
#[inline]
fn quad_face_key(a: u32, b: u32, c: u32, d: u32) -> QuadFaceKey {
    let mut arr = [a, b, c, d];
    arr.sort_unstable();
    (arr[0], arr[1], arr[2], arr[3])
}

/// Internal face record stored in the hash map during boundary extraction.
struct FaceRecord {
    /// Index of the first cell that produced this face.
    cell_index: usize,
    /// Original winding (preserves outward normal direction).
    winding: [u32; 3],
    /// How many cells have contributed this face.  >1 means interior.
    count: u32,
    /// Interior reference point of the owning cell, used for winding correction.
    ///
    /// For tet faces this is the vertex opposite the face, which is a more
    /// reliable inward reference than the cell centroid. For hex faces this
    /// remains the cell centroid.
    interior_ref: [f32; 3],
}

/// Internal quad-face record stored for hex boundary extraction.
struct QuadFaceRecord {
    /// Index of the first cell that produced this face.
    cell_index: usize,
    /// Original quad winding.
    winding: [u32; 4],
    /// How many cells contributed this face. >1 means interior.
    count: u32,
    /// Interior reference point used for triangle winding correction.
    interior_ref: [f32; 3],
}

/// Convert [`VolumeMeshData`] into a standard [`MeshData`] by extracting the
/// boundary surface and remapping per-cell attributes to per-face attributes.
///
/// This is the core of Phase 9: after this step the boundary mesh is uploaded
/// via the existing [`upload_mesh_data`](super::ViewportGpuResources::upload_mesh_data)
/// path and rendered exactly like any other surface mesh.
pub(crate) fn extract_boundary_faces(data: &VolumeMeshData) -> MeshData {
    let n_verts = data.positions.len();

    // Accumulate triangles here; we'll build index buffer from unique vertices later.
    // Strategy: collect boundary triangles as (winding, cell_index) then build
    // a flat non-indexed triangle list and compute normals.
    let mut face_map: HashMap<FaceKey, FaceRecord> = HashMap::new();
    let mut quad_face_map: HashMap<QuadFaceKey, QuadFaceRecord> = HashMap::new();

    for (cell_idx, cell) in data.cells.iter().enumerate() {
        let is_tet = cell[4] == TET_SENTINEL;

        if is_tet {
            // 4 triangular faces
            for (face_idx, face_local) in TET_FACES.iter().enumerate() {
                let a = cell[face_local[0]];
                let b = cell[face_local[1]];
                let c = cell[face_local[2]];
                let opposite = data.positions[cell[face_idx] as usize];
                let key = face_key(a, b, c);
                let entry = face_map.entry(key).or_insert(FaceRecord {
                    cell_index: cell_idx,
                    winding: [a, b, c],
                    count: 0,
                    interior_ref: opposite,
                });
                entry.count += 1;
            }
        } else {
            // 6 quad faces. Deduplicate quads before triangulating; otherwise
            // adjacent hexes can choose different diagonals for the same shared
            // quad and leak interior triangles into the boundary surface.
            for (face_idx, quad) in HEX_FACES.iter().enumerate() {
                let v: [u32; 4] = [cell[quad[0]], cell[quad[1]], cell[quad[2]], cell[quad[3]]];
                let interior_ref = {
                    let opposite = &HEX_FACES[HEX_FACE_OPPOSITE[face_idx]];
                    let mut c = [0.0f32; 3];
                    for &local_vi in opposite {
                        let p = data.positions[cell[local_vi] as usize];
                        c[0] += p[0];
                        c[1] += p[1];
                        c[2] += p[2];
                    }
                    [c[0] / 4.0, c[1] / 4.0, c[2] / 4.0]
                };
                let key = quad_face_key(v[0], v[1], v[2], v[3]);
                let entry = quad_face_map.entry(key).or_insert(QuadFaceRecord {
                    cell_index: cell_idx,
                    winding: v,
                    count: 0,
                    interior_ref,
                });
                entry.count += 1;
            }
        }
    }

    // Collect boundary triangles (count == 1) in a stable order.
    let mut boundary: Vec<(usize, [u32; 3], [f32; 3])> = face_map
        .into_values()
        .filter(|r| r.count == 1)
        .map(|r| (r.cell_index, r.winding, r.interior_ref))
        .collect();

    for quad in quad_face_map.into_values().filter(|r| r.count == 1) {
        boundary.push((
            quad.cell_index,
            [quad.winding[0], quad.winding[1], quad.winding[2]],
            quad.interior_ref,
        ));
        boundary.push((
            quad.cell_index,
            [quad.winding[0], quad.winding[2], quad.winding[3]],
            quad.interior_ref,
        ));
    }

    // Sort by cell index for deterministic output (useful for testing).
    boundary.sort_unstable_by_key(|(cell_idx, _, _)| *cell_idx);

    // Geometric winding correction: ensure each boundary face's normal points
    // outward (away from the owning cell centroid). This is a safety net for
    // degenerate cells or unexpected orientations, and is also the primary
    // correctness mechanism for tet faces where the table winding may be inward.
    for (_, tri, interior_ref) in &mut boundary {
        let pa = data.positions[tri[0] as usize];
        let pb = data.positions[tri[1] as usize];
        let pc = data.positions[tri[2] as usize];

        // Face normal (cross product of two edges; not normalized).
        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let normal = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];

        // Direction from an interior point of the owning cell to the face
        // centroid (outward reference).
        let fc = [
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ];
        let out = [
            fc[0] - interior_ref[0],
            fc[1] - interior_ref[1],
            fc[2] - interior_ref[2],
        ];

        // If the face normal points inward, flip the winding.
        let dot = normal[0] * out[0] + normal[1] * out[1] + normal[2] * out[2];
        if dot < 0.0 {
            tri.swap(1, 2);
        }
    }

    let n_boundary_tris = boundary.len();

    // Build flat triangle lists (positions & indices).
    // We re-use original vertex indices and build a compact index buffer.
    // To avoid the complexity of deduplication, we use the original vertex
    // indices directly and build an index buffer.  Normal accumulation uses
    // the original vertex indices.

    let mut indices: Vec<u32> = Vec::with_capacity(n_boundary_tris * 3);
    // Track which original vertices are used by boundary faces.
    let mut normal_accum: Vec<[f64; 3]> = vec![[0.0; 3]; n_verts];

    for (_, tri, _) in &boundary {
        indices.push(tri[0]);
        indices.push(tri[1]);
        indices.push(tri[2]);

        // Accumulate area-weighted face normal to each vertex.
        let pa = data.positions[tri[0] as usize];
        let pb = data.positions[tri[1] as usize];
        let pc = data.positions[tri[2] as usize];

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
        // Cross product (area-weighted normal)
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for &vi in tri {
            let acc = &mut normal_accum[vi as usize];
            acc[0] += n[0];
            acc[1] += n[1];
            acc[2] += n[2];
        }
    }

    // Normalize accumulated normals.
    let mut normals: Vec<[f32; 3]> = normal_accum
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
                [0.0, 1.0, 0.0] // degenerate fallback
            }
        })
        .collect();

    // Ensure normals vec length matches positions.
    normals.resize(n_verts, [0.0, 1.0, 0.0]);

    // ---------------------------------------------------------------------------
    // Build per-cell -> per-face attribute remapping
    // ---------------------------------------------------------------------------

    let mut attributes: HashMap<String, AttributeData> = HashMap::new();

    for (name, cell_vals) in &data.cell_scalars {
        let face_scalars: Vec<f32> = boundary
            .iter()
            .map(|(cell_idx, _, _)| cell_vals.get(*cell_idx).copied().unwrap_or(0.0))
            .collect();
        attributes.insert(name.clone(), AttributeData::Face(face_scalars));
    }

    for (name, cell_vals) in &data.cell_colors {
        let face_colors: Vec<[f32; 4]> = boundary
            .iter()
            .map(|(cell_idx, _, _)| cell_vals.get(*cell_idx).copied().unwrap_or([1.0; 4]))
            .collect();
        attributes.insert(name.clone(), AttributeData::FaceColor(face_colors));
    }

    MeshData {
        positions: data.positions.clone(),
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

    const TEST_TET_LOCAL: [[usize; 4]; 6] = [
        [0, 1, 5, 6],
        [0, 1, 2, 6],
        [0, 4, 5, 6],
        [0, 4, 7, 6],
        [0, 3, 2, 6],
        [0, 3, 7, 6],
    ];

    fn single_tet() -> VolumeMeshData {
        VolumeMeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            cells: vec![[
                0,
                1,
                2,
                3,
                TET_SENTINEL,
                TET_SENTINEL,
                TET_SENTINEL,
                TET_SENTINEL,
            ]],
            ..Default::default()
        }
    }

    fn two_tets_sharing_face() -> VolumeMeshData {
        // Two tets glued along face [0, 1, 2].
        // Tet A: [0,1,2,3], Tet B: [0,2,1,4]  (reversed to share face outwardly)
        VolumeMeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.5, -1.0],
            ],
            cells: vec![
                [
                    0,
                    1,
                    2,
                    3,
                    TET_SENTINEL,
                    TET_SENTINEL,
                    TET_SENTINEL,
                    TET_SENTINEL,
                ],
                [
                    0,
                    2,
                    1,
                    4,
                    TET_SENTINEL,
                    TET_SENTINEL,
                    TET_SENTINEL,
                    TET_SENTINEL,
                ],
            ],
            ..Default::default()
        }
    }

    fn single_hex() -> VolumeMeshData {
        VolumeMeshData {
            positions: vec![
                [0.0, 0.0, 0.0], // 0
                [1.0, 0.0, 0.0], // 1
                [1.0, 0.0, 1.0], // 2
                [0.0, 0.0, 1.0], // 3
                [0.0, 1.0, 0.0], // 4
                [1.0, 1.0, 0.0], // 5
                [1.0, 1.0, 1.0], // 6
                [0.0, 1.0, 1.0], // 7
            ],
            cells: vec![[0, 1, 2, 3, 4, 5, 6, 7]],
            ..Default::default()
        }
    }

    fn structured_tet_grid(grid_n: usize) -> VolumeMeshData {
        let grid_v = grid_n + 1;
        let vid =
            |ix: usize, iy: usize, iz: usize| (iz * grid_v * grid_v + iy * grid_v + ix) as u32;

        let mut positions = Vec::with_capacity(grid_v * grid_v * grid_v);
        for iz in 0..grid_v {
            for iy in 0..grid_v {
                for ix in 0..grid_v {
                    positions.push([ix as f32, iy as f32, iz as f32]);
                }
            }
        }

        let mut cells = Vec::with_capacity(grid_n * grid_n * grid_n * TEST_TET_LOCAL.len());
        for iz in 0..grid_n {
            for iy in 0..grid_n {
                for ix in 0..grid_n {
                    let cube_verts = [
                        vid(ix, iy, iz),
                        vid(ix + 1, iy, iz),
                        vid(ix + 1, iy, iz + 1),
                        vid(ix, iy, iz + 1),
                        vid(ix, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz + 1),
                        vid(ix, iy + 1, iz + 1),
                    ];
                    for tet in &TEST_TET_LOCAL {
                        cells.push([
                            cube_verts[tet[0]],
                            cube_verts[tet[1]],
                            cube_verts[tet[2]],
                            cube_verts[tet[3]],
                            TET_SENTINEL,
                            TET_SENTINEL,
                            TET_SENTINEL,
                            TET_SENTINEL,
                        ]);
                    }
                }
            }
        }

        VolumeMeshData {
            positions,
            cells,
            ..Default::default()
        }
    }

    fn projected_sphere_tet_grid(grid_n: usize, radius: f32) -> VolumeMeshData {
        let grid_v = grid_n + 1;
        let half = grid_n as f32 / 2.0;
        let vid =
            |ix: usize, iy: usize, iz: usize| (iz * grid_v * grid_v + iy * grid_v + ix) as u32;

        let mut positions = Vec::with_capacity(grid_v * grid_v * grid_v);
        for iz in 0..grid_v {
            for iy in 0..grid_v {
                for ix in 0..grid_v {
                    let x = ix as f32 - half;
                    let y = iy as f32 - half;
                    let z = iz as f32 - half;
                    let len = (x * x + y * y + z * z).sqrt();
                    let s = radius / len;
                    positions.push([x * s, y * s, z * s]);
                }
            }
        }

        let mut cells = Vec::with_capacity(grid_n * grid_n * grid_n * TEST_TET_LOCAL.len());
        for iz in 0..grid_n {
            for iy in 0..grid_n {
                for ix in 0..grid_n {
                    let cube_verts = [
                        vid(ix, iy, iz),
                        vid(ix + 1, iy, iz),
                        vid(ix + 1, iy, iz + 1),
                        vid(ix, iy, iz + 1),
                        vid(ix, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz + 1),
                        vid(ix, iy + 1, iz + 1),
                    ];
                    for tet in &TEST_TET_LOCAL {
                        cells.push([
                            cube_verts[tet[0]],
                            cube_verts[tet[1]],
                            cube_verts[tet[2]],
                            cube_verts[tet[3]],
                            TET_SENTINEL,
                            TET_SENTINEL,
                            TET_SENTINEL,
                            TET_SENTINEL,
                        ]);
                    }
                }
            }
        }

        VolumeMeshData {
            positions,
            cells,
            ..Default::default()
        }
    }

    fn cube_to_sphere([x, y, z]: [f32; 3]) -> [f32; 3] {
        let x2 = x * x;
        let y2 = y * y;
        let z2 = z * z;
        [
            x * (1.0 - 0.5 * (y2 + z2) + (y2 * z2) / 3.0).sqrt(),
            y * (1.0 - 0.5 * (z2 + x2) + (z2 * x2) / 3.0).sqrt(),
            z * (1.0 - 0.5 * (x2 + y2) + (x2 * y2) / 3.0).sqrt(),
        ]
    }

    fn cube_sphere_hex_grid(grid_n: usize, radius: f32) -> VolumeMeshData {
        let grid_v = grid_n + 1;
        let half = grid_n as f32 / 2.0;
        let vid =
            |ix: usize, iy: usize, iz: usize| (iz * grid_v * grid_v + iy * grid_v + ix) as u32;

        let mut positions = Vec::with_capacity(grid_v * grid_v * grid_v);
        for iz in 0..grid_v {
            for iy in 0..grid_v {
                for ix in 0..grid_v {
                    let p = [ix as f32 - half, iy as f32 - half, iz as f32 - half];
                    let cube = [p[0] / half, p[1] / half, p[2] / half];
                    let s = cube_to_sphere(cube);
                    positions.push([s[0] * radius, s[1] * radius, s[2] * radius]);
                }
            }
        }

        let mut cells = Vec::with_capacity(grid_n * grid_n * grid_n);
        for iz in 0..grid_n {
            for iy in 0..grid_n {
                for ix in 0..grid_n {
                    cells.push([
                        vid(ix, iy, iz),
                        vid(ix + 1, iy, iz),
                        vid(ix + 1, iy, iz + 1),
                        vid(ix, iy, iz + 1),
                        vid(ix, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz + 1),
                        vid(ix, iy + 1, iz + 1),
                    ]);
                }
            }
        }

        VolumeMeshData {
            positions,
            cells,
            ..Default::default()
        }
    }

    fn structured_hex_grid(grid_n: usize) -> VolumeMeshData {
        let grid_v = grid_n + 1;
        let vid =
            |ix: usize, iy: usize, iz: usize| (iz * grid_v * grid_v + iy * grid_v + ix) as u32;

        let mut positions = Vec::with_capacity(grid_v * grid_v * grid_v);
        for iz in 0..grid_v {
            for iy in 0..grid_v {
                for ix in 0..grid_v {
                    positions.push([ix as f32, iy as f32, iz as f32]);
                }
            }
        }

        let mut cells = Vec::with_capacity(grid_n * grid_n * grid_n);
        for iz in 0..grid_n {
            for iy in 0..grid_n {
                for ix in 0..grid_n {
                    cells.push([
                        vid(ix, iy, iz),
                        vid(ix + 1, iy, iz),
                        vid(ix + 1, iy, iz + 1),
                        vid(ix, iy, iz + 1),
                        vid(ix, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz + 1),
                        vid(ix, iy + 1, iz + 1),
                    ]);
                }
            }
        }

        VolumeMeshData {
            positions,
            cells,
            ..Default::default()
        }
    }

    #[test]
    fn single_tet_has_four_boundary_faces() {
        let data = single_tet();
        let mesh = extract_boundary_faces(&data);
        assert_eq!(
            mesh.indices.len(),
            4 * 3,
            "single tet -> 4 boundary triangles"
        );
    }

    #[test]
    fn two_tets_sharing_face_eliminates_shared_face() {
        let data = two_tets_sharing_face();
        let mesh = extract_boundary_faces(&data);
        // 4 + 4 - 2 = 6 boundary triangles (shared face contributes 2 tris
        // that cancel, leaving 6)
        assert_eq!(
            mesh.indices.len(),
            6 * 3,
            "two tets sharing a face -> 6 boundary triangles"
        );
    }

    #[test]
    fn single_hex_has_twelve_boundary_triangles() {
        let data = single_hex();
        let mesh = extract_boundary_faces(&data);
        // 6 quad faces × 2 triangles each = 12
        assert_eq!(
            mesh.indices.len(),
            12 * 3,
            "single hex -> 12 boundary triangles"
        );
    }

    #[test]
    fn structured_tet_grid_has_expected_boundary_triangle_count() {
        let grid_n = 3;
        let data = structured_tet_grid(grid_n);
        let mesh = extract_boundary_faces(&data);
        let expected_boundary_tris = 6 * grid_n * grid_n * 2;
        assert_eq!(
            mesh.indices.len(),
            expected_boundary_tris * 3,
            "3x3x3 tet grid should expose 108 boundary triangles"
        );
    }

    #[test]
    fn structured_hex_grid_has_expected_boundary_triangle_count() {
        let grid_n = 3;
        let data = structured_hex_grid(grid_n);
        let mesh = extract_boundary_faces(&data);
        let expected_boundary_tris = 6 * grid_n * grid_n * 2;
        assert_eq!(
            mesh.indices.len(),
            expected_boundary_tris * 3,
            "3x3x3 hex grid should expose 108 boundary triangles"
        );
    }

    #[test]
    fn structured_tet_grid_boundary_is_edge_manifold() {
        let data = structured_tet_grid(3);
        let mesh = extract_boundary_faces(&data);

        let mut edge_counts: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        for tri in mesh.indices.chunks_exact(3) {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let edge = if a < b { (a, b) } else { (b, a) };
                *edge_counts.entry(edge).or_insert(0) += 1;
            }
        }

        let non_manifold: Vec<((u32, u32), usize)> = edge_counts
            .into_iter()
            .filter(|(_, count)| *count != 2)
            .collect();

        assert!(
            non_manifold.is_empty(),
            "boundary should be watertight; bad edges: {non_manifold:?}"
        );
    }

    #[test]
    fn structured_hex_grid_boundary_is_edge_manifold() {
        let data = structured_hex_grid(3);
        let mesh = extract_boundary_faces(&data);

        let mut edge_counts: std::collections::HashMap<(u32, u32), usize> =
            std::collections::HashMap::new();
        for tri in mesh.indices.chunks_exact(3) {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let edge = if a < b { (a, b) } else { (b, a) };
                *edge_counts.entry(edge).or_insert(0) += 1;
            }
        }

        let non_manifold: Vec<((u32, u32), usize)> = edge_counts
            .into_iter()
            .filter(|(_, count)| *count != 2)
            .collect();

        assert!(
            non_manifold.is_empty(),
            "boundary should be watertight; bad edges: {non_manifold:?}"
        );
    }

    #[test]
    fn projected_sphere_tet_grid_boundary_faces_point_outward() {
        let data = projected_sphere_tet_grid(3, 2.0);
        let mesh = extract_boundary_faces(&data);

        for tri in mesh.indices.chunks_exact(3) {
            let pa = mesh.positions[tri[0] as usize];
            let pb = mesh.positions[tri[1] as usize];
            let pc = mesh.positions[tri[2] as usize];

            let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let fc = [
                (pa[0] + pb[0] + pc[0]) / 3.0,
                (pa[1] + pb[1] + pc[1]) / 3.0,
                (pa[2] + pb[2] + pc[2]) / 3.0,
            ];
            let dot = normal[0] * fc[0] + normal[1] * fc[1] + normal[2] * fc[2];
            assert!(dot > 0.0, "boundary face points inward: tri={tri:?}, dot={dot}");
        }
    }

    #[test]
    fn cube_sphere_hex_grid_boundary_faces_point_outward() {
        let data = cube_sphere_hex_grid(3, 2.0);
        let mesh = extract_boundary_faces(&data);

        for tri in mesh.indices.chunks_exact(3) {
            let pa = mesh.positions[tri[0] as usize];
            let pb = mesh.positions[tri[1] as usize];
            let pc = mesh.positions[tri[2] as usize];

            let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let fc = [
                (pa[0] + pb[0] + pc[0]) / 3.0,
                (pa[1] + pb[1] + pc[1]) / 3.0,
                (pa[2] + pb[2] + pc[2]) / 3.0,
            ];
            let dot = normal[0] * fc[0] + normal[1] * fc[1] + normal[2] * fc[2];
            assert!(dot > 0.0, "boundary face points inward: tri={tri:?}, dot={dot}");
        }
    }

    #[test]
    fn normals_have_correct_length() {
        let data = single_tet();
        let mesh = extract_boundary_faces(&data);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        for n in &mesh.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5 || len < 1e-5,
                "normal not unit: {n:?}"
            );
        }
    }

    #[test]
    fn cell_scalar_remaps_to_face_attribute() {
        let mut data = single_tet();
        data.cell_scalars.insert("pressure".to_string(), vec![42.0]);
        let mesh = extract_boundary_faces(&data);
        match mesh.attributes.get("pressure") {
            Some(AttributeData::Face(vals)) => {
                assert_eq!(vals.len(), 4, "one value per boundary triangle");
                for &v in vals {
                    assert_eq!(v, 42.0);
                }
            }
            other => panic!("expected Face attribute, got {other:?}"),
        }
    }

    #[test]
    fn cell_color_remaps_to_face_color_attribute() {
        let mut data = two_tets_sharing_face();
        data.cell_colors.insert(
            "label".to_string(),
            vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        );
        let mesh = extract_boundary_faces(&data);
        match mesh.attributes.get("label") {
            Some(AttributeData::FaceColor(colors)) => {
                assert_eq!(colors.len(), 6, "6 boundary faces");
            }
            other => panic!("expected FaceColor attribute, got {other:?}"),
        }
    }

    #[test]
    fn positions_preserved_unchanged() {
        let data = single_hex();
        let mesh = extract_boundary_faces(&data);
        assert_eq!(mesh.positions, data.positions);
    }
}
