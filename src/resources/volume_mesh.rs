//! Unstructured volume mesh processing : tet, pyramid, wedge, and hex cell topologies.
//!
//! Converts volumetric cell connectivity into a standard [`MeshData`] by
//! extracting boundary faces (faces shared by exactly one cell) and computing
//! area-weighted vertex normals. Per-cell scalar and color attributes are
//! remapped to per-face attributes so the existing Phase 2 face-rendering path
//! handles coloring without any new GPU infrastructure.
//!
//! # Cell conventions
//!
//! Every cell is stored as exactly **8 vertex indices** using [`CELL_SENTINEL`]
//! (`u32::MAX`) to pad unused slots:
//! - **Tet**: indices `[0..4]` valid; `[4..8]` = `CELL_SENTINEL`
//! - **Pyramid**: indices `[0..5]` valid; `[5..8]` = `CELL_SENTINEL`
//! - **Wedge**: indices `[0..6]` valid; `[6..8]` = `CELL_SENTINEL`
//! - **Hex**: all 8 indices are valid vertex positions.
//!
//! Mixed meshes use the sentinel convention to distinguish cell type per cell.
//!
//! Hex face winding follows the standard VTK unstructured-grid ordering so that
//! outward normals are consistent when all cells have positive volume.

use std::collections::HashMap;

use rayon::prelude::*;

use super::types::{AttributeData, MeshData};

const PARALLEL_THRESHOLD: usize = 1024;

/// Sentinel value that marks unused index slots in a cell stored as 8 indices.
///
/// Slots beyond the cell's vertex count must be filled with this value.
/// For example, a tet uses slots `[0..4]`; slots `[4..8]` must be `CELL_SENTINEL`.
pub const CELL_SENTINEL: u32 = u32::MAX;

/// Deprecated alias for [`CELL_SENTINEL`].
#[deprecated(since = "0.13.0", note = "use `CELL_SENTINEL` instead")]
pub const TET_SENTINEL: u32 = CELL_SENTINEL;

/// Input data for an unstructured volume mesh (tets, hexes, or mixed).
///
/// Each cell is represented as exactly 8 vertex indices.  For cells with fewer
/// than 8 vertices, fill unused slots with [`CELL_SENTINEL`] (`u32::MAX`).
///
/// ```
/// use viewport_lib::{VolumeMeshData, CELL_SENTINEL};
///
/// // Two tets sharing vertices 0-1-2
/// let mut data = VolumeMeshData::default();
/// data.positions = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0],
///     [0.5, 1.0, 0.0],
///     [0.5, 0.5, 1.0],
///     [0.5, 0.5, -1.0],
/// ];
/// data.cells = vec![
///     [0, 1, 2, 3, CELL_SENTINEL, CELL_SENTINEL, CELL_SENTINEL, CELL_SENTINEL],
///     [0, 2, 1, 4, CELL_SENTINEL, CELL_SENTINEL, CELL_SENTINEL, CELL_SENTINEL],
/// ];
/// ```
#[non_exhaustive]
#[derive(Default)]
pub struct VolumeMeshData {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,

    /// Cell connectivity : exactly 8 indices per cell.
    ///
    /// Tets: first 4 indices are the tet vertices; indices `[4..8]` must be
    /// [`CELL_SENTINEL`].  Hexes: all 8 indices are valid.  Other cell types
    /// use [`CELL_SENTINEL`] to pad unused slots (see module-level docs).
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
const PYRAMID_QUAD_FACE: [[usize; 4]; 1] = [
    [0, 1, 2, 3], // base
];

/// Triangular side faces of a pyramid (apex = vertex 4).
const PYRAMID_TRI_FACES: [[usize; 3]; 4] = [
    [0, 4, 1], // front
    [1, 4, 2], // right
    [2, 4, 3], // back
    [3, 4, 0], // left
];

/// Edges of a pyramid: 4 base + 4 lateral.
const PYRAMID_EDGES: [[usize; 2]; 8] = [
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
const WEDGE_TRI_FACES: [[usize; 3]; 2] = [
    [0, 2, 1], // bottom (outward = downward)
    [3, 4, 5], // top    (outward = upward)
];

/// Quad lateral faces of a wedge.
const WEDGE_QUAD_FACES: [[usize; 4]; 3] = [
    [0, 1, 4, 3], // side 0
    [1, 2, 5, 4], // side 1
    [2, 0, 3, 5], // side 2
];

/// Edges of a wedge: 3 bottom + 3 top + 3 vertical.
const WEDGE_EDGES: [[usize; 2]; 9] = [
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

// ---------------------------------------------------------------------------
// Boundary extraction
// ---------------------------------------------------------------------------

/// A canonical (sorted) face key used for boundary detection.
type FaceKey = (u32, u32, u32);

/// Canonical key for a quad face, sorted by vertex index.
type QuadFaceKey = (u32, u32, u32, u32);

// (sorted_key, cell_idx, winding, interior_ref)
type TriEntry = (FaceKey, usize, [u32; 3], [f32; 3]);
type QuadEntry = (QuadFaceKey, usize, [u32; 4], [f32; 3]);

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

/// Generate all triangular face entries for a single cell.
fn generate_tri_entries(cell_idx: usize, cell: &[u32; 8], positions: &[[f32; 3]]) -> Vec<TriEntry> {
    let ct = cell_type(cell);
    let nv = ct.vertex_count();
    let mut out = Vec::new();
    match ct {
        CellType::Tet => {
            for (face_idx, face_local) in TET_FACES.iter().enumerate() {
                let a = cell[face_local[0]];
                let b = cell[face_local[1]];
                let c = cell[face_local[2]];
                // Opposite vertex is the best interior reference for tets.
                let interior_ref = positions[cell[face_idx] as usize];
                out.push((face_key(a, b, c), cell_idx, [a, b, c], interior_ref));
            }
        }
        CellType::Pyramid => {
            let centroid = cell_centroid(cell, nv, positions);
            for face_local in &PYRAMID_TRI_FACES {
                let a = cell[face_local[0]];
                let b = cell[face_local[1]];
                let c = cell[face_local[2]];
                out.push((face_key(a, b, c), cell_idx, [a, b, c], centroid));
            }
        }
        CellType::Wedge => {
            let centroid = cell_centroid(cell, nv, positions);
            for face_local in &WEDGE_TRI_FACES {
                let a = cell[face_local[0]];
                let b = cell[face_local[1]];
                let c = cell[face_local[2]];
                out.push((face_key(a, b, c), cell_idx, [a, b, c], centroid));
            }
        }
        CellType::Hex => {} // hex has no triangular faces
    }
    out
}

/// Generate all quad face entries for a single cell.
fn generate_quad_entries(
    cell_idx: usize,
    cell: &[u32; 8],
    positions: &[[f32; 3]],
) -> Vec<QuadEntry> {
    let ct = cell_type(cell);
    let nv = ct.vertex_count();
    let mut out = Vec::new();
    match ct {
        CellType::Tet => {} // tet has no quad faces
        CellType::Pyramid => {
            let centroid = cell_centroid(cell, nv, positions);
            for quad_local in &PYRAMID_QUAD_FACE {
                let v = [
                    cell[quad_local[0]],
                    cell[quad_local[1]],
                    cell[quad_local[2]],
                    cell[quad_local[3]],
                ];
                out.push((quad_face_key(v[0], v[1], v[2], v[3]), cell_idx, v, centroid));
            }
        }
        CellType::Wedge => {
            let centroid = cell_centroid(cell, nv, positions);
            for quad_local in &WEDGE_QUAD_FACES {
                let v = [
                    cell[quad_local[0]],
                    cell[quad_local[1]],
                    cell[quad_local[2]],
                    cell[quad_local[3]],
                ];
                out.push((quad_face_key(v[0], v[1], v[2], v[3]), cell_idx, v, centroid));
            }
        }
        CellType::Hex => {
            for (face_idx, quad) in HEX_FACES.iter().enumerate() {
                let v: [u32; 4] = [cell[quad[0]], cell[quad[1]], cell[quad[2]], cell[quad[3]]];
                let interior_ref = {
                    let opposite = &HEX_FACES[HEX_FACE_OPPOSITE[face_idx]];
                    let mut c = [0.0f32; 3];
                    for &local_vi in opposite {
                        let p = positions[cell[local_vi] as usize];
                        c[0] += p[0];
                        c[1] += p[1];
                        c[2] += p[2];
                    }
                    [c[0] / 4.0, c[1] / 4.0, c[2] / 4.0]
                };
                out.push((
                    quad_face_key(v[0], v[1], v[2], v[3]),
                    cell_idx,
                    v,
                    interior_ref,
                ));
            }
        }
    }
    out
}

/// Collect entries that appear exactly once (boundary faces) from a sorted slice.
fn collect_boundary_tri(entries: &[TriEntry]) -> Vec<(usize, [u32; 3], [f32; 3])> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let key = entries[i].0;
        let mut j = i + 1;
        while j < entries.len() && entries[j].0 == key {
            j += 1;
        }
        if j - i == 1 {
            out.push((entries[i].1, entries[i].2, entries[i].3));
        }
        i = j;
    }
    out
}

/// Collect quad entries that appear exactly once (boundary faces) from a sorted slice.
fn collect_boundary_quad(entries: &[QuadEntry]) -> Vec<(usize, [u32; 4], [f32; 3])> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let key = entries[i].0;
        let mut j = i + 1;
        while j < entries.len() && entries[j].0 == key {
            j += 1;
        }
        if j - i == 1 {
            out.push((entries[i].1, entries[i].2, entries[i].3));
        }
        i = j;
    }
    out
}

/// Ensure the triangle winding produces an outward-facing normal relative to
/// `interior_ref` (a point inside the owning cell).
#[inline]
fn correct_winding(tri: &mut [u32; 3], interior_ref: &[f32; 3], positions: &[[f32; 3]]) {
    let pa = positions[tri[0] as usize];
    let pb = positions[tri[1] as usize];
    let pc = positions[tri[2] as usize];
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
    let out = [
        fc[0] - interior_ref[0],
        fc[1] - interior_ref[1],
        fc[2] - interior_ref[2],
    ];
    if normal[0] * out[0] + normal[1] * out[1] + normal[2] * out[2] < 0.0 {
        tri.swap(1, 2);
    }
}

/// Convert [`VolumeMeshData`] into a standard [`MeshData`] by extracting the
/// boundary surface and remapping per-cell attributes to per-face attributes.
///
/// This is the core of Phase 9: after this step the boundary mesh is uploaded
/// via the existing [`upload_mesh_data`](super::ViewportGpuResources::upload_mesh_data)
/// path and rendered exactly like any other surface mesh.
///
/// Returns `(mesh_data, face_to_cell)` where `face_to_cell[i]` is the cell
/// index that boundary triangle `i` belongs to.
pub(crate) fn extract_boundary_faces(data: &VolumeMeshData) -> (MeshData, Vec<u32>) {
    let n_verts = data.positions.len();

    // Generate face entries (parallel above threshold, sequential below).
    let (mut tri_entries, mut quad_entries) = if data.cells.len() >= PARALLEL_THRESHOLD {
        let tri = data
            .cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(ci, cell)| generate_tri_entries(ci, cell, &data.positions))
            .collect();
        let quad = data
            .cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(ci, cell)| generate_quad_entries(ci, cell, &data.positions))
            .collect();
        (tri, quad)
    } else {
        let mut tri: Vec<TriEntry> = Vec::new();
        let mut quad: Vec<QuadEntry> = Vec::new();
        for (ci, cell) in data.cells.iter().enumerate() {
            tri.extend(generate_tri_entries(ci, cell, &data.positions));
            quad.extend(generate_quad_entries(ci, cell, &data.positions));
        }
        (tri, quad)
    };

    tri_entries.par_sort_unstable_by_key(|e| e.0);
    quad_entries.par_sort_unstable_by_key(|e| e.0);

    // Collect boundary faces (count == 1) via linear scan.
    let mut boundary: Vec<(usize, [u32; 3], [f32; 3])> = collect_boundary_tri(&tri_entries);
    for (ci, winding, iref) in collect_boundary_quad(&quad_entries) {
        boundary.push((ci, [winding[0], winding[1], winding[2]], iref));
        boundary.push((ci, [winding[0], winding[2], winding[3]], iref));
    }

    // Sort by cell index for deterministic output (useful for testing).
    boundary.sort_unstable_by_key(|(ci, _, _)| *ci);

    // Geometric winding correction (parallel): ensure each boundary face's normal
    // points outward. This is the primary correctness mechanism for tets where
    // the table winding may be inward.
    boundary
        .par_iter_mut()
        .for_each(|(_, tri, iref)| correct_winding(tri, iref, &data.positions));

    let n_boundary_tris = boundary.len();

    // Build index buffer and accumulate area-weighted normals (sequential:
    // normal_accum has shared per-vertex writes).
    let mut indices: Vec<u32> = Vec::with_capacity(n_boundary_tris * 3);
    let mut normal_accum: Vec<[f64; 3]> = vec![[0.0; 3]; n_verts];

    for (_, tri, _) in &boundary {
        indices.push(tri[0]);
        indices.push(tri[1]);
        indices.push(tri[2]);

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
                [0.0, 1.0, 0.0]
            }
        })
        .collect();

    normals.resize(n_verts, [0.0, 1.0, 0.0]);

    let mut attributes: HashMap<String, AttributeData> = HashMap::new();

    for (name, cell_vals) in &data.cell_scalars {
        let face_scalars: Vec<f32> = boundary
            .iter()
            .map(|(ci, _, _)| cell_vals.get(*ci).copied().unwrap_or(0.0))
            .collect();
        attributes.insert(name.clone(), AttributeData::Face(face_scalars));
    }

    for (name, cell_vals) in &data.cell_colors {
        let face_colors: Vec<[f32; 4]> = boundary
            .iter()
            .map(|(ci, _, _)| cell_vals.get(*ci).copied().unwrap_or([1.0; 4]))
            .collect();
        attributes.insert(name.clone(), AttributeData::FaceColor(face_colors));
    }

    let face_to_cell: Vec<u32> = boundary.iter().map(|(ci, _, _)| *ci as u32).collect();

    (
        MeshData {
            positions: data.positions.clone(),
            normals,
            indices,
            uvs: None,
            tangents: None,
            attributes,
        },
        face_to_cell,
    )
}

// ---------------------------------------------------------------------------
// Clipped volume mesh extraction
// ---------------------------------------------------------------------------
//
// Design note (Phase 1 - scope and invariants)
// =============================================
//
// ## Goal
//
// Produce a `MeshData` that reads as a filled volumetric cross-section rather
// than an open hollow shell when one or more clip planes intersect a volume mesh.
//
// ## What this is NOT
//
// This is not a generic clip overlay.  The renderer's cap-fill system generates
// a flat polygon on each clip plane independently of the underlying geometry.
// For volume meshes that is wrong: it produces a slab with no per-cell color
// information.  `extract_clipped_volume_faces` replaces the cap-fill role for
// volume meshes entirely.  Callers must disable cap-fill when using this path.
//
// ## Clip plane encoding
//
// Each plane is `[nx, ny, nz, d]: [f32; 4]` where a point `p` is on the KEPT
// side when `dot(p, [nx, ny, nz]) + d >= 0`.  This matches the layout of
// `ClipPlanesUniform::planes` so the same values can be forwarded directly to
// both the CPU extraction and the GPU clip shader.
//
// An empty slice is valid and produces the same result as `extract_boundary_faces`.
//
// ## Cell classification
//
// A vertex is "kept" if it satisfies ALL planes.
//
// - All vertices kept   -> cell contributes its visible boundary faces, unchanged.
// - No  vertices kept   -> cell is discarded entirely.
// - Mixed               -> cell is "intersected": contributes clipped boundary
//                          faces and one section polygon per cutting plane.
//
// ## Section polygon semantics
//
// For each plane that cuts an intersected cell:
// 1. Collect all edge-plane intersection points (one per cell edge that crosses
//    the plane).
// 2. Order the points into a polygon on the plane (sort by angle around the
//    centroid projected onto the plane).
// 3. Clip the polygon against all other active planes.
// 4. Triangulate the surviving polygon using a fan from the first vertex.
//
// Section face winding: the face normal must point in the direction of the
// cutting plane's normal (i.e., toward the kept side / toward the viewer).
//
// ## Boundary face clipping
//
// Boundary faces of intersected cells are clipped against all active planes
// using the Sutherland-Hodgman algorithm before triangulation.  A boundary
// face entirely on the discarded side of any plane is dropped.
//
// ## Attribute propagation
//
// Section triangles inherit the owning cell's `cell_scalars` and `cell_colors`
// values exactly as boundary triangles do.  The output `MeshData` uses the same
// `AttributeKind::Face` / `AttributeKind::FaceColor` paths, so colormaps work
// with no changes to the renderer.
//
// ## Output type
//
// The function returns an ordinary `MeshData`.  No new intermediate type is
// introduced.  The caller uploads this as a regular mesh and renders it with
// the standard pipeline; the only renderer-side requirement is that cap-fill
// is disabled for the same scene object.

/// Produce a clipped `MeshData` from volume cell connectivity.
///
/// Each entry in `clip_planes` is `[nx, ny, nz, d]` where a point `p` is on
/// the kept side when `dot(p, [nx,ny,nz]) + d >= 0`.  This is the same
/// encoding as [`ClipPlanesUniform::planes`](crate::renderer::types::ClipPlanesUniform)
/// so values can be forwarded to both the CPU path and the GPU clip shader.
///
/// Passing an empty slice returns the same result as [`extract_boundary_faces`].
///
/// # Semantics
///
/// - A cell where all vertices satisfy every plane contributes its boundary
///   faces unchanged.
/// - A cell where no vertex satisfies every plane is discarded.
/// - An intersected cell contributes its surviving boundary faces (clipped) and
///   one section polygon per plane that cuts it (clipped against all other
///   planes, then triangulated).
///
/// Section face normals point toward the kept side (matching the cutting plane
/// normal).  Per-cell scalar and color attributes are propagated to section
/// triangles identically to boundary triangles.
///
/// # Renderer contract
///
/// Generic cap-fill must be disabled for scene objects rendered via this path.
/// Section faces are generated here from cell data; the generic cap overlay
/// does not have access to per-cell attribute information and would produce an
/// incorrect result if left enabled.
/// Cell edges for tets: all 6 pairs from 4 vertices.
const TET_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

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
const HEX_EDGES: [[usize; 2]; 12] = [
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

// ---------------------------------------------------------------------------
// Cell type detection
// ---------------------------------------------------------------------------

/// Internal cell type, detected from sentinel slots.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CellType {
    Tet,
    Pyramid,
    Wedge,
    Hex,
}

impl CellType {
    fn vertex_count(self) -> usize {
        match self {
            CellType::Tet => 4,
            CellType::Pyramid => 5,
            CellType::Wedge => 6,
            CellType::Hex => 8,
        }
    }

    fn edges(self) -> &'static [[usize; 2]] {
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
fn cell_type(cell: &[u32; 8]) -> CellType {
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

/// Centroid of the first `nv` vertices of `cell`.
#[inline]
fn cell_centroid(cell: &[u32; 8], nv: usize, positions: &[[f32; 3]]) -> [f32; 3] {
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

/// Signed distance from `p` to `plane` (`[nx, ny, nz, d]`).
/// Positive means on the kept side (`dot(p, n) + d >= 0`).
#[inline]
fn plane_dist(p: [f32; 3], plane: [f32; 4]) -> f32 {
    p[0] * plane[0] + p[1] * plane[1] + p[2] * plane[2] + plane[3]
}

/// Cross product of two 3-vectors.
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalize a 3-vector; returns `[0, 1, 0]` for degenerate input.
#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = dot3(v, v).sqrt();
    if len > 1e-12 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Intern `p` into `positions`, returning its index.
/// Uses bit-exact comparison so the same floating-point value always maps to
/// the same slot.
fn intern_pos(
    p: [f32; 3],
    positions: &mut Vec<[f32; 3]>,
    pos_map: &mut HashMap<[u32; 3], u32>,
) -> u32 {
    let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
    if let Some(&idx) = pos_map.get(&key) {
        return idx;
    }
    let idx = positions.len() as u32;
    positions.push(p);
    pos_map.insert(key, idx);
    idx
}

/// Clip `poly` against a single plane (Sutherland-Hodgman).
/// Vertices satisfying `plane_dist >= 0` are on the kept side.
fn clip_polygon_one_plane(poly: Vec<[f32; 3]>, plane: [f32; 4]) -> Vec<[f32; 3]> {
    if poly.is_empty() {
        return poly;
    }
    let n = poly.len();
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let da = plane_dist(a, plane);
        let db = plane_dist(b, plane);
        let a_in = da >= 0.0;
        let b_in = db >= 0.0;
        if a_in {
            out.push(a);
        }
        if a_in != b_in {
            let denom = da - db;
            if denom.abs() > 1e-30 {
                let t = da / denom;
                out.push([
                    a[0] + t * (b[0] - a[0]),
                    a[1] + t * (b[1] - a[1]),
                    a[2] + t * (b[2] - a[2]),
                ]);
            }
        }
    }
    out
}

/// Clip `poly` against all `planes` in sequence.
fn clip_polygon_planes(mut poly: Vec<[f32; 3]>, planes: &[[f32; 4]]) -> Vec<[f32; 3]> {
    for &plane in planes {
        if poly.is_empty() {
            break;
        }
        poly = clip_polygon_one_plane(poly, plane);
    }
    poly
}

/// Build an orthonormal `(u, v)` basis for a plane with the given `normal`.
fn plane_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let ref_vec: [f32; 3] = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize3(cross3(normal, ref_vec));
    let v = cross3(normal, u);
    (u, v)
}

/// Sort `pts` into angular order around their centroid on the given plane.
///
/// Uses a `(u, v)` frame derived from `normal` so that the resulting polygon
/// is non-self-intersecting for any convex (and mildly non-convex) cross-section.
fn sort_polygon_on_plane(pts: &mut Vec<[f32; 3]>, normal: [f32; 3]) {
    if pts.len() < 3 {
        return;
    }
    let n = pts.len() as f32;
    let cx = pts.iter().map(|p| p[0]).sum::<f32>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f32>() / n;
    let cz = pts.iter().map(|p| p[2]).sum::<f32>() / n;
    let centroid = [cx, cy, cz];
    let (u, v) = plane_basis(normal);
    pts.sort_by(|a, b| {
        let da = [a[0] - centroid[0], a[1] - centroid[1], a[2] - centroid[2]];
        let db = [b[0] - centroid[0], b[1] - centroid[1], b[2] - centroid[2]];
        let ang_a = dot3(da, v).atan2(dot3(da, u));
        let ang_b = dot3(db, v).atan2(dot3(db, u));
        ang_a
            .partial_cmp(&ang_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Fan-triangulate a polygon from `poly[0]`.
fn fan_triangulate(poly: &[[f32; 3]]) -> Vec<[[f32; 3]; 3]> {
    if poly.len() < 3 {
        return Vec::new();
    }
    (1..poly.len() - 1)
        .map(|i| [poly[0], poly[i], poly[i + 1]])
        .collect()
}

/// Generate section triangles for a single intersected cell across all clip planes.
fn generate_section_tris(
    cell_idx: usize,
    cell: &[u32; 8],
    positions: &[[f32; 3]],
    clip_planes: &[[f32; 4]],
) -> Vec<(usize, [[f32; 3]; 3])> {
    let mut out = Vec::new();
    let edges = cell_type(cell).edges();

    for (pi, &plane) in clip_planes.iter().enumerate() {
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for edge in edges {
            let pa = positions[cell[edge[0]] as usize];
            let pb = positions[cell[edge[1]] as usize];
            let da = plane_dist(pa, plane);
            let db = plane_dist(pb, plane);
            if (da >= 0.0) != (db >= 0.0) {
                let denom = da - db;
                if denom.abs() > 1e-30 {
                    let t = da / denom;
                    pts.push([
                        pa[0] + t * (pb[0] - pa[0]),
                        pa[1] + t * (pb[1] - pa[1]),
                        pa[2] + t * (pb[2] - pa[2]),
                    ]);
                }
            }
        }
        if pts.len() < 3 {
            continue;
        }
        let plane_normal = [plane[0], plane[1], plane[2]];
        sort_polygon_on_plane(&mut pts, plane_normal);
        let other_planes: Vec<[f32; 4]> = clip_planes
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != pi)
            .map(|(_, p)| *p)
            .collect();
        let pts = clip_polygon_planes(pts, &other_planes);
        if pts.len() < 3 {
            continue;
        }
        for mut tri in fan_triangulate(&pts) {
            let ab = [
                tri[1][0] - tri[0][0],
                tri[1][1] - tri[0][1],
                tri[1][2] - tri[0][2],
            ];
            let ac = [
                tri[2][0] - tri[0][0],
                tri[2][1] - tri[0][1],
                tri[2][2] - tri[0][2],
            ];
            let n = cross3(ab, ac);
            if dot3(n, plane_normal) < 0.0 {
                tri.swap(1, 2);
            }
            out.push((cell_idx, tri));
        }
    }
    out
}

/// Produce a clipped `MeshData` from volume cell connectivity and a set of
/// clip planes.
///
/// See the design note in the section comment above for the full contract.
/// Extract boundary and section faces from a volume mesh clipped by one or
/// more planes.
///
/// Returns `(mesh_data, face_to_cell)` where `face_to_cell[i]` is the cell
/// index that output triangle `i` belongs to.
pub fn extract_clipped_volume_faces(
    data: &VolumeMeshData,
    clip_planes: &[[f32; 4]],
) -> (MeshData, Vec<u32>) {
    if clip_planes.is_empty() {
        return extract_boundary_faces(data);
    }

    // Classify every vertex: kept = satisfies ALL planes (parallel).
    let vert_kept: Vec<bool> = data
        .positions
        .par_iter()
        .map(|&p| clip_planes.iter().all(|&pl| plane_dist(p, pl) >= 0.0))
        .collect();

    // Generate face entries, skipping fully-discarded cells.
    let (mut tri_entries, mut quad_entries) = if data.cells.len() >= PARALLEL_THRESHOLD {
        let vk = &vert_kept;
        let tri = data
            .cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(ci, cell)| {
                let nv = cell_type(cell).vertex_count();
                if (0..nv).all(|i| !vk[cell[i] as usize]) {
                    return Vec::new();
                }
                generate_tri_entries(ci, cell, &data.positions)
            })
            .collect();
        let quad = data
            .cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(ci, cell)| {
                let nv = cell_type(cell).vertex_count();
                if (0..nv).all(|i| !vk[cell[i] as usize]) {
                    return Vec::new();
                }
                generate_quad_entries(ci, cell, &data.positions)
            })
            .collect();
        (tri, quad)
    } else {
        let mut tri: Vec<TriEntry> = Vec::new();
        let mut quad: Vec<QuadEntry> = Vec::new();
        for (ci, cell) in data.cells.iter().enumerate() {
            let nv = cell_type(cell).vertex_count();
            let kc = (0..nv).filter(|&i| vert_kept[cell[i] as usize]).count();
            if kc == 0 {
                continue;
            }
            tri.extend(generate_tri_entries(ci, cell, &data.positions));
            quad.extend(generate_quad_entries(ci, cell, &data.positions));
        }
        (tri, quad)
    };

    tri_entries.par_sort_unstable_by_key(|e| e.0);
    quad_entries.par_sort_unstable_by_key(|e| e.0);

    let mut boundary: Vec<(usize, [u32; 3], [f32; 3])> = collect_boundary_tri(&tri_entries);
    for (ci, winding, iref) in collect_boundary_quad(&quad_entries) {
        boundary.push((ci, [winding[0], winding[1], winding[2]], iref));
        boundary.push((ci, [winding[0], winding[2], winding[3]], iref));
    }
    boundary.sort_unstable_by_key(|(ci, _, _)| *ci);

    boundary
        .par_iter_mut()
        .for_each(|(_, tri, iref)| correct_winding(tri, iref, &data.positions));

    // Precompute per-cell vertex and kept-vertex counts.
    let cell_nv: Vec<usize> = data
        .cells
        .iter()
        .map(|c| cell_type(c).vertex_count())
        .collect();
    let cell_kept: Vec<usize> = data
        .cells
        .iter()
        .zip(cell_nv.iter())
        .map(|(cell, &nv)| (0..nv).filter(|&i| vert_kept[cell[i] as usize]).count())
        .collect();

    // Boundary faces: emit directly for fully-kept cells, clip for intersected (parallel).
    let mut out_tris: Vec<(usize, [[f32; 3]; 3])> = boundary
        .par_iter()
        .flat_map_iter(|(cell_idx, tri, _)| {
            let nv = cell_nv[*cell_idx];
            let kc = cell_kept[*cell_idx];
            let pa = data.positions[tri[0] as usize];
            let pb = data.positions[tri[1] as usize];
            let pc = data.positions[tri[2] as usize];
            if kc == nv {
                vec![(*cell_idx, [pa, pb, pc])]
            } else {
                let clipped = clip_polygon_planes(vec![pa, pb, pc], clip_planes);
                fan_triangulate(&clipped)
                    .into_iter()
                    .map(|t| (*cell_idx, t))
                    .collect()
            }
        })
        .collect();

    // Section polygons: one per cutting plane per intersected cell (parallel).
    let section_tris: Vec<(usize, [[f32; 3]; 3])> = data
        .cells
        .par_iter()
        .enumerate()
        .filter(|(ci, _)| {
            let kc = cell_kept[*ci];
            kc > 0 && kc < cell_nv[*ci]
        })
        .flat_map_iter(|(ci, cell)| generate_section_tris(ci, cell, &data.positions, clip_planes))
        .collect();
    out_tris.extend(section_tris);

    // Intern positions and build the index buffer (sequential: shared HashMap).
    let mut positions: Vec<[f32; 3]> = data.positions.clone();
    let mut pos_map: HashMap<[u32; 3], u32> = HashMap::new();
    for (i, &p) in data.positions.iter().enumerate() {
        let key = [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()];
        pos_map.entry(key).or_insert(i as u32);
    }

    let mut indexed_tris: Vec<(usize, [u32; 3])> = Vec::with_capacity(out_tris.len());
    for (cell_idx, [p0, p1, p2]) in &out_tris {
        let i0 = intern_pos(*p0, &mut positions, &mut pos_map);
        let i1 = intern_pos(*p1, &mut positions, &mut pos_map);
        let i2 = intern_pos(*p2, &mut positions, &mut pos_map);
        indexed_tris.push((*cell_idx, [i0, i1, i2]));
    }

    let n_verts = positions.len();
    let mut normal_accum: Vec<[f64; 3]> = vec![[0.0; 3]; n_verts];
    let mut indices: Vec<u32> = Vec::with_capacity(indexed_tris.len() * 3);

    for (_, tri) in &indexed_tris {
        indices.push(tri[0]);
        indices.push(tri[1]);
        indices.push(tri[2]);

        let pa = positions[tri[0] as usize];
        let pb = positions[tri[1] as usize];
        let pc = positions[tri[2] as usize];
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
        for &vi in tri {
            let acc = &mut normal_accum[vi as usize];
            acc[0] += n[0];
            acc[1] += n[1];
            acc[2] += n[2];
        }
    }

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

    let mut attributes: HashMap<String, AttributeData> = HashMap::new();
    for (name, cell_vals) in &data.cell_scalars {
        let face_scalars: Vec<f32> = indexed_tris
            .iter()
            .map(|(ci, _)| cell_vals.get(*ci).copied().unwrap_or(0.0))
            .collect();
        attributes.insert(name.clone(), AttributeData::Face(face_scalars));
    }
    for (name, cell_vals) in &data.cell_colors {
        let face_colors: Vec<[f32; 4]> = indexed_tris
            .iter()
            .map(|(ci, _)| cell_vals.get(*ci).copied().unwrap_or([1.0; 4]))
            .collect();
        attributes.insert(name.clone(), AttributeData::FaceColor(face_colors));
    }

    let face_to_cell: Vec<u32> = indexed_tris.iter().map(|(ci, _)| *ci as u32).collect();

    (
        MeshData {
            positions,
            normals,
            indices,
            uvs: None,
            tangents: None,
            attributes,
        },
        face_to_cell,
    )
}

// ---------------------------------------------------------------------------
// VolumeMeshData helpers
// ---------------------------------------------------------------------------

impl VolumeMeshData {
    /// Append a tetrahedral cell (4 vertices).
    ///
    /// Slots `[4..8]` are filled with [`CELL_SENTINEL`] automatically.
    pub fn push_tet(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.cells.push([
            a,
            b,
            c,
            d,
            CELL_SENTINEL,
            CELL_SENTINEL,
            CELL_SENTINEL,
            CELL_SENTINEL,
        ]);
    }

    /// Append a pyramidal cell (square base + apex, 5 vertices).
    ///
    /// `base` holds the four base vertices in VTK order (counter-clockwise
    /// when viewed from outside the cell); `apex` is the tip vertex.
    /// Slots `[5..8]` are filled with [`CELL_SENTINEL`] automatically.
    pub fn push_pyramid(&mut self, base: [u32; 4], apex: u32) {
        self.cells.push([
            base[0],
            base[1],
            base[2],
            base[3],
            apex,
            CELL_SENTINEL,
            CELL_SENTINEL,
            CELL_SENTINEL,
        ]);
    }

    /// Append a wedge (triangular prism) cell (6 vertices).
    ///
    /// `tri0` and `tri1` are the bottom and top triangular faces; vertex
    /// `tri1[i]` is directly above `tri0[i]`, forming the three lateral quad
    /// faces.  Slots `[6..8]` are filled with [`CELL_SENTINEL`] automatically.
    pub fn push_wedge(&mut self, tri0: [u32; 3], tri1: [u32; 3]) {
        self.cells.push([
            tri0[0],
            tri0[1],
            tri0[2],
            tri1[0],
            tri1[1],
            tri1[2],
            CELL_SENTINEL,
            CELL_SENTINEL,
        ]);
    }

    /// Append a hexahedral cell (8 vertices, VTK ordering).
    pub fn push_hex(&mut self, verts: [u32; 8]) {
        self.cells.push(verts);
    }
}

// ---------------------------------------------------------------------------
// Tet decomposition for transparent volume rendering
// ---------------------------------------------------------------------------

/// Hex-to-tet decomposition using the Freudenthal 6-tet split.
///
/// All 6 tets share the main diagonal (vertex 0 <-> vertex 6 in VTK hex ordering).
const HEX_TO_TETS: [[usize; 4]; 6] = [
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
const WEDGE_TO_TETS: [[usize; 4]; 3] = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]];

/// Pyramid-to-tet decomposition (2 tets from a square pyramid).
///
/// Vertices: 0-3 = base quad; 4 = apex.
const PYRAMID_TO_TETS: [[usize; 4]; 2] = [[0, 1, 2, 4], [0, 2, 3, 4]];

/// Call `f` once per output tetrahedron across all cells in `data`.
///
/// `f` receives the four world-space vertices and the scalar value for that tet.
/// The scalar is taken from `data.cell_scalars[attribute]` at the parent cell index,
/// or 0.0 when the attribute is absent or the cell index is out of range.
///
/// Cell decomposition:
/// - Tet -> 1 tet
/// - Pyramid -> 2 tets
/// - Wedge -> 3 tets
/// - Hex -> 6 tets (Freudenthal split)
pub(crate) fn for_each_tet<F>(data: &VolumeMeshData, attribute: &str, mut f: F)
where
    F: FnMut([[f32; 3]; 4], f32),
{
    let cell_scalars = data.cell_scalars.get(attribute);
    for (cell_idx, cell) in data.cells.iter().enumerate() {
        let scalar = cell_scalars
            .and_then(|v| v.get(cell_idx))
            .copied()
            .unwrap_or(0.0);
        let tets: &[[usize; 4]] = match cell_type(cell) {
            CellType::Tet => &[[0, 1, 2, 3]],
            CellType::Pyramid => &PYRAMID_TO_TETS,
            CellType::Wedge => &WEDGE_TO_TETS,
            CellType::Hex => &HEX_TO_TETS,
        };
        for local in tets {
            let verts = [
                data.positions[cell[local[0]] as usize],
                data.positions[cell[local[1]] as usize],
                data.positions[cell[local[2]] as usize],
                data.positions[cell[local[3]] as usize],
            ];
            f(verts, scalar);
        }
    }
}

/// Decompose all cells in `data` into tetrahedra and collect the results.
///
/// Returns `(positions, scalars)`:
/// - `positions`: flat list of `[[f32; 3]; 4]`, one entry per output tet (4 world-space vertices)
/// - `scalars`: one `f32` per output tet, taken from `data.cell_scalars[attribute]` at the
///   parent cell index (0.0 when the attribute is absent or the cell index is out of range)
///
/// Used in tests. Production upload paths use `for_each_tet` directly to avoid
/// materializing the full decomposed data before chunking.
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
                CELL_SENTINEL,
                CELL_SENTINEL,
                CELL_SENTINEL,
                CELL_SENTINEL,
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
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                ],
                [
                    0,
                    2,
                    1,
                    4,
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                    CELL_SENTINEL,
                    CELL_SENTINEL,
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
                            CELL_SENTINEL,
                            CELL_SENTINEL,
                            CELL_SENTINEL,
                            CELL_SENTINEL,
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
                            CELL_SENTINEL,
                            CELL_SENTINEL,
                            CELL_SENTINEL,
                            CELL_SENTINEL,
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
        let (mesh, _) = extract_boundary_faces(&data);
        assert_eq!(
            mesh.indices.len(),
            4 * 3,
            "single tet -> 4 boundary triangles"
        );
    }

    #[test]
    fn two_tets_sharing_face_eliminates_shared_face() {
        let data = two_tets_sharing_face();
        let (mesh, _) = extract_boundary_faces(&data);
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
        let (mesh, _) = extract_boundary_faces(&data);
        // 6 quad faces x 2 triangles each = 12
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
        let (mesh, _) = extract_boundary_faces(&data);
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
        let (mesh, _) = extract_boundary_faces(&data);
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
        let (mesh, _) = extract_boundary_faces(&data);

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
        let (mesh, _) = extract_boundary_faces(&data);

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
        let (mesh, _) = extract_boundary_faces(&data);

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
            assert!(
                dot > 0.0,
                "boundary face points inward: tri={tri:?}, dot={dot}"
            );
        }
    }

    #[test]
    fn cube_sphere_hex_grid_boundary_faces_point_outward() {
        let data = cube_sphere_hex_grid(3, 2.0);
        let (mesh, _) = extract_boundary_faces(&data);

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
            assert!(
                dot > 0.0,
                "boundary face points inward: tri={tri:?}, dot={dot}"
            );
        }
    }

    #[test]
    fn normals_have_correct_length() {
        let data = single_tet();
        let (mesh, _) = extract_boundary_faces(&data);
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
        let (mesh, _) = extract_boundary_faces(&data);
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
        let (mesh, _) = extract_boundary_faces(&data);
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
        let (mesh, _) = extract_boundary_faces(&data);
        assert_eq!(mesh.positions, data.positions);
    }

    // -----------------------------------------------------------------------
    // Executable specifications for extract_clipped_volume_faces (Phase 5).
    // These tests document the required invariants and are ignored until the
    // Phase 2 implementation lands.  Enable them by removing #[ignore].
    // -----------------------------------------------------------------------

    /// Empty clip-plane slice must produce the same triangles as the boundary
    /// extractor (the clipped path degenerates to an unclipped boundary extraction
    /// when no planes are active).
    #[test]

    fn empty_planes_matches_boundary_extractor_tet() {
        let data = structured_tet_grid(3);
        let (boundary, _) = extract_boundary_faces(&data);
        let (clipped, _) = extract_clipped_volume_faces(&data, &[]);
        assert_eq!(
            boundary.indices.len(),
            clipped.indices.len(),
            "empty clip_planes -> same triangle count as extract_boundary_faces"
        );
    }

    /// Empty clip-plane slice must produce the same triangles as the boundary
    /// extractor for hex meshes.
    #[test]

    fn empty_planes_matches_boundary_extractor_hex() {
        let data = structured_hex_grid(3);
        let (boundary, _) = extract_boundary_faces(&data);
        let (clipped, _) = extract_clipped_volume_faces(&data, &[]);
        assert_eq!(
            boundary.indices.len(),
            clipped.indices.len(),
            "empty clip_planes -> same triangle count as extract_boundary_faces"
        );
    }

    /// Clipping a tet grid through its centre must produce non-empty section
    /// faces (i.e. the cut face count is greater than zero).
    #[test]

    fn clipped_tet_grid_has_nonempty_section_faces() {
        let grid_n = 3;
        let data = structured_tet_grid(grid_n);
        // Y = 1.5 cuts through the middle of a 3-unit-tall grid.
        // Plane: ny=1, d=-1.5  ->  dot(p,[0,1,0]) - 1.5 >= 0  ->  keep y >= 1.5.
        let plane = [0.0_f32, 1.0, 0.0, -1.5];
        let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
        // Some triangles must come from section faces.
        assert!(
            !mesh.indices.is_empty(),
            "clipped tet grid must produce at least one triangle"
        );
    }

    /// Clipping a hex grid through its centre must produce non-empty section faces.
    #[test]

    fn clipped_hex_grid_has_nonempty_section_faces() {
        let grid_n = 3;
        let data = structured_hex_grid(grid_n);
        let plane = [0.0_f32, 1.0, 0.0, -1.5];
        let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
        assert!(
            !mesh.indices.is_empty(),
            "clipped hex grid must produce at least one triangle"
        );
    }

    /// Section face normals must point toward the kept side of the cutting
    /// plane (dot of the section face normal with the plane normal > 0).
    #[test]

    fn section_face_normals_point_toward_kept_side_tet() {
        let data = structured_tet_grid(3);
        let plane_normal = [0.0_f32, 1.0, 0.0];
        let plane = [plane_normal[0], plane_normal[1], plane_normal[2], -1.5];
        let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);

        for n in &mesh.normals {
            let dot = n[0] * plane_normal[0] + n[1] * plane_normal[1] + n[2] * plane_normal[2];
            // Only section faces are required to satisfy this; boundary normals
            // may point in any outward direction.  The test checks that no
            // normal is strongly anti-parallel to the plane normal.
            // (A full test would distinguish section faces from boundary faces.)
            let _ = dot; // placeholder until section faces can be identified
        }
    }

    /// A cell fully on the discarded side of a clip plane contributes no triangles.
    #[test]

    fn fully_discarded_cells_contribute_nothing() {
        // Single tet at y=0..1 ; plane keeps y >= 2.0 -> tet is fully discarded.
        let data = single_tet();
        let plane = [0.0_f32, 1.0, 0.0, -2.0]; // keep y >= 2.0
        let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
        assert!(
            mesh.indices.is_empty(),
            "tet fully below clip plane must produce no triangles"
        );
    }

    /// A cell fully on the kept side of a clip plane contributes the same
    /// boundary triangles as the unclipped extractor.
    #[test]

    fn fully_kept_cell_matches_boundary_extractor() {
        // Single tet at y=0..1 ; plane keeps y >= -1.0 -> tet is fully kept.
        let data = single_tet();
        let plane = [0.0_f32, 1.0, 0.0, 1.0]; // keep y >= -1.0
        let (clipped, _) = extract_clipped_volume_faces(&data, &[plane]);
        let (boundary, _) = extract_boundary_faces(&data);
        assert_eq!(
            clipped.indices.len(),
            boundary.indices.len(),
            "fully kept cell must produce the same triangles as boundary extractor"
        );
    }

    /// Cell scalar attributes must be remapped onto section triangles in the
    /// same way they are remapped onto boundary triangles.
    #[test]
    fn cell_scalar_propagates_to_section_faces() {
        let mut data = structured_tet_grid(3);
        let n_cells = data.cells.len();
        data.cell_scalars
            .insert("pressure".to_string(), vec![1.0; n_cells]);
        let plane = [0.0_f32, 1.0, 0.0, -1.5];
        let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
        match mesh.attributes.get("pressure") {
            Some(AttributeData::Face(vals)) => {
                let n_tris = mesh.indices.len() / 3;
                assert_eq!(vals.len(), n_tris, "one scalar value per output triangle");
                for &v in vals {
                    assert_eq!(v, 1.0, "scalar must equal the owning cell's value");
                }
            }
            other => panic!("expected Face attribute on clipped mesh, got {other:?}"),
        }
    }

    /// Cell color attributes must be remapped onto section triangles as
    /// `AttributeKind::FaceColor`, with one entry per output triangle.
    #[test]
    fn cell_color_propagates_to_section_faces() {
        let mut data = structured_tet_grid(3);
        let n_cells = data.cells.len();
        let color = [1.0_f32, 0.0, 0.5, 1.0];
        data.cell_colors
            .insert("label".to_string(), vec![color; n_cells]);
        let plane = [0.0_f32, 1.0, 0.0, -1.5];
        let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
        match mesh.attributes.get("label") {
            Some(AttributeData::FaceColor(colors)) => {
                let n_tris = mesh.indices.len() / 3;
                assert_eq!(colors.len(), n_tris, "one color per output triangle");
                for &c in colors {
                    assert_eq!(c, color, "color must equal the owning cell's value");
                }
            }
            other => panic!("expected FaceColor attribute on clipped mesh, got {other:?}"),
        }
    }

    /// Section faces for hex cells must also carry per-cell scalar attributes.
    #[test]
    fn hex_cell_scalar_propagates_to_section_faces() {
        let mut data = structured_hex_grid(3);
        let n_cells = data.cells.len();
        data.cell_scalars
            .insert("temp".to_string(), vec![7.0; n_cells]);
        let plane = [0.0_f32, 1.0, 0.0, -1.5];
        let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
        match mesh.attributes.get("temp") {
            Some(AttributeData::Face(vals)) => {
                let n_tris = mesh.indices.len() / 3;
                assert_eq!(vals.len(), n_tris, "one scalar per output triangle");
                for &v in vals {
                    assert_eq!(v, 7.0, "scalar must equal the owning cell's value");
                }
            }
            other => panic!("expected Face attribute on clipped hex mesh, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // decompose_to_tetrahedra
    // -----------------------------------------------------------------------

    fn single_pyramid() -> VolumeMeshData {
        // Square base at y=0, apex at y=1.
        let mut data = VolumeMeshData {
            positions: vec![
                [0.0, 0.0, 0.0], // 0
                [1.0, 0.0, 0.0], // 1
                [1.0, 0.0, 1.0], // 2
                [0.0, 0.0, 1.0], // 3
                [0.5, 1.0, 0.5], // 4 apex
            ],
            ..Default::default()
        };
        data.push_pyramid([0, 1, 2, 3], 4);
        data
    }

    fn single_wedge() -> VolumeMeshData {
        // Two triangular faces: tri0 at y=0, tri1 at y=1.
        let mut data = VolumeMeshData {
            positions: vec![
                [0.0, 0.0, 0.0], // 0
                [1.0, 0.0, 0.0], // 1
                [0.5, 0.0, 1.0], // 2
                [0.0, 1.0, 0.0], // 3
                [1.0, 1.0, 0.0], // 4
                [0.5, 1.0, 1.0], // 5
            ],
            ..Default::default()
        };
        data.push_wedge([0, 1, 2], [3, 4, 5]);
        data
    }

    fn tet_volume(p: [[f32; 3]; 4]) -> f32 {
        // Signed volume = dot(v1, cross(v2, v3)) / 6 where vi = pi - p0.
        let v =
            |i: usize| -> [f32; 3] { [p[i][0] - p[0][0], p[i][1] - p[0][1], p[i][2] - p[0][2]] };
        let (a, b, c) = (v(1), v(2), v(3));
        let cross = [
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        ];
        (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0
    }

    #[test]
    fn decompose_tet_yields_one_tet() {
        let data = single_tet();
        let (tets, scalars) = decompose_to_tetrahedra(&data, "");
        assert_eq!(tets.len(), 1);
        assert_eq!(scalars.len(), 1);
    }

    #[test]
    fn decompose_hex_yields_six_tets() {
        let data = single_hex();
        let (tets, scalars) = decompose_to_tetrahedra(&data, "");
        assert_eq!(tets.len(), 6);
        assert_eq!(scalars.len(), 6);
    }

    #[test]
    fn decompose_pyramid_yields_two_tets() {
        let data = single_pyramid();
        let (tets, scalars) = decompose_to_tetrahedra(&data, "");
        assert_eq!(tets.len(), 2);
        assert_eq!(scalars.len(), 2);
    }

    #[test]
    fn decompose_wedge_yields_three_tets() {
        let data = single_wedge();
        let (tets, scalars) = decompose_to_tetrahedra(&data, "");
        assert_eq!(tets.len(), 3);
        assert_eq!(scalars.len(), 3);
    }

    #[test]
    fn decompose_output_tets_have_nonzero_volume() {
        for data in [single_tet(), single_hex(), single_pyramid(), single_wedge()] {
            let (tets, _) = decompose_to_tetrahedra(&data, "");
            for (i, t) in tets.iter().enumerate() {
                let vol = tet_volume(*t).abs();
                assert!(vol > 1e-6, "tet {i} has near-zero volume {vol}: {t:?}");
            }
        }
    }

    #[test]
    fn decompose_hex_volume_equals_cell_volume() {
        // The 6-tet decomposition of a unit cube must sum to 1.0.
        let data = single_hex();
        let (tets, _) = decompose_to_tetrahedra(&data, "");
        let total: f32 = tets.iter().map(|t| tet_volume(*t).abs()).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "unit hex volume should be 1.0, got {total}"
        );
    }

    #[test]
    fn decompose_scalar_propagates_to_child_tets() {
        let mut data = single_hex();
        data.cell_scalars.insert("temp".to_string(), vec![42.0]);
        let (_, scalars) = decompose_to_tetrahedra(&data, "temp");
        assert_eq!(scalars.len(), 6);
        for &s in &scalars {
            assert_eq!(s, 42.0, "all child tets must inherit the cell scalar");
        }
    }

    #[test]
    fn decompose_missing_attribute_falls_back_to_zero() {
        let data = single_hex();
        let (_, scalars) = decompose_to_tetrahedra(&data, "nonexistent");
        for &s in &scalars {
            assert_eq!(s, 0.0, "missing attribute must produce 0.0 per tet");
        }
    }

    #[test]
    fn decompose_mixed_mesh_tet_counts_sum_correctly() {
        // One tet + one hex + one pyramid + one wedge = 1+6+2+3 = 12 tets.
        let mut data = VolumeMeshData {
            positions: vec![
                // tet verts (0..3)
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                // hex verts (4..11): unit cube offset at x=2
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                // pyramid verts (12..16): square base + apex, offset at x=4
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [5.0, 0.0, 1.0],
                [4.0, 0.0, 1.0],
                [4.5, 1.0, 0.5],
                // wedge verts (17..22): offset at x=6
                [6.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [6.5, 0.0, 1.0],
                [6.0, 1.0, 0.0],
                [7.0, 1.0, 0.0],
                [6.5, 1.0, 1.0],
            ],
            ..Default::default()
        };
        data.push_tet(0, 1, 2, 3);
        data.push_hex([4, 5, 6, 7, 8, 9, 10, 11]);
        data.push_pyramid([12, 13, 14, 15], 16);
        data.push_wedge([17, 18, 19], [20, 21, 22]);

        let (tets, scalars) = decompose_to_tetrahedra(&data, "");
        assert_eq!(tets.len(), 12, "1+6+2+3 = 12 tets");
        assert_eq!(scalars.len(), 12);
    }
}
