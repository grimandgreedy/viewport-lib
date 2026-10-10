//! Boundary extraction: the faces of a volume mesh that belong to exactly one cell.

use super::cells::{
    CellType, HEX_FACES, PYRAMID_QUAD_FACE, PYRAMID_TRI_FACES, TET_FACES, WEDGE_QUAD_FACES,
    WEDGE_TRI_FACES, boundary_interior_ref, cell_type,
};
use super::{PARALLEL_THRESHOLD, VolumeMeshData};
use crate::util::par::*;
use std::collections::HashMap;
use viewport_lib_types::data::{attribute::AttributeData, mesh::MeshData};

/// A canonical (sorted) face key used for boundary detection.
pub(super) type FaceKey = (u32, u32, u32);

/// Canonical key for a quad face, sorted by vertex index.
pub(super) type QuadFaceKey = (u32, u32, u32, u32);

// (sorted_key, cell_idx, winding). The interior reference point for winding
// correction is NOT carried here: it is only needed for the ~few-percent of
// faces that turn out to be boundary faces, so it is computed lazily from the
// owning cell after boundary detection (see `extract_boundary_faces`). Carrying
// it per face would compute it for every interior face too and fatten the
// sorted entry, both pure waste at scale.
pub(super) type TriEntry = (FaceKey, usize, [u32; 3]);
pub(super) type QuadEntry = (QuadFaceKey, usize, [u32; 4]);

/// Build a sorted key from three vertex indices.
#[inline]
pub(super) fn face_key(a: u32, b: u32, c: u32) -> FaceKey {
    let mut arr = [a, b, c];
    arr.sort_unstable();
    (arr[0], arr[1], arr[2])
}

/// Build a sorted key from four vertex indices.
#[inline]
pub(super) fn quad_face_key(a: u32, b: u32, c: u32, d: u32) -> QuadFaceKey {
    let mut arr = [a, b, c, d];
    arr.sort_unstable();
    (arr[0], arr[1], arr[2], arr[3])
}

/// Maximum triangular faces a single cell can contribute (tet/pyramid = 4).
pub(super) const MAX_TRI_FACES: usize = 4;
/// Maximum quad faces a single cell can contribute (hex = 6).
pub(super) const MAX_QUAD_FACES: usize = 6;

/// Generate all triangular face entries for a single cell as a stack-bounded
/// iterator.
///
/// A cell contributes at most [`MAX_TRI_FACES`] triangular faces, so the entries
/// live in a fixed-size array on the stack rather than a fresh heap `Vec` per
/// cell: at 1M cells the per-cell `Vec` was 1M-6M throwaway allocations on the
/// extraction critical path. The returned iterator is consumed the same way a
/// `Vec` would be (`flat_map_iter` / `extend` only need `IntoIterator`), so the
/// output and its order are unchanged.
#[inline]
pub(super) fn generate_tri_entries(
    cell_idx: usize,
    cell: &[u32; 8],
) -> impl Iterator<Item = TriEntry> {
    let mut out = [((0, 0, 0), 0usize, [0u32; 3]); MAX_TRI_FACES];
    let mut len = 0;
    let mut push_tri = |a: u32, b: u32, c: u32| {
        out[len] = (face_key(a, b, c), cell_idx, [a, b, c]);
        len += 1;
    };
    let faces: &[[usize; 3]] = match cell_type(cell) {
        CellType::Tet => &TET_FACES,
        CellType::Pyramid => &PYRAMID_TRI_FACES,
        CellType::Wedge => &WEDGE_TRI_FACES,
        CellType::Hex => &[], // hex has no triangular faces
    };
    for face_local in faces {
        push_tri(
            cell[face_local[0]],
            cell[face_local[1]],
            cell[face_local[2]],
        );
    }
    out.into_iter().take(len)
}

/// Generate all quad face entries for a single cell as a stack-bounded iterator.
///
/// Same rationale as [`generate_tri_entries`]: at most [`MAX_QUAD_FACES`] quads per cell,
/// held in a stack array so extraction pays no per-cell heap allocation.
#[inline]
pub(super) fn generate_quad_entries(
    cell_idx: usize,
    cell: &[u32; 8],
) -> impl Iterator<Item = QuadEntry> {
    let mut out = [((0, 0, 0, 0), 0usize, [0u32; 4]); MAX_QUAD_FACES];
    let mut len = 0;
    let mut push_quad = |q: &[usize; 4]| {
        let v = [cell[q[0]], cell[q[1]], cell[q[2]], cell[q[3]]];
        out[len] = (quad_face_key(v[0], v[1], v[2], v[3]), cell_idx, v);
        len += 1;
    };
    let faces: &[[usize; 4]] = match cell_type(cell) {
        CellType::Tet => &[], // tet has no quad faces
        CellType::Pyramid => &PYRAMID_QUAD_FACE,
        CellType::Wedge => &WEDGE_QUAD_FACES,
        CellType::Hex => &HEX_FACES,
    };
    for quad_local in faces {
        push_quad(quad_local);
    }
    out.into_iter().take(len)
}

/// Collect entries that appear exactly once (boundary faces) from a sorted slice.
pub(super) fn collect_boundary_tri(entries: &[TriEntry]) -> Vec<(usize, [u32; 3])> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let key = entries[i].0;
        let mut j = i + 1;
        while j < entries.len() && entries[j].0 == key {
            j += 1;
        }
        if j - i == 1 {
            out.push((entries[i].1, entries[i].2));
        }
        i = j;
    }
    out
}

/// Collect quad entries that appear exactly once (boundary faces) from a sorted slice.
pub(super) fn collect_boundary_quad(entries: &[QuadEntry]) -> Vec<(usize, [u32; 4])> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let key = entries[i].0;
        let mut j = i + 1;
        while j < entries.len() && entries[j].0 == key {
            j += 1;
        }
        if j - i == 1 {
            out.push((entries[i].1, entries[i].2));
        }
        i = j;
    }
    out
}

/// Ensure the triangle winding produces an outward-facing normal relative to
/// `interior_ref` (a point inside the owning cell).
#[inline]
pub(super) fn correct_winding(tri: &mut [u32; 3], interior_ref: &[f32; 3], positions: &[[f32; 3]]) {
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
/// After this step the boundary mesh is uploaded via
/// `DeviceResources::upload_mesh_data` and rendered exactly like any other
/// surface mesh.
///
/// Returns `(mesh_data, face_to_cell)` where `face_to_cell[i]` is the cell
/// index that boundary triangle `i` belongs to.
#[doc(hidden)]
pub fn extract_boundary_faces(data: &VolumeMeshData) -> (MeshData, Vec<u32>) {
    let n_verts = data.positions.len();

    // Generate face entries (parallel above threshold, sequential below). Each
    // cell yields a stack-bounded iterator (no per-cell heap allocation); the
    // final order is irrelevant since both lists are sorted by face key next.
    let (mut tri_entries, mut quad_entries) = if data.cells.len() >= PARALLEL_THRESHOLD {
        let tri = data
            .cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(ci, cell)| generate_tri_entries(ci, cell))
            .collect();
        let quad = data
            .cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(ci, cell)| generate_quad_entries(ci, cell))
            .collect();
        (tri, quad)
    } else {
        let mut tri: Vec<TriEntry> = Vec::new();
        let mut quad: Vec<QuadEntry> = Vec::new();
        for (ci, cell) in data.cells.iter().enumerate() {
            tri.extend(generate_tri_entries(ci, cell));
            quad.extend(generate_quad_entries(ci, cell));
        }
        (tri, quad)
    };

    tri_entries.par_sort_unstable_by_key(|e| e.0);
    quad_entries.par_sort_unstable_by_key(|e| e.0);

    // Collect boundary faces (count == 1) via linear scan.
    let mut boundary: Vec<(usize, [u32; 3])> = collect_boundary_tri(&tri_entries);
    for (ci, winding) in collect_boundary_quad(&quad_entries) {
        boundary.push((ci, [winding[0], winding[1], winding[2]]));
        boundary.push((ci, [winding[0], winding[2], winding[3]]));
    }

    // Sort by cell index for deterministic output (useful for testing).
    boundary.sort_unstable_by_key(|(ci, _)| *ci);

    // Geometric winding correction (parallel): ensure each boundary face's normal
    // points outward. This is the primary correctness mechanism for tets where
    // the table winding may be inward. The interior reference is the owning
    // cell's centroid, computed here for the few boundary faces rather than for
    // every face during generation.
    boundary.par_iter_mut().for_each(|(ci, tri)| {
        let cell = &data.cells[*ci];
        let iref = boundary_interior_ref(cell, tri, &data.positions);
        correct_winding(tri, &iref, &data.positions);
    });

    let n_boundary_tris = boundary.len();

    // Build index buffer and accumulate area-weighted normals (sequential:
    // normal_accum has shared per-vertex writes).
    let mut indices: Vec<u32> = Vec::with_capacity(n_boundary_tris * 3);
    let mut normal_accum: Vec<[f64; 3]> = vec![[0.0; 3]; n_verts];

    for (_, tri) in &boundary {
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
            .map(|(ci, _)| cell_vals.get(*ci).copied().unwrap_or(0.0))
            .collect();
        attributes.insert(name.clone(), AttributeData::Face(face_scalars));
    }

    for (name, cell_vals) in &data.cell_colours {
        let face_colours: Vec<[f32; 4]> = boundary
            .iter()
            .map(|(ci, _)| cell_vals.get(*ci).copied().unwrap_or([1.0; 4]))
            .collect();
        attributes.insert(name.clone(), AttributeData::FaceColour(face_colours));
    }

    // The surface is built over the volume's own vertex list, so a node array
    // is already in its vertex order.
    for (name, node_vals) in &data.node_scalars {
        if attributes.contains_key(name) {
            continue;
        }
        let mut values = node_vals.clone();
        values.resize(n_verts, 0.0);
        attributes.insert(name.clone(), AttributeData::Vertex(values));
    }

    let face_to_cell: Vec<u32> = boundary.iter().map(|(ci, _)| *ci as u32).collect();

    (
        {
            let mut m = MeshData::default();
            m.positions = data.positions.clone();
            m.normals = normals;
            m.indices = indices;
            m.uvs = None;
            m.tangents = None;
            m.vertex_colours = None;
            m.attributes = attributes;
            m.extension_attributes = None;
            m.submeshes = Vec::new();
            m
        },
        face_to_cell,
    )
}
