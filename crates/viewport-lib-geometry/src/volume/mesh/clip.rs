//! Clipped extraction: the boundary of the part of a volume mesh on the kept side
//! of a set of planes, with the cut faces filled in.

use super::boundary::{
    MAX_QUAD_FACES, MAX_TRI_FACES, QuadEntry, TriEntry, collect_boundary_quad,
    collect_boundary_tri, correct_winding, extract_boundary_faces, generate_quad_entries,
    generate_tri_entries,
};
use super::cells::{boundary_interior_ref, cell_type};
use super::tets::cell_tets;
use super::{PARALLEL_THRESHOLD, VolumeMeshData};
use crate::maths::vec3::{cross3, dot3, normalize3};
use crate::util::par::*;
use std::collections::HashMap;
use viewport_lib_types::data::{attribute::AttributeData, mesh::MeshData};

//
// Design note: scope and invariants
// ==================================
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
// For volume meshes that is wrong: it produces a slab with no per-cell colour
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
// Section triangles inherit the owning cell's `cell_scalars` and `cell_colours`
// values exactly as boundary triangles do.  The output `MeshData` uses the same
// `AttributeKind::Face` / `AttributeKind::FaceColour` paths, so colourmaps work
// with no changes to the renderer.
//
// ## Output type
//
// The function returns an ordinary `MeshData`.  No new intermediate type is
// introduced.  The caller uploads this as a regular mesh and renders it with
// the standard pipeline; the only renderer-side requirement is that cap-fill
// is disabled for the same scene object.

/// Signed distance from `p` to `plane` (`[nx, ny, nz, d]`).
/// Positive means on the kept side (`dot(p, n) + d >= 0`).
#[inline]
pub(super) fn plane_dist(p: [f32; 3], plane: [f32; 4]) -> f32 {
    p[0] * plane[0] + p[1] * plane[1] + p[2] * plane[2] + plane[3]
}

/// Intern `p` into `positions`, returning its index.
/// Uses bit-exact comparison so the same floating-point value always maps to
/// the same slot.
pub(super) fn intern_pos(
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
pub(super) fn clip_polygon_one_plane(poly: Vec<[f32; 3]>, plane: [f32; 4]) -> Vec<[f32; 3]> {
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
pub(super) fn clip_polygon_planes(mut poly: Vec<[f32; 3]>, planes: &[[f32; 4]]) -> Vec<[f32; 3]> {
    for &plane in planes {
        if poly.is_empty() {
            break;
        }
        poly = clip_polygon_one_plane(poly, plane);
    }
    poly
}

/// Build an orthonormal `(u, v)` basis for a plane with the given `normal`.
pub(super) fn plane_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
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
pub(super) fn sort_polygon_on_plane(pts: &mut Vec<[f32; 3]>, normal: [f32; 3]) {
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
pub(super) fn fan_triangulate(poly: &[[f32; 3]]) -> Vec<[[f32; 3]; 3]> {
    if poly.len() < 3 {
        return Vec::new();
    }
    (1..poly.len() - 1)
        .map(|i| [poly[0], poly[i], poly[i + 1]])
        .collect()
}

/// Barycentric coordinates of `p` in the tet `v`, or `None` when the tet has
/// no volume.
pub(super) fn tet_barycentric(p: [f32; 3], v: [[f32; 3]; 4]) -> Option<[f32; 4]> {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (e1, e2, e3) = (sub(v[1], v[0]), sub(v[2], v[0]), sub(v[3], v[0]));
    let det = dot3(e1, cross3(e2, e3));
    if det.abs() < 1e-20 {
        return None;
    }
    let d = sub(p, v[0]);
    let b1 = dot3(d, cross3(e2, e3)) / det;
    let b2 = dot3(e1, cross3(d, e3)) / det;
    let b3 = dot3(e1, cross3(e2, d)) / det;
    Some([1.0 - b1 - b2 - b3, b1, b2, b3])
}

/// Where a node value at `p` comes from inside `cell`: four vertices and
/// their weights, which sum to one.
///
/// The cell is split into the tets the transparent mode uses and `p` is
/// interpolated linearly in the one that contains it. A point on a cell edge
/// therefore takes the value linear along that edge, which is where every
/// cut vertex of a section starts.
pub(super) fn node_weights_in_cell(
    cell: &[u32; 8],
    p: [f32; 3],
    positions: &[[f32; 3]],
) -> ([u32; 4], [f32; 4]) {
    let mut best: Option<(f32, [u32; 4], [f32; 4])> = None;
    for tet in cell_tets(cell) {
        let ids = tet.map(|slot| cell[slot]);
        let Some(bary) = tet_barycentric(p, ids.map(|i| positions[i as usize])) else {
            continue;
        };
        // The containing tet is the one whose smallest coordinate is largest.
        let inside = bary.iter().copied().fold(f32::INFINITY, f32::min);
        if best.as_ref().is_none_or(|(b, _, _)| inside > *b) {
            best = Some((inside, ids, bary));
        }
    }
    match best {
        Some((_, ids, bary)) => {
            // Rounding can leave a point just outside; pull it back in.
            let clamped = bary.map(|b| b.max(0.0));
            let sum: f32 = clamped.iter().sum();
            (ids, clamped.map(|b| b / sum))
        }
        // Every tet is flat: fall back to the cell's first corner.
        None => ([cell[0]; 4], [1.0, 0.0, 0.0, 0.0]),
    }
}

/// Generate section triangles for a single intersected cell across all clip planes.
pub(super) fn generate_section_tris(
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

/// Extract boundary and section faces from a volume mesh clipped by one or
/// more planes.
///
/// Each entry in `clip_planes` is `[nx, ny, nz, d]` where a point `p` is on
/// the kept side when `dot(p, [nx,ny,nz]) + d >= 0`.  This is the same
/// encoding as `viewport-lib`'s clip-plane uniform, so values can be forwarded
/// to both the CPU path and the GPU clip shader.
///
/// Passing an empty slice returns the same result as [`extract_boundary_faces`].
/// See the design note in the section comment above for the full contract.
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
/// normal).  Per-cell scalar and colour attributes are propagated to section
/// triangles identically to boundary triangles.
///
/// # Renderer contract
///
/// Generic cap-fill must be disabled for scene objects rendered via this path.
/// Section faces are generated here from cell data; the generic cap overlay
/// does not have access to per-cell attribute information and would produce an
/// incorrect result if left enabled.
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

    // Generate face entries, skipping fully-discarded cells. Each cell yields a
    // stack-bounded iterator (no per-cell heap allocation); a discarded cell
    // yields nothing via an outer `take(0)` so every closure branch keeps the
    // same iterator type. Order is irrelevant since entries are sorted next.
    let cell_kept = |cell: &[u32; 8]| -> bool {
        let nv = cell_type(cell).vertex_count();
        (0..nv).any(|i| vert_kept[cell[i] as usize])
    };
    let (mut tri_entries, mut quad_entries) = if data.cells.len() >= PARALLEL_THRESHOLD {
        let tri = data
            .cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(ci, cell)| {
                let cap = if cell_kept(cell) { MAX_TRI_FACES } else { 0 };
                generate_tri_entries(ci, cell).take(cap)
            })
            .collect();
        let quad = data
            .cells
            .par_iter()
            .enumerate()
            .flat_map_iter(|(ci, cell)| {
                let cap = if cell_kept(cell) { MAX_QUAD_FACES } else { 0 };
                generate_quad_entries(ci, cell).take(cap)
            })
            .collect();
        (tri, quad)
    } else {
        let mut tri: Vec<TriEntry> = Vec::new();
        let mut quad: Vec<QuadEntry> = Vec::new();
        for (ci, cell) in data.cells.iter().enumerate() {
            if !cell_kept(cell) {
                continue;
            }
            tri.extend(generate_tri_entries(ci, cell));
            quad.extend(generate_quad_entries(ci, cell));
        }
        (tri, quad)
    };

    tri_entries.par_sort_unstable_by_key(|e| e.0);
    quad_entries.par_sort_unstable_by_key(|e| e.0);

    let mut boundary: Vec<(usize, [u32; 3])> = collect_boundary_tri(&tri_entries);
    for (ci, winding) in collect_boundary_quad(&quad_entries) {
        boundary.push((ci, [winding[0], winding[1], winding[2]]));
        boundary.push((ci, [winding[0], winding[2], winding[3]]));
    }
    boundary.sort_unstable_by_key(|(ci, _)| *ci);

    boundary.par_iter_mut().for_each(|(ci, tri)| {
        let cell = &data.cells[*ci];
        let iref = boundary_interior_ref(cell, tri, &data.positions);
        correct_winding(tri, &iref, &data.positions);
    });

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
        .flat_map_iter(|(cell_idx, tri)| {
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

    // The cell that produced each vertex made by a cut, in the order those
    // vertices were appended after the volume's own. A node scalar is
    // evaluated there by interpolating inside that cell.
    let n_original = data.positions.len();
    let mut cut_vertex_cell: Vec<usize> = Vec::new();

    let mut indexed_tris: Vec<(usize, [u32; 3])> = Vec::with_capacity(out_tris.len());
    for (cell_idx, tri) in &out_tris {
        let mut idx = [0u32; 3];
        for (slot, p) in tri.iter().enumerate() {
            idx[slot] = intern_pos(*p, &mut positions, &mut pos_map);
            if positions.len() > n_original + cut_vertex_cell.len() {
                cut_vertex_cell.push(*cell_idx);
            }
        }
        indexed_tris.push((*cell_idx, idx));
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
    for (name, cell_vals) in &data.cell_colours {
        let face_colours: Vec<[f32; 4]> = indexed_tris
            .iter()
            .map(|(ci, _)| cell_vals.get(*ci).copied().unwrap_or([1.0; 4]))
            .collect();
        attributes.insert(name.clone(), AttributeData::FaceColour(face_colours));
    }

    // Node scalars: the volume's own vertices keep their values, and each
    // vertex made by a cut takes the value interpolated inside its cell.
    if !data.node_scalars.is_empty() {
        let cut_sources: Vec<([u32; 4], [f32; 4])> = cut_vertex_cell
            .iter()
            .enumerate()
            .map(|(i, &ci)| {
                node_weights_in_cell(&data.cells[ci], positions[n_original + i], &data.positions)
            })
            .collect();
        for (name, node_vals) in &data.node_scalars {
            if attributes.contains_key(name) {
                continue;
            }
            let at = |vi: u32| node_vals.get(vi as usize).copied().unwrap_or(0.0);
            let mut values = node_vals.clone();
            values.resize(n_original, 0.0);
            values.extend(
                cut_sources
                    .iter()
                    .map(|(ids, w)| (0..4).map(|k| w[k] * at(ids[k])).sum::<f32>()),
            );
            attributes.insert(name.clone(), AttributeData::Vertex(values));
        }
    }

    let face_to_cell: Vec<u32> = indexed_tris.iter().map(|(ci, _)| *ci as u32).collect();

    (
        {
            let mut m = MeshData::new(positions, normals, indices);
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
