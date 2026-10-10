//! CPU-side marching cubes isosurface extraction from volumetric scalar data.
//!
//! The output is a standard [`MeshData`] that `viewport-lib` can upload via
//! `DeviceResources::upload_mesh_data` or `replace_mesh_data`.
//!
//! # Example
//!
//! ```ignore
//! let volume = VolumeData {
//!     data: vec![/* scalar values */],
//!     dims: [32, 32, 32],
//!     origin: [0.0, 0.0, 0.0],
//!     spacing: [0.1, 0.1, 0.1],
//! };
//! let mesh = extract_isosurface(&volume, 0.5);
//! // mesh.positions, mesh.normals, mesh.indices ready for upload.
//! ```

mod tables;

use crate::util::par::*;
use crate::volume::grid::{VolumeData, trilinear_sample};
use std::collections::HashMap;
use tables::{EDGE_TABLE, EDGE_VERTICES};
use viewport_lib_types::data::mesh::MeshData;

pub use tables::TRI_TABLE;

/// Compute the gradient at a world-space position via central differences.
fn gradient_at(volume: &VolumeData, pos: [f32; 3]) -> [f32; 3] {
    let hx = volume.spacing[0] * 0.5;
    let hy = volume.spacing[1] * 0.5;
    let hz = volume.spacing[2] * 0.5;

    let gx = trilinear_sample(volume, [pos[0] + hx, pos[1], pos[2]])
        - trilinear_sample(volume, [pos[0] - hx, pos[1], pos[2]]);
    let gy = trilinear_sample(volume, [pos[0], pos[1] + hy, pos[2]])
        - trilinear_sample(volume, [pos[0], pos[1] - hy, pos[2]]);
    let gz = trilinear_sample(volume, [pos[0], pos[1], pos[2] + hz])
        - trilinear_sample(volume, [pos[0], pos[1], pos[2] - hz]);

    let len = (gx * gx + gy * gy + gz * gz).sqrt();
    if len > 1e-10 {
        [gx / len, gy / len, gz / len]
    } else {
        [0.0, 1.0, 0.0] // fallback normal
    }
}

/// Extract an isosurface from a volume at the given `isovalue` using marching cubes.
///
/// Returns a [`MeshData`] with positions, normals (from volume gradient), and triangle
/// indices. The mesh can be uploaded to the viewport via the standard mesh pipeline.
///
/// For volumes larger than 64x64x64 cells the extraction is parallelised via
/// Z-slab decomposition. Edges on slab boundaries are independently interpolated
/// by each adjacent slab, producing a small number of geometrically coincident
/// but topologically disconnected vertices along each Z-boundary. For rendering
/// this is invisible.
pub fn extract_isosurface(volume: &VolumeData, isovalue: f32) -> MeshData {
    let [nx, ny, nz] = volume.dims;
    if nx < 2 || ny < 2 || nz < 2 {
        return {
            let mut m = MeshData::default();
            m.positions = Vec::new();
            m.normals = Vec::new();
            m.indices = Vec::new();
            m.uvs = None;
            m.tangents = None;
            m.vertex_colours = None;
            m.attributes = HashMap::new();
            m.extension_attributes = None;
            m.submeshes = Vec::new();
            m
        };
    }

    const SLAB_THRESHOLD: usize = 64 * 64 * 64;
    let cell_count = (nx - 1) as usize * (ny - 1) as usize * (nz - 1) as usize;

    let (positions, normals, indices) = if cell_count >= SLAB_THRESHOLD {
        // Parallel Z-slab decomposition: divide the nz-1 cell layers into
        // independent slabs, each with its own edge cache. Slab outputs are
        // concatenated with adjusted index offsets after the parallel phase.
        let nz_cells = (nz - 1) as usize;
        let num_threads = current_num_threads();
        let slab_height = (nz_cells / num_threads).max(1);

        let slabs: Vec<_> = (0..nz_cells)
            .step_by(slab_height)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&iz_start| {
                let iz_end = (iz_start + slab_height).min(nz_cells);
                extract_isosurface_slab(volume, isovalue, iz_start as u32, iz_end as u32)
            })
            .collect();

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for (slab_pos, slab_nrm, slab_idx) in slabs {
            let offset = positions.len() as u32;
            positions.extend_from_slice(&slab_pos);
            normals.extend_from_slice(&slab_nrm);
            indices.extend(slab_idx.into_iter().map(|i| i + offset));
        }
        (positions, normals, indices)
    } else {
        extract_isosurface_slab(volume, isovalue, 0, nz - 1)
    };

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = None;
    m.tangents = None;
    m.vertex_colours = None;
    m.attributes = HashMap::new();
    m.extension_attributes = None;
    m.submeshes = Vec::new();
    m
}

/// Process one Z-slab of cells: `iz_start..iz_end` (exclusive end).
fn extract_isosurface_slab(
    volume: &VolumeData,
    isovalue: f32,
    iz_start: u32,
    iz_end: u32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let [nx, ny, _nz] = volume.dims;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut edge_cache: HashMap<(u32, u32, u32, u8), u32> = HashMap::new();

    for iz in iz_start..iz_end {
        for iy in 0..(ny - 1) {
            for ix in 0..(nx - 1) {
                // 8 corner values in standard marching cubes corner order.
                let corners = [
                    volume.sample(ix, iy, iz),             // 0
                    volume.sample(ix + 1, iy, iz),         // 1
                    volume.sample(ix + 1, iy + 1, iz),     // 2
                    volume.sample(ix, iy + 1, iz),         // 3
                    volume.sample(ix, iy, iz + 1),         // 4
                    volume.sample(ix + 1, iy, iz + 1),     // 5
                    volume.sample(ix + 1, iy + 1, iz + 1), // 6
                    volume.sample(ix, iy + 1, iz + 1),     // 7
                ];

                // Compute 8-bit cube index.
                let mut cube_index = 0u8;
                for (i, &val) in corners.iter().enumerate() {
                    if val < isovalue {
                        cube_index |= 1 << i;
                    }
                }

                let edge_bits = EDGE_TABLE[cube_index as usize];
                if edge_bits == 0 {
                    continue;
                }

                // Corner world positions.
                let corner_pos = corner_positions(volume, ix, iy, iz);

                // For each triangle edge in TRI_TABLE, get or create vertex.
                let tri_row = &TRI_TABLE[cube_index as usize];
                let mut i = 0;
                while i < 16 && tri_row[i] >= 0 {
                    let mut tri_verts = [0u32; 3];
                    for v in 0..3 {
                        let edge_id = tri_row[i + v] as u8;
                        let (a, b) = EDGE_VERTICES[edge_id as usize];

                        // Canonical edge key: lower-corner cell + axis.
                        let cache_key = canonical_edge_key(ix, iy, iz, edge_id);

                        tri_verts[v] = *edge_cache.entry(cache_key).or_insert_with(|| {
                            let va = corners[a as usize];
                            let vb = corners[b as usize];
                            let t = if (va - vb).abs() > 1e-10 {
                                (isovalue - va) / (vb - va)
                            } else {
                                0.5
                            };
                            let t = t.clamp(0.0, 1.0);

                            let pa = corner_pos[a as usize];
                            let pb = corner_pos[b as usize];
                            let pos = [
                                pa[0] + t * (pb[0] - pa[0]),
                                pa[1] + t * (pb[1] - pa[1]),
                                pa[2] + t * (pb[2] - pa[2]),
                            ];

                            let normal = gradient_at(volume, pos);

                            let idx = positions.len() as u32;
                            positions.push(pos);
                            normals.push(normal);
                            idx
                        });
                    }
                    // Emit triangle (swap v1/v2 to match renderer's CCW front-face convention).
                    indices.push(tri_verts[0]);
                    indices.push(tri_verts[2]);
                    indices.push(tri_verts[1]);
                    i += 3;
                }
            }
        }
    }

    (positions, normals, indices)
}

/// Compute world-space positions for the 8 corners of a cell.
fn corner_positions(volume: &VolumeData, ix: u32, iy: u32, iz: u32) -> [[f32; 3]; 8] {
    let ox = volume.origin[0] + ix as f32 * volume.spacing[0];
    let oy = volume.origin[1] + iy as f32 * volume.spacing[1];
    let oz = volume.origin[2] + iz as f32 * volume.spacing[2];
    let dx = volume.spacing[0];
    let dy = volume.spacing[1];
    let dz = volume.spacing[2];

    [
        [ox, oy, oz],                // 0
        [ox + dx, oy, oz],           // 1
        [ox + dx, oy + dy, oz],      // 2
        [ox, oy + dy, oz],           // 3
        [ox, oy, oz + dz],           // 4
        [ox + dx, oy, oz + dz],      // 5
        [ox + dx, oy + dy, oz + dz], // 6
        [ox, oy + dy, oz + dz],      // 7
    ]
}

/// Canonical edge key for the vertex deduplication cache.
///
/// Shared edges between adjacent cells must produce the same key. We encode each
/// edge as the lower-index corner cell coordinate + the axis of the edge.
fn canonical_edge_key(cx: u32, cy: u32, cz: u32, edge_id: u8) -> (u32, u32, u32, u8) {
    // Each edge is shared by up to 4 cells. We pick the canonical owner as the cell
    // that has the smallest coordinates among all cells sharing this edge, encoding
    // the edge as (cell_x, cell_y, cell_z, local_edge_axis).
    //
    // Edge axis encoding:
    //   0-3: edges along X
    //   4-7: edges along Y
    //   8-11: edges along Z
    match edge_id {
        // X-axis edges
        0 => (cx, cy, cz, 0),         // edge 0: corner 0-1, bottom-front
        2 => (cx, cy + 1, cz, 0),     // edge 2: corner 3-2, top-front
        4 => (cx, cy, cz + 1, 0),     // edge 4: corner 4-5, bottom-back
        6 => (cx, cy + 1, cz + 1, 0), // edge 6: corner 7-6, top-back
        // Y-axis edges
        3 => (cx, cy, cz, 1),         // edge 3: corner 0-3, left-front
        1 => (cx + 1, cy, cz, 1),     // edge 1: corner 1-2, right-front
        7 => (cx, cy, cz + 1, 1),     // edge 7: corner 4-7, left-back
        5 => (cx + 1, cy, cz + 1, 1), // edge 5: corner 5-6, right-back
        // Z-axis edges
        8 => (cx, cy, cz, 2),          // edge 8: corner 0-4, bottom-left
        9 => (cx + 1, cy, cz, 2),      // edge 9: corner 1-5, bottom-right
        10 => (cx + 1, cy + 1, cz, 2), // edge 10: corner 2-6, top-right
        11 => (cx, cy + 1, cz, 2),     // edge 11: corner 3-7, top-left
        _ => (cx, cy, cz, edge_id),    // fallback (should not happen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_table_edge_consistency() {
        // Every edge index referenced in TRI_TABLE[i] must appear in EDGE_TABLE[i].
        let mut failures = Vec::new();
        for cube_index in 0u16..256 {
            let edge_bits = EDGE_TABLE[cube_index as usize];
            let tri_row = &TRI_TABLE[cube_index as usize];
            let mut j = 0;
            while j < 16 && tri_row[j] >= 0 {
                let edge_id = tri_row[j] as u8;
                if edge_bits & (1 << edge_id) == 0 {
                    failures.push(format!(
                        "TRI_TABLE[{}]: edge {} not in EDGE_TABLE ({:#014b})",
                        cube_index, edge_id, edge_bits
                    ));
                    break; // one failure per cube_index is enough
                }
                j += 1;
            }
        }
        if !failures.is_empty() {
            panic!(
                "{} TRI_TABLE entries inconsistent with EDGE_TABLE:\n{}",
                failures.len(),
                failures.join("\n")
            );
        }
    }

    #[test]
    fn test_sphere_isosurface() {
        let n = 32u32;
        let mut data = vec![0.0f32; (n * n * n) as usize];
        let center = n as f32 / 2.0;

        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let dx = ix as f32 - center;
                    let dy = iy as f32 - center;
                    let dz = iz as f32 - center;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    data[(ix + iy * n + iz * n * n) as usize] = dist;
                }
            }
        }

        let volume = VolumeData {
            data,
            dims: [n, n, n],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        };

        let mesh = extract_isosurface(&volume, 8.0);

        // Mesh was generated.
        assert!(
            !mesh.positions.is_empty(),
            "Sphere isosurface should produce vertices"
        );

        // Valid triangle list.
        assert_eq!(mesh.indices.len() % 3, 0, "Indices must be a multiple of 3");

        // Reasonable triangle count for a sphere.
        assert!(
            mesh.indices.len() > 100,
            "Expected > 100 indices for sphere, got {}",
            mesh.indices.len()
        );

        // All positions within expected bounding box (centre 16 +/- radius 8 + margin).
        for pos in &mesh.positions {
            for c in pos {
                assert!(
                    *c >= 4.0 && *c <= 28.0,
                    "Position component {} out of expected range [4, 28]",
                    c
                );
            }
        }

        // All normals approximately unit length.
        for n in &mesh.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                len > 0.95 && len < 1.05,
                "Normal length {} not approximately 1.0",
                len
            );
        }

        // positions and normals must be same length.
        assert_eq!(mesh.positions.len(), mesh.normals.len());
    }

    #[test]
    fn test_sphere_winding_order() {
        // Extract a sphere isosurface and verify geometric normals (cross product)
        // align with the gradient-based vertex normals (which point outward for SDF).
        // A winding mismatch means some fraction of triangles would be back-face culled.
        let n = 32u32;
        let center = n as f32 / 2.0;
        let mut data = vec![0.0f32; (n * n * n) as usize];
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let dx = ix as f32 - center;
                    let dy = iy as f32 - center;
                    let dz = iz as f32 - center;
                    data[(ix + iy * n + iz * n * n) as usize] =
                        (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }
        }
        let volume = VolumeData {
            data,
            dims: [n, n, n],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        };
        let mesh = extract_isosurface(&volume, 8.0);
        assert!(!mesh.positions.is_empty(), "expected vertices");

        let mut correct = 0usize;
        let mut flipped = 0usize;
        let tri_count = mesh.indices.len() / 3;
        for t in 0..tri_count {
            let i0 = mesh.indices[t * 3] as usize;
            let i1 = mesh.indices[t * 3 + 1] as usize;
            let i2 = mesh.indices[t * 3 + 2] as usize;
            let p0 = mesh.positions[i0];
            let p1 = mesh.positions[i1];
            let p2 = mesh.positions[i2];
            // Geometric normal via cross product.
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let gn = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            // Vertex normal (gradient-based, points outward for this SDF).
            let vn = mesh.normals[i0];
            let dot = gn[0] * vn[0] + gn[1] * vn[1] + gn[2] * vn[2];
            if dot >= 0.0 {
                correct += 1;
            } else {
                flipped += 1;
            }
        }

        let total = correct + flipped;
        let flipped_pct = flipped as f32 / total as f32 * 100.0;
        assert!(
            flipped_pct < 5.0,
            "{}/{} triangles ({:.1}%) have geometric normal opposing the gradient : winding is wrong",
            flipped,
            total,
            flipped_pct
        );
    }

    #[test]
    fn test_empty_volume() {
        // All values above isovalue.
        let data = vec![10.0f32; 8]; // 2x2x2, all 10.0
        let volume = VolumeData {
            data,
            dims: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        };

        let mesh = extract_isosurface(&volume, 5.0);
        assert_eq!(
            mesh.positions.len(),
            0,
            "All-above should produce empty mesh"
        );
        assert_eq!(mesh.indices.len(), 0);
    }
}
