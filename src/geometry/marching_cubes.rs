//! CPU-side marching cubes isosurface extraction from volumetric scalar data.
//!
//! The output is a standard [`MeshData`](crate::resources::MeshData) that can be uploaded via
//! [`upload_mesh_data()`](crate::ViewportGpuResources::upload_mesh_data) or
//! [`replace_mesh_data()`](crate::ViewportGpuResources::replace_mesh_data).
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

use crate::resources::MeshData;
use std::collections::HashMap;

/// A structured 3D scalar field on a regular grid.
#[derive(Debug, Clone)]
pub struct VolumeData {
    /// Flattened scalar values in x-fastest order: `index = x + y*nx + z*nx*ny`.
    pub data: Vec<f32>,
    /// Grid dimensions `[nx, ny, nz]`.
    pub dims: [u32; 3],
    /// World-space origin of the grid corner `(0, 0, 0)`.
    pub origin: [f32; 3],
    /// Cell size in each axis direction.
    pub spacing: [f32; 3],
}

impl VolumeData {
    /// Check whether grid indices are within bounds.
    pub fn in_bounds(&self, ix: u32, iy: u32, iz: u32) -> bool {
        ix < self.dims[0] && iy < self.dims[1] && iz < self.dims[2]
    }

    /// Read the scalar value at grid point `(ix, iy, iz)`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn sample(&self, ix: u32, iy: u32, iz: u32) -> f32 {
        let nx = self.dims[0] as usize;
        let ny = self.dims[1] as usize;
        self.data[ix as usize + iy as usize * nx + iz as usize * nx * ny]
    }
}

/// Trilinear interpolation of the volume at an arbitrary world-space position.
///
/// Returns the interpolated scalar value. Points outside the grid are clamped
/// to the boundary.
pub fn trilinear_sample(volume: &VolumeData, world_pos: [f32; 3]) -> f32 {
    let [nx, ny, nz] = volume.dims;

    // Convert world position to continuous grid coordinates.
    let gx = (world_pos[0] - volume.origin[0]) / volume.spacing[0];
    let gy = (world_pos[1] - volume.origin[1]) / volume.spacing[1];
    let gz = (world_pos[2] - volume.origin[2]) / volume.spacing[2];

    // Clamp to valid range.
    let gx = gx.clamp(0.0, (nx as f32) - 1.001);
    let gy = gy.clamp(0.0, (ny as f32) - 1.001);
    let gz = gz.clamp(0.0, (nz as f32) - 1.001);

    let ix = gx.floor() as u32;
    let iy = gy.floor() as u32;
    let iz = gz.floor() as u32;

    let fx = gx - ix as f32;
    let fy = gy - iy as f32;
    let fz = gz - iz as f32;

    let ix1 = (ix + 1).min(nx - 1);
    let iy1 = (iy + 1).min(ny - 1);
    let iz1 = (iz + 1).min(nz - 1);

    // 8-corner trilinear interpolation.
    let c000 = volume.sample(ix, iy, iz);
    let c100 = volume.sample(ix1, iy, iz);
    let c010 = volume.sample(ix, iy1, iz);
    let c110 = volume.sample(ix1, iy1, iz);
    let c001 = volume.sample(ix, iy, iz1);
    let c101 = volume.sample(ix1, iy, iz1);
    let c011 = volume.sample(ix, iy1, iz1);
    let c111 = volume.sample(ix1, iy1, iz1);

    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    c0 * (1.0 - fz) + c1 * fz
}

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
/// Vertices along shared edges are deduplicated via an edge-vertex cache, producing
/// a clean mesh suitable for smooth rendering.
pub fn extract_isosurface(volume: &VolumeData, isovalue: f32) -> MeshData {
    let [nx, ny, nz] = volume.dims;
    if nx < 2 || ny < 2 || nz < 2 {
        return MeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
            uvs: None,
            tangents: None,
            attributes: HashMap::new(),
        };
    }

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Edge-vertex cache: key = (cell_x, cell_y, cell_z, edge_id) -> vertex index.
    // We encode shared edges using the canonical lower-corner + axis direction.
    let mut edge_cache: HashMap<(u32, u32, u32, u8), u32> = HashMap::new();

    for iz in 0..(nz - 1) {
        for iy in 0..(ny - 1) {
            for ix in 0..(nx - 1) {
                // 8 corner values in standard marching cubes order (Bourke numbering).
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

    MeshData {
        positions,
        normals,
        indices,
        uvs: None,
        tangents: None,
        attributes: HashMap::new(),
    }
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

/// Edge vertex pairs: for edge i, EDGE_VERTICES[i] = (corner_a, corner_b).
const EDGE_VERTICES: [(u8, u8); 12] = [
    (0, 1), // edge 0
    (1, 2), // edge 1
    (2, 3), // edge 2
    (3, 0), // edge 3
    (4, 5), // edge 4
    (5, 6), // edge 5
    (6, 7), // edge 6
    (7, 4), // edge 7
    (0, 4), // edge 8
    (1, 5), // edge 9
    (2, 6), // edge 10
    (3, 7), // edge 11
];

// ---------------------------------------------------------------------------
// Standard marching cubes lookup tables (Paul Bourke)
// ---------------------------------------------------------------------------

/// 12-bit edge intersection bitmask per cube configuration.
#[rustfmt::skip]
const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// Triangle table: up to 5 triangles (15 edge indices) per configuration. -1 = sentinel.
#[rustfmt::skip]
pub(crate) const TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 9, 8, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 0, 2, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 8, 3, 2,10, 8,10, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 8,11, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11, 2, 1, 9,11, 9, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 1,11,10, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,10, 1, 0, 8,10, 8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 9, 0, 3,11, 9,11,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 7, 3, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 1, 9, 4, 7, 1, 7, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 4, 7, 3, 0, 4, 1, 2,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 2,10, 9, 0, 2, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 8, 4, 7, 3,11, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 4, 7,11, 2, 4, 2, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 8, 4, 7, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 7,11, 9, 4,11, 9,11, 2, 9, 2, 1,-1,-1,-1,-1],
    [ 3,10, 1, 3,11,10, 7, 8, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 1,11,10, 1, 4,11, 1, 0, 4, 7,11, 4,-1,-1,-1,-1],
    [ 4, 7, 8, 9, 0,11, 9,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 4, 7,11, 4,11, 9, 9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 1, 5, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 5, 4, 8, 3, 5, 3, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2,10, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 2,10, 5, 4, 2, 4, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 2,10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8,-1,-1,-1,-1],
    [ 9, 5, 4, 2, 3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0,11, 2, 0, 8,11, 4, 9, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 5, 4, 0, 1, 5, 2, 3,11,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 1, 5, 2, 5, 8, 2, 8,11, 4, 8, 5,-1,-1,-1,-1],
    [10, 3,11,10, 1, 3, 9, 5, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 0, 8, 1, 8,10, 1, 8,11,10,-1,-1,-1,-1],
    [ 5, 4, 0, 5, 0,11, 5,11,10,11, 0, 3,-1,-1,-1,-1],
    [ 5, 4, 8, 5, 8,10,10, 8,11,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 5, 7, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 3, 0, 9, 5, 3, 5, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 8, 0, 1, 7, 1, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 7, 8, 9, 5, 7,10, 1, 2,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3,-1,-1,-1,-1],
    [ 8, 0, 2, 8, 2, 5, 8, 5, 7,10, 5, 2,-1,-1,-1,-1],
    [ 2,10, 5, 2, 5, 3, 3, 5, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 9, 5, 7, 8, 9, 3,11, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 2, 3,11, 0, 1, 8, 1, 7, 8, 1, 5, 7,-1,-1,-1,-1],
    [11, 2, 1,11, 1, 7, 7, 1, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 8, 8, 5, 7,10, 1, 3,10, 3,11,-1,-1,-1,-1],
    [ 5, 7, 0, 5, 0, 9, 7,11, 0, 1, 0,10,11,10, 0,-1],
    [11,10, 0,11, 0, 3,10, 5, 0, 8, 0, 7, 5, 7, 0,-1],
    [11,10, 5, 7,11, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 8, 3, 1, 9, 8, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 2, 6, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 5, 1, 2, 6, 3, 0, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 6, 5, 9, 0, 6, 0, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 0, 8,11, 2, 0,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9, 2, 3,11, 5,10, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 1, 9, 2, 9,11, 2, 9, 8,11,-1,-1,-1,-1],
    [ 6, 3,11, 6, 5, 3, 5, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8,11, 0,11, 5, 0, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 3,11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9,-1,-1,-1,-1],
    [ 6, 5, 9, 6, 9,11,11, 9, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 3, 0, 4, 7, 3, 6, 5,10,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 5,10, 6, 8, 4, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4,-1,-1,-1,-1],
    [ 6, 1, 2, 6, 5, 1, 4, 7, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7,-1,-1,-1,-1],
    [ 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6,-1,-1,-1,-1],
    [ 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9,-1],
    [ 3,11, 2, 7, 8, 4,10, 6, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 5,10, 6, 4, 7, 2, 4, 2, 0, 2, 7,11,-1,-1,-1,-1],
    [ 0, 1, 9, 4, 7, 8, 2, 3,11, 5,10, 6,-1,-1,-1,-1],
    [ 9, 2, 1, 9,11, 2, 9, 4,11, 7,11, 4, 5,10, 6,-1],
    [ 8, 4, 7, 3,11, 5, 3, 5, 1, 5,11, 6,-1,-1,-1,-1],
    [ 5, 1,11, 5,11, 6, 1, 0,11, 7,11, 4, 0, 4,11,-1],
    [ 0, 5, 9, 0, 6, 5, 0, 3, 6,11, 6, 3, 8, 4, 7,-1],
    [ 6, 5, 9, 6, 9,11, 4, 7, 9, 7,11, 9,-1,-1,-1,-1],
    [10, 4, 9, 6, 4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,10, 6, 4, 9,10, 0, 8, 3,-1,-1,-1,-1,-1,-1,-1],
    [10, 0, 1,10, 6, 0, 6, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 1, 4, 9, 1, 2, 4, 2, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4,-1,-1,-1,-1],
    [ 0, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 3, 2, 8, 2, 4, 4, 2, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 4, 9,10, 6, 4,11, 2, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 2, 2, 8,11, 4, 9,10, 4,10, 6,-1,-1,-1,-1],
    [ 3,11, 2, 0, 1, 6, 0, 6, 4, 6, 1,10,-1,-1,-1,-1],
    [ 6, 4, 1, 6, 1,10, 4, 8, 1, 2, 1,11, 8,11, 1,-1],
    [ 9, 6, 4, 9, 3, 6, 9, 1, 3,11, 6, 3,-1,-1,-1,-1],
    [ 8,11, 1, 8, 1, 0,11, 6, 1, 9, 1, 4, 6, 4, 1,-1],
    [ 3,11, 6, 3, 6, 0, 0, 6, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 4, 8,11, 6, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7,10, 6, 7, 8,10, 8, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 7, 3, 0,10, 7, 0, 9,10, 6, 7,10,-1,-1,-1,-1],
    [10, 6, 7, 1,10, 7, 1, 7, 8, 1, 8, 0,-1,-1,-1,-1],
    [10, 6, 7,10, 7, 1, 1, 7, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9,-1],
    [ 7, 8, 0, 7, 0, 6, 6, 0, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 3, 2, 6, 7, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3,11,10, 6, 8,10, 8, 9, 8, 6, 7,-1,-1,-1,-1],
    [ 2, 0, 7, 2, 7,11, 0, 9, 7, 6, 7,10, 9,10, 7,-1],
    [ 1, 8, 0, 1, 7, 8, 1,10, 7, 6, 7,10, 2, 3,11,-1],
    [11, 2, 1,11, 1, 7,10, 6, 1, 6, 7, 1,-1,-1,-1,-1],
    [ 8, 9, 6, 8, 6, 7, 9, 1, 6,11, 6, 3, 1, 3, 6,-1],
    [ 0, 9, 1,11, 6, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 8, 0, 7, 0, 6, 3,11, 0,11, 6, 0,-1,-1,-1,-1],
    [ 7,11, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 8,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1, 9,11, 7, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 9, 8, 3, 1,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [10, 1, 2, 6,11, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0, 8, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 9, 0, 2,10, 9, 6,11, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 2,10, 3,10, 8, 3,10, 9, 8,-1,-1,-1,-1],
    [ 7, 2, 3, 6, 2, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 7, 0, 8, 7, 6, 0, 6, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 7, 6, 2, 3, 7, 0, 1, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6,-1,-1,-1,-1],
    [10, 7, 6,10, 1, 7, 1, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 6, 1, 7,10, 1, 8, 7, 1, 0, 8,-1,-1,-1,-1],
    [ 0, 3, 7, 0, 7,10, 0,10, 9, 6,10, 7,-1,-1,-1,-1],
    [ 7, 6,10, 7,10, 8, 8,10, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 6, 8, 4,11, 8, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 3, 0, 6, 0, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 6,11, 8, 4, 6, 9, 0, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 4, 6, 9, 6, 3, 9, 3, 1,11, 3, 6,-1,-1,-1,-1],
    [ 6, 8, 4, 6,11, 8, 2,10, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 3, 0,11, 0, 6,11, 0, 4, 6,-1,-1,-1,-1],
    [ 4,11, 8, 4, 6,11, 0, 2, 9, 2,10, 9,-1,-1,-1,-1],
    [10, 9, 3,10, 3, 2, 9, 4, 3,11, 3, 6, 4, 6, 3,-1],
    [ 8, 2, 3, 8, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 2, 4, 6, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8,-1,-1,-1,-1],
    [ 1, 9, 4, 1, 4, 2, 2, 4, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 1, 3, 8, 6, 1, 8, 4, 6, 6,10, 1,-1,-1,-1,-1],
    [10, 1, 0,10, 0, 6, 6, 0, 4,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 6, 3, 4, 3, 8, 6,10, 3, 0, 3, 9,10, 9, 3,-1],
    [10, 9, 4, 6,10, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 5, 7, 6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 5,11, 7, 6,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 1, 5, 4, 0, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5,-1,-1,-1,-1],
    [ 9, 5, 4,10, 1, 2, 7, 6,11,-1,-1,-1,-1,-1,-1,-1],
    [ 6,11, 7, 1, 2,10, 0, 8, 3, 4, 9, 5,-1,-1,-1,-1],
    [ 7, 6,11, 5, 4,10, 4, 2,10, 4, 0, 2,-1,-1,-1,-1],
    [ 3, 4, 8, 3, 5, 4, 3, 2, 5,10, 5, 2,11, 7, 6,-1],
    [ 7, 2, 3, 7, 6, 2, 5, 4, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7,-1,-1,-1,-1],
    [ 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0,-1,-1,-1,-1],
    [ 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8,-1],
    [ 9, 5, 4,10, 1, 6, 1, 7, 6, 1, 3, 7,-1,-1,-1,-1],
    [ 1, 6,10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4,-1],
    [ 4, 0,10, 4,10, 5, 0, 3,10, 6,10, 7, 3, 7,10,-1],
    [ 7, 6,10, 7,10, 8, 5, 4,10, 4, 8,10,-1,-1,-1,-1],
    [ 6, 9, 5, 6,11, 9,11, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 6,11, 0, 6, 3, 0, 5, 6, 0, 9, 5,-1,-1,-1,-1],
    [ 0,11, 8, 0, 5,11, 0, 1, 5, 5, 6,11,-1,-1,-1,-1],
    [ 6,11, 3, 6, 3, 5, 5, 3, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,10, 9, 5,11, 9,11, 8,11, 5, 6,-1,-1,-1,-1],
    [ 0,11, 3, 0, 6,11, 0, 9, 6, 5, 6, 9, 1, 2,10,-1],
    [11, 8, 5,11, 5, 6, 8, 0, 5,10, 5, 2, 0, 2, 5,-1],
    [ 6,11, 3, 6, 3, 5, 2,10, 3,10, 5, 3,-1,-1,-1,-1],
    [ 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2,-1,-1,-1,-1],
    [ 9, 5, 6, 9, 6, 0, 0, 6, 2,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8,-1],
    [ 1, 5, 6, 2, 1, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 6, 1, 6,10, 3, 8, 6, 5, 6, 9, 8, 9, 6,-1],
    [10, 1, 0,10, 0, 6, 9, 5, 0, 5, 6, 0,-1,-1,-1,-1],
    [ 0, 3, 8, 5, 6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10, 5, 6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10, 7, 5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11, 5,10,11, 7, 5, 8, 3, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 5,11, 7, 5,10,11, 1, 9, 0,-1,-1,-1,-1,-1,-1,-1],
    [10, 7, 5,10,11, 7, 9, 8, 1, 8, 3, 1,-1,-1,-1,-1],
    [11, 1, 2,11, 7, 1, 7, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2,11,-1,-1,-1,-1],
    [ 9, 7, 5, 9, 2, 7, 9, 0, 2, 2,11, 7,-1,-1,-1,-1],
    [ 7, 5, 2, 7, 2,11, 5, 9, 2, 3, 2, 8, 9, 8, 2,-1],
    [ 2, 5,10, 2, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 2, 0, 8, 5, 2, 8, 7, 5,10, 2, 5,-1,-1,-1,-1],
    [ 9, 0, 1, 5,10, 3, 5, 3, 7, 3,10, 2,-1,-1,-1,-1],
    [ 9, 8, 2, 9, 2, 1, 8, 7, 2,10, 2, 5, 7, 5, 2,-1],
    [ 1, 3, 5, 3, 7, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 7, 0, 7, 1, 1, 7, 5,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 0, 3, 9, 3, 5, 5, 3, 7,-1,-1,-1,-1,-1,-1,-1],
    [ 9, 8, 7, 5, 9, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 8, 4, 5,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 5, 0, 4, 5,11, 0, 5,10,11,11, 3, 0,-1,-1,-1,-1],
    [ 0, 1, 9, 8, 4,10, 8,10,11,10, 4, 5,-1,-1,-1,-1],
    [10,11, 4,10, 4, 5,11, 3, 4, 9, 4, 1, 3, 1, 4,-1],
    [ 2, 5, 1, 2, 8, 5, 2,11, 8, 4, 5, 8,-1,-1,-1,-1],
    [ 0, 4,11, 0,11, 3, 4, 5,11, 2,11, 1, 5, 1,11,-1],
    [ 0, 2, 5, 0, 5, 9, 2,11, 5, 4, 5, 8,11, 8, 5,-1],
    [ 9, 4, 5, 2,11, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 5,10, 3, 5, 2, 3, 4, 5, 3, 8, 4,-1,-1,-1,-1],
    [ 5,10, 2, 5, 2, 4, 4, 2, 0,-1,-1,-1,-1,-1,-1,-1],
    [ 3,10, 2, 3, 5,10, 3, 8, 5, 4, 5, 8, 0, 1, 9,-1],
    [ 5,10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 3, 5, 1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 4, 5, 1, 0, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,-1,-1,-1,-1],
    [ 9, 4, 5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4,11, 7, 4, 9,11, 9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 8, 3, 4, 9, 7, 9,11, 7, 9,10,11,-1,-1,-1,-1],
    [ 1,10,11, 1,11, 4, 1, 4, 0, 7, 4,11,-1,-1,-1,-1],
    [ 3, 1, 4, 3, 4, 8, 1,10, 4, 7, 4,11,10,11, 4,-1],
    [ 4,11, 7, 9,11, 4, 9, 2,11, 9, 1, 2,-1,-1,-1,-1],
    [ 9, 7, 4, 9,11, 7, 9, 1,11, 2,11, 1, 0, 8, 3,-1],
    [11, 7, 4,11, 4, 2, 2, 4, 0,-1,-1,-1,-1,-1,-1,-1],
    [11, 7, 4,11, 4, 2, 8, 3, 4, 3, 2, 4,-1,-1,-1,-1],
    [ 2, 9,10, 2, 7, 9, 2, 3, 7, 7, 4, 9,-1,-1,-1,-1],
    [ 9,10, 7, 9, 7, 4,10, 2, 7, 8, 7, 0, 2, 0, 7,-1],
    [ 3, 7,10, 3,10, 2, 7, 4,10, 1,10, 0, 4, 0,10,-1],
    [ 1,10, 2, 8, 7, 4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 7, 1, 3,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1,-1,-1,-1,-1],
    [ 4, 0, 3, 7, 4, 3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 4, 8, 7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 8,10,11, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11,11, 9,10,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 1,10, 0,10, 8, 8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 1,10,11, 3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 2,11, 1,11, 9, 9,11, 8,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 0, 9, 3, 9,11, 1, 2, 9, 2,11, 9,-1,-1,-1,-1],
    [ 0, 2,11, 8, 0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 3, 2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10,10, 8, 9,-1,-1,-1,-1,-1,-1,-1],
    [ 9,10, 2, 0, 9, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 2, 3, 8, 2, 8,10, 0, 1, 8, 1,10, 8,-1,-1,-1,-1],
    [ 1,10, 2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 1, 3, 8, 9, 1, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 9, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [ 0, 3, 8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

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

        // All positions within expected bounding box (center 16 +/- radius 8 + margin).
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
            let e1 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
            let e2 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
            let gn = [
                e1[1]*e2[2] - e1[2]*e2[1],
                e1[2]*e2[0] - e1[0]*e2[2],
                e1[0]*e2[1] - e1[1]*e2[0],
            ];
            // Vertex normal (gradient-based, points outward for this SDF).
            let vn = mesh.normals[i0];
            let dot = gn[0]*vn[0] + gn[1]*vn[1] + gn[2]*vn[2];
            if dot >= 0.0 { correct += 1; } else { flipped += 1; }
        }

        let total = correct + flipped;
        let flipped_pct = flipped as f32 / total as f32 * 100.0;
        assert!(
            flipped_pct < 5.0,
            "{}/{} triangles ({:.1}%) have geometric normal opposing the gradient — winding is wrong",
            flipped, total, flipped_pct
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

    #[test]
    fn test_volume_data_sample() {
        let data = vec![
            1.0, 2.0, // z=0 row: (0,0,0)=1, (1,0,0)=2
            3.0, 4.0, // z=0 row: (0,1,0)=3, (1,1,0)=4
            5.0, 6.0, // z=1 row: (0,0,1)=5, (1,0,1)=6
            7.0, 8.0, // z=1 row: (0,1,1)=7, (1,1,1)=8
        ];
        let vol = VolumeData {
            data,
            dims: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        };

        assert_eq!(vol.sample(0, 0, 0), 1.0);
        assert_eq!(vol.sample(1, 0, 0), 2.0);
        assert_eq!(vol.sample(0, 1, 0), 3.0);
        assert_eq!(vol.sample(1, 1, 0), 4.0);
        assert_eq!(vol.sample(0, 0, 1), 5.0);
        assert_eq!(vol.sample(1, 0, 1), 6.0);
        assert_eq!(vol.sample(0, 1, 1), 7.0);
        assert_eq!(vol.sample(1, 1, 1), 8.0);
    }

    #[test]
    fn test_trilinear_interpolation() {
        let data = vec![
            0.0, 1.0, // (0,0,0)=0, (1,0,0)=1
            0.0, 1.0, // (0,1,0)=0, (1,1,0)=1
            0.0, 1.0, // (0,0,1)=0, (1,0,1)=1
            0.0, 1.0, // (0,1,1)=0, (1,1,1)=1
        ];
        let vol = VolumeData {
            data,
            dims: [2, 2, 2],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
        };

        // At grid points.
        let v00 = trilinear_sample(&vol, [0.0, 0.0, 0.0]);
        assert!(
            (v00 - 0.0).abs() < 0.01,
            "Expected 0.0 at origin, got {}",
            v00
        );

        let v10 = trilinear_sample(&vol, [0.999, 0.0, 0.0]);
        assert!(
            (v10 - 1.0).abs() < 0.02,
            "Expected ~1.0 at (1,0,0), got {}",
            v10
        );

        // At midpoint along X: linear gradient means value = 0.5.
        let mid = trilinear_sample(&vol, [0.5, 0.0, 0.0]);
        assert!(
            (mid - 0.5).abs() < 0.01,
            "Expected 0.5 at midpoint, got {}",
            mid
        );

        // At midpoint in all axes.
        let center = trilinear_sample(&vol, [0.5, 0.5, 0.5]);
        assert!(
            (center - 0.5).abs() < 0.01,
            "Expected 0.5 at center, got {}",
            center
        );
    }
}
