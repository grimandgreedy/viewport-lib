//! Hex meshes built from the occupied cells of a regular grid.

use super::VolumeMeshData;
use std::collections::HashMap;

/// A volume mesh built from occupied cells of a regular grid, with the grid
/// node each vertex sits on.
///
/// Returned by [`VolumeMeshData::from_grid_cells`].
#[derive(Default, Clone)]
pub struct GridCells {
    /// One hex per occupied cell, in the order the cells were given, so
    /// per-cell attributes parallel to that list go straight onto
    /// `cell_scalars` and `cell_colours`.
    pub data: VolumeMeshData,
    /// The grid node `[i, j, k]` under each entry of `data.positions`. Node
    /// `[i, j, k]` is the low corner of cell `[i, j, k]`.
    pub vertex_nodes: Vec<[u32; 3]>,
}

impl GridCells {
    /// Per-vertex values picked out of a dense array over the grid's nodes.
    ///
    /// `dense` is indexed `k * (dims[0] * dims[1]) + j * dims[0] + i`, where
    /// `dims` is the node count on each axis (one more than the cell count).
    /// A node outside `dims`, or past the end of `dense`, reads as `0.0`.
    ///
    /// The result is in vertex order, ready for
    /// [`VolumeMeshData::node_scalars`].
    pub fn node_values(&self, dense: &[f32], dims: [usize; 3]) -> Vec<f32> {
        self.vertex_nodes
            .iter()
            .map(|&[i, j, k]| {
                let (i, j, k) = (i as usize, j as usize, k as usize);
                if i >= dims[0] || j >= dims[1] || k >= dims[2] {
                    return 0.0;
                }
                dense
                    .get(k * dims[0] * dims[1] + j * dims[0] + i)
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

impl VolumeMeshData {
    /// Build a hex mesh from the occupied cells of a regular grid.
    ///
    /// Cell `[i, j, k]` spans `origin + [i, j, k] * cell_size` to
    /// `origin + [i + 1, j + 1, k + 1] * cell_size`. Neighbouring cells share
    /// their corner vertices, so the face between two occupied cells is
    /// interior and [`extract_boundary_faces`](super::extract_boundary_faces) keeps only the outer shell.
    ///
    /// A cell listed twice makes two coincident cells, whose faces then all
    /// read as interior; the caller keeps the list free of duplicates.
    pub fn from_grid_cells(
        origin: [f32; 3],
        cell_size: [f32; 3],
        active_cells: &[[u32; 3]],
    ) -> GridCells {
        // Corner offsets in the module's hex vertex order.
        const CORNERS: [[u32; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ];

        // Vertex index per grid node. A table over the cells' bounding box
        // when that box is not much larger than the cell list, which is the
        // common case and avoids hashing eight nodes per cell; a map when the
        // cells are scattered over a large extent.
        const UNSET: u32 = u32::MAX;
        let mut lo = [u32::MAX; 3];
        let mut hi = [0u32; 3];
        for cell in active_cells {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(cell[axis]);
                hi[axis] = hi[axis].max(cell[axis]);
            }
        }
        // Node counts per axis of the bounding box.
        let dims = [0, 1, 2].map(|a| (hi[a].saturating_sub(lo[a])) as usize + 2);
        let box_nodes = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
        let mut table: Vec<u32> = if box_nodes <= active_cells.len().saturating_mul(64) {
            vec![UNSET; box_nodes]
        } else {
            Vec::new()
        };
        let mut map: HashMap<[u32; 3], u32> = HashMap::new();

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut vertex_nodes: Vec<[u32; 3]> = Vec::new();
        let mut cells: Vec<[u32; 8]> = Vec::with_capacity(active_cells.len());

        for &[ci, cj, ck] in active_cells {
            let mut cell = [0u32; 8];
            for (slot, [di, dj, dk]) in CORNERS.iter().enumerate() {
                let node = [ci + di, cj + dj, ck + dk];
                let entry = if table.is_empty() {
                    map.entry(node).or_insert(UNSET)
                } else {
                    let [i, j, k] = [0, 1, 2].map(|a| (node[a] - lo[a]) as usize);
                    &mut table[(k * dims[1] + j) * dims[0] + i]
                };
                if *entry == UNSET {
                    *entry = positions.len() as u32;
                    positions.push([
                        origin[0] + node[0] as f32 * cell_size[0],
                        origin[1] + node[1] as f32 * cell_size[1],
                        origin[2] + node[2] as f32 * cell_size[2],
                    ]);
                    vertex_nodes.push(node);
                }
                cell[slot] = *entry;
            }
            cells.push(cell);
        }

        GridCells {
            data: VolumeMeshData {
                positions,
                cells,
                ..Default::default()
            },
            vertex_nodes,
        }
    }
}
