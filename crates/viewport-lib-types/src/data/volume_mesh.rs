//! Volume mesh submission payloads: tetrahedral meshes with their per-vertex
//! attributes, and unstructured meshes of tet, pyramid, wedge and hex cells.

use std::collections::HashMap;

use glam::Vec3;

/// Per-vertex attributes. Each `Vec` is either empty (attribute absent)
/// or has length equal to the vertex count.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TetMeshAttributes {
    /// Region/material id per vertex.
    pub region_tag: Vec<u32>,
    /// Target lumped mass per vertex.
    pub target_mass: Vec<f32>,
    /// Target influence radius per vertex.
    pub target_radius: Vec<f32>,
}

impl TetMeshAttributes {
    /// True when no attribute arrays are present.
    pub fn is_empty(&self) -> bool {
        self.region_tag.is_empty() && self.target_mass.is_empty() && self.target_radius.is_empty()
    }

    /// True when every present attribute array matches `vertex_count` (an empty
    /// array counts as absent and is always allowed).
    pub fn is_consistent(&self, vertex_count: usize) -> bool {
        let ok = |len: usize| len == 0 || len == vertex_count;
        ok(self.region_tag.len()) && ok(self.target_mass.len()) && ok(self.target_radius.len())
    }
}

/// A tetrahedral mesh. Pure data, no behaviour beyond accessors and
/// per-tet geometry helpers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TetMesh {
    positions: Vec<Vec3>,
    tets: Vec<[u32; 4]>,
    attributes: TetMeshAttributes,
}

impl TetMesh {
    /// Build a tet mesh from vertex positions and tet connectivity, with no
    /// per-vertex attributes.
    pub fn new(positions: Vec<Vec3>, tets: Vec<[u32; 4]>) -> Self {
        Self {
            positions,
            tets,
            attributes: TetMeshAttributes::default(),
        }
    }

    /// Attach per-vertex attributes, consuming and returning the mesh.
    pub fn with_attributes(mut self, attributes: TetMeshAttributes) -> Self {
        self.attributes = attributes;
        self
    }

    /// Vertex positions.
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Mutable vertex positions.
    pub fn positions_mut(&mut self) -> &mut Vec<Vec3> {
        &mut self.positions
    }

    /// Tet connectivity (four vertex indices per tet).
    pub fn tets(&self) -> &[[u32; 4]] {
        &self.tets
    }

    /// Mutable tet connectivity.
    pub fn tets_mut(&mut self) -> &mut Vec<[u32; 4]> {
        &mut self.tets
    }

    /// Per-vertex attributes.
    pub fn attributes(&self) -> &TetMeshAttributes {
        &self.attributes
    }

    /// Mutable per-vertex attributes.
    pub fn attributes_mut(&mut self) -> &mut TetMeshAttributes {
        &mut self.attributes
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of tets.
    pub fn tet_count(&self) -> usize {
        self.tets.len()
    }

    /// Signed volume of a single tet.
    pub fn signed_volume(&self, tet_index: usize) -> f32 {
        let [a, b, c, d] = self.tets[tet_index];
        let p = &self.positions;
        let va = p[a as usize];
        let vb = p[b as usize];
        let vc = p[c as usize];
        let vd = p[d as usize];
        (vb - va).cross(vc - va).dot(vd - va) / 6.0
    }

    /// Sum of the absolute volumes of all tets.
    pub fn total_volume(&self) -> f32 {
        (0..self.tets.len())
            .map(|i| self.signed_volume(i).abs())
            .sum()
    }
}

/// Sentinel value that marks unused index slots in a cell stored as 8 indices.
///
/// Slots beyond the cell's vertex count must be filled with this value.
/// For example, a tet uses slots `[0..4]`; slots `[4..8]` must be `CELL_SENTINEL`.
pub const CELL_SENTINEL: u32 = u32::MAX;

/// Input data for an unstructured volume mesh (tets, hexes, or mixed).
///
/// Each cell is represented as exactly 8 vertex indices.  For cells with fewer
/// than 8 vertices, fill unused slots with [`CELL_SENTINEL`] (`u32::MAX`):
///
/// - **Tet**: indices `[0..4]` valid; `[4..8]` = `CELL_SENTINEL`
/// - **Pyramid**: indices `[0..5]` valid; `[5..8]` = `CELL_SENTINEL`
/// - **Wedge**: indices `[0..6]` valid; `[6..8]` = `CELL_SENTINEL`
/// - **Hex**: all 8 indices are valid, in VTK unstructured-grid order.
///
/// `PartialEq` compares every field in full; `Debug` prints counts and
/// attribute names rather than the vertices.
///
/// ```
/// use viewport_lib_types::data::volume_mesh::{VolumeMeshData, CELL_SENTINEL};
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
#[derive(Default, Clone, PartialEq)]
pub struct VolumeMeshData {
    /// Vertex positions in local space.
    pub positions: Vec<[f32; 3]>,

    /// Cell connectivity : exactly 8 indices per cell.
    ///
    /// Tets: first 4 indices are the tet vertices; indices `[4..8]` must be
    /// [`CELL_SENTINEL`].  Hexes: all 8 indices are valid.  Other cell types
    /// use [`CELL_SENTINEL`] to pad unused slots (see the type's docs).
    pub cells: Vec<[u32; 8]>,

    /// Named per-cell scalar attributes (one `f32` per cell).
    ///
    /// Automatically remapped to boundary face scalars during upload so they
    /// can be visualised via [`AttributeKind::Face`](crate::data::attribute::AttributeKind::Face).
    pub cell_scalars: HashMap<String, Vec<f32>>,

    /// Named per-cell RGBA colour attributes (one `[f32; 4]` per cell).
    ///
    /// Automatically remapped to boundary face colours during upload, rendered
    /// via [`AttributeKind::FaceColour`](crate::data::attribute::AttributeKind::FaceColour).
    pub cell_colours: HashMap<String, Vec<[f32; 4]>>,

    /// Named per-vertex scalar attributes (one `f32` per entry of `positions`).
    ///
    /// Carried onto the boundary surface as per-vertex values, so they
    /// interpolate across each face and are visualised via
    /// [`AttributeKind::Vertex`](crate::data::attribute::AttributeKind::Vertex).
    /// A short array is padded with `0.0`. A name also present in
    /// `cell_scalars` or `cell_colours` is skipped: the surface holds one
    /// attribute per name, and the cell entry keeps it.
    ///
    /// In a clipped extraction the vertices a cut creates take the value
    /// interpolated inside their cell. The transparent mode draws one value
    /// per tet, so there a node scalar is the mean of the tet's four corners.
    pub node_scalars: HashMap<String, Vec<f32>>,
}

impl std::fmt::Debug for VolumeMeshData {
    /// Prints counts and attribute names rather than the vertices: a volume
    /// mesh is large enough that dumping it buries whatever else is logged.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn names<V>(map: &HashMap<String, V>) -> Vec<&str> {
            let mut names: Vec<&str> = map.keys().map(String::as_str).collect();
            names.sort_unstable();
            names
        }
        f.debug_struct("VolumeMeshData")
            .field("positions", &self.positions.len())
            .field("cells", &self.cells.len())
            .field("cell_scalars", &names(&self.cell_scalars))
            .field("cell_colours", &names(&self.cell_colours))
            .field("node_scalars", &names(&self.node_scalars))
            .finish()
    }
}

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

    /// Centroid of every cell, one per entry in [`VolumeMeshData::cells`].
    ///
    /// The centroid is the average of the cell's valid vertex positions, so
    /// [`CELL_SENTINEL`] padding and out-of-range indices are ignored and every
    /// cell shape is handled the same way. This is where per-cell quantities go
    /// when they need a position: the result is index-aligned with `cells`, and
    /// a cell with no usable vertex at all sits at the origin.
    pub fn cell_centroids(&self) -> Vec<[f32; 3]> {
        self.cells
            .iter()
            .map(|cell| {
                let mut sum = [0.0f32; 3];
                let mut count = 0u32;
                for &idx in cell {
                    if idx == CELL_SENTINEL {
                        continue;
                    }
                    let Some(p) = self.positions.get(idx as usize) else {
                        continue;
                    };
                    sum[0] += p[0];
                    sum[1] += p[1];
                    sum[2] += p[2];
                    count += 1;
                }
                if count == 0 {
                    return [0.0; 3];
                }
                let inv = 1.0 / count as f32;
                [sum[0] * inv, sum[1] * inv, sum[2] * inv]
            })
            .collect()
    }

    /// Extract all tetrahedral cells and return a [`TetMesh`].
    ///
    /// Cells whose slots `[4..8]` are all [`CELL_SENTINEL`] are tets; every
    /// other cell shape is dropped and counted in the returned
    /// [`ConversionReport`]. Returns an error if no tet cells are present or
    /// if a cell references a vertex index out of range.
    pub fn to_tet_mesh(&self) -> Result<(TetMesh, ConversionReport), ToTetMeshError> {
        let mut tet_indices: Vec<[u32; 4]> = Vec::new();
        let mut dropped = 0usize;
        for cell in &self.cells {
            if cell[4] == CELL_SENTINEL
                && cell[5] == CELL_SENTINEL
                && cell[6] == CELL_SENTINEL
                && cell[7] == CELL_SENTINEL
            {
                tet_indices.push([cell[0], cell[1], cell[2], cell[3]]);
            } else {
                dropped += 1;
            }
        }

        if tet_indices.is_empty() {
            return Err(ToTetMeshError::NoTetCells);
        }

        let vertex_count = self.positions.len();
        let mut remap = vec![u32::MAX; vertex_count];
        let mut positions: Vec<Vec3> = Vec::new();
        let mut tets: Vec<[u32; 4]> = Vec::with_capacity(tet_indices.len());
        for raw in tet_indices {
            let mut out = [0_u32; 4];
            for (slot, &idx) in raw.iter().enumerate() {
                let idx_usize = idx as usize;
                if idx_usize >= vertex_count {
                    return Err(ToTetMeshError::OutOfRangeIndex(idx));
                }
                if remap[idx_usize] == u32::MAX {
                    remap[idx_usize] = positions.len() as u32;
                    let p = self.positions[idx_usize];
                    positions.push(Vec3::new(p[0], p[1], p[2]));
                }
                out[slot] = remap[idx_usize];
            }
            tets.push(out);
        }

        Ok((
            TetMesh::new(positions, tets),
            ConversionReport {
                dropped_non_tet_cells: dropped,
            },
        ))
    }
}

/// Side data returned by [`VolumeMeshData::to_tet_mesh`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConversionReport {
    /// Non-tet cells (pyramid, wedge, hex) dropped during extraction.
    pub dropped_non_tet_cells: usize,
}

/// Error returned by [`VolumeMeshData::to_tet_mesh`].
#[derive(Debug)]
pub enum ToTetMeshError {
    /// The mesh contained no tetrahedral cells.
    NoTetCells,
    /// A cell referenced a vertex index beyond the position array.
    OutOfRangeIndex(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_prints_counts_not_vertices() {
        let mut data = VolumeMeshData::default();
        data.positions = vec![[1.25, 0.0, 0.0]; 4];
        data.push_tet(0, 1, 2, 3);
        data.cell_scalars.insert("stress".into(), vec![0.0]);
        let text = format!("{data:?}");
        assert!(text.contains("cells: 1"), "{text}");
        assert!(text.contains("\"stress\""), "{text}");
        assert!(!text.contains("1.25"), "{text}");
    }
}
