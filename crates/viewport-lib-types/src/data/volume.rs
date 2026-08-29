//! Volume submission payloads: tetrahedral meshes and their per-vertex
//! attributes.

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
