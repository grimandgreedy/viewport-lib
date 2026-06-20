//! Tetrahedral mesh data type.

use glam::Vec3;

/// Per-vertex attributes. Each `Vec` is either empty (attribute absent)
/// or has length equal to the vertex count.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TetMeshAttributes {
    pub region_tag: Vec<u32>,
    pub target_mass: Vec<f32>,
    pub target_radius: Vec<f32>,
}

impl TetMeshAttributes {
    pub fn is_empty(&self) -> bool {
        self.region_tag.is_empty() && self.target_mass.is_empty() && self.target_radius.is_empty()
    }

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
    pub fn new(positions: Vec<Vec3>, tets: Vec<[u32; 4]>) -> Self {
        Self { positions, tets, attributes: TetMeshAttributes::default() }
    }

    pub fn with_attributes(mut self, attributes: TetMeshAttributes) -> Self {
        self.attributes = attributes;
        self
    }

    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    pub fn positions_mut(&mut self) -> &mut Vec<Vec3> {
        &mut self.positions
    }

    pub fn tets(&self) -> &[[u32; 4]] {
        &self.tets
    }

    pub fn tets_mut(&mut self) -> &mut Vec<[u32; 4]> {
        &mut self.tets
    }

    pub fn attributes(&self) -> &TetMeshAttributes {
        &self.attributes
    }

    pub fn attributes_mut(&mut self) -> &mut TetMeshAttributes {
        &mut self.attributes
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

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

    pub fn total_volume(&self) -> f32 {
        (0..self.tets.len()).map(|i| self.signed_volume(i).abs()).sum()
    }
}
