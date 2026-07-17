//! Tetrahedral mesh data type.

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

/// Uniform buffer layout for the projected tetrahedra pass.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ProjectedTetUniform {
    pub(crate) density: f32,
    pub(crate) scalar_min: f32,
    pub(crate) scalar_max: f32,
    pub(crate) threshold_min: f32,
    pub(crate) threshold_max: f32,
    /// 1 = skip Beer-Lambert thickness modulation and emit a flat density-scaled
    /// alpha per visible fragment. Wired from `ItemSettings.unlit`.
    pub(crate) unlit: u32,
    /// Multiplied into the final alpha. Wired from `ItemSettings.opacity`.
    pub(crate) opacity: f32,
    pub(crate) _pad: f32,
}

/// One device-limit-bounded chunk of a projected-tet mesh.
pub(crate) struct ProjectedTetChunk {
    /// Storage buffer for this chunk's tetrahedra geometry (kept alive for the
    /// bind group). Vertex positions only; the per-tet scalar lives in
    /// `scalar_buffer` so a scalar-only refresh does not touch geometry.
    #[allow(dead_code)]
    pub tet_buffer: crate::gpu::Buffer,
    /// Per-tet scalar storage buffer (one `f32` per tet), parallel to
    /// `tet_buffer`. Rewritten in place by `replace_projected_tet_scalar`
    /// without rebuilding the geometry.
    #[allow(dead_code)]
    pub scalar_buffer: crate::gpu::Buffer,
    /// Number of tetrahedra in this chunk (= instanced draw count).
    pub tet_count: u32,
    /// Bind group: shared uniform + this chunk's tet + scalar buffers + colourmap.
    pub bind_group: crate::gpu::BindGroup,
}

/// Uploaded projected-tetrahedra mesh, stored persistently on the GPU.
///
/// Large meshes are split into multiple chunks so each storage buffer
/// stays within `max_storage_buffer_binding_size`.
/// Created by [`DeviceResources::upload_projected_tet`].
pub(crate) struct GpuProjectedTetMesh {
    /// One or more device-limit-bounded chunks.
    pub chunks: Vec<ProjectedTetChunk>,
    /// Uniform buffer shared across all chunks (density, scalar_min/max). Written each frame.
    pub uniform_buffer: crate::gpu::Buffer,
    /// Auto-detected scalar range from the uploaded data (min, max).
    pub scalar_range: (f32, f32),
}
