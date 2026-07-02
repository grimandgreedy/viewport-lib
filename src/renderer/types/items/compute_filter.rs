/// Whether a filter runs on CPU or GPU compute shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterMode {
    /// CPU-side index compaction (existing path, always works).
    #[default]
    Cpu,
    /// GPU compute shader index compaction (faster for large meshes).
    Gpu,
}

/// Kind of GPU compute filter operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeFilterKind {
    /// Clip: discard triangles where all 3 vertices are on the negative side of a plane.
    /// Plane defined as (normal.xyz, distance) where dot(pos, normal) < distance => clipped.
    Clip {
        /// Unit normal of the clip plane.
        plane_normal: [f32; 3],
        /// Signed distance from origin along the plane normal.
        plane_dist: f32,
    },
    /// Box clip: discard triangles where all 3 vertices are outside an oriented box region.
    ClipBox {
        /// Box center in world space.
        center: [f32; 3],
        /// Half-extents (per-axis radius) of the box.
        half_extents: [f32; 3],
        /// Rotation matrix columns: local X, Y, Z axes in world space.
        orientation: [[f32; 3]; 3],
    },
    /// Sphere clip: discard triangles where all 3 vertices are outside a sphere region.
    ClipSphere {
        /// Sphere center in world space.
        center: [f32; 3],
        /// Sphere radius in world units.
        radius: f32,
    },
    /// Threshold: discard triangles where all 3 vertex scalars are outside [min, max].
    Threshold {
        /// Minimum scalar value (inclusive).
        min: f32,
        /// Maximum scalar value (inclusive).
        max: f32,
    },
}

/// A GPU compute filter item : references an existing uploaded mesh.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ComputeFilterItem {
    /// `MeshId` of the uploaded GPU mesh for this compute filter item.
    pub mesh_id: crate::resources::mesh_store::MeshId,
    /// Which filter to apply.
    pub kind: ComputeFilterKind,
    /// Name of the scalar attribute buffer (for Threshold). Ignored for Clip.
    pub attribute_name: Option<String>,
}

impl Default for ComputeFilterItem {
    fn default() -> Self {
        Self {
            mesh_id: crate::resources::mesh_store::MeshId(0),
            kind: ComputeFilterKind::Clip {
                plane_normal: [0.0, 0.0, 1.0],
                plane_dist: 0.0,
            },
            attribute_name: None,
        }
    }
}
