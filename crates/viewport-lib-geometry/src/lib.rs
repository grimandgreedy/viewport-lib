//! CPU geometry algorithms that operate on `viewport-lib-types` payloads without
//! touching the GPU.
//!
//! Two examples: turning a scalar field into a surface mesh
//! ([`marching_cubes`]) and turning unstructured cell connectivity into a
//! renderable boundary mesh ([`volume_mesh`]). Both produce a
//! [`MeshData`](viewport_lib_types::data::mesh::MeshData) that the renderer can
//! upload, but neither needs the renderer to run.
//!
//! Renderer-coupled geometry (BVH picking, implicit sphere-marching, the
//! primitive builders) lives in `viewport-lib`.

/// Closed-form eigendecomposition of symmetric 3x3 matrices (tensor fields).
pub mod eigen;
/// Ray/primitive intersection helpers.
pub mod intersect;
/// Tangent-plane (intrinsic) vector fields to world-space vectors.
pub mod intrinsic_vectors;
pub mod marching_cubes;
/// Pure CPU mesh operations: tangent computation, attribute expansion, validation.
pub mod mesh_ops;
/// Whitney reconstruction of a vector field from an edge one-form.
pub mod one_forms;
/// Geometry primitives: cube, sphere, plane, cylinder, cone, capsule, torus and friends.
pub mod primitives;
/// Per-vertex and per-face tangent-frame computation (Gram-Schmidt).
pub mod tangent_frames;
/// Positions paired with a world-space vector, as the quantity helpers return them.
pub mod vector_samples;
pub mod volume_mesh;

/// Clip-plane cap mesh generation (plane-mesh contour extraction + triangulation).
pub mod cap_geometry;
/// Polyline construction helpers (circle loops and similar wireframe primitives).
pub mod polyline;

pub mod prelude {
    //! The geometry entry points, in one glob import:
    //! `use viewport_lib_geometry::prelude::*;`.
    pub use crate::eigen::{SymmetricEigen, symmetric_eigen_3x3};
    pub use crate::marching_cubes::{VolumeData, extract_isosurface};
    pub use crate::vector_samples::VectorSamples;
    pub use crate::volume_mesh::{CELL_SENTINEL, GridCells, VolumeMeshData};
}
