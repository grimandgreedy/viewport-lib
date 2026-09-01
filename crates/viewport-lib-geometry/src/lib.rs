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

/// Ray/primitive intersection helpers.
pub mod intersect;
pub mod marching_cubes;
/// Pure CPU mesh operations: tangent computation, attribute expansion, validation.
pub mod mesh_ops;
/// Geometry primitives: cube, sphere, plane, cylinder, cone, capsule, torus and friends.
pub mod primitives;
/// On-surface vector quantities: vector fields and one-forms to glyph sets.
pub mod quantities;
/// Per-vertex and per-face tangent-frame computation (Gram-Schmidt).
pub mod tangent_frames;
pub mod volume_mesh;

/// Clip-plane cap mesh generation (plane-mesh contour extraction + triangulation).
pub mod cap_geometry;
/// Polyline construction helpers (circle loops and similar wireframe primitives).
pub mod polyline;

pub mod prelude {
    //! The geometry entry points, in one glob import:
    //! `use viewport_lib_geometry::prelude::*;`.
    pub use crate::marching_cubes::{VolumeData, extract_isosurface};
    pub use crate::volume_mesh::{CELL_SENTINEL, VolumeMeshData};
}
