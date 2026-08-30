//! CPU geometry algorithms that operate on `viewport-lib-types` payloads without
//! touching the GPU.
//!
//! These are the pieces of `viewport-lib`'s geometry that a CPU-side consumer
//! genuinely needs on its own: turning a scalar field into a surface mesh
//! ([`marching_cubes`]) and turning unstructured cell connectivity into a
//! renderable boundary mesh ([`volume_mesh`]). Both produce a
//! [`MeshData`](viewport_lib_types::data::mesh::MeshData) that the renderer can
//! upload, but neither needs the renderer to run.
//!
//! The renderer-coupled geometry (BVH picking, implicit sphere-marching, the
//! primitive builders) stays in `viewport-lib`.

pub mod marching_cubes;
pub mod volume_mesh;

pub mod prelude {
    //! The geometry entry points, in one glob import:
    //! `use viewport_lib_geometry::prelude::*;`.
    pub use crate::marching_cubes::{VolumeData, extract_isosurface};
    pub use crate::volume_mesh::{CELL_SENTINEL, VolumeMeshData};
}
