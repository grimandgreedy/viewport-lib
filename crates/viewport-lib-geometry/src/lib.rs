//! CPU geometry algorithms that operate on `viewport-lib-types` payloads without
//! touching the GPU.
//!
//! - [`primitives`]: built-in meshes (cube, sphere, torus and the rest).
//! - [`mesh`]: operations on triangle meshes, including clip-plane caps and
//!   isolines.
//! - [`volume`]: regular scalar grids, marching cubes isosurfaces, and
//!   unstructured volume meshes.
//! - [`maths`]: small helpers such as ray/plane intersection.
//!
//! Everything returns plain data, usually a
//! [`MeshData`](viewport_lib_types::data::mesh::MeshData) the renderer can
//! upload, but nothing here needs the renderer to run.

pub mod maths;
pub mod mesh;
pub mod primitives;
pub mod volume;

mod util;

pub mod prelude {
    //! The geometry entry points, in one glob import:
    //! `use viewport_lib_geometry::prelude::*;`.
    pub use crate::volume::grid::VolumeData;
    pub use crate::volume::marching_cubes::extract_isosurface;
    pub use crate::volume::mesh::{CELL_SENTINEL, GridCells, VolumeMeshData};
}
