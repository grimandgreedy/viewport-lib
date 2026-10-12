//! Unstructured volume mesh processing: boundary and clipped surface
//! extraction, tet decomposition, and hex meshes from occupied grid cells. The
//! cell conventions are documented on [`VolumeMeshData`].

mod boundary;
mod cells;
mod clip;
mod grid_cells;
#[cfg(test)]
mod tests;
mod tets;

pub use boundary::extract_boundary_faces;
pub use cells::TET_FACES;
pub use clip::extract_clipped_volume_faces;
pub use grid_cells::GridCells;
pub use tets::{for_each_tet, tet_scalars};
pub use viewport_lib_types::data::volume_mesh::{
    CELL_SENTINEL, ConversionReport, ToTetMeshError, VolumeMeshData,
};
