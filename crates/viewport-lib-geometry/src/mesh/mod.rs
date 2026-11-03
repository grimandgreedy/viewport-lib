//! Operations on triangle meshes: attribute expansion, tangents, validation,
//! clip-plane caps and isolines.
//!
//! These operate on plain slices and `MeshData`, with no GPU dependency, so a
//! loader or mesh-processing consumer can run them without the renderer. The
//! renderer's upload path calls them before handing data to the GPU.

mod attributes;
pub mod cap;
pub mod isoline;
mod tangents;
mod validate;

pub use attributes::{
    expand_cell_to_vertex, expand_edge_to_vertex, expand_face_colours_to_3n,
    expand_face_scalars_to_3n,
};
pub use tangents::compute_tangents;
pub use validate::validate_mesh_data;
