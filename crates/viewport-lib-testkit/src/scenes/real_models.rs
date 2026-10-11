//! Optional loading of real models through `viewport-lib-io` (feature
//! `real_models`).
//!
//! The procedural corpus covers concavity we control; this brings in real-world
//! topology (Suzanne, the Stanford bunny, a CAD part) for cases procedural
//! shapes do not represent well. Off by default so the base build needs no model
//! files.

use std::path::Path;
use viewport_lib::MeshData;

/// Load an STL file as `MeshData`. io fills in normals when the file has none.
pub fn load_stl(path: impl AsRef<Path>) -> Result<MeshData, viewport_lib_io::IoError> {
    viewport_lib_io::loaders::stl::mesh_from_path(path.as_ref())
}
