//! Optional loading of real models through `viewport-lib-io` (feature
//! `real_models`).
//!
//! The procedural corpus covers concavity we control; this brings in real-world
//! topology (Suzanne, the Stanford bunny, a CAD part) for cases procedural
//! shapes do not represent well. Off by default so the base build needs no model
//! files.

use std::path::Path;
use viewport_lib::MeshData;
use viewport_lib_io::SurfaceMesh;

/// Convert an io `SurfaceMesh` into a `viewport-lib` `MeshData`, recomputing
/// normals when the source has none.
fn convert(m: SurfaceMesh) -> MeshData {
    let normals = if m.normals.len() == m.positions.len() && !m.normals.is_empty() {
        m.normals
    } else {
        super::meshes::recompute_normals(&m.positions, &m.indices)
    };
    let mut out = MeshData::default();
    out.positions = m.positions;
    out.normals = normals;
    out.indices = m.indices;
    out.uvs = m.uvs;
    out
}

/// Load an STL file as `MeshData`.
pub fn load_stl(path: impl AsRef<Path>) -> Result<MeshData, viewport_lib_io::IoError> {
    Ok(convert(viewport_lib_io::loaders::stl::mesh_from_path(
        path.as_ref(),
    )?))
}
