//! Checks run on `MeshData` before upload.

use viewport_lib_types::data::mesh::MeshData;
use viewport_lib_types::error::{ViewportError, ViewportResult};

/// Validate mesh data before upload.
pub fn validate_mesh_data(data: &MeshData) -> ViewportResult<()> {
    if data.positions.is_empty() || data.indices.is_empty() {
        return Err(ViewportError::EmptyMesh {
            positions: data.positions.len(),
            indices: data.indices.len(),
        });
    }
    if data.positions.len() != data.normals.len() {
        return Err(ViewportError::MeshLengthMismatch {
            positions: data.positions.len(),
            normals: data.normals.len(),
        });
    }
    let vertex_count = data.positions.len();
    for &idx in &data.indices {
        if (idx as usize) >= vertex_count {
            return Err(ViewportError::InvalidVertexIndex {
                vertex_index: idx,
                vertex_count,
            });
        }
    }
    let index_count = data.indices.len() as u64;
    for range in &data.submeshes {
        let end = range.first_index as u64 + range.index_count as u64;
        if end > index_count {
            return Err(ViewportError::SubmeshRangeOutOfBounds {
                first_index: range.first_index,
                range_count: range.index_count,
                index_count: data.indices.len(),
            });
        }
    }
    Ok(())
}
