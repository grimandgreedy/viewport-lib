//! Error types for the viewport library.

/// Errors that can occur during mesh upload and manipulation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ViewportError {
    /// Mesh data has empty positions or indices.
    #[error("empty mesh: {positions} positions, {indices} indices")]
    EmptyMesh {
        /// Number of positions provided.
        positions: usize,
        /// Number of indices provided.
        indices: usize,
    },

    /// Positions and normals arrays have different lengths.
    #[error("mesh length mismatch: {positions} positions vs {normals} normals")]
    MeshLengthMismatch {
        /// Number of positions provided.
        positions: usize,
        /// Number of normals provided.
        normals: usize,
    },

    /// Mesh index is out of bounds for the mesh storage.
    #[error("mesh index {index} out of bounds (mesh count: {count})")]
    MeshIndexOutOfBounds {
        /// The requested mesh index.
        index: usize,
        /// The number of meshes currently stored.
        count: usize,
    },

    /// An index buffer entry references a vertex that does not exist.
    #[error("invalid vertex index {vertex_index} (vertex count: {vertex_count})")]
    InvalidVertexIndex {
        /// The offending index value.
        vertex_index: u32,
        /// Total number of vertices.
        vertex_count: usize,
    },

    /// Texture RGBA data has an unexpected size.
    #[error("invalid texture data: expected {expected} bytes, got {actual}")]
    InvalidTextureData {
        /// Expected byte count (width * height * 4).
        expected: usize,
        /// Actual byte count provided.
        actual: usize,
    },

    /// Attempted to access or replace a mesh slot that is empty (previously removed).
    #[error("mesh slot {index} is empty")]
    MeshSlotEmpty {
        /// The slot index that was accessed.
        index: usize,
    },

    /// Named scalar attribute not found on the given mesh.
    #[error("attribute '{name}' not found on mesh {mesh_id}")]
    AttributeNotFound {
        /// The mesh index that was accessed.
        mesh_id: usize,
        /// The attribute name that was requested.
        name: String,
    },

    /// Attribute data length does not match the existing buffer.
    ///
    /// `replace_attribute` requires the same vertex count as the original upload.
    #[error("attribute length mismatch: expected {expected} f32 elements, got {got}")]
    AttributeLengthMismatch {
        /// Expected number of f32 elements (original attribute length).
        expected: usize,
        /// Actual number of f32 elements provided.
        got: usize,
    },

    /// A buffer required for GPU marching cubes upload exceeds the device's
    /// `max_buffer_size` limit. The caller should fall back to CPU marching cubes.
    #[error("GPU MC buffer '{buffer}' needs {needed} bytes but device limit is {limit}")]
    McBufferTooLarge {
        /// Short name identifying which buffer exceeded the limit (e.g. `"vertex_buf"`).
        buffer: &'static str,
        /// Bytes required for the allocation.
        needed: u64,
        /// The device's `max_buffer_size` limit.
        limit: u64,
    },
}

/// Convenience alias for `Result<T, ViewportError>`.
pub type ViewportResult<T> = Result<T, ViewportError>;
