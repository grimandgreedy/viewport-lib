//! Error types for the viewport library.

/// Errors that can occur during mesh upload and manipulation.
#[derive(Debug, Clone, thiserror::Error)]
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

    /// Volume scalar data length does not match the declared grid dimensions.
    #[error("volume data length {actual} does not match dims {dims:?} (expected {expected})")]
    VolumeDataLengthMismatch {
        /// Number of scalar values provided.
        actual: usize,
        /// Expected length, equal to `dims[0] * dims[1] * dims[2]`.
        expected: usize,
        /// Grid dimensions `[nx, ny, nz]`.
        dims: [u32; 3],
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

    /// Gaussian splat data is malformed or empty.
    #[error("invalid gaussian splat data: {reason}")]
    InvalidGaussianSplatData {
        /// Short description of what is wrong.
        reason: &'static str,
    },

    /// A background upload worker panicked or dropped its channel before
    /// sending a result.
    #[error("upload worker did not complete: {reason}")]
    JobWorkerLost {
        /// Short description of how the worker was lost.
        reason: &'static str,
    },

    /// `upload_result_*` was called before the corresponding job reached the
    /// `Ready` state. Poll `upload_status` and retry on the next frame.
    #[error("upload job is not yet ready to take its result")]
    JobNotReady,

    /// `upload_result_*` was called with a `JobId` that does not belong to
    /// the matching upload type, has already been taken, or was never
    /// issued.
    #[error("upload job has no result available: {reason}")]
    JobResultMissing {
        /// Short description of why the result is unavailable.
        reason: &'static str,
    },
}

/// Convenience alias for `Result<T, ViewportError>`.
pub type ViewportResult<T> = Result<T, ViewportError>;
