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

    /// A mesh's vertex or index buffer would exceed the device's maximum
    /// buffer size. Creating it anyway raises a validation error that takes
    /// the whole device down, so the upload is refused instead.
    #[error("mesh needs a {bytes} byte buffer but the device maximum is {max}")]
    MeshTooLarge {
        /// Size of the largest buffer the mesh needs.
        bytes: u64,
        /// The device's `max_buffer_size` limit.
        max: u64,
    },

    /// A position/normal override buffer was bound to a mesh whose vertex
    /// count differs from the buffer's. The shader indexes the override per
    /// mesh vertex, so a smaller buffer reads out of bounds and a different
    /// count silently scrambles the geometry.
    #[error(
        "override buffer has {override_vertices} vertices but mesh {mesh_id} has {mesh_vertices}"
    )]
    OverrideBufferVertexCountMismatch {
        /// Index of the mesh the override was bound to.
        mesh_id: usize,
        /// Vertex count of the mesh.
        mesh_vertices: usize,
        /// Vertex count the override buffer holds.
        override_vertices: usize,
    },

    /// A consumer-owned buffer handed to the renderer was created without a
    /// usage flag the binding requires (for example `STORAGE` for an
    /// external instance positions buffer, or `COPY_SRC` for an external
    /// marching-cubes scalar source).
    #[error("external buffer is missing the {missing} usage flag")]
    ExternalBufferUsageMissing {
        /// Name of the missing `wgpu::BufferUsages` flag.
        missing: &'static str,
    },

    /// An external marching-cubes scalar source does not cover the volume:
    /// the offset is not 4-byte aligned, or the buffer holds fewer bytes past
    /// the offset than the volume's `nx * ny * nz` `f32` nodes need.
    #[error(
        "external scalar source needs {needed_bytes} bytes but the buffer has {available_bytes} past offset {offset_bytes}"
    )]
    McScalarSourceMismatch {
        /// Bytes the volume's scalar field occupies (`nx * ny * nz * 4`).
        needed_bytes: u64,
        /// Bytes available in the buffer from `offset_bytes` to its end.
        available_bytes: u64,
        /// The requested byte offset into the buffer.
        offset_bytes: u64,
    },

    /// A sliced override binding does not fit inside the supplied buffer:
    /// `base_element + element_count` vec3 elements (12 bytes each) exceed
    /// the buffer's size.
    #[error(
        "override slice [{base_element}..{base_element}+{element_count}) exceeds the buffer's {buffer_elements} vec3 elements"
    )]
    OverrideSliceOutOfRange {
        /// First vec3 element of the requested window.
        base_element: u32,
        /// Number of vec3 elements in the requested window.
        element_count: u32,
        /// How many whole vec3 elements the buffer holds.
        buffer_elements: u64,
    },

    /// A handle does not resolve to a live resource: its slot was freed, reused
    /// at a newer generation, or its index is out of range for the store.
    #[error("handle {index} does not resolve to a live resource (store holds {count})")]
    StaleHandle {
        /// The slot index the handle points at.
        index: usize,
        /// The number of live entries currently in the store.
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

    /// A compressed-texture upload requested a format the device cannot sample,
    /// or a format that is not block-compressed.
    ///
    /// The device check reads the format's required wgpu feature (for example
    /// `TEXTURE_COMPRESSION_BC`, `_ASTC`, or `_ETC2`). When this is returned for
    /// a hardware reason, the caller should fall back to `upload_texture` /
    /// `upload_normal_map` or ship a variant the device supports.
    #[error(
        "unsupported texture format {format:?}: device lacks its required feature, or the format is not block-compressed"
    )]
    UnsupportedTextureFormat {
        /// The format that was rejected.
        format: crate::gpu::TextureFormat,
    },

    /// A compressed-texture upload had base dimensions that are not a multiple
    /// of the format's block size. Block-compressed textures must be created
    /// with block-aligned dimensions; pad the image (and adjust UVs) or upload
    /// it uncompressed.
    #[error(
        "compressed texture {width}x{height} is not block-aligned (block {block_width}x{block_height})"
    )]
    CompressedTextureNotBlockAligned {
        /// Base (mip 0) width in texels.
        width: u32,
        /// Base (mip 0) height in texels.
        height: u32,
        /// Format block width.
        block_width: u32,
        /// Format block height.
        block_height: u32,
    },

    /// A mip level of a compressed-texture upload had the wrong byte length for
    /// its block-packed dimensions.
    #[error(
        "invalid compressed texture data at mip {level}: expected {expected} bytes, got {actual}"
    )]
    InvalidCompressedTextureData {
        /// The mip level whose data was the wrong size (0 is the base level).
        level: u32,
        /// Expected block-packed byte count for this level.
        expected: usize,
        /// Actual byte count provided for this level.
        actual: usize,
    },

    /// Attempted to access or replace a slot that is empty (its resource was
    /// previously freed).
    #[error("slot {index} is empty")]
    SlotEmpty {
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

    /// No deformer slots remain. The registry has a fixed budget set by the
    /// vertex-stage storage-buffer limit; once all slots are taken, callers
    /// must release one with `unregister_deformer` (when implemented) before
    /// registering another.
    #[error("deformer registry exhausted: {max} slots in use")]
    DeformSlotsExhausted {
        /// Total number of slots the registry exposes.
        max: usize,
    },

    /// A deformer with this name is already registered.
    #[error("deformer name '{name}' already registered")]
    DeformNameTaken {
        /// The duplicate name that was rejected.
        name: String,
    },

    /// The composed mesh-family shader failed validation, either because the
    /// supplied `wgsl_body` is malformed or because composition produced an
    /// invalid module. The reason carries the underlying validator message
    /// and identifies which shader failed.
    #[error("deformer shader invalid: {reason}")]
    DeformShaderInvalid {
        /// Validator message prefixed by the shader name that failed.
        reason: String,
    },

    /// A shading hook with this name is already registered (or the name
    /// collides with a registered deformer; both splice `<name>__`-prefixed
    /// declarations into the same modules).
    #[error("shading hook name '{name}' already registered")]
    ShadeNameTaken {
        /// The duplicate name that was rejected.
        name: String,
    },

    /// The composed lit shader failed validation, either because the supplied
    /// `wgsl_body` is malformed, defines none of the three hook functions, or
    /// because composition produced an invalid module. The reason carries the
    /// underlying validator message and identifies which shader failed.
    #[error("shading hook shader invalid: {reason}")]
    ShadeShaderInvalid {
        /// Validator message prefixed by the shader name that failed.
        reason: String,
    },

    /// A plugin's `install` needed a piece the install context did not carry,
    /// most often a `ViewportRuntime` on a host that does not use one. The
    /// message names what was missing.
    #[error("plugin install requires {needed}, but the install context did not provide it")]
    PluginInstallMissing {
        /// Name of the missing piece, e.g. "a ViewportRuntime".
        needed: &'static str,
    },

    /// A LOD group was registered with no levels.
    #[error("LOD group has no levels")]
    LodGroupEmpty,

    /// The mesh list and threshold list passed to `register_lod_group` differ
    /// in length. Each level needs exactly one threshold.
    #[error("LOD level count mismatch: {meshes} meshes vs {thresholds} thresholds")]
    LodLevelCountMismatch {
        /// Number of meshes provided.
        meshes: usize,
        /// Number of thresholds provided.
        thresholds: usize,
    },

    /// LOD thresholds must strictly decrease from the full level to the crudest,
    /// since each level applies to a smaller on-screen size than the one before.
    #[error(
        "LOD thresholds must strictly decrease (level {level} is not smaller than the previous)"
    )]
    LodThresholdsNotDescending {
        /// The level whose threshold is not smaller than its predecessor's.
        level: usize,
    },

    /// A level's mesh is not compatible with the other levels in the group: a
    /// switch to it would change rendering. Today this means its named
    /// attribute set or its deformer attachment differs from level 0.
    #[error("LOD level {level} is incompatible with level 0: {reason}")]
    LodLevelIncompatible {
        /// The level that does not match level 0.
        level: usize,
        /// What differs (a missing attribute name, or a deform mismatch).
        reason: String,
    },

    /// A LOD group id does not refer to a registered group.
    #[error("LOD group {index} is not registered")]
    LodGroupNotFound {
        /// The group index that was looked up.
        index: usize,
    },
}

/// Convenience alias for `Result<T, ViewportError>`.
pub type ViewportResult<T> = Result<T, ViewportError>;
