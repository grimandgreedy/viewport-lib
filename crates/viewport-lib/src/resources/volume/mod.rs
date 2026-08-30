/// GPU marching cubes compute pipeline.
pub mod gpu_marching_cubes;
/// GPU implicit surface types and pipeline.
pub mod implicit;
/// Scatter-volume participating-media pipeline state and uploads.
pub mod scatter_volume;
/// Sparse voxel grid topology processing (boundary face extraction).
pub mod sparse_volume;
/// Unstructured volume mesh topology processing (tet / hex boundary extraction).
pub mod tetmesh;
/// Unstructured volume mesh boundary extraction. Lives in the
/// `viewport-lib-geometry` crate; re-exported here so the renderer keeps its
/// `crate::resources::volume::volume_mesh` path.
pub use viewport_lib_geometry::volume_mesh;
pub(crate) mod volumes;
