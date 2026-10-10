/// GPU marching cubes compute pipeline.
/// Scatter-volume participating-media pipeline state and uploads.
/// Unstructured volume mesh topology processing (tet / hex boundary extraction).
pub mod tetmesh;
/// Unstructured volume mesh boundary extraction. Lives in the
/// `viewport-lib-geometry` crate as `volume::mesh`; re-exported here so the
/// renderer keeps its `crate::resources::volume::volume_mesh` path.
pub use viewport_lib_geometry::volume::mesh as volume_mesh;
pub(crate) mod volumes;
