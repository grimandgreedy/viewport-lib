/// Geometry helpers for building glyph and primitive meshes.
pub(crate) mod geometry;
/// Per-mesh GPU buffers and bind group.
pub(crate) mod gpu_mesh;
pub(crate) mod instancing;
pub mod lod;
/// Factory functions for the mesh-family pipelines that share a single
/// shader source. Used at init time and on `register_deformer` rebuild.
pub(crate) mod mesh_pipelines;
/// Slotted GPU mesh storage with free-list removal.
pub mod mesh_store;
pub(crate) mod meshes;
