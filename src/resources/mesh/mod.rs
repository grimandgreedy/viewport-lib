pub(crate) mod instancing;
pub mod lod;
/// Factory functions for the mesh-family pipelines that share a single
/// shader source. Used at init time and on `register_deformer` rebuild.
pub(crate) mod mesh_pipelines;
/// Slotted GPU mesh storage with free-list removal.
pub mod mesh_store;
mod meshes;
