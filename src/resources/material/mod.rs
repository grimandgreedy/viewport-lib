/// Built-in colourmap LUT data.
pub mod colourmap_data;
/// IBL precomputation and environment map upload.
pub mod environment;
/// GPU compute path for IBL precomputation (selected at runtime when supported).
mod ibl_compute;
/// Built-in matcap texture data (procedurally generated).
pub mod matcap_data;
/// Slotted texture storage with generational ids.
pub(crate) mod texture_store;
pub(crate) mod textures;
