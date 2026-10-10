//! Deformers: WGSL bodies the renderer splices into every mesh-family pass.
//!
//! A deformer is registered once with `DeviceResources::register_deformer` and
//! runs for any mesh draw that carries data in its slot, per mesh or per
//! instance. Because it is spliced into the colour, transparent, shadow,
//! outline and pick passes alike, what it does to a mesh holds in all of them.
//!
//! Each deformer here is a handle type with an `install` call, in the module
//! named after it:
//!
//! - [`cut`]: removes part of a mesh, for section views and scalar
//!   thresholds.

pub mod cut;
