//! Per-mesh sidecar data attached to uploaded meshes.
//!
//! Per-vertex deformation (skinning, displacement, wind, morph targets, and
//! the like) flows through `deform` and `registry`: a host calls
//! `register_deformer` with a WGSL body, attaches slot data per mesh and
//! per instance through `attach_deform_slot` /
//! `attach_deform_slot_instance`, and the body runs in the standard mesh
//! shader's object-space or world-space hook. Skinning is shipped in-crate
//! against this same path on a reserved internal slot.

pub(crate) mod deform;
pub(crate) mod registry;
pub(crate) mod skin;
