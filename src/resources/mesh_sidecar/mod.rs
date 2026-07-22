//! Per-mesh sidecar data attached to uploaded meshes.
//!
//! Per-vertex deformation (skinning, displacement, wind, morph targets, and
//! the like) flows through `deform` and `registry`: a host calls
//! `register_deformer` with a WGSL body, attaches slot data per mesh and
//! per instance through `attach_deform_slot` /
//! `attach_deform_slot_instance`, and the body runs in the standard mesh
//! shader's object-space or world-space hook. Skinning is shipped in-crate
//! against this same path on a reserved internal slot.
//!
//! The registry is the single mechanism for per-vertex deformation. There
//! used to be a parallel set of pipelines (skinned solid / shadow / outline
//! / OIT, plus a standalone vertex-displacement sidecar) that hard-coded
//! one deformation kind per pipeline; those are gone. New deformers
//! register through this module and inherit every mesh-family pass for
//! free. See [`registry::DeformerDesc`] for the hook contract,
//! composition-order rules, and pass-scope policy.

pub(crate) mod deform;
pub(crate) mod registry;
pub(crate) mod shade;
