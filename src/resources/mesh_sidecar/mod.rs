//! Per-mesh sidecar data attached to uploaded meshes.
//!
//! A sidecar is an opt-in piece of GPU state keyed by `MeshId`. The mesh's
//! own vertex buffer is not modified; the sidecar lives alongside it and the
//! renderer routes draws through a matching pipeline variant when one is
//! present.
//!
//! Each sidecar follows the same shape:
//!
//! - A `set_*_weights` (or equivalent) entry point on
//!   [`crate::ViewportGpuResources`] uploads the data and marks the mesh.
//! - A `bind_group_layout` is owned by the renderer and reused by every
//!   pipeline variant that consumes the sidecar.
//! - Presence of the sidecar selects the variant at draw time; absence is
//!   zero-cost.
//!
//! Today this houses the GPU skinning sidecar. The displacement sidecar
//! lands next to it.

pub(crate) mod deform;
pub(crate) mod displacement;
pub(crate) mod registry;
pub(crate) mod skin;
