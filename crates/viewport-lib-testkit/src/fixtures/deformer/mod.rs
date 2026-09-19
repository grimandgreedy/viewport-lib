//! Fixtures for [`DeformerDesc`](viewport_lib::DeformerDesc): a WGSL body
//! spliced into the mesh shader family, registered with
//! `DeviceResources::register_deformer`.
//!
//! A deformer is data, not a trait impl, so the fixtures here are
//! constructors returning a descriptor plus the helper that attaches the slot
//! data a body needs to run.
//!
//! - [`constant_offset_deformer`] / [`ConstantOffsetDeformer`]: moves every
//!   vertex by a constant read from the deformer's slot params.
//! - [`per_vertex_offset_deformer`] / [`PerVertexOffsetDeformer`]: moves each
//!   vertex by its own value, read from the slot's per-vertex data.

mod constant_offset;
mod per_vertex_offset;

pub use constant_offset::{ConstantOffsetDeformer, constant_offset_deformer};
pub use per_vertex_offset::{PerVertexOffsetDeformer, per_vertex_offset_deformer};
