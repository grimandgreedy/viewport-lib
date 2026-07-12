//! Internal alias for the wgpu version this build targets.
//!
//! The crate names `crate::gpu::*` instead of `wgpu::*` so that selecting a
//! wgpu version is this one module's concern. Today it re-exports the single
//! pinned wgpu; the version-window feature selection (renamed `wgpu27` /
//! `wgpu29` dependencies) slots in here without touching call sites.

pub(crate) use wgpu::*;

// The glob above does not bring in macros, so re-export by name the wgpu macros
// the crate uses.
pub(crate) use wgpu::vertex_attr_array;
