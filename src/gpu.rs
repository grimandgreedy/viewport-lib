//! Internal alias for the wgpu version this build targets.
//!
//! The crate names `crate::gpu::*` instead of `wgpu::*` so that selecting a
//! wgpu version is this one module's concern. The `wgpu27` / `wgpu29` cargo
//! features pick the version by re-exporting the matching renamed dependency;
//! exactly one must be enabled, and only that one's crate is compiled.

#[cfg(all(feature = "wgpu27", feature = "wgpu29"))]
compile_error!("the `wgpu27` and `wgpu29` features are mutually exclusive: enable exactly one");

#[cfg(not(any(feature = "wgpu27", feature = "wgpu29")))]
compile_error!(
    "viewport-lib needs a wgpu version: enable exactly one of the `wgpu27` or `wgpu29` features (the default is `wgpu27`)"
);

// The 27 crate keeps its real name `wgpu`; only 29 is aliased.
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
pub(crate) use wgpu::*;
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
pub(crate) use wgpu29::*;

// The globs above do not bring in macros, so re-export by name the wgpu macros
// the crate uses.
#[cfg(all(feature = "wgpu27", not(feature = "wgpu29")))]
pub(crate) use wgpu::vertex_attr_array;
#[cfg(all(feature = "wgpu29", not(feature = "wgpu27")))]
pub(crate) use wgpu29::vertex_attr_array;
