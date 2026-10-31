//! Plugin descriptor vocabulary: the pure data and traits a consumer supplies
//! to define a deformer or material plugin, and the runtime plugin handle. The
//! registration machinery that consumes these (WGSL composition, validation,
//! GPU resources) lives in `viewport-lib`.

pub mod deformer;
pub mod material;
pub mod runtime;

pub use runtime::RuntimePluginId;
