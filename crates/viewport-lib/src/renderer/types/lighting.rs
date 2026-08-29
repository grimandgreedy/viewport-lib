//! Lighting and shadow configuration.
//!
//! These settings live in `viewport-lib-types` so CPU-side tools can build
//! lighting config without the renderer. They are re-exported here so the
//! renderer's `crate::renderer::types::lighting::*` paths keep resolving.

pub use viewport_lib_types::effects::lighting::*;
