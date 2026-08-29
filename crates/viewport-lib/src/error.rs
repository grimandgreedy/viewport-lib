//! Error types for the viewport library.
//!
//! `ViewportError` and `ViewportResult` live in `viewport-lib-types` so CPU-side
//! tools (loaders, mesh processors) can produce and match the same errors
//! without the renderer. They are re-exported here so the renderer's
//! `crate::error::*` paths keep resolving.

pub use viewport_lib_types::error::{ViewportError, ViewportResult};
