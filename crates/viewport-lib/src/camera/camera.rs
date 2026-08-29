//! Arcball camera with perspective and orthographic projections.
//!
//! `Camera`, `CameraTarget`, and `Projection` live in `viewport-lib-types` so
//! CPU-side tools can drive view math without the renderer. They are re-exported
//! here so the renderer's `crate::camera::camera::*` paths keep resolving.

pub use viewport_lib_types::camera::camera::*;
