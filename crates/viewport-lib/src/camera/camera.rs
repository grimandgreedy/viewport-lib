//! The camera: pose, projection and exposure.
//!
//! `Camera`, `Pose`, `Projection`, `PhysicalExposure` and `CameraTarget` live in
//! `viewport-lib-types` so CPU-side tools can drive view math without the
//! renderer. They are re-exported here so the renderer's
//! `crate::camera::camera::*` paths keep resolving.

pub use viewport_lib_types::camera::camera::*;
pub use viewport_lib_types::camera::exposure::PhysicalExposure;
pub use viewport_lib_types::camera::pose::Pose;
pub use viewport_lib_types::camera::projection::{DEFAULT_FAR, Projection};
