/// The camera: pose, projection and exposure.
pub mod camera;
/// Orbit, first-person, third-person, turntable, and fly-to camera controllers.
/// Camera controllers. Live in the `viewport-lib-input` crate; re-exported here
/// so the renderer keeps its `crate::camera::controllers` path.
pub use viewport_lib_input::controllers;
/// View frustum planes and AABB culling.
pub mod frustum;
/// Keyframe camera animation with Catmull-Rom interpolation.
pub mod track;
/// Named standard camera orientations (front, top, isometric, etc.).
pub mod view_preset;

// Re-export the most-used types at the module level so that
// `viewport_lib::camera::Camera` continues to resolve.
pub use camera::{Camera, CameraTarget, PhysicalExposure, Pose, Projection};
#[cfg(feature = "serde")]
pub use viewport_lib_types::camera::camera_v0::{CameraV0, ProjectionV0};
pub use viewport_lib_types::camera::look::{self, LookAngles};
