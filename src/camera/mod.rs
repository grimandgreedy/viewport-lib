/// Arcball camera with perspective and orthographic projections.
pub mod camera;
/// Orbit, first-person, third-person, turntable, and fly-to camera controllers.
pub mod controllers;
/// View frustum planes and AABB culling.
pub mod frustum;
/// Keyframe camera animation with Catmull-Rom interpolation.
pub mod track;
/// Named standard camera orientations (front, top, isometric, etc.).
pub mod view_preset;

// Re-export the most-used types at the module level so that
// `viewport_lib::camera::Camera` continues to resolve.
pub use camera::{Camera, CameraTarget, Projection};
