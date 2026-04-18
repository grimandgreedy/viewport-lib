/// Smooth camera motion with exponential damping and fly-to animations.
pub mod animator;
/// Arcball camera with perspective and orthographic projections.
pub mod camera;
/// View frustum planes and AABB culling.
pub mod frustum;
/// Named standard camera orientations (front, top, isometric, etc.).
pub mod view_preset;

// Re-export the most-used types at the module level so that
// `viewport_lib::camera::Camera` continues to resolve.
pub use camera::{Camera, CameraTarget, Projection};
