/// Arcball camera with perspective and orthographic projections.
pub mod camera;
/// Smooth camera motion with exponential damping and fly-to animations.
pub mod animator;
/// Named standard camera orientations (front, top, isometric, etc.).
pub mod view_preset;
/// View frustum planes and AABB culling.
pub mod frustum;

// Re-export the most-used types at the module level so that
// `viewport_lib::camera::Camera` continues to resolve.
pub use camera::{Camera, Projection};
