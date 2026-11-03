//! Camera state and view math: the camera's pose, projection and exposure,
//! the orbit target, the view frustum, keyframe tracks, and named orientation
//! presets.
//!
//! The camera controllers (orbit, fly-to, first/third person) are input-driven
//! behaviour and live in `viewport-lib-input`; this module holds the pure state
//! and math they operate on.

/// The camera and its orbit target.
pub mod camera;
/// Reading cameras saved in the orbit form.
#[cfg(feature = "serde")]
pub mod camera_v0;
/// Physical exposure settings and focus distance.
pub mod exposure;
/// View frustum planes and AABB culling.
pub mod frustum;
/// A look direction as yaw and pitch.
pub mod look;
/// Eye position and orientation.
pub mod pose;
/// Perspective, orthographic and host-built projections.
pub mod projection;
/// Keyframe camera animation with Catmull-Rom interpolation.
pub mod track;
/// Named standard camera orientations (front, top, isometric, etc.).
pub mod view_preset;

pub use camera::{Camera, CameraTarget};
#[cfg(feature = "serde")]
pub use camera_v0::{CameraV0, ProjectionV0};
pub use exposure::PhysicalExposure;
pub use look::LookAngles;
pub use pose::Pose;
pub use projection::Projection;
