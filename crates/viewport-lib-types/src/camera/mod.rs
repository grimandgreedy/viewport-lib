//! Camera state and view math: the arcball camera, its projection and target,
//! the view frustum, keyframe tracks, and named orientation presets.
//!
//! The camera controllers (orbit, fly-to, first/third person) are input-driven
//! behaviour and stay in `viewport-lib`; this module holds the pure state and
//! math they operate on.

/// Arcball camera with perspective and orthographic projections.
pub mod camera;
/// View frustum planes and AABB culling.
pub mod frustum;
/// Keyframe camera animation with Catmull-Rom interpolation.
pub mod track;
/// Named standard camera orientations (front, top, isometric, etc.).
pub mod view_preset;

pub use camera::{Camera, CameraTarget, Projection};
