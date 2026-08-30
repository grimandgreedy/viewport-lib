//! Camera controllers: orbit, first-person, third-person, turntable, and fly-to animation.

/// Smooth camera motion with exponential damping and fly-to animations.
pub mod animator;
/// Body-attached first-person camera controller.
pub mod first_person;
/// Shared look-basis math for the character cameras.
mod look;
/// Movement-input helper for the character cameras.
pub mod movement;
/// High-level orbit/pan/zoom camera controller.
pub mod orbit;
/// Body-attached third-person camera controller.
pub mod third_person;
/// Continuous turntable (orbit) camera controller.
pub mod turntable;

pub use animator::{CameraAnimator, CameraDamping, Easing};
pub use first_person::FirstPersonCameraController;
pub use movement::wish_xy_from_actions;
pub use orbit::OrbitCameraController;
pub use third_person::ThirdPersonCameraController;
pub use turntable::TurntableController;
