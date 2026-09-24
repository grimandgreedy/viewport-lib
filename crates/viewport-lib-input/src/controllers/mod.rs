//! Camera controllers: orbit, first-person, third-person, turntable, and fly-to animation.
//!
//! # What each one reads
//!
//! A controller takes a resolved [`ActionFrame`](crate::input::ActionFrame) and
//! writes a [`Camera`](crate::Camera); it never sees events, and it cannot tell
//! which pointer filled the frame. That is what lets one binding set drive all of
//! them, and it is also why a controller that reads nothing from a field simply
//! ignores input bound to it.
//!
//! | Controller | Reads | Ignores |
//! | --- | --- | --- |
//! | [`OrbitCameraController`](crate::controllers::orbit::OrbitCameraController) | `navigation.orbit`, `pan`, `zoom`, `twist` | pointer flags |
//! | [`FirstPersonCameraController`](crate::controllers::first_person::FirstPersonCameraController) | `navigation.orbit`, plus the fly actions through [`wish_xy_from_actions`](crate::controllers::movement::wish_xy_from_actions) | `pan`, `zoom`, `twist` |
//! | [`ThirdPersonCameraController`](crate::controllers::third_person::ThirdPersonCameraController) | `navigation.orbit`, plus the fly actions | `pan`, `zoom`, `twist` |
//! | [`TurntableController`](crate::controllers::turntable::TurntableController) | nothing: it advances on `dt` alone | the whole frame |
//!
//! Two consequences worth knowing before switching between them.
//!
//! **Look is a drag, not mouselook.** Every controller here takes look from
//! `navigation.orbit`, which the resolver fills from a bound gesture: the middle
//! drag, ctrl+scroll, or a one-finger touch drag. So moving the mouse with nothing
//! held turns the first-person camera exactly as much as it turns the orbit camera,
//! which is not at all. Mouselook needs the raw device motion in
//! [`ViewportEvent::RawMotion`](crate::input::ViewportEvent::RawMotion), which the
//! resolver does not model and a host reads from the event stream itself, usually
//! alongside a cursor grab.
//!
//! **Zoom and pan go nowhere in the character views.** Switching from orbit to
//! first-person with the default bindings leaves the wheel bound to zoom and
//! resolving into a field nothing reads. Rebind or ignore it; the library will not
//! decide that for you.

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
