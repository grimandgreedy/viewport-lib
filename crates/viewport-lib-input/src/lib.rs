//! Input handling for `viewport-lib`: a replaceable layer, not baked-in core.
//!
//! The design intent is that a consumer can bring their own. Input handling
//! splits into two halves, and either is usable on its own:
//!
//! 1. **Resolve**: [`input::ViewportInput`] turns `ViewportEvent`s into a single
//!    [`ActionFrame`](viewport_lib_types::input::ActionFrame) per frame.
//! 2. **Apply**: a camera controller takes that `ActionFrame` and a
//!    `&mut Camera` and produces a new pose. This is the half a custom controller
//!    replaces; see [`CameraController`].
//!
//! The built-in [`controllers`] (orbit, first/third-person, turntable, and the
//! `dt`-driven animator) are the batteries-included defaults. The winit/egui
//! [adapters](input::adapters) that translate native events are optional
//! features (`winit-adapter`, `egui-adapter`), so the crate builds with no
//! framework dependency by default.
//!
//! This crate carries no `wgpu` and no renderer. It stands on
//! `viewport-lib-types` (the input vocabulary and the `Camera` type) plus `glam`.
//! `viewport-lib` re-exports it, so `viewport_lib::OrbitCameraController` and
//! friends keep working.

#![allow(deprecated)]

// Re-exported so controllers can name `crate::Camera` and consumers can get the
// camera type from this crate.
pub use viewport_lib_types::camera::Camera;

/// The batteries-included camera controllers.
pub mod controllers;
/// The event resolver, the legacy query model, and the native-event adapters.
pub mod input;

/// A camera controller: the "apply" half of input handling.
///
/// Implement this to drive a `Camera` from a resolved
/// [`ActionFrame`](viewport_lib_types::input::ActionFrame) however you like - the
/// built-in controllers (`OrbitCameraController`, ...) are one set of policies,
/// not the only ones. Resolve the frame with [`input::ViewportInput`], then hand
/// it to your controller each frame.
///
/// The built-in controllers implement this, but the trait is not required to use
/// them: their inherent `apply` methods and free-function forms still work. It
/// exists so a custom controller has a concrete shape to fill and so controllers
/// are interchangeable by type where a consumer wants that.
pub trait CameraController {
    /// Apply one frame of resolved input to the camera.
    fn apply(&mut self, camera: &mut Camera, frame: &viewport_lib_types::input::ActionFrame);
}

impl CameraController for controllers::OrbitCameraController {
    fn apply(&mut self, camera: &mut Camera, frame: &viewport_lib_types::input::ActionFrame) {
        // Delegate to the inherent apply, which drives the camera per the
        // controller's `navigation_mode`.
        controllers::OrbitCameraController::apply(self, camera, frame)
    }
}
// The first/third-person controllers take an extra per-frame position (eye /
// target), so they do not fit this two-argument shape; use their inherent
// `apply` methods directly. See `controllers`.

pub mod prelude {
    //! The input types most code reaches for, in one glob import:
    //! `use viewport_lib_input::prelude::*;`.
    //!
    //! The common set, not the whole surface. Reach into [`crate::input`] and
    //! [`crate::controllers`] for the rest.
    pub use crate::CameraController;
    pub use crate::controllers::{
        CameraAnimator, FirstPersonCameraController, OrbitCameraController,
        ThirdPersonCameraController, TurntableController,
    };
    pub use crate::input::ViewportInput;
    pub use viewport_lib_types::camera::Camera;
    pub use viewport_lib_types::input::{Action, ActionFrame, ViewportContext, ViewportEvent};
}
