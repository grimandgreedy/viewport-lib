//! The shared camera rig used by every showcase: orbit and first-person
//! controllers with a backtick / button toggle and WASD + arrow-key movement.
//!
//! One rig is owned by the host and kept across showcase switches, so the camera
//! and its mode persist when you cycle showcases. Showcases never touch the
//! camera directly: they call `ctx.drive_camera()`, which runs this rig.

use eframe::egui;
use glam::Vec3;
use viewport_lib as vpl;
use vpl::{
    Action, FirstPersonCameraController, OrbitCameraController, ViewportContext, ViewportInstance,
};

#[derive(Clone, Copy, PartialEq)]
enum CameraMode {
    Orbit,
    FirstPerson,
}

/// Held arrow/WASD movement intent for one frame. WASD is resolved through the
/// input actions; the arrow keys are read as raw held state and passed in here.
#[derive(Clone, Copy, Default)]
pub struct MoveKeys {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
}

pub struct CameraRig {
    orbit: OrbitCameraController,
    fp: FirstPersonCameraController,
    mode: CameraMode,
    /// First-person eye position, moved by WASD/arrows; the FP controller looks.
    eye: Vec3,
    move_speed: f32,
    /// Set by the toggle button; consumed on the next `drive`, like a backtick.
    toggle_requested: bool,
}

impl CameraRig {
    pub fn new() -> Self {
        Self {
            orbit: OrbitCameraController::viewport_all(),
            fp: FirstPersonCameraController::new(
                FirstPersonCameraController::DEFAULT_SENSITIVITY,
                FirstPersonCameraController::DEFAULT_PITCH_CLAMP,
            ),
            mode: CameraMode::Orbit,
            eye: Vec3::ZERO,
            move_speed: 6.0,
            toggle_requested: false,
        }
    }

    /// Drive the camera and assemble the frame. `toggle` is the backtick press
    /// this frame; `move_keys` is the held arrow-key state. Camera motion is left
    /// to the session's manipulation while a G/R/S drag owns the pointer.
    pub fn drive(
        &mut self,
        session: &mut ViewportInstance,
        dt: f32,
        view_ctx: ViewportContext,
        toggle: bool,
        move_keys: MoveKeys,
    ) {
        let action = session.resolve().clone();

        // Backtick or the overlay button toggles the controller, syncing so the
        // view does not jump.
        if toggle || self.toggle_requested {
            self.toggle_requested = false;
            self.mode = match self.mode {
                CameraMode::Orbit => {
                    self.fp.sync_from_camera(session.camera());
                    self.eye = session.camera().eye_position();
                    CameraMode::FirstPerson
                }
                CameraMode::FirstPerson => {
                    // Hand the orbit controller a sane centre ahead of the eye.
                    let eye = session.camera().eye_position();
                    let aim = self.fp.aim_dir();
                    let camera = session.camera_mut();
                    camera.center = eye + aim * 8.0;
                    camera.distance = 8.0;
                    CameraMode::Orbit
                }
            };
        }

        match self.mode {
            CameraMode::Orbit => {
                session.update_orbit(&mut self.orbit);
            }
            CameraMode::FirstPerson => {
                // A manipulation drag owns the pointer; do not also look/move then.
                if !session.is_manipulating() {
                    let mut dir = Vec3::ZERO;
                    if action.is_active(Action::FlyForward) || move_keys.forward {
                        dir += self.fp.forward_dir();
                    }
                    if action.is_active(Action::FlyBackward) || move_keys.back {
                        dir -= self.fp.forward_dir();
                    }
                    if action.is_active(Action::FlyRight) || move_keys.right {
                        dir += self.fp.right_dir();
                    }
                    if action.is_active(Action::FlyLeft) || move_keys.left {
                        dir -= self.fp.right_dir();
                    }
                    if action.is_active(Action::FlyUp) {
                        dir += Vec3::Z;
                    }
                    if action.is_active(Action::FlyDown) {
                        dir -= Vec3::Z;
                    }
                    let speed = if action.is_active(Action::FlySpeedBoost) {
                        self.move_speed * 3.0
                    } else {
                        self.move_speed
                    };
                    if dir != Vec3::ZERO {
                        self.eye += dir.normalize() * speed * dt;
                    }
                    self.fp.apply(session.camera_mut(), &action, self.eye);
                }
                session.frame(view_ctx);
            }
        }
    }

    /// The `orbit | fly` toggle, drawn as a viewport overlay by the host.
    pub fn overlay(&mut self, ui: &mut egui::Ui) {
        let active = match self.mode {
            CameraMode::Orbit => 0,
            CameraMode::FirstPerson => 1,
        };
        // Two modes, so any pick is a toggle; defer it to the next `drive`.
        if crate::ui::segmented(ui, active, &["orbit", "fly"]).is_some() {
            self.toggle_requested = true;
        }
    }

    /// The general camera controls, listed in the `?` modal above the active
    /// showcase's own controls.
    pub fn controls(&self, ui: &mut egui::Ui) {
        ui.strong("Camera");
        ui.label("` or the top-right toggle: switch orbit / fly");
        ui.label("Fly mode: WASD or arrow keys to move");
    }
}
