//! The shared camera rig used by every showcase: orbit and first-person
//! controllers with a backtick / button toggle and WASD + arrow-key movement.
//!
//! One rig is owned by the host and kept across showcase switches, so the camera
//! and its mode persist when you cycle showcases. Showcases never touch the
//! camera directly: they call `ctx.drive_camera()`, which runs this rig.
//!
//! The rig resolves its own input, separate from the session's. A showcase with
//! a tool drawn in the viewport (the transform gizmo) reports through
//! [`CameraRig::set_tool`] when the tool wants the pointer; the rig's resolver
//! then sees the pointer only as inside, so a press or drag on the tool does not
//! also move the camera, while the wheel still zooms.

use crate::eframe::egui;
use glam::Vec3;
use viewport_lib as vpl;
use vpl::{
    Action, BindingPreset, FirstPersonCameraController, InputOwnership, OrbitCameraController,
    PointerOwnership, ViewportContext, ViewportEvent, ViewportInput, ViewportInstance,
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
    /// The camera's own resolver; the session's resolves for picking and tools.
    input: ViewportInput,
    /// Whether a tool wanted the pointer last frame, and whether it had a
    /// session running. Set by [`set_tool`](Self::set_tool), read and cleared
    /// by [`begin_frame`](Self::begin_frame).
    tool_wants_pointer: bool,
    tool_active: bool,
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
            input: ViewportInput::from_preset(BindingPreset::Default),
            tool_wants_pointer: false,
            tool_active: false,
            orbit: OrbitCameraController::new_stateless(),
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

    /// Whether a tool had a session running last frame, for the host's egui
    /// ownership.
    pub fn tool_active(&self) -> bool {
        self.tool_active
    }

    /// Report this frame's tool state for the next frame: `wants_pointer` while a
    /// handle is hovered or a session runs, `active` while a session runs.
    pub fn set_tool(&mut self, wants_pointer: bool, active: bool) {
        self.tool_wants_pointer = wants_pointer;
        self.tool_active = active;
    }

    /// Drop the camera's gesture for a press a tool took this frame. The
    /// camera's ownership was decided last frame, so a press with no hover frame
    /// before it has already started a drag here.
    pub fn cancel_gesture(&mut self) {
        self.input.cancel_gesture();
    }

    /// Begin the camera's input frame with the viewport's `ownership`, keeping
    /// the pointer from the camera while a tool wants it. Clears the tool state
    /// so a showcase without a tool leaves none behind.
    pub fn begin_frame(&mut self, ownership: InputOwnership, size: [f32; 2]) {
        let mut ownership = ownership;
        if self.tool_wants_pointer && ownership.pointer != PointerOwnership::Elsewhere {
            ownership.pointer = PointerOwnership::Inside;
        }
        self.input
            .begin_frame(ViewportContext::new(ownership, size));
        self.orbit.set_viewport_size(size);
        self.tool_wants_pointer = false;
        self.tool_active = false;
    }

    /// Feed one event to the camera's resolver.
    pub fn push_event(&mut self, event: ViewportEvent) {
        self.input.push_event(event);
    }

    /// Drive the camera and assemble the frame. `toggle` is the backtick press
    /// this frame; `move_keys` is the held arrow-key state.
    pub fn drive(
        &mut self,
        session: &mut ViewportInstance,
        dt: f32,
        view_ctx: ViewportContext,
        toggle: bool,
        move_keys: MoveKeys,
    ) {
        let action = self.input.resolve();

        // Backtick or the overlay button toggles the controller, syncing so the
        // view does not jump.
        if toggle || self.toggle_requested {
            self.toggle_requested = false;
            self.mode = match self.mode {
                CameraMode::Orbit => {
                    self.fp.sync_from_camera(session.camera());
                    self.eye = session.camera().eye().as_vec3();
                    CameraMode::FirstPerson
                }
                CameraMode::FirstPerson => {
                    // Hand the orbit controller a sane centre ahead of the eye.
                    let eye = session.camera().eye();
                    let aim = self.fp.aim_dir();
                    let camera = session.camera_mut();
                    camera.set_orbit(eye + (aim * 8.0).as_dvec3(), 8.0, camera.orientation());
                    CameraMode::Orbit
                }
            };
        }

        match self.mode {
            CameraMode::Orbit => {
                self.orbit.apply(session.camera_mut(), &action, dt);
            }
            CameraMode::FirstPerson => {
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
        }
        session.frame(view_ctx);
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
