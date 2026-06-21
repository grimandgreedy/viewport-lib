//! Body-attached third-person camera controller.
//!
//! [`ThirdPersonCameraController`] orbits a host-owned [`Camera`] around a
//! target point supplied each frame (typically a player body from physics).
//! Look is driven from the resolved [`ActionFrame`], decomposed into explicit
//! `yaw` and `pitch`. A `distance` boom pulls the eye back behind the target
//! along the look direction; a `height` offset raises the orbit pivot above the
//! target's feet.
//!
//! The coordinate conventions match
//! [`FirstPersonCameraController`](super::FirstPersonCameraController), so the
//! shared [`wish_xy_from_actions`](super::wish_xy_from_actions) helper produces
//! the same world-space movement vector in either view.

use glam::{Vec2, Vec3};

use super::action_frame::ActionFrame;
use super::look::{build_orientation, orientation_to_aim, yaw_pitch_from_orientation};
use crate::Camera;

/// Body-attached third-person orbit-and-boom camera controller.
///
/// Call [`apply`](Self::apply) once per frame with the resolved
/// [`ActionFrame`] and the world-space target position. The host owns the
/// [`Camera`]; this controller writes its `center` / `distance` / `orientation`
/// each frame.
pub struct ThirdPersonCameraController {
    /// Horizontal look angle in radians (rotation around world Z).
    pub yaw: f32,
    /// Vertical look angle in radians above the horizontal plane: positive
    /// looks up.
    pub pitch: f32,
    /// Horizontal look sensitivity, in radians per unit of `navigation.orbit.x`.
    pub yaw_sensitivity: f32,
    /// Vertical look sensitivity, in radians per unit of `navigation.orbit.y`.
    pub pitch_sensitivity: f32,
    /// `(min, max)` allowed pitch in radians. Defaults to roughly
    /// `(-PI/4, +PI/4)` so the camera neither dives into the ground nor flips
    /// overhead.
    pub pitch_clamp: (f32, f32),
    /// Distance from the orbit pivot to the camera eye. Larger values pull the
    /// eye further back behind the target.
    pub distance: f32,
    /// Height offset above the target used as the orbit pivot (roughly chest
    /// height for a character).
    pub height: f32,
    /// `(right, up)` offset of the orbit pivot for over-the-shoulder framing.
    /// `Vec2::ZERO` is centred.
    pub shoulder_offset: Vec2,
}

impl ThirdPersonCameraController {
    /// Default boom distance.
    pub const DEFAULT_DISTANCE: f32 = 6.0;
    /// Default pivot height above the target.
    pub const DEFAULT_HEIGHT: f32 = 1.2;

    /// Create a controller with the given look sensitivities and third-person
    /// defaults (`distance = 6.0`, `height = 1.2`, pitch clamped to
    /// `+/- PI/4`).
    pub fn new(yaw_sensitivity: f32, pitch_sensitivity: f32) -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            yaw_sensitivity,
            pitch_sensitivity,
            pitch_clamp: (-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4),
            distance: Self::DEFAULT_DISTANCE,
            height: Self::DEFAULT_HEIGHT,
            shoulder_offset: Vec2::ZERO,
        }
    }

    /// Builder: set the boom distance.
    pub fn with_distance(mut self, distance: f32) -> Self {
        self.distance = distance;
        self
    }

    /// Builder: set the pivot height above the target.
    pub fn with_height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Builder: set the `(right, up)` shoulder offset.
    pub fn with_shoulder_offset(mut self, offset: Vec2) -> Self {
        self.shoulder_offset = offset;
        self
    }

    /// Builder: set the `(min, max)` pitch clamp in radians.
    pub fn with_pitch_clamp(mut self, min: f32, max: f32) -> Self {
        self.pitch_clamp = if min <= max { (min, max) } else { (max, min) };
        self
    }

    /// Apply the frame's look delta, then orbit the camera around `target`.
    ///
    /// The orbit pivot lands `height` above the target, shifted by
    /// `shoulder_offset` in the camera-local right / up axes.
    pub fn apply(&mut self, camera: &mut Camera, frame: &ActionFrame, target: Vec3) {
        let look = frame.navigation.orbit;
        self.yaw += look.x * self.yaw_sensitivity;
        self.pitch -= look.y * self.pitch_sensitivity;
        self.pitch = self.pitch.clamp(self.pitch_clamp.0, self.pitch_clamp.1);

        let orientation = build_orientation(self.yaw, self.pitch);

        let mut pivot = target + Vec3::Z * self.height;
        if self.shoulder_offset.x != 0.0 {
            pivot += self.right_dir() * self.shoulder_offset.x;
        }
        if self.shoulder_offset.y != 0.0 {
            pivot += Vec3::Z * self.shoulder_offset.y;
        }

        camera.center = pivot;
        camera.distance = self.distance;
        camera.orientation = orientation;
    }

    /// Adopt the current view of `camera` as this controller's `yaw` / `pitch`,
    /// so switching into third-person continues from the existing view instead
    /// of snapping. Call this when entering third-person mode.
    pub fn sync_from_camera(&mut self, camera: &Camera) {
        let (yaw, pitch) = yaw_pitch_from_orientation(camera.orientation);
        self.yaw = yaw;
        self.pitch = pitch.clamp(self.pitch_clamp.0, self.pitch_clamp.1);
    }

    /// Jump to an explicit look direction. `pitch` is clamped.
    pub fn set_look(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(self.pitch_clamp.0, self.pitch_clamp.1);
    }

    /// Horizontal forward vector (yaw only, `z = 0`) for movement.
    pub fn forward_dir(&self) -> Vec3 {
        let (s, c) = self.yaw.sin_cos();
        Vec3::new(s, c, 0.0)
    }

    /// Horizontal right vector, for strafing and the shoulder offset.
    pub fn right_dir(&self) -> Vec3 {
        self.forward_dir().cross(Vec3::Z).normalize_or_zero()
    }

    /// Full look direction including pitch.
    pub fn aim_dir(&self) -> Vec3 {
        orientation_to_aim(build_orientation(self.yaw, self.pitch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> ThirdPersonCameraController {
        ThirdPersonCameraController::new(0.0035, 0.0030)
    }

    fn look(x: f32, y: f32) -> ActionFrame {
        let mut f = ActionFrame::default();
        f.navigation.orbit = Vec2::new(x, y);
        f
    }

    fn approx_eq(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-4
    }

    #[test]
    fn default_forward_is_plus_y() {
        assert!(approx_eq(cam().forward_dir(), Vec3::Y));
    }

    #[test]
    fn right_look_increases_yaw() {
        let mut c = cam();
        let mut camera = Camera::default();
        c.apply(&mut camera, &look(10.0, 0.0), Vec3::ZERO);
        assert!(c.yaw > 0.0);
    }

    #[test]
    fn pitch_clamp_is_respected() {
        let mut c = cam().with_pitch_clamp(-0.1, 0.5);
        let mut camera = Camera::default();
        c.apply(&mut camera, &look(0.0, 100.0), Vec3::ZERO);
        assert!((c.pitch - -0.1).abs() < 1e-5, "pitch={}", c.pitch);
    }

    #[test]
    fn pivot_is_above_target_and_boom_pulls_back() {
        let mut c = cam().with_distance(6.0).with_height(1.2);
        let mut camera = Camera::default();
        c.apply(&mut camera, &look(0.0, 0.0), Vec3::ZERO);
        // yaw 0, pitch 0: aim +Y, so the eye sits 6 units along -Y at z=1.2.
        let eye = camera.eye_position();
        assert!(approx_eq(eye, Vec3::new(0.0, -6.0, 1.2)), "eye={eye:?}");
    }

    #[test]
    fn shoulder_offset_shifts_pivot_right() {
        let mut c = cam()
            .with_distance(6.0)
            .with_height(1.2)
            .with_shoulder_offset(Vec2::new(0.5, 0.0));
        let mut camera = Camera::default();
        c.apply(&mut camera, &look(0.0, 0.0), Vec3::ZERO);
        assert!((camera.center.x - 0.5).abs() < 1e-5);
        assert!((camera.center.z - 1.2).abs() < 1e-5);
    }

    #[test]
    fn sync_from_camera_round_trips() {
        let mut c = cam();
        let mut camera = Camera::default();
        c.set_look(0.6, 0.2);
        c.apply(&mut camera, &look(0.0, 0.0), Vec3::ZERO);
        let mut c2 = cam();
        c2.sync_from_camera(&camera);
        assert!((c2.yaw - 0.6).abs() < 1e-4, "yaw={}", c2.yaw);
        assert!((c2.pitch - 0.2).abs() < 1e-4, "pitch={}", c2.pitch);
    }
}
