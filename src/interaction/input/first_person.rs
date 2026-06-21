//! Body-attached first-person camera controller.
//!
//! [`FirstPersonCameraController`] drives the look direction of a host-owned
//! [`Camera`] from the resolved [`ActionFrame`] and attaches the eye to a
//! world-space position supplied each frame (typically a player body from
//! physics). Look is decomposed into explicit `yaw` and `pitch` so clamping is
//! clean and the movement basis is cheap to derive.
//!
//! # Coordinate conventions (Z-up)
//!
//! - `yaw` rotates around world Z: yaw 0 looks along +Y, yaw `+PI/2` looks
//!   along +X.
//! - `pitch` is the angle above the horizontal plane: positive looks up,
//!   clamped to `+/- pitch_clamp`.
//!
//! Unlike [`OrbitCameraController`](super::OrbitCameraController), this
//! controller does not own a camera: call
//! [`apply`](FirstPersonCameraController::apply) with the host's `&mut Camera`.
//! Look is read from `frame.navigation.orbit`, so the same input pipeline and
//! binding presets that drive orbit also drive first-person look.

use glam::Vec3;

use super::action_frame::ActionFrame;
use super::look::{build_orientation, orientation_to_aim, yaw_pitch_from_orientation};
use crate::Camera;

/// Body-attached first-person camera controller.
///
/// Call [`apply`](Self::apply) once per frame with the resolved
/// [`ActionFrame`] and the world-space eye position. The host owns the
/// [`Camera`]; this controller writes its `center` / `distance` / `orientation`
/// each frame. Read [`forward_dir`](Self::forward_dir) /
/// [`right_dir`](Self::right_dir) to drive character movement.
pub struct FirstPersonCameraController {
    /// Horizontal look angle in radians (rotation around world Z).
    pub yaw: f32,
    /// Vertical look angle in radians above the horizontal plane: positive
    /// looks up.
    pub pitch: f32,
    /// Look sensitivity, in radians per unit of `navigation.orbit` delta.
    pub sensitivity: f32,
    /// Maximum absolute pitch in radians before clamping (e.g. `1.4` is roughly
    /// 80 degrees).
    pub pitch_clamp: f32,
}

impl FirstPersonCameraController {
    /// Default look sensitivity applied to the resolved orbit delta.
    pub const DEFAULT_SENSITIVITY: f32 = 0.005;
    /// Default pitch clamp, roughly 80 degrees.
    pub const DEFAULT_PITCH_CLAMP: f32 = 1.4;

    /// Create a controller with the given sensitivity and pitch clamp, looking
    /// along +Y.
    pub fn new(sensitivity: f32, pitch_clamp: f32) -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            sensitivity,
            pitch_clamp: pitch_clamp.abs(),
        }
    }

    /// Apply the frame's look delta, then attach the camera eye to
    /// `eye_position`.
    ///
    /// `eye_position` is the world-space eye point, typically the player body
    /// origin offset by eye height.
    pub fn apply(&mut self, camera: &mut Camera, frame: &ActionFrame, eye_position: Vec3) {
        let look = frame.navigation.orbit;
        // Mouse right -> yaw increases (turn right, clockwise from above).
        // Mouse down -> pitch decreases (look down).
        self.yaw += look.x * self.sensitivity;
        self.pitch -= look.y * self.sensitivity;
        self.pitch = self.pitch.clamp(-self.pitch_clamp, self.pitch_clamp);

        let orientation = build_orientation(self.yaw, self.pitch);
        // distance 1 keeps the orbit centre one unit ahead of the eye so the
        // view matrix is non-degenerate; it does not affect rendering.
        camera.center = eye_position + orientation_to_aim(orientation);
        camera.distance = 1.0;
        camera.orientation = orientation;
    }

    /// Adopt the current view of `camera` as this controller's `yaw` / `pitch`,
    /// so switching into first-person continues from the existing view instead
    /// of snapping. Call this when entering first-person mode.
    pub fn sync_from_camera(&mut self, camera: &Camera) {
        let (yaw, pitch) = yaw_pitch_from_orientation(camera.orientation);
        self.yaw = yaw;
        self.pitch = pitch.clamp(-self.pitch_clamp, self.pitch_clamp);
    }

    /// Jump to an explicit look direction. `pitch` is clamped.
    pub fn set_look(&mut self, yaw: f32, pitch: f32) {
        self.yaw = yaw;
        self.pitch = pitch.clamp(-self.pitch_clamp, self.pitch_clamp);
    }

    /// Horizontal forward vector (yaw only, `z = 0`) for movement. At yaw 0:
    /// `+Y`; at yaw `PI/2`: `+X`.
    pub fn forward_dir(&self) -> Vec3 {
        let (s, c) = self.yaw.sin_cos();
        Vec3::new(s, c, 0.0)
    }

    /// Horizontal right vector perpendicular to [`forward_dir`](Self::forward_dir),
    /// for strafing.
    pub fn right_dir(&self) -> Vec3 {
        self.forward_dir().cross(Vec3::Z).normalize_or_zero()
    }

    /// Full look direction including pitch, for raycasts and aim.
    pub fn aim_dir(&self) -> Vec3 {
        orientation_to_aim(build_orientation(self.yaw, self.pitch))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn cam() -> FirstPersonCameraController {
        FirstPersonCameraController::new(0.002, 1.4)
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
    fn default_aim_is_forward_along_y() {
        assert!(approx_eq(cam().aim_dir(), Vec3::Y));
    }

    #[test]
    fn right_look_increases_yaw() {
        let mut c = cam();
        let mut camera = Camera::default();
        c.apply(&mut camera, &look(100.0, 0.0), Vec3::ZERO);
        assert!(c.yaw > 0.0, "look right should increase yaw");
    }

    #[test]
    fn down_look_decreases_pitch() {
        let mut c = cam();
        let mut camera = Camera::default();
        c.apply(&mut camera, &look(0.0, 100.0), Vec3::ZERO);
        assert!(c.pitch < 0.0, "look down should decrease pitch");
    }

    #[test]
    fn pitch_is_clamped() {
        let mut c = FirstPersonCameraController::new(1.0, 1.0);
        let mut camera = Camera::default();
        c.apply(&mut camera, &look(0.0, -9999.0), Vec3::ZERO);
        assert!(c.pitch <= 1.0 && c.pitch >= -1.0, "pitch={}", c.pitch);
    }

    #[test]
    fn eye_attaches_to_supplied_position() {
        let mut c = cam();
        let mut camera = Camera::default();
        let eye = Vec3::new(3.0, 4.0, 1.8);
        c.apply(&mut camera, &look(0.0, 0.0), eye);
        assert!((camera.eye_position() - eye).length() < 1e-4);
    }

    #[test]
    fn forward_dir_is_horizontal_and_rotates() {
        let mut c = cam();
        c.set_look(std::f32::consts::FRAC_PI_2, 0.0);
        let fwd = c.forward_dir();
        assert!(fwd.z.abs() < 1e-6, "forward must be horizontal");
        assert!(approx_eq(fwd, Vec3::X), "forward at yaw=PI/2 should be +X");
    }

    #[test]
    fn sync_from_camera_round_trips() {
        let mut c = cam();
        let mut camera = Camera::default();
        c.set_look(0.7, 0.3);
        c.apply(&mut camera, &look(0.0, 0.0), Vec3::ZERO);
        // A fresh controller adopting that camera recovers the same look.
        let mut c2 = cam();
        c2.sync_from_camera(&camera);
        assert!((c2.yaw - 0.7).abs() < 1e-4, "yaw={}", c2.yaw);
        assert!((c2.pitch - 0.3).abs() < 1e-4, "pitch={}", c2.pitch);
    }
}
