//! Turntable (continuous orbit) camera controller.

use crate::camera::camera::Camera;

// ---------------------------------------------------------------------------
// TurntableController
// ---------------------------------------------------------------------------

/// Continuously orbits the camera around the Z axis at a fixed elevation.
///
/// Each frame, advance the azimuth by `angular_velocity * dt` and apply the
/// result to the camera. The orbit distance and center are left unchanged.
///
/// ```rust,ignore
/// let mut turntable = TurntableController::from_camera(&camera, 0.5);
/// // in the render loop:
/// turntable.update(dt, &mut camera);
/// ```
#[derive(Clone, Debug)]
pub struct TurntableController {
    /// Angular velocity in radians per second. Positive rotates counter-clockwise
    /// when viewed from above (Z-up world).
    pub angular_velocity: f32,
    /// Pitch angle in radians (angle from the vertical axis). A value of PI/4
    /// places the eye 45 degrees from the top. Matches the convention used by
    /// `Camera::orbit` where `Quat::from_rotation_x(tilt)` is the pitch.
    pub tilt: f32,
    /// Current azimuth angle in radians.
    pub azimuth: f32,
}

impl TurntableController {
    /// Create a turntable that starts at azimuth zero.
    ///
    /// `tilt` is the pitch angle in radians: `PI/2` looks from the side
    /// (horizontal), smaller values look more from above.
    pub fn new(angular_velocity: f32, tilt: f32) -> Self {
        Self {
            angular_velocity,
            tilt,
            azimuth: 0.0,
        }
    }

    /// Create a turntable that continues from the camera's current orientation.
    ///
    /// The azimuth and tilt are extracted from the camera's current orientation
    /// so the orbit continues smoothly from wherever the user left off.
    pub fn from_camera(camera: &Camera, angular_velocity: f32) -> Self {
        let eye_dir = camera.orientation() * glam::Vec3::Z;
        // Tilt (colatitude): angle from +Z. cos(tilt) = eye_dir.z.
        let tilt = eye_dir.z.clamp(-1.0, 1.0).acos();
        // Azimuth: atan2 of the horizontal projection.
        // From the camera model: eye_dir = [sin(az)*sin(tilt), -cos(az)*sin(tilt), cos(tilt)]
        // So az = atan2(eye_dir.x, -eye_dir.y).
        let azimuth = eye_dir.x.atan2(-eye_dir.y);
        Self {
            angular_velocity,
            tilt,
            azimuth,
        }
    }

    /// Advance the turntable by `dt` seconds and write the new orientation into
    /// `camera`. Distance and center are unchanged.
    pub fn update(&mut self, dt: f32, camera: &mut Camera) {
        self.azimuth += self.angular_velocity * dt;
        // Normalize azimuth to [-PI, PI] to avoid float drift over time.
        self.azimuth = normalize_angle(self.azimuth);
        camera.set_orientation(
            glam::Quat::from_rotation_z(self.azimuth) * glam::Quat::from_rotation_x(self.tilt),
        );
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Wrap `angle` into `[-PI, PI]`.
fn normalize_angle(angle: f32) -> f32 {
    use std::f32::consts::TAU;
    let a = angle % TAU;
    if a > std::f32::consts::PI {
        a - TAU
    } else if a < -std::f32::consts::PI {
        a + TAU
    } else {
        a
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_camera() -> Camera {
        Camera::default()
    }

    #[test]
    fn test_turntable_advances_azimuth() {
        let mut cam = default_camera();
        let mut tt = TurntableController::new(1.0, 1.0);
        let start_az = tt.azimuth;
        tt.update(0.1, &mut cam);
        assert!((tt.azimuth - (start_az + 0.1)).abs() < 1e-5);
    }

    #[test]
    fn test_turntable_does_not_change_distance() {
        let mut cam = default_camera();
        cam.set_distance(7.5);
        let mut tt = TurntableController::new(1.0, 1.0);
        tt.update(1.0, &mut cam);
        assert!((cam.distance() - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_turntable_does_not_change_center() {
        let mut cam = default_camera();
        cam.set_center(glam::Vec3::new(1.0, 2.0, 3.0));
        let mut tt = TurntableController::new(1.0, 1.0);
        tt.update(1.0, &mut cam);
        assert!((cam.center() - glam::Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn test_from_camera_round_trip() {
        let cam = default_camera();
        let tt = TurntableController::from_camera(&cam, 0.5);
        // Apply to a fresh camera and check orientation matches original.
        let mut cam2 = default_camera();
        cam2.set_orientation(
            glam::Quat::from_rotation_z(tt.azimuth) * glam::Quat::from_rotation_x(tt.tilt),
        );
        let orig_eye = cam.orientation() * glam::Vec3::Z;
        let new_eye = cam2.orientation() * glam::Vec3::Z;
        let diff = (orig_eye - new_eye).length();
        assert!(diff < 1e-4, "round-trip eye direction diff={diff}");
    }

    #[test]
    fn test_azimuth_normalization() {
        let mut cam = default_camera();
        // Spin fast for many frames -- azimuth should stay bounded.
        let mut tt = TurntableController::new(10.0, 1.0);
        for _ in 0..1000 {
            tt.update(0.1, &mut cam);
        }
        assert!(
            tt.azimuth.abs() <= std::f32::consts::PI + 1e-4,
            "azimuth out of range: {}",
            tt.azimuth
        );
    }

    #[test]
    fn test_negative_velocity_reverses() {
        let mut cam = default_camera();
        let mut tt_fwd = TurntableController::new(1.0, 1.0);
        let mut tt_rev = TurntableController::new(-1.0, 1.0);
        tt_fwd.update(0.5, &mut cam);
        tt_rev.update(0.5, &mut cam);
        assert!(tt_fwd.azimuth > 0.0, "forward azimuth should be positive");
        assert!(tt_rev.azimuth < 0.0, "reverse azimuth should be negative");
    }
}
