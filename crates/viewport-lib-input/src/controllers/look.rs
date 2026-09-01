//! Shared look-basis math for the body-attached character cameras.
//!
//! Both [`crate::controllers::first_person::FirstPersonCameraController`] and
//! [`crate::controllers::third_person::ThirdPersonCameraController`] decompose the look direction into
//! explicit `yaw` and `pitch` angles (Z-up): yaw rotates around world Z with
//! yaw 0 looking along +Y, pitch is the angle above the horizontal plane.

use glam::{Quat, Vec3};

/// Build the camera orientation quaternion from yaw and pitch.
///
/// At yaw 0, pitch 0 the aim is +Y (horizontal forward). Increasing pitch looks
/// up; increasing yaw rotates clockwise from above (+Y toward +X). The
/// `FRAC_PI_2 + pitch` term converts the from-horizontal pitch used here into
/// the from-overhead convention the viewport `Camera` orientation expects.
pub(super) fn build_orientation(yaw: f32, pitch: f32) -> Quat {
    (Quat::from_rotation_z(-yaw) * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2 + pitch))
        .normalize()
}

/// World-space aim direction for a camera orientation (the camera looks along
/// -Z in camera space).
pub(super) fn orientation_to_aim(orientation: Quat) -> Vec3 {
    -(orientation * Vec3::Z)
}

/// Decompose an orientation into `(yaw, pitch)`, the inverse of
/// [`build_orientation`] away from the straight-up / straight-down singularities.
///
/// Used by `sync_from_camera` so switching into a character camera continues
/// from the current view rather than snapping.
pub(super) fn yaw_pitch_from_orientation(orientation: Quat) -> (f32, f32) {
    let aim = orientation_to_aim(orientation);
    let yaw = aim.x.atan2(aim.y);
    let pitch = aim.z.clamp(-1.0, 1.0).asin();
    (yaw, pitch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn zero_yaw_pitch_aims_along_plus_y() {
        // The documented convention: yaw 0, pitch 0 looks along +Y (Z-up).
        let aim = orientation_to_aim(build_orientation(0.0, 0.0));
        assert!((aim - Vec3::Y).length() < 1e-4, "aim was {aim:?}");
    }

    #[test]
    fn positive_pitch_looks_up_negative_looks_down() {
        let up = orientation_to_aim(build_orientation(0.0, 0.5));
        assert!(up.z > 0.0, "positive pitch should aim upward, got {up:?}");
        let down = orientation_to_aim(build_orientation(0.0, -0.5));
        assert!(
            down.z < 0.0,
            "negative pitch should aim downward, got {down:?}"
        );
    }

    #[test]
    fn yaw_pitch_decompose_reconstructs_the_same_aim() {
        // yaw_pitch_from_orientation is the inverse of build_orientation away
        // from the straight-up/down singularities: decomposing then rebuilding
        // must reproduce the same aim direction (what sync_from_camera relies on).
        for &(yaw, pitch) in &[
            (0.0, 0.0),
            (0.5, 0.3),
            (-1.2, -0.4),
            (2.0, 0.7),
            (-2.7, 0.1),
        ] {
            let aim = orientation_to_aim(build_orientation(yaw, pitch));
            let (y2, p2) = yaw_pitch_from_orientation(build_orientation(yaw, pitch));
            let aim2 = orientation_to_aim(build_orientation(y2, p2));
            assert!(
                (aim - aim2).length() < 1e-4,
                "round trip changed aim: {aim:?} -> {aim2:?} (yaw {yaw}, pitch {pitch})"
            );
        }
    }

    #[test]
    fn aim_is_always_unit_length() {
        for &(yaw, pitch) in &[(0.0, 0.0), (1.0, FRAC_PI_2 - 0.01), (-1.0, -1.0)] {
            let aim = orientation_to_aim(build_orientation(yaw, pitch));
            assert!((aim.length() - 1.0).abs() < 1e-4, "aim {aim:?} not unit");
        }
    }
}
