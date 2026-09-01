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
