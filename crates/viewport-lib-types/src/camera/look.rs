//! A look direction as yaw and pitch, and its conversion to a [`Camera`]
//! orientation.
//!
//! [`Camera`]: crate::camera::Camera

use glam::{Quat, Vec3};

/// A look direction as yaw and pitch, Z-up.
///
/// - `yaw` turns around world Z. Yaw 0 looks along +Y and increasing yaw turns
///   toward +X (clockwise seen from above).
/// - `pitch` is the angle above the horizontal plane: positive looks up.
///
/// There is no roll. [`orientation`](Self::orientation) gives the matching
/// [`Camera::orientation`](crate::camera::Camera::orientation), and
/// [`from_orientation`](Self::from_orientation) reads one back, dropping any
/// roll it carried. The first- and third-person controllers store their look
/// this way.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LookAngles {
    /// Rotation around world Z, in radians.
    pub yaw: f32,
    /// Angle above the horizontal plane, in radians.
    pub pitch: f32,
}

impl LookAngles {
    /// Looking along +Y, level.
    pub const FORWARD: Self = Self::new(0.0, 0.0);

    /// Build from yaw and pitch in radians.
    pub const fn new(yaw: f32, pitch: f32) -> Self {
        Self { yaw, pitch }
    }

    /// Read the look out of a camera orientation. Roll is dropped.
    ///
    /// Looking straight up or down leaves yaw undefined; the result then takes
    /// whatever yaw the rounding gives.
    pub fn from_orientation(orientation: Quat) -> Self {
        Self::from_direction(-(orientation * Vec3::Z))
    }

    /// Read the look out of a direction. A zero direction gives
    /// [`FORWARD`](Self::FORWARD).
    pub fn from_direction(direction: Vec3) -> Self {
        let Some(d) = direction.try_normalize() else {
            return Self::FORWARD;
        };
        Self {
            yaw: d.x.atan2(d.y),
            pitch: d.z.clamp(-1.0, 1.0).asin(),
        }
    }

    /// The camera orientation for this look.
    pub fn orientation(self) -> Quat {
        // The camera looks along its local -Z, and a zero orientation looks
        // straight down, so pitch is offset by a quarter turn.
        (Quat::from_rotation_z(-self.yaw)
            * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2 + self.pitch))
        .normalize()
    }

    /// The full look direction, pitch included. Unit length.
    pub fn direction(self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, cy * cp, sp)
    }

    /// The look direction flattened onto the ground plane, for movement. Unit
    /// length.
    pub fn forward(self) -> Vec3 {
        let (s, c) = self.yaw.sin_cos();
        Vec3::new(s, c, 0.0)
    }

    /// The ground-plane right of [`forward`](Self::forward), for strafing.
    /// Unit length.
    pub fn right(self) -> Vec3 {
        let (s, c) = self.yaw.sin_cos();
        Vec3::new(c, -s, 0.0)
    }

    /// Turned by `yaw` and `pitch` radians. Pitch is not clamped; follow with
    /// [`with_pitch_clamped`](Self::with_pitch_clamped).
    pub fn turned(self, yaw: f32, pitch: f32) -> Self {
        Self {
            yaw: self.yaw + yaw,
            pitch: self.pitch + pitch,
        }
    }

    /// With pitch held to `min..=max`.
    pub fn with_pitch_clamped(self, min: f32, max: f32) -> Self {
        Self {
            yaw: self.yaw,
            pitch: self.pitch.clamp(min, max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-4
    }

    #[test]
    fn forward_looks_along_plus_y() {
        let look = LookAngles::FORWARD;
        assert!(close(look.direction(), Vec3::Y));
        assert!(close(-(look.orientation() * Vec3::Z), Vec3::Y));
    }

    #[test]
    fn yaw_turns_toward_plus_x() {
        let look = LookAngles::new(FRAC_PI_2, 0.0);
        assert!(close(look.direction(), Vec3::X));
        assert!(close(look.forward(), Vec3::X));
        assert!(close(look.right(), -Vec3::Y));
    }

    #[test]
    fn positive_pitch_looks_up() {
        assert!(LookAngles::new(0.0, 0.5).direction().z > 0.0);
        assert!(LookAngles::new(0.0, -0.5).direction().z < 0.0);
    }

    #[test]
    fn direction_matches_orientation() {
        for &(yaw, pitch) in &[(0.0, 0.0), (0.7, 0.3), (-2.1, -0.9), (3.0, 1.2)] {
            let look = LookAngles::new(yaw, pitch);
            let from_quat = -(look.orientation() * Vec3::Z);
            assert!(close(look.direction(), from_quat), "{look:?}");
        }
    }

    #[test]
    fn orientation_round_trips() {
        for &(yaw, pitch) in &[(0.0, 0.0), (0.7, 0.3), (-2.1, -0.9), (3.0, 1.2)] {
            let look = LookAngles::new(yaw, pitch);
            let back = LookAngles::from_orientation(look.orientation());
            assert!(close(back.direction(), look.direction()), "{look:?}");
            assert!((back.pitch - pitch).abs() < 1e-4, "{look:?}");
        }
    }

    #[test]
    fn right_is_forward_cross_up() {
        let look = LookAngles::new(0.9, 0.4);
        assert!(close(look.right(), look.forward().cross(Vec3::Z)));
    }

    #[test]
    fn camera_right_matches_look_right_when_level() {
        let look = LookAngles::new(1.3, 0.0);
        assert!(close(look.orientation() * Vec3::X, look.right()));
    }

    #[test]
    fn zero_direction_is_forward() {
        assert_eq!(LookAngles::from_direction(Vec3::ZERO), LookAngles::FORWARD);
    }

    #[test]
    fn clamp_holds_pitch_only() {
        let look = LookAngles::new(1.0, 2.0).with_pitch_clamped(-1.0, 1.0);
        assert_eq!(look, LookAngles::new(1.0, 1.0));
    }
}
