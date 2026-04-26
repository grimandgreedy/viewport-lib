//! Named camera orientations for standard engineering views.
//!
//! Use with [`CameraAnimator::fly_to`](crate::camera::animator::CameraAnimator::fly_to)
//! for smooth animated transitions.

use crate::camera::camera::Projection;

/// Standard viewport orientations matching engineering CAD conventions (Z-up).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ViewPreset {
    /// Front view : looking from +Y toward -Y, with Z up.
    Front,
    /// Back view : looking from -Y toward +Y, with Z up.
    Back,
    /// Left view : looking from -X toward +X, with Z up.
    Left,
    /// Right view : looking from +X toward -X, with Z up.
    Right,
    /// Top view : looking down along -Z from the +Z side, with Y forward.
    Top,
    /// Bottom view : looking up along +Z from the -Z side, with Y forward.
    Bottom,
    /// True isometric view at 45° yaw (around Z) and ~35.26° pitch.
    Isometric,
}

impl ViewPreset {
    /// All preset variants in display order.
    pub fn all() -> &'static [ViewPreset] {
        &[
            Self::Front,
            Self::Back,
            Self::Left,
            Self::Right,
            Self::Top,
            Self::Bottom,
            Self::Isometric,
        ]
    }

    /// Human-readable display name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Front => "Front",
            Self::Back => "Back",
            Self::Left => "Left",
            Self::Right => "Right",
            Self::Top => "Top",
            Self::Bottom => "Bottom",
            Self::Isometric => "Isometric",
        }
    }

    /// Target camera orientation quaternion.
    ///
    /// Convention: eye = center + orientation * Vec3::Z * distance.
    /// Identity = looking down from +Z (top view in Z-up world).
    pub fn orientation(self) -> glam::Quat {
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        // Front view basis: eye at +Y, viewport up = world Z.
        // Derived: orientation * Vec3::Z = Vec3::Y, orientation * Vec3::Y = Vec3::Z.
        // This is a 180° rotation around the (0, 1, 1)/√2 axis.
        let frac_1_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
        let front = glam::Quat::from_xyzw(0.0, frac_1_sqrt_2, frac_1_sqrt_2, 0.0);
        match self {
            // Eye at +Y, up = Z.
            Self::Front => front,
            // Eye at -Y, up = Z.
            Self::Back => glam::Quat::from_rotation_x(FRAC_PI_2),
            // Eye at -X, up = Z.
            Self::Left => glam::Quat::from_rotation_z(FRAC_PI_2) * front,
            // Eye at +X, up = Z.
            Self::Right => glam::Quat::from_rotation_z(-FRAC_PI_2) * front,
            // Eye at +Z, looking down at XY plane, up = +Y (identity).
            Self::Top => glam::Quat::IDENTITY,
            // Eye at -Z, up = +Y.
            Self::Bottom => glam::Quat::from_rotation_y(PI),
            Self::Isometric => {
                // True isometric: yaw 45° around Z, then arctan(1/√2) ≈ 35.264° pitch.
                // The negative pitch keeps the eye on the +Z side so the default
                // isometric view sits above the XY plane in this Z-up world.
                let iso_pitch = (1.0_f32 / 2.0_f32.sqrt()).atan();
                glam::Quat::from_rotation_z(FRAC_PI_4)
                    * front
                    * glam::Quat::from_rotation_x(-iso_pitch)
            }
        }
    }

    /// Preferred projection for this preset, if any.
    ///
    /// Orthographic views (Front, Back, Left, Right, Top, Bottom) return
    /// `Some(Orthographic)`. Isometric returns `None` (keep current projection).
    pub fn preferred_projection(self) -> Option<Projection> {
        match self {
            Self::Isometric => None,
            _ => Some(Projection::Orthographic),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_orientations_are_unit_quaternions() {
        for preset in ViewPreset::all() {
            let q = preset.orientation();
            let len = q.length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "{:?}: quaternion length = {len}, expected 1.0",
                preset
            );
        }
    }

    #[test]
    fn test_front_eye_along_positive_y() {
        let q = ViewPreset::Front.orientation();
        // Z-up Front view: eye is along +Y (orientation * Z = Y).
        let eye_dir = q * glam::Vec3::Z;
        assert!(
            (eye_dir - glam::Vec3::Y).length() < 1e-5,
            "front eye_dir = {eye_dir:?}, expected +Y"
        );
        // Viewport up should be +Z (orientation * Y = Z).
        let up = q * glam::Vec3::Y;
        assert!(
            (up - glam::Vec3::Z).length() < 1e-5,
            "front up = {up:?}, expected +Z"
        );
    }

    #[test]
    fn test_opposite_presets_are_180_apart() {
        let pairs = [
            (ViewPreset::Front, ViewPreset::Back),
            (ViewPreset::Left, ViewPreset::Right),
            (ViewPreset::Top, ViewPreset::Bottom),
        ];
        for (a, b) in &pairs {
            let qa = a.orientation();
            let qb = b.orientation();
            // The angle between opposite views should be ~180° = π radians.
            let angle = qa.angle_between(qb);
            assert!(
                (angle - std::f32::consts::PI).abs() < 0.01,
                "{:?}/{:?}: angle = {angle:.4}, expected π",
                a,
                b
            );
        }
    }

    #[test]
    fn test_isometric_eye_is_above_xy_plane() {
        let eye_dir = ViewPreset::Isometric.orientation() * glam::Vec3::Z;
        assert!(
            eye_dir.z > 0.0,
            "isometric eye_dir = {eye_dir:?}, expected positive Z"
        );
    }

    #[test]
    fn test_all_presets_unique() {
        let all = ViewPreset::all();
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                let qi = all[i].orientation();
                let qj = all[j].orientation();
                let angle = qi.angle_between(qj);
                assert!(
                    angle > 0.01,
                    "{:?} and {:?} have nearly identical orientations (angle={angle})",
                    all[i],
                    all[j]
                );
            }
        }
    }
}
