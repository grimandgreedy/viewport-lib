//! Named camera orientations for standard engineering views.
//!
//! Use with [`CameraAnimator::fly_to`](crate::camera::animator::CameraAnimator::fly_to)
//! for smooth animated transitions.

use crate::camera::camera::Projection;

/// Standard viewport orientations matching engineering CAD conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ViewPreset {
    /// Front view — looking along -Z from the +Z side.
    Front,
    /// Back view — looking along +Z from the -Z side.
    Back,
    /// Left view — looking along +X from the -X side.
    Left,
    /// Right view — looking along -X from the +X side.
    Right,
    /// Top view — looking down along -Y from the +Y side.
    Top,
    /// Bottom view — looking up along +Y from the -Y side.
    Bottom,
    /// True isometric view at 45° yaw and ~35.26° pitch.
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
    /// Convention: camera looks along -Z in its local space, so identity = front view.
    pub fn orientation(self) -> glam::Quat {
        use std::f32::consts::PI;
        match self {
            Self::Front => glam::Quat::IDENTITY,
            Self::Back => glam::Quat::from_rotation_y(PI),
            Self::Left => glam::Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
            Self::Right => glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            Self::Top => glam::Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            Self::Bottom => glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            Self::Isometric => {
                // True isometric: rotate 45° around Y, then arctan(1/√2) ≈ 35.264° around X.
                let iso_pitch = (1.0_f32 / 2.0_f32.sqrt()).atan();
                glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_4)
                    * glam::Quat::from_rotation_x(iso_pitch)
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
    fn test_front_looks_along_negative_z() {
        let q = ViewPreset::Front.orientation();
        // Identity quaternion: camera at +Z looking at origin (-Z direction).
        let forward = q * glam::Vec3::NEG_Z;
        assert!(
            (forward - glam::Vec3::NEG_Z).length() < 1e-5,
            "front forward = {forward:?}, expected -Z"
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
