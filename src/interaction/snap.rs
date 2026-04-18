//! Pure-math snap helpers for gizmo transforms.
//!
//! All functions are stateless — the application wraps gizmo drag results with
//! these when snapping is enabled.

/// Configuration for transform snapping.
#[derive(Clone, Debug, Default)]
pub struct SnapConfig {
    /// Translation snap increment in world units (e.g. 0.25, 0.5, 1.0).
    pub translation: Option<f32>,
    /// Rotation snap increment in radians (e.g. `PI / 12` for 15°).
    pub rotation: Option<f32>,
    /// Scale snap increment as a fraction (e.g. 0.1 for 10% steps).
    pub scale: Option<f32>,
}

/// Snap a scalar value to the nearest increment.
///
/// Returns `value` unchanged if `increment <= 0`.
pub fn snap_value(value: f32, increment: f32) -> f32 {
    if increment <= 0.0 {
        return value;
    }
    (value / increment).round() * increment
}

/// Snap each component of a `Vec3` to the nearest increment.
pub fn snap_vec3(v: glam::Vec3, increment: f32) -> glam::Vec3 {
    glam::Vec3::new(
        snap_value(v.x, increment),
        snap_value(v.y, increment),
        snap_value(v.z, increment),
    )
}

/// Snap an angle (in radians) to the nearest increment.
pub fn snap_angle(angle_rad: f32, increment_rad: f32) -> f32 {
    snap_value(angle_rad, increment_rad)
}

/// Snap a scale factor to the nearest increment.
///
/// Operates the same as [`snap_value`]; named separately for discoverability.
pub fn snap_scale(scale: f32, increment: f32) -> f32 {
    snap_value(scale, increment)
}

/// Describes a visual overlay for an active constraint (data only, not rendering).
///
/// The application layer maps these to actual draw calls.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ConstraintOverlay {
    /// An infinite line through `origin` along `direction`.
    AxisLine {
        /// World-space point on the line.
        origin: glam::Vec3,
        /// Unit direction vector of the line.
        direction: glam::Vec3,
        /// RGBA display color.
        color: [f32; 4],
    },
    /// A plane through `origin` spanned by `axis_a` and `axis_b`.
    Plane {
        /// World-space point on the plane.
        origin: glam::Vec3,
        /// First tangent axis of the plane.
        axis_a: glam::Vec3,
        /// Second tangent axis of the plane.
        axis_b: glam::Vec3,
        /// RGBA display color.
        color: [f32; 4],
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snap_value_rounds_to_nearest() {
        assert!((snap_value(0.7, 0.5) - 0.5).abs() < 1e-6);
        assert!((snap_value(0.8, 0.5) - 1.0).abs() < 1e-6);
        assert!((snap_value(-0.3, 0.5) - -0.5).abs() < 1e-6);
        assert!((snap_value(0.25, 0.5) - 0.5).abs() < 1e-6); // 0.25/0.5=0.5, rounds to 1, *0.5=0.5
    }

    #[test]
    fn test_snap_vec3_per_component() {
        let v = glam::Vec3::new(0.7, 1.3, -0.8);
        let snapped = snap_vec3(v, 0.5);
        assert!((snapped.x - 0.5).abs() < 1e-6);
        assert!((snapped.y - 1.5).abs() < 1e-6);
        assert!((snapped.z - -1.0).abs() < 1e-6);
    }

    #[test]
    fn test_snap_angle_15_degrees() {
        let deg15 = std::f32::consts::PI / 12.0;
        let deg20 = 20.0_f32.to_radians();
        let deg40 = 40.0_f32.to_radians();
        let snapped_20 = snap_angle(deg20, deg15);
        let snapped_40 = snap_angle(deg40, deg15);
        // 20° -> 15° (1 × 15)
        assert!(
            (snapped_20 - deg15).abs() < 1e-5,
            "20° snapped to {}, expected {}",
            snapped_20.to_degrees(),
            15.0
        );
        // 40° -> 45° (3 × 15)
        let deg45 = 45.0_f32.to_radians();
        assert!(
            (snapped_40 - deg45).abs() < 1e-5,
            "40° snapped to {}, expected {}",
            snapped_40.to_degrees(),
            45.0
        );
    }

    #[test]
    fn test_snap_scale_around_one() {
        let snapped = snap_scale(1.37, 0.1);
        assert!(
            (snapped - 1.4).abs() < 1e-5,
            "1.37 @ 0.1 -> {snapped}, expected 1.4"
        );
    }

    #[test]
    fn test_snap_config_none_passthrough() {
        let config = SnapConfig::default();
        // When increments are None, a typical usage pattern:
        let value = 1.234;
        let result = config
            .translation
            .map(|inc| snap_value(value, inc))
            .unwrap_or(value);
        assert!((result - value).abs() < 1e-6, "None should pass through");
    }
}
