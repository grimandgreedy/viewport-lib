//! Calibration for touch gesture recognition.

/// The numbers the touch recogniser is calibrated on.
///
/// Every distance is in logical pixels and every time is in seconds, so a value
/// means the same thing on a phone and on a desktop touchscreen. The defaults are
/// what the built-in presets use; override the ones your device or your users
/// disagree with.
///
/// Sensitivities are multipliers over the resolved delta, and a negative one
/// inverts that axis. They exist because "drag up to look up" and "drag up to look
/// down" are both conventions people hold strongly, and neither is the library's to
/// pick.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TouchSettings {
    /// Multiplier on one-finger orbit, per axis. Negative inverts that axis.
    pub orbit_sensitivity: glam::Vec2,
    /// Multiplier on two-finger pan, per axis. Negative inverts that axis.
    pub pan_sensitivity: glam::Vec2,
    /// Multiplier on pinch-to-zoom. Negative swaps spread and pinch.
    pub pinch_sensitivity: f32,
    /// Multiplier on two-finger twist. Negative reverses the direction.
    pub twist_sensitivity: f32,
    /// How far a contact may travel and still count as a tap.
    pub tap_tolerance: f32,
    /// Longest gap between two taps for the second to count as a double tap.
    pub double_tap_interval: f32,
    /// How far apart two taps may land and still count as a double tap.
    pub double_tap_distance: f32,
    /// How long a contact must stay down to count as a long press.
    pub long_press_dwell: f32,
    /// How far a contact may travel before it can no longer become a long press.
    pub long_press_tolerance: f32,
}

impl Default for TouchSettings {
    fn default() -> Self {
        Self {
            orbit_sensitivity: glam::Vec2::ONE,
            pan_sensitivity: glam::Vec2::ONE,
            pinch_sensitivity: 1.0,
            twist_sensitivity: 1.0,
            // A finger is a blunt instrument: the mouse click threshold of 5px is
            // too tight for one, and the platforms land around 10.
            tap_tolerance: 10.0,
            double_tap_interval: 0.3,
            double_tap_distance: 40.0,
            long_press_dwell: 0.5,
            long_press_tolerance: 10.0,
        }
    }
}
