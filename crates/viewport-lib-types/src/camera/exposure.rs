//! The camera's physical exposure settings.

use crate::effects::postprocess::ExposureMode;

/// Aperture, shutter and sensitivity of a physical camera, plus the distance
/// it focuses at.
///
/// The defaults are the sunny-16 rule: f/16, 1/125 s, ISO 100 (about
/// EV100 15), focused 5 units out.
///
/// The renderer reads exposure from `PostProcessSettings::exposure`, not from
/// here. To expose a frame with these settings, set that field's mode to
/// [`to_exposure_mode`](Self::to_exposure_mode).
///
/// `focus_distance` is also the distance [`Camera::target`](crate::camera::Camera::target)
/// reports and that the automatic depth range reaches past; orbit controllers
/// set it to their orbit distance.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[non_exhaustive]
pub struct PhysicalExposure {
    /// f-number `N`. Larger is a smaller aperture and a darker image.
    pub aperture: f32,
    /// Shutter time in seconds. Longer is brighter.
    pub shutter_seconds: f32,
    /// Sensor sensitivity (ISO). Higher is brighter.
    pub iso: f32,
    /// Distance from the eye to the point in focus, in world units.
    pub focus_distance: f32,
}

impl Default for PhysicalExposure {
    fn default() -> Self {
        Self {
            aperture: 16.0,
            shutter_seconds: 1.0 / 125.0,
            iso: 100.0,
            focus_distance: 5.0,
        }
    }
}

impl PhysicalExposure {
    /// An exposure from aperture, shutter time and ISO, with the default
    /// focus distance.
    pub fn new(aperture: f32, shutter_seconds: f32, iso: f32) -> Self {
        Self {
            aperture,
            shutter_seconds,
            iso,
            ..Self::default()
        }
    }

    /// Builder: set the focus distance.
    pub fn with_focus_distance(mut self, focus_distance: f32) -> Self {
        self.focus_distance = focus_distance;
        self
    }

    /// The matching [`ExposureMode::PhysicalCamera`], for
    /// `PostProcessSettings::exposure`.
    pub fn to_exposure_mode(&self) -> ExposureMode {
        ExposureMode::PhysicalCamera {
            aperture: self.aperture,
            shutter: self.shutter_seconds,
            iso: self.iso,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::postprocess::ExposureSettings;

    #[test]
    fn default_is_sunny_sixteen() {
        let ev = ExposureSettings::from_mode(PhysicalExposure::default().to_exposure_mode())
            .base_ev100()
            .unwrap();
        assert!((ev - 15.0).abs() < 0.1, "ev={ev}");
    }

    #[test]
    fn to_exposure_mode_carries_the_triangle() {
        let mode = PhysicalExposure::new(2.8, 0.5, 400.0).to_exposure_mode();
        assert_eq!(
            mode,
            ExposureMode::PhysicalCamera {
                aperture: 2.8,
                shutter: 0.5,
                iso: 400.0
            }
        );
    }
}
