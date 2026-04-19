//! Per-frame resolved action output for the new input pipeline.

use super::viewport_binding::ViewportAction;

/// Resolved navigation actions for one frame.
///
/// Produced by [`super::viewport_input::ViewportInput`] after processing all
/// events for a frame. Non-zero fields indicate active input in that direction.
#[derive(Debug, Clone, Default)]
pub struct NavigationActions {
    /// Orbit delta in radians (x = yaw, y = pitch). Zero if no orbit input.
    pub orbit: glam::Vec2,
    /// Pan delta in viewport-local pixels (x = right, y = down). Zero if no pan input.
    pub pan: glam::Vec2,
    /// Zoom factor delta. Positive = zoom in. Zero if no zoom input.
    pub zoom: f32,
}

/// Per-frame resolved action output.
///
/// Returned by [`super::controller::OrbitCameraController::apply_to_camera`] and
/// available from [`super::viewport_input::ViewportInput`] after a frame.
#[derive(Debug, Clone, Default)]
pub struct ActionFrame {
    /// Resolved camera navigation actions.
    pub navigation: NavigationActions,
}

impl ActionFrame {
    /// Returns the raw delta for the given action, or `glam::Vec2::ZERO` if inactive.
    pub fn delta(&self, action: ViewportAction) -> glam::Vec2 {
        match action {
            ViewportAction::Orbit => self.navigation.orbit,
            ViewportAction::Pan => self.navigation.pan,
            ViewportAction::Zoom => glam::Vec2::new(self.navigation.zoom, self.navigation.zoom),
        }
    }
}
