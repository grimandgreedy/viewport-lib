//! Per-frame resolved action output for the new input pipeline.

use std::collections::HashMap;

use super::action::Action;

/// State of a resolved action for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResolvedActionState {
    /// The action was triggered this frame (KeyPress).
    Pressed,
    /// The action is actively held (KeyHold, Drag in progress).
    Held,
    /// The action is producing a two-axis delta (Drag, WheelXY).
    Delta(glam::Vec2),
}

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
    /// Two-finger trackpad rotation gesture delta, in radians.
    /// Positive = counter-clockwise. Zero if no gesture this frame.
    pub twist: f32,
}

/// Per-frame resolved action output.
///
/// Returned by [`super::controller::OrbitCameraController::apply_to_camera`] and
/// available from [`super::viewport_input::ViewportInput`] after a frame.
#[derive(Debug, Clone, Default)]
pub struct ActionFrame {
    /// Resolved camera navigation actions.
    pub navigation: NavigationActions,
    /// General action states resolved this frame (key presses, holds, etc.).
    pub actions: HashMap<Action, ResolvedActionState>,
    /// Characters typed this frame, for numeric manipulation input.
    ///
    /// Only populated when the app forwards [`super::event::ViewportEvent::Character`]
    /// events (which it should do only while `ManipulationController::is_active()`).
    /// Already filtered to `0-9`, `.`, `-` by the input layer.
    pub typed_chars: Vec<char>,
}

impl ActionFrame {
    /// Returns the resolved state for the given action, if active this frame.
    pub fn action(&self, action: Action) -> Option<&ResolvedActionState> {
        self.actions.get(&action)
    }

    /// Returns `true` if the action is active (pressed, held, or producing a delta) this frame.
    pub fn is_active(&self, action: Action) -> bool {
        self.actions.contains_key(&action)
    }
}
