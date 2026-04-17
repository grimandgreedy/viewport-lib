use super::binding::{ActivationMode, KeyCode, Modifiers, MouseButton, TriggerKind};
use std::collections::HashSet;

/// Per-frame input snapshot populated by the egui adapter.
///
/// This is framework-agnostic - the egui-specific translation happens in the
/// host application's viewport adapter.
#[derive(Debug, Clone)]
pub struct FrameInput {
    /// Keys pressed this frame (rising edge).
    pub keys_pressed: HashSet<KeyCode>,
    /// Keys held this frame.
    pub keys_held: HashSet<KeyCode>,
    /// Current modifier state.
    pub modifiers: Modifiers,
    /// Mouse buttons that started a drag this frame.
    pub drag_started: HashSet<MouseButton>,
    /// Mouse buttons currently being dragged.
    pub dragging: HashSet<MouseButton>,
    /// Accumulated drag delta this frame (pixels).
    pub drag_delta: glam::Vec2,
    /// Scroll delta this frame (positive = up/zoom-in).
    pub scroll_delta: f32,
    /// Mouse buttons clicked this frame (press + release without drag).
    pub clicked: HashSet<MouseButton>,
    /// Raw pointer movement delta this frame (pixels), regardless of button state.
    /// Used for manipulation modes where mouse movement alone drives transforms.
    pub pointer_delta: glam::Vec2,
    /// Whether the input area (viewport) is hovered.
    pub hovered: bool,
    /// Ctrl+scroll orbit delta in logical pixels (x = yaw, y = pitch).
    /// Read from raw MouseWheel events to preserve 2D direction — smooth_scroll_delta
    /// loses directional data when modifiers are held.
    pub ctrl_scroll_orbit_delta: glam::Vec2,
    /// Shift+scroll pan delta in logical pixels (x = right, y = up).
    /// Read from raw MouseWheel events for the same reason as ctrl_scroll_orbit_delta.
    pub shift_scroll_pan_delta: glam::Vec2,
}

impl Default for FrameInput {
    fn default() -> Self {
        Self {
            keys_pressed: Default::default(),
            keys_held: Default::default(),
            modifiers: Default::default(),
            drag_started: Default::default(),
            dragging: Default::default(),
            drag_delta: glam::Vec2::ZERO,
            scroll_delta: 0.0,
            clicked: Default::default(),
            pointer_delta: glam::Vec2::ZERO,
            hovered: false,
            ctrl_scroll_orbit_delta: glam::Vec2::ZERO,
            shift_scroll_pan_delta: glam::Vec2::ZERO,
        }
    }
}

/// Result of querying whether an action is active this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActionState {
    /// Action is not active.
    Inactive,
    /// Action was just triggered (single-fire).
    Pressed,
    /// Action is continuously active with an associated delta.
    Active {
        /// Motion delta for this frame (pixels for drag/scroll actions, zero for key-held).
        delta: glam::Vec2,
    },
}

impl ActionState {
    /// Returns true if the action was pressed this frame.
    pub fn pressed(self) -> bool {
        matches!(self, ActionState::Pressed)
    }

    /// Returns true if the action is active (pressed or held with delta).
    pub fn is_active(self) -> bool {
        !matches!(self, ActionState::Inactive)
    }

    /// Extract delta if active, otherwise zero.
    pub fn delta(self) -> glam::Vec2 {
        match self {
            ActionState::Active { delta } => delta,
            _ => glam::Vec2::ZERO,
        }
    }
}

/// Check whether a trigger's modifiers match the current frame modifiers.
pub(crate) fn modifiers_match(required: &Modifiers, current: &Modifiers, ignore: bool) -> bool {
    if ignore {
        return true;
    }
    required == current
}

/// Evaluate a single trigger against the current frame input.
pub(crate) fn evaluate_trigger(
    kind: &TriggerKind,
    activation: &ActivationMode,
    required_mods: &Modifiers,
    ignore_mods: bool,
    input: &FrameInput,
) -> ActionState {
    if !modifiers_match(required_mods, &input.modifiers, ignore_mods) {
        return ActionState::Inactive;
    }

    match (kind, activation) {
        (TriggerKind::Key(key), ActivationMode::OnPress) => {
            if input.keys_pressed.contains(key) {
                ActionState::Pressed
            } else {
                ActionState::Inactive
            }
        }
        (TriggerKind::Key(key), ActivationMode::WhileHeld) => {
            if input.keys_held.contains(key) {
                ActionState::Active {
                    delta: glam::Vec2::ZERO,
                }
            } else {
                ActionState::Inactive
            }
        }
        (TriggerKind::MouseButton(btn), ActivationMode::OnPress) => {
            if input.clicked.contains(btn) {
                ActionState::Pressed
            } else {
                ActionState::Inactive
            }
        }
        (TriggerKind::MouseButton(btn), ActivationMode::OnDrag) => {
            if input.dragging.contains(btn) {
                ActionState::Active {
                    delta: input.drag_delta,
                }
            } else {
                ActionState::Inactive
            }
        }
        (TriggerKind::Scroll, ActivationMode::OnScroll) => {
            if input.scroll_delta.abs() > 0.0 {
                ActionState::Active {
                    delta: glam::Vec2::new(0.0, input.scroll_delta),
                }
            } else {
                ActionState::Inactive
            }
        }
        _ => ActionState::Inactive,
    }
}
