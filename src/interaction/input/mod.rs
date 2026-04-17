//! Input system: action-based input mapping with mode-sensitive bindings.
//!
//! Decouples semantic actions (Orbit, Pan, Zoom, ...) from physical triggers
//! (key/mouse combinations), enabling future key reconfiguration and
//! context-sensitive controls (Normal / FlyMode / Manipulating).

/// Semantic action enum.
pub mod action;
/// Binding, trigger, and modifier types.
pub mod binding;
/// Default key/mouse bindings for the viewport.
pub mod defaults;
/// Input mode enum (Normal, FlyMode, Manipulating).
pub mod mode;
/// Per-frame input snapshot and action-state query evaluation.
pub mod query;

pub use action::Action;
pub use binding::{ActivationMode, Binding, KeyCode, Modifiers, MouseButton, Trigger, TriggerKind};
pub use defaults::default_bindings;
pub use mode::InputMode;
pub use query::{ActionState, FrameInput};

/// Central input system that evaluates action queries against the current
/// binding table and input mode.
pub struct InputSystem {
    bindings: Vec<Binding>,
    mode: InputMode,
}

impl InputSystem {
    /// Create a new input system with default bindings in Normal mode.
    pub fn new() -> Self {
        Self {
            bindings: default_bindings(),
            mode: InputMode::Normal,
        }
    }

    /// Current input mode.
    pub fn mode(&self) -> InputMode {
        self.mode
    }

    /// Set the input mode.
    pub fn set_mode(&mut self, mode: InputMode) {
        self.mode = mode;
    }

    /// Query whether an action is active this frame.
    ///
    /// Iterates bindings matching the action and current mode, evaluates
    /// each trigger against the frame input. First match wins.
    pub fn query(&self, action: Action, input: &FrameInput) -> ActionState {
        for binding in &self.bindings {
            if binding.action != action {
                continue;
            }
            // Check mode filter.
            if !binding.active_modes.is_empty() && !binding.active_modes.contains(&self.mode) {
                continue;
            }
            let state = query::evaluate_trigger(
                &binding.trigger.kind,
                &binding.trigger.activation,
                &binding.trigger.modifiers,
                binding.trigger.ignore_modifiers,
                input,
            );
            if !matches!(state, ActionState::Inactive) {
                return state;
            }
        }
        ActionState::Inactive
    }

    /// Access the current binding table.
    pub fn bindings(&self) -> &[Binding] {
        &self.bindings
    }

    /// Replace the binding table.
    pub fn set_bindings(&mut self, bindings: Vec<Binding>) {
        self.bindings = bindings;
    }
}

impl Default for InputSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binding::{KeyCode, Modifiers, MouseButton};
    use query::FrameInput;

    fn input_with_left_drag() -> FrameInput {
        let mut input = FrameInput::default();
        input.dragging.insert(MouseButton::Left);
        input.drag_delta = glam::Vec2::new(10.0, 5.0);
        input.hovered = true;
        input
    }

    #[test]
    fn test_query_orbit_active() {
        let sys = InputSystem::new();
        let mut input = input_with_left_drag();
        input.modifiers = Modifiers::ALT;
        let state = sys.query(Action::Orbit, &input);
        assert!(
            state.is_active(),
            "orbit should be active on alt+left-drag in Normal mode"
        );
    }

    #[test]
    fn test_query_orbit_inactive_without_alt() {
        let sys = InputSystem::new();
        let input = input_with_left_drag();
        let state = sys.query(Action::Orbit, &input);
        assert!(
            !state.is_active(),
            "orbit should be inactive on plain left-drag in Normal mode"
        );
    }

    #[test]
    fn test_mode_filtering() {
        let mut sys = InputSystem::new();
        sys.set_mode(InputMode::FlyMode);
        let input = input_with_left_drag();
        // Orbit is bound to Normal mode only, should be inactive in FlyMode.
        let state = sys.query(Action::Orbit, &input);
        assert!(!state.is_active(), "orbit should be inactive in FlyMode");
    }

    #[test]
    fn test_modifier_matching() {
        let sys = InputSystem::new();
        // Pan requires Shift + left drag.
        let mut input = FrameInput::default();
        input.dragging.insert(MouseButton::Left);
        input.drag_delta = glam::Vec2::new(10.0, 5.0);
        input.modifiers = Modifiers::SHIFT;
        let state = sys.query(Action::Pan, &input);
        assert!(
            state.is_active(),
            "pan should be active with shift+left drag"
        );

        // Without shift, pan should be inactive (orbit takes it instead).
        let mut input2 = FrameInput::default();
        input2.dragging.insert(MouseButton::Left);
        input2.drag_delta = glam::Vec2::new(10.0, 5.0);
        input2.modifiers = Modifiers::CTRL;
        let state2 = sys.query(Action::Pan, &input2);
        assert!(
            !state2.is_active(),
            "pan should be inactive with ctrl modifier"
        );
    }

    #[test]
    fn test_ignore_modifiers() {
        let mut sys = InputSystem::new();
        sys.set_mode(InputMode::FlyMode);
        // FlyForward (W) uses ignore_modifiers, so it should fire even with Shift held.
        let mut input = FrameInput::default();
        input.keys_held.insert(KeyCode::W);
        input.modifiers = Modifiers::SHIFT;
        let state = sys.query(Action::FlyForward, &input);
        assert!(
            state.is_active(),
            "fly forward should be active with shift held (ignore_modifiers)"
        );
    }

    #[test]
    fn test_empty_input_inactive() {
        let sys = InputSystem::new();
        let input = FrameInput::default();
        assert!(!sys.query(Action::Orbit, &input).is_active());
        assert!(!sys.query(Action::Pan, &input).is_active());
        assert!(!sys.query(Action::Zoom, &input).is_active());
        assert!(!sys.query(Action::FocusObject, &input).is_active());
    }
}
