use super::action::Action;
use super::binding::*;
use super::mode::InputMode;

/// Helper: key press trigger with given modifiers.
fn key_press(key: KeyCode, mods: Modifiers) -> Trigger {
    Trigger {
        kind: TriggerKind::Key(key),
        modifiers: mods,
        activation: ActivationMode::OnPress,
        ignore_modifiers: false,
    }
}

/// Helper: key held trigger that ignores modifiers (for fly mode WASD).
fn key_held_any_mod(key: KeyCode) -> Trigger {
    Trigger {
        kind: TriggerKind::Key(key),
        modifiers: Modifiers::NONE,
        activation: ActivationMode::WhileHeld,
        ignore_modifiers: true,
    }
}

/// Helper: mouse drag trigger.
fn mouse_drag(btn: MouseButton, mods: Modifiers) -> Trigger {
    Trigger {
        kind: TriggerKind::MouseButton(btn),
        modifiers: mods,
        activation: ActivationMode::OnDrag,
        ignore_modifiers: false,
    }
}

/// Helper: scroll trigger.
fn scroll(mods: Modifiers) -> Trigger {
    Trigger {
        kind: TriggerKind::Scroll,
        modifiers: mods,
        activation: ActivationMode::OnScroll,
        ignore_modifiers: false,
    }
}

/// Helper: mouse click trigger.
fn mouse_click(btn: MouseButton) -> Trigger {
    Trigger {
        kind: TriggerKind::MouseButton(btn),
        modifiers: Modifiers::NONE,
        activation: ActivationMode::OnPress,
        ignore_modifiers: false,
    }
}

/// Returns the default binding table for viewport navigation and manipulation.
pub fn default_bindings() -> Vec<Binding> {
    let normal = &[InputMode::Normal];
    let fly = &[InputMode::FlyMode];
    let manip = &[InputMode::Manipulating];
    let fly_manip = &[InputMode::FlyMode, InputMode::Manipulating];

    vec![
        // -- Viewport navigation --
        Binding::in_modes(
            Action::Orbit,
            mouse_drag(MouseButton::Left, Modifiers::ALT),
            normal,
        ),
        Binding::in_modes(
            Action::Pan,
            mouse_drag(MouseButton::Left, Modifiers::SHIFT),
            normal,
        ),
        Binding::in_modes(
            Action::Pan,
            mouse_drag(MouseButton::Middle, Modifiers::NONE),
            normal,
        ),
        Binding::in_modes(Action::Zoom, scroll(Modifiers::NONE), normal),
        Binding::in_modes(
            Action::Zoom,
            mouse_drag(MouseButton::Right, Modifiers::ALT),
            normal,
        ),
        Binding::in_modes(
            Action::FocusObject,
            key_press(KeyCode::F, Modifiers::NONE),
            normal,
        ),
        Binding::in_modes(
            Action::ResetView,
            key_press(KeyCode::R, Modifiers::NONE),
            normal,
        ),
        Binding::in_modes(
            Action::ToggleWireframe,
            key_press(KeyCode::W, Modifiers::NONE),
            normal,
        ),
        Binding::in_modes(
            Action::CycleGizmoMode,
            key_press(KeyCode::Tab, Modifiers::NONE),
            normal,
        ),
        // -- Fly mode --
        Binding::in_modes(
            Action::EnterFlyMode,
            key_press(KeyCode::Backtick, Modifiers::SHIFT),
            normal,
        ),
        Binding::in_modes(Action::FlyForward, key_held_any_mod(KeyCode::W), fly),
        Binding::in_modes(Action::FlyBackward, key_held_any_mod(KeyCode::S), fly),
        Binding::in_modes(Action::FlyLeft, key_held_any_mod(KeyCode::A), fly),
        Binding::in_modes(Action::FlyRight, key_held_any_mod(KeyCode::D), fly),
        Binding::in_modes(Action::FlyUp, key_held_any_mod(KeyCode::E), fly),
        Binding::in_modes(Action::FlyDown, key_held_any_mod(KeyCode::Q), fly),
        // FlySpeedBoost is handled by checking modifiers.shift directly in fly-mode logic,
        // not via a binding, because Shift is a modifier and not a standalone key.
        Binding::in_modes(Action::FlySpeedIncrease, scroll(Modifiers::NONE), fly),
        Binding::in_modes(Action::FlySpeedDecrease, scroll(Modifiers::SHIFT), fly),
        // -- Gizmo space toggle --
        Binding::in_modes(
            Action::ToggleGizmoSpace,
            key_press(KeyCode::Backtick, Modifiers::NONE),
            normal,
        ),
        // -- Object manipulation --
        Binding::in_modes(
            Action::BeginMove,
            key_press(KeyCode::G, Modifiers::NONE),
            normal,
        ),
        Binding::in_modes(
            Action::BeginRotate,
            key_press(KeyCode::R, Modifiers::NONE),
            normal,
        ),
        Binding::in_modes(
            Action::BeginScale,
            key_press(KeyCode::S, Modifiers::NONE),
            normal,
        ),
        Binding::in_modes(
            Action::ConstrainX,
            key_press(KeyCode::X, Modifiers::NONE),
            manip,
        ),
        Binding::in_modes(
            Action::ConstrainY,
            key_press(KeyCode::Y, Modifiers::NONE),
            manip,
        ),
        Binding::in_modes(
            Action::ConstrainZ,
            key_press(KeyCode::Z, Modifiers::NONE),
            manip,
        ),
        Binding::in_modes(
            Action::ExcludeX,
            key_press(KeyCode::X, Modifiers::SHIFT),
            manip,
        ),
        Binding::in_modes(
            Action::ExcludeY,
            key_press(KeyCode::Y, Modifiers::SHIFT),
            manip,
        ),
        Binding::in_modes(
            Action::ExcludeZ,
            key_press(KeyCode::Z, Modifiers::SHIFT),
            manip,
        ),
        // -- Confirm / Cancel (fly mode + manipulation) --
        Binding::in_modes(
            Action::Confirm,
            key_press(KeyCode::Enter, Modifiers::NONE),
            fly_manip,
        ),
        Binding::in_modes(Action::Confirm, mouse_click(MouseButton::Left), fly_manip),
        Binding::in_modes(
            Action::Cancel,
            key_press(KeyCode::Escape, Modifiers::NONE),
            fly_manip,
        ),
        Binding::in_modes(Action::Cancel, mouse_click(MouseButton::Right), fly_manip),
        // -- Scene object shortcuts --
        Binding::in_modes(
            Action::OpenAddMenu,
            key_press(KeyCode::A, Modifiers::SHIFT),
            normal,
        ),
        Binding::in_modes(
            Action::DeleteSelected,
            key_press(KeyCode::X, Modifiers::NONE),
            normal,
        ),
        // -- Global (all modes) --
        Binding::global(Action::Undo, key_press(KeyCode::Z, Modifiers::CTRL)),
        Binding::global(Action::Redo, key_press(KeyCode::Z, Modifiers::CTRL_SHIFT)),
        Binding::global(Action::Redo, key_press(KeyCode::Y, Modifiers::CTRL)),
    ]
}
