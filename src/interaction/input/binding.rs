//! Input binding types: triggers, modifiers, and bindings that map actions to physical inputs.

use super::action::Action;
use super::mode::InputMode;

/// Modifier key state : exact-match semantics (all must match).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Modifiers {
    /// Alt/Option key held.
    pub alt: bool,
    /// Shift key held.
    pub shift: bool,
    /// Ctrl/Cmd key held.
    pub ctrl: bool,
}

impl Modifiers {
    /// No modifier keys held.
    pub const NONE: Self = Self {
        alt: false,
        shift: false,
        ctrl: false,
    };
    /// Only Alt held.
    pub const ALT: Self = Self {
        alt: true,
        shift: false,
        ctrl: false,
    };
    /// Only Shift held.
    pub const SHIFT: Self = Self {
        alt: false,
        shift: true,
        ctrl: false,
    };
    /// Only Ctrl held.
    pub const CTRL: Self = Self {
        alt: false,
        shift: false,
        ctrl: true,
    };
    /// Ctrl + Shift held.
    pub const CTRL_SHIFT: Self = Self {
        alt: false,
        shift: true,
        ctrl: true,
    };
}

/// Keyboard key codes : subset covering keys used in the default bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KeyCode {
    /// Letter key A.
    A,
    /// Letter key B.
    B,
    /// Letter key C.
    C,
    /// Letter key D.
    D,
    /// Letter key E.
    E,
    /// Letter key F.
    F,
    /// Letter key G.
    G,
    /// Letter key H.
    H,
    /// Letter key I.
    I,
    /// Letter key J.
    J,
    /// Letter key K.
    K,
    /// Letter key L.
    L,
    /// Letter key M.
    M,
    /// Letter key N.
    N,
    /// Letter key O.
    O,
    /// Letter key P.
    P,
    /// Letter key Q.
    Q,
    /// Letter key R.
    R,
    /// Letter key S.
    S,
    /// Letter key T.
    T,
    /// Letter key U.
    U,
    /// Letter key V.
    V,
    /// Letter key W.
    W,
    /// Letter key X.
    X,
    /// Letter key Y.
    Y,
    /// Letter key Z.
    Z,
    /// Tab key.
    Tab,
    /// Enter/Return key.
    Enter,
    /// Escape key.
    Escape,
    /// Backtick/grave-accent key.
    Backtick,
    /// Backspace key.
    Backspace,
    /// Comma key.
    Comma,
    /// Period/full-stop key.
    Period,
    /// Left square bracket `[` key.
    LeftBracket,
    /// Right square bracket `]` key.
    RightBracket,
    /// Forward-slash `/` key.
    Slash,
    /// Left Shift key.
    LeftShift,
    /// Right Shift key.
    RightShift,
    /// Digit row `0`.
    Num0,
    /// Digit row `1`.
    Num1,
    /// Digit row `2`.
    Num2,
    /// Digit row `3`.
    Num3,
    /// Digit row `4`.
    Num4,
    /// Digit row `5`.
    Num5,
    /// Digit row `6`.
    Num6,
    /// Digit row `7`.
    Num7,
    /// Digit row `8`.
    Num8,
    /// Digit row `9`.
    Num9,
    /// Space bar.
    Space,
    /// Arrow up.
    Up,
    /// Arrow down.
    Down,
    /// Arrow left.
    Left,
    /// Arrow right.
    Right,
    /// F1 function key.
    F1,
    /// F2 function key.
    F2,
    /// F3 function key.
    F3,
    /// F4 function key.
    F4,
    /// F5 function key.
    F5,
    /// F6 function key.
    F6,
    /// F7 function key.
    F7,
    /// F8 function key.
    F8,
    /// F9 function key.
    F9,
    /// F10 function key.
    F10,
    /// F11 function key.
    F11,
    /// F12 function key.
    F12,
    /// F13 function key.
    F13,
    /// F14 function key.
    F14,
    /// F15 function key.
    F15,
    /// F16 function key.
    F16,
    /// F17 function key.
    F17,
    /// F18 function key.
    F18,
    /// F19 function key.
    F19,
    /// F20 function key.
    F20,
    /// F21 function key.
    F21,
    /// F22 function key.
    F22,
    /// F23 function key.
    F23,
    /// F24 function key.
    F24,
    /// Semicolon `;`.
    Semicolon,
    /// Apostrophe `'`.
    Apostrophe,
    /// Backslash `\`.
    Backslash,
    /// Minus `-`.
    Minus,
    /// Equals `=`.
    Equals,
    /// Left Ctrl key.
    LeftCtrl,
    /// Right Ctrl key.
    RightCtrl,
    /// Left Alt key.
    LeftAlt,
    /// Right Alt key.
    RightAlt,
    /// Left Super (Windows/Command) key.
    LeftSuper,
    /// Right Super (Windows/Command) key.
    RightSuper,
    /// Caps Lock key.
    CapsLock,
    /// Delete key.
    Delete,
    /// Insert key.
    Insert,
    /// Home key.
    Home,
    /// End key.
    End,
    /// Page Up key.
    PageUp,
    /// Page Down key.
    PageDown,
    /// Numpad `0`.
    Numpad0,
    /// Numpad `1`.
    Numpad1,
    /// Numpad `2`.
    Numpad2,
    /// Numpad `3`.
    Numpad3,
    /// Numpad `4`.
    Numpad4,
    /// Numpad `5`.
    Numpad5,
    /// Numpad `6`.
    Numpad6,
    /// Numpad `7`.
    Numpad7,
    /// Numpad `8`.
    Numpad8,
    /// Numpad `9`.
    Numpad9,
    /// Numpad `+`.
    NumpadAdd,
    /// Numpad `-`.
    NumpadSubtract,
    /// Numpad `*`.
    NumpadMultiply,
    /// Numpad `/`.
    NumpadDivide,
    /// Numpad `.` (decimal point).
    NumpadDecimal,
    /// Numpad Enter.
    NumpadEnter,
    /// Num Lock key.
    NumLock,
    /// Print Screen key.
    PrintScreen,
    /// Pause/Break key.
    Pause,
    /// Scroll Lock key.
    ScrollLock,
}

/// Mouse buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MouseButton {
    /// Primary (left) mouse button.
    Left,
    /// Secondary (right) mouse button.
    Right,
    /// Middle mouse button (scroll wheel click).
    Middle,
}

/// What physical input fires the trigger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TriggerKind {
    /// A keyboard key.
    Key(KeyCode),
    /// A mouse button.
    MouseButton(MouseButton),
    /// The scroll wheel (used with `OnScroll` activation).
    Scroll,
}

/// How the trigger activates the action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ActivationMode {
    /// Fires once on key/button press.
    OnPress,
    /// Active every frame while held.
    WhileHeld,
    /// Active while dragging (mouse button held + moved).
    OnDrag,
    /// Active when scroll wheel moves.
    OnScroll,
}

/// A physical trigger that can activate an action.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Trigger {
    /// The physical input kind (key, mouse button, or scroll).
    pub kind: TriggerKind,
    /// Required modifier state (exact match unless `ignore_modifiers` is set).
    pub modifiers: Modifiers,
    /// How the trigger fires relative to the input event.
    pub activation: ActivationMode,
    /// When true, modifier keys are not checked. Useful for fly-mode WASD
    /// so that holding Shift for speed boost doesn't break movement keys.
    pub ignore_modifiers: bool,
}

/// Maps an action to a physical trigger, optionally restricted to specific modes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Binding {
    /// The semantic action this binding fires.
    pub action: Action,
    /// The physical trigger that fires the action.
    pub trigger: Trigger,
    /// Which input modes this binding is active in. Empty = all modes.
    pub active_modes: Vec<InputMode>,
}

impl Binding {
    /// Convenience: create a binding active in all modes.
    pub fn global(action: Action, trigger: Trigger) -> Self {
        Self {
            action,
            trigger,
            active_modes: Vec::new(),
        }
    }

    /// Convenience: create a binding active only in the given modes.
    pub fn in_modes(action: Action, trigger: Trigger, modes: &[InputMode]) -> Self {
        Self {
            action,
            trigger,
            active_modes: modes.to_vec(),
        }
    }
}
