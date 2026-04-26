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
