//! Semantic action identifiers for the viewport input system.

/// Semantic actions that can be triggered by user input.
///
/// Actions are decoupled from their physical triggers (keys/mouse buttons),
/// so bindings can be remapped and the same action can be reached through
/// different triggers depending on context. The named variants are the ones
/// the library's controllers act on, plus [`Confirm`](Action::Confirm) and
/// [`Cancel`](Action::Cancel) for any modal tool; an application or a tool crate
/// binds its own actions to [`Action::Custom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Action {
    // -- Viewport navigation --
    /// Rotate the camera around the orbit centre (arcball).
    Orbit,
    /// Translate the orbit centre in the camera plane.
    Pan,
    /// Zoom in/out by adjusting camera distance.
    Zoom,
    /// Frame the selected object (zoom-to-fit).
    FocusObject,
    /// Reset camera to the default view.
    ResetView,

    // -- Fly mode --
    /// Enter first-person fly-through mode.
    EnterFlyMode,
    /// Move forward in fly mode.
    FlyForward,
    /// Move backward in fly mode.
    FlyBackward,
    /// Strafe left in fly mode.
    FlyLeft,
    /// Strafe right in fly mode.
    FlyRight,
    /// Move up in fly mode.
    FlyUp,
    /// Move down in fly mode.
    FlyDown,
    /// Hold to increase fly-mode movement speed.
    FlySpeedBoost,

    // -- Shared modal --
    /// Confirm the current operation (Enter / left-click).
    Confirm,
    /// Cancel the current operation (Escape / right-click).
    Cancel,

    // -- Application-defined --
    /// An action the application defines. The library binds nothing to it and
    /// never acts on it; the resolver reports it like any other action, so an
    /// application shortcut gets the same gestures, modifiers and keyboard
    /// ownership as a built-in one. Give each shortcut its own constant.
    /// Library crates take values from `0x1000_0000` upwards; an application
    /// keeps the range below.
    Custom(u32),
}
