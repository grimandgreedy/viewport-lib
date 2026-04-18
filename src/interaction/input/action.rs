//! Semantic action identifiers for the viewport input system.

/// Semantic actions that can be triggered by user input.
///
/// Actions are decoupled from their physical triggers (keys/mouse buttons),
/// enabling future key reconfiguration and context-sensitive bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Action {
    // -- Viewport navigation --
    /// Rotate the camera around the orbit center (arcball).
    Orbit,
    /// Translate the orbit center in the camera plane.
    Pan,
    /// Zoom in/out by adjusting camera distance.
    Zoom,
    /// Frame the selected object (zoom-to-fit).
    FocusObject,
    /// Reset camera to the default view.
    ResetView,
    /// Toggle between solid and wireframe render modes.
    ToggleWireframe,
    /// Cycle the gizmo mode (Translate -> Rotate -> Scale).
    CycleGizmoMode,

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

    // -- Object manipulation (keyboard G/R/S) --
    /// Begin keyboard-driven move (G key).
    BeginMove,
    /// Begin keyboard-driven rotate (R key).
    BeginRotate,
    /// Begin keyboard-driven scale (S key).
    BeginScale,
    /// Constrain transform to the X axis.
    ConstrainX,
    /// Constrain transform to the Y axis.
    ConstrainY,
    /// Constrain transform to the Z axis.
    ConstrainZ,
    /// Shift+X/Y/Z — exclude that axis, operate in the perpendicular plane.
    ExcludeX,
    /// Exclude the Y axis; operate in the XZ plane.
    ExcludeY,
    /// Exclude the Z axis; operate in the XY plane.
    ExcludeZ,

    // -- Shared modal --
    /// Confirm the current operation (Enter / left-click).
    Confirm,
    /// Cancel the current operation (Escape / right-click).
    Cancel,

    // -- Global --
    /// Undo the last action.
    Undo,
    /// Redo the previously undone action.
    Redo,

    // -- Fly mode speed --
    /// Increase fly-mode movement speed.
    FlySpeedIncrease,
    /// Decrease fly-mode movement speed.
    FlySpeedDecrease,

    // -- Gizmo --
    /// Toggle gizmo between World and Local coordinate space.
    ToggleGizmoSpace,

    // -- Scene object shortcuts --
    /// Shift+A: open the radial "Add Object" menu.
    OpenAddMenu,
    /// X: prompt to delete the currently selected object.
    DeleteSelected,
}
