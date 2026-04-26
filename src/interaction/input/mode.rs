/// The current input mode determines which bindings are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum InputMode {
    /// Default mode: orbit/pan/zoom, object selection, shortcuts.
    Normal,
    /// WASD fly-through camera mode.
    FlyMode,
    /// Keyboard-driven object manipulation (G/R/S + axis constraint).
    Manipulating,
}

/// How the camera responds to orbit drag input.
///
/// Set on [`super::controller::OrbitCameraController`] to switch between the
/// four navigation styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum NavigationMode {
    /// Unconstrained arcball rotation (default).
    ///
    /// Drag rotates the camera freely around the orbit center in any direction.
    /// Equivalent to the behavior before navigation modes were introduced.
    #[default]
    Arcball,
    /// Yaw around the world Z axis; pitch clamped to ±89°.
    ///
    /// The up vector always stays vertical : the camera cannot roll or go
    /// upside-down. Preferred by users who expect a fixed-up-axis orbit
    /// (common in scientific / CAD applications).
    Turntable,
    /// Pan only : no rotation, no center-aimed zoom.
    ///
    /// Drag translates the orbit center in the camera plane. Scroll still
    /// adjusts the zoom level. Intended for 2D data inspection where orbit
    /// would be disorienting.
    Planar,
    /// First-person fly-through.
    ///
    /// Mouse drag acts as mouselook (yaw + pitch with the eye held fixed).
    /// WASD / Q / E translate the camera position at [`super::controller::OrbitCameraController::fly_speed`]
    /// units per frame. The `ViewportAll` binding preset must be active for
    /// the movement keys to be resolved.
    FirstPerson,
}
