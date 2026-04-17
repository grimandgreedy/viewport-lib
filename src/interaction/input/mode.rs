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
