//! Runtime mode enum for [`super::ViewportRuntime`].

/// Execution mode for [`super::ViewportRuntime`].
///
/// Selects default behavior for timestep, gizmo, and camera policies.
/// Individual settings can always be overridden after construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SceneRuntimeMode {
    /// No fixed timestep. Picking, selection, and manipulation gizmo are active.
    /// Default for visualization and CAD-style applications.
    #[default]
    Visualization,
    /// Variable-dt animation steps. No fixed timestep or physics.
    Animation,
    /// Fixed timestep with physics plugins active.
    /// Picking and selection remain on.
    Simulation,
    /// Fixed timestep with physics. Manipulation gizmo is off by default.
    /// Orbit camera can be overridden by a camera tracking binding.
    Game,
}
