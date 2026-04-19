//! Public types for the manipulation controller: kinds, results, state, and context.

use crate::interaction::gizmo::{GizmoAxis, GizmoMode};

/// Which kind of transform manipulation is in progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManipulationKind {
    /// Move (translate) selected objects.
    Move,
    /// Rotate selected objects.
    Rotate,
    /// Scale selected objects.
    Scale,
}

/// Solved per-frame transform increment. Zero/identity values mean "no change on this axis."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformDelta {
    /// Add to object position.
    pub translation: glam::Vec3,
    /// Multiply with current rotation (world-space).
    pub rotation: glam::Quat,
    /// Multiply componentwise with current scale.
    pub scale: glam::Vec3,
    /// Set individual position axes (from numeric input). None = no override.
    pub position_override: [Option<f32>; 3],
    /// Set individual scale axes (from numeric input). None = no override.
    pub scale_override: [Option<f32>; 3],
}

impl Default for TransformDelta {
    fn default() -> Self {
        Self {
            translation: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            position_override: [None; 3],
            scale_override: [None; 3],
        }
    }
}

/// What the manipulation controller produced this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ManipResult {
    /// No manipulation is active or no change this frame.
    None,
    /// Active session produced a transform delta — apply to selected objects.
    Update(TransformDelta),
    /// Session completed (Enter or left-click). Apply and finalize.
    Commit,
    /// Session cancelled (Escape). Restore original transforms.
    Cancel,
}

/// Inspectable snapshot of the current manipulation session.
#[derive(Debug, Clone)]
pub struct ManipulationState {
    /// Which transform operation is active.
    pub kind: ManipulationKind,
    /// Active axis constraint, if any.
    pub axis: Option<GizmoAxis>,
    /// Whether the axis is excluded (Shift+axis) rather than constrained.
    pub exclude_axis: bool,
    /// Whether the session began from a gizmo drag (vs keyboard G/R/S).
    pub is_gizmo_drag: bool,
    /// World-space center of the manipulation (selection centroid at session start).
    pub center: glam::Vec3,
    /// Numeric input display string for HUD rendering, e.g. "X: 2.50 Y: _ Z: _".
    pub numeric_display: Option<String>,
}

/// What the app knows about the gizmo this frame.
///
/// Pass `None` to skip gizmo drag detection.
#[derive(Debug, Clone, Copy)]
pub struct GizmoInfo {
    /// World-space center of the gizmo.
    pub center: glam::Vec3,
    /// World-space arm length (screen-size scale factor).
    pub scale: f32,
    /// Gizmo orientation (for local-space gizmos).
    pub orientation: glam::Quat,
    /// Current gizmo mode.
    pub mode: GizmoMode,
}

/// Everything the manipulation controller needs to run one frame.
#[derive(Clone)]
pub struct ManipulationContext {
    /// Current camera state.
    pub camera: crate::camera::camera::Camera,
    /// Viewport size in pixels.
    pub viewport_size: glam::Vec2,
    /// Cursor position in viewport-local pixels. `None` if outside viewport.
    pub cursor_viewport: Option<glam::Vec2>,
    /// Mouse movement in pixels since last frame.
    pub pointer_delta: glam::Vec2,
    /// World-space center of the current selection. `None` disables G/R/S and gizmo drag.
    pub selection_center: Option<glam::Vec3>,
    /// Gizmo state for this frame. `None` disables gizmo drag detection.
    pub gizmo: Option<GizmoInfo>,
    /// `true` on the frame a primary drag begins.
    pub drag_started: bool,
    /// `true` while a primary drag is ongoing.
    pub dragging: bool,
    /// `true` on a primary click-without-drag.
    pub clicked: bool,
}
