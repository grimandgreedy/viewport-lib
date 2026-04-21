//! Private session state machine for the manipulation controller.
//!
//! This module is not part of the public API. All types are `pub(super)`.

use crate::interaction::gizmo::GizmoAxis;
use crate::interaction::input::{Action, ActionFrame};

use super::types::{ManipulationKind, ManipulationState};

/// Buffers numeric input from the keyboard during a manipulation session.
///
/// Activated when the user types a digit during an active session.
/// Each active axis gets its own string buffer.
///
/// Fields and methods are unused until `Action::NumericDigit/Backspace/Tab` are added.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(super) struct NumericInputState {
    /// Which axis indices are accepting input (depends on constraint).
    pub(super) active_axes: Vec<usize>,
    /// Index into `active_axes` pointing at the currently focused axis.
    pub(super) current_axis_idx: usize,
    /// Digit-string buffers for each of the three axes (X=0, Y=1, Z=2).
    pub(super) axis_inputs: [String; 3],
}

#[allow(dead_code)]
impl NumericInputState {
    /// Create a new numeric input state for the given axis constraint.
    pub(super) fn new(axis: Option<GizmoAxis>, excluded: bool) -> Self {
        let active_axes = match axis {
            None => vec![0, 1, 2],
            Some(GizmoAxis::X) => {
                if excluded { vec![1, 2] } else { vec![0] }
            }
            Some(GizmoAxis::Y) => {
                if excluded { vec![0, 2] } else { vec![1] }
            }
            Some(GizmoAxis::Z) | Some(GizmoAxis::None) => {
                if excluded { vec![0, 1] } else { vec![2] }
            }
            _ => vec![0, 1, 2],
        };
        Self {
            active_axes,
            current_axis_idx: 0,
            axis_inputs: [String::new(), String::new(), String::new()],
        }
    }

    /// Index of the axis currently receiving typed input.
    pub(super) fn current_axis(&self) -> usize {
        self.active_axes[self.current_axis_idx]
    }

    /// Parse each axis buffer into an `f32`, returning `None` for empty or invalid input.
    pub(super) fn parsed_values(&self) -> [Option<f32>; 3] {
        core::array::from_fn(|i| self.axis_inputs[i].parse::<f32>().ok())
    }

    /// Format a HUD-friendly display string, e.g. "X: 2.50 Y: _ Z: _".
    pub(super) fn display_string(&self) -> String {
        let labels = ["X", "Y", "Z"];
        let mut parts = Vec::new();
        for (i, label) in labels.iter().enumerate() {
            if self.active_axes.contains(&i) {
                let val = if self.axis_inputs[i].is_empty() {
                    "_".to_string()
                } else {
                    self.axis_inputs[i].clone()
                };
                parts.push(format!("{label}: {val}"));
            }
        }
        parts.join("  ")
    }
}

/// Private session state for an in-progress manipulation.
#[derive(Debug, Clone)]
pub(super) struct ManipulationSession {
    /// Which transform operation is active.
    pub(super) kind: ManipulationKind,
    /// Active axis constraint, if any.
    pub(super) axis: Option<GizmoAxis>,
    /// Whether the constrained axis is excluded (Shift+axis).
    pub(super) exclude_axis: bool,
    /// Numeric input state, active once the user begins typing digits.
    pub(super) numeric: Option<NumericInputState>,
    /// Whether this session began from a gizmo drag.
    pub(super) is_gizmo_drag: bool,
    /// World-space center captured when the session began.
    pub(super) gizmo_center: glam::Vec3,
    /// Viewport-local cursor position when the session started or the constraint last changed.
    ///
    /// Used to compute true per-frame increments from the absolute cursor position,
    /// so that changing constraints resets the delta accumulation cleanly.
    pub(super) cursor_anchor: Option<glam::Vec2>,
    /// Total cursor displacement (current − anchor) that was used on the previous frame.
    ///
    /// The per-frame delta passed to solvers is `(current − anchor) − cursor_last_total`.
    /// Reset to zero whenever `cursor_anchor` is updated.
    pub(super) cursor_last_total: glam::Vec2,
    /// The cumulative distance-ratio scale factor applied on the previous frame.
    ///
    /// Used in scale mode to convert the cumulative factor (current_dist / anchor_dist)
    /// into a per-frame incremental factor.  Reset to 1.0 whenever `cursor_anchor` is updated.
    pub(super) last_scale_factor: f32,
}

impl ManipulationSession {
    /// Convert to the public inspectable snapshot.
    pub(super) fn to_state(&self) -> ManipulationState {
        let numeric_display = self
            .numeric
            .as_ref()
            .map(|n| n.display_string())
            .filter(|s| !s.is_empty());
        ManipulationState {
            kind: self.kind,
            axis: self.axis,
            exclude_axis: self.exclude_axis,
            is_gizmo_drag: self.is_gizmo_drag,
            center: self.gizmo_center,
            numeric_display,
        }
    }
}

/// Update axis constraint on the session.
///
/// Resets numeric input whenever the constraint changes so that previously
/// typed values do not carry over to the new axis.
pub(super) fn update_constraint(
    session: &mut ManipulationSession,
    constrain_x: bool,
    constrain_y: bool,
    constrain_z: bool,
    exclude_x: bool,
    exclude_y: bool,
    exclude_z: bool,
) {
    let mut set_axis = |axis: GizmoAxis, exclude: bool| {
        session.axis = Some(axis);
        session.exclude_axis = exclude;
        session.numeric = None;
    };

    if constrain_x { set_axis(GizmoAxis::X, false); }
    if constrain_y { set_axis(GizmoAxis::Y, false); }
    if constrain_z { set_axis(GizmoAxis::Z, false); }
    if exclude_x   { set_axis(GizmoAxis::X, true);  }
    if exclude_y   { set_axis(GizmoAxis::Y, true);  }
    if exclude_z   { set_axis(GizmoAxis::Z, true);  }
}

/// Update numeric buffering for the session.
///
/// Reads `frame.typed_chars` (digits, `.`, `-`) and `Action::NumericBackspace` /
/// `Action::NumericNextAxis` to maintain per-axis string buffers. The first digit
/// typed automatically initialises the numeric state.
pub(super) fn update_numeric_state(session: &mut ManipulationSession, frame: &ActionFrame) {
    // Bootstrap: create numeric state on the first typed digit.
    if session.numeric.is_none() && !frame.typed_chars.is_empty() {
        session.numeric = Some(NumericInputState::new(session.axis, session.exclude_axis));
    }

    let Some(ref mut numeric) = session.numeric else { return };
    let axis_idx = numeric.current_axis();

    // Append typed characters to the current axis buffer.
    for &c in &frame.typed_chars {
        let buf = &mut numeric.axis_inputs[axis_idx];
        // Allow '-' only as the first character.
        if c == '-' && buf.is_empty() {
            buf.push(c);
        } else if c.is_ascii_digit() || c == '.' {
            // Allow at most one decimal point.
            if c != '.' || !buf.contains('.') {
                buf.push(c);
            }
        }
    }

    // Backspace: pop the last character.
    if frame.is_active(Action::NumericBackspace) {
        numeric.axis_inputs[axis_idx].pop();
    }

    // Tab: advance to the next active axis.
    if frame.is_active(Action::NumericNextAxis) {
        let len = numeric.active_axes.len();
        if len > 1 {
            numeric.current_axis_idx = (numeric.current_axis_idx + 1) % len;
        }
    }
}
