//! Object manipulation controller: move, rotate, and scale with axis constraints.
//!
//! # Quick start
//!
//! ```rust,ignore
//! let mut manip = ManipulationController::new();
//!
//! // Each frame:
//! let result = manip.update(&frame, ManipulationContext { ... });
//! match result {
//!     ManipResult::Update(delta) => { /* apply delta to selected objects */ }
//!     ManipResult::Commit        => { /* finalize / push undo */ }
//!     ManipResult::Cancel        => { /* restore snapshot */ }
//!     ManipResult::None          => {}
//! }
//!
//! // Suppress orbit while manipulating:
//! if manip.is_active() {
//!     orbit_controller.resolve();
//! } else {
//!     orbit_controller.apply_to_camera(&mut camera);
//! }
//! ```

mod session;
pub mod solvers;
pub mod types;

pub use types::*;

use crate::interaction::gizmo::{Gizmo, GizmoAxis, GizmoMode, GizmoSpace};
use crate::interaction::input::{Action, ActionFrame};
use session::{ManipulationSession, update_constraint, update_numeric_state};

/// Manages a single object-manipulation session (G/R/S + axis constraints + gizmo drag).
///
/// Owns all session state; the app only supplies per-frame context and applies the
/// resulting [`TransformDelta`].
pub struct ManipulationController {
    session: Option<ManipulationSession>,
}

impl ManipulationController {
    /// Create a controller with no active session.
    pub fn new() -> Self {
        Self { session: None }
    }

    /// Drive the controller for one frame.
    ///
    /// Priority order:
    /// 1. Confirm (Enter, or left-click while not a gizmo drag) → [`ManipResult::Commit`]
    /// 2. Cancel (Escape) → [`ManipResult::Cancel`]
    /// 3. Gizmo drag release → [`ManipResult::Commit`]
    /// 4. Update constraints and numeric input
    /// 5. Compute and return [`ManipResult::Update`]
    /// 6. Gizmo drag start → begins session, returns [`ManipResult::None`] this frame
    /// 7. G/R/S keys (when `selection_center` is `Some`) → begins session
    /// 8. Otherwise → [`ManipResult::None`]
    pub fn update(&mut self, frame: &ActionFrame, ctx: ManipulationContext) -> ManipResult {
        if let Some(ref mut session) = self.session {
            // 1. Confirm: Enter key, or left-click when not a gizmo drag.
            let click_confirm = ctx.clicked && !session.is_gizmo_drag;
            if frame.is_active(Action::Confirm) || click_confirm {
                self.session = None;
                return ManipResult::Commit;
            }

            // 2. Cancel: Escape key.
            if frame.is_active(Action::Cancel) {
                self.session = None;
                return ManipResult::Cancel;
            }

            // 3. Gizmo drag released.
            if session.is_gizmo_drag && !ctx.dragging {
                self.session = None;
                return ManipResult::Commit;
            }

            // 4. Constraint and numeric updates.
            let axis_before = session.axis;
            let exclude_before = session.exclude_axis;
            update_constraint(
                session,
                frame.is_active(Action::ConstrainX),
                frame.is_active(Action::ConstrainY),
                frame.is_active(Action::ConstrainZ),
                frame.is_active(Action::ExcludeX),
                frame.is_active(Action::ExcludeY),
                frame.is_active(Action::ExcludeZ),
            );
            update_numeric_state(session, frame);

            // If the constraint changed, reset the cursor anchor so the next
            // frame's delta is computed relative to the current cursor position
            // with the new constraint — and tell the app to restore its snapshot.
            if session.axis != axis_before || session.exclude_axis != exclude_before {
                session.cursor_anchor = ctx.cursor_viewport;
                session.cursor_last_total = glam::Vec2::ZERO;
                return ManipResult::ConstraintChanged;
            }

            // 5. Compute delta.
            //
            // Prefer absolute-cursor arithmetic over raw pointer_delta so that
            // the per-frame increment is stable even if the OS coalesces events.
            // Falls back to ctx.pointer_delta when cursor_viewport is unavailable.
            let pointer_delta = if session.numeric.is_some() {
                glam::Vec2::ZERO
            } else if let (Some(current), Some(anchor)) =
                (ctx.cursor_viewport, session.cursor_anchor)
            {
                let total = current - anchor;
                let increment = total - session.cursor_last_total;
                session.cursor_last_total = total;
                increment
            } else {
                ctx.pointer_delta
            };

            let mut delta = TransformDelta::default();

            let camera_view = ctx.camera.view_matrix();
            let view_proj = ctx.camera.proj_matrix() * camera_view;

            match session.kind {
                ManipulationKind::Move => {
                    delta.translation = solvers::constrained_translation(
                        pointer_delta,
                        session.axis,
                        session.exclude_axis,
                        session.gizmo_center,
                        &ctx.camera,
                        ctx.viewport_size,
                    );
                    // Numeric position override.
                    if let Some(ref numeric) = session.numeric {
                        delta.position_override = numeric.parsed_values();
                    }
                }

                ManipulationKind::Rotate => {
                    let rot = if let Some(ax) = session.axis {
                        if session.exclude_axis {
                            // Excluded axis: rotate around the dominant of the two remaining axes.
                            let (ax1, ax2) = solvers::excluded_axes(ax);
                            let a1 = solvers::drag_onto_rotation(pointer_delta, ax1, camera_view);
                            let a2 = solvers::drag_onto_rotation(pointer_delta, ax2, camera_view);
                            let (chosen_axis, angle) =
                                if a1.abs() >= a2.abs() { (ax1, a1) } else { (ax2, a2) };
                            glam::Quat::from_axis_angle(chosen_axis, angle)
                        } else {
                            // Constrained to a single axis: angular sweep around screen center.
                            let axis_world = solvers::gizmo_axis_to_vec3(ax);
                            let angle = solvers::angular_rotation_from_cursor(
                                ctx.cursor_viewport,
                                pointer_delta,
                                session.gizmo_center,
                                axis_world,
                                view_proj,
                                ctx.viewport_size,
                                camera_view,
                            );
                            glam::Quat::from_axis_angle(axis_world, angle)
                        }
                    } else {
                        // Unconstrained: rotate around camera view direction.
                        let view_dir =
                            (ctx.camera.center - ctx.camera.eye_position()).normalize();
                        glam::Quat::from_axis_angle(view_dir, pointer_delta.x * 0.01)
                    };
                    delta.rotation = rot;
                }

                ManipulationKind::Scale => {
                    delta.scale = solvers::constrained_scale(
                        pointer_delta,
                        session.axis,
                        session.exclude_axis,
                        session.gizmo_center,
                        view_proj,
                        ctx.viewport_size,
                    );
                    // Numeric scale override.
                    if let Some(ref numeric) = session.numeric {
                        delta.scale_override = numeric.parsed_values();
                    }
                }
            }

            return ManipResult::Update(delta);
        }

        // No active session — check for session starts.

        // 6. Gizmo drag start.
        if ctx.drag_started {
            if let (Some(gizmo_info), Some(center), Some(cursor)) =
                (&ctx.gizmo, ctx.selection_center, ctx.cursor_viewport)
            {
                let camera_view = ctx.camera.view_matrix();
                let view_proj = ctx.camera.proj_matrix() * camera_view;

                // Build a ray from the cursor position.
                let ray_origin = ctx.camera.eye_position();
                let ray_dir =
                    unproject_cursor_to_ray(cursor, &ctx.camera, view_proj, ctx.viewport_size);

                let temp_gizmo = Gizmo {
                    mode: gizmo_info.mode,
                    space: GizmoSpace::World,
                    hovered_axis: GizmoAxis::None,
                    active_axis: GizmoAxis::None,
                    drag_start_mouse: None,
                    pivot_mode: crate::interaction::gizmo::PivotMode::SelectionCentroid,
                };
                let hit = temp_gizmo.hit_test_oriented(
                    ray_origin,
                    ray_dir,
                    gizmo_info.center,
                    gizmo_info.scale,
                    gizmo_info.orientation,
                );

                if hit != GizmoAxis::None {
                    let kind = match gizmo_info.mode {
                        GizmoMode::Translate => ManipulationKind::Move,
                        GizmoMode::Rotate    => ManipulationKind::Rotate,
                        GizmoMode::Scale     => ManipulationKind::Scale,
                    };
                    self.session = Some(ManipulationSession {
                        kind,
                        axis: Some(hit),
                        exclude_axis: false,
                        numeric: None,
                        is_gizmo_drag: true,
                        gizmo_center: center,
                        cursor_anchor: ctx.cursor_viewport,
                        cursor_last_total: glam::Vec2::ZERO,
                    });
                    return ManipResult::None;
                }
            }
        }

        // 7. G/R/S keyboard shortcuts.
        if let Some(center) = ctx.selection_center {
            let kind = if frame.is_active(Action::BeginMove) {
                Some(ManipulationKind::Move)
            } else if frame.is_active(Action::BeginRotate) {
                Some(ManipulationKind::Rotate)
            } else if frame.is_active(Action::BeginScale) {
                Some(ManipulationKind::Scale)
            } else {
                None
            };

            if let Some(kind) = kind {
                self.session = Some(ManipulationSession {
                    kind,
                    axis: None,
                    exclude_axis: false,
                    numeric: None,
                    is_gizmo_drag: false,
                    gizmo_center: center,
                    cursor_anchor: ctx.cursor_viewport,
                    cursor_last_total: glam::Vec2::ZERO,
                });
                return ManipResult::None;
            }
        }

        ManipResult::None
    }

    /// Returns `true` when a manipulation session is in progress.
    ///
    /// Use this to suppress camera orbit:
    /// ```rust,ignore
    /// if manip.is_active() { orbit.resolve() } else { orbit.apply_to_camera(&mut cam) }
    /// ```
    pub fn is_active(&self) -> bool {
        self.session.is_some()
    }

    /// Returns an inspectable snapshot of the current session, or `None` when idle.
    pub fn state(&self) -> Option<ManipulationState> {
        self.session.as_ref().map(|s| s.to_state())
    }

    /// Force-begin a manipulation (e.g. from a UI button).
    ///
    /// No-op if a session is already active.
    pub fn begin(&mut self, kind: ManipulationKind, center: glam::Vec3) {
        if self.session.is_some() {
            return;
        }
        self.session = Some(ManipulationSession {
            kind,
            axis: None,
            exclude_axis: false,
            numeric: None,
            is_gizmo_drag: false,
            gizmo_center: center,
            cursor_anchor: None,
            cursor_last_total: glam::Vec2::ZERO,
        });
    }

    /// Force-cancel any active session without emitting [`ManipResult::Cancel`].
    pub fn reset(&mut self) {
        self.session = None;
    }
}

impl Default for ManipulationController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute a world-space ray direction from a viewport-local cursor position.
fn unproject_cursor_to_ray(
    cursor_viewport: glam::Vec2,
    camera: &crate::camera::camera::Camera,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> glam::Vec3 {
    // Convert cursor from viewport pixels (Y-down) to NDC.
    let ndc_x = (cursor_viewport.x / viewport_size.x.max(1.0)) * 2.0 - 1.0;
    let ndc_y = 1.0 - (cursor_viewport.y / viewport_size.y.max(1.0)) * 2.0;

    let inv_vp = view_proj.inverse();

    // Near plane point and far plane point in world space.
    let near_world = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 0.0));
    let far_world  = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));

    // Use the camera eye position for accuracy (same as the gizmo hit-test origin).
    let _ = near_world;
    let eye = camera.eye_position();
    (far_world - eye).normalize_or(glam::Vec3::NEG_Z)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::input::ActionFrame;
    use session::{NumericInputState, update_constraint};

    fn make_camera() -> crate::camera::camera::Camera {
        crate::camera::camera::Camera::default()
    }

    fn idle_ctx() -> ManipulationContext {
        ManipulationContext {
            camera: make_camera(),
            viewport_size: glam::Vec2::new(800.0, 600.0),
            cursor_viewport: None,
            pointer_delta: glam::Vec2::ZERO,
            selection_center: None,
            gizmo: None,
            drag_started: false,
            dragging: false,
            clicked: false,
        }
    }

    // -----------------------------------------------------------------------
    // Constraint transition tests
    // -----------------------------------------------------------------------

    #[test]
    fn constraint_transitions_x_y_shift_z() {
        let mut session = ManipulationSession {
            kind: ManipulationKind::Move,
            axis: None,
            exclude_axis: false,
            numeric: None,
            is_gizmo_drag: false,
            gizmo_center: glam::Vec3::ZERO,
            cursor_anchor: None,
            cursor_last_total: glam::Vec2::ZERO,
        };

        // X: constrained, not excluded.
        update_constraint(&mut session, true, false, false, false, false, false);
        assert_eq!(session.axis, Some(GizmoAxis::X));
        assert!(!session.exclude_axis);

        // Y: constrained, not excluded.
        update_constraint(&mut session, false, true, false, false, false, false);
        assert_eq!(session.axis, Some(GizmoAxis::Y));
        assert!(!session.exclude_axis);

        // Shift+Z: excluded.
        update_constraint(&mut session, false, false, false, false, false, true);
        assert_eq!(session.axis, Some(GizmoAxis::Z));
        assert!(session.exclude_axis);
    }

    // -----------------------------------------------------------------------
    // Numeric parse test (deferred — Action enum lacks NumericDigit/Backspace/Tab)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "numeric input deferred: Action enum lacks NumericDigit/Backspace/Tab variants"]
    fn numeric_parse_x_axis() {
        // When numeric input actions are added, this test should verify:
        //   NumericInputState with axis=Some(X), after typing "2", ".", "5", "0"
        //   -> parsed_values() returns [Some(2.5), None, None]
        let mut state = NumericInputState::new(Some(GizmoAxis::X), false);
        // Simulated digit pushes (would be driven by Action events):
        state.axis_inputs[0] = "2.50".to_string();
        let parsed = state.parsed_values();
        assert_eq!(parsed[0], Some(2.5));
        assert_eq!(parsed[1], None);
        assert_eq!(parsed[2], None);
    }

    // -----------------------------------------------------------------------
    // angular_rotation_from_cursor sign tests
    // -----------------------------------------------------------------------

    fn make_view_proj_looking_neg_z() -> (glam::Mat4, glam::Mat4) {
        // Camera at (0, 0, 5) looking at origin.
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_4,
            800.0 / 600.0,
            0.1,
            100.0,
        );
        (view, proj * view)
    }

    #[test]
    fn angular_rotation_z_toward_camera_cw_is_positive() {
        // Axis = +Z, camera at +Z => axis points toward camera (axis_z_cam > 0).
        // CW screen motion (cursor sweeps CW) should produce positive world angle.
        let (camera_view, view_proj) = make_view_proj_looking_neg_z();
        let gizmo_center = glam::Vec3::ZERO;
        let viewport_size = glam::Vec2::new(800.0, 600.0);

        // Place cursor to the right of center, move it upward (CW sweep).
        let cursor = glam::Vec2::new(500.0, 300.0); // right of screen center
        let pointer_delta = glam::Vec2::new(0.0, -20.0); // upward = CW for right-side cursor

        let angle = solvers::angular_rotation_from_cursor(
            Some(cursor),
            pointer_delta,
            gizmo_center,
            glam::Vec3::Z,
            view_proj,
            viewport_size,
            camera_view,
        );
        assert!(
            angle > 0.0,
            "CW motion with +Z axis (toward camera) should give positive angle, got {angle}"
        );
    }

    #[test]
    fn angular_rotation_neg_z_away_from_camera_cw_is_negative() {
        // Axis = -Z points away from camera.  Same CW cursor motion should give negative angle.
        let (camera_view, view_proj) = make_view_proj_looking_neg_z();
        let gizmo_center = glam::Vec3::ZERO;
        let viewport_size = glam::Vec2::new(800.0, 600.0);

        let cursor = glam::Vec2::new(500.0, 300.0);
        let pointer_delta = glam::Vec2::new(0.0, -20.0);

        let angle = solvers::angular_rotation_from_cursor(
            Some(cursor),
            pointer_delta,
            gizmo_center,
            glam::Vec3::NEG_Z,
            view_proj,
            viewport_size,
            camera_view,
        );
        assert!(
            angle < 0.0,
            "CW motion with -Z axis (away from camera) should give negative angle, got {angle}"
        );
    }

    // -----------------------------------------------------------------------
    // Controller lifecycle tests
    // -----------------------------------------------------------------------

    #[test]
    fn controller_lifecycle_begin_reset() {
        let mut ctrl = ManipulationController::new();
        assert!(!ctrl.is_active());

        ctrl.begin(ManipulationKind::Move, glam::Vec3::ZERO);
        assert!(ctrl.is_active());

        ctrl.reset();
        assert!(!ctrl.is_active());
    }

    #[test]
    fn controller_begin_no_op_when_active() {
        let mut ctrl = ManipulationController::new();
        ctrl.begin(ManipulationKind::Move, glam::Vec3::ONE);
        ctrl.begin(ManipulationKind::Rotate, glam::Vec3::ZERO);
        // Should still be Move (second begin was no-op).
        let state = ctrl.state().unwrap();
        assert_eq!(state.kind, ManipulationKind::Move);
    }

    #[test]
    fn controller_idle_returns_none() {
        let mut ctrl = ManipulationController::new();
        let frame = ActionFrame::default();
        let result = ctrl.update(&frame, idle_ctx());
        assert_eq!(result, ManipResult::None);
        assert!(!ctrl.is_active());
    }

    #[test]
    fn controller_no_session_without_selection_center() {
        let mut ctrl = ManipulationController::new();
        // No selection_center → G/R/S should not start a session.
        let mut frame = ActionFrame::default();
        frame.actions.insert(
            crate::interaction::input::Action::BeginMove,
            crate::interaction::input::ResolvedActionState::Pressed,
        );
        let result = ctrl.update(&frame, idle_ctx());
        assert_eq!(result, ManipResult::None);
        assert!(!ctrl.is_active());
    }

    #[test]
    fn controller_g_key_starts_move_session() {
        let mut ctrl = ManipulationController::new();
        let mut frame = ActionFrame::default();
        frame.actions.insert(
            crate::interaction::input::Action::BeginMove,
            crate::interaction::input::ResolvedActionState::Pressed,
        );
        let mut ctx = idle_ctx();
        ctx.selection_center = Some(glam::Vec3::new(1.0, 2.0, 3.0));

        let result = ctrl.update(&frame, ctx);
        assert_eq!(result, ManipResult::None); // None on first frame
        assert!(ctrl.is_active());
        assert_eq!(ctrl.state().unwrap().kind, ManipulationKind::Move);
    }
}
