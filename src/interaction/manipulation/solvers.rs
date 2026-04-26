//! Pure math solver functions for manipulation transforms.
//!
//! All functions are stateless. They translate raw input (pointer delta, camera,
//! axis constraint) into world-space transform components that the app can apply directly.

use crate::camera::camera::Camera;
use crate::interaction::gizmo::{GizmoAxis, project_drag_onto_axis};

/// Compute the signed rotation angle (radians) produced by the cursor sweeping around the
/// screen-projected gizmo center.
///
/// Returns 0.0 when the cursor is within `MIN_RADIUS` pixels of the center, or when
/// `cursor_viewport` is `None`.
///
/// # Arguments
/// * `cursor_viewport` : current cursor position in viewport-local pixels (Y-down).
/// * `pointer_delta` : mouse movement since last frame in pixels.
/// * `gizmo_center` : world-space rotation pivot.
/// * `axis_world` : world-space rotation axis (unit vector).
/// * `view_proj` : camera view-projection matrix.
/// * `viewport_size` : viewport dimensions in pixels.
/// * `camera_view` : camera view matrix (used to orient the angle sign).
pub fn angular_rotation_from_cursor(
    cursor_viewport: Option<glam::Vec2>,
    pointer_delta: glam::Vec2,
    gizmo_center: glam::Vec3,
    axis_world: glam::Vec3,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
    camera_view: glam::Mat4,
) -> f32 {
    const MIN_RADIUS: f32 = 10.0;

    let cursor = match cursor_viewport {
        Some(c) => c,
        None => return 0.0,
    };

    // Project gizmo center into viewport pixel space (Y-down).
    let ndc = view_proj.project_point3(gizmo_center);
    let center_screen = glam::Vec2::new(
        (ndc.x + 1.0) * 0.5 * viewport_size.x,
        (1.0 - ndc.y) * 0.5 * viewport_size.y,
    );

    let r_curr = cursor - center_screen;
    let r_prev = r_curr - pointer_delta;

    if r_curr.length() < MIN_RADIUS || r_prev.length() < MIN_RADIUS {
        return 0.0;
    }

    // Signed angle from r_prev to r_curr in screen space (Y-down).
    // Positive cross2d = CW visual rotation (because screen Y is down).
    let cross2d = r_prev.x * r_curr.y - r_prev.y * r_curr.x;
    let dot = r_prev.dot(r_curr);
    let screen_angle = cross2d.atan2(dot);

    // Correct for axis orientation relative to camera.
    // In a look_at_rh view matrix, camera +Z points backward (away from scene),
    // so axis_z_cam > 0 means the axis points toward the camera.
    // For a toward-camera axis, RH positive rotation = CCW from camera = CCW visual.
    // In Y-down screen space, CCW visual -> negative cross2d -> negative screen_angle,
    // so we negate to recover the positive world angle.
    let axis_z_cam = (camera_view * axis_world.extend(0.0)).z;
    if axis_z_cam >= 0.0 {
        -screen_angle
    } else {
        screen_angle
    }
}

/// Compute the world-space translation vector from a pointer delta given an axis constraint.
///
/// # Arguments
/// * `pointer_delta` : mouse movement in pixels since last frame.
/// * `axis` : optional axis constraint. `None` = free camera-plane movement.
/// * `exclude_axis` : if `true`, `axis` names the axis to *exclude*; movement is in the perpendicular plane.
/// * `gizmo_center` : world-space pivot (used for axis projection).
/// * `camera` : current camera (provides right, up, distance, fov_y).
/// * `viewport_size` : viewport dimensions in pixels.
pub fn constrained_translation(
    pointer_delta: glam::Vec2,
    axis: Option<GizmoAxis>,
    exclude_axis: bool,
    gizmo_center: glam::Vec3,
    camera: &Camera,
    viewport_size: glam::Vec2,
) -> glam::Vec3 {
    let pan_scale = 2.0 * camera.distance * (camera.fov_y / 2.0).tan() / viewport_size.y.max(1.0);
    let camera_right = camera.right();
    let camera_up = camera.up();

    let camera_view = camera.view_matrix();
    let view_proj = camera.proj_matrix() * camera_view;

    match axis {
        None => {
            // Free: move in the camera plane.
            camera_right * pointer_delta.x * pan_scale - camera_up * pointer_delta.y * pan_scale
        }
        Some(ax) => {
            if exclude_axis {
                // Move in camera plane, then zero the excluded axis component.
                let mut world_delta = camera_right * pointer_delta.x * pan_scale
                    - camera_up * pointer_delta.y * pan_scale;
                match ax {
                    GizmoAxis::X => world_delta.x = 0.0,
                    GizmoAxis::Y => world_delta.y = 0.0,
                    GizmoAxis::Z | GizmoAxis::None => world_delta.z = 0.0,
                    _ => world_delta.z = 0.0,
                }
                world_delta
            } else {
                // Constrained to a single axis.
                let axis_world = gizmo_axis_to_vec3(ax);
                let amount = project_drag_onto_axis(
                    pointer_delta,
                    axis_world,
                    view_proj,
                    gizmo_center,
                    viewport_size,
                );
                axis_world * amount
            }
        }
    }
}

/// Compute per-axis scale multipliers from a pointer delta given an axis constraint.
///
/// All returned scale factors are clamped to `>= 0.001`.
///
/// # Arguments
/// * `pointer_delta` : mouse movement in pixels since last frame.
/// * `axis` : optional axis constraint. `None` = uniform scale.
/// * `exclude_axis` : if `true`, `axis` names the axis to hold at 1.0; the other two axes scale uniformly.
/// * `position` : world-space object position (used for axis projection onto screen).
/// * `view_proj` : camera view-projection matrix.
/// * `viewport_size` : viewport dimensions in pixels.
pub fn constrained_scale(
    pointer_delta: glam::Vec2,
    axis: Option<GizmoAxis>,
    exclude_axis: bool,
    position: glam::Vec3,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> glam::Vec3 {
    const MIN_SCALE: f32 = 0.001;
    let sensitivity = 8.0 / viewport_size.x.max(1.0);

    match axis {
        None => {
            // Uniform scale from horizontal drag.
            let factor = (1.0 + pointer_delta.x * sensitivity).max(MIN_SCALE);
            glam::Vec3::splat(factor)
        }
        Some(ax) => {
            if exclude_axis {
                // Uniform scale on the two non-excluded axes; excluded axis stays at 1.0.
                let factor = (1.0 + pointer_delta.x * sensitivity).max(MIN_SCALE);
                let mut scale = glam::Vec3::ONE;
                match ax {
                    GizmoAxis::X => {
                        scale.y = factor;
                        scale.z = factor;
                    }
                    GizmoAxis::Y => {
                        scale.x = factor;
                        scale.z = factor;
                    }
                    GizmoAxis::Z | GizmoAxis::None => {
                        scale.x = factor;
                        scale.y = factor;
                    }
                    _ => {
                        scale.x = factor;
                        scale.y = factor;
                    }
                }
                scale
            } else {
                // Project drag onto the screen-space axis direction.
                let axis_world = gizmo_axis_to_vec3(ax);
                let base_ndc = view_proj.project_point3(position);
                let tip_ndc = view_proj.project_point3(position + axis_world);
                let base_screen = glam::Vec2::new(
                    (base_ndc.x + 1.0) * 0.5 * viewport_size.x,
                    (1.0 - base_ndc.y) * 0.5 * viewport_size.y,
                );
                let tip_screen = glam::Vec2::new(
                    (tip_ndc.x + 1.0) * 0.5 * viewport_size.x,
                    (1.0 - tip_ndc.y) * 0.5 * viewport_size.y,
                );
                let axis_screen = tip_screen - base_screen;
                let axis_screen_len = axis_screen.length();
                let amount = if axis_screen_len > 1e-4 {
                    pointer_delta.dot(axis_screen / axis_screen_len) / viewport_size.x.max(1.0)
                        * 8.0
                } else {
                    0.0
                };
                let factor = (1.0 + amount).max(MIN_SCALE);
                let mut scale = glam::Vec3::ONE;
                match ax {
                    GizmoAxis::X => scale.x = factor,
                    GizmoAxis::Y => scale.y = factor,
                    GizmoAxis::Z | GizmoAxis::None => scale.z = factor,
                    _ => scale.z = factor,
                }
                scale
            }
        }
    }
}

/// Map a `GizmoAxis` single-axis variant to its world-space unit vector.
///
/// Plane and screen axes fall back to Z.
pub(super) fn gizmo_axis_to_vec3(axis: GizmoAxis) -> glam::Vec3 {
    match axis {
        GizmoAxis::X => glam::Vec3::X,
        GizmoAxis::Y => glam::Vec3::Y,
        GizmoAxis::Z | GizmoAxis::None => glam::Vec3::Z,
        _ => glam::Vec3::Z,
    }
}

/// Map a `GizmoAxis` exclude-axis variant to the two perpendicular axes.
pub(super) fn excluded_axes(axis: GizmoAxis) -> (glam::Vec3, glam::Vec3) {
    match axis {
        GizmoAxis::X => (glam::Vec3::Y, glam::Vec3::Z),
        GizmoAxis::Y => (glam::Vec3::X, glam::Vec3::Z),
        GizmoAxis::Z | GizmoAxis::None => (glam::Vec3::X, glam::Vec3::Y),
        _ => (glam::Vec3::X, glam::Vec3::Y),
    }
}

// Re-export project_drag_onto_rotation for use within the crate.
pub(super) use crate::interaction::gizmo::project_drag_onto_rotation as drag_onto_rotation;
