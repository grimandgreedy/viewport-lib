//! Interactive clip plane controller: position and orient section planes in the viewport.
//!
//! The library owns the session state machine, hit testing, solvers, and visual feedback
//! data. The application owns which clip planes exist, how they are stored, and when to
//! activate the controller.
//!
//! # Quick start
//!
//! ```rust,ignore
//! let mut ctrl = ClipPlaneController::new();
//!
//! // Each frame:
//! let result = ctrl.update(&frame, ClipPlaneContext {
//!     plane: my_plane,
//!     camera: camera.clone(),
//!     viewport_size,
//!     cursor_viewport,
//!     pointer_delta,
//!     drag_started,
//!     dragging,
//!     clicked,
//!     plane_extent: 2.0,
//! });
//! match result {
//!     ClipPlaneResult::Update(delta) => { my_plane.distance += delta.distance_delta; }
//!     ClipPlaneResult::Commit        => { /* finalize / push undo */ }
//!     ClipPlaneResult::Cancel        => { /* restore snapshot */ }
//!     ClipPlaneResult::None          => {}
//! }
//!
//! // Suppress orbit while a drag is active:
//! if ctrl.is_active() { orbit.resolve(); } else { orbit.apply_to_camera(&mut camera); }
//!
//! // Draw the plane indicator: build the outline from `visual` and submit it as an
//! // ordinary scene polyline, tagged `ignore_clip` so it stays visible through the cut:
//! frame.scene.polylines.push(visual::outline(&clip.shape, extent, colour));
//! ```

use crate::camera::camera::Camera;
use crate::interaction::input::{Action, ActionFrame};
use crate::interaction::query::snap::snap_value;
use crate::renderer::{ClipObject, ClipShape};

/// World-space visual builders for the clip shapes (outlines + plane fill).
pub mod visual;

// ---------------------------------------------------------------------------
// Solver functions
// ---------------------------------------------------------------------------

/// Project a screen-space pointer drag onto a world-space plane normal,
/// returning the signed distance delta along the normal.
///
/// This is the clip-plane equivalent of `project_drag_onto_axis` in `gizmo.rs`,
/// generalised to an arbitrary axis (the plane normal). The pointer delta is
/// projected onto the screen-space direction of the normal and converted back
/// to world units.
pub fn project_drag_onto_normal(
    pointer_delta: glam::Vec2,
    plane_normal: glam::Vec3,
    plane_point: glam::Vec3,
    camera: &Camera,
    viewport_size: glam::Vec2,
) -> f32 {
    let view_proj = camera.proj_matrix() * camera.view_matrix();

    let base_ndc = view_proj.project_point3(plane_point);
    let tip_ndc = view_proj.project_point3(plane_point + plane_normal);

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

    if axis_screen_len < 1e-4 {
        return 0.0;
    }

    let axis_screen_norm = axis_screen / axis_screen_len;
    let drag_along_axis = pointer_delta.dot(axis_screen_norm);
    // 1 world unit projects to `axis_screen_len` pixels.
    drag_along_axis / axis_screen_len
}

/// Snap a distance value to the nearest grid increment.
///
/// Wraps [`snap_value`] : named separately for discoverability.
pub fn snap_plane_distance(distance: f32, increment: f32) -> f32 {
    snap_value(distance, increment)
}

/// Axis preset for constructing axis-aligned clip planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipAxis {
    /// +X axis normal.
    X,
    /// +Y axis normal.
    Y,
    /// +Z axis normal.
    Z,
}

/// Construct a [`ClipObject`] from an axis preset and a distance from the origin.
///
/// Convenience for the common case of axis-aligned section planes.
pub fn plane_from_axis_preset(axis: ClipAxis, distance: f32) -> ClipObject {
    let normal = match axis {
        ClipAxis::X => [1.0, 0.0, 0.0],
        ClipAxis::Y => [0.0, 1.0, 0.0],
        ClipAxis::Z => [0.0, 0.0, 1.0],
    };
    ClipObject {
        shape: ClipShape::Plane {
            normal,
            distance,
            cap_colour: None,
        },
        ..ClipObject::default()
    }
}

pub use crate::geometry::intersect::ray_plane_intersection;

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

/// Result of a hit test against a clip plane handle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipPlaneHit {
    /// No hit.
    None,
    /// Hit the plane quad (suitable for distance drag).
    Quad {
        /// Ray parameter `t` at the hit point.
        t: f32,
    },
    /// Hit the normal-axis indicator handle (suitable for orient drag).
    NormalHandle {
        /// Ray parameter `t` at the hit point.
        t: f32,
    },
}

/// Test whether a ray hits the finite clip plane quad.
///
/// Returns [`ClipPlaneHit::Quad`] if the ray intersects the quad, otherwise [`ClipPlaneHit::None`].
pub fn hit_test_plane_quad(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    plane_center: glam::Vec3,
    plane_normal: glam::Vec3,
    extent: f32,
) -> ClipPlaneHit {
    let denom = plane_normal.dot(ray_dir);
    if denom.abs() < 1e-6 {
        return ClipPlaneHit::None;
    }
    let t = plane_normal.dot(plane_center - ray_origin) / denom;
    if t < 0.0 {
        return ClipPlaneHit::None;
    }
    let hit_point = ray_origin + ray_dir * t;
    let local = hit_point - plane_center;
    let (t1, t2) = plane_tangents(plane_normal);
    let u = local.dot(t1);
    let v = local.dot(t2);
    if u.abs() <= extent && v.abs() <= extent {
        ClipPlaneHit::Quad { t }
    } else {
        ClipPlaneHit::None
    }
}

/// Test whether a ray hits the normal-axis indicator handle (a thin cylinder).
///
/// The cylinder runs from `plane_center` to `plane_center + plane_normal * handle_length`
/// with the given radius.
pub fn hit_test_normal_handle(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    plane_center: glam::Vec3,
    plane_normal: glam::Vec3,
    handle_length: f32,
    handle_radius: f32,
) -> ClipPlaneHit {
    // Decompose the ray into components parallel and perpendicular to the cylinder axis.
    let d = ray_dir - plane_normal * ray_dir.dot(plane_normal);
    let offset = ray_origin - plane_center;
    let e = offset - plane_normal * offset.dot(plane_normal);

    let a = d.dot(d);
    let b = 2.0 * d.dot(e);
    let c = e.dot(e) - handle_radius * handle_radius;

    if a < 1e-10 {
        return ClipPlaneHit::None; // ray parallel to cylinder axis
    }

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return ClipPlaneHit::None;
    }

    let sqrt_disc = discriminant.sqrt();
    let t0 = (-b - sqrt_disc) / (2.0 * a);
    let t1 = (-b + sqrt_disc) / (2.0 * a);

    let t = if t0 > 0.0 {
        t0
    } else if t1 > 0.0 {
        t1
    } else {
        return ClipPlaneHit::None;
    };

    // Verify the hit point lies within the finite cylinder length.
    let hit_point = ray_origin + ray_dir * t;
    let h = (hit_point - plane_center).dot(plane_normal);
    if h >= 0.0 && h <= handle_length {
        ClipPlaneHit::NormalHandle { t }
    } else {
        ClipPlaneHit::None
    }
}

// ---------------------------------------------------------------------------
// ClipPlaneController
// ---------------------------------------------------------------------------

/// Which degree of freedom is being manipulated in a clip plane session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipPlaneSessionKind {
    /// Dragging the plane along its normal (distance change).
    Distance,
    /// Reorienting the plane normal (rotation).
    Orient,
}

/// What the clip plane controller produced this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipPlaneResult {
    /// No interaction this frame.
    None,
    /// Active session produced a delta : apply to the clip plane.
    Update(ClipPlaneDelta),
    /// Session completed (mouse release). Apply and finalise.
    Commit,
    /// Session cancelled (Escape). Restore the original plane state.
    Cancel,
}

/// Per-frame incremental change to apply to a clip plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipPlaneDelta {
    /// Change in signed distance along the normal. Add to the plane's `distance` field.
    pub distance_delta: f32,
    /// New normal orientation. Only set during Orient sessions; `None` during Distance.
    pub normal_override: Option<[f32; 3]>,
}

/// Everything the clip plane controller needs to run one frame.
#[derive(Clone)]
pub struct ClipPlaneContext {
    /// The clip object being manipulated (must be a `ClipShape::Plane` variant).
    pub plane: ClipObject,
    /// Current camera state.
    pub camera: Camera,
    /// Viewport size in pixels.
    pub viewport_size: glam::Vec2,
    /// Cursor position in viewport-local pixels. `None` if outside the viewport.
    pub cursor_viewport: Option<glam::Vec2>,
    /// Mouse movement in pixels since last frame.
    pub pointer_delta: glam::Vec2,
    /// `true` on the frame a primary drag begins.
    pub drag_started: bool,
    /// `true` while a primary drag is ongoing.
    pub dragging: bool,
    /// `true` on a primary click-without-drag (unused by the controller, available for apps).
    pub clicked: bool,
    /// Visual half-extent of the plane quad in world units.
    ///
    /// Controls the rendered size of the handle and the hit-test area.
    pub plane_extent: f32,
}

/// Internal per-session state.
struct ClipPlaneSession {
    kind: ClipPlaneSessionKind,
    /// Normal at session start, used to restore on Cancel.
    original_normal: [f32; 3],
    /// Distance at session start, used to restore on Cancel.
    original_distance: f32,
    /// Cursor position when the drag began.
    cursor_anchor: Option<glam::Vec2>,
    /// Cumulative cursor displacement used on the previous frame (for stable deltas).
    cursor_last_total: glam::Vec2,
}

/// Interactive clip plane controller.
///
/// Manages one clip-plane drag session at a time. The application supplies
/// per-frame context via [`ClipPlaneContext`] and applies the resulting
/// [`ClipPlaneDelta`] to its own plane state.
pub struct ClipPlaneController {
    session: Option<ClipPlaneSession>,
    /// Whether the handle was hovered on the last idle frame (used for overlay colour).
    hovered: bool,
}

impl ClipPlaneController {
    /// Create a controller with no active session.
    pub fn new() -> Self {
        Self {
            session: None,
            hovered: false,
        }
    }

    /// Drive the controller for one frame.
    ///
    /// Priority order:
    /// 1. Cancel (Escape) -> [`ClipPlaneResult::Cancel`]
    /// 2. Mouse release (dragging -> false) -> [`ClipPlaneResult::Commit`]
    /// 3. Compute delta from cursor movement -> [`ClipPlaneResult::Update`]
    /// 4. If idle and drag started: hit-test the handle -> begin session
    /// 5. Otherwise -> [`ClipPlaneResult::None`]
    pub fn update(&mut self, frame: &ActionFrame, ctx: ClipPlaneContext) -> ClipPlaneResult {
        if let Some(ref mut session) = self.session {
            // 1. Cancel.
            if frame.is_active(Action::Cancel) {
                self.session = None;
                self.hovered = false;
                return ClipPlaneResult::Cancel;
            }

            // 2. Mouse release.
            if !ctx.dragging {
                self.session = None;
                return ClipPlaneResult::Commit;
            }

            // 3. Compute delta.
            //
            // Prefer absolute-cursor arithmetic over raw pointer_delta so the
            // per-frame increment is stable even if the OS coalesces events.
            let pointer_delta = if let (Some(current), Some(anchor)) =
                (ctx.cursor_viewport, session.cursor_anchor)
            {
                let total = current - anchor;
                let increment = total - session.cursor_last_total;
                session.cursor_last_total = total;
                increment
            } else {
                ctx.pointer_delta
            };

            let delta = match session.kind {
                ClipPlaneSessionKind::Distance => {
                    let normal = glam::Vec3::from(session.original_normal);
                    let plane_point = normal * session.original_distance;
                    let distance_delta = project_drag_onto_normal(
                        pointer_delta,
                        normal,
                        plane_point,
                        &ctx.camera,
                        ctx.viewport_size,
                    );
                    ClipPlaneDelta {
                        distance_delta,
                        normal_override: None,
                    }
                }
                ClipPlaneSessionKind::Orient => {
                    // Orient session deferred per plan : emit zero delta.
                    ClipPlaneDelta {
                        distance_delta: 0.0,
                        normal_override: None,
                    }
                }
            };

            return ClipPlaneResult::Update(delta);
        }

        // No active session : update hover state and check for a new drag.
        self.hovered = false;

        if let Some(cursor) = ctx.cursor_viewport {
            let view_proj = ctx.camera.proj_matrix() * ctx.camera.view_matrix();
            let ray_origin = ctx.camera.eye_position();
            let ray_dir =
                unproject_cursor_to_ray(cursor, &ctx.camera, view_proj, ctx.viewport_size);

            let (plane_normal_arr, plane_dist) = if let ClipShape::Plane {
                normal, distance, ..
            } = ctx.plane.shape
            {
                (normal, distance)
            } else {
                return ClipPlaneResult::None;
            };
            let plane_normal = glam::Vec3::from(plane_normal_arr);
            let plane_center = plane_normal * plane_dist;
            let handle_length = ctx.plane_extent * 0.4;
            let handle_radius = ctx.plane_extent * 0.04;

            // Normal handle takes priority over the quad.
            let hit = {
                let nh = hit_test_normal_handle(
                    ray_origin,
                    ray_dir,
                    plane_center,
                    plane_normal,
                    handle_length,
                    handle_radius,
                );
                if nh != ClipPlaneHit::None {
                    nh
                } else {
                    hit_test_plane_quad(
                        ray_origin,
                        ray_dir,
                        plane_center,
                        plane_normal,
                        ctx.plane_extent,
                    )
                }
            };

            let is_hit = hit != ClipPlaneHit::None;
            self.hovered = is_hit;

            // 4. Drag start on a hit -> begin session.
            if ctx.drag_started && is_hit {
                let kind = match hit {
                    ClipPlaneHit::NormalHandle { .. } => ClipPlaneSessionKind::Orient,
                    _ => ClipPlaneSessionKind::Distance,
                };
                self.session = Some(ClipPlaneSession {
                    kind,
                    original_normal: plane_normal_arr,
                    original_distance: plane_dist,
                    cursor_anchor: ctx.cursor_viewport,
                    cursor_last_total: glam::Vec2::ZERO,
                });
                return ClipPlaneResult::None;
            }
        }

        ClipPlaneResult::None
    }

    /// Returns `true` when a drag session is in progress.
    pub fn is_active(&self) -> bool {
        self.session.is_some()
    }

    /// Returns `true` when the handle is being hovered.
    pub fn is_hovered(&self) -> bool {
        self.hovered
    }

    /// Force-begin a distance drag session (e.g. from a UI button or keyboard shortcut).
    ///
    /// No-op if a session is already active or the object is not a `Plane` variant.
    pub fn begin_distance(&mut self, obj: &ClipObject) {
        if self.session.is_some() {
            return;
        }
        if let ClipShape::Plane {
            normal, distance, ..
        } = obj.shape
        {
            self.session = Some(ClipPlaneSession {
                kind: ClipPlaneSessionKind::Distance,
                original_normal: normal,
                original_distance: distance,
                cursor_anchor: None,
                cursor_last_total: glam::Vec2::ZERO,
            });
        }
    }

    /// Force-cancel any active session without emitting [`ClipPlaneResult::Cancel`].
    pub fn reset(&mut self) {
        self.session = None;
    }
}

impl Default for ClipPlaneController {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute two orthogonal tangent vectors for a plane with the given normal.
pub(crate) fn plane_tangents(normal: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let up = if normal.z.abs() < 0.9 {
        glam::Vec3::Z
    } else {
        glam::Vec3::X
    };
    let t1 = (up - normal * up.dot(normal)).normalize_or(glam::Vec3::X);
    let t2 = normal.cross(t1);
    (t1, t2)
}

/// Unproject a viewport-pixel cursor position to a world-space ray direction.
fn unproject_cursor_to_ray(
    cursor_viewport: glam::Vec2,
    camera: &Camera,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> glam::Vec3 {
    let ndc_x = (cursor_viewport.x / viewport_size.x.max(1.0)) * 2.0 - 1.0;
    let ndc_y = 1.0 - (cursor_viewport.y / viewport_size.y.max(1.0)) * 2.0;
    let inv_vp = view_proj.inverse();
    let far_world = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
    let eye = camera.eye_position();
    (far_world - eye).normalize_or(glam::Vec3::NEG_Z)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::input::{ActionFrame, ResolvedActionState};

    fn default_camera() -> Camera {
        Camera::default()
    }

    fn default_plane() -> ClipObject {
        ClipObject {
            shape: ClipShape::Plane {
                normal: [0.0, 0.0, 1.0],
                distance: 0.0,
                cap_colour: None,
            },
            enabled: true,
            ..ClipObject::default()
        }
    }

    fn idle_ctx() -> ClipPlaneContext {
        ClipPlaneContext {
            plane: default_plane(),
            camera: default_camera(),
            viewport_size: glam::Vec2::new(800.0, 600.0),
            cursor_viewport: None,
            pointer_delta: glam::Vec2::ZERO,
            drag_started: false,
            dragging: false,
            clicked: false,
            plane_extent: 2.0,
        }
    }

    // --- Solver tests ---

    #[test]
    fn snap_plane_distance_rounds_to_increment() {
        assert!((snap_plane_distance(0.7, 0.5) - 0.5).abs() < 1e-6);
        assert!((snap_plane_distance(0.8, 0.5) - 1.0).abs() < 1e-6);
        assert!((snap_plane_distance(-0.3, 0.5) - -0.5).abs() < 1e-6);
    }

    #[test]
    fn snap_plane_distance_zero_increment_passthrough() {
        assert!((snap_plane_distance(1.23, 0.0) - 1.23).abs() < 1e-6);
    }

    #[test]
    fn plane_from_axis_preset_x() {
        let p = plane_from_axis_preset(ClipAxis::X, 3.0);
        if let ClipShape::Plane {
            normal, distance, ..
        } = p.shape
        {
            assert_eq!(normal, [1.0, 0.0, 0.0]);
            assert!((distance - 3.0).abs() < 1e-6);
        } else {
            panic!("expected Plane shape");
        }
        assert!(p.enabled);
    }

    #[test]
    fn plane_from_axis_preset_y() {
        let p = plane_from_axis_preset(ClipAxis::Y, -2.0);
        if let ClipShape::Plane {
            normal, distance, ..
        } = p.shape
        {
            assert_eq!(normal, [0.0, 1.0, 0.0]);
            assert!((distance - (-2.0)).abs() < 1e-6);
        } else {
            panic!("expected Plane shape");
        }
    }

    #[test]
    fn plane_from_axis_preset_z() {
        let p = plane_from_axis_preset(ClipAxis::Z, 0.5);
        if let ClipShape::Plane { normal, .. } = p.shape {
            assert_eq!(normal, [0.0, 0.0, 1.0]);
        } else {
            panic!("expected Plane shape");
        }
    }

    // --- Hit-test tests ---

    #[test]
    fn hit_test_quad_center_hit() {
        let result = hit_test_plane_quad(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            1.0,
        );
        assert!(matches!(result, ClipPlaneHit::Quad { t } if (t - 5.0).abs() < 1e-5));
    }

    #[test]
    fn hit_test_quad_outside_returns_none() {
        let result = hit_test_plane_quad(
            glam::Vec3::new(2.0, 0.0, 5.0), // x=2 > extent=1
            glam::Vec3::new(0.0, 0.0, -1.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            1.0,
        );
        assert_eq!(result, ClipPlaneHit::None);
    }

    #[test]
    fn hit_test_quad_parallel_returns_none() {
        let result = hit_test_plane_quad(
            glam::Vec3::ZERO,
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
            1.0,
        );
        assert_eq!(result, ClipPlaneHit::None);
    }

    #[test]
    fn plane_tangents_y_normal_produces_orthogonal_vectors() {
        let n = glam::Vec3::Y;
        let (t1, t2) = plane_tangents(n);
        assert!(t1.dot(n).abs() < 1e-5, "t1 not perpendicular to normal");
        assert!(t2.dot(n).abs() < 1e-5, "t2 not perpendicular to normal");
        assert!(
            t1.dot(t2).abs() < 1e-5,
            "t1 and t2 not perpendicular to each other"
        );
        assert!((t1.length() - 1.0).abs() < 1e-5, "t1 not unit length");
        assert!((t2.length() - 1.0).abs() < 1e-5, "t2 not unit length");
    }

    #[test]
    fn plane_tangents_x_normal_produces_orthogonal_vectors() {
        let n = glam::Vec3::X;
        let (t1, t2) = plane_tangents(n);
        assert!(t1.dot(n).abs() < 1e-5);
        assert!(t2.dot(n).abs() < 1e-5);
        assert!(t1.dot(t2).abs() < 1e-5);
    }

    // --- Controller lifecycle tests ---

    #[test]
    fn controller_new_is_idle() {
        assert!(!ClipPlaneController::new().is_active());
    }

    #[test]
    fn controller_begin_distance_activates() {
        let mut ctrl = ClipPlaneController::new();
        ctrl.begin_distance(&default_plane());
        assert!(ctrl.is_active());
    }

    #[test]
    fn controller_reset_clears_session() {
        let mut ctrl = ClipPlaneController::new();
        ctrl.begin_distance(&default_plane());
        ctrl.reset();
        assert!(!ctrl.is_active());
    }

    #[test]
    fn controller_begin_distance_no_op_when_active() {
        let mut ctrl = ClipPlaneController::new();
        ctrl.begin_distance(&default_plane());
        // Second call is a no-op.
        let other = ClipObject {
            shape: ClipShape::Plane {
                normal: [1.0, 0.0, 0.0],
                distance: 5.0,
                cap_colour: None,
            },
            enabled: true,
            ..ClipObject::default()
        };
        ctrl.begin_distance(&other);
        // Still in the first session's plane.
        assert!(ctrl.is_active());
    }

    #[test]
    fn controller_idle_returns_none() {
        let mut ctrl = ClipPlaneController::new();
        let result = ctrl.update(&ActionFrame::default(), idle_ctx());
        assert_eq!(result, ClipPlaneResult::None);
        assert!(!ctrl.is_active());
    }

    #[test]
    fn controller_escape_cancels_active_session() {
        let mut ctrl = ClipPlaneController::new();
        ctrl.begin_distance(&default_plane());

        let mut frame = ActionFrame::default();
        frame
            .actions
            .insert(Action::Cancel, ResolvedActionState::Pressed);
        let mut ctx = idle_ctx();
        ctx.dragging = true;

        let result = ctrl.update(&frame, ctx);
        assert_eq!(result, ClipPlaneResult::Cancel);
        assert!(!ctrl.is_active());
    }

    #[test]
    fn controller_drag_release_commits() {
        let mut ctrl = ClipPlaneController::new();
        ctrl.begin_distance(&default_plane());
        // dragging = false -> commit
        let result = ctrl.update(&ActionFrame::default(), idle_ctx());
        assert_eq!(result, ClipPlaneResult::Commit);
        assert!(!ctrl.is_active());
    }
}
