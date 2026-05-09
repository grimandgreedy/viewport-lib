//! Interactive 3D probe and region widgets.
//!
//! Each widget is a pure CPU state struct (like `Gizmo`) that the host app owns.
//! Push render items from the widget into `SceneFrame` each frame, call `update()`
//! to advance state, and read public fields for results.
//!
//! Suppress orbit while a widget is active using the same pattern as
//! `ManipulationController`:
//!
//! ```rust,ignore
//! if probe.is_active() {
//!     orbit.resolve();
//! } else {
//!     orbit.apply_to_camera(&mut camera);
//! }
//! ```

pub mod box_widget;
pub mod cylinder;
pub mod disk;
pub mod line_probe;
pub mod plane;
pub mod polyline_widget;
pub mod sphere;
pub mod spline;

pub use box_widget::BoxWidget;
pub use cylinder::CylinderWidget;
pub use disk::DiskWidget;
pub use line_probe::LineProbeWidget;
pub use plane::PlaneWidget;
pub use polyline_widget::PolylineWidget;
pub use sphere::SphereWidget;
pub use spline::SplineWidget;

use crate::renderer::RenderCamera;

// ---------------------------------------------------------------------------
// WidgetContext
// ---------------------------------------------------------------------------

/// Per-frame input state passed to widget `update()` methods.
///
/// Build this from the `ActionFrame` and `CameraFrame` your app already has.
/// Mirrors the shape of [`crate::ManipulationContext`].
#[derive(Clone, Debug)]
pub struct WidgetContext {
    /// Camera state for this frame (used for ray construction and drag projection).
    pub camera: RenderCamera,
    /// Viewport width and height in pixels.
    pub viewport_size: glam::Vec2,
    /// Mouse cursor position relative to the viewport top-left, in pixels.
    pub cursor_viewport: glam::Vec2,
    /// True on the first frame that a left-button drag crosses the egui drag threshold.
    pub drag_started: bool,
    /// True while the left mouse button is held after crossing the drag threshold.
    pub dragging: bool,
    /// True on the frame the left mouse button is released.
    pub released: bool,
    /// True on the second click within the double-click time window.
    ///
    /// Used by `PolylineWidget` to insert or remove control points. Set from the
    /// framework's double-click event (e.g. `egui::Response::double_clicked()`).
    /// Leave `false` if the host does not need double-click interactions.
    pub double_clicked: bool,
}

// ---------------------------------------------------------------------------
// WidgetResult
// ---------------------------------------------------------------------------

/// Result returned by widget `update()` calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WidgetResult {
    /// Nothing changed this frame.
    None,
    /// The widget state changed (endpoint moved, size changed, point added/removed, etc.).
    Updated,
}

// ---------------------------------------------------------------------------
// Shared internal helpers
// ---------------------------------------------------------------------------

/// Compute a world-space radius that maps to `target_px` pixels on screen.
///
/// Used to keep handle spheres at a constant apparent screen size.
pub(super) fn handle_world_radius(
    pos: glam::Vec3,
    camera: &RenderCamera,
    viewport_height: f32,
    target_px: f32,
) -> f32 {
    let eye = glam::Vec3::from(camera.eye_position);
    let dist = (pos - eye).length().max(0.001);
    let world_per_px = 2.0 * (camera.fov * 0.5).tan() * dist / viewport_height.max(1.0);
    world_per_px * target_px
}

/// Build a ray from the context cursor position.
pub(super) fn ctx_ray(ctx: &WidgetContext) -> (glam::Vec3, glam::Vec3) {
    let vp = ctx.camera.projection * ctx.camera.view;
    crate::interaction::picking::screen_to_ray(ctx.cursor_viewport, ctx.viewport_size, vp.inverse())
}

/// Shortest distance from a ray to a point.
pub(super) fn ray_point_dist(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    point: glam::Vec3,
) -> f32 {
    let t = (point - ray_origin).dot(ray_dir).max(0.0);
    (ray_origin + ray_dir * t - point).length()
}

/// Returns a unit vector perpendicular to `n`.
pub(super) fn any_perpendicular(n: glam::Vec3) -> glam::Vec3 {
    let len = n.length();
    if len < 1e-6 { return glam::Vec3::X; }
    let n = n / len;
    if n.x.abs() < 0.9 {
        n.cross(glam::Vec3::X).normalize()
    } else {
        n.cross(glam::Vec3::Y).normalize()
    }
}

/// Returns two unit vectors `(u, v)` that are mutually perpendicular and perpendicular to `n`.
pub(super) fn any_perpendicular_pair(n: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let u = any_perpendicular(n);
    let len = n.length();
    let n_unit = if len > 1e-6 { n / len } else { glam::Vec3::Z };
    let v = n_unit.cross(u);
    (u, v)
}
