//! Custom transform gizmo: translate, rotate, and scale handles rendered over
//! the selected object.
//!
//! # Architecture
//!
//! `Gizmo` is a pure CPU state struct that lives in the host application. It stores the
//! current mode (translate/rotate/scale), which axis is hovered or active, and the
//! transform snapshot captured at drag start for undo.
//!
//! Rendering goes through the 2D overlay system: `gizmo_overlay::build_gizmo_overlays`
//! projects the handles to overlay shapes and polylines each frame, so the gizmo
//! draws on top of the scene without a dedicated GPU pipeline. This module keeps
//! the CPU state, the shared proportions/colours, `compute_gizmo_scale`, and hit
//! testing; the overlay builder reuses all of them so what is drawn matches what
//! is picked.
//!
//! Hit testing uses cylinder-distance approximation via `parry3d`-style math:
//! we compute the closest-approach distance from the ray to each axis line segment,
//! then compare against a threshold. This avoids the parry3d dependency in the gizmo
//! module itself (the gizmo vertices are already in gizmo-local space).

/// Pivot point mode for the gizmo : determines where the transform center is.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PivotMode {
    /// Average of all selected objects' positions (default).
    SelectionCentroid,
    /// Each object transforms around its own origin.
    IndividualOrigins,
    /// Median of selected object positions (alias for SelectionCentroid in single-pivot ops).
    MedianPoint,
    /// World origin (0, 0, 0).
    WorldOrigin,
    /// Arbitrary 3D cursor position.
    Cursor3D(glam::Vec3),
}

impl PivotMode {
    /// Advance to the next mode in the cycle.
    ///
    /// Cycle order: `SelectionCentroid` -> `IndividualOrigins` -> `MedianPoint` -> `WorldOrigin` -> ...
    ///
    /// `Cursor3D` is excluded from the cycle (it requires an explicit position) and
    /// falls back to `SelectionCentroid`.
    pub fn cycle_next(self) -> Self {
        match self {
            PivotMode::SelectionCentroid => PivotMode::IndividualOrigins,
            PivotMode::IndividualOrigins => PivotMode::MedianPoint,
            PivotMode::MedianPoint => PivotMode::WorldOrigin,
            PivotMode::WorldOrigin => PivotMode::SelectionCentroid,
            PivotMode::Cursor3D(_) => PivotMode::SelectionCentroid,
        }
    }

    /// Step back to the previous mode in the cycle.
    pub fn cycle_prev(self) -> Self {
        match self {
            PivotMode::SelectionCentroid => PivotMode::WorldOrigin,
            PivotMode::IndividualOrigins => PivotMode::SelectionCentroid,
            PivotMode::MedianPoint => PivotMode::IndividualOrigins,
            PivotMode::WorldOrigin => PivotMode::MedianPoint,
            PivotMode::Cursor3D(_) => PivotMode::SelectionCentroid,
        }
    }

    /// Short human-readable label for HUD display.
    pub fn label(self) -> &'static str {
        match self {
            PivotMode::SelectionCentroid => "Selection Centroid",
            PivotMode::IndividualOrigins => "Individual Origins",
            PivotMode::MedianPoint => "Median Point",
            PivotMode::WorldOrigin => "World Origin",
            PivotMode::Cursor3D(_) => "3D Cursor",
        }
    }
}

/// Compute the gizmo center based on the given `PivotMode`, selection, and position resolver.
///
/// Returns `None` if the selection is empty or positions are unavailable.
pub fn gizmo_center_for_pivot(
    pivot: &PivotMode,
    selection: &crate::interaction::select::selection::Selection,
    position_fn: impl Fn(crate::interaction::select::selection::NodeId) -> Option<glam::Vec3>,
) -> Option<glam::Vec3> {
    if selection.is_empty() {
        return None;
    }
    match pivot {
        PivotMode::SelectionCentroid | PivotMode::MedianPoint => selection.centroid(position_fn),
        PivotMode::IndividualOrigins => selection.primary().and_then(position_fn),
        PivotMode::WorldOrigin => Some(glam::Vec3::ZERO),
        PivotMode::Cursor3D(pos) => Some(*pos),
    }
}

/// Gizmo interaction mode.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum GizmoMode {
    /// Move selected objects along one or two axes.
    Translate,
    /// Rotate selected objects around an axis.
    Rotate,
    /// Scale selected objects along one or two axes.
    Scale,
}

/// Which axis or handle is being hovered or dragged.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum GizmoAxis {
    /// World or local X axis.
    X,
    /// World or local Y axis.
    Y,
    /// World or local Z axis.
    Z,
    /// XY plane handle (translate/scale in X+Y simultaneously).
    XY,
    /// XZ plane handle.
    XZ,
    /// YZ plane handle.
    YZ,
    /// Screen-space handle (translate in the camera plane).
    Screen,
    /// No axis : used when nothing is hovered or active.
    None,
}

/// Coordinate space for gizmo axis orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoSpace {
    /// Axes aligned to world X/Y/Z.
    World,
    /// Axes aligned to the selected object's local orientation.
    Local,
}

/// Gizmo CPU state : lives in the host application, not a renderer state struct.
///
/// Gizmo state is transient and must NOT be serialized.
/// Note: the drag-start transform snapshot (for undo) is stored in the
/// application struct, not here, to avoid coupling with app-specific types.
pub struct Gizmo {
    /// Current interaction mode.
    pub mode: GizmoMode,
    /// Coordinate space for axis directions.
    pub space: GizmoSpace,
    /// Axis under the mouse cursor (updated each hover frame).
    pub hovered_axis: GizmoAxis,
    /// Axis currently being dragged (set on drag start, cleared on drag end).
    pub active_axis: GizmoAxis,
    /// Mouse position (in viewport-local pixels) at drag start.
    pub drag_start_mouse: Option<glam::Vec2>,
    /// Pivot point mode : determines the transform center for multi-selections.
    pub pivot_mode: PivotMode,
}

impl Gizmo {
    /// Create a gizmo with default translate mode in world space.
    pub fn new() -> Self {
        Self {
            mode: GizmoMode::Translate,
            space: GizmoSpace::World,
            hovered_axis: GizmoAxis::None,
            active_axis: GizmoAxis::None,
            drag_start_mouse: None,
            pivot_mode: PivotMode::SelectionCentroid,
        }
    }

    /// Advance the pivot mode to the next in the cycle.
    ///
    /// Call this when [`crate::interaction::input::Action::CyclePivotModeForward`] fires.
    pub fn cycle_pivot_forward(&mut self) {
        self.pivot_mode = self.pivot_mode.cycle_next();
    }

    /// Step the pivot mode back to the previous in the cycle.
    ///
    /// Call this when [`crate::interaction::input::Action::CyclePivotModeBackward`] fires.
    pub fn cycle_pivot_backward(&mut self) {
        self.pivot_mode = self.pivot_mode.cycle_prev();
    }

    /// Resolve the three axis directions based on the current space and the
    /// given object orientation.
    fn axis_directions(&self, object_orientation: glam::Quat) -> [glam::Vec3; 3] {
        match self.space {
            GizmoSpace::World => [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z],
            GizmoSpace::Local => [
                object_orientation * glam::Vec3::X,
                object_orientation * glam::Vec3::Y,
                object_orientation * glam::Vec3::Z,
            ],
        }
    }

    /// Hit test: given a ray in world space, the gizmo's center position, and
    /// the selected object's orientation, return which handle is under the cursor.
    ///
    /// Uses closest-approach distance from ray to axis line segment for axis
    /// handles, and ray-plane intersection for plane/screen handles.
    ///
    /// # Arguments
    ///
    /// * `ray_origin` : world-space origin of the picking ray
    /// * `ray_dir` : world-space direction of the picking ray (normalized)
    /// * `gizmo_center` : world-space position of the gizmo (== selected object position)
    /// * `gizmo_scale` : world-space length of each axis arm
    pub fn hit_test(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        gizmo_center: glam::Vec3,
        gizmo_scale: f32,
    ) -> GizmoAxis {
        self.hit_test_oriented(
            ray_origin,
            ray_dir,
            gizmo_center,
            gizmo_scale,
            glam::Quat::IDENTITY,
        )
    }

    /// Hit test with an explicit object orientation for local-space gizmo.
    pub fn hit_test_oriented(
        &self,
        ray_origin: glam::Vec3,
        ray_dir: glam::Vec3,
        gizmo_center: glam::Vec3,
        gizmo_scale: f32,
        object_orientation: glam::Quat,
    ) -> GizmoAxis {
        let dirs = self.axis_directions(object_orientation);

        match self.mode {
            GizmoMode::Rotate => {
                // Test rotation rings: each ring is a circle in the plane perpendicular
                // to its axis. We intersect the ray with that plane and check if the
                // hit point lies on the ring (within a tolerance band).
                let ring_radius = gizmo_scale * ROTATION_RING_RADIUS;
                let ring_tolerance = gizmo_scale * 0.15;

                let axis_ids = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];
                let mut best: Option<(GizmoAxis, f32)> = None;

                for i in 0..3 {
                    let normal = dirs[i];
                    let denom = ray_dir.dot(normal);
                    if denom.abs() < 1e-6 {
                        continue;
                    }
                    let t = (gizmo_center - ray_origin).dot(normal) / denom;
                    if t < 0.0 {
                        continue;
                    }
                    let hit_point = ray_origin + ray_dir * t;
                    let dist_from_center = (hit_point - gizmo_center).length();
                    if (dist_from_center - ring_radius).abs() < ring_tolerance
                        && (best.is_none() || t < best.unwrap().1)
                    {
                        best = Some((axis_ids[i], t));
                    }
                }

                best.map(|(a, _)| a).unwrap_or(GizmoAxis::None)
            }
            _ => {
                // Translate / Scale modes: arrows + plane handles + screen handle.
                let hit_radius = gizmo_scale * 0.18;

                // --- Screen handle (center square): check first, highest priority ---
                let screen_size = gizmo_scale * 0.15;
                let to_center = gizmo_center - ray_origin;
                let t_center = to_center.dot(ray_dir);
                if t_center > 0.0 {
                    let closest = ray_origin + ray_dir * t_center;
                    let offset = closest - gizmo_center;
                    if offset.length() < screen_size {
                        return GizmoAxis::Screen;
                    }
                }

                // --- Plane handles (small quads at axis-pair corners) ---
                let plane_offset = gizmo_scale * PLANE_OFFSET;
                let plane_size = gizmo_scale * PLANE_SIZE;

                let plane_handles = [
                    (GizmoAxis::XY, dirs[0], dirs[1], dirs[2]),
                    (GizmoAxis::XZ, dirs[0], dirs[2], dirs[1]),
                    (GizmoAxis::YZ, dirs[1], dirs[2], dirs[0]),
                ];

                let mut best_plane: Option<(GizmoAxis, f32)> = None;
                for (axis, dir_a, dir_b, normal) in &plane_handles {
                    let quad_center = gizmo_center + *dir_a * plane_offset + *dir_b * plane_offset;
                    let denom = ray_dir.dot(*normal);
                    if denom.abs() < 1e-6 {
                        continue;
                    }
                    let t = (quad_center - ray_origin).dot(*normal) / denom;
                    if t < 0.0 {
                        continue;
                    }
                    let hit_point = ray_origin + ray_dir * t;
                    let local = hit_point - quad_center;
                    let a_dist = local.dot(*dir_a).abs();
                    let b_dist = local.dot(*dir_b).abs();
                    if a_dist < plane_size
                        && b_dist < plane_size
                        && (best_plane.is_none() || t < best_plane.unwrap().1)
                    {
                        best_plane = Some((*axis, t));
                    }
                }
                if let Some((axis, _)) = best_plane {
                    return axis;
                }

                // --- Single-axis handles ---
                let axis_ids = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];
                let mut best: Option<(GizmoAxis, f32)> = None;

                for i in 0..3 {
                    let arm_end = gizmo_center + dirs[i] * gizmo_scale;
                    let dist = ray_to_segment_distance(ray_origin, ray_dir, gizmo_center, arm_end);
                    if dist < hit_radius {
                        let t = ray_segment_t(ray_origin, ray_dir, gizmo_center, arm_end);
                        if best.is_none() || t < best.unwrap().1 {
                            best = Some((axis_ids[i], t));
                        }
                    }
                }

                best.map(|(a, _)| a).unwrap_or(GizmoAxis::None)
            }
        }
    }
}

impl Default for Gizmo {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Ray / segment distance math
// ---------------------------------------------------------------------------

/// Compute the shortest distance between a ray and a line segment.
///
/// Returns the distance in world units. Uses the standard closest-point-on-ray
/// to closest-point-on-segment approach.
fn ray_to_segment_distance(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    seg_a: glam::Vec3,
    seg_b: glam::Vec3,
) -> f32 {
    let seg_dir = seg_b - seg_a;
    let w0 = ray_origin - seg_a;

    let a = ray_dir.dot(ray_dir); // |ray_dir|^2 (== 1 if normalized)
    let b = ray_dir.dot(seg_dir);
    let c = seg_dir.dot(seg_dir);
    let d = ray_dir.dot(w0);
    let e = seg_dir.dot(w0);

    let denom = a * c - b * b;

    let (t_ray, t_seg) = if denom.abs() > 1e-8 {
        let t_r = (b * e - c * d) / denom;
        let t_s = (a * e - b * d) / denom;
        (t_r.max(0.0), t_s.clamp(0.0, 1.0))
    } else {
        // Parallel: closest point on segment start.
        (0.0, 0.0)
    };

    let closest_ray = ray_origin + ray_dir * t_ray;
    let closest_seg = seg_a + seg_dir * t_seg;
    (closest_ray - closest_seg).length()
}

/// Return the ray parameter `t` at which the ray is closest to the segment.
///
/// Used to pick the nearest axis when multiple axes are within hit radius.
fn ray_segment_t(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    seg_a: glam::Vec3,
    seg_b: glam::Vec3,
) -> f32 {
    let seg_dir = seg_b - seg_a;
    let w0 = ray_origin - seg_a;

    let a = ray_dir.dot(ray_dir);
    let b = ray_dir.dot(seg_dir);
    let c = seg_dir.dot(seg_dir);
    let d = ray_dir.dot(w0);
    let e = seg_dir.dot(w0);

    let denom = a * c - b * b;
    if denom.abs() > 1e-8 {
        let t_r = (b * e - c * d) / denom;
        t_r.max(0.0)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Gizmo colours (shared by the overlay gizmo builder)
// ---------------------------------------------------------------------------

/// Axis colour definitions (per UI-SPEC).
/// X = red, Y = green, Z = blue; brightened variants for hover.
const X_COLOUR: [f32; 4] = [0.878, 0.322, 0.322, 1.0]; // #e05252
const Y_COLOUR: [f32; 4] = [0.361, 0.722, 0.361, 1.0]; // #5cb85c
const Z_COLOUR: [f32; 4] = [0.290, 0.620, 1.0, 1.0]; // #4a9eff

const X_COLOUR_HOV: [f32; 4] = [1.0, 0.518, 0.518, 1.0]; // X * 1.3 clamped
const Y_COLOUR_HOV: [f32; 4] = [0.469, 0.938, 0.469, 1.0]; // Y * 1.3 clamped
const Z_COLOUR_HOV: [f32; 4] = [0.377, 0.806, 1.0, 1.0]; // Z * 1.3 clamped

const SCREEN_COLOUR: [f32; 4] = [0.9, 0.9, 0.9, 0.6];
const SCREEN_COLOUR_HOV: [f32; 4] = [1.0, 1.0, 1.0, 0.8];
const PLANE_ALPHA: f32 = 0.3;
const PLANE_ALPHA_HOV: f32 = 0.5;

/// Select the base or hover colour for an axis based on whether it's hovered.
pub(crate) fn axis_colour(axis: GizmoAxis, hovered: GizmoAxis) -> [f32; 4] {
    let is_hovered = axis == hovered;
    match axis {
        GizmoAxis::X => {
            if is_hovered {
                X_COLOUR_HOV
            } else {
                X_COLOUR
            }
        }
        GizmoAxis::Y => {
            if is_hovered {
                Y_COLOUR_HOV
            } else {
                Y_COLOUR
            }
        }
        GizmoAxis::Z => {
            if is_hovered {
                Z_COLOUR_HOV
            } else {
                Z_COLOUR
            }
        }
        GizmoAxis::Screen => {
            if is_hovered {
                SCREEN_COLOUR_HOV
            } else {
                SCREEN_COLOUR
            }
        }
        _ => [1.0; 4],
    }
}

/// Get the colour for a plane handle, blending the two axis colours.
/// On hover, RGB is brightened by 1.3x (clamped) in addition to the alpha bump.
pub(crate) fn plane_colour(axis: GizmoAxis, hovered: GizmoAxis) -> [f32; 4] {
    let is_hovered = axis == hovered;
    let alpha = if is_hovered {
        PLANE_ALPHA_HOV
    } else {
        PLANE_ALPHA
    };
    let brightness = if is_hovered { 1.3 } else { 1.0 };
    let (c1, c2) = match axis {
        GizmoAxis::XY => (X_COLOUR, Y_COLOUR),
        GizmoAxis::XZ => (X_COLOUR, Z_COLOUR),
        GizmoAxis::YZ => (Y_COLOUR, Z_COLOUR),
        _ => return [1.0, 1.0, 1.0, alpha],
    };
    [
        ((c1[0] + c2[0]) * 0.5 * brightness).min(1.0),
        ((c1[1] + c2[1]) * 0.5 * brightness).min(1.0),
        ((c1[2] + c2[2]) * 0.5 * brightness).min(1.0),
        alpha,
    ]
}

/// Arrow proportions in gizmo-local units. Shared by the overlay gizmo builder
/// (`gizmo_overlay`) so the drawn handles match the CPU `hit_test` extents.
pub(crate) const SHAFT_RADIUS: f32 = 0.035;
pub(crate) const SHAFT_LENGTH: f32 = 0.70;
/// Major radius of the rotation rings : used in both mesh generation and hit testing.
pub const ROTATION_RING_RADIUS: f32 = 0.85;
/// Tube (minor) radius of the rotation rings, in gizmo-local units.
pub(crate) const RING_TUBE_RADIUS: f32 = 0.025;
pub(crate) const CONE_RADIUS: f32 = 0.09;
pub(crate) const CONE_LENGTH: f32 = 0.30;
pub(crate) const CUBE_HALF: f32 = 0.06;
/// Distance from the gizmo centre to the plane-handle centre, in gizmo-local units.
pub(crate) const PLANE_OFFSET: f32 = 0.25;
/// Half-extent of a plane handle, in gizmo-local units.
pub(crate) const PLANE_SIZE: f32 = 0.15;
/// Half-extent of the drawn screen-space centre handle, in gizmo-local units.
/// The hit-test grab area is deliberately larger (see `hit_test`).
pub(crate) const SCREEN_HANDLE_SIZE: f32 = 0.08;

// ---------------------------------------------------------------------------
// Gizmo scale computation
// ---------------------------------------------------------------------------

/// Compute the world-space scale for the gizmo so it appears at a consistent
/// screen size regardless of camera distance.
///
/// `gizmo_center_world` : position of the gizmo (selected object)
/// `camera_eye` : camera eye position
/// `fov_y` : camera vertical field of view (radians)
/// `viewport_height` : viewport height in pixels
/// Target: gizmo should be approximately 100px tall on screen.
pub fn compute_gizmo_scale(
    gizmo_center_world: glam::Vec3,
    camera_eye: glam::Vec3,
    fov_y: f32,
    viewport_height: f32,
) -> f32 {
    let dist = (gizmo_center_world - camera_eye).length();
    // world_units_per_pixel at distance = 2 * tan(fov_y/2) * dist / viewport_height
    let world_per_px = 2.0 * (fov_y * 0.5).tan() * dist / viewport_height;
    // Target: 100 pixels = gizmo total length.
    let target_px = 100.0_f32;
    world_per_px * target_px
}

// ---------------------------------------------------------------------------
// Drag math helpers
// ---------------------------------------------------------------------------

/// Project mouse delta (in screen pixels) onto a world-space axis direction.
///
/// Returns the signed scalar amount to move along the axis.
///
/// # Arguments
/// * `drag_delta` : mouse movement in pixels since drag start (egui drag_delta())
/// * `axis_world` : world-space axis direction (X, Y, or Z unit vector)
/// * `view_proj` : camera view-projection matrix
/// * `gizmo_center` : world-space gizmo center (selected object position)
/// * `viewport_size` : viewport size in pixels
pub fn project_drag_onto_axis(
    drag_delta: glam::Vec2,
    axis_world: glam::Vec3,
    view_proj: glam::Mat4,
    gizmo_center: glam::Vec3,
    viewport_size: glam::Vec2,
) -> f32 {
    // Project the axis tip and base to screen space.
    let base_ndc = view_proj.project_point3(gizmo_center);
    let tip_ndc = view_proj.project_point3(gizmo_center + axis_world);

    // Convert NDC to screen pixels.
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

    // Project the mouse drag onto the screen-space axis direction.
    let axis_screen_norm = axis_screen / axis_screen_len;
    let drag_along_axis = drag_delta.dot(axis_screen_norm);

    // Convert screen pixels back to world units.
    // 1 world unit projects to `axis_screen_len` pixels.
    drag_along_axis / axis_screen_len
}

/// Project mouse delta onto a rotation axis, returning an angle in radians.
///
/// For rotation, we use: move right/up on screen = positive rotation.
/// The axis being dragged is perpendicular in screen space.
pub fn project_drag_onto_rotation(
    drag_delta: glam::Vec2,
    axis_world: glam::Vec3,
    view: glam::Mat4,
) -> f32 {
    // Project the rotation axis into camera space to find the perpendicular
    // screen direction.
    let axis_cam = (view * axis_world.extend(0.0))
        .truncate()
        .normalize_or_zero();

    // The perpendicular of the axis in screen space.
    let perp = glam::Vec2::new(-axis_cam.y, axis_cam.x);
    let perp_len = perp.length();
    if perp_len < 1e-4 {
        return 0.0;
    }

    // Project drag delta onto the perpendicular direction.
    let perp_norm = perp / perp_len;
    let drag_amount = drag_delta.dot(perp_norm);

    // Scale: 1 full-screen drag = 2pi radians (reasonable sensitivity).
    drag_amount * 0.02
}

/// Project mouse delta onto a world-space plane defined by two axis directions.
///
/// Returns the 3D world-space displacement vector in the plane.
pub fn project_drag_onto_plane(
    drag_delta: glam::Vec2,
    axis_a: glam::Vec3,
    axis_b: glam::Vec3,
    view_proj: glam::Mat4,
    gizmo_center: glam::Vec3,
    viewport_size: glam::Vec2,
) -> glam::Vec3 {
    let a_amount =
        project_drag_onto_axis(drag_delta, axis_a, view_proj, gizmo_center, viewport_size);
    let b_amount =
        project_drag_onto_axis(drag_delta, axis_b, view_proj, gizmo_center, viewport_size);
    axis_a * a_amount + axis_b * b_amount
}

/// Project mouse delta onto the camera-facing (screen) plane.
///
/// Returns the 3D world-space displacement vector.
pub fn project_drag_onto_screen_plane(
    drag_delta: glam::Vec2,
    camera_right: glam::Vec3,
    camera_up: glam::Vec3,
    view_proj: glam::Mat4,
    gizmo_center: glam::Vec3,
    viewport_size: glam::Vec2,
) -> glam::Vec3 {
    project_drag_onto_plane(
        drag_delta,
        camera_right,
        camera_up,
        view_proj,
        gizmo_center,
        viewport_size,
    )
}

/// Compute the gizmo center from a multi-selection by averaging positions.
///
/// Thin wrapper around `Selection::centroid()` for discoverability in gizmo workflows.
pub fn gizmo_center_from_selection(
    selection: &crate::interaction::select::selection::Selection,
    position_fn: impl Fn(crate::interaction::select::selection::NodeId) -> Option<glam::Vec3>,
) -> Option<glam::Vec3> {
    selection.centroid(position_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gizmo() -> Gizmo {
        Gizmo::new()
    }

    #[test]
    fn test_hit_test_x_axis() {
        let g = gizmo();
        let center = glam::Vec3::ZERO;
        let scale = 1.0;
        let axis = g.hit_test(
            glam::Vec3::new(0.5, 0.5, 0.0),
            glam::Vec3::new(0.0, -1.0, 0.0),
            center,
            scale,
        );
        assert_eq!(axis, GizmoAxis::X);
    }

    #[test]
    fn test_hit_test_y_axis() {
        let g = gizmo();
        let center = glam::Vec3::ZERO;
        let scale = 1.0;
        let axis = g.hit_test(
            glam::Vec3::new(0.5, 0.5, 0.0),
            glam::Vec3::new(-1.0, 0.0, 0.0),
            center,
            scale,
        );
        assert_eq!(axis, GizmoAxis::Y);
    }

    #[test]
    fn test_hit_test_z_axis() {
        let g = gizmo();
        let center = glam::Vec3::ZERO;
        let scale = 1.0;
        let axis = g.hit_test(
            glam::Vec3::new(0.0, 0.5, 0.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            center,
            scale,
        );
        assert_eq!(axis, GizmoAxis::Z);
    }

    #[test]
    fn test_hit_test_miss() {
        let g = gizmo();
        let center = glam::Vec3::ZERO;
        let scale = 1.0;
        let axis = g.hit_test(
            glam::Vec3::new(10.0, 10.0, 10.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            center,
            scale,
        );
        assert_eq!(axis, GizmoAxis::None);
    }

    #[test]
    fn test_hit_test_plane_handle_xy() {
        let g = gizmo();
        let center = glam::Vec3::ZERO;
        let scale = 1.0;
        // Ray coming from +Z, hitting the XY plane handle area (at ~0.25, 0.25).
        let axis = g.hit_test_oriented(
            glam::Vec3::new(0.25, 0.25, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            center,
            scale,
            glam::Quat::IDENTITY,
        );
        assert_eq!(axis, GizmoAxis::XY, "expected XY plane handle hit");
    }

    #[test]
    fn test_hit_test_local_orientation() {
        let mut g = gizmo();
        g.space = GizmoSpace::Local;
        let center = glam::Vec3::ZERO;
        let scale = 1.0;
        // Rotate the object 90 deg around Y: local X -> world -Z, local Z -> world +X.
        let rot = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);

        // A ray that hits along world -Z direction should hit local X axis.
        // Local X arm goes from origin to (0, 0, -1).
        let axis = g.hit_test_oriented(
            glam::Vec3::new(0.0, 0.5, -0.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            center,
            scale,
            rot,
        );
        assert_eq!(
            axis,
            GizmoAxis::X,
            "local X axis should be along world -Z after 90 deg Y rotation"
        );
    }

    #[test]
    fn test_project_drag_onto_axis() {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let vp = proj * view;
        let viewport_size = glam::Vec2::new(800.0, 600.0);
        let center = glam::Vec3::ZERO;

        let result = project_drag_onto_axis(
            glam::Vec2::new(100.0, 0.0),
            glam::Vec3::X,
            vp,
            center,
            viewport_size,
        );
        assert!(result > 0.0, "expected positive drag along X, got {result}");
    }

    #[test]
    fn test_project_drag_onto_plane() {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 5.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let vp = proj * view;
        let viewport_size = glam::Vec2::new(800.0, 600.0);
        let center = glam::Vec3::ZERO;

        let result = project_drag_onto_plane(
            glam::Vec2::new(100.0, 0.0),
            glam::Vec3::X,
            glam::Vec3::Z,
            vp,
            center,
            viewport_size,
        );
        // Should have components along X and Z.
        assert!(
            result.length() > 0.0,
            "plane drag should produce non-zero displacement"
        );
        assert!(
            result.y.abs() < 1e-4,
            "XZ plane drag should have no Y component"
        );
    }

    #[test]
    fn test_screen_handle_hit() {
        let g = gizmo();
        let center = glam::Vec3::ZERO;
        let scale = 1.0;
        // Ray aimed directly at origin from +Z.
        let axis = g.hit_test(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::new(0.0, 0.0, -1.0),
            center,
            scale,
        );
        assert_eq!(
            axis,
            GizmoAxis::Screen,
            "ray at center should hit Screen handle"
        );
    }

    #[test]
    fn test_compute_gizmo_scale() {
        let scale = compute_gizmo_scale(
            glam::Vec3::ZERO,
            glam::Vec3::new(0.0, 0.0, 10.0),
            std::f32::consts::FRAC_PI_4,
            600.0,
        );
        assert!(scale > 0.0, "gizmo scale should be positive");
        assert!((scale - 1.381).abs() < 0.1, "unexpected scale: {scale}");
    }

    #[test]
    fn test_gizmo_center_single_selection() {
        let mut sel = crate::interaction::select::selection::Selection::new();
        sel.select_one(1);
        let center = gizmo_center_from_selection(&sel, |id| match id {
            1 => Some(glam::Vec3::new(3.0, 0.0, 0.0)),
            _ => None,
        });
        let c = center.unwrap();
        assert!((c.x - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_gizmo_center_multi_selection() {
        let mut sel = crate::interaction::select::selection::Selection::new();
        sel.add(1);
        sel.add(2);
        let center = gizmo_center_from_selection(&sel, |id| match id {
            1 => Some(glam::Vec3::new(0.0, 0.0, 0.0)),
            2 => Some(glam::Vec3::new(4.0, 0.0, 0.0)),
            _ => None,
        });
        let c = center.unwrap();
        assert!((c.x - 2.0).abs() < 1e-5);
    }

    // --- PivotMode tests ---

    #[test]
    fn test_pivot_selection_centroid_matches_centroid() {
        let mut sel = crate::interaction::select::selection::Selection::new();
        sel.add(1);
        sel.add(2);
        let pos_fn = |id: crate::interaction::select::selection::NodeId| match id {
            1 => Some(glam::Vec3::new(0.0, 0.0, 0.0)),
            2 => Some(glam::Vec3::new(4.0, 0.0, 0.0)),
            _ => None,
        };
        let centroid = gizmo_center_from_selection(&sel, pos_fn);
        let pivot = gizmo_center_for_pivot(&PivotMode::SelectionCentroid, &sel, pos_fn);
        assert_eq!(centroid, pivot);
    }

    #[test]
    fn test_pivot_world_origin_returns_zero() {
        let mut sel = crate::interaction::select::selection::Selection::new();
        sel.add(1);
        let result = gizmo_center_for_pivot(&PivotMode::WorldOrigin, &sel, |_| {
            Some(glam::Vec3::new(5.0, 0.0, 0.0))
        });
        assert_eq!(result, Some(glam::Vec3::ZERO));
    }

    #[test]
    fn test_pivot_world_origin_empty_selection_returns_none() {
        let sel = crate::interaction::select::selection::Selection::new();
        let result = gizmo_center_for_pivot(&PivotMode::WorldOrigin, &sel, |_| None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_pivot_individual_origins_uses_primary() {
        let mut sel = crate::interaction::select::selection::Selection::new();
        sel.add(1);
        sel.add(2); // primary = 2
        let result = gizmo_center_for_pivot(&PivotMode::IndividualOrigins, &sel, |id| match id {
            1 => Some(glam::Vec3::new(1.0, 0.0, 0.0)),
            2 => Some(glam::Vec3::new(9.0, 0.0, 0.0)),
            _ => None,
        });
        let c = result.unwrap();
        assert!(
            (c.x - 9.0).abs() < 1e-5,
            "expected primary (node 2) position x=9, got {}",
            c.x
        );
    }

    #[test]
    fn test_pivot_median_point_same_as_centroid() {
        let mut sel = crate::interaction::select::selection::Selection::new();
        sel.add(1);
        sel.add(2);
        let pos_fn = |id: crate::interaction::select::selection::NodeId| match id {
            1 => Some(glam::Vec3::new(0.0, 0.0, 0.0)),
            2 => Some(glam::Vec3::new(6.0, 0.0, 0.0)),
            _ => None,
        };
        let result = gizmo_center_for_pivot(&PivotMode::MedianPoint, &sel, pos_fn);
        let c = result.unwrap();
        assert!((c.x - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_pivot_cursor3d_returns_cursor_pos() {
        let mut sel = crate::interaction::select::selection::Selection::new();
        sel.add(1);
        let cursor = glam::Vec3::new(7.0, 2.0, 3.0);
        let result = gizmo_center_for_pivot(&PivotMode::Cursor3D(cursor), &sel, |_| {
            Some(glam::Vec3::ZERO)
        });
        assert_eq!(result, Some(cursor));
    }

    #[test]
    fn test_pivot_cursor3d_empty_selection_returns_none() {
        let sel = crate::interaction::select::selection::Selection::new();
        let cursor = glam::Vec3::new(1.0, 2.0, 3.0);
        let result = gizmo_center_for_pivot(&PivotMode::Cursor3D(cursor), &sel, |_| None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_gizmo_pivot_mode_field_defaults_to_selection_centroid() {
        let g = Gizmo::new();
        assert!(matches!(g.pivot_mode, PivotMode::SelectionCentroid));
    }

    // --- Pivot cycling tests ---

    #[test]
    fn test_cycle_next_full_round_trip() {
        let start = PivotMode::SelectionCentroid;
        let after_one = start.cycle_next();
        assert!(matches!(after_one, PivotMode::IndividualOrigins));
        let after_two = after_one.cycle_next();
        assert!(matches!(after_two, PivotMode::MedianPoint));
        let after_three = after_two.cycle_next();
        assert!(matches!(after_three, PivotMode::WorldOrigin));
        let wrapped = after_three.cycle_next();
        assert!(matches!(wrapped, PivotMode::SelectionCentroid));
    }

    #[test]
    fn test_cycle_prev_full_round_trip() {
        let start = PivotMode::SelectionCentroid;
        let after_one = start.cycle_prev();
        assert!(matches!(after_one, PivotMode::WorldOrigin));
        let after_two = after_one.cycle_prev();
        assert!(matches!(after_two, PivotMode::MedianPoint));
        let after_three = after_two.cycle_prev();
        assert!(matches!(after_three, PivotMode::IndividualOrigins));
        let wrapped = after_three.cycle_prev();
        assert!(matches!(wrapped, PivotMode::SelectionCentroid));
    }

    #[test]
    fn test_cycle_next_and_prev_are_inverses() {
        for mode in [
            PivotMode::SelectionCentroid,
            PivotMode::IndividualOrigins,
            PivotMode::MedianPoint,
            PivotMode::WorldOrigin,
        ] {
            assert_eq!(mode.cycle_next().cycle_prev(), mode);
            assert_eq!(mode.cycle_prev().cycle_next(), mode);
        }
    }

    #[test]
    fn test_cursor3d_falls_back_to_selection_centroid_on_cycle() {
        let cursor = PivotMode::Cursor3D(glam::Vec3::ONE);
        assert!(matches!(cursor.cycle_next(), PivotMode::SelectionCentroid));
        assert!(matches!(cursor.cycle_prev(), PivotMode::SelectionCentroid));
    }

    #[test]
    fn test_label_returns_non_empty_str() {
        for mode in [
            PivotMode::SelectionCentroid,
            PivotMode::IndividualOrigins,
            PivotMode::MedianPoint,
            PivotMode::WorldOrigin,
            PivotMode::Cursor3D(glam::Vec3::ZERO),
        ] {
            assert!(!mode.label().is_empty());
        }
    }

    #[test]
    fn test_gizmo_cycle_pivot_forward_and_backward() {
        let mut g = Gizmo::new();
        assert!(matches!(g.pivot_mode, PivotMode::SelectionCentroid));
        g.cycle_pivot_forward();
        assert!(matches!(g.pivot_mode, PivotMode::IndividualOrigins));
        g.cycle_pivot_backward();
        assert!(matches!(g.pivot_mode, PivotMode::SelectionCentroid));
    }
}
