//! Custom transform gizmo: translate, rotate, and scale handles rendered over
//! the selected object.
//!
//! # Architecture
//!
//! `Gizmo` is a pure CPU state struct that lives in the host application. It stores the
//! current mode (translate/rotate/scale), which axis is hovered or active, and the
//! transform snapshot captured at drag start for undo.
//!
//! Rendering is handled by `gizmo_pipeline` in `ViewportGpuResources`, which uses
//! `depth_compare: Always` so the gizmo always appears on top of scene geometry.
//!
//! Hit testing uses cylinder-distance approximation via `parry3d`-style math:
//! we compute the closest-approach distance from the ray to each axis line segment,
//! then compare against a threshold. This avoids the parry3d dependency in the gizmo
//! module itself (the gizmo vertices are already in gizmo-local space).

/// Pivot point mode for the gizmo — determines where the transform center is.
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

/// Compute the gizmo center based on the given `PivotMode`, selection, and position resolver.
///
/// Returns `None` if the selection is empty or positions are unavailable.
pub fn gizmo_center_for_pivot(
    pivot: &PivotMode,
    selection: &crate::interaction::selection::Selection,
    position_fn: impl Fn(crate::interaction::selection::NodeId) -> Option<glam::Vec3>,
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
    /// No axis — used when nothing is hovered or active.
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

/// Gizmo CPU state — lives in the host application, not a renderer state struct.
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
    /// Pivot point mode — determines the transform center for multi-selections.
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
    /// * `ray_origin` — world-space origin of the picking ray
    /// * `ray_dir` — world-space direction of the picking ray (normalized)
    /// * `gizmo_center` — world-space position of the gizmo (== selected object position)
    /// * `gizmo_scale` — world-space length of each axis arm
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
                let plane_offset = gizmo_scale * 0.25;
                let plane_size = gizmo_scale * 0.15;

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
// Gizmo mesh generation (mode-aware)
// ---------------------------------------------------------------------------

/// Vertex type reused from resources module (position, normal, color).
pub use crate::resources::Vertex;

/// Axis color definitions (per UI-SPEC).
/// X = red, Y = green, Z = blue; brightened variants for hover.
const X_COLOR: [f32; 4] = [0.878, 0.322, 0.322, 1.0]; // #e05252
const Y_COLOR: [f32; 4] = [0.361, 0.722, 0.361, 1.0]; // #5cb85c
const Z_COLOR: [f32; 4] = [0.290, 0.620, 1.0, 1.0]; // #4a9eff

const X_COLOR_HOV: [f32; 4] = [1.0, 0.518, 0.518, 1.0]; // X * 1.3 clamped
const Y_COLOR_HOV: [f32; 4] = [0.469, 0.938, 0.469, 1.0]; // Y * 1.3 clamped
const Z_COLOR_HOV: [f32; 4] = [0.377, 0.806, 1.0, 1.0]; // Z * 1.3 clamped

const SCREEN_COLOR: [f32; 4] = [0.9, 0.9, 0.9, 0.6];
const SCREEN_COLOR_HOV: [f32; 4] = [1.0, 1.0, 1.0, 0.8];
const PLANE_ALPHA: f32 = 0.3;
const PLANE_ALPHA_HOV: f32 = 0.5;

/// Select the base or hover color for an axis based on whether it's hovered.
fn axis_color(axis: GizmoAxis, hovered: GizmoAxis) -> [f32; 4] {
    let is_hovered = axis == hovered;
    match axis {
        GizmoAxis::X => {
            if is_hovered {
                X_COLOR_HOV
            } else {
                X_COLOR
            }
        }
        GizmoAxis::Y => {
            if is_hovered {
                Y_COLOR_HOV
            } else {
                Y_COLOR
            }
        }
        GizmoAxis::Z => {
            if is_hovered {
                Z_COLOR_HOV
            } else {
                Z_COLOR
            }
        }
        GizmoAxis::Screen => {
            if is_hovered {
                SCREEN_COLOR_HOV
            } else {
                SCREEN_COLOR
            }
        }
        _ => [1.0; 4],
    }
}

/// Get the color for a plane handle, blending the two axis colors.
/// On hover, RGB is brightened by 1.3× (clamped) in addition to the alpha bump.
fn plane_color(axis: GizmoAxis, hovered: GizmoAxis) -> [f32; 4] {
    let is_hovered = axis == hovered;
    let alpha = if is_hovered {
        PLANE_ALPHA_HOV
    } else {
        PLANE_ALPHA
    };
    let brightness = if is_hovered { 1.3 } else { 1.0 };
    let (c1, c2) = match axis {
        GizmoAxis::XY => (X_COLOR, Y_COLOR),
        GizmoAxis::XZ => (X_COLOR, Z_COLOR),
        GizmoAxis::YZ => (Y_COLOR, Z_COLOR),
        _ => return [1.0, 1.0, 1.0, alpha],
    };
    [
        ((c1[0] + c2[0]) * 0.5 * brightness).min(1.0),
        ((c1[1] + c2[1]) * 0.5 * brightness).min(1.0),
        ((c1[2] + c2[2]) * 0.5 * brightness).min(1.0),
        alpha,
    ]
}

/// Build gizmo mesh for the specified mode.
///
/// - **Translate:** arrows + plane quads + center square
/// - **Rotate:** torus ring segments around each axis
/// - **Scale:** arrows with cube tips instead of cones
///
/// All geometry is in gizmo-local space. The `space_orientation` quaternion
/// rotates axis geometry for local-space mode (identity for world space).
///
/// Returns `(vertices, indices)`.
pub(crate) fn build_gizmo_mesh(
    mode: GizmoMode,
    hovered: GizmoAxis,
    space_orientation: glam::Quat,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    match mode {
        GizmoMode::Translate => {
            build_arrows(
                &mut vertices,
                &mut indices,
                hovered,
                space_orientation,
                false,
            );
            build_plane_quads(&mut vertices, &mut indices, hovered, space_orientation);
            build_screen_handle(&mut vertices, &mut indices, hovered);
        }
        GizmoMode::Rotate => {
            build_rotation_rings(&mut vertices, &mut indices, hovered, space_orientation);
        }
        GizmoMode::Scale => {
            build_arrows(
                &mut vertices,
                &mut indices,
                hovered,
                space_orientation,
                true,
            );
        }
    }

    (vertices, indices)
}

/// Arrow proportions in gizmo-local units.
const SHAFT_RADIUS: f32 = 0.035;
const SHAFT_LENGTH: f32 = 0.70;
/// Major radius of the rotation rings — used in both mesh generation and hit testing.
pub const ROTATION_RING_RADIUS: f32 = 0.85;
const CONE_RADIUS: f32 = 0.09;
const CONE_LENGTH: f32 = 0.30;
const CUBE_HALF: f32 = 0.06;
const SEGMENTS: u32 = 8;

/// Build arrow geometry for all 3 axes. If `cube_tips` is true, use cubes
/// instead of cones (for Scale mode).
fn build_arrows(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    hovered: GizmoAxis,
    orientation: glam::Quat,
    cube_tips: bool,
) {
    let base_axes = [
        (GizmoAxis::X, glam::Vec3::X, glam::Vec3::Y),
        (GizmoAxis::Y, glam::Vec3::Y, glam::Vec3::X),
        (GizmoAxis::Z, glam::Vec3::Z, glam::Vec3::Y),
    ];

    for (axis, raw_dir, raw_up) in &base_axes {
        let axis_dir = orientation * *raw_dir;
        let up_hint = orientation * *raw_up;
        let color = axis_color(*axis, hovered);

        let tangent = if axis_dir.abs().dot(orientation * glam::Vec3::Y) > 0.9 {
            axis_dir.cross(up_hint).normalize()
        } else {
            axis_dir.cross(orientation * glam::Vec3::Y).normalize()
        };
        let bitangent = axis_dir.cross(tangent).normalize();

        let base_index = vertices.len() as u32;

        // --- Shaft cylinder ---
        let shaft_bottom = glam::Vec3::ZERO;
        let shaft_top = axis_dir * SHAFT_LENGTH;

        for i in 0..SEGMENTS {
            let angle = (i as f32) * std::f32::consts::TAU / (SEGMENTS as f32);
            let radial = tangent * angle.cos() + bitangent * angle.sin();

            vertices.push(Vertex {
                position: (shaft_bottom + radial * SHAFT_RADIUS).to_array(),
                normal: radial.to_array(),
                color,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
            vertices.push(Vertex {
                position: (shaft_top + radial * SHAFT_RADIUS).to_array(),
                normal: radial.to_array(),
                color,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }

        // Shaft side indices.
        for i in 0..SEGMENTS {
            let next = (i + 1) % SEGMENTS;
            let b0 = base_index + i * 2;
            let t0 = base_index + i * 2 + 1;
            let b1 = base_index + next * 2;
            let t1 = base_index + next * 2 + 1;
            indices.extend_from_slice(&[b0, b1, t0, t0, b1, t1]);
        }

        // Shaft bottom cap.
        let shaft_bottom_center = vertices.len() as u32;
        vertices.push(Vertex {
            position: shaft_bottom.to_array(),
            normal: (-axis_dir).to_array(),
            color,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        });
        for i in 0..SEGMENTS {
            let next = (i + 1) % SEGMENTS;
            let v0 = base_index + i * 2;
            let v1 = base_index + next * 2;
            indices.extend_from_slice(&[shaft_bottom_center, v1, v0]);
        }

        // --- Tip ---
        let tip_base = shaft_top;
        if cube_tips {
            build_cube_tip(
                vertices, indices, tip_base, axis_dir, tangent, bitangent, color,
            );
        } else {
            build_cone_tip(
                vertices, indices, tip_base, axis_dir, tangent, bitangent, color,
            );
        }
    }
}

/// Build a cone tip for translate arrows.
fn build_cone_tip(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    base_center: glam::Vec3,
    axis_dir: glam::Vec3,
    tangent: glam::Vec3,
    bitangent: glam::Vec3,
    color: [f32; 4],
) {
    let cone_tip = base_center + axis_dir * CONE_LENGTH;
    let cone_base_start = vertices.len() as u32;

    // Base ring.
    for i in 0..SEGMENTS {
        let angle = (i as f32) * std::f32::consts::TAU / (SEGMENTS as f32);
        let radial = tangent * angle.cos() + bitangent * angle.sin();
        vertices.push(Vertex {
            position: (base_center + radial * CONE_RADIUS).to_array(),
            normal: (-axis_dir).to_array(),
            color,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        });
    }

    // Base cap center.
    let base_cap_center = vertices.len() as u32;
    vertices.push(Vertex {
        position: base_center.to_array(),
        normal: (-axis_dir).to_array(),
        color,
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 1.0],
    });
    for i in 0..SEGMENTS {
        let next = (i + 1) % SEGMENTS;
        indices.extend_from_slice(&[base_cap_center, cone_base_start + i, cone_base_start + next]);
    }

    // Tip vertex + side triangles.
    let tip_idx = vertices.len() as u32;
    vertices.push(Vertex {
        position: cone_tip.to_array(),
        normal: axis_dir.to_array(),
        color,
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 1.0],
    });
    for i in 0..SEGMENTS {
        let next = (i + 1) % SEGMENTS;
        indices.extend_from_slice(&[cone_base_start + i, cone_base_start + next, tip_idx]);
    }
}

/// Build a cube tip for scale arrows.
fn build_cube_tip(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    center: glam::Vec3,
    axis_dir: glam::Vec3,
    tangent: glam::Vec3,
    bitangent: glam::Vec3,
    color: [f32; 4],
) {
    let cube_center = center + axis_dir * CUBE_HALF;
    let h = CUBE_HALF;

    // 8 corners of the cube.
    let corners = [
        cube_center + axis_dir * (-h) + tangent * (-h) + bitangent * (-h),
        cube_center + axis_dir * h + tangent * (-h) + bitangent * (-h),
        cube_center + axis_dir * h + tangent * h + bitangent * (-h),
        cube_center + axis_dir * (-h) + tangent * h + bitangent * (-h),
        cube_center + axis_dir * (-h) + tangent * (-h) + bitangent * h,
        cube_center + axis_dir * h + tangent * (-h) + bitangent * h,
        cube_center + axis_dir * h + tangent * h + bitangent * h,
        cube_center + axis_dir * (-h) + tangent * h + bitangent * h,
    ];

    // 6 faces (each face = 4 vertices with face normal + 2 triangles).
    let faces: [([usize; 4], glam::Vec3); 6] = [
        ([1, 2, 6, 5], axis_dir),   // +axis
        ([0, 4, 7, 3], -axis_dir),  // -axis
        ([2, 3, 7, 6], tangent),    // +tangent
        ([0, 1, 5, 4], -tangent),   // -tangent
        ([4, 5, 6, 7], bitangent),  // +bitangent
        ([0, 3, 2, 1], -bitangent), // -bitangent
    ];

    for (corner_ids, normal) in &faces {
        let base = vertices.len() as u32;
        for &ci in corner_ids {
            vertices.push(Vertex {
                position: corners[ci].to_array(),
                normal: normal.to_array(),
                color,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Build small transparent plane-handle quads at axis-pair corners.
fn build_plane_quads(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    hovered: GizmoAxis,
    orientation: glam::Quat,
) {
    let plane_offset = 0.25_f32;
    let plane_size = 0.15_f32;

    let planes = [
        (GizmoAxis::XY, glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z),
        (GizmoAxis::XZ, glam::Vec3::X, glam::Vec3::Z, glam::Vec3::Y),
        (GizmoAxis::YZ, glam::Vec3::Y, glam::Vec3::Z, glam::Vec3::X),
    ];

    for (axis, dir_a, dir_b, normal_dir) in &planes {
        let a = orientation * *dir_a;
        let b = orientation * *dir_b;
        let n = orientation * *normal_dir;
        let center = a * plane_offset + b * plane_offset;
        let color = plane_color(*axis, hovered);

        let base = vertices.len() as u32;
        let corners = [
            center + a * (-plane_size) + b * (-plane_size),
            center + a * plane_size + b * (-plane_size),
            center + a * plane_size + b * plane_size,
            center + a * (-plane_size) + b * plane_size,
        ];
        for c in &corners {
            vertices.push(Vertex {
                position: c.to_array(),
                normal: n.to_array(),
                color,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
        // Two triangles (double-sided via two-face winding).
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
}

/// Build a small center square for screen-space translate.
fn build_screen_handle(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, hovered: GizmoAxis) {
    let size = 0.08_f32;
    let color = axis_color(GizmoAxis::Screen, hovered);
    let base = vertices.len() as u32;

    // Small quad in XY plane at the origin. The gizmo uniform will orient it
    // toward the camera via billboard, but for simplicity we emit in XY.
    let corners = [
        glam::Vec3::new(-size, -size, 0.0),
        glam::Vec3::new(size, -size, 0.0),
        glam::Vec3::new(size, size, 0.0),
        glam::Vec3::new(-size, size, 0.0),
    ];
    for c in &corners {
        vertices.push(Vertex {
            position: c.to_array(),
            normal: [0.0, 0.0, 1.0],
            color,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        });
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
}

/// Build torus ring segments for rotation mode.
fn build_rotation_rings(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    hovered: GizmoAxis,
    orientation: glam::Quat,
) {
    let ring_radius = ROTATION_RING_RADIUS; // major radius
    let tube_radius = 0.025_f32; // minor radius
    let ring_segments = 40_u32;
    let tube_segments = 8_u32;

    let axis_data = [
        (GizmoAxis::X, glam::Vec3::X),
        (GizmoAxis::Y, glam::Vec3::Y),
        (GizmoAxis::Z, glam::Vec3::Z),
    ];

    for (axis, raw_dir) in &axis_data {
        let axis_dir = orientation * *raw_dir;
        let color = axis_color(*axis, hovered);

        // Build two perpendicular vectors in the plane of the ring.
        let (ring_u, ring_v) = perpendicular_pair(axis_dir);

        let base = vertices.len() as u32;

        for i in 0..ring_segments {
            let theta = (i as f32) * std::f32::consts::TAU / (ring_segments as f32);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            // Point on the ring centerline.
            let ring_center = (ring_u * cos_t + ring_v * sin_t) * ring_radius;
            // Outward direction from the torus center.
            let outward = (ring_u * cos_t + ring_v * sin_t).normalize();

            for j in 0..tube_segments {
                let phi = (j as f32) * std::f32::consts::TAU / (tube_segments as f32);
                let cos_p = phi.cos();
                let sin_p = phi.sin();
                let normal = outward * cos_p + axis_dir * sin_p;
                let pos = ring_center + normal * tube_radius;

                vertices.push(Vertex {
                    position: pos.to_array(),
                    normal: normal.to_array(),
                    color,
                    uv: [0.0, 0.0],
                    tangent: [0.0, 0.0, 0.0, 1.0],
                });
            }
        }

        // Indices: connect adjacent ring segments.
        for i in 0..ring_segments {
            let next_i = (i + 1) % ring_segments;
            for j in 0..tube_segments {
                let next_j = (j + 1) % tube_segments;
                let v00 = base + i * tube_segments + j;
                let v01 = base + i * tube_segments + next_j;
                let v10 = base + next_i * tube_segments + j;
                let v11 = base + next_i * tube_segments + next_j;
                indices.extend_from_slice(&[v00, v10, v01, v01, v10, v11]);
            }
        }
    }
}

/// Compute two perpendicular unit vectors to the given axis.
fn perpendicular_pair(axis: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let hint = if axis.dot(glam::Vec3::Y).abs() > 0.9 {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    };
    let u = axis.cross(hint).normalize();
    let v = axis.cross(u).normalize();
    (u, v)
}

// ---------------------------------------------------------------------------
// Gizmo scale computation
// ---------------------------------------------------------------------------

/// Compute the world-space scale for the gizmo so it appears at a consistent
/// screen size regardless of camera distance.
///
/// `gizmo_center_world` — position of the gizmo (selected object)
/// `camera_eye` — camera eye position
/// `fov_y` — camera vertical field of view (radians)
/// `viewport_height` — viewport height in pixels
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
// GizmoUniform (mirrors the WGSL struct)
// ---------------------------------------------------------------------------

/// Uniform data for the gizmo shader: model matrix positioning the gizmo.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GizmoUniform {
    /// World-space model matrix (Translation * Scale; no rotation for world-space gizmo).
    pub(crate) model: [[f32; 4]; 4],
}

// ---------------------------------------------------------------------------
// Drag math helpers
// ---------------------------------------------------------------------------

/// Project mouse delta (in screen pixels) onto a world-space axis direction.
///
/// Returns the signed scalar amount to move along the axis.
///
/// # Arguments
/// * `drag_delta` — mouse movement in pixels since drag start (egui drag_delta())
/// * `axis_world` — world-space axis direction (X, Y, or Z unit vector)
/// * `view_proj` — camera view-projection matrix
/// * `gizmo_center` — world-space gizmo center (selected object position)
/// * `viewport_size` — viewport size in pixels
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

    // Scale: 1 full-screen drag = 2π radians (reasonable sensitivity).
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
    selection: &crate::interaction::selection::Selection,
    position_fn: impl Fn(crate::interaction::selection::NodeId) -> Option<glam::Vec3>,
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
        // Rotate the object 90° around Y: local X → world -Z, local Z → world +X.
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
            "local X axis should be along world -Z after 90° Y rotation"
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
    fn test_build_mesh_translate_has_plane_quads() {
        let (verts, idxs) =
            build_gizmo_mesh(GizmoMode::Translate, GizmoAxis::None, glam::Quat::IDENTITY);
        // Translate mode has arrows + plane quads + screen handle — substantially more geometry.
        assert!(
            verts.len() > 80,
            "translate mesh should have significant vertex count, got {}",
            verts.len()
        );
        assert!(!idxs.is_empty());
    }

    #[test]
    fn test_build_mesh_rotate_produces_rings() {
        let (verts, _) = build_gizmo_mesh(GizmoMode::Rotate, GizmoAxis::None, glam::Quat::IDENTITY);
        // 3 rings × 40 ring_segments × 8 tube_segments = 960 vertices.
        assert!(
            verts.len() >= 960,
            "rotate mesh should have ring vertices, got {}",
            verts.len()
        );
    }

    #[test]
    fn test_build_mesh_scale_has_cubes() {
        let (verts_translate, _) =
            build_gizmo_mesh(GizmoMode::Translate, GizmoAxis::None, glam::Quat::IDENTITY);
        let (verts_scale, _) =
            build_gizmo_mesh(GizmoMode::Scale, GizmoAxis::None, glam::Quat::IDENTITY);
        // Scale mode has cube tips (6 faces × 4 verts = 24 per axis = 72 for cube tips)
        // instead of cone tips (segments + 1 + segments per axis). Different counts expected.
        assert!(
            verts_scale.len() > 50,
            "scale mesh should have geometry, got {}",
            verts_scale.len()
        );
        assert_ne!(
            verts_translate.len(),
            verts_scale.len(),
            "translate and scale should have different vertex counts (cone vs cube tips)"
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
        let mut sel = crate::interaction::selection::Selection::new();
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
        let mut sel = crate::interaction::selection::Selection::new();
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
        let mut sel = crate::interaction::selection::Selection::new();
        sel.add(1);
        sel.add(2);
        let pos_fn = |id: crate::interaction::selection::NodeId| match id {
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
        let mut sel = crate::interaction::selection::Selection::new();
        sel.add(1);
        let result = gizmo_center_for_pivot(&PivotMode::WorldOrigin, &sel, |_| {
            Some(glam::Vec3::new(5.0, 0.0, 0.0))
        });
        assert_eq!(result, Some(glam::Vec3::ZERO));
    }

    #[test]
    fn test_pivot_world_origin_empty_selection_returns_none() {
        let sel = crate::interaction::selection::Selection::new();
        let result = gizmo_center_for_pivot(&PivotMode::WorldOrigin, &sel, |_| None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_pivot_individual_origins_uses_primary() {
        let mut sel = crate::interaction::selection::Selection::new();
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
        let mut sel = crate::interaction::selection::Selection::new();
        sel.add(1);
        sel.add(2);
        let pos_fn = |id: crate::interaction::selection::NodeId| match id {
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
        let mut sel = crate::interaction::selection::Selection::new();
        sel.add(1);
        let cursor = glam::Vec3::new(7.0, 2.0, 3.0);
        let result = gizmo_center_for_pivot(&PivotMode::Cursor3D(cursor), &sel, |_| {
            Some(glam::Vec3::ZERO)
        });
        assert_eq!(result, Some(cursor));
    }

    #[test]
    fn test_pivot_cursor3d_empty_selection_returns_none() {
        let sel = crate::interaction::selection::Selection::new();
        let cursor = glam::Vec3::new(1.0, 2.0, 3.0);
        let result = gizmo_center_for_pivot(&PivotMode::Cursor3D(cursor), &sel, |_| None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_gizmo_pivot_mode_field_defaults_to_selection_centroid() {
        let g = Gizmo::new();
        assert!(matches!(g.pivot_mode, PivotMode::SelectionCentroid));
    }
}
