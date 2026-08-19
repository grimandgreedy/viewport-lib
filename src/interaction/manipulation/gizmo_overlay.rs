//! Draw the transform gizmo as projected 2D overlay primitives.
//!
//! [`build_gizmo_overlays`] takes the gizmo centre, screen scale, mode, hovered
//! axis, and camera, and emits overlay shapes and polylines that reproduce the
//! translate arrows, rotate rings, and scale cubes. The renderer calls this each
//! frame from `frame.interaction` and merges the result into the viewport's
//! overlay batches, so the gizmo needs no dedicated GPU pipeline.
//!
//! The proportions and colours come from [`super::gizmo`] (the same constants
//! [`Gizmo::hit_test`](super::gizmo::Gizmo::hit_test) uses), so what is drawn
//! matches what is picked. Sizing tracks camera distance through the caller's
//! `scale` (from [`super::gizmo::compute_gizmo_scale`]); arrow heads and rings
//! foreshorten because they are projected from real world-space geometry.

use glam::{Mat4, Vec2, Vec3};

use super::gizmo::{
    self, CONE_LENGTH, CONE_RADIUS, CUBE_HALF, GizmoAxis, GizmoMode, PLANE_OFFSET, PLANE_SIZE,
    RING_TUBE_RADIUS, ROTATION_RING_RADIUS, SCREEN_HANDLE_SIZE, SHAFT_LENGTH, SHAFT_RADIUS,
};
use crate::{OverlayFill, OverlayPolylineItem, OverlayShape, OverlayShapeItem, TriangleDirection};

/// Collects the overlay primitives produced for one gizmo.
struct Out<'a> {
    shapes: &'a mut Vec<OverlayShapeItem>,
    polylines: &'a mut Vec<OverlayPolylineItem>,
}

/// Build the overlay items for one transform gizmo.
///
/// * `mode` : which handle set to draw (translate arrows, rotate rings, scale cubes).
/// * `view_proj` : the viewport's combined view-projection matrix.
/// * `forward` : camera forward direction in world space (for arrow-head foreshortening).
/// * `viewport` : viewport size in logical pixels `[w, h]`.
/// * `centre_world` : gizmo centre (the selected object's pivot) in world space.
/// * `scale` : world-space arm length from [`gizmo::compute_gizmo_scale`].
/// * `hovered` : the axis under the cursor, brightened in the output.
/// * `space_orientation` : identity for world space, the object orientation for local space.
///
/// Shapes go into `shapes` (arrow shafts and cone heads, gradient-filled), and
/// polylines into `polylines` (rings, plane quads, cube faces, centre handle).
#[allow(clippy::too_many_arguments)]
pub fn build_gizmo_overlays(
    mode: GizmoMode,
    view_proj: Mat4,
    forward: Vec3,
    viewport: [f32; 2],
    centre_world: Vec3,
    scale: f32,
    hovered: GizmoAxis,
    space_orientation: glam::Quat,
    shapes: &mut Vec<OverlayShapeItem>,
    polylines: &mut Vec<OverlayPolylineItem>,
) {
    if scale <= 0.0 || viewport[0] <= 0.0 || viewport[1] <= 0.0 {
        return;
    }
    let Some((origin_px, _)) = project(view_proj, centre_world, viewport) else {
        return;
    };
    let mut out = Out { shapes, polylines };

    let axes = [
        (GizmoAxis::X, space_orientation * Vec3::X),
        (GizmoAxis::Y, space_orientation * Vec3::Y),
        (GizmoAxis::Z, space_orientation * Vec3::Z),
    ];

    match mode {
        GizmoMode::Rotate => {
            for (axis, dir) in axes {
                build_ring(
                    &mut out,
                    view_proj,
                    viewport,
                    centre_world,
                    dir,
                    scale,
                    gizmo::axis_colour(axis, hovered),
                );
            }
        }
        GizmoMode::Translate | GizmoMode::Scale => {
            let cube = mode == GizmoMode::Scale;
            for (axis, dir) in axes {
                build_arrow(
                    &mut out,
                    view_proj,
                    forward,
                    viewport,
                    centre_world,
                    origin_px,
                    dir,
                    scale,
                    gizmo::axis_colour(axis, hovered),
                    cube,
                );
            }
            // Plane handles (XY, XZ, YZ) are Translate-only, matching the mesh gizmo.
            if !cube {
                let planes = [
                    (GizmoAxis::XY, Vec3::X, Vec3::Y),
                    (GizmoAxis::XZ, Vec3::X, Vec3::Z),
                    (GizmoAxis::YZ, Vec3::Y, Vec3::Z),
                ];
                for (axis, a, b) in planes {
                    let a = space_orientation * a;
                    let b = space_orientation * b;
                    let c = centre_world + (a + b) * (PLANE_OFFSET * scale);
                    let s = PLANE_SIZE * scale;
                    let corners = [c - a * s - b * s, c + a * s - b * s, c + a * s + b * s, c - a * s + b * s];
                    let col = gizmo::plane_colour(axis, hovered);
                    let stroke = [col[0], col[1], col[2], 1.0];
                    push_quad(&mut out, view_proj, viewport, corners, col, stroke, 1.0, 15_000);
                }
            }
            // Flat X-Y centre handle: drawn for both Translate and Scale.
            let s = SCREEN_HANDLE_SIZE * scale;
            let corners = [
                centre_world + Vec3::new(-s, -s, 0.0),
                centre_world + Vec3::new(s, -s, 0.0),
                centre_world + Vec3::new(s, s, 0.0),
                centre_world + Vec3::new(-s, s, 0.0),
            ];
            let col = gizmo::axis_colour(GizmoAxis::Screen, hovered);
            push_quad(&mut out, view_proj, viewport, corners, col, [1.0, 1.0, 1.0, 1.0], 1.0, 20_000);
        }
    }
}

// ---------------------------------------------------------------------------
// Projection helpers
// ---------------------------------------------------------------------------

/// Project a world point to logical-pixel screen space (top-left origin).
///
/// Returns the pixel position and the NDC depth, or `None` when the point is at
/// or behind the eye.
fn project(view_proj: Mat4, world: Vec3, size: [f32; 2]) -> Option<(Vec2, f32)> {
    let clip = view_proj * world.extend(1.0);
    if clip.w <= 1e-6 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    let px = (ndc.x * 0.5 + 0.5) * size[0];
    let py = (1.0 - (ndc.y * 0.5 + 0.5)) * size[1];
    Some((glam::vec2(px, py), ndc.z))
}

/// A unit vector perpendicular to `dir` that is as broad on screen as possible.
fn perp_of(dir: Vec3, forward: Vec3) -> Vec3 {
    let p = dir.cross(forward);
    if p.length_squared() < 1e-8 {
        dir.cross(Vec3::Z).normalize_or(Vec3::X)
    } else {
        p.normalize()
    }
}

/// An orthonormal (tangent, bitangent) pair spanning the plane perpendicular to `dir`.
fn basis(dir: Vec3) -> (Vec3, Vec3) {
    let t = if dir.z.abs() < 0.9 {
        dir.cross(Vec3::Z).normalize()
    } else {
        dir.cross(Vec3::X).normalize()
    };
    (t, dir.cross(t).normalize())
}

/// Screen pixels spanned by a world-space length `r` at `centre`, offset along
/// `dir_unit`. Used to size radii (shaft, cone, ring tube) authored in world units.
fn screen_len(vp: Mat4, size: [f32; 2], centre: Vec3, dir_unit: Vec3, r: f32) -> f32 {
    match (project(vp, centre, size), project(vp, centre + dir_unit * r, size)) {
        (Some((a, _)), Some((b, _))) => (b - a).length(),
        _ => 3.0,
    }
}

/// `OverlayShapeItem::position` is the top-left of the bounding box; build one
/// centred on `centre` so callers can think in centres.
fn centred(shape: OverlayShape, centre: Vec2, size: [f32; 2]) -> OverlayShapeItem {
    let tl = [centre.x - size[0] * 0.5, centre.y - size[1] * 0.5];
    OverlayShapeItem::new(shape, tl, size)
}

fn scale_rgb(c: [f32; 4], k: f32) -> [f32; 4] {
    [
        (c[0] * k).clamp(0.0, 1.0),
        (c[1] * k).clamp(0.0, 1.0),
        (c[2] * k).clamp(0.0, 1.0),
        c[3],
    ]
}

/// Project four world corners into a closed, filled polyline (a general quad).
fn push_quad(
    out: &mut Out,
    vp: Mat4,
    size: [f32; 2],
    corners_w: [Vec3; 4],
    fill: [f32; 4],
    stroke: [f32; 4],
    thickness: f32,
    z: i32,
) {
    let mut pts = Vec::with_capacity(4);
    for c in corners_w {
        match project(vp, c, size) {
            Some((p, _)) => pts.push(p.to_array()),
            None => return,
        }
    }
    out.polylines.push(
        OverlayPolylineItem::new(pts)
            .with_closed(true)
            .with_fill(OverlayFill::Solid(fill))
            .with_colour(stroke)
            .with_thickness(thickness)
            .with_z_order(z),
    );
}

// ---------------------------------------------------------------------------
// Gizmo parts
// ---------------------------------------------------------------------------

/// One axis arrow: shaft capsule plus a cone head (Translate) or cube (Scale).
#[allow(clippy::too_many_arguments)]
fn build_arrow(
    out: &mut Out,
    vp: Mat4,
    forward: Vec3,
    size: [f32; 2],
    origin_w: Vec3,
    origin_px: Vec2,
    dir: Vec3,
    ws: f32,
    colour: [f32; 4],
    cube_tip: bool,
) {
    let shaft_tip_w = origin_w + dir * (ws * SHAFT_LENGTH);
    let tip_end_w = origin_w + dir * (ws * (SHAFT_LENGTH + CONE_LENGTH));
    let (Some((shaft_tip_px, _)), Some((tip_end_px, tip_z))) =
        (project(vp, shaft_tip_w, size), project(vp, tip_end_w, size))
    else {
        return;
    };

    let perp = perp_of(dir, forward);
    let shaft_thick = (2.0 * screen_len(vp, size, origin_w, perp, ws * SHAFT_RADIUS)).max(2.0);
    let head_w = (2.0 * screen_len(vp, size, shaft_tip_w, perp, ws * CONE_RADIUS)).max(4.0);
    let z = ((1.0 - tip_z) * 10_000.0) as i32;

    let head_vec = tip_end_px - shaft_tip_px;
    let head_len = head_vec.length();

    // Shaft, tucked slightly under the head so there's no gap at the join.
    let shaft_end = shaft_tip_px + head_vec * 0.35;
    let shaft_draw = shaft_end - origin_px;
    if shaft_draw.length() > 1.0 {
        let mid = (origin_px + shaft_end) * 0.5;
        out.shapes.push(
            centred(OverlayShape::Capsule, mid, [shaft_draw.length(), shaft_thick])
                .with_rotation(shaft_draw.y.atan2(shaft_draw.x))
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: scale_rgb(colour, 1.2),
                    end_colour: scale_rgb(colour, 0.65),
                    angle: 0.0,
                })
                .with_z_order(z),
        );
    }

    if cube_tip {
        build_cube_tip(out, vp, forward, size, shaft_tip_w, dir, ws, colour, z + 1);
    } else {
        // Cone = base disc + sides. The base is a circle in the plane
        // perpendicular to the axis, so it projects to an ellipse: a thin sliver
        // when the axis is side-on, a full circle when it points at the camera.
        // The triangle draws the cone's sides on top. Together they read as a
        // cone and morph continuously as the axis rotates toward the eye.
        let head_on = dir.dot(forward).abs();
        let along = head_vec.y.atan2(head_vec.x); // screen angle of the axis
        let minor = (head_w * head_on).max(head_w * 0.12); // base extent along the axis
        out.shapes.push(
            // Local +X (minor) runs along the axis; local +Y (major) across it.
            centred(OverlayShape::Ellipse, shaft_tip_px, [minor, head_w])
                .with_rotation(along)
                .with_fill(OverlayFill::RadialGradient {
                    centre_colour: scale_rgb(colour, 1.2),
                    edge_colour: scale_rgb(colour, 0.55),
                })
                .with_z_order(z + 1),
        );
        if head_len > 1.0 {
            let mid = (shaft_tip_px + tip_end_px) * 0.5;
            out.shapes.push(
                centred(
                    OverlayShape::Triangle {
                        direction: TriangleDirection::Up,
                    },
                    mid,
                    [head_w, head_len],
                )
                .with_rotation(head_vec.x.atan2(-head_vec.y))
                .with_fill(OverlayFill::LinearGradient {
                    start_colour: scale_rgb(colour, 1.3),
                    end_colour: scale_rgb(colour, 0.6),
                    angle: 0.0,
                })
                .with_z_order(z + 2),
            );
        }
    }
}

/// Cube handle (Scale): the three camera-facing faces as shaded quads.
#[allow(clippy::too_many_arguments)]
fn build_cube_tip(
    out: &mut Out,
    vp: Mat4,
    forward: Vec3,
    size: [f32; 2],
    tip_base_w: Vec3,
    dir: Vec3,
    ws: f32,
    colour: [f32; 4],
    z: i32,
) {
    let h = CUBE_HALF * ws;
    let centre = tip_base_w + dir * h;
    let (t, b) = basis(dir);
    // Each of the three axes contributes the face whose outward normal faces the
    // camera. Draw brightest-facing last so it sits on top.
    let mut faces: Vec<([Vec3; 4], f32)> = Vec::new();
    for (n_axis, u, v) in [(dir, t, b), (t, dir, b), (b, dir, t)] {
        let sign = if n_axis.dot(-forward) >= 0.0 { 1.0 } else { -1.0 };
        let normal = n_axis * sign;
        let fc = centre + normal * h;
        let corners = [
            fc + u * h + v * h,
            fc + u * h - v * h,
            fc - u * h - v * h,
            fc - u * h + v * h,
        ];
        faces.push((corners, normal.dot(-forward).max(0.0)));
    }
    faces.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (corners, facing) in faces {
        let bright = 0.5 + 0.5 * facing;
        push_quad(out, vp, size, corners, scale_rgb(colour, bright), scale_rgb(colour, 0.4), 1.0, z);
    }
}

/// One rotation ring: the world circle (radius `ROTATION_RING_RADIUS`) in the
/// plane perpendicular to `dir`, sampled and stroked as a closed polyline.
fn build_ring(
    out: &mut Out,
    vp: Mat4,
    size: [f32; 2],
    origin_w: Vec3,
    dir: Vec3,
    ws: f32,
    colour: [f32; 4],
) {
    let (u, v) = basis(dir);
    let r = ROTATION_RING_RADIUS * ws;
    let thick = (2.0 * screen_len(vp, size, origin_w, u, ws * RING_TUBE_RADIUS)).max(2.0);
    const SEGMENTS: usize = 48;
    let mut pts = Vec::with_capacity(SEGMENTS);
    let mut depth = 0.0f32;
    for i in 0..SEGMENTS {
        let a = (i as f32) * std::f32::consts::TAU / (SEGMENTS as f32);
        let p_w = origin_w + (u * a.cos() + v * a.sin()) * r;
        match project(vp, p_w, size) {
            Some((p, d)) => {
                pts.push(p.to_array());
                depth += d;
            }
            None => return,
        }
    }
    let z = ((1.0 - depth / SEGMENTS as f32) * 10_000.0) as i32;
    out.polylines.push(
        OverlayPolylineItem::new(pts)
            .with_closed(true)
            .with_colour(colour)
            .with_thickness(thick)
            .with_z_order(z),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple look-at camera pointing down -Y at the origin, with the world
    /// Z axis up. Returns (view_proj, forward, viewport).
    fn camera() -> (Mat4, Vec3, [f32; 2]) {
        let eye = Vec3::new(0.0, -8.0, 4.0);
        let target = Vec3::ZERO;
        let up = Vec3::Z;
        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1200.0 / 720.0, 0.1, 100.0);
        (proj * view, (target - eye).normalize(), [1200.0, 720.0])
    }

    #[test]
    fn translate_emits_arrows_planes_and_centre() {
        let (vp, fwd, size) = camera();
        let mut shapes = Vec::new();
        let mut polys = Vec::new();
        build_gizmo_overlays(
            GizmoMode::Translate,
            vp,
            fwd,
            size,
            Vec3::ZERO,
            1.0,
            GizmoAxis::None,
            glam::Quat::IDENTITY,
            &mut shapes,
            &mut polys,
        );
        // Three arrows contribute shaft + head shapes.
        assert!(shapes.len() >= 3, "expected arrow shapes, got {}", shapes.len());
        // Three plane quads + one centre handle are closed, filled polylines.
        assert_eq!(polys.len(), 4, "expected 3 plane handles + 1 centre quad");
        assert!(polys.iter().all(|p| p.closed && p.fill.is_some()));
    }

    #[test]
    fn scale_has_centre_but_no_plane_handles() {
        let (vp, fwd, size) = camera();
        let mut shapes = Vec::new();
        let mut polys = Vec::new();
        build_gizmo_overlays(
            GizmoMode::Scale,
            vp,
            fwd,
            size,
            Vec3::ZERO,
            1.0,
            GizmoAxis::None,
            glam::Quat::IDENTITY,
            &mut shapes,
            &mut polys,
        );
        // Cube tips (3 faces each) + the centre handle are polylines; no plane handles.
        assert!(!shapes.is_empty(), "scale arrows still emit shaft shapes");
        assert!(polys.len() >= 3 * 3 + 1, "cube faces + centre handle, got {}", polys.len());
    }

    #[test]
    fn rotate_emits_three_rings() {
        let (vp, fwd, size) = camera();
        let mut shapes = Vec::new();
        let mut polys = Vec::new();
        build_gizmo_overlays(
            GizmoMode::Rotate,
            vp,
            fwd,
            size,
            Vec3::ZERO,
            1.0,
            GizmoAxis::None,
            glam::Quat::IDENTITY,
            &mut shapes,
            &mut polys,
        );
        assert!(shapes.is_empty(), "rings are polylines, not shapes");
        assert_eq!(polys.len(), 3, "one ring per axis");
        assert!(polys.iter().all(|p| p.closed && p.points.len() == 48));
    }

    #[test]
    fn hover_brightens_the_hovered_axis() {
        let (vp, fwd, size) = camera();
        let axis_shaft_colour = |hovered| {
            let mut shapes = Vec::new();
            let mut polys = Vec::new();
            build_gizmo_overlays(
                GizmoMode::Translate,
                vp,
                fwd,
                size,
                Vec3::ZERO,
                1.0,
                hovered,
                glam::Quat::IDENTITY,
                &mut shapes,
                &mut polys,
            );
            // First shape is the X-axis shaft; its gradient start encodes the colour.
            match &shapes[0].fill {
                OverlayFill::LinearGradient { start_colour, .. } => *start_colour,
                _ => panic!("shaft should use a linear gradient"),
            }
        };
        let base = axis_shaft_colour(GizmoAxis::None);
        let hovered = axis_shaft_colour(GizmoAxis::X);
        assert_ne!(base, hovered, "hovering X must change the X shaft colour");
    }

    #[test]
    fn behind_camera_centre_emits_nothing() {
        let (vp, fwd, size) = camera();
        let mut shapes = Vec::new();
        let mut polys = Vec::new();
        // Far behind the eye (eye is at y=-8 looking toward +y).
        build_gizmo_overlays(
            GizmoMode::Translate,
            vp,
            fwd,
            size,
            Vec3::new(0.0, -40.0, 0.0),
            1.0,
            GizmoAxis::None,
            glam::Quat::IDENTITY,
            &mut shapes,
            &mut polys,
        );
        assert!(shapes.is_empty() && polys.is_empty());
    }
}
