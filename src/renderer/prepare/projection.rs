//! World-to-screen projection and clip helpers used when placing overlays.

/// Project a world-space position to NDC.
/// Returns `None` only if the point is behind the camera (`clip.w <= 0`).
/// Does NOT reject points outside the [-1,1] viewport box.
pub(super) fn project_to_ndc(
    pos: [f32; 3],
    view: &glam::Mat4,
    proj: &glam::Mat4,
) -> Option<[f32; 2]> {
    let clip = *proj * *view * glam::Vec3::from(pos).extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    Some([clip.x / clip.w, clip.y / clip.w])
}

/// Convert NDC coordinates to screen pixels (top-left origin).
pub(super) fn ndc_to_screen_px(ndc: [f32; 2], vp_w: f32, vp_h: f32) -> [f32; 2] {
    [
        (ndc[0] * 0.5 + 0.5) * vp_w,
        (1.0 - (ndc[1] * 0.5 + 0.5)) * vp_h,
    ]
}

/// Returns true when the NDC point lies within the viewport square.
pub(super) fn ndc_in_viewport(ndc: [f32; 2]) -> bool {
    ndc[0] >= -1.0 && ndc[0] <= 1.0 && ndc[1] >= -1.0 && ndc[1] <= 1.0
}

/// Clip a line segment [a, b] in NDC to the [-1,1]^2 viewport box
/// using the Liang-Barsky algorithm.
/// Returns the clipped endpoints, or `None` if the segment is entirely outside.
pub(super) fn clip_line_ndc(a: [f32; 2], b: [f32; 2]) -> Option<([f32; 2], [f32; 2])> {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;

    // (p, q) pairs for left, right, bottom, top boundaries.
    for (p, q) in [
        (-dx, a[0] + 1.0),
        (dx, 1.0 - a[0]),
        (-dy, a[1] + 1.0),
        (dy, 1.0 - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }

    if t0 > t1 {
        return None;
    }
    Some((
        [a[0] + t0 * dx, a[1] + t0 * dy],
        [a[0] + t1 * dx, a[1] + t1 * dy],
    ))
}

/// Project a world-space position to screen pixels (top-left origin).
/// Returns `None` if behind the camera or outside the frustum.
pub(super) fn project_to_screen(
    pos: [f32; 3],
    view: &glam::Mat4,
    proj: &glam::Mat4,
    vp_w: f32,
    vp_h: f32,
) -> Option<[f32; 2]> {
    let p = glam::Vec3::from(pos);
    let clip = *proj * *view * p.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    if ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0 {
        return None;
    }
    let x = (ndc_x * 0.5 + 0.5) * vp_w;
    let y = (1.0 - (ndc_y * 0.5 + 0.5)) * vp_h;
    Some([x, y])
}

/// Convert screen pixel coordinates to NDC.
#[inline]
pub(super) fn px_to_ndc(px_x: f32, px_y: f32, vp_w: f32, vp_h: f32) -> [f32; 2] {
    [px_x / vp_w * 2.0 - 1.0, 1.0 - px_y / vp_h * 2.0]
}

pub(super) fn polyline_bounds(points: &[[f32; 2]]) -> Option<([f32; 2], [f32; 2])> {
    let first = *points.first()?;
    let mut min = first;
    let mut max = first;
    for p in points.iter().skip(1) {
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
    }
    Some((min, max))
}
