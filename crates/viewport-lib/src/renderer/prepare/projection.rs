//! World-to-screen projection and clip helpers used when placing overlays.

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

/// The screen-pixel position stored in an overlay vertex.
///
/// Overlay geometry is stored in local logical-pixel space; the overlay shaders
/// apply the pixel-to-NDC transform from a per-frame viewport-size uniform, so a
/// compiled or immediate overlay is independent of the viewport size (a resize
/// updates the uniform, not the vertices). The viewport-size arguments are kept
/// in the signature so call sites do not have to change while the transform lives
/// in the shader; they are unused here.
#[inline]
pub(super) fn overlay_local_px(px_x: f32, px_y: f32, _vp_w: f32, _vp_h: f32) -> [f32; 2] {
    [px_x, px_y]
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
