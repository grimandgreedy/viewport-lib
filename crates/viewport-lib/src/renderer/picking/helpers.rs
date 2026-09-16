//! Free helper functions shared by the CPU pick and rect-pick paths.
//!
//! Ray/segment intersection and strip-index mapping.

/// Warn once if a CPU pick runs while the pick cache is disabled, so the call does
/// not silently return nothing.
pub(super) fn warn_pick_cache_disabled() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "renderer.pick()/pick_rect() was called but the CPU pick cache is disabled; \
             enable it with ViewportRenderer::set_cpu_pick_cache(true)"
        );
    });
}

/// Project a world-space point to pixel coordinates (origin top-left, y
/// down), or `None` when the point is behind the camera (`clip.w <= 0`).
///
/// This is the projection every CPU pick proximity test uses; plugin pick
/// implementations use it so their screen-space tolerances agree with the
/// built-in item types'.
pub fn project_to_screen(
    world: glam::Vec3,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> Option<glam::Vec2> {
    let clip = view_proj * world.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    Some(glam::Vec2::new(
        (clip.x / clip.w + 1.0) * 0.5 * viewport_size.x,
        (1.0 - clip.y / clip.w) * 0.5 * viewport_size.y,
    ))
}

/// Pixel radius of a world-space radius `world_r` measured at `world_centre`.
///
/// Item types whose instances are drawn at a world size but picked by
/// screen-space proximity (glyphs, tensor glyphs, world-space sprites) need
/// their pick tolerance in pixels. Measuring at the instance centroid rather
/// than at the model origin keeps the estimate right when the instances sit
/// far from the origin. The result is floored at 4 pixels so a distant set is
/// still clickable, and falls back to a scaled world radius when the centre
/// projects to or behind the eye.
pub fn world_radius_in_pixels(
    world_centre: glam::Vec3,
    world_r: f32,
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
) -> f32 {
    let p0 = view_proj * world_centre.extend(1.0);
    let p1 = view_proj * (world_centre + glam::Vec3::X * world_r).extend(1.0);
    if p0.w.abs() > 1e-6 && p1.w.abs() > 1e-6 {
        let n0 = glam::Vec2::new(p0.x, p0.y) / p0.w;
        let n1 = glam::Vec2::new(p1.x, p1.y) / p1.w;
        ((n1 - n0).length() * 0.5 * viewport_size.x.max(viewport_size.y)).max(4.0)
    } else {
        (world_r * 100.0_f32).max(4.0)
    }
}

/// Ray versus the local unit box `[-0.5, 0.5]^3`, used for decal projection
/// volumes. `origin` and `dir` are the ray in the box's local space (the world
/// ray transformed by the inverse of the box's model matrix).
///
/// Returns the entry parameter `t` along `dir`, or `None` if the ray misses.
/// Because an affine transform maps the world ray parameter to the same
/// parameter on the local ray, the returned `t` is directly comparable to the
/// world-space `time_of_impact` used by the other pick sections. If the origin
/// is inside the box, returns `0.0`.
pub fn ray_unit_box_toi(origin: glam::Vec3, dir: glam::Vec3) -> Option<f32> {
    const HALF: f32 = 0.5;
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    for i in 0..3 {
        let o = origin[i];
        let d = dir[i];
        if d.abs() < 1e-9 {
            // Ray is parallel to this slab: a miss unless the origin lies within it.
            if o < -HALF || o > HALF {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut t0 = (-HALF - o) * inv;
            let mut t1 = (HALF - o) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
            if t_enter > t_exit {
                return None;
            }
        }
    }
    if t_exit < 0.0 {
        // The box is entirely behind the ray origin.
        return None;
    }
    Some(t_enter.max(0.0))
}

// ---------------------------------------------------------------------------
// Strip index helpers (shared by polyline, tube, ribbon picking)
// ---------------------------------------------------------------------------

/// Map a global node index to its strip index by walking `strip_lengths`.
pub(crate) fn strip_for_node(node_idx: u32, strip_lengths: &[u32]) -> u32 {
    let mut offset = 0u32;
    for (i, &len) in strip_lengths.iter().enumerate() {
        offset += len;
        if node_idx < offset {
            return i as u32;
        }
    }
    strip_lengths.len().saturating_sub(1) as u32
}

/// Find the closest polyline segment to `click_pos` within `threshold_px` pixels.
///
/// Returns `(global_seg_idx, world_hit_pos)` on hit, `None` otherwise. Positions
/// are treated as world-space (polylines are always submitted without a model
/// transform). The hit position is the closest point on the segment in 3D,
/// interpolated at the same screen-space parameter `t` as the closest screen point.
pub fn pick_closest_polyline_segment(
    click_pos: glam::Vec2,
    viewport_size: glam::Vec2,
    view_proj: glam::Mat4,
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    threshold_px: f32,
) -> Option<(u32, glam::Vec3)> {
    let project = |p: [f32; 3]| -> Option<glam::Vec2> {
        project_to_screen(p.into(), view_proj, viewport_size)
    };

    let mut best_dist = threshold_px;
    let mut best: Option<(u32, glam::Vec3)> = None;

    macro_rules! try_seg {
        ($ai:expr, $bi:expr, $seg:expr) => {{
            if let (Some(sa), Some(sb)) = (project(positions[$ai]), project(positions[$bi])) {
                let ab = sb - sa;
                let len_sq = ab.length_squared();
                let t = if len_sq < 1e-6 {
                    0.0f32
                } else {
                    ((click_pos - sa).dot(ab) / len_sq).clamp(0.0, 1.0)
                };
                let dist = (click_pos - (sa + ab * t)).length();
                if dist < best_dist {
                    best_dist = dist;
                    let wa = glam::Vec3::from(positions[$ai]);
                    let wb = glam::Vec3::from(positions[$bi]);
                    best = Some(($seg as u32, wa.lerp(wb, t)));
                }
            }
        }};
    }

    if strip_lengths.is_empty() {
        for j in 0..positions.len().saturating_sub(1) {
            try_seg!(j, j + 1, j);
        }
    } else {
        let mut node_off = 0usize;
        let mut seg_off = 0u32;
        for &slen in strip_lengths {
            let slen = slen as usize;
            for j in 0..slen.saturating_sub(1) {
                try_seg!(node_off + j, node_off + j + 1, seg_off + j as u32);
            }
            seg_off += slen.saturating_sub(1) as u32;
            node_off += slen;
        }
    }

    best
}

/// Returns `true` if the 2D segment [a, b] touches or crosses the axis-aligned rect.
pub fn segment_in_rect(
    a: glam::Vec2,
    b: glam::Vec2,
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
) -> bool {
    // Quick AABB reject.
    if a.x.min(b.x) > rect_max.x
        || a.x.max(b.x) < rect_min.x
        || a.y.min(b.y) > rect_max.y
        || a.y.max(b.y) < rect_min.y
    {
        return false;
    }
    // Either endpoint inside?
    let in_r = |p: glam::Vec2| {
        p.x >= rect_min.x && p.x <= rect_max.x && p.y >= rect_min.y && p.y <= rect_max.y
    };
    if in_r(a) || in_r(b) {
        return true;
    }
    // Segment crosses one of the 4 edges (parametric intersection test).
    let crosses = |p0: glam::Vec2, p1: glam::Vec2, q0: glam::Vec2, q1: glam::Vec2| -> bool {
        let d = p1 - p0;
        let e = q1 - q0;
        let denom = d.x * e.y - d.y * e.x;
        if denom.abs() < 1e-10 {
            return false;
        }
        let diff = q0 - p0;
        let t = (diff.x * e.y - diff.y * e.x) / denom;
        let u = (diff.x * d.y - diff.y * d.x) / denom;
        t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0
    };
    let tl = rect_min;
    let tr = glam::Vec2::new(rect_max.x, rect_min.y);
    let bl = glam::Vec2::new(rect_min.x, rect_max.y);
    let br = rect_max;
    crosses(a, b, tl, tr) || crosses(a, b, tr, br) || crosses(a, b, br, bl) || crosses(a, b, bl, tl)
}

/// Map a global segment index to its strip index by walking `strip_lengths`.
pub(crate) fn strip_for_segment(seg_idx: u32, strip_lengths: &[u32]) -> u32 {
    let mut offset = 0u32;
    for (i, &len) in strip_lengths.iter().enumerate() {
        let segs = len.saturating_sub(1);
        offset += segs;
        if seg_idx < offset {
            return i as u32;
        }
    }
    strip_lengths.len().saturating_sub(1) as u32
}

/// Moller-Trumbore ray-triangle intersection.
///
/// Returns the ray parameter `t > 0` on hit, or `None` on miss or backface cull.
/// Call twice with reversed winding to test both faces.
#[inline]
pub fn ray_triangle(
    ray_orig: glam::Vec3,
    ray_dir: glam::Vec3,
    v0: glam::Vec3,
    v1: glam::Vec3,
    v2: glam::Vec3,
) -> Option<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = ray_dir.cross(e2);
    let a = e1.dot(h);
    if a.abs() < 1e-10 {
        return None;
    }
    let f = 1.0 / a;
    let s = ray_orig - v0;
    let u = f * s.dot(h);
    if u < 0.0 || u > 1.0 {
        return None;
    }
    let q = s.cross(e1);
    let v = f * ray_dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * e2.dot(q);
    if t > 0.0 { Some(t) } else { None }
}
