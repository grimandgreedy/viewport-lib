//! Free helper functions shared by the CPU pick and rect-pick paths.
//!
//! Ray/segment intersection, strip-index mapping, and CPU evaluators for
//! implicit-surface and marching-cubes picking.

use super::*;

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

// ---------------------------------------------------------------------------
// Strip index helpers (shared by polyline, tube, ribbon picking)
// ---------------------------------------------------------------------------

/// Map a global node index to its strip index by walking `strip_lengths`.
pub(super) fn strip_for_node(node_idx: u32, strip_lengths: &[u32]) -> u32 {
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
pub(super) fn pick_closest_polyline_segment(
    click_pos: glam::Vec2,
    viewport_size: glam::Vec2,
    view_proj: glam::Mat4,
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    threshold_px: f32,
) -> Option<(u32, glam::Vec3)> {
    let project = |p: [f32; 3]| -> Option<glam::Vec2> {
        let clip = view_proj * glam::Vec4::new(p[0], p[1], p[2], 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        Some(glam::Vec2::new(
            (clip.x / clip.w + 1.0) * 0.5 * viewport_size.x,
            (1.0 - clip.y / clip.w) * 0.5 * viewport_size.y,
        ))
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
pub(super) fn segment_in_rect(
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
pub(super) fn strip_for_segment(seg_idx: u32, strip_lengths: &[u32]) -> u32 {
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
pub(super) fn ray_triangle(
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

/// Reconstruct per-vertex (lateral direction, half-width) for a ribbon item.
///
/// Replicates the parallel-transport frame built by `upload_ribbon()` in
/// `prepare.rs` so click and rect picking can test the actual swept quad
/// rather than a midpoint proxy.
pub(super) fn ribbon_lateral_frames(
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    width: f32,
    width_attribute: Option<&[f32]>,
    twist_attribute: Option<&[[f32; 3]]>,
) -> Vec<(glam::Vec3, f32)> {
    let n = positions.len();
    // Initialise with a sentinel so any unvisited vertex has zero width.
    let mut frames: Vec<(glam::Vec3, f32)> = vec![(glam::Vec3::X, 0.0); n];

    let single;
    let strips: &[u32] = if strip_lengths.is_empty() {
        single = [positions.len() as u32];
        &single
    } else {
        strip_lengths
    };

    let mut node_off = 0usize;
    for &slen in strips {
        let slen = slen as usize;
        if slen < 2 {
            node_off += slen;
            continue;
        }

        let pts: Vec<glam::Vec3> = positions[node_off..node_off + slen]
            .iter()
            .map(|&p| glam::Vec3::from(p))
            .collect();

        let t0 = (pts[1] - pts[0]).normalize_or_zero();
        if t0.length_squared() < 1e-10 {
            node_off += slen;
            continue;
        }
        let ref_v = if t0.x.abs() < 0.9 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let mut u = t0.cross(ref_v).normalize();

        for k in 0..slen {
            let tangent = if k + 1 < slen {
                (pts[k + 1] - pts[k]).normalize_or_zero()
            } else {
                (pts[k] - pts[k - 1]).normalize_or_zero()
            };

            // Parallel transport: rotate u to stay perpendicular to the new tangent.
            if k > 0 {
                let t_prev = (pts[k] - pts[k - 1]).normalize_or_zero();
                let axis = t_prev.cross(tangent);
                let sin_a = axis.length().min(1.0);
                if sin_a > 1e-6 {
                    let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                    let ax = axis / sin_a;
                    u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                    u = u.normalize_or_zero();
                }
            }

            // Apply per-point twist if supplied.
            let mut lateral = u;
            if let Some(twist) = twist_attribute {
                if let Some(&tv) = twist.get(node_off + k) {
                    let tv = glam::Vec3::from(tv);
                    let proj = tv - tangent * tangent.dot(tv);
                    if proj.length_squared() > 1e-10 {
                        lateral = proj.normalize();
                    }
                }
            }

            let half_w = width_attribute
                .and_then(|wa| wa.get(node_off + k).copied())
                .unwrap_or(width)
                * 0.5;

            frames[node_off + k] = (lateral, half_w);
        }

        node_off += slen;
    }

    frames
}

// ---------------------------------------------------------------------------
// CPU SDF evaluation for GPU implicit surfaces (mirrors implicit.wgsl)
// ---------------------------------------------------------------------------

/// Evaluate one implicit primitive's signed distance from `p`.
pub(super) fn eval_implicit_primitive(
    p: glam::Vec3,
    prim: &crate::resources::ImplicitPrimitive,
) -> f32 {
    match prim.kind {
        1 => {
            // Sphere: center=params[0..3], radius=params[3]
            let center = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
            (p - center).length() - prim.params[3]
        }
        2 => {
            // Box: center=params[0..3], half-extents=params[4..7]
            let center = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
            let half = glam::Vec3::new(prim.params[4], prim.params[5], prim.params[6]);
            let q = (p - center).abs() - half;
            q.max(glam::Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0)
        }
        3 => {
            // Plane: normal=params[0..3], offset=params[3]
            let n =
                glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]).normalize_or_zero();
            p.dot(n) + prim.params[3]
        }
        4 => {
            // Capsule: a=params[0..3], radius=params[3], b=params[4..7]
            let a = glam::Vec3::new(prim.params[0], prim.params[1], prim.params[2]);
            let r = prim.params[3];
            let b = glam::Vec3::new(prim.params[4], prim.params[5], prim.params[6]);
            let pa = p - a;
            let ba = b - a;
            let h = (pa.dot(ba) / ba.dot(ba).max(1e-10)).clamp(0.0, 1.0);
            (pa - ba * h).length() - r
        }
        _ => f32::MAX,
    }
}

/// Polynomial smooth minimum.
#[inline]
pub(super) fn smin_implicit(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    a * h + b * (1.0 - h) - k * h * (1.0 - h)
}

/// Evaluate the combined SDF for all primitives in one GPU implicit item.
pub(super) fn eval_implicit_sdf(p: glam::Vec3, item: &GpuImplicitPickItem) -> f32 {
    use crate::resources::ImplicitBlendMode;
    let mut d = item.max_distance;
    for (i, prim) in item.primitives.iter().enumerate() {
        let pd = eval_implicit_primitive(p, prim);
        match item.blend_mode {
            ImplicitBlendMode::Union => {
                d = d.min(pd);
            }
            ImplicitBlendMode::SmoothUnion => {
                let k = if prim.blend > 0.0 { prim.blend } else { 1e-5 };
                d = smin_implicit(d, pd, k);
            }
            ImplicitBlendMode::Intersection => {
                if i == 0 {
                    d = pd;
                } else {
                    d = d.max(pd);
                }
            }
        }
    }
    d
}

/// CPU ray-march against the implicit SDF. Returns `(toi, world_pos)` on hit.
pub(super) fn pick_implicit_sdf(
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
    item: &GpuImplicitPickItem,
) -> Option<(f32, glam::Vec3)> {
    let max_steps = item.max_steps.min(512) as usize;
    let scale = item.step_scale.clamp(0.01, 1.0);
    let hit_thr = item.hit_threshold;
    let max_dist = item.max_distance;
    let min_step = hit_thr * 0.5;

    let mut t = 0.0f32;
    for _ in 0..max_steps {
        if t > max_dist {
            break;
        }
        let p = ray_origin + ray_dir * t;
        let d = eval_implicit_sdf(p, item);
        if d < hit_thr {
            return Some((t, p));
        }
        t += d.abs().max(min_step) * scale;
    }
    None
}

// ---------------------------------------------------------------------------
// CPU volume ray-march for GPU marching cubes isosurface picking
// ---------------------------------------------------------------------------

/// Slab test: returns (t_enter, t_exit) for a ray vs axis-aligned box, or None.
pub(super) fn ray_aabb_slab(
    ray_orig: glam::Vec3,
    ray_dir: glam::Vec3,
    bbox_min: glam::Vec3,
    bbox_max: glam::Vec3,
) -> Option<(f32, f32)> {
    // Avoid division by zero for axis-aligned rays.
    let inv = glam::Vec3::new(
        if ray_dir.x.abs() > 1e-30 {
            1.0 / ray_dir.x
        } else {
            f32::INFINITY * ray_dir.x.signum()
        },
        if ray_dir.y.abs() > 1e-30 {
            1.0 / ray_dir.y
        } else {
            f32::INFINITY * ray_dir.y.signum()
        },
        if ray_dir.z.abs() > 1e-30 {
            1.0 / ray_dir.z
        } else {
            f32::INFINITY * ray_dir.z.signum()
        },
    );
    let t1 = (bbox_min - ray_orig) * inv;
    let t2 = (bbox_max - ray_orig) * inv;
    let tmin = t1.min(t2);
    let tmax = t1.max(t2);
    let t_enter = tmin.x.max(tmin.y).max(tmin.z);
    let t_exit = tmax.x.min(tmax.y).min(tmax.z);
    if t_enter <= t_exit && t_exit >= 0.0 {
        Some((t_enter, t_exit))
    } else {
        None
    }
}

/// Bisect to refine the isovalue crossing between t_lo and t_hi (8 iterations).
pub(super) fn bisect_mc_crossing(
    ray_orig: glam::Vec3,
    ray_dir: glam::Vec3,
    vol: &crate::geometry::marching_cubes::VolumeData,
    isovalue: f32,
    mut t_lo: f32,
    mut t_hi: f32,
) -> f32 {
    let s0 = crate::geometry::marching_cubes::trilinear_sample(
        vol,
        (ray_orig + ray_dir * t_lo).to_array(),
    ) - isovalue;
    let mut lo_sign = s0 < 0.0;
    for _ in 0..8 {
        let mid = (t_lo + t_hi) * 0.5;
        let s = crate::geometry::marching_cubes::trilinear_sample(
            vol,
            (ray_orig + ray_dir * mid).to_array(),
        ) - isovalue;
        if (s < 0.0) == lo_sign {
            t_lo = mid;
        } else {
            t_hi = mid;
            lo_sign = !lo_sign;
        }
    }
    (t_lo + t_hi) * 0.5
}

/// CPU ray-march against a MC isosurface. Returns `(toi, world_pos)` on hit.
///
/// Steps through the volume AABB at half-cell intervals and refines any
/// isovalue crossing to 8 bisection steps.
pub(super) fn pick_mc_volume(
    ray_orig: glam::Vec3,
    ray_dir: glam::Vec3,
    item: &GpuMcPickItem,
) -> Option<(f32, glam::Vec3)> {
    use crate::geometry::marching_cubes::trilinear_sample;

    let vol = &item.volume_data;
    let isovalue = item.isovalue;
    let [nx, ny, nz] = vol.dims;
    let origin = glam::Vec3::from(vol.origin);
    let spacing = glam::Vec3::from(vol.spacing);
    let extent = spacing * glam::Vec3::new(nx as f32, ny as f32, nz as f32);

    let (t_enter, t_exit) = ray_aabb_slab(ray_orig, ray_dir, origin, origin + extent)?;
    let t_start = t_enter.max(0.0);
    if t_start >= t_exit {
        return None;
    }

    // Step at half the smallest cell spacing so we don't skip thin features.
    let step = spacing.min_element() * 0.5;
    let mut t = t_start;
    let mut prev = trilinear_sample(vol, (ray_orig + ray_dir * t).to_array()) - isovalue;

    loop {
        t += step;
        if t > t_exit {
            break;
        }
        let p = ray_orig + ray_dir * t;
        let cur = trilinear_sample(vol, p.to_array()) - isovalue;
        if prev * cur <= 0.0 {
            // Sign change detected: bisect and return.
            let t_hit = bisect_mc_crossing(ray_orig, ray_dir, vol, isovalue, t - step, t);
            let world_pos = ray_orig + ray_dir * t_hit;
            return Some((t_hit, world_pos));
        }
        prev = cur;
    }
    None
}
