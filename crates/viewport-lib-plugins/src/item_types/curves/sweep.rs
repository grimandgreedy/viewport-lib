//! Sweeping polyline strips into meshes on the CPU: closed tubes for the
//! streamtube and tube builders, flat strips for the ribbon builder.
//!
//! Both orient each point by a parallel transport frame. Consecutive points
//! that coincide (submitted twice, or rounded together by f32 far from the
//! origin) sweep a zero-length segment: its points take the tangent of the
//! nearest segment that has a length, so the rest of the strip still draws.

use glam::Vec3;
use viewport_lib::plugin_api::builders::Vertex;

/// The swept mesh, with per-triangle segment and strip maps filled in
/// lockstep with `indices` so a GPU pick's `primitive_index` resolves to a
/// segment or strip.
#[derive(Default)]
pub(super) struct SweptMesh {
    pub verts: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub tri_segment: Vec<u32>,
    pub tri_strip: Vec<u32>,
}

/// Sweep every strip of `positions` into a tube with `sides` vertices per
/// ring. `radius_at` and `colour_at` take a point's index into `positions`.
///
/// Segments are numbered the way the CPU picker walks `strip_lengths`: each
/// strip owns `len - 1` of them, whether or not it draws.
pub(super) fn sweep_tube(
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    sides: usize,
    radius_at: impl Fn(usize) -> f32,
    colour_at: impl Fn(usize) -> [f32; 4],
) -> SweptMesh {
    let mut out = SweptMesh::default();
    let mut seg_acc: u32 = 0;
    let mut strip_start = 0usize;

    for (strip_idx, &strip_len) in strip_lengths.iter().enumerate() {
        let strip_idx = strip_idx as u32;
        let strip_len = strip_len as usize;
        let seg_base = seg_acc;
        seg_acc += strip_len.saturating_sub(1) as u32;
        let strip_end = (strip_start + strip_len).min(positions.len());
        let pts: Vec<Vec3> = positions[strip_start..strip_end]
            .iter()
            .map(|&p| Vec3::from(p))
            .collect();
        let first = strip_start;
        strip_start += strip_len;

        if pts.len() < 2 {
            continue;
        }

        // Seed: the first segment with a length, and a perpendicular to it. A
        // strip whose points all coincide draws nothing.
        let Some(t0) = pts.windows(2).find_map(|w| direction(w[1] - w[0])) else {
            continue;
        };
        let ref_v = if t0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let mut u = t0.cross(ref_v).normalize();

        let ring_base = out.verts.len() as u32;
        let n_rings = pts.len();
        // The tangent the previous ring was built with.
        let mut t_prev = t0;
        let mut first_tangent = t0;

        for (k, &pt) in pts.iter().enumerate() {
            // Forward difference, except at the last point; a zero-length
            // difference keeps the previous tangent.
            let diff = if k + 1 < pts.len() {
                pts[k + 1] - pt
            } else {
                pt - pts[k - 1]
            };
            let tangent = direction(diff).unwrap_or(t_prev);
            if k == 0 {
                first_tangent = tangent;
            }

            // Rotate u by the angle between the previous tangent and this one
            // (Rodrigues).
            if k > 0 {
                let axis = t_prev.cross(tangent);
                let sin_a = axis.length().min(1.0);
                if sin_a > 1e-6 {
                    let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                    let ax = axis / sin_a;
                    u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                    u = u.normalize_or_zero();
                }
            }

            let v = tangent.cross(u).normalize_or_zero();
            t_prev = tangent;
            let radius = radius_at(first + k);
            let colour = colour_at(first + k);

            for s in 0..sides {
                let theta = 2.0 * std::f32::consts::PI * (s as f32) / (sides as f32);
                let nx = theta.cos() * u.x + theta.sin() * v.x;
                let ny = theta.cos() * u.y + theta.sin() * v.y;
                let nz = theta.cos() * u.z + theta.sin() * v.z;
                let normal = Vec3::new(nx, ny, nz);
                out.verts.push(Vertex {
                    position: (pt + normal * radius).to_array(),
                    normal: normal.to_array(),
                    colour,
                    uv: [0.0, 0.0],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
            }

            // Quads between ring k-1 and ring k, wound counter-clockwise seen
            // from outside.
            if k > 0 {
                let r0 = ring_base + ((k - 1) * sides) as u32;
                let r1 = ring_base + (k * sides) as u32;
                let seg = seg_base + (k - 1) as u32;
                for s in 0..sides {
                    let s1 = (s + 1) % sides;
                    out.indices.extend_from_slice(&[
                        r0 + s as u32,
                        r0 + s1 as u32,
                        r1 + s as u32,
                        r0 + s1 as u32,
                        r1 + s1 as u32,
                        r1 + s as u32,
                    ]);
                    out.tri_segment.extend_from_slice(&[seg, seg]);
                    out.tri_strip.extend_from_slice(&[strip_idx, strip_idx]);
                }
            }
        }

        // The end cap belongs to the last segment, the start cap to the first.
        let last_seg = seg_base + (n_rings - 2) as u32;

        // End cap: a fan facing forward.
        {
            let last_ring = ring_base + ((n_rings - 1) * sides) as u32;
            let tangent = t_prev;
            let centre = out.verts.len() as u32;
            out.verts.push(Vertex {
                position: pts[n_rings - 1].to_array(),
                normal: tangent.to_array(),
                colour: colour_at(first + n_rings - 1),
                uv: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
            for s in 0..sides {
                let s1 = (s + 1) % sides;
                out.indices.extend_from_slice(&[
                    centre,
                    last_ring + s as u32,
                    last_ring + s1 as u32,
                ]);
                out.tri_segment.push(last_seg);
                out.tri_strip.push(strip_idx);
            }
        }

        // Start cap: a fan facing backward.
        {
            let tangent = -first_tangent;
            let centre = out.verts.len() as u32;
            out.verts.push(Vertex {
                position: pts[0].to_array(),
                normal: tangent.to_array(),
                colour: colour_at(first),
                uv: [0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
            });
            for s in 0..sides {
                let s1 = (s + 1) % sides;
                out.indices.extend_from_slice(&[
                    centre,
                    ring_base + s1 as u32,
                    ring_base + s as u32,
                ]);
                out.tri_segment.push(seg_base);
                out.tri_strip.push(strip_idx);
            }
        }
    }
    out
}

/// Sweep every strip of `positions` into a flat ribbon: two vertices per
/// point, `half_width_at` either side along the frame's lateral, joined by
/// two triangles per segment and no caps.
///
/// `uv.x` runs along the strip, from `u_attribute` when it is not empty and
/// otherwise the arc length normalised to [0, 1]; `uv.y` picks the edge. A
/// `twist` vector at a point turns the lateral towards it. The index closures
/// and the segment numbering work as in [`sweep_tube`].
pub(super) fn sweep_ribbon(
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    u_attribute: &[f32],
    twist: Option<&[[f32; 3]]>,
    half_width_at: impl Fn(usize) -> f32,
    colour_at: impl Fn(usize) -> [f32; 4],
) -> SweptMesh {
    let mut out = SweptMesh::default();
    let mut seg_acc: u32 = 0;
    let mut strip_start = 0usize;

    for (strip_idx, &strip_len) in strip_lengths.iter().enumerate() {
        let strip_idx = strip_idx as u32;
        let strip_len = strip_len as usize;
        let seg_base = seg_acc;
        seg_acc += strip_len.saturating_sub(1) as u32;
        let strip_end = (strip_start + strip_len).min(positions.len());
        let pts: Vec<Vec3> = positions[strip_start..strip_end]
            .iter()
            .map(|&p| Vec3::from(p))
            .collect();
        let first = strip_start;
        strip_start += strip_len;

        if pts.len() < 2 {
            continue;
        }

        let Some(t0) = pts.windows(2).find_map(|w| direction(w[1] - w[0])) else {
            continue;
        };
        let ref_v = if t0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let mut u = t0.cross(ref_v).normalize();

        let mut strip_u: Vec<f32> = Vec::with_capacity(pts.len());
        if u_attribute.is_empty() {
            let mut cum = 0.0_f32;
            strip_u.push(0.0);
            for k in 1..pts.len() {
                cum += (pts[k] - pts[k - 1]).length();
                strip_u.push(cum);
            }
            let total = strip_u.last().copied().unwrap_or(1.0).max(1e-6);
            for v in &mut strip_u {
                *v /= total;
            }
        } else {
            for k in 0..pts.len() {
                strip_u.push(*u_attribute.get(first + k).unwrap_or(&0.0));
            }
        }

        let base = out.verts.len() as u32;
        let mut t_prev = t0;

        for (k, &pt) in pts.iter().enumerate() {
            let diff = if k + 1 < pts.len() {
                pts[k + 1] - pt
            } else {
                pt - pts[k - 1]
            };
            let tangent = direction(diff).unwrap_or(t_prev);

            if k > 0 {
                let axis = t_prev.cross(tangent);
                let sin_a = axis.length().min(1.0);
                if sin_a > 1e-6 {
                    let cos_a = t_prev.dot(tangent).clamp(-1.0, 1.0);
                    let ax = axis / sin_a;
                    u = u * cos_a + ax.cross(u) * sin_a + ax * ax.dot(u) * (1.0 - cos_a);
                    u = u.normalize_or_zero();
                }
            }
            t_prev = tangent;

            let mut lateral = u;
            if let Some(&tv) = twist.and_then(|t| t.get(first + k)) {
                let tv = Vec3::from(tv);
                let proj = tv - tangent * tangent.dot(tv);
                if proj.length_squared() > 1e-10 {
                    lateral = proj.normalize();
                }
            }

            let normal = tangent.cross(lateral).normalize_or_zero();
            let half_w = half_width_at(first + k);
            let colour = colour_at(first + k);
            let uval = strip_u[k];
            for (position, edge) in [(pt + lateral * half_w, 0.0), (pt - lateral * half_w, 1.0)] {
                out.verts.push(Vertex {
                    position: position.to_array(),
                    normal: normal.to_array(),
                    colour,
                    uv: [uval, edge],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                });
            }

            if k > 0 {
                let r0 = base + ((k - 1) * 2) as u32;
                let r1 = base + (k * 2) as u32;
                out.indices
                    .extend_from_slice(&[r0, r0 + 1, r1, r0 + 1, r1 + 1, r1]);
                let seg = seg_base + (k - 1) as u32;
                out.tri_segment.extend_from_slice(&[seg, seg]);
                out.tri_strip.extend_from_slice(&[strip_idx, strip_idx]);
            }
        }
    }
    out
}

/// `v` normalised, or `None` when it has no usable length.
fn direction(v: Vec3) -> Option<Vec3> {
    let d = v.normalize_or_zero();
    (d.length_squared() >= 1e-10).then_some(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: [f32; 4] = [1.0; 4];

    /// A helix of radius 0.5 m rising 1.25 m, its points rounded to 1/8 m
    /// and moved `y` out along +Y in f32.
    fn helix(y: f32, n: usize) -> Vec<[f32; 3]> {
        let q = |v: f32| (v * 8.0).round() / 8.0;
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let theta = t * std::f32::consts::TAU * 1.5;
                [
                    q(0.5 * theta.cos()),
                    y + q(0.5 * theta.sin()),
                    q(0.25 + 1.25 * t),
                ]
            })
            .collect()
    }

    fn bounds(verts: &[Vertex]) -> (Vec3, Vec3) {
        verts.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), v| {
                let p = Vec3::from(v.position);
                (lo.min(p), hi.max(p))
            },
        )
    }

    #[test]
    fn a_helix_6400_km_out_sweeps_a_whole_tube() {
        let n = 32;
        let sides = 8;
        let r = 0.08;
        let far = helix(6_400_000.0, n);
        // f32 steps by half a metre this far out, so neighbours round
        // together, the first two among them.
        assert_eq!(far[0], far[1]);
        let (anchor, local) = crate::item_types::origin::anchored(&far);
        assert_ne!(anchor, Vec3::ZERO);
        let local = local.unwrap();

        let mesh = sweep_tube(&local, &[n as u32], sides, |_| r, |_| WHITE);
        assert_eq!(mesh.verts.len(), n * sides + 2);
        assert_eq!(mesh.indices.len(), ((n - 1) * 2 + 2) * sides * 3);
        assert_eq!(mesh.tri_segment.len(), mesh.indices.len() / 3);
        for v in &mesh.verts {
            assert!(Vec3::from(v.position).is_finite());
            let len = Vec3::from(v.normal).length();
            assert!((len - 1.0).abs() < 1e-3, "degenerate normal {:?}", v.normal);
        }
        // Every ring is a full circle of radius r around its point.
        for (k, p) in local.iter().enumerate() {
            for v in &mesh.verts[k * sides..(k + 1) * sides] {
                let d = (Vec3::from(v.position) - Vec3::from(*p)).length();
                assert!((d - r).abs() < 1e-4, "ring {k} vertex {d} m from its point");
            }
        }
        let (lo, hi) = bounds(&mesh.verts);
        let size = hi - lo;
        assert!(
            (size.x - (1.0 + 2.0 * r)).abs() < 0.05,
            "x extent {}",
            size.x
        );
        assert!(
            size.y > 1.0 && size.y < 1.5 + 2.0 * r,
            "y extent {}",
            size.y
        );
        assert!(
            (size.z - (1.25 + 2.0 * r)).abs() < 0.05,
            "z extent {}",
            size.z
        );
    }

    #[test]
    fn near_points_sweep_the_same_with_and_without_a_repeat() {
        let pts = helix(0.0, 16);
        let plain = sweep_tube(&pts, &[16], 6, |_| 0.1, |_| WHITE);
        let mut doubled = pts.clone();
        doubled.insert(0, pts[0]);
        let swept = sweep_tube(&doubled, &[17], 6, |_| 0.1, |_| WHITE);
        // The repeated point adds one ring and the segment between the two
        // copies; the rest of the mesh is the same.
        assert_eq!(swept.verts.len(), plain.verts.len() + 6);
        for (a, b) in swept.verts[6..17 * 6].iter().zip(&plain.verts[..16 * 6]) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.normal, b.normal);
        }
    }

    fn ribbon(positions: &[[f32; 3]], len: u32) -> SweptMesh {
        sweep_ribbon(positions, &[len], &[], None, |_| 0.125, |_| WHITE)
    }

    #[test]
    fn a_ribbon_whose_first_points_coincide_still_draws() {
        let pts = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let mesh = ribbon(&pts, 3);
        assert_eq!(mesh.verts.len(), 6);
        assert_eq!(mesh.indices.len(), 2 * 2 * 3);
        for v in &mesh.verts {
            assert!(Vec3::from(v.position).is_finite());
            assert!((Vec3::from(v.normal).length() - 1.0).abs() < 1e-3);
        }
        let (lo, hi) = bounds(&mesh.verts);
        assert!((hi.x - lo.x - 2.0).abs() < 1e-6);
        assert!(((hi - lo).length() - (4.0f32 + 0.0625).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn near_points_sweep_the_same_ribbon_with_and_without_a_repeat() {
        let pts = helix(0.0, 16);
        let plain = ribbon(&pts, 16);
        let mut doubled = pts.clone();
        doubled.insert(0, pts[0]);
        let swept = ribbon(&doubled, 17);
        assert_eq!(swept.verts.len(), plain.verts.len() + 2);
        for (a, b) in swept.verts[2..].iter().zip(&plain.verts) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.normal, b.normal);
            assert_eq!(a.uv, b.uv);
        }
    }

    #[test]
    fn a_strip_of_one_repeated_point_draws_nothing_but_keeps_its_segments() {
        let pts = [[1.0, 2.0, 3.0]; 3]
            .into_iter()
            .chain([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .collect::<Vec<_>>();
        let mesh = sweep_tube(&pts, &[3, 2], 4, |_| 0.1, |_| WHITE);
        assert_eq!(mesh.verts.len(), 2 * 4 + 2);
        assert!(mesh.tri_segment.iter().all(|&s| s == 2));
        assert!(mesh.tri_strip.iter().all(|&s| s == 1));
    }
}
