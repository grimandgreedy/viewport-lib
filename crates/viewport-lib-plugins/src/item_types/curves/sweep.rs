//! Sweeping polyline strips into a closed tube mesh on the CPU, shared by the
//! streamtube and tube builders.
//!
//! Each strip gets one ring of vertices per point, oriented by a parallel
//! transport frame, with quads between consecutive rings and a flat fan cap
//! at each end.

use glam::Vec3;
use viewport_lib::plugin_api::builders::Vertex;

/// The swept mesh, with per-triangle segment and strip maps filled in
/// lockstep with `indices` so a GPU pick's `primitive_index` resolves to a
/// segment or strip.
#[derive(Default)]
pub(super) struct SweptTube {
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
) -> SweptTube {
    let mut out = SweptTube::default();
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

        // Seed: the initial tangent and a perpendicular to it.
        let t0 = (pts[1] - pts[0]).normalize_or_zero();
        if t0.length_squared() < 1e-10 {
            continue;
        }
        let ref_v = if t0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        let mut u = t0.cross(ref_v).normalize();

        let ring_base = out.verts.len() as u32;
        let n_rings = pts.len();

        for (k, &pt) in pts.iter().enumerate() {
            // Forward difference, except at the last point.
            let tangent = if k + 1 < pts.len() {
                (pts[k + 1] - pt).normalize_or_zero()
            } else {
                (pt - pts[k - 1]).normalize_or_zero()
            };

            // Rotate u by the angle between the previous tangent and this one
            // (Rodrigues).
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

            let v = tangent.cross(u).normalize_or_zero();
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
            let tangent = (pts[n_rings - 1] - pts[n_rings - 2]).normalize_or_zero();
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
            let tangent = (pts[0] - pts[1]).normalize_or_zero();
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
