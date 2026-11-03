//! Cap geometry generation for section-view cross-section fill.
//!
//! When clip planes cut through solid meshes, this module generates filled cap
//! surfaces at the cross-section. The algorithm:
//! 1. Extract contour segments (plane-triangle intersections)
//! 2. Assemble segments into closed loops
//! 3. Nest the loops by even-odd depth and bridge each hole into its outer loop
//! 4. Triangulate each merged polygon via ear-clipping

use std::collections::HashMap;

/// Result of cap mesh generation: positions, indices, and flat normal.
#[allow(dead_code)]
pub struct CapMesh {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub normal: [f32; 3],
}

/// Epsilon for vertex-on-plane classification.
const PLANE_EPS: f32 = 1e-6;

/// Scale factor for coordinate quantization in loop assembly.
const QUANT_SCALE: f32 = 1e4;

// ---------------------------------------------------------------------------
// Step 1: Extract contour segments (plane-mesh intersection)
// ---------------------------------------------------------------------------

/// For each triangle that straddles the plane, compute the two intersection
/// points where the triangle edges cross the plane. Returns unordered segments.
fn extract_contour_segments(
    positions: &[[f32; 3]],
    indices: &[u32],
    model: &glam::Mat4,
    plane_normal: glam::Vec3,
    plane_distance: f32,
) -> Vec<[glam::Vec3; 2]> {
    let mut segments = Vec::new();

    for tri in indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }
        let v0 = model.transform_point3(glam::Vec3::from(positions[tri[0] as usize]));
        let v1 = model.transform_point3(glam::Vec3::from(positions[tri[1] as usize]));
        let v2 = model.transform_point3(glam::Vec3::from(positions[tri[2] as usize]));

        let d0 = plane_normal.dot(v0) + plane_distance;
        let d1 = plane_normal.dot(v1) + plane_distance;
        let d2 = plane_normal.dot(v2) + plane_distance;

        // Classify: positive side (>= eps) or negative side
        let s0 = if d0.abs() < PLANE_EPS { 0.0 } else { d0 };
        let s1 = if d1.abs() < PLANE_EPS { 0.0 } else { d1 };
        let s2 = if d2.abs() < PLANE_EPS { 0.0 } else { d2 };

        let verts = [v0, v1, v2];
        let dists = [s0, s1, s2];

        // Collect crossing points
        let mut crossings = Vec::new();
        for i in 0..3 {
            let j = (i + 1) % 3;
            let di = dists[i];
            let dj = dists[j];
            if (di > 0.0 && dj < 0.0) || (di < 0.0 && dj > 0.0) {
                let t = di / (di - dj);
                crossings.push(verts[i].lerp(verts[j], t));
            } else if di == 0.0 {
                crossings.push(verts[i]);
            }
        }

        // Deduplicate near-coincident crossings
        crossings.dedup_by(|a, b| (*a - *b).length_squared() < PLANE_EPS * PLANE_EPS);

        if crossings.len() == 2 {
            segments.push([crossings[0], crossings[1]]);
        }
    }

    // Deduplicate segments: when vertices lie on the clip plane, multiple
    // triangles sharing those vertices produce identical segments.
    dedup_segments(&mut segments);

    segments
}

/// Remove duplicate segments using quantized endpoint pairs.
/// Two segments match if their quantized endpoints are identical in either order.
fn dedup_segments(segments: &mut Vec<[glam::Vec3; 2]>) {
    use std::collections::HashSet;
    type Key = ((i32, i32, i32), (i32, i32, i32));

    let mut seen = HashSet::new();
    segments.retain(|seg| {
        let a = quantize(seg[0]);
        let b = quantize(seg[1]);
        // Canonical order so (A,B) and (B,A) are the same key.
        let key: Key = if a <= b { (a, b) } else { (b, a) };
        seen.insert(key)
    });
}

// ---------------------------------------------------------------------------
// Step 2: Assemble segments into closed loops
// ---------------------------------------------------------------------------

/// Quantize a 3D point to an integer key for adjacency lookup.
fn quantize(v: glam::Vec3) -> (i32, i32, i32) {
    (
        (v.x * QUANT_SCALE).round() as i32,
        (v.y * QUANT_SCALE).round() as i32,
        (v.z * QUANT_SCALE).round() as i32,
    )
}

/// Chain unordered segments into closed (or force-closed) loops.
fn assemble_loops(segments: &[[glam::Vec3; 2]]) -> Vec<Vec<glam::Vec3>> {
    if segments.is_empty() {
        return Vec::new();
    }

    // Build adjacency: quantized point -> list of (segment_index, other_endpoint)
    type AdjEntry = (usize, glam::Vec3, glam::Vec3);
    let mut adj: HashMap<(i32, i32, i32), Vec<AdjEntry>> = HashMap::new();
    for (i, seg) in segments.iter().enumerate() {
        let k0 = quantize(seg[0]);
        let k1 = quantize(seg[1]);
        adj.entry(k0).or_default().push((i, seg[0], seg[1]));
        adj.entry(k1).or_default().push((i, seg[1], seg[0]));
    }

    let mut used = vec![false; segments.len()];
    let mut loops = Vec::new();

    for start_idx in 0..segments.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;

        let mut chain = vec![segments[start_idx][0], segments[start_idx][1]];
        let start_key = quantize(chain[0]);

        // Walk forward from chain end
        loop {
            let end_key = quantize(*chain.last().unwrap());
            if chain.len() > 2 && end_key == start_key {
                // Loop closed
                chain.pop(); // Remove duplicate endpoint
                break;
            }
            let mut found = false;
            if let Some(neighbors) = adj.get(&end_key) {
                for &(seg_i, _this_pt, other_pt) in neighbors {
                    if !used[seg_i] {
                        used[seg_i] = true;
                        chain.push(other_pt);
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                break; // Open chain : force-close below
            }
        }

        if chain.len() >= 3 {
            loops.push(chain);
        }
    }

    loops
}

// ---------------------------------------------------------------------------
// Step 3: Ear-clipping triangulation
// ---------------------------------------------------------------------------

/// Project a 3D loop onto the clip plane's 2D coordinate system.
fn project_to_2d(
    loop_3d: &[glam::Vec3],
    plane_normal: glam::Vec3,
) -> (Vec<[f32; 2]>, glam::Vec3, glam::Vec3) {
    // Build tangent/bitangent from plane normal
    let up = if plane_normal.y.abs() < 0.99 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let tangent = plane_normal.cross(up).normalize();
    let bitangent = plane_normal.cross(tangent).normalize();

    let pts: Vec<[f32; 2]> = loop_3d
        .iter()
        .map(|p| [tangent.dot(*p), bitangent.dot(*p)])
        .collect();

    (pts, tangent, bitangent)
}

/// Signed area of a 2D polygon (positive = CCW).
fn signed_area_2d(pts: &[[f32; 2]]) -> f32 {
    let n = pts.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
    }
    area * 0.5
}

/// Check if point p is inside triangle (a, b, c) using cross products.
fn point_in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let cross = |o: [f32; 2], u: [f32; 2], v: [f32; 2]| -> f32 {
        (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    };
    let d1 = cross(p, a, b);
    let d2 = cross(p, b, c);
    let d3 = cross(p, c, a);
    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    !(has_neg && has_pos)
}

/// Ear-clipping triangulation of a 2D polygon (assumed CCW).
/// Returns triangle indices into the original vertex array.
fn ear_clip(pts: &[[f32; 2]]) -> Vec<[usize; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..n).collect();
    let mut triangles = Vec::new();

    while indices.len() > 3 {
        let len = indices.len();
        let mut found_ear = false;

        for i in 0..len {
            let prev = indices[(i + len - 1) % len];
            let curr = indices[i];
            let next = indices[(i + 1) % len];

            let a = pts[prev];
            let b = pts[curr];
            let c = pts[next];

            // Check convexity (cross product > 0 for CCW)
            let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if cross <= 0.0 {
                continue; // Reflex vertex
            }

            // Check no other vertex inside this triangle
            let mut ear = true;
            for &idx in &indices {
                // Bridged holes repeat two positions; a copy of a corner is
                // not inside the ear.
                if idx == prev || idx == curr || idx == next {
                    continue;
                }
                let p = pts[idx];
                if p == a || p == b || p == c {
                    continue;
                }
                if point_in_triangle(pts[idx], a, b, c) {
                    ear = false;
                    break;
                }
            }

            if ear {
                triangles.push([prev, curr, next]);
                indices.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            // Degenerate polygon : bail with what we have
            break;
        }
    }

    if indices.len() == 3 {
        triangles.push([indices[0], indices[1], indices[2]]);
    }

    triangles
}

// ---------------------------------------------------------------------------
// Step 4: Holes
// ---------------------------------------------------------------------------

/// Even-odd point-in-polygon test.
fn point_in_polygon(p: [f32; 2], poly: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if p[0] < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Whether segments `p0-p1` and `q0-q1` touch anywhere, endpoints included.
fn segments_touch(p0: [f32; 2], p1: [f32; 2], q0: [f32; 2], q1: [f32; 2]) -> bool {
    let orient = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| -> f32 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let on_segment = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| -> bool {
        c[0] >= a[0].min(b[0])
            && c[0] <= a[0].max(b[0])
            && c[1] >= a[1].min(b[1])
            && c[1] <= a[1].max(b[1])
    };
    let d1 = orient(q0, q1, p0);
    let d2 = orient(q0, q1, p1);
    let d3 = orient(p0, p1, q0);
    let d4 = orient(p0, p1, q1);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    (d1 == 0.0 && on_segment(q0, q1, p0))
        || (d2 == 0.0 && on_segment(q0, q1, p1))
        || (d3 == 0.0 && on_segment(p0, p1, q0))
        || (d4 == 0.0 && on_segment(p0, p1, q1))
}

/// A polygon in the plane's 2D frame with its 3D points alongside.
struct Ring {
    pts: Vec<[f32; 2]>,
    world: Vec<glam::Vec3>,
}

impl Ring {
    fn reversed(mut self) -> Self {
        self.pts.reverse();
        self.world.reverse();
        self
    }

    fn max_x(&self) -> f32 {
        self.pts.iter().fold(f32::MIN, |m, p| m.max(p[0]))
    }
}

/// Whether the segment `a-b` touches any edge of `ring`, skipping edges that
/// share an endpoint position with the segment.
fn segment_blocked(a: [f32; 2], b: [f32; 2], ring: &[[f32; 2]]) -> bool {
    let n = ring.len();
    for i in 0..n {
        let (q0, q1) = (ring[i], ring[(i + 1) % n]);
        if q0 == a || q0 == b || q1 == a || q1 == b {
            continue;
        }
        if segments_touch(a, b, q0, q1) {
            return true;
        }
    }
    false
}

/// Join `hole` (wound CW) into `outer` (wound CCW) with a two-way bridge from
/// the hole's rightmost vertex to the nearest outer vertex it can see past
/// both rings and every hole still to be joined. Returns `false`, leaving
/// `outer` untouched, if no such vertex exists.
fn bridge_hole(outer: &mut Ring, hole: &Ring, pending: &[Ring]) -> bool {
    let mi = (0..hole.pts.len())
        .max_by(|&a, &b| hole.pts[a][0].total_cmp(&hole.pts[b][0]))
        .unwrap();
    let m = hole.pts[mi];
    let mut candidates: Vec<usize> = (0..outer.pts.len()).collect();
    let dist = |i: usize| {
        let p = outer.pts[i];
        (p[0] - m[0]).powi(2) + (p[1] - m[1]).powi(2)
    };
    candidates.sort_by(|&a, &b| dist(a).total_cmp(&dist(b)));
    let visible = candidates.into_iter().find(|&vi| {
        let v = outer.pts[vi];
        v != m
            && !segment_blocked(m, v, &outer.pts)
            && !segment_blocked(m, v, &hole.pts)
            && !pending.iter().any(|r| segment_blocked(m, v, &r.pts))
            // The bridge must run through the region between the rings, not
            // through the hole or outside the outer ring.
            && point_in_polygon(
                [(m[0] + v[0]) * 0.5, (m[1] + v[1]) * 0.5],
                &outer.pts,
            )
            && !point_in_polygon(
                [(m[0] + v[0]) * 0.5, (m[1] + v[1]) * 0.5],
                &hole.pts,
            )
    });
    let Some(vi) = visible else {
        return false;
    };
    let n = hole.pts.len();
    let mut pts = Vec::with_capacity(outer.pts.len() + n + 2);
    let mut world = Vec::with_capacity(pts.capacity());
    pts.extend_from_slice(&outer.pts[..=vi]);
    world.extend_from_slice(&outer.world[..=vi]);
    for k in 0..=n {
        let h = (mi + k) % n;
        pts.push(hole.pts[h]);
        world.push(hole.world[h]);
    }
    pts.extend_from_slice(&outer.pts[vi..]);
    world.extend_from_slice(&outer.world[vi..]);
    outer.pts = pts;
    outer.world = world;
    true
}

/// Group loops into outer boundaries and their holes by even-odd nesting
/// depth, then join each outer's holes into it. Each returned ring is wound
/// CCW and ready to ear-clip.
fn merge_holes(loops: &[Vec<glam::Vec3>], plane_normal: glam::Vec3) -> Vec<Ring> {
    let mut rings: Vec<(Ring, f32)> = Vec::new();
    for loop_3d in loops {
        if loop_3d.len() < 3 {
            continue;
        }
        let (pts, _, _) = project_to_2d(loop_3d, plane_normal);
        let area = signed_area_2d(&pts);
        if area.abs() < 1e-8 {
            continue;
        }
        rings.push((
            Ring {
                pts,
                world: loop_3d.clone(),
            },
            area,
        ));
    }

    let n = rings.len();
    let contains = |outer: usize, inner: usize| -> bool {
        point_in_polygon(rings[inner].0.pts[0], &rings[outer].0.pts)
    };
    let depth: Vec<usize> = (0..n)
        .map(|i| (0..n).filter(|&j| j != i && contains(j, i)).count())
        .collect();
    // Each hole belongs to the smallest containing ring one level up.
    let parent: Vec<Option<usize>> = (0..n)
        .map(|i| {
            if depth[i] % 2 == 0 {
                return None;
            }
            (0..n)
                .filter(|&j| j != i && depth[j] + 1 == depth[i] && contains(j, i))
                .min_by(|&a, &b| rings[a].1.abs().total_cmp(&rings[b].1.abs()))
        })
        .collect();

    let mut rings: Vec<Option<(Ring, f32)>> = rings.into_iter().map(Some).collect();
    let mut holes_of: Vec<Vec<Ring>> = (0..n).map(|_| Vec::new()).collect();
    for i in 0..n {
        if let Some(p) = parent[i] {
            let (ring, area) = rings[i].take().unwrap();
            holes_of[p].push(if area > 0.0 { ring.reversed() } else { ring });
        }
    }

    let mut out = Vec::new();
    for i in 0..n {
        // A hole with no parent (an odd depth reached through a malformed
        // section) is dropped rather than filled.
        if depth[i] % 2 == 1 {
            continue;
        }
        let Some((ring, area)) = rings[i].take() else {
            continue;
        };
        let mut outer = if area < 0.0 { ring.reversed() } else { ring };
        let mut holes = std::mem::take(&mut holes_of[i]);
        holes.sort_by(|a, b| b.max_x().total_cmp(&a.max_x()));
        for k in 0..holes.len() {
            bridge_hole(&mut outer, &holes[k], &holes[k + 1..]);
        }
        out.push(outer);
    }
    out
}

// ---------------------------------------------------------------------------
// Step 5: Top-level entry point
// ---------------------------------------------------------------------------

/// Triangulate closed section loops into one flat fill. Loops nested inside
/// another are holes, and loops inside a hole are filled again.
fn fill_loops(loops: &[Vec<glam::Vec3>], plane_normal: glam::Vec3) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    for ring in merge_holes(loops, plane_normal) {
        let tris = ear_clip(&ring.pts);
        if tris.is_empty() {
            continue;
        }
        let base = all_positions.len() as u32;
        all_positions.extend(ring.world.iter().map(|v| v.to_array()));
        for [a, b, c] in &tris {
            all_indices.extend_from_slice(&[base + *a as u32, base + *b as u32, base + *c as u32]);
        }
    }
    (all_positions, all_indices)
}

/// Generate cap mesh geometry for a single mesh intersected by a single clip plane.
///
/// A section with several loops fills each outer boundary and leaves its
/// holes open, so a hollow tube caps as a ring. Returns `None` if the mesh
/// doesn't intersect the plane or produces degenerate geometry.
pub fn generate_cap_mesh(
    positions: &[[f32; 3]],
    indices: &[u32],
    model: &glam::Mat4,
    plane_normal: glam::Vec3,
    plane_distance: f32,
) -> Option<CapMesh> {
    let segments =
        extract_contour_segments(positions, indices, model, plane_normal, plane_distance);
    if segments.is_empty() {
        return None;
    }

    let loops = assemble_loops(&segments);
    if loops.is_empty() {
        return None;
    }

    let (all_positions, all_indices) = fill_loops(&loops, plane_normal);
    if all_positions.is_empty() || all_indices.is_empty() {
        return None;
    }

    Some(CapMesh {
        positions: all_positions,
        indices: all_indices,
        normal: plane_normal.to_array(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A single triangle straddling the YZ plane (x=0).
    fn single_tri_positions() -> Vec<[f32; 3]> {
        vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    }

    #[test]
    fn test_extract_no_intersection() {
        let positions = vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 1.0, 0.0]];
        let indices = vec![0, 1, 2];
        let segs = extract_contour_segments(
            &positions,
            &indices,
            &glam::Mat4::IDENTITY,
            glam::Vec3::X,
            0.0,
        );
        // All vertices on positive side
        assert!(segs.is_empty());
    }

    #[test]
    fn test_extract_one_crossing() {
        let positions = single_tri_positions();
        let indices = vec![0, 1, 2];
        let segs = extract_contour_segments(
            &positions,
            &indices,
            &glam::Mat4::IDENTITY,
            glam::Vec3::X,
            0.0,
        );
        assert_eq!(segs.len(), 1);
    }

    #[test]
    fn test_extract_all_below() {
        let positions = vec![[-3.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-2.5, 1.0, 0.0]];
        let indices = vec![0, 1, 2];
        let segs = extract_contour_segments(
            &positions,
            &indices,
            &glam::Mat4::IDENTITY,
            glam::Vec3::X,
            0.0,
        );
        assert!(segs.is_empty());
    }

    #[test]
    fn test_loop_assembly_square() {
        // Four segments forming a square
        let segments = vec![
            [
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 0.0, 0.0),
            ],
            [
                glam::Vec3::new(1.0, 0.0, 0.0),
                glam::Vec3::new(1.0, 1.0, 0.0),
            ],
            [
                glam::Vec3::new(1.0, 1.0, 0.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
            ],
            [
                glam::Vec3::new(0.0, 1.0, 0.0),
                glam::Vec3::new(0.0, 0.0, 0.0),
            ],
        ];
        let loops = assemble_loops(&segments);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 4);
    }

    #[test]
    fn test_ear_clip_triangle() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let tris = ear_clip(&pts);
        assert_eq!(tris.len(), 1);
    }

    #[test]
    fn test_ear_clip_square() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tris = ear_clip(&pts);
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn test_ear_clip_concave_l() {
        // L-shaped concave polygon (CCW)
        let pts = vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let tris = ear_clip(&pts);
        assert_eq!(tris.len(), 4); // 6 vertices -> 4 triangles
    }

    #[test]
    fn test_generate_cap_mesh_box() {
        // A unit cube : 12 triangles
        let positions = vec![
            // Front face (z=0.5)
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            // Back face (z=-0.5)
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
        ];
        #[rustfmt::skip]
        let indices = vec![
            0,1,2, 0,2,3, // front
            5,4,7, 5,7,6, // back
            4,0,3, 4,3,7, // left
            1,5,6, 1,6,2, // right
            3,2,6, 3,6,7, // top
            4,5,1, 4,1,0, // bottom
        ];
        let cap = generate_cap_mesh(
            &positions,
            &indices,
            &glam::Mat4::IDENTITY,
            glam::Vec3::X,
            0.0, // YZ plane at x=0
        );
        assert!(cap.is_some());
        let cap = cap.unwrap();
        assert!(!cap.positions.is_empty());
        assert!(!cap.indices.is_empty());
        // Should produce a square cap
        assert!(cap.indices.len() >= 3);
        let (area, consistent) = fill_area(&cap.positions, &cap.indices, glam::Vec3::X);
        assert!((area - 1.0).abs() < 1e-4, "area {area}");
        assert!(consistent);
    }

    /// UV sphere sliced through its centre: the clip plane passes through
    /// on-plane vertices, which previously produced duplicate segments that
    /// broke loop assembly.
    #[test]
    fn test_generate_cap_mesh_uv_sphere() {
        // Build a small UV sphere (same structure as the showcase).
        let lon_segs = 48usize;
        let lat_segs = 24usize;
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        for lat in 0..=lat_segs {
            let theta = std::f32::consts::PI * lat as f32 / lat_segs as f32;
            let (sin_t, cos_t) = theta.sin_cos();
            for lon in 0..=lon_segs {
                let phi = 2.0 * std::f32::consts::PI * lon as f32 / lon_segs as f32;
                let (sin_p, cos_p) = phi.sin_cos();
                positions.push([sin_t * cos_p, cos_t, sin_t * sin_p]);
            }
        }
        let stride = lon_segs + 1;
        for lat in 0..lat_segs {
            for lon in 0..lon_segs {
                let a = (lat * stride + lon) as u32;
                let b = ((lat + 1) * stride + lon) as u32;
                let c = ((lat + 1) * stride + lon + 1) as u32;
                let d = (lat * stride + lon + 1) as u32;
                if lat > 0 {
                    indices.extend_from_slice(&[a, d, b]);
                }
                if lat < lat_segs - 1 {
                    indices.extend_from_slice(&[b, d, c]);
                }
            }
        }

        // Clip at X=0 through the centre.
        let model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 1.0));
        let cap = generate_cap_mesh(&positions, &indices, &model, glam::Vec3::X, 0.0);
        assert!(cap.is_some(), "sphere cap mesh must be generated");
        let cap = cap.unwrap();
        assert!(cap.indices.len() >= 3);
    }

    /// Total area of a triangle fill, and whether every triangle faces the
    /// same way along `normal` (a fold or overlap breaks that).
    fn fill_area(positions: &[[f32; 3]], indices: &[u32], normal: glam::Vec3) -> (f32, bool) {
        let mut area = 0.0;
        let mut pos = 0;
        let mut neg = 0;
        for t in indices.chunks(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| glam::Vec3::from(positions[i as usize]));
            let n = (b - a).cross(c - a);
            area += n.length() * 0.5;
            let d = n.dot(normal);
            if d > 1e-9 {
                pos += 1;
            } else if d < -1e-9 {
                neg += 1;
            }
        }
        (area, pos == 0 || neg == 0)
    }

    /// An axis-aligned square loop in the z = 0 plane, counter-clockwise
    /// about +Z unless `cw`.
    fn square(half: f32, cw: bool) -> Vec<glam::Vec3> {
        let mut v = vec![
            glam::Vec3::new(-half, -half, 0.0),
            glam::Vec3::new(half, -half, 0.0),
            glam::Vec3::new(half, half, 0.0),
            glam::Vec3::new(-half, half, 0.0),
        ];
        if cw {
            v.reverse();
        }
        v
    }

    #[test]
    fn test_fill_square_with_square_hole() {
        // The hole is given the same winding as the outer loop, so the
        // nesting test, not the input winding, has to classify it.
        let loops = vec![square(2.0, false), square(1.0, false)];
        let (positions, indices) = fill_loops(&loops, glam::Vec3::Z);
        let (area, consistent) = fill_area(&positions, &indices, glam::Vec3::Z);
        assert!((area - (16.0 - 4.0)).abs() < 1e-4, "area {area}");
        assert!(consistent, "triangles fold over");
    }

    #[test]
    fn test_fill_island_inside_hole() {
        let loops = vec![square(1.0, true), square(3.0, false), square(2.0, true)];
        let (positions, indices) = fill_loops(&loops, glam::Vec3::Z);
        let (area, consistent) = fill_area(&positions, &indices, glam::Vec3::Z);
        assert!((area - (36.0 - 16.0 + 4.0)).abs() < 1e-4, "area {area}");
        assert!(consistent, "triangles fold over");
    }

    #[test]
    fn test_fill_two_holes() {
        let shift = |l: Vec<glam::Vec3>, x: f32| -> Vec<glam::Vec3> {
            l.into_iter()
                .map(|v| v + glam::Vec3::new(x, 0.0, 0.0))
                .collect()
        };
        let loops = vec![
            square(3.0, false),
            shift(square(0.5, false), -1.5),
            shift(square(0.5, true), 1.5),
        ];
        let (positions, indices) = fill_loops(&loops, glam::Vec3::Z);
        let (area, consistent) = fill_area(&positions, &indices, glam::Vec3::Z);
        assert!((area - (36.0 - 2.0)).abs() < 1e-4, "area {area}");
        assert!(consistent, "triangles fold over");
    }

    /// A hollow round tube along Z, cut across: the cap is a ring.
    #[test]
    fn test_generate_cap_mesh_hollow_tube() {
        let segs = 32usize;
        let (r_out, r_in) = (1.0f32, 0.6f32);
        let mut positions = Vec::new();
        for r in [r_out, r_in] {
            for z in [-1.0f32, 1.0] {
                for k in 0..segs {
                    let a = std::f32::consts::TAU * k as f32 / segs as f32;
                    positions.push([r * a.cos(), r * a.sin(), z]);
                }
            }
        }
        let at =
            |wall: usize, top: usize, k: usize| (wall * 2 * segs + top * segs + k % segs) as u32;
        let mut indices = Vec::new();
        for wall in 0..2 {
            for k in 0..segs {
                let (a, b, c, d) = (
                    at(wall, 0, k),
                    at(wall, 0, k + 1),
                    at(wall, 1, k + 1),
                    at(wall, 1, k),
                );
                indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
        // End rings joining the walls.
        for top in 0..2 {
            for k in 0..segs {
                let (a, b, c, d) = (
                    at(0, top, k),
                    at(0, top, k + 1),
                    at(1, top, k + 1),
                    at(1, top, k),
                );
                indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
        let cap = generate_cap_mesh(
            &positions,
            &indices,
            &glam::Mat4::IDENTITY,
            glam::Vec3::Z,
            0.0,
        )
        .expect("tube cap");
        let (area, consistent) = fill_area(&cap.positions, &cap.indices, glam::Vec3::Z);
        let polygon =
            |r: f32| 0.5 * segs as f32 * r * r * (std::f32::consts::TAU / segs as f32).sin();
        let expected = polygon(r_out) - polygon(r_in);
        assert!(
            (area - expected).abs() < 1e-3,
            "area {area}, expected {expected}"
        );
        assert!(consistent, "triangles fold over");
    }
}
