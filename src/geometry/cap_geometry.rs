//! Cap geometry generation for section-view cross-section fill.
//!
//! When clip planes cut through solid meshes, this module generates filled cap
//! surfaces at the cross-section. The algorithm:
//! 1. Extract contour segments (plane-triangle intersections)
//! 2. Assemble segments into closed loops
//! 3. Triangulate loops via ear-clipping

use std::collections::HashMap;

/// Result of cap mesh generation: positions, indices, and flat normal.
#[allow(dead_code)]
pub(crate) struct CapMesh {
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

    segments
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
                break; // Open chain — force-close below
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
                if idx == prev || idx == curr || idx == next {
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
            // Degenerate polygon — bail with what we have
            break;
        }
    }

    if indices.len() == 3 {
        triangles.push([indices[0], indices[1], indices[2]]);
    }

    triangles
}

// ---------------------------------------------------------------------------
// Step 4: Top-level entry point
// ---------------------------------------------------------------------------

/// Generate cap mesh geometry for a single mesh intersected by a single clip plane.
///
/// Returns `None` if the mesh doesn't intersect the plane or produces degenerate geometry.
pub(crate) fn generate_cap_mesh(
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

    let mut all_positions: Vec<[f32; 3]> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();

    for loop_3d in &loops {
        if loop_3d.len() < 3 {
            continue;
        }

        let (pts_2d, _tangent, _bitangent) = project_to_2d(loop_3d, plane_normal);

        // Ensure CCW winding
        let area = signed_area_2d(&pts_2d);
        let pts_2d = if area < 0.0 {
            pts_2d.into_iter().rev().collect::<Vec<_>>()
        } else {
            pts_2d
        };
        let loop_3d: Vec<glam::Vec3> = if area < 0.0 {
            loop_3d.iter().copied().rev().collect()
        } else {
            loop_3d.clone()
        };

        if area.abs() < 1e-8 {
            continue; // Degenerate
        }

        let tris = ear_clip(&pts_2d);
        if tris.is_empty() {
            continue;
        }

        let base = all_positions.len() as u32;
        for v in &loop_3d {
            all_positions.push(v.to_array());
        }
        for [a, b, c] in &tris {
            all_indices.push(base + *a as u32);
            all_indices.push(base + *b as u32);
            all_indices.push(base + *c as u32);
        }
    }

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
        // A unit cube — 12 triangles
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
    }
}
