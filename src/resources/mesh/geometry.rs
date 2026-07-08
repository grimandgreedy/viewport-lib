//! Geometry builders for glyph base meshes and primitive shapes.

use crate::resources::types::*;

pub(crate) fn generate_edge_indices(triangle_indices: &[u32]) -> Vec<u32> {
    // Canonical form: smaller index first, packed into a u64 key so
    // (a,b) and (b,a) collapse to the same edge under sort+dedup. Sorting
    // is several times faster than hashing every edge, and this runs for
    // every uploaded mesh. The output is sorted rather than first-seen
    // order, which line rendering does not care about.
    let mut keys: Vec<u64> = Vec::with_capacity(triangle_indices.len());
    for tri in triangle_indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            keys.push((u64::from(lo) << 32) | u64::from(hi));
        }
    }
    keys.sort_unstable();
    keys.dedup();
    let mut result = Vec::with_capacity(keys.len() * 2);
    for k in keys {
        result.push((k >> 32) as u32);
        result.push(k as u32);
    }
    result
}

// ---------------------------------------------------------------------------
// Procedural unit cube mesh (24 vertices, 4 per face, 36 indices)
// ---------------------------------------------------------------------------

/// Generate a unit cube centered at the origin.
///
/// 24 vertices (4 per face with shared normals), 36 indices (2 triangles per face).
/// All vertices are white [1,1,1,1].
pub(crate) fn build_unit_cube() -> (Vec<Vertex>, Vec<u32>) {
    let white = [1.0f32, 1.0, 1.0, 1.0];
    let mut verts: Vec<Vertex> = Vec::with_capacity(24);
    let mut idx: Vec<u32> = Vec::with_capacity(36);

    // Helper: add a face quad (4 vertices in CCW order) and its 2 triangles.
    let mut add_face = |positions: [[f32; 3]; 4], normal: [f32; 3]| {
        let base = verts.len() as u32;
        for pos in &positions {
            verts.push(Vertex {
                position: *pos,
                normal,
                colour: white,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
        // Two triangles: (base, base+1, base+2) and (base, base+2, base+3)
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // +X face (right), normal [1, 0, 0]
    add_face(
        [
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
        ],
        [1.0, 0.0, 0.0],
    );

    // -X face (left), normal [-1, 0, 0]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5],
        ],
        [-1.0, 0.0, 0.0],
    );

    // +Y face (top), normal [0, 1, 0]
    add_face(
        [
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
        ],
        [0.0, 1.0, 0.0],
    );

    // -Y face (bottom), normal [0, -1, 0]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
        ],
        [0.0, -1.0, 0.0],
    );

    // +Z face (front), normal [0, 0, 1]
    add_face(
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        [0.0, 0.0, 1.0],
    );

    // -Z face (back), normal [0, 0, -1]
    add_face(
        [
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ],
        [0.0, 0.0, -1.0],
    );

    (verts, idx)
}

// ---------------------------------------------------------------------------
// Procedural glyph arrow mesh (cone tip + cylinder shaft, local +Z axis)
// ---------------------------------------------------------------------------

/// Generate a unit arrow mesh aligned to local +Z.
///
/// The arrow consists of:
/// - A cylinder shaft from Z=0 to Z=0.7, radius 0.05.
/// - A cone tip from Z=0.7 to Z=1.0, base radius 0.12.
///
/// 16 segments around the circumference gives ~300 vertices.
pub(crate) fn build_glyph_arrow() -> (Vec<Vertex>, Vec<u32>) {
    let white = [1.0f32, 1.0, 1.0, 1.0];
    let segments = 16usize;
    let mut verts: Vec<Vertex> = Vec::new();
    let mut idx: Vec<u32> = Vec::new();

    let shaft_r = 0.05f32;
    let shaft_bot = 0.0f32;
    let shaft_top = 0.7f32;
    let cone_r = 0.12f32;
    let cone_bot = shaft_top;
    let cone_tip = 1.0f32;

    // Helper: append ring vertices at a given Z and radius with outward normals.
    let ring_verts = |verts: &mut Vec<Vertex>, z: f32, r: f32, normal_z: f32| {
        for i in 0..segments {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
            let (s, c) = angle.sin_cos();
            let nx = if r > 0.0 { c } else { 0.0 };
            let ny = if r > 0.0 { s } else { 0.0 };
            let len = (nx * nx + ny * ny + normal_z * normal_z).sqrt();
            verts.push(Vertex {
                position: [c * r, s * r, z],
                normal: [nx / len, ny / len, normal_z / len],
                colour: white,
                uv: [0.0, 0.0],
                tangent: [0.0, 0.0, 0.0, 1.0],
            });
        }
    };

    // --- Shaft ---
    // Bottom ring (face down for the cap).
    let shaft_bot_base = verts.len() as u32;
    ring_verts(&mut verts, shaft_bot, shaft_r, 0.0);

    // Bottom cap center.
    let shaft_bot_center = verts.len() as u32;
    verts.push(Vertex {
        position: [0.0, 0.0, shaft_bot],
        normal: [0.0, 0.0, -1.0],
        colour: white,
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 1.0],
    });

    // Bottom cap triangles.
    for i in 0..segments {
        let a = shaft_bot_base + i as u32;
        let b = shaft_bot_base + ((i + 1) % segments) as u32;
        idx.extend_from_slice(&[shaft_bot_center, b, a]);
    }

    // Side quads: two rings of shaft.
    let shaft_top_ring_base = verts.len() as u32;
    ring_verts(&mut verts, shaft_bot, shaft_r, 0.0); // duplicate bottom ring for side normals
    let shaft_top_ring_top = verts.len() as u32;
    ring_verts(&mut verts, shaft_top, shaft_r, 0.0);
    for i in 0..segments {
        let a = shaft_top_ring_base + i as u32;
        let b = shaft_top_ring_base + ((i + 1) % segments) as u32;
        let c = shaft_top_ring_top + i as u32;
        let d = shaft_top_ring_top + ((i + 1) % segments) as u32;
        idx.extend_from_slice(&[a, b, d, a, d, c]);
    }

    // --- Cone ---
    // Slanted normal angle for cone surface: rise=(cone_tip-cone_bot), run=cone_r.
    let cone_len = ((cone_tip - cone_bot).powi(2) + cone_r * cone_r).sqrt();
    let normal_z_cone = cone_r / cone_len; // outward Z component of slanted normal
    let normal_r_cone = (cone_tip - cone_bot) / cone_len;

    let cone_base_ring = verts.len() as u32;
    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let (s, c) = angle.sin_cos();
        verts.push(Vertex {
            position: [c * cone_r, s * cone_r, cone_bot],
            normal: [c * normal_r_cone, s * normal_r_cone, normal_z_cone],
            colour: white,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        });
    }

    // Cone tip vertex (normals averaged around tip : just point up).
    let cone_tip_v = verts.len() as u32;
    verts.push(Vertex {
        position: [0.0, 0.0, cone_tip],
        normal: [0.0, 0.0, 1.0],
        colour: white,
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 1.0],
    });

    for i in 0..segments {
        let a = cone_base_ring + i as u32;
        let b = cone_base_ring + ((i + 1) % segments) as u32;
        idx.extend_from_slice(&[a, b, cone_tip_v]);
    }

    // Cone base cap (flat, faces -Z).
    let cone_cap_base = verts.len() as u32;
    for i in 0..segments {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
        let (s, c) = angle.sin_cos();
        verts.push(Vertex {
            position: [c * cone_r, s * cone_r, cone_bot],
            normal: [0.0, 0.0, -1.0],
            colour: white,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        });
    }
    let cone_cap_center = verts.len() as u32;
    verts.push(Vertex {
        position: [0.0, 0.0, cone_bot],
        normal: [0.0, 0.0, -1.0],
        colour: white,
        uv: [0.0, 0.0],
        tangent: [0.0, 0.0, 0.0, 1.0],
    });
    for i in 0..segments {
        let a = cone_cap_base + i as u32;
        let b = cone_cap_base + ((i + 1) % segments) as u32;
        idx.extend_from_slice(&[cone_cap_center, b, a]);
    }

    // The arrow above is authored along local +Z. The glyph shader aligns the
    // local +Y axis with the vector direction, so rotate the whole mesh so its
    // axis becomes +Y. This is a rotation about X by -90 degrees,
    // (x, y, z) -> (x, z, -y), which preserves triangle winding and keeps the
    // normals pointing outward (no need to reorder indices).
    for v in &mut verts {
        let [px, py, pz] = v.position;
        v.position = [px, pz, -py];
        let [nx, ny, nz] = v.normal;
        v.normal = [nx, nz, -ny];
    }

    (verts, idx)
}

// ---------------------------------------------------------------------------
// Procedural icosphere (2 subdivisions, ~240 triangles)
// ---------------------------------------------------------------------------

/// Generate a unit sphere as an icosphere with 2 subdivisions.
///
/// Starts from a regular icosahedron and subdivides each triangle 2x.
pub(crate) fn build_glyph_sphere() -> (Vec<Vertex>, Vec<u32>) {
    let white = [1.0f32, 1.0, 1.0, 1.0];

    // Icosahedron constants.
    let t = (1.0 + 5.0f32.sqrt()) / 2.0;

    // 12 vertices of a regular icosahedron (not yet normalised).
    let raw_verts = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];

    let mut positions: Vec<[f32; 3]> = raw_verts
        .iter()
        .map(|v| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / l, v[1] / l, v[2] / l]
        })
        .collect();

    // 20 base triangles.
    let mut triangles: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    // Subdivide 2 times.
    for _ in 0..2 {
        let mut mid_cache: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let mut new_triangles: Vec<[usize; 3]> = Vec::with_capacity(triangles.len() * 4);

        let midpoint = |positions: &mut Vec<[f32; 3]>,
                        a: usize,
                        b: usize,
                        cache: &mut std::collections::HashMap<(usize, usize), usize>|
         -> usize {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&idx) = cache.get(&key) {
                return idx;
            }
            let pa = positions[a];
            let pb = positions[b];
            let mx = (pa[0] + pb[0]) * 0.5;
            let my = (pa[1] + pb[1]) * 0.5;
            let mz = (pa[2] + pb[2]) * 0.5;
            let l = (mx * mx + my * my + mz * mz).sqrt();
            let idx = positions.len();
            positions.push([mx / l, my / l, mz / l]);
            cache.insert(key, idx);
            idx
        };

        for tri in &triangles {
            let a = tri[0];
            let b = tri[1];
            let c = tri[2];
            let ab = midpoint(&mut positions, a, b, &mut mid_cache);
            let bc = midpoint(&mut positions, b, c, &mut mid_cache);
            let ca = midpoint(&mut positions, c, a, &mut mid_cache);
            new_triangles.push([a, ab, ca]);
            new_triangles.push([b, bc, ab]);
            new_triangles.push([c, ca, bc]);
            new_triangles.push([ab, bc, ca]);
        }
        triangles = new_triangles;
    }

    let verts: Vec<Vertex> = positions
        .iter()
        .map(|&p| Vertex {
            position: p,
            normal: p, // unit sphere: position = normal
            colour: white,
            uv: [0.0, 0.0],
            tangent: [0.0, 0.0, 0.0, 1.0],
        })
        .collect();

    let idx: Vec<u32> = triangles
        .iter()
        .flat_map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
        .collect();

    (verts, idx)
}
