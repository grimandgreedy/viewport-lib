//! Round primitives: spheres, the icosphere, hemispheres, ellipsoids and capsules.

use viewport_lib_types::data::mesh::MeshData;

/// UV sphere centred at the origin.
///
/// `radius` : sphere radius.
/// `sectors` : longitude subdivisions (minimum 3).
/// `stacks` : latitude subdivisions (minimum 2).
pub fn sphere(radius: f32, sectors: u32, stacks: u32) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(2);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let sector_step = 2.0 * std::f32::consts::PI / sectors as f32;
    let stack_step = std::f32::consts::PI / stacks as f32;

    for i in 0..=stacks {
        let stack_angle = std::f32::consts::FRAC_PI_2 - i as f32 * stack_step;
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();

        for j in 0..=sectors {
            let sector_angle = j as f32 * sector_step;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();
            positions.push([x, y, z]);
            normals.push([x / radius, y / radius, z / radius]);
            uvs.push([j as f32 / sectors as f32, i as f32 / stacks as f32]);
        }
    }

    for i in 0..stacks {
        let k1 = i * (sectors + 1);
        let k2 = k1 + sectors + 1;
        for j in 0..sectors {
            if i != 0 {
                indices.push(k1 + j);
                indices.push(k2 + j);
                indices.push(k1 + j + 1);
            }
            if i != stacks - 1 {
                indices.push(k1 + j + 1);
                indices.push(k2 + j);
                indices.push(k2 + j + 1);
            }
        }
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Capsule (cylinder body with hemispherical caps) centred at the origin, axis along Z.
///
/// `radius` : sphere cap radius. `height` : total height (clamped so body >= 0).
/// `sectors` : longitude subdivisions (minimum 3). `stacks` : latitude subdivisions (minimum 2).
pub fn capsule(radius: f32, height: f32, sectors: u32, stacks: u32) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(2);
    let body_height = (height - 2.0 * radius).max(0.0);
    let half_body = body_height / 2.0;
    let hemi_stacks = (stacks / 2).max(1);
    let cols = sectors + 1;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Top hemisphere (tip at i=0, equator at i=hemi_stacks, offset center +half_body)
    for i in 0..=hemi_stacks {
        let phi = std::f32::consts::FRAC_PI_2 * (1.0 - i as f32 / hemi_stacks as f32);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for j in 0..=sectors {
            let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
            let nx = cos_phi * theta.cos();
            let ny = cos_phi * theta.sin();
            positions.push([radius * nx, radius * ny, half_body + radius * sin_phi]);
            normals.push([nx, ny, sin_phi]);
        }
    }

    // Bottom hemisphere (equator at i=0, tip at i=hemi_stacks, offset center -half_body)
    let bottom_off = (hemi_stacks + 1) as u32;
    for i in 0..=hemi_stacks {
        let phi = -std::f32::consts::FRAC_PI_2 * i as f32 / hemi_stacks as f32;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for j in 0..=sectors {
            let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
            let nx = cos_phi * theta.cos();
            let ny = cos_phi * theta.sin();
            positions.push([radius * nx, radius * ny, -half_body + radius * sin_phi]);
            normals.push([nx, ny, sin_phi]);
        }
    }

    // Top hemisphere quads (skip degenerate upper triangle at pole)
    for i in 0..hemi_stacks {
        let k1 = i * cols;
        let k2 = k1 + cols;
        for j in 0..sectors {
            if i != 0 {
                indices.extend_from_slice(&[k1 + j, k2 + j, k1 + j + 1]);
            }
            indices.extend_from_slice(&[k1 + j + 1, k2 + j, k2 + j + 1]);
        }
    }

    // Body strip connecting the two equators
    if body_height > 1e-6 {
        let k1 = hemi_stacks * cols;
        let k2 = bottom_off * cols;
        for j in 0..sectors {
            indices.extend_from_slice(&[
                k1 + j,
                k2 + j,
                k1 + j + 1,
                k1 + j + 1,
                k2 + j,
                k2 + j + 1,
            ]);
        }
    }

    // Bottom hemisphere quads (skip degenerate lower triangle at pole)
    for i in 0..hemi_stacks {
        let k1 = (bottom_off + i) * cols;
        let k2 = k1 + cols;
        for j in 0..sectors {
            indices.extend_from_slice(&[k1 + j, k2 + j, k1 + j + 1]);
            if i != hemi_stacks - 1 {
                indices.extend_from_slice(&[k1 + j + 1, k2 + j, k2 + j + 1]);
            }
        }
    }

    // V mapped proportionally to Z so UVs flow smoothly from bottom tip (v=0) to top tip (v=1).
    let total_h = 2.0 * radius + body_height;
    let z_min = -(half_body + radius);
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    for i in 0..=hemi_stacks {
        let phi = std::f32::consts::FRAC_PI_2 * (1.0 - i as f32 / hemi_stacks as f32);
        let z = half_body + radius * phi.sin();
        let v = if total_h > 0.0 {
            (z - z_min) / total_h
        } else {
            1.0
        };
        for j in 0..=sectors {
            uvs.push([j as f32 / sectors as f32, v]);
        }
    }
    for i in 0..=hemi_stacks {
        let phi = -std::f32::consts::FRAC_PI_2 * i as f32 / hemi_stacks as f32;
        let z = -half_body + radius * phi.sin();
        let v = if total_h > 0.0 {
            (z - z_min) / total_h
        } else {
            0.0
        };
        for j in 0..=sectors {
            uvs.push([j as f32 / sectors as f32, v]);
        }
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Icosphere centred at the origin (better tessellation than UV sphere; no pole pinching).
///
/// `radius` : sphere radius. `subdivisions` : refinement level (0 = raw icosahedron, 20 faces).
pub fn icosphere(radius: f32, subdivisions: u32) -> MeshData {
    let phi = (1.0 + 5.0f32.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = 1.0 / norm;
    let b = phi / norm;

    let mut verts: Vec<[f32; 3]> = vec![
        [-a, b, 0.0],
        [a, b, 0.0],
        [-a, -b, 0.0],
        [a, -b, 0.0],
        [0.0, -a, b],
        [0.0, a, b],
        [0.0, -a, -b],
        [0.0, a, -b],
        [b, 0.0, -a],
        [b, 0.0, a],
        [-b, 0.0, -a],
        [-b, 0.0, a],
    ];
    let mut faces: Vec<[u32; 3]> = vec![
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

    for _ in 0..subdivisions {
        let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
        let mut cache: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        for &[va, vb, vc] in &faces {
            let mab = ico_midpoint(&mut verts, &mut cache, va, vb);
            let mbc = ico_midpoint(&mut verts, &mut cache, vb, vc);
            let mca = ico_midpoint(&mut verts, &mut cache, vc, va);
            new_faces.push([va, mab, mca]);
            new_faces.push([vb, mbc, mab]);
            new_faces.push([vc, mca, mbc]);
            new_faces.push([mab, mbc, mca]);
        }
        faces = new_faces;
    }

    let normals: Vec<[f32; 3]> = verts.clone();
    let positions: Vec<[f32; 3]> = verts
        .iter()
        .map(|v| [v[0] * radius, v[1] * radius, v[2] * radius])
        .collect();
    let indices: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();

    // Spherical UV projection. Triangles that span the antimeridian will have a
    // seam artefact (unavoidable without duplicating vertices at the seam)
    let uvs: Vec<[f32; 2]> = verts
        .iter()
        .map(|p| {
            let u = (p[2].atan2(p[0]) + std::f32::consts::PI) / std::f32::consts::TAU;
            let lat = p[1].clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
            [u, lat]
        })
        .collect();

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

fn ico_midpoint(
    verts: &mut Vec<[f32; 3]>,
    cache: &mut std::collections::HashMap<u64, u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = if a < b {
        (a as u64) << 32 | b as u64
    } else {
        (b as u64) << 32 | a as u64
    };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let va = verts[a as usize];
    let vb = verts[b as usize];
    let mx = (va[0] + vb[0]) * 0.5;
    let my = (va[1] + vb[1]) * 0.5;
    let mz = (va[2] + vb[2]) * 0.5;
    let len = (mx * mx + my * my + mz * mz).sqrt();
    let idx = verts.len() as u32;
    verts.push([mx / len, my / len, mz / len]);
    cache.insert(key, idx);
    idx
}

/// Hemisphere (upper half of a UV sphere) centred at the origin, dome facing +Z.
///
/// `radius` : sphere radius.
/// `sectors` : longitude subdivisions (minimum 3). `stacks` : latitude subdivisions (minimum 1).
pub fn hemisphere(radius: f32, sectors: u32, stacks: u32) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(1);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=stacks {
        let phi = std::f32::consts::FRAC_PI_2 * (1.0 - i as f32 / stacks as f32);
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for j in 0..=sectors {
            let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
            let nx = cos_phi * theta.cos();
            let ny = cos_phi * theta.sin();
            positions.push([radius * nx, radius * ny, radius * sin_phi]);
            normals.push([nx, ny, sin_phi]);
        }
    }

    let cols = sectors + 1;
    for i in 0..stacks {
        let k1 = i * cols;
        let k2 = k1 + cols;
        for j in 0..sectors {
            if i != 0 {
                indices.extend_from_slice(&[k1 + j, k2 + j, k1 + j + 1]);
            }
            indices.extend_from_slice(&[k1 + j + 1, k2 + j, k2 + j + 1]);
        }
    }

    // Equator disk cap (faces -Z)
    let center = positions.len() as u32;
    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, 0.0, -1.0]);
    let rim_start = positions.len() as u32;
    for j in 0..sectors {
        let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
        positions.push([radius * theta.cos(), radius * theta.sin(), 0.0]);
        normals.push([0.0, 0.0, -1.0]);
    }
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[center, rim_start + next, rim_start + j]);
    }

    let mut uvs: Vec<[f32; 2]> = Vec::new();
    for i in 0..=stacks {
        for j in 0..=sectors {
            uvs.push([j as f32 / sectors as f32, i as f32 / stacks as f32]);
        }
    }
    uvs.push([0.5, 0.5]); // cap center
    for j in 0..sectors {
        let a = j as f32 * std::f32::consts::TAU / sectors as f32;
        uvs.push([0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Ellipsoid centred at the origin.
///
/// `rx`, `ry`, `rz` : semi-axes along X, Y, Z.
/// `sectors` : longitude subdivisions (minimum 3). `stacks` : latitude subdivisions (minimum 2).
pub fn ellipsoid(rx: f32, ry: f32, rz: f32, sectors: u32, stacks: u32) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(2);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let sector_step = std::f32::consts::TAU / sectors as f32;
    let stack_step = std::f32::consts::PI / stacks as f32;

    for i in 0..=stacks {
        let stack_angle = std::f32::consts::FRAC_PI_2 - i as f32 * stack_step;
        let cos_sa = stack_angle.cos();
        let sin_sa = stack_angle.sin();

        for j in 0..=sectors {
            let sector_angle = j as f32 * sector_step;
            let cos_se = sector_angle.cos();
            let sin_se = sector_angle.sin();

            let x = rx * cos_sa * cos_se;
            let y = ry * sin_sa;
            let z = rz * cos_sa * sin_se;
            positions.push([x, y, z]);

            // Gradient of the implicit ellipsoid equation gives outward normal direction.
            let nx = x / (rx * rx);
            let ny = y / (ry * ry);
            let nz = z / (rz * rz);
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            normals.push(if len > 0.0 {
                [nx / len, ny / len, nz / len]
            } else {
                [0.0, 1.0, 0.0]
            });
        }
    }

    for i in 0..stacks {
        let k1 = i * (sectors + 1);
        let k2 = k1 + sectors + 1;
        for j in 0..sectors {
            if i != 0 {
                indices.extend_from_slice(&[k1 + j, k1 + j + 1, k2 + j]);
            }
            if i != stacks - 1 {
                indices.extend_from_slice(&[k1 + j + 1, k2 + j + 1, k2 + j]);
            }
        }
    }

    let mut uvs: Vec<[f32; 2]> = Vec::new();
    for i in 0..=stacks {
        for j in 0..=sectors {
            uvs.push([j as f32 / sectors as f32, i as f32 / stacks as f32]);
        }
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::tests::{assert_mesh_invariants, assert_normals_unit_length};

    // ---- sphere ----

    #[test]
    fn sphere_vertices_at_radius() {
        let r = 2.5;
        let m = sphere(r, 16, 8);
        for (i, p) in m.positions.iter().enumerate() {
            let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(
                (dist - r).abs() < 1e-4,
                "sphere vertex[{i}] at distance {dist}, expected {r}"
            );
        }
    }

    #[test]
    fn sphere_vertex_count() {
        let s = 16u32;
        let t = 8u32;
        let m = sphere(1.0, s, t);
        assert_eq!(m.positions.len(), ((t + 1) * (s + 1)) as usize);
    }

    #[test]
    fn sphere_normals_unit_length() {
        assert_normals_unit_length("sphere", &sphere(1.0, 16, 8));
    }

    #[test]
    fn sphere_has_uvs() {
        assert!(sphere(1.0, 16, 8).uvs.is_some());
    }

    #[test]
    fn sphere_minimum_sectors_clamped() {
        let m = sphere(1.0, 1, 1); // should clamp to 3 sectors, 2 stacks
        assert_mesh_invariants("sphere_min", &m);
        assert_eq!(m.positions.len(), ((2 + 1) * (3 + 1)) as usize);
    }

    // ---- icosphere ----

    #[test]
    fn icosphere_subdivision_0_counts() {
        let m = icosphere(1.0, 0);
        assert_eq!(m.positions.len(), 12); // icosahedron
        assert_eq!(m.indices.len(), 60); // 20 faces * 3
    }

    #[test]
    fn icosphere_subdivision_1_counts() {
        let m = icosphere(1.0, 1);
        assert_eq!(m.positions.len(), 42); // 12 + 30 edge midpoints
        assert_eq!(m.indices.len(), 240); // 80 faces * 3
    }

    #[test]
    fn icosphere_vertices_at_radius() {
        let r = 3.0;
        let m = icosphere(r, 2);
        for (i, p) in m.positions.iter().enumerate() {
            let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(
                (dist - r).abs() < 1e-4,
                "icosphere vertex[{i}] at distance {dist}, expected {r}"
            );
        }
    }

    #[test]
    fn icosphere_normals_unit_length() {
        assert_normals_unit_length("icosphere", &icosphere(1.0, 2));
    }

    // ---- hemisphere ----

    #[test]
    fn hemisphere_all_dome_vertices_non_negative_z() {
        let m = hemisphere(1.0, 12, 6);
        // Dome vertices (before cap) should have Z >= 0
        let dome_count = (6 + 1) * (12 + 1);
        for (i, p) in m.positions.iter().take(dome_count as usize).enumerate() {
            assert!(
                p[2] >= -1e-5,
                "hemisphere dome vertex[{i}] has Z = {}",
                p[2]
            );
        }
    }

    // ---- ellipsoid ----

    #[test]
    fn ellipsoid_vertices_on_surface() {
        let (rx, ry, rz) = (2.0, 1.0, 3.0);
        let m = ellipsoid(rx, ry, rz, 16, 8);
        for (i, p) in m.positions.iter().enumerate() {
            let val = (p[0] / rx).powi(2) + (p[1] / ry).powi(2) + (p[2] / rz).powi(2);
            assert!(
                (val - 1.0).abs() < 1e-3,
                "ellipsoid vertex[{i}] has implicit value {val}, expected ~1.0"
            );
        }
    }

    #[test]
    fn ellipsoid_normals_unit_length() {
        assert_normals_unit_length("ellipsoid", &ellipsoid(2.0, 1.0, 3.0, 16, 8));
    }

    // ---- capsule ----

    #[test]
    fn capsule_all_vertices_within_bounding_sphere() {
        let r = 0.5;
        let h = 3.0;
        let m = capsule(r, h, 12, 6);
        let half_body = (h - 2.0 * r).max(0.0) / 2.0;
        for (i, p) in m.positions.iter().enumerate() {
            // Each vertex should be within radius of the closest point on the capsule axis
            let axis_z = p[2].clamp(-half_body, half_body);
            let dz = p[2] - axis_z;
            let dist = (p[0] * p[0] + p[1] * p[1] + dz * dz).sqrt();
            assert!(
                dist <= r + 1e-3,
                "capsule vertex[{i}] at dist {dist} from axis, expected <= {r}"
            );
        }
    }

    #[test]
    fn capsule_zero_body_height() {
        // When height <= 2*radius, body height is 0 (pure sphere shape)
        let r = 1.0;
        let m = capsule(r, 2.0 * r, 12, 6);
        assert_mesh_invariants("capsule_zero_body", &m);
    }
}
