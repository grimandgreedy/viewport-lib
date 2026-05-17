use crate::resources::MeshData;

/// Unit cube (side length 1, centered at the origin).
///
/// `size` scales all three axes uniformly.
pub fn cube(size: f32) -> MeshData {
    let h = size / 2.0;

    // 6 faces × 4 vertices each = 24 vertices
    #[rustfmt::skip]
    let positions: Vec<[f32; 3]> = vec![
        // +Z
        [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],
        // -Z
        [ h, -h, -h], [-h, -h, -h], [-h,  h, -h], [ h,  h, -h],
        // +Y
        [-h,  h,  h], [ h,  h,  h], [ h,  h, -h], [-h,  h, -h],
        // -Y
        [-h, -h, -h], [ h, -h, -h], [ h, -h,  h], [-h, -h,  h],
        // +X
        [ h, -h,  h], [ h, -h, -h], [ h,  h, -h], [ h,  h,  h],
        // -X
        [-h, -h, -h], [-h, -h,  h], [-h,  h,  h], [-h,  h, -h],
    ];

    // Build per-face flat normals
    let face_normals: [[f32; 3]; 6] = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    let normals: Vec<[f32; 3]> = face_normals
        .iter()
        .flat_map(|n| std::iter::repeat(*n).take(4))
        .collect();

    // 6 faces × 2 triangles × 3 indices
    let indices: Vec<u32> = (0..6u32)
        .flat_map(|f| {
            let b = f * 4;
            [b, b + 1, b + 2, b, b + 2, b + 3]
        })
        .collect();

    // Each face gets [0,0] [1,0] [1,1] [0,1] UVs.
    let uvs: Vec<[f32; 2]> = (0..6)
        .flat_map(|_| [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        .collect();

    MeshData {
        positions,
        normals,
        indices,
        uvs: Some(uvs),
        ..MeshData::default()
    }
}

/// UV sphere centered at the origin.
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

    MeshData {
        positions,
        normals,
        indices,
        uvs: Some(uvs),
        ..MeshData::default()
    }
}

/// Flat XY plane centered at the origin (Z-up world: this is the ground plane).
///
/// `width` : extent along X. `depth` : extent along Y. Normal points +Z.
pub fn plane(width: f32, depth: f32) -> MeshData {
    let hw = width / 2.0;
    let hd = depth / 2.0;

    let positions = vec![
        [-hw, -hd, 0.0],
        [hw, -hd, 0.0],
        [hw, hd, 0.0],
        [-hw, hd, 0.0],
    ];
    let normals = vec![[0.0, 0.0, 1.0]; 4];
    let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let indices = vec![0, 1, 2, 0, 2, 3];

    MeshData {
        positions,
        normals,
        indices,
        uvs: Some(uvs),
        ..MeshData::default()
    }
}

/// Cylinder centered at the origin, axis along Z.
///
/// `radius` : circle radius. `height` : total height. `sectors` : circumference subdivisions (minimum 3).
pub fn cylinder(radius: f32, height: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let half_h = height / 2.0;
    let step = 2.0 * std::f32::consts::PI / sectors as f32;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Side vertices: two rings (bottom then top)
    for &z in &[-half_h, half_h] {
        for j in 0..sectors {
            let angle = j as f32 * step;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            positions.push([x, y, z]);
            normals.push([angle.cos(), angle.sin(), 0.0]);
        }
    }

    // Side faces
    for j in 0..sectors {
        let b = j;
        let next = (j + 1) % sectors;
        let t = j + sectors;
        let t_next = next + sectors;
        indices.extend_from_slice(&[b, next, t_next, b, t_next, t]);
    }

    // Cap centers
    let bottom_center = positions.len() as u32;
    positions.push([0.0, 0.0, -half_h]);
    normals.push([0.0, 0.0, -1.0]);

    let top_center = positions.len() as u32;
    positions.push([0.0, 0.0, half_h]);
    normals.push([0.0, 0.0, 1.0]);

    // Cap rim vertices (separate so normals point up/down)
    let bottom_rim_start = positions.len() as u32;
    for j in 0..sectors {
        let angle = j as f32 * step;
        positions.push([radius * angle.cos(), radius * angle.sin(), -half_h]);
        normals.push([0.0, 0.0, -1.0]);
    }

    let top_rim_start = positions.len() as u32;
    for j in 0..sectors {
        let angle = j as f32 * step;
        positions.push([radius * angle.cos(), radius * angle.sin(), half_h]);
        normals.push([0.0, 0.0, 1.0]);
    }

    // Cap faces
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        // Bottom
        indices.extend_from_slice(&[bottom_center, bottom_rim_start + next, bottom_rim_start + j]);
        // Top
        indices.extend_from_slice(&[top_center, top_rim_start + j, top_rim_start + next]);
    }

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Non-uniform box (cuboid) centered at the origin.
///
/// `width` : X extent. `height` : Y extent. `depth` : Z extent.
pub fn cuboid(width: f32, height: f32, depth: f32) -> MeshData {
    let hw = width / 2.0;
    let hh = height / 2.0;
    let hd = depth / 2.0;

    #[rustfmt::skip]
    let positions: Vec<[f32; 3]> = vec![
        // +Z
        [-hw, -hh,  hd], [ hw, -hh,  hd], [ hw,  hh,  hd], [-hw,  hh,  hd],
        // -Z
        [ hw, -hh, -hd], [-hw, -hh, -hd], [-hw,  hh, -hd], [ hw,  hh, -hd],
        // +Y
        [-hw,  hh,  hd], [ hw,  hh,  hd], [ hw,  hh, -hd], [-hw,  hh, -hd],
        // -Y
        [-hw, -hh, -hd], [ hw, -hh, -hd], [ hw, -hh,  hd], [-hw, -hh,  hd],
        // +X
        [ hw, -hh,  hd], [ hw, -hh, -hd], [ hw,  hh, -hd], [ hw,  hh,  hd],
        // -X
        [-hw, -hh, -hd], [-hw, -hh,  hd], [-hw,  hh,  hd], [-hw,  hh, -hd],
    ];

    let face_normals: [[f32; 3]; 6] = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    let normals: Vec<[f32; 3]> = face_normals
        .iter()
        .flat_map(|n| std::iter::repeat(*n).take(4))
        .collect();

    let indices: Vec<u32> = (0..6u32)
        .flat_map(|f| {
            let b = f * 4;
            [b, b + 1, b + 2, b, b + 2, b + 3]
        })
        .collect();

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Cone with tip at +Z and base at −Z, centered at the origin.
///
/// `radius` : base radius. `height` : total height. `sectors` : circumference subdivisions (minimum 3).
pub fn cone(radius: f32, height: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let half_h = height / 2.0;
    let step = 2.0 * std::f32::consts::PI / sectors as f32;

    // Side normal components: outward radial and upward Z.
    let hyp = (radius * radius + height * height).sqrt();
    let nz = radius / hyp;
    let nr = height / hyp;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Side faces : one duplicated tip vertex per sector so each has the bisector normal.
    for j in 0..sectors {
        let a0 = j as f32 * step;
        let a1 = (j + 1) as f32 * step;
        let amid = (a0 + a1) * 0.5;
        let base = positions.len() as u32;

        positions.push([0.0, 0.0, half_h]);
        normals.push([nr * amid.cos(), nr * amid.sin(), nz]);

        positions.push([radius * a0.cos(), radius * a0.sin(), -half_h]);
        normals.push([nr * a0.cos(), nr * a0.sin(), nz]);

        positions.push([radius * a1.cos(), radius * a1.sin(), -half_h]);
        normals.push([nr * a1.cos(), nr * a1.sin(), nz]);

        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Bottom cap
    let bottom_center = positions.len() as u32;
    positions.push([0.0, 0.0, -half_h]);
    normals.push([0.0, 0.0, -1.0]);

    let rim_start = positions.len() as u32;
    for j in 0..sectors {
        let a = j as f32 * step;
        positions.push([radius * a.cos(), radius * a.sin(), -half_h]);
        normals.push([0.0, 0.0, -1.0]);
    }
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[bottom_center, rim_start + next, rim_start + j]);
    }

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Capsule (cylinder body with hemispherical caps) centered at the origin, axis along Z.
///
/// `radius` : sphere cap radius. `height` : total height (clamped so body ≥ 0).
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

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Torus centered at the origin, lying in the XY plane.
///
/// `major_radius` : distance from the torus centre to the tube centre.
/// `minor_radius` : radius of the tube.
/// `sectors` : segments around the tube (minimum 3).
/// `stacks` : segments around the torus ring (minimum 3).
pub fn torus(major_radius: f32, minor_radius: f32, sectors: u32, stacks: u32) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(3);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=stacks {
        let phi = i as f32 * std::f32::consts::TAU / stacks as f32;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let ring_x = major_radius * cos_phi;
        let ring_y = major_radius * sin_phi;

        for j in 0..=sectors {
            let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            // Tube cross-section normal in XY-plane torus.
            let nx = cos_phi * cos_theta;
            let ny = sin_phi * cos_theta;
            let nz = sin_theta;

            positions.push([
                ring_x + minor_radius * nx,
                ring_y + minor_radius * ny,
                minor_radius * nz,
            ]);
            normals.push([nx, ny, nz]);
            uvs.push([j as f32 / sectors as f32, i as f32 / stacks as f32]);
        }
    }

    let cols = sectors + 1;
    for i in 0..stacks {
        let k1 = i * cols;
        let k2 = k1 + cols;
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

    MeshData {
        positions,
        normals,
        indices,
        uvs: Some(uvs),
        ..MeshData::default()
    }
}

/// Icosphere centered at the origin (better tessellation than UV sphere; no pole pinching).
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

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
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

/// Arrow along +Z, centered at the origin (total length 1).
///
/// `shaft_radius` : cylinder shaft radius.
/// `head_radius` : cone head base radius.
/// `head_fraction` : fraction of total length occupied by the cone head (clamped to 0.1-0.9).
/// `sectors` : circumference subdivisions (minimum 3).
pub fn arrow(shaft_radius: f32, head_radius: f32, head_fraction: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let head_fraction = head_fraction.clamp(0.1, 0.9);
    let step = std::f32::consts::TAU / sectors as f32;

    let shaft_bot: f32 = -0.5;
    let shaft_top: f32 = 0.5 - head_fraction;
    let head_bot: f32 = shaft_top;
    let head_top: f32 = 0.5;
    let head_h = head_fraction;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Shaft side rings (bottom then top)
    for &z in &[shaft_bot, shaft_top] {
        for j in 0..sectors {
            let a = j as f32 * step;
            positions.push([shaft_radius * a.cos(), shaft_radius * a.sin(), z]);
            normals.push([a.cos(), a.sin(), 0.0]);
        }
    }
    for j in 0..sectors {
        let next = (j + 1) % sectors;
        let t = j + sectors;
        let t_next = next + sectors;
        indices.extend_from_slice(&[j, next, t_next, j, t_next, t]);
    }

    // Shaft bottom cap
    let sb_center = positions.len() as u32;
    positions.push([0.0, 0.0, shaft_bot]);
    normals.push([0.0, 0.0, -1.0]);
    let sb_rim = positions.len() as u32;
    for j in 0..sectors {
        let a = j as f32 * step;
        positions.push([shaft_radius * a.cos(), shaft_radius * a.sin(), shaft_bot]);
        normals.push([0.0, 0.0, -1.0]);
    }
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[sb_center, sb_rim + next, sb_rim + j]);
    }

    // Cone head side (one duplicated tip per sector)
    let cone_hyp = (head_radius * head_radius + head_h * head_h).sqrt();
    let cnz = head_radius / cone_hyp;
    let cnr = head_h / cone_hyp;
    for j in 0..sectors {
        let a0 = j as f32 * step;
        let a1 = (j + 1) as f32 * step;
        let amid = (a0 + a1) * 0.5;
        let base = positions.len() as u32;

        positions.push([0.0, 0.0, head_top]);
        normals.push([cnr * amid.cos(), cnr * amid.sin(), cnz]);

        positions.push([head_radius * a0.cos(), head_radius * a0.sin(), head_bot]);
        normals.push([cnr * a0.cos(), cnr * a0.sin(), cnz]);

        positions.push([head_radius * a1.cos(), head_radius * a1.sin(), head_bot]);
        normals.push([cnr * a1.cos(), cnr * a1.sin(), cnz]);

        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Cone base cap
    let hb_center = positions.len() as u32;
    positions.push([0.0, 0.0, head_bot]);
    normals.push([0.0, 0.0, -1.0]);
    let hb_rim = positions.len() as u32;
    for j in 0..sectors {
        let a = j as f32 * step;
        positions.push([head_radius * a.cos(), head_radius * a.sin(), head_bot]);
        normals.push([0.0, 0.0, -1.0]);
    }
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[hb_center, hb_rim + next, hb_rim + j]);
    }

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Flat disk in the XY plane, centered at the origin, normal pointing +Z.
///
/// `radius` : disk radius. `sectors` : circumference subdivisions (minimum 3).
pub fn disk(radius: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let step = std::f32::consts::TAU / sectors as f32;

    let mut positions: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 1.0]];
    let mut indices: Vec<u32> = Vec::new();

    for j in 0..sectors {
        let a = j as f32 * step;
        positions.push([radius * a.cos(), radius * a.sin(), 0.0]);
        normals.push([0.0, 0.0, 1.0]);
    }

    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[0, j + 1, next + 1]);
    }

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Camera frustum mesh for visualization.
///
/// The camera sits at the origin looking along −Z.
/// `fov_y` : vertical field of view in radians. `aspect` : width / height.
/// `near`, `far` : clip plane distances (positive values).
pub fn frustum(fov_y: f32, aspect: f32, near: f32, far: f32) -> MeshData {
    let half_h_n = near * (fov_y * 0.5).tan();
    let half_w_n = half_h_n * aspect;
    let half_h_f = far * (fov_y * 0.5).tan();
    let half_w_f = half_h_f * aspect;

    // 8 corners
    let nbl = [-half_w_n, -half_h_n, -near];
    let nbr = [half_w_n, -half_h_n, -near];
    let ntr = [half_w_n, half_h_n, -near];
    let ntl = [-half_w_n, half_h_n, -near];
    let fbl = [-half_w_f, -half_h_f, -far];
    let fbr = [half_w_f, -half_h_f, -far];
    let ftr = [half_w_f, half_h_f, -far];
    let ftl = [-half_w_f, half_h_f, -far];

    // 6 faces as (v0, v1, v2, v3); normal from (v1-v0) × (v3-v0)
    let face_quads: [[[f32; 3]; 4]; 6] = [
        [ntl, ntr, nbr, nbl], // near
        [fbl, fbr, ftr, ftl], // far
        [ntl, ftl, ftr, ntr], // top
        [nbr, fbr, fbl, nbl], // bottom
        [ntr, ftr, fbr, nbr], // right
        [nbl, fbl, ftl, ntl], // left
    ];

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for quad in &face_quads {
        let [v0, v1, _, v3] = quad;
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]];
        let nr = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = (nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]).sqrt();
        let n = if len > 0.0 {
            [nr[0] / len, nr[1] / len, nr[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        };

        let base = positions.len() as u32;
        for v in quad {
            positions.push(*v);
            normals.push(n);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Hemisphere (upper half of a UV sphere) centered at the origin, dome facing +Z.
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

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Flat ring (annulus) in the XY plane, centered at the origin, normal pointing +Z.
///
/// `inner_radius` : inner edge. `outer_radius` : outer edge. `sectors` : circumference subdivisions (minimum 3).
pub fn ring(inner_radius: f32, outer_radius: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let step = std::f32::consts::TAU / sectors as f32;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Interleaved inner/outer pairs: [inner_0, outer_0, inner_1, outer_1, ...]
    for j in 0..=sectors {
        let a = j as f32 * step;
        let cos_a = a.cos();
        let sin_a = a.sin();
        positions.push([inner_radius * cos_a, inner_radius * sin_a, 0.0]);
        normals.push([0.0, 0.0, 1.0]);
        positions.push([outer_radius * cos_a, outer_radius * sin_a, 0.0]);
        normals.push([0.0, 0.0, 1.0]);
    }

    for j in 0..sectors as u32 {
        let i0 = j * 2;
        let o0 = i0 + 1;
        let i1 = i0 + 2;
        let o1 = i0 + 3;
        indices.extend_from_slice(&[i0, o0, i1, i1, o0, o1]);
    }

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Ellipsoid centered at the origin.
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

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Helical spring centered at the origin, axis along Z.
///
/// `radius` : distance from Z axis to tube centre.
/// `coil_radius` : cross-section tube radius.
/// `turns` : number of complete coil turns.
/// `sectors` : tube cross-section subdivisions (minimum 3).
pub fn spring(radius: f32, coil_radius: f32, turns: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let pitch = 2.5 * coil_radius; // height per turn; keeps coils from overlapping
    let height = turns * pitch;
    let n_segs = (turns * 16.0).ceil() as u32;
    let total_t = std::f32::consts::TAU * turns;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for seg in 0..=n_segs {
        let t = seg as f32 / n_segs as f32 * total_t;

        let cx = radius * t.cos();
        let cy = radius * t.sin();
        let cz = t / std::f32::consts::TAU * pitch - height * 0.5;

        // Helix tangent
        let dtx = -radius * t.sin();
        let dty = radius * t.cos();
        let dtz = pitch / std::f32::consts::TAU;
        let dt_len = (dtx * dtx + dty * dty + dtz * dtz).sqrt();
        let (tx, ty, tz) = (dtx / dt_len, dty / dt_len, dtz / dt_len);

        // Principal normal: inward radial toward Z axis
        let pnx = -t.cos();
        let pny = -t.sin();
        let pnz = 0.0f32;

        // Binormal = T × principal_normal
        let bx = ty * pnz - tz * pny;
        let by = tz * pnx - tx * pnz;
        let bz = tx * pny - ty * pnx;

        for sec in 0..=sectors {
            let phi = sec as f32 * std::f32::consts::TAU / sectors as f32;
            let cp = phi.cos();
            let sp = phi.sin();

            let on_x = cp * pnx + sp * bx;
            let on_y = cp * pny + sp * by;
            let on_z = cp * pnz + sp * bz;

            positions.push([
                cx + coil_radius * on_x,
                cy + coil_radius * on_y,
                cz + coil_radius * on_z,
            ]);
            normals.push([on_x, on_y, on_z]);
        }
    }

    let cols = sectors + 1;
    for seg in 0..n_segs {
        let k1 = seg * cols;
        let k2 = k1 + cols;
        for sec in 0..sectors {
            indices.extend_from_slice(&[
                k1 + sec,
                k1 + sec + 1,
                k2 + sec,
                k1 + sec + 1,
                k2 + sec + 1,
                k2 + sec,
            ]);
        }
    }

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

/// Subdivided plane in the XY plane, centered at the origin, normal pointing +Z.
///
/// `width` : X extent. `depth` : Y extent.
/// `cols` : column subdivisions (minimum 1). `rows` : row subdivisions (minimum 1).
pub fn grid_plane(width: f32, depth: f32, cols: u32, rows: u32) -> MeshData {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let hw = width * 0.5;
    let hd = depth * 0.5;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for row in 0..=rows {
        let y = -hd + row as f32 / rows as f32 * depth;
        for col in 0..=cols {
            let x = -hw + col as f32 / cols as f32 * width;
            positions.push([x, y, 0.0]);
            normals.push([0.0, 0.0, 1.0]);
        }
    }

    let v_cols = cols + 1;
    for row in 0..rows {
        for col in 0..cols {
            let tl = row * v_cols + col;
            let tr = tl + 1;
            let bl = tl + v_cols;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, tr, bl, tr, br, bl]);
        }
    }

    MeshData {
        positions,
        normals,
        indices,
        ..MeshData::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_triangle_winding_matches_normals(mesh: &MeshData) {
        for tri in mesh.indices.chunks_exact(3) {
            let ia = tri[0] as usize;
            let ib = tri[1] as usize;
            let ic = tri[2] as usize;

            let a = glam::Vec3::from_array(mesh.positions[ia]);
            let b = glam::Vec3::from_array(mesh.positions[ib]);
            let c = glam::Vec3::from_array(mesh.positions[ic]);
            let face_normal = (b - a).cross(c - a);
            if face_normal.length_squared() <= 1e-12 {
                continue;
            }

            let avg_vertex_normal = glam::Vec3::from_array(mesh.normals[ia])
                + glam::Vec3::from_array(mesh.normals[ib])
                + glam::Vec3::from_array(mesh.normals[ic]);

            assert!(
                face_normal.dot(avg_vertex_normal) > 0.0,
                "triangle winding does not match vertex normals: {tri:?}"
            );
        }
    }

    /// Validates structural invariants that every generated mesh must satisfy.
    fn assert_mesh_invariants(name: &str, mesh: &MeshData) {
        assert!(
            !mesh.positions.is_empty(),
            "{name}: positions must not be empty"
        );
        assert_eq!(
            mesh.positions.len(),
            mesh.normals.len(),
            "{name}: positions and normals length mismatch"
        );
        assert_eq!(
            mesh.indices.len() % 3,
            0,
            "{name}: index count must be a multiple of 3"
        );
        let n = mesh.positions.len() as u32;
        for (i, &idx) in mesh.indices.iter().enumerate() {
            assert!(idx < n, "{name}: index[{i}] = {idx} out of bounds (n={n})");
        }
        if let Some(ref uvs) = mesh.uvs {
            assert_eq!(
                uvs.len(),
                mesh.positions.len(),
                "{name}: uvs length mismatch"
            );
        }
        if let Some(ref tangents) = mesh.tangents {
            assert_eq!(
                tangents.len(),
                mesh.positions.len(),
                "{name}: tangents length mismatch"
            );
        }
    }

    /// Checks that all normals are unit length (within tolerance).
    fn assert_normals_unit_length(name: &str, mesh: &MeshData) {
        for (i, n) in mesh.normals.iter().enumerate() {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "{name}: normal[{i}] has length {len}"
            );
        }
    }

    /// Checks that all positions are within the given axis-aligned bounds.
    fn assert_positions_bounded(name: &str, mesh: &MeshData, half: [f32; 3]) {
        for (i, p) in mesh.positions.iter().enumerate() {
            for axis in 0..3 {
                assert!(
                    p[axis].abs() <= half[axis] + 1e-5,
                    "{name}: position[{i}][{axis}] = {} exceeds bound {}",
                    p[axis],
                    half[axis]
                );
            }
        }
    }

    #[test]
    fn generated_primitives_have_consistent_outward_winding() {
        let meshes = [
            ("cube", cube(1.0)),
            ("sphere", sphere(1.0, 24, 12)),
            ("plane", plane(1.0, 1.0)),
            ("cylinder", cylinder(1.0, 2.0, 24)),
            ("cuboid", cuboid(1.0, 1.5, 2.0)),
            ("cone", cone(1.0, 2.0, 24)),
            ("capsule", capsule(1.0, 3.0, 24, 12)),
            ("torus", torus(2.0, 0.5, 24, 24)),
            ("icosphere", icosphere(1.0, 2)),
            ("arrow", arrow(0.2, 0.4, 0.3, 24)),
            ("disk", disk(1.0, 24)),
            (
                "frustum",
                frustum(std::f32::consts::FRAC_PI_4, 1.5, 0.1, 2.0),
            ),
            ("hemisphere", hemisphere(1.0, 24, 12)),
            ("ring", ring(0.5, 1.0, 24)),
            ("ellipsoid", ellipsoid(1.0, 0.75, 1.25, 24, 12)),
            ("spring", spring(2.0, 0.25, 3.0, 16)),
            ("grid_plane", grid_plane(1.0, 1.0, 4, 4)),
        ];

        for (name, mesh) in &meshes {
            eprintln!("checking {name}");
            assert_triangle_winding_matches_normals(mesh);
        }
    }

    // ---- structural invariants for every primitive ----

    #[test]
    fn all_primitives_pass_mesh_invariants() {
        let meshes: Vec<(&str, MeshData)> = vec![
            ("cube", cube(1.0)),
            ("sphere", sphere(1.0, 16, 8)),
            ("plane", plane(2.0, 3.0)),
            ("cylinder", cylinder(1.0, 2.0, 16)),
            ("cuboid", cuboid(1.0, 2.0, 3.0)),
            ("cone", cone(1.0, 2.0, 16)),
            ("capsule", capsule(0.5, 2.0, 12, 6)),
            ("torus", torus(2.0, 0.5, 12, 12)),
            ("icosphere_0", icosphere(1.0, 0)),
            ("icosphere_2", icosphere(1.0, 2)),
            ("arrow", arrow(0.1, 0.3, 0.3, 12)),
            ("disk", disk(1.0, 12)),
            ("frustum", frustum(1.0, 1.5, 0.1, 10.0)),
            ("hemisphere", hemisphere(1.0, 12, 6)),
            ("ring", ring(0.5, 1.0, 12)),
            ("ellipsoid", ellipsoid(1.0, 0.5, 1.5, 12, 6)),
            ("spring", spring(1.0, 0.2, 2.0, 8)),
            ("grid_plane", grid_plane(1.0, 1.0, 4, 4)),
        ];
        for (name, mesh) in &meshes {
            assert_mesh_invariants(name, mesh);
        }
    }

    // ---- cube ----

    #[test]
    fn cube_vertex_and_index_counts() {
        let m = cube(1.0);
        assert_eq!(m.positions.len(), 24); // 6 faces * 4 verts
        assert_eq!(m.indices.len(), 36); // 6 faces * 2 tris * 3
    }

    #[test]
    fn cube_positions_bounded_by_half_size() {
        let size = 2.0;
        let m = cube(size);
        let h = size / 2.0;
        assert_positions_bounded("cube", &m, [h, h, h]);
    }

    #[test]
    fn cube_has_uvs() {
        let m = cube(1.0);
        assert!(m.uvs.is_some());
    }

    #[test]
    fn cube_normals_unit_length() {
        assert_normals_unit_length("cube", &cube(1.0));
    }

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

    // ---- plane ----

    #[test]
    fn plane_all_z_zero() {
        let m = plane(5.0, 3.0);
        for (i, p) in m.positions.iter().enumerate() {
            assert!(p[2].abs() < 1e-6, "plane vertex[{i}] has Z = {}", p[2]);
        }
    }

    #[test]
    fn plane_extents_match() {
        let w = 4.0;
        let d = 6.0;
        let m = plane(w, d);
        assert_positions_bounded("plane", &m, [w / 2.0, d / 2.0, 0.0]);
    }

    #[test]
    fn plane_vertex_count() {
        assert_eq!(plane(1.0, 1.0).positions.len(), 4);
        assert_eq!(plane(1.0, 1.0).indices.len(), 6);
    }

    // ---- cylinder ----

    #[test]
    fn cylinder_side_vertices_at_radius() {
        let r = 1.5;
        let m = cylinder(r, 3.0, 16);
        // Side vertices are first 2*sectors entries
        for (i, p) in m.positions.iter().take(32).enumerate() {
            let radial = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (radial - r).abs() < 1e-4,
                "cylinder side vertex[{i}] at radial dist {radial}, expected {r}"
            );
        }
    }

    #[test]
    fn cylinder_z_bounded() {
        let h = 4.0;
        let m = cylinder(1.0, h, 12);
        for (i, p) in m.positions.iter().enumerate() {
            assert!(
                p[2].abs() <= h / 2.0 + 1e-5,
                "cylinder vertex[{i}] Z = {} exceeds half height",
                p[2]
            );
        }
    }

    // ---- cuboid ----

    #[test]
    fn cuboid_positions_bounded() {
        let (w, h, d) = (2.0, 3.0, 4.0);
        let m = cuboid(w, h, d);
        assert_positions_bounded("cuboid", &m, [w / 2.0, h / 2.0, d / 2.0]);
        assert_eq!(m.positions.len(), 24);
        assert_eq!(m.indices.len(), 36);
    }

    // ---- cone ----

    #[test]
    fn cone_tip_at_positive_z() {
        let h = 3.0;
        let m = cone(1.0, h, 12);
        let tip_z = h / 2.0;
        let has_tip = m.positions.iter().any(|p| (p[2] - tip_z).abs() < 1e-5);
        assert!(has_tip, "cone should have a tip vertex at Z = {tip_z}");
    }

    #[test]
    fn cone_base_vertices_at_radius() {
        let r = 2.0;
        let h = 3.0;
        let m = cone(r, h, 16);
        let base_z = -h / 2.0;
        for (i, p) in m.positions.iter().enumerate() {
            if (p[2] - base_z).abs() < 1e-5 && p[0].abs() > 1e-5 {
                let radial = (p[0] * p[0] + p[1] * p[1]).sqrt();
                assert!(
                    (radial - r).abs() < 1e-4,
                    "cone base vertex[{i}] at radial {radial}, expected {r}"
                );
            }
        }
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

    // ---- torus ----

    #[test]
    fn torus_vertices_within_radial_bounds() {
        let major = 3.0;
        let minor = 0.5;
        let m = torus(major, minor, 12, 12);
        for (i, p) in m.positions.iter().enumerate() {
            let radial_xy = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                radial_xy >= major - minor - 1e-3 && radial_xy <= major + minor + 1e-3,
                "torus vertex[{i}] radial_xy = {radial_xy}, expected in [{}, {}]",
                major - minor,
                major + minor
            );
        }
    }

    #[test]
    fn torus_has_uvs() {
        assert!(torus(2.0, 0.5, 8, 8).uvs.is_some());
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

    // ---- arrow ----

    #[test]
    fn arrow_total_height_is_one() {
        let m = arrow(0.1, 0.3, 0.3, 12);
        let min_z = m
            .positions
            .iter()
            .map(|p| p[2])
            .fold(f32::INFINITY, f32::min);
        let max_z = m
            .positions
            .iter()
            .map(|p| p[2])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            ((max_z - min_z) - 1.0).abs() < 1e-4,
            "arrow total height = {}, expected 1.0",
            max_z - min_z
        );
    }

    #[test]
    fn arrow_head_fraction_clamped() {
        // head_fraction < 0.1 should clamp to 0.1
        let m = arrow(0.1, 0.3, 0.0, 12);
        assert_mesh_invariants("arrow_clamp_low", &m);
        // head_fraction > 0.9 should clamp to 0.9
        let m = arrow(0.1, 0.3, 1.0, 12);
        assert_mesh_invariants("arrow_clamp_high", &m);
    }

    // ---- disk ----

    #[test]
    fn disk_center_at_origin() {
        let m = disk(2.0, 12);
        assert!((m.positions[0][0]).abs() < 1e-6);
        assert!((m.positions[0][1]).abs() < 1e-6);
        assert!((m.positions[0][2]).abs() < 1e-6);
    }

    #[test]
    fn disk_rim_at_radius() {
        let r = 2.0;
        let m = disk(r, 16);
        for (i, p) in m.positions.iter().skip(1).enumerate() {
            let dist = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (dist - r).abs() < 1e-4,
                "disk rim vertex[{i}] at dist {dist}, expected {r}"
            );
            assert!(p[2].abs() < 1e-6, "disk vertex should be at Z=0");
        }
    }

    #[test]
    fn disk_vertex_count() {
        let sectors = 12u32;
        let m = disk(1.0, sectors);
        assert_eq!(m.positions.len(), (sectors + 1) as usize); // center + rim
    }

    // ---- frustum ----

    #[test]
    fn frustum_has_24_vertices_and_36_indices() {
        let m = frustum(1.0, 1.5, 0.1, 10.0);
        assert_eq!(m.positions.len(), 24); // 6 faces * 4 verts
        assert_eq!(m.indices.len(), 36);
    }

    #[test]
    fn frustum_near_plane_smaller_than_far() {
        let m = frustum(std::f32::consts::FRAC_PI_4, 1.5, 0.1, 10.0);
        let near_z = -0.1f32;
        let far_z = -10.0f32;
        let near_verts: Vec<_> = m
            .positions
            .iter()
            .filter(|p| (p[2] - near_z).abs() < 1e-3)
            .collect();
        let far_verts: Vec<_> = m
            .positions
            .iter()
            .filter(|p| (p[2] - far_z).abs() < 1e-3)
            .collect();
        assert!(!near_verts.is_empty());
        assert!(!far_verts.is_empty());
        let near_w = near_verts.iter().map(|p| p[0].abs()).fold(0.0f32, f32::max);
        let far_w = far_verts.iter().map(|p| p[0].abs()).fold(0.0f32, f32::max);
        assert!(far_w > near_w, "far plane should be wider than near plane");
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

    // ---- ring ----

    #[test]
    fn ring_radial_bounds() {
        let inner = 1.0;
        let outer = 2.0;
        let m = ring(inner, outer, 16);
        for (i, p) in m.positions.iter().enumerate() {
            let dist = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                dist >= inner - 1e-4 && dist <= outer + 1e-4,
                "ring vertex[{i}] radial = {dist}, expected in [{inner}, {outer}]"
            );
            assert!(p[2].abs() < 1e-6, "ring vertex should be at Z=0");
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

    // ---- spring ----

    #[test]
    fn spring_invariants() {
        let m = spring(2.0, 0.25, 3.0, 8);
        assert_mesh_invariants("spring", &m);
        assert!(!m.positions.is_empty());
    }

    #[test]
    fn spring_normals_unit_length() {
        assert_normals_unit_length("spring", &spring(2.0, 0.25, 2.0, 8));
    }

    // ---- grid_plane ----

    #[test]
    fn grid_plane_vertex_count() {
        let cols = 4u32;
        let rows = 3u32;
        let m = grid_plane(1.0, 1.0, cols, rows);
        assert_eq!(m.positions.len(), ((cols + 1) * (rows + 1)) as usize);
    }

    #[test]
    fn grid_plane_all_z_zero() {
        let m = grid_plane(5.0, 3.0, 8, 6);
        for (i, p) in m.positions.iter().enumerate() {
            assert!(p[2].abs() < 1e-6, "grid_plane vertex[{i}] Z = {}", p[2]);
        }
    }

    #[test]
    fn grid_plane_extents() {
        let (w, d) = (4.0, 6.0);
        let m = grid_plane(w, d, 4, 4);
        assert_positions_bounded("grid_plane", &m, [w / 2.0, d / 2.0, 0.0]);
    }

    #[test]
    fn grid_plane_index_count() {
        let cols = 4u32;
        let rows = 3u32;
        let m = grid_plane(1.0, 1.0, cols, rows);
        // Each cell = 2 triangles * 3 indices
        assert_eq!(m.indices.len(), (cols * rows * 6) as usize);
    }
}
