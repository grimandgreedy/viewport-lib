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
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
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

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// UV sphere centered at the origin.
///
/// `radius` — sphere radius.
/// `sectors` — longitude subdivisions (minimum 3).
/// `stacks` — latitude subdivisions (minimum 2).
pub fn sphere(radius: f32, sectors: u32, stacks: u32) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(2);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
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

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Flat XZ plane centered at the origin.
///
/// `width` — extent along X. `depth` — extent along Z.
pub fn plane(width: f32, depth: f32) -> MeshData {
    let hw = width / 2.0;
    let hd = depth / 2.0;

    let positions = vec![
        [-hw, 0.0, -hd],
        [ hw, 0.0, -hd],
        [ hw, 0.0,  hd],
        [-hw, 0.0,  hd],
    ];
    let normals = vec![[0.0, 1.0, 0.0]; 4];
    let indices = vec![0, 1, 2, 0, 2, 3];

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Cylinder centered at the origin, axis along Y.
///
/// `radius` — circle radius. `height` — total height. `sectors` — circumference subdivisions (minimum 3).
pub fn cylinder(radius: f32, height: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let half_h = height / 2.0;
    let step = 2.0 * std::f32::consts::PI / sectors as f32;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Side vertices: two rings (bottom then top)
    for &y in &[-half_h, half_h] {
        for j in 0..sectors {
            let angle = j as f32 * step;
            let x = radius * angle.cos();
            let z = radius * angle.sin();
            positions.push([x, y, z]);
            normals.push([angle.cos(), 0.0, angle.sin()]);
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
    positions.push([0.0, -half_h, 0.0]);
    normals.push([0.0, -1.0, 0.0]);

    let top_center = positions.len() as u32;
    positions.push([0.0, half_h, 0.0]);
    normals.push([0.0, 1.0, 0.0]);

    // Cap rim vertices (separate so normals point up/down)
    let bottom_rim_start = positions.len() as u32;
    for j in 0..sectors {
        let angle = j as f32 * step;
        positions.push([radius * angle.cos(), -half_h, radius * angle.sin()]);
        normals.push([0.0, -1.0, 0.0]);
    }

    let top_rim_start = positions.len() as u32;
    for j in 0..sectors {
        let angle = j as f32 * step;
        positions.push([radius * angle.cos(), half_h, radius * angle.sin()]);
        normals.push([0.0, 1.0, 0.0]);
    }

    // Cap faces
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        // Bottom (winding reversed so normal faces down)
        indices.extend_from_slice(&[bottom_center, bottom_rim_start + next, bottom_rim_start + j]);
        // Top
        indices.extend_from_slice(&[top_center, top_rim_start + j, top_rim_start + next]);
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Non-uniform box (cuboid) centered at the origin.
///
/// `width` — X extent. `height` — Y extent. `depth` — Z extent.
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
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
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

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Cone with tip at +Y and base at −Y, centered at the origin.
///
/// `radius` — base radius. `height` — total height. `sectors` — circumference subdivisions (minimum 3).
pub fn cone(radius: f32, height: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let half_h = height / 2.0;
    let step = 2.0 * std::f32::consts::PI / sectors as f32;

    // Side normal components: outward radial and upward Y.
    let hyp = (radius * radius + height * height).sqrt();
    let ny = radius / hyp;
    let nr = height / hyp;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Side faces — one duplicated tip vertex per sector so each has the bisector normal.
    for j in 0..sectors {
        let a0 = j as f32 * step;
        let a1 = (j + 1) as f32 * step;
        let amid = (a0 + a1) * 0.5;
        let base = positions.len() as u32;

        positions.push([0.0, half_h, 0.0]);
        normals.push([nr * amid.cos(), ny, nr * amid.sin()]);

        positions.push([radius * a0.cos(), -half_h, radius * a0.sin()]);
        normals.push([nr * a0.cos(), ny, nr * a0.sin()]);

        positions.push([radius * a1.cos(), -half_h, radius * a1.sin()]);
        normals.push([nr * a1.cos(), ny, nr * a1.sin()]);

        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Bottom cap
    let bottom_center = positions.len() as u32;
    positions.push([0.0, -half_h, 0.0]);
    normals.push([0.0, -1.0, 0.0]);

    let rim_start = positions.len() as u32;
    for j in 0..sectors {
        let a = j as f32 * step;
        positions.push([radius * a.cos(), -half_h, radius * a.sin()]);
        normals.push([0.0, -1.0, 0.0]);
    }
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[bottom_center, rim_start + next, rim_start + j]);
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Capsule (cylinder body with hemispherical caps) centered at the origin, axis along Y.
///
/// `radius` — sphere cap radius. `height` — total height (clamped so body ≥ 0).
/// `sectors` — longitude subdivisions (minimum 3). `stacks` — latitude subdivisions (minimum 2).
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
            let nz = cos_phi * theta.sin();
            positions.push([radius * nx, half_body + radius * sin_phi, radius * nz]);
            normals.push([nx, sin_phi, nz]);
        }
    }

    // Bottom hemisphere (equator at i=0, tip at i=hemi_stacks, offset center −half_body)
    let bottom_off = (hemi_stacks + 1) as u32;
    for i in 0..=hemi_stacks {
        let phi = -std::f32::consts::FRAC_PI_2 * i as f32 / hemi_stacks as f32;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for j in 0..=sectors {
            let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
            let nx = cos_phi * theta.cos();
            let nz = cos_phi * theta.sin();
            positions.push([radius * nx, -half_body + radius * sin_phi, radius * nz]);
            normals.push([nx, sin_phi, nz]);
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
                k1 + j, k2 + j, k1 + j + 1,
                k1 + j + 1, k2 + j, k2 + j + 1,
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

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Torus centered at the origin, lying in the XZ plane.
///
/// `major_radius` — distance from the torus centre to the tube centre.
/// `minor_radius` — radius of the tube.
/// `sectors` — segments around the tube (minimum 3).
/// `stacks` — segments around the torus ring (minimum 3).
pub fn torus(major_radius: f32, minor_radius: f32, sectors: u32, stacks: u32) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(3);

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=stacks {
        let phi = i as f32 * std::f32::consts::TAU / stacks as f32;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let cx = major_radius * cos_phi;
        let cz = major_radius * sin_phi;

        for j in 0..=sectors {
            let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let nx = cos_phi * cos_theta;
            let ny = sin_theta;
            let nz = sin_phi * cos_theta;

            positions.push([cx + minor_radius * nx, minor_radius * ny, cz + minor_radius * nz]);
            normals.push([nx, ny, nz]);
        }
    }

    let cols = sectors + 1;
    for i in 0..stacks {
        let k1 = i * cols;
        let k2 = k1 + cols;
        for j in 0..sectors {
            indices.extend_from_slice(&[
                k1 + j, k2 + j, k1 + j + 1,
                k1 + j + 1, k2 + j, k2 + j + 1,
            ]);
        }
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Icosphere centered at the origin (better tessellation than UV sphere; no pole pinching).
///
/// `radius` — sphere radius. `subdivisions` — refinement level (0 = raw icosahedron, 20 faces).
pub fn icosphere(radius: f32, subdivisions: u32) -> MeshData {
    let phi = (1.0 + 5.0f32.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let a = 1.0 / norm;
    let b = phi / norm;

    let mut verts: Vec<[f32; 3]> = vec![
        [-a,  b,  0.0], [ a,  b,  0.0], [-a, -b,  0.0], [ a, -b,  0.0],
        [ 0.0, -a,  b], [ 0.0,  a,  b], [ 0.0, -a, -b], [ 0.0,  a, -b],
        [ b,  0.0, -a], [ b,  0.0,  a], [-b,  0.0, -a], [-b,  0.0,  a],
    ];
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
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
    let positions: Vec<[f32; 3]> = verts.iter().map(|v| [v[0] * radius, v[1] * radius, v[2] * radius]).collect();
    let indices: Vec<u32> = faces.iter().flat_map(|f| f.iter().copied()).collect();

    MeshData { positions, normals, indices, ..MeshData::default() }
}

fn ico_midpoint(
    verts: &mut Vec<[f32; 3]>,
    cache: &mut std::collections::HashMap<u64, u32>,
    a: u32,
    b: u32,
) -> u32 {
    let key = if a < b { (a as u64) << 32 | b as u64 } else { (b as u64) << 32 | a as u64 };
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

/// Arrow along +Y, centered at the origin (total length 1).
///
/// `shaft_radius` — cylinder shaft radius.
/// `head_radius` — cone head base radius.
/// `head_fraction` — fraction of total length occupied by the cone head (clamped to 0.1–0.9).
/// `sectors` — circumference subdivisions (minimum 3).
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
    for &y in &[shaft_bot, shaft_top] {
        for j in 0..sectors {
            let a = j as f32 * step;
            positions.push([shaft_radius * a.cos(), y, shaft_radius * a.sin()]);
            normals.push([a.cos(), 0.0, a.sin()]);
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
    positions.push([0.0, shaft_bot, 0.0]);
    normals.push([0.0, -1.0, 0.0]);
    let sb_rim = positions.len() as u32;
    for j in 0..sectors {
        let a = j as f32 * step;
        positions.push([shaft_radius * a.cos(), shaft_bot, shaft_radius * a.sin()]);
        normals.push([0.0, -1.0, 0.0]);
    }
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[sb_center, sb_rim + next, sb_rim + j]);
    }

    // Cone head side (one duplicated tip per sector)
    let cone_hyp = (head_radius * head_radius + head_h * head_h).sqrt();
    let cny = head_radius / cone_hyp;
    let cnr = head_h / cone_hyp;
    for j in 0..sectors {
        let a0 = j as f32 * step;
        let a1 = (j + 1) as f32 * step;
        let amid = (a0 + a1) * 0.5;
        let base = positions.len() as u32;

        positions.push([0.0, head_top, 0.0]);
        normals.push([cnr * amid.cos(), cny, cnr * amid.sin()]);

        positions.push([head_radius * a0.cos(), head_bot, head_radius * a0.sin()]);
        normals.push([cnr * a0.cos(), cny, cnr * a0.sin()]);

        positions.push([head_radius * a1.cos(), head_bot, head_radius * a1.sin()]);
        normals.push([cnr * a1.cos(), cny, cnr * a1.sin()]);

        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Cone base cap
    let hb_center = positions.len() as u32;
    positions.push([0.0, head_bot, 0.0]);
    normals.push([0.0, -1.0, 0.0]);
    let hb_rim = positions.len() as u32;
    for j in 0..sectors {
        let a = j as f32 * step;
        positions.push([head_radius * a.cos(), head_bot, head_radius * a.sin()]);
        normals.push([0.0, -1.0, 0.0]);
    }
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[hb_center, hb_rim + next, hb_rim + j]);
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Flat disk in the XZ plane, centered at the origin, normal pointing +Y.
///
/// `radius` — disk radius. `sectors` — circumference subdivisions (minimum 3).
pub fn disk(radius: f32, sectors: u32) -> MeshData {
    let sectors = sectors.max(3);
    let step = std::f32::consts::TAU / sectors as f32;

    let mut positions: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]];
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 1.0, 0.0]];
    let mut indices: Vec<u32> = Vec::new();

    for j in 0..sectors {
        let a = j as f32 * step;
        positions.push([radius * a.cos(), 0.0, radius * a.sin()]);
        normals.push([0.0, 1.0, 0.0]);
    }

    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[0, j + 1, next + 1]);
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Camera frustum mesh for visualization.
///
/// The camera sits at the origin looking along −Z.
/// `fov_y` — vertical field of view in radians. `aspect` — width / height.
/// `near`, `far` — clip plane distances (positive values).
pub fn frustum(fov_y: f32, aspect: f32, near: f32, far: f32) -> MeshData {
    let half_h_n = near * (fov_y * 0.5).tan();
    let half_w_n = half_h_n * aspect;
    let half_h_f = far * (fov_y * 0.5).tan();
    let half_w_f = half_h_f * aspect;

    // 8 corners
    let nbl = [-half_w_n, -half_h_n, -near];
    let nbr = [ half_w_n, -half_h_n, -near];
    let ntr = [ half_w_n,  half_h_n, -near];
    let ntl = [-half_w_n,  half_h_n, -near];
    let fbl = [-half_w_f, -half_h_f, -far];
    let fbr = [ half_w_f, -half_h_f, -far];
    let ftr = [ half_w_f,  half_h_f, -far];
    let ftl = [-half_w_f,  half_h_f, -far];

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
        let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
        let e2 = [v3[0]-v0[0], v3[1]-v0[1], v3[2]-v0[2]];
        let nr = [
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        ];
        let len = (nr[0]*nr[0] + nr[1]*nr[1] + nr[2]*nr[2]).sqrt();
        let n = if len > 0.0 { [nr[0]/len, nr[1]/len, nr[2]/len] } else { [0.0, 0.0, 1.0] };

        let base = positions.len() as u32;
        for v in quad {
            positions.push(*v);
            normals.push(n);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Hemisphere (upper half of a UV sphere) centered at the origin, dome facing +Y.
///
/// `radius` — sphere radius.
/// `sectors` — longitude subdivisions (minimum 3). `stacks` — latitude subdivisions (minimum 1).
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
            let nz = cos_phi * theta.sin();
            positions.push([radius * nx, radius * sin_phi, radius * nz]);
            normals.push([nx, sin_phi, nz]);
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

    // Equator disk cap (faces −Y)
    let center = positions.len() as u32;
    positions.push([0.0, 0.0, 0.0]);
    normals.push([0.0, -1.0, 0.0]);
    let rim_start = positions.len() as u32;
    for j in 0..sectors {
        let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
        positions.push([radius * theta.cos(), 0.0, radius * theta.sin()]);
        normals.push([0.0, -1.0, 0.0]);
    }
    for j in 0..sectors as u32 {
        let next = (j + 1) % sectors as u32;
        indices.extend_from_slice(&[center, rim_start + next, rim_start + j]);
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Flat ring (annulus) in the XZ plane, centered at the origin, normal pointing +Y.
///
/// `inner_radius` — inner edge. `outer_radius` — outer edge. `sectors` — circumference subdivisions (minimum 3).
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
        positions.push([inner_radius * cos_a, 0.0, inner_radius * sin_a]);
        normals.push([0.0, 1.0, 0.0]);
        positions.push([outer_radius * cos_a, 0.0, outer_radius * sin_a]);
        normals.push([0.0, 1.0, 0.0]);
    }

    for j in 0..sectors as u32 {
        let i0 = j * 2;
        let o0 = i0 + 1;
        let i1 = i0 + 2;
        let o1 = i0 + 3;
        indices.extend_from_slice(&[i0, o0, i1, i1, o0, o1]);
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Ellipsoid centered at the origin.
///
/// `rx`, `ry`, `rz` — semi-axes along X, Y, Z.
/// `sectors` — longitude subdivisions (minimum 3). `stacks` — latitude subdivisions (minimum 2).
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
            let len = (nx*nx + ny*ny + nz*nz).sqrt();
            normals.push(if len > 0.0 { [nx/len, ny/len, nz/len] } else { [0.0, 1.0, 0.0] });
        }
    }

    for i in 0..stacks {
        let k1 = i * (sectors + 1);
        let k2 = k1 + sectors + 1;
        for j in 0..sectors {
            if i != 0 {
                indices.extend_from_slice(&[k1 + j, k2 + j, k1 + j + 1]);
            }
            if i != stacks - 1 {
                indices.extend_from_slice(&[k1 + j + 1, k2 + j, k2 + j + 1]);
            }
        }
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Helical spring centered at the origin, axis along Y.
///
/// `radius` — distance from Y axis to tube centre.
/// `coil_radius` — cross-section tube radius.
/// `turns` — number of complete coil turns.
/// `sectors` — tube cross-section subdivisions (minimum 3).
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
        let cy = t / std::f32::consts::TAU * pitch - height * 0.5;
        let cz = radius * t.sin();

        // Helix tangent
        let dtx = -radius * t.sin();
        let dty = pitch / std::f32::consts::TAU;
        let dtz = radius * t.cos();
        let dt_len = (dtx*dtx + dty*dty + dtz*dtz).sqrt();
        let (tx, ty, tz) = (dtx / dt_len, dty / dt_len, dtz / dt_len);

        // Principal normal: inward radial toward helix axis
        let pnx = -t.cos();
        let pny = 0.0f32;
        let pnz = -t.sin();

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

            positions.push([cx + coil_radius * on_x, cy + coil_radius * on_y, cz + coil_radius * on_z]);
            normals.push([on_x, on_y, on_z]);
        }
    }

    let cols = sectors + 1;
    for seg in 0..n_segs {
        let k1 = seg * cols;
        let k2 = k1 + cols;
        for sec in 0..sectors {
            indices.extend_from_slice(&[
                k1 + sec, k2 + sec, k1 + sec + 1,
                k1 + sec + 1, k2 + sec, k2 + sec + 1,
            ]);
        }
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}

/// Subdivided plane in the XZ plane, centered at the origin, normal pointing +Y.
///
/// `width` — X extent. `depth` — Z extent.
/// `cols` — column subdivisions (minimum 1). `rows` — row subdivisions (minimum 1).
pub fn grid_plane(width: f32, depth: f32, cols: u32, rows: u32) -> MeshData {
    let cols = cols.max(1);
    let rows = rows.max(1);
    let hw = width * 0.5;
    let hd = depth * 0.5;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for row in 0..=rows {
        let z = -hd + row as f32 / rows as f32 * depth;
        for col in 0..=cols {
            let x = -hw + col as f32 / cols as f32 * width;
            positions.push([x, 0.0, z]);
            normals.push([0.0, 1.0, 0.0]);
        }
    }

    let v_cols = cols + 1;
    for row in 0..rows {
        for col in 0..cols {
            let tl = row * v_cols + col;
            let tr = tl + 1;
            let bl = tl + v_cols;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    MeshData { positions, normals, indices, ..MeshData::default() }
}
