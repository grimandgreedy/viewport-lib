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
