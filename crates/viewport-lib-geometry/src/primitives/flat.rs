//! Flat primitives in the XY plane: planes, grids, disks and rings.

use viewport_lib_types::data::mesh::MeshData;

/// Flat XY plane centred at the origin (Z-up world: this is the ground plane).
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

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Flat disk in the XY plane, centred at the origin, normal pointing +Z.
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

    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(sectors as usize + 1);
    uvs.push([0.5, 0.5]);
    for j in 0..sectors {
        let a = j as f32 * step;
        uvs.push([0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Flat ring (annulus) in the XY plane, centred at the origin, normal pointing +Z.
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

    // Inner edge v=0, outer edge v=1; u wraps 0->1 around the ring.
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(2 * (sectors as usize + 1));
    for j in 0..=sectors {
        let u = j as f32 / sectors as f32;
        uvs.push([u, 0.0]); // inner
        uvs.push([u, 1.0]); // outer
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Subdivided plane in the XY plane, centred at the origin, normal pointing +Z.
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

    let mut uvs: Vec<[f32; 2]> = Vec::new();
    for row in 0..=rows {
        for col in 0..=cols {
            uvs.push([col as f32 / cols as f32, row as f32 / rows as f32]);
        }
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::tests::assert_positions_bounded;

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
}
