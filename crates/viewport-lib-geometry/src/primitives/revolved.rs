//! Primitives revolved around the Z axis: cylinders, cones and the arrow glyph.

use viewport_lib_types::data::mesh::MeshData;

/// Cylinder centred at the origin, axis along Z.
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

    // UVs follow the same vertex order as the position build above.
    // Side UVs have a single-edge seam (u wraps at j=0/j=sectors) which are accepted for non-seam geometry.
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(4 * sectors as usize + 2);
    for j in 0..sectors {
        uvs.push([j as f32 / sectors as f32, 0.0]);
    }
    for j in 0..sectors {
        uvs.push([j as f32 / sectors as f32, 1.0]);
    }
    uvs.push([0.5, 0.5]); // bottom center
    uvs.push([0.5, 0.5]); // top center
    for j in 0..sectors {
        let a = j as f32 * step;
        uvs.push([0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
    }
    for j in 0..sectors {
        let a = j as f32 * step;
        uvs.push([0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Cone with tip at +Z and base at -Z, centred at the origin.
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

    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(4 * sectors as usize + 1);
    let tau = std::f32::consts::TAU;
    for j in 0..sectors {
        let a0 = j as f32 * step;
        let a1 = (j + 1) as f32 * step;
        let amid = (a0 + a1) * 0.5;
        uvs.push([amid / tau, 1.0]); // tip
        uvs.push([a0 / tau, 0.0]); // base left
        uvs.push([a1 / tau, 0.0]); // base right
    }
    uvs.push([0.5, 0.5]); // bottom center
    for j in 0..sectors {
        let a = j as f32 * step;
        uvs.push([0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Arrow along +Z, centred at the origin (total length 1).
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

    // V is proportional to Z: shaft_bot -> 0, head_top -> 1.
    let v_shaft_top = 1.0 - head_fraction; // head_fraction already clamped
    let tau = std::f32::consts::TAU;
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(7 * sectors as usize + 2);
    for j in 0..sectors {
        uvs.push([j as f32 / sectors as f32, 0.0]);
    }
    for j in 0..sectors {
        uvs.push([j as f32 / sectors as f32, v_shaft_top]);
    }
    uvs.push([0.5, 0.5]); // sb_center
    for j in 0..sectors {
        let a = j as f32 * step;
        uvs.push([0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
    }
    for j in 0..sectors {
        let a0 = j as f32 * step;
        let a1 = (j + 1) as f32 * step;
        let amid = (a0 + a1) * 0.5;
        uvs.push([amid / tau, 1.0]);
        uvs.push([a0 / tau, v_shaft_top]);
        uvs.push([a1 / tau, v_shaft_top]);
    }
    uvs.push([0.5, 0.5]); // hb_center
    for j in 0..sectors {
        let a = j as f32 * step;
        uvs.push([0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]);
    }

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::tests::assert_mesh_invariants;

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
}
