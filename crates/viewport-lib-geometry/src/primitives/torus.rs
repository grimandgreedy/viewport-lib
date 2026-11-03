//! Tube primitives swept along a closed path: tori and the spring.

use viewport_lib_types::data::mesh::MeshData;

/// Torus centred at the origin, lying in the XY plane.
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

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Sweep a circular tube of radius `minor_radius` along a closed centre-line in
/// the XY plane.
///
/// `centre_at(i)` returns the centre-line point and its tangent (a 2D vector in
/// the XY plane) for each ring `i` in `0..=stacks`. The tube cross-section is
/// built perpendicular to the tangent, spanning the in-plane normal and the
/// world up axis, so the tube stays normal to the path everywhere. Traverse the
/// centre-line counter-clockwise to keep the outward winding consistent.
fn swept_tube<F: Fn(u32) -> ([f32; 3], [f32; 2])>(
    minor_radius: f32,
    sectors: u32,
    stacks: u32,
    centre_at: F,
) -> MeshData {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..=stacks {
        let (centre, tangent) = centre_at(i);

        // In-plane normal perpendicular to the tangent: (tx, ty, 0) x (0, 0, 1).
        let tlen = (tangent[0] * tangent[0] + tangent[1] * tangent[1])
            .sqrt()
            .max(1e-6);
        let tx = tangent[0] / tlen;
        let ty = tangent[1] / tlen;
        let side = [ty, -tx, 0.0];

        for j in 0..=sectors {
            let theta = j as f32 * std::f32::consts::TAU / sectors as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            // Cross-section spans the in-plane side vector and the up axis.
            let nx = side[0] * cos_theta;
            let ny = side[1] * cos_theta;
            let nz = sin_theta;

            positions.push([
                centre[0] + minor_radius * nx,
                centre[1] + minor_radius * ny,
                centre[2] + minor_radius * nz,
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

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Elliptic torus centred at the origin, lying in the XY plane.
///
/// Like [`torus`], but the ring centre-line is an ellipse instead of a circle,
/// so the tube traces an oval. `torus(r, m, s, st)` is the special case
/// `torus_ellipse(r, r, m, s, st)`.
///
/// `radius_x` / `radius_y` : ellipse semi-axes of the ring centre-line.
/// `minor_radius` : radius of the tube.
/// `sectors` : segments around the tube (minimum 3).
/// `stacks` : segments around the ring (minimum 3).
pub fn torus_ellipse(
    radius_x: f32,
    radius_y: f32,
    minor_radius: f32,
    sectors: u32,
    stacks: u32,
) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(3);

    swept_tube(minor_radius, sectors, stacks, |i| {
        let phi = i as f32 * std::f32::consts::TAU / stacks as f32;
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let centre = [radius_x * cos_phi, radius_y * sin_phi, 0.0];
        // Tangent of the ellipse centre-line, traversed counter-clockwise.
        let tangent = [-radius_x * sin_phi, radius_y * cos_phi];
        (centre, tangent)
    })
}

/// Stadium (capsule) torus centred at the origin, lying in the XY plane.
///
/// The ring centre-line is a stadium: two semicircular ends joined by two
/// straight runs, which gives the chain-link / oval-link outline. The straight
/// runs lie along the X axis and the ends bulge out along +/-X.
///
/// `straight_length` : length of each straight run between the two ends.
/// `bend_radius` : radius of the semicircular ends (half the stadium height).
/// `minor_radius` : radius of the tube.
/// `sectors` : segments around the tube (minimum 3).
/// `stacks` : segments along the ring (minimum 3).
pub fn torus_stadium(
    straight_length: f32,
    bend_radius: f32,
    minor_radius: f32,
    sectors: u32,
    stacks: u32,
) -> MeshData {
    let sectors = sectors.max(3);
    let stacks = stacks.max(3);

    let sl = straight_length.max(0.0);
    let br = bend_radius.max(1e-4);
    let half = sl * 0.5;
    let arc = std::f32::consts::PI * br; // length of one semicircular end
    let perimeter = 2.0 * sl + 2.0 * arc;

    swept_tube(minor_radius, sectors, stacks, |i| {
        let mut s = i as f32 / stacks as f32 * perimeter;
        if s >= perimeter {
            s -= perimeter;
        }

        // Walk the stadium counter-clockwise:
        // right end -> top straight -> left end -> bottom straight.
        if s < arc {
            // Right semicircle, centre (half, 0), angle -pi/2 .. pi/2.
            let a = -std::f32::consts::FRAC_PI_2 + (s / arc) * std::f32::consts::PI;
            (
                [half + br * a.cos(), br * a.sin(), 0.0],
                [-a.sin(), a.cos()],
            )
        } else if s < arc + sl {
            // Top straight, from (half, br) to (-half, br).
            let u = (s - arc) / sl;
            ([half - u * sl, br, 0.0], [-1.0, 0.0])
        } else if s < 2.0 * arc + sl {
            // Left semicircle, centre (-half, 0), angle pi/2 .. 3pi/2.
            let a = std::f32::consts::FRAC_PI_2 + ((s - arc - sl) / arc) * std::f32::consts::PI;
            (
                [-half + br * a.cos(), br * a.sin(), 0.0],
                [-a.sin(), a.cos()],
            )
        } else {
            // Bottom straight, from (-half, -br) to (half, -br).
            let u = (s - 2.0 * arc - sl) / sl;
            ([-half + u * sl, -br, 0.0], [1.0, 0.0])
        }
    })
}

/// Helical spring centred at the origin, axis along Z.
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

        // Binormal = T x principal_normal
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

    // u: around the tube cross-section; v: along the helix from 0 to 1.
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    for seg in 0..=n_segs {
        let v = seg as f32 / n_segs as f32;
        for sec in 0..=sectors {
            uvs.push([sec as f32 / sectors as f32, v]);
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

    #[test]
    fn torus_ellipse_matches_torus_when_axes_equal() {
        // An ellipse with equal semi-axes is a circle, so the elliptic torus
        // must reproduce the plain torus vertex-for-vertex.
        let a = torus(2.0, 0.5, 16, 12);
        let b = torus_ellipse(2.0, 2.0, 0.5, 16, 12);
        assert_eq!(a.positions.len(), b.positions.len());
        for (pa, pb) in a.positions.iter().zip(b.positions.iter()) {
            for k in 0..3 {
                assert!(
                    (pa[k] - pb[k]).abs() < 1e-4,
                    "torus_ellipse diverged from torus: {pa:?} vs {pb:?}"
                );
            }
        }
    }

    #[test]
    fn torus_ellipse_vertices_within_bounds() {
        let (rx, ry, minor) = (3.0f32, 1.5f32, 0.4f32);
        let m = torus_ellipse(rx, ry, minor, 16, 16);
        let max_r = rx.max(ry) + minor + 1e-3;
        for (i, p) in m.positions.iter().enumerate() {
            let radial_xy = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                radial_xy <= max_r,
                "torus_ellipse vertex[{i}] radial_xy = {radial_xy}, expected <= {max_r}"
            );
            assert!(
                p[2].abs() <= minor + 1e-3,
                "torus_ellipse vertex[{i}] z = {} out of tube bounds",
                p[2]
            );
        }
    }

    #[test]
    fn torus_stadium_extent_matches_path() {
        // Stadium straights run along X; the X extent reaches half the straight
        // length plus an end radius plus the tube radius.
        let (sl, br, minor) = (4.0f32, 1.0f32, 0.25f32);
        let m = torus_stadium(sl, br, minor, 16, 48);
        let max_x = m.positions.iter().fold(f32::MIN, |a, p| a.max(p[0]));
        let max_y = m.positions.iter().fold(f32::MIN, |a, p| a.max(p[1]));
        let expect_x = sl * 0.5 + br + minor;
        let expect_y = br + minor;
        assert!(
            (max_x - expect_x).abs() < 0.05,
            "stadium max_x = {max_x}, expected ~{expect_x}"
        );
        assert!(
            (max_y - expect_y).abs() < 0.05,
            "stadium max_y = {max_y}, expected ~{expect_y}"
        );
    }

    #[test]
    fn torus_stadium_has_uvs() {
        assert!(torus_stadium(2.0, 0.5, 0.2, 8, 16).uvs.is_some());
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
}
