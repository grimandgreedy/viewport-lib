//! Box-shaped primitives: cubes, cuboids and the camera frustum.

use viewport_lib_types::data::mesh::MeshData;

/// Unit cube (side length 1, centred at the origin).
///
/// `size` scales all three axes uniformly.
pub fn cube(size: f32) -> MeshData {
    let h = size / 2.0;

    // 6 faces x 4 vertices each = 24 vertices
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

    // 6 faces x 2 triangles x 3 indices
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

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Non-uniform box (cuboid) centred at the origin.
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

    // Each face gets the same quad UVs as cube.
    let uvs: Vec<[f32; 2]> = (0..6)
        .flat_map(|_| [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        .collect();

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

/// Box-unwrap UVs for the shared 24-vertex cube / cuboid layout: each of the six
/// faces gets its own tile in a 3x2 atlas, so a texture (or a paint stroke) maps
/// to each face independently rather than repeating identically across all six
/// (as the overlapping `[0,1]x[0,1]` UVs of [`cube`] / [`cuboid`] do).
///
/// A small inset keeps each face's UVs a texel or so inside its tile border, so
/// bilinear filtering at a tile edge does not bleed one face's colour into its
/// neighbour's.
fn box_unwrap_uvs() -> Vec<[f32; 2]> {
    // Roughly one texel of a 512-map; large enough to stop cross-face bleed at
    // common resolutions, small enough not to visibly crop a face.
    const INSET: f32 = 1.0 / 512.0;
    (0..6u32)
        .flat_map(|f| {
            let (col, row) = (f % 3, f / 3);
            let u0 = col as f32 / 3.0 + INSET;
            let u1 = (col + 1) as f32 / 3.0 - INSET;
            let v0 = row as f32 / 2.0 + INSET;
            let v1 = (row + 1) as f32 / 2.0 - INSET;
            // Same per-face corner order as the wrapped layout: [0,0] [1,0] [1,1] [0,1].
            [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
        })
        .collect()
}

/// Unit cube with a **box-unwrap** UV layout ([`box_unwrap_uvs`]).
///
/// Geometry is identical to [`cube`]; only the UVs differ. Where [`cube`]'s six
/// faces share one `[0,1]x[0,1]` square (so a texture repeats identically on every
/// face), each face here owns a distinct atlas tile, making per-face texturing
/// and painting independent.
pub fn cube_unwrapped(size: f32) -> MeshData {
    let mut m = cube(size);
    m.uvs = Some(box_unwrap_uvs());
    m
}

/// Rectangular box with a **box-unwrap** UV layout: the [`cuboid`] counterpart of
/// [`cube_unwrapped`]. Geometry matches [`cuboid`]; each face owns its own atlas
/// tile so faces texture / paint independently.
pub fn cuboid_unwrapped(width: f32, height: f32, depth: f32) -> MeshData {
    let mut m = cuboid(width, height, depth);
    m.uvs = Some(box_unwrap_uvs());
    m
}

/// Camera frustum mesh for visualisation.
///
/// The camera sits at the origin looking along -Z.
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

    // 6 faces as (v0, v1, v2, v3); normal from (v1-v0) x (v3-v0)
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

    let uvs: Vec<[f32; 2]> = (0..6)
        .flat_map(|_| [[0.0f32, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        .collect();

    let mut m = MeshData::new(positions, normals, indices);
    m.uvs = Some(uvs);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::tests::{assert_normals_unit_length, assert_positions_bounded};

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

    #[test]
    fn wrapped_cube_shares_one_uv_square_across_faces() {
        // The overlapping layout: every face's four UVs are the same [0,1] quad.
        let uvs = cube(1.0).uvs.unwrap();
        for face in 0..6 {
            assert_eq!(&uvs[face * 4..face * 4 + 4], &uvs[0..4]);
        }
    }

    #[test]
    fn unwrapped_cube_gives_each_face_a_distinct_tile() {
        // The box unwrap: each face's UVs fall inside a unique 3x2 atlas cell, so
        // no two faces share texels (the point of per-face painting).
        let m = cube_unwrapped(1.0);
        assert_eq!(m.uvs.as_ref().map(Vec::len), Some(24));
        let uvs = m.uvs.unwrap();
        let mut cells = Vec::new();
        for face in 0..6 {
            let quad = &uvs[face * 4..face * 4 + 4];
            // Every vertex of a face lands in the same cell.
            let cell = |uv: &[f32; 2]| ((uv[0] * 3.0) as u32, (uv[1] * 2.0) as u32);
            let c = cell(&quad[0]);
            for uv in quad {
                assert_eq!(cell(uv), c, "face {face} spans more than one atlas cell");
            }
            cells.push(c);
        }
        cells.sort();
        cells.dedup();
        assert_eq!(cells.len(), 6, "faces must occupy six distinct tiles");
    }

    #[test]
    fn unwrapped_cube_keeps_cube_geometry() {
        // Only UVs change: positions / normals / indices match the wrapped cube.
        let wrapped = cube(2.0);
        let unwrapped = cube_unwrapped(2.0);
        assert_eq!(wrapped.positions, unwrapped.positions);
        assert_eq!(wrapped.normals, unwrapped.normals);
        assert_eq!(wrapped.indices, unwrapped.indices);
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
}
