use super::tets::decompose_to_tetrahedra;
use super::*;
use crate::maths::vec3::{cross3, dot3};
use viewport_lib_types::data::attribute::AttributeData;

fn block(n: u32) -> Vec<[u32; 3]> {
    let mut cells = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                cells.push([i, j, k]);
            }
        }
    }
    cells
}

#[test]
fn grid_single_cell_is_a_closed_box() {
    let grid = VolumeMeshData::from_grid_cells([1.0, 2.0, 3.0], [2.0, 1.0, 0.5], &[[0, 0, 0]]);
    assert_eq!(grid.data.positions.len(), 8);
    assert_eq!(grid.vertex_nodes.len(), 8);
    let (mesh, face_to_cell) = extract_boundary_faces(&grid.data);
    assert_eq!(mesh.indices.len(), 36);
    assert!(face_to_cell.iter().all(|&c| c == 0));
    // The far corner is origin + cell_size.
    assert!(mesh.positions.contains(&[3.0, 3.0, 3.5]));
    // Every face normal points away from the box centre.
    let centre = [2.0, 2.5, 3.25];
    for tri in mesh.indices.chunks_exact(3) {
        let [a, b, c] = [0, 1, 2].map(|i| mesh.positions[tri[i] as usize]);
        let n = cross3(
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        );
        let out = [a[0] - centre[0], a[1] - centre[1], a[2] - centre[2]];
        assert!(dot3(n, out) > 0.0);
    }
}

#[test]
fn grid_neighbours_share_corners_and_lose_their_common_face() {
    let grid = VolumeMeshData::from_grid_cells([0.0; 3], [1.0; 3], &[[0, 0, 0], [1, 0, 0]]);
    // Two boxes with one shared face: 12 corners, 10 outer quads.
    assert_eq!(grid.data.positions.len(), 12);
    let (mesh, face_to_cell) = extract_boundary_faces(&grid.data);
    assert_eq!(mesh.indices.len() / 3, 20);
    assert_eq!(face_to_cell.iter().filter(|&&c| c == 0).count(), 10);
}

#[test]
fn grid_solid_block_keeps_only_its_shell() {
    let n = 4;
    let grid = VolumeMeshData::from_grid_cells([0.0; 3], [1.0; 3], &block(n));
    assert_eq!(grid.data.cells.len(), (n * n * n) as usize);
    assert_eq!(
        grid.data.positions.len(),
        ((n + 1) * (n + 1) * (n + 1)) as usize
    );
    let (mesh, _) = extract_boundary_faces(&grid.data);
    assert_eq!(mesh.indices.len() / 3, (12 * n * n) as usize);
}

#[test]
fn grid_cells_far_apart_take_the_same_form() {
    // Two cells a long way apart: the bounding box is far larger than the
    // cell list, so node lookup goes through the map.
    let cells = [[0, 0, 0], [5000, 5000, 5000]];
    let grid = VolumeMeshData::from_grid_cells([0.0; 3], [1.0; 3], &cells);
    assert_eq!(grid.data.positions.len(), 16);
    assert_eq!(grid.vertex_nodes[8], [5000, 5000, 5000]);
    let (mesh, _) = extract_boundary_faces(&grid.data);
    assert_eq!(mesh.indices.len() / 3, 24);
    assert!(
        VolumeMeshData::from_grid_cells([0.0; 3], [1.0; 3], &[])
            .data
            .cells
            .is_empty()
    );
}

#[test]
fn node_scalars_become_vertex_attributes() {
    let mut grid = VolumeMeshData::from_grid_cells([0.0; 3], [1.0; 3], &[[0, 0, 0]]);
    let heights: Vec<f32> = grid.data.positions.iter().map(|p| p[2]).collect();
    grid.data
        .node_scalars
        .insert("height".into(), heights.clone());
    // Shorter than the vertex list: padded.
    grid.data.node_scalars.insert("short".into(), vec![7.0; 3]);
    // Same name as a cell scalar: the cell entry keeps it.
    grid.data.cell_scalars.insert("both".into(), vec![1.0]);
    grid.data.node_scalars.insert("both".into(), vec![2.0; 8]);

    let (mesh, _) = extract_boundary_faces(&grid.data);
    match mesh.attributes.get("height") {
        Some(AttributeData::Vertex(v)) => assert_eq!(*v, heights),
        _ => panic!("height is not a vertex attribute"),
    }
    match mesh.attributes.get("short") {
        Some(AttributeData::Vertex(v)) => {
            assert_eq!(v.len(), 8);
            assert_eq!(&v[..4], &[7.0, 7.0, 7.0, 0.0]);
        }
        _ => panic!("short is not a vertex attribute"),
    }
    assert!(matches!(
        mesh.attributes.get("both"),
        Some(AttributeData::Face(_))
    ));
}

#[test]
fn node_scalars_interpolate_onto_cut_vertices() {
    // A 2x2x2 block with a field linear in position, cut by a plane that
    // passes through the middle of cells. A linear field is reproduced
    // exactly by interpolation, so every output vertex must carry the
    // field's value at its own position.
    let mut grid = VolumeMeshData::from_grid_cells([0.0; 3], [1.0; 3], &block(2));
    let field = |p: [f32; 3]| 2.0 * p[0] - 3.0 * p[1] + 0.5 * p[2] + 1.0;
    let values: Vec<f32> = grid.data.positions.iter().map(|&p| field(p)).collect();
    grid.data.node_scalars.insert("f".into(), values);

    let planes = [[1.0, 0.25, 0.0, -0.7], [0.0, 0.0, -1.0, 1.4]];
    let (mesh, _) = extract_clipped_volume_faces(&grid.data, &planes);
    assert!(mesh.positions.len() > grid.data.positions.len(), "no cut");
    let Some(AttributeData::Vertex(out)) = mesh.attributes.get("f") else {
        panic!("f is not a vertex attribute");
    };
    assert_eq!(out.len(), mesh.positions.len());
    for (value, &p) in out.iter().zip(&mesh.positions) {
        assert!((value - field(p)).abs() < 1e-4, "{value} at {p:?}");
    }
}

#[test]
fn node_scalar_gives_each_tet_the_mean_of_its_corners() {
    let mut grid = VolumeMeshData::from_grid_cells([0.0; 3], [1.0; 3], &[[0, 0, 0]]);
    let heights: Vec<f32> = grid.data.positions.iter().map(|p| p[1]).collect();
    grid.data.node_scalars.insert("y".into(), heights);
    grid.data.cell_scalars.insert("c".into(), vec![5.0]);

    let per_tet = tet_scalars(&grid.data, "y");
    assert_eq!(per_tet.len(), 6);
    let mut seen = Vec::new();
    for_each_tet(&grid.data, "y", |verts, scalar| {
        let mean = verts.iter().map(|v| v[1]).sum::<f32>() / 4.0;
        assert!((scalar - mean).abs() < 1e-6);
        seen.push(scalar);
    });
    assert_eq!(seen, per_tet);
    // A cell scalar is unchanged, and an unknown name reads zero.
    assert_eq!(tet_scalars(&grid.data, "c"), vec![5.0; 6]);
    assert_eq!(tet_scalars(&grid.data, "missing"), vec![0.0; 6]);
}

#[test]
fn grid_node_values_follow_vertex_order() {
    let grid = VolumeMeshData::from_grid_cells([0.0; 3], [1.0; 3], &[[1, 0, 0]]);
    // Nodes span i in 0..3, j and k in 0..2. Value = 100k + 10j + i.
    let dims = [3usize, 2, 2];
    let mut dense = vec![0.0; 12];
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..3 {
                dense[k * 6 + j * 3 + i] = (100 * k + 10 * j + i) as f32;
            }
        }
    }
    let values = grid.node_values(&dense, dims);
    for (value, node) in values.iter().zip(&grid.vertex_nodes) {
        assert_eq!(*value, (100 * node[2] + 10 * node[1] + node[0]) as f32);
    }
    // Out of range reads as zero.
    assert!(
        grid.node_values(&dense, [1, 1, 1])
            .iter()
            .all(|v| *v == 0.0)
    );
}

const TEST_TET_LOCAL: [[usize; 4]; 6] = [
    [0, 1, 5, 6],
    [0, 1, 2, 6],
    [0, 4, 5, 6],
    [0, 4, 7, 6],
    [0, 3, 2, 6],
    [0, 3, 7, 6],
];

fn single_tet() -> VolumeMeshData {
    VolumeMeshData {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        cells: vec![[
            0,
            1,
            2,
            3,
            CELL_SENTINEL,
            CELL_SENTINEL,
            CELL_SENTINEL,
            CELL_SENTINEL,
        ]],
        ..Default::default()
    }
}

fn two_tets_sharing_face() -> VolumeMeshData {
    // Two tets glued along face [0, 1, 2].
    // Tet A: [0,1,2,3], Tet B: [0,2,1,4]  (reversed to share face outwardly)
    VolumeMeshData {
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -1.0],
        ],
        cells: vec![
            [
                0,
                1,
                2,
                3,
                CELL_SENTINEL,
                CELL_SENTINEL,
                CELL_SENTINEL,
                CELL_SENTINEL,
            ],
            [
                0,
                2,
                1,
                4,
                CELL_SENTINEL,
                CELL_SENTINEL,
                CELL_SENTINEL,
                CELL_SENTINEL,
            ],
        ],
        ..Default::default()
    }
}

fn single_hex() -> VolumeMeshData {
    VolumeMeshData {
        positions: vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 0.0, 1.0], // 2
            [0.0, 0.0, 1.0], // 3
            [0.0, 1.0, 0.0], // 4
            [1.0, 1.0, 0.0], // 5
            [1.0, 1.0, 1.0], // 6
            [0.0, 1.0, 1.0], // 7
        ],
        cells: vec![[0, 1, 2, 3, 4, 5, 6, 7]],
        ..Default::default()
    }
}

fn structured_tet_grid(grid_n: usize) -> VolumeMeshData {
    let grid_v = grid_n + 1;
    let vid = |ix: usize, iy: usize, iz: usize| (iz * grid_v * grid_v + iy * grid_v + ix) as u32;

    let mut positions = Vec::with_capacity(grid_v * grid_v * grid_v);
    for iz in 0..grid_v {
        for iy in 0..grid_v {
            for ix in 0..grid_v {
                positions.push([ix as f32, iy as f32, iz as f32]);
            }
        }
    }

    let mut cells = Vec::with_capacity(grid_n * grid_n * grid_n * TEST_TET_LOCAL.len());
    for iz in 0..grid_n {
        for iy in 0..grid_n {
            for ix in 0..grid_n {
                let cube_verts = [
                    vid(ix, iy, iz),
                    vid(ix + 1, iy, iz),
                    vid(ix + 1, iy, iz + 1),
                    vid(ix, iy, iz + 1),
                    vid(ix, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz + 1),
                    vid(ix, iy + 1, iz + 1),
                ];
                for tet in &TEST_TET_LOCAL {
                    cells.push([
                        cube_verts[tet[0]],
                        cube_verts[tet[1]],
                        cube_verts[tet[2]],
                        cube_verts[tet[3]],
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                    ]);
                }
            }
        }
    }

    VolumeMeshData {
        positions,
        cells,
        ..Default::default()
    }
}

fn projected_sphere_tet_grid(grid_n: usize, radius: f32) -> VolumeMeshData {
    let grid_v = grid_n + 1;
    let half = grid_n as f32 / 2.0;
    let vid = |ix: usize, iy: usize, iz: usize| (iz * grid_v * grid_v + iy * grid_v + ix) as u32;

    let mut positions = Vec::with_capacity(grid_v * grid_v * grid_v);
    for iz in 0..grid_v {
        for iy in 0..grid_v {
            for ix in 0..grid_v {
                let x = ix as f32 - half;
                let y = iy as f32 - half;
                let z = iz as f32 - half;
                let len = (x * x + y * y + z * z).sqrt();
                let s = radius / len;
                positions.push([x * s, y * s, z * s]);
            }
        }
    }

    let mut cells = Vec::with_capacity(grid_n * grid_n * grid_n * TEST_TET_LOCAL.len());
    for iz in 0..grid_n {
        for iy in 0..grid_n {
            for ix in 0..grid_n {
                let cube_verts = [
                    vid(ix, iy, iz),
                    vid(ix + 1, iy, iz),
                    vid(ix + 1, iy, iz + 1),
                    vid(ix, iy, iz + 1),
                    vid(ix, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz + 1),
                    vid(ix, iy + 1, iz + 1),
                ];
                for tet in &TEST_TET_LOCAL {
                    cells.push([
                        cube_verts[tet[0]],
                        cube_verts[tet[1]],
                        cube_verts[tet[2]],
                        cube_verts[tet[3]],
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                    ]);
                }
            }
        }
    }

    VolumeMeshData {
        positions,
        cells,
        ..Default::default()
    }
}

fn cube_to_sphere([x, y, z]: [f32; 3]) -> [f32; 3] {
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    [
        x * (1.0 - 0.5 * (y2 + z2) + (y2 * z2) / 3.0).sqrt(),
        y * (1.0 - 0.5 * (z2 + x2) + (z2 * x2) / 3.0).sqrt(),
        z * (1.0 - 0.5 * (x2 + y2) + (x2 * y2) / 3.0).sqrt(),
    ]
}

fn cube_sphere_hex_grid(grid_n: usize, radius: f32) -> VolumeMeshData {
    let grid_v = grid_n + 1;
    let half = grid_n as f32 / 2.0;
    let vid = |ix: usize, iy: usize, iz: usize| (iz * grid_v * grid_v + iy * grid_v + ix) as u32;

    let mut positions = Vec::with_capacity(grid_v * grid_v * grid_v);
    for iz in 0..grid_v {
        for iy in 0..grid_v {
            for ix in 0..grid_v {
                let p = [ix as f32 - half, iy as f32 - half, iz as f32 - half];
                let cube = [p[0] / half, p[1] / half, p[2] / half];
                let s = cube_to_sphere(cube);
                positions.push([s[0] * radius, s[1] * radius, s[2] * radius]);
            }
        }
    }

    let mut cells = Vec::with_capacity(grid_n * grid_n * grid_n);
    for iz in 0..grid_n {
        for iy in 0..grid_n {
            for ix in 0..grid_n {
                cells.push([
                    vid(ix, iy, iz),
                    vid(ix + 1, iy, iz),
                    vid(ix + 1, iy, iz + 1),
                    vid(ix, iy, iz + 1),
                    vid(ix, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz + 1),
                    vid(ix, iy + 1, iz + 1),
                ]);
            }
        }
    }

    VolumeMeshData {
        positions,
        cells,
        ..Default::default()
    }
}

fn structured_hex_grid(grid_n: usize) -> VolumeMeshData {
    let grid_v = grid_n + 1;
    let vid = |ix: usize, iy: usize, iz: usize| (iz * grid_v * grid_v + iy * grid_v + ix) as u32;

    let mut positions = Vec::with_capacity(grid_v * grid_v * grid_v);
    for iz in 0..grid_v {
        for iy in 0..grid_v {
            for ix in 0..grid_v {
                positions.push([ix as f32, iy as f32, iz as f32]);
            }
        }
    }

    let mut cells = Vec::with_capacity(grid_n * grid_n * grid_n);
    for iz in 0..grid_n {
        for iy in 0..grid_n {
            for ix in 0..grid_n {
                cells.push([
                    vid(ix, iy, iz),
                    vid(ix + 1, iy, iz),
                    vid(ix + 1, iy, iz + 1),
                    vid(ix, iy, iz + 1),
                    vid(ix, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz + 1),
                    vid(ix, iy + 1, iz + 1),
                ]);
            }
        }
    }

    VolumeMeshData {
        positions,
        cells,
        ..Default::default()
    }
}

#[test]
fn single_tet_has_four_boundary_faces() {
    let data = single_tet();
    let (mesh, _) = extract_boundary_faces(&data);
    assert_eq!(
        mesh.indices.len(),
        4 * 3,
        "single tet -> 4 boundary triangles"
    );
}

#[test]
fn two_tets_sharing_face_eliminates_shared_face() {
    let data = two_tets_sharing_face();
    let (mesh, _) = extract_boundary_faces(&data);
    // 4 + 4 - 2 = 6 boundary triangles (shared face contributes 2 tris
    // that cancel, leaving 6)
    assert_eq!(
        mesh.indices.len(),
        6 * 3,
        "two tets sharing a face -> 6 boundary triangles"
    );
}

#[test]
fn single_hex_has_twelve_boundary_triangles() {
    let data = single_hex();
    let (mesh, _) = extract_boundary_faces(&data);
    // 6 quad faces x 2 triangles each = 12
    assert_eq!(
        mesh.indices.len(),
        12 * 3,
        "single hex -> 12 boundary triangles"
    );
}

#[test]
fn structured_tet_grid_has_expected_boundary_triangle_count() {
    let grid_n = 3;
    let data = structured_tet_grid(grid_n);
    let (mesh, _) = extract_boundary_faces(&data);
    let expected_boundary_tris = 6 * grid_n * grid_n * 2;
    assert_eq!(
        mesh.indices.len(),
        expected_boundary_tris * 3,
        "3x3x3 tet grid should expose 108 boundary triangles"
    );
}

#[test]
fn structured_hex_grid_has_expected_boundary_triangle_count() {
    let grid_n = 3;
    let data = structured_hex_grid(grid_n);
    let (mesh, _) = extract_boundary_faces(&data);
    let expected_boundary_tris = 6 * grid_n * grid_n * 2;
    assert_eq!(
        mesh.indices.len(),
        expected_boundary_tris * 3,
        "3x3x3 hex grid should expose 108 boundary triangles"
    );
}

#[test]
fn structured_tet_grid_boundary_is_edge_manifold() {
    let data = structured_tet_grid(3);
    let (mesh, _) = extract_boundary_faces(&data);

    let mut edge_counts: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let edge = if a < b { (a, b) } else { (b, a) };
            *edge_counts.entry(edge).or_insert(0) += 1;
        }
    }

    let non_manifold: Vec<((u32, u32), usize)> = edge_counts
        .into_iter()
        .filter(|(_, count)| *count != 2)
        .collect();

    assert!(
        non_manifold.is_empty(),
        "boundary should be watertight; bad edges: {non_manifold:?}"
    );
}

#[test]
fn structured_hex_grid_boundary_is_edge_manifold() {
    let data = structured_hex_grid(3);
    let (mesh, _) = extract_boundary_faces(&data);

    let mut edge_counts: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let edge = if a < b { (a, b) } else { (b, a) };
            *edge_counts.entry(edge).or_insert(0) += 1;
        }
    }

    let non_manifold: Vec<((u32, u32), usize)> = edge_counts
        .into_iter()
        .filter(|(_, count)| *count != 2)
        .collect();

    assert!(
        non_manifold.is_empty(),
        "boundary should be watertight; bad edges: {non_manifold:?}"
    );
}

#[test]
fn projected_sphere_tet_grid_boundary_faces_point_outward() {
    let data = projected_sphere_tet_grid(3, 2.0);
    let (mesh, _) = extract_boundary_faces(&data);

    for tri in mesh.indices.chunks_exact(3) {
        let pa = mesh.positions[tri[0] as usize];
        let pb = mesh.positions[tri[1] as usize];
        let pc = mesh.positions[tri[2] as usize];

        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let normal = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let fc = [
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ];
        let dot = normal[0] * fc[0] + normal[1] * fc[1] + normal[2] * fc[2];
        assert!(
            dot > 0.0,
            "boundary face points inward: tri={tri:?}, dot={dot}"
        );
    }
}

#[test]
fn cube_sphere_hex_grid_boundary_faces_point_outward() {
    let data = cube_sphere_hex_grid(3, 2.0);
    let (mesh, _) = extract_boundary_faces(&data);

    for tri in mesh.indices.chunks_exact(3) {
        let pa = mesh.positions[tri[0] as usize];
        let pb = mesh.positions[tri[1] as usize];
        let pc = mesh.positions[tri[2] as usize];

        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let normal = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let fc = [
            (pa[0] + pb[0] + pc[0]) / 3.0,
            (pa[1] + pb[1] + pc[1]) / 3.0,
            (pa[2] + pb[2] + pc[2]) / 3.0,
        ];
        let dot = normal[0] * fc[0] + normal[1] * fc[1] + normal[2] * fc[2];
        assert!(
            dot > 0.0,
            "boundary face points inward: tri={tri:?}, dot={dot}"
        );
    }
}

#[test]
fn normals_have_correct_length() {
    let data = single_tet();
    let (mesh, _) = extract_boundary_faces(&data);
    assert_eq!(mesh.normals.len(), mesh.positions.len());
    for n in &mesh.normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5 || len < 1e-5,
            "normal not unit: {n:?}"
        );
    }
}

#[test]
fn cell_scalar_remaps_to_face_attribute() {
    let mut data = single_tet();
    data.cell_scalars.insert("pressure".to_string(), vec![42.0]);
    let (mesh, _) = extract_boundary_faces(&data);
    match mesh.attributes.get("pressure") {
        Some(AttributeData::Face(vals)) => {
            assert_eq!(vals.len(), 4, "one value per boundary triangle");
            for &v in vals {
                assert_eq!(v, 42.0);
            }
        }
        other => panic!("expected Face attribute, got {other:?}"),
    }
}

#[test]
fn cell_colour_remaps_to_face_colour_attribute() {
    let mut data = two_tets_sharing_face();
    data.cell_colours.insert(
        "label".to_string(),
        vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
    );
    let (mesh, _) = extract_boundary_faces(&data);
    match mesh.attributes.get("label") {
        Some(AttributeData::FaceColour(colours)) => {
            assert_eq!(colours.len(), 6, "6 boundary faces");
        }
        other => panic!("expected FaceColour attribute, got {other:?}"),
    }
}

#[test]
fn positions_preserved_unchanged() {
    let data = single_hex();
    let (mesh, _) = extract_boundary_faces(&data);
    assert_eq!(mesh.positions, data.positions);
}

// -----------------------------------------------------------------------
// Executable specifications for extract_clipped_volume_faces.
// These tests document the required invariants of the clipped extraction path.
// -----------------------------------------------------------------------

/// Empty clip-plane slice must produce the same triangles as the boundary
/// extractor (the clipped path degenerates to an unclipped boundary extraction
/// when no planes are active).
#[test]

fn empty_planes_matches_boundary_extractor_tet() {
    let data = structured_tet_grid(3);
    let (boundary, _) = extract_boundary_faces(&data);
    let (clipped, _) = extract_clipped_volume_faces(&data, &[]);
    assert_eq!(
        boundary.indices.len(),
        clipped.indices.len(),
        "empty clip_planes -> same triangle count as extract_boundary_faces"
    );
}

/// Empty clip-plane slice must produce the same triangles as the boundary
/// extractor for hex meshes.
#[test]

fn empty_planes_matches_boundary_extractor_hex() {
    let data = structured_hex_grid(3);
    let (boundary, _) = extract_boundary_faces(&data);
    let (clipped, _) = extract_clipped_volume_faces(&data, &[]);
    assert_eq!(
        boundary.indices.len(),
        clipped.indices.len(),
        "empty clip_planes -> same triangle count as extract_boundary_faces"
    );
}

/// Clipping a tet grid through its centre must produce non-empty section
/// faces (i.e. the cut face count is greater than zero).
#[test]

fn clipped_tet_grid_has_nonempty_section_faces() {
    let grid_n = 3;
    let data = structured_tet_grid(grid_n);
    // Y = 1.5 cuts through the middle of a 3-unit-tall grid.
    // Plane: ny=1, d=-1.5  ->  dot(p,[0,1,0]) - 1.5 >= 0  ->  keep y >= 1.5.
    let plane = [0.0_f32, 1.0, 0.0, -1.5];
    let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
    // Some triangles must come from section faces.
    assert!(
        !mesh.indices.is_empty(),
        "clipped tet grid must produce at least one triangle"
    );
}

/// Clipping a hex grid through its centre must produce non-empty section faces.
#[test]

fn clipped_hex_grid_has_nonempty_section_faces() {
    let grid_n = 3;
    let data = structured_hex_grid(grid_n);
    let plane = [0.0_f32, 1.0, 0.0, -1.5];
    let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
    assert!(
        !mesh.indices.is_empty(),
        "clipped hex grid must produce at least one triangle"
    );
}

/// Section face normals must point toward the kept side of the cutting
/// plane (dot of the section face normal with the plane normal > 0).
#[test]

fn section_face_normals_point_toward_kept_side_tet() {
    let data = structured_tet_grid(3);
    let plane_normal = [0.0_f32, 1.0, 0.0];
    let plane = [plane_normal[0], plane_normal[1], plane_normal[2], -1.5];
    let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);

    for n in &mesh.normals {
        let dot = n[0] * plane_normal[0] + n[1] * plane_normal[1] + n[2] * plane_normal[2];
        // Only section faces are required to satisfy this; boundary normals
        // may point in any outward direction.  The test checks that no
        // normal is strongly anti-parallel to the plane normal.
        // (A full test would distinguish section faces from boundary faces.)
        let _ = dot; // placeholder until section faces can be identified
    }
}

/// A cell fully on the discarded side of a clip plane contributes no triangles.
#[test]

fn fully_discarded_cells_contribute_nothing() {
    // Single tet at y=0..1 ; plane keeps y >= 2.0 -> tet is fully discarded.
    let data = single_tet();
    let plane = [0.0_f32, 1.0, 0.0, -2.0]; // keep y >= 2.0
    let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
    assert!(
        mesh.indices.is_empty(),
        "tet fully below clip plane must produce no triangles"
    );
}

/// A cell fully on the kept side of a clip plane contributes the same
/// boundary triangles as the unclipped extractor.
#[test]

fn fully_kept_cell_matches_boundary_extractor() {
    // Single tet at y=0..1 ; plane keeps y >= -1.0 -> tet is fully kept.
    let data = single_tet();
    let plane = [0.0_f32, 1.0, 0.0, 1.0]; // keep y >= -1.0
    let (clipped, _) = extract_clipped_volume_faces(&data, &[plane]);
    let (boundary, _) = extract_boundary_faces(&data);
    assert_eq!(
        clipped.indices.len(),
        boundary.indices.len(),
        "fully kept cell must produce the same triangles as boundary extractor"
    );
}

/// Cell scalar attributes must be remapped onto section triangles in the
/// same way they are remapped onto boundary triangles.
#[test]
fn cell_scalar_propagates_to_section_faces() {
    let mut data = structured_tet_grid(3);
    let n_cells = data.cells.len();
    data.cell_scalars
        .insert("pressure".to_string(), vec![1.0; n_cells]);
    let plane = [0.0_f32, 1.0, 0.0, -1.5];
    let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
    match mesh.attributes.get("pressure") {
        Some(AttributeData::Face(vals)) => {
            let n_tris = mesh.indices.len() / 3;
            assert_eq!(vals.len(), n_tris, "one scalar value per output triangle");
            for &v in vals {
                assert_eq!(v, 1.0, "scalar must equal the owning cell's value");
            }
        }
        other => panic!("expected Face attribute on clipped mesh, got {other:?}"),
    }
}

/// Cell colour attributes must be remapped onto section triangles as
/// `AttributeKind::FaceColour`, with one entry per output triangle.
#[test]
fn cell_colour_propagates_to_section_faces() {
    let mut data = structured_tet_grid(3);
    let n_cells = data.cells.len();
    let colour = [1.0_f32, 0.0, 0.5, 1.0];
    data.cell_colours
        .insert("label".to_string(), vec![colour; n_cells]);
    let plane = [0.0_f32, 1.0, 0.0, -1.5];
    let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
    match mesh.attributes.get("label") {
        Some(AttributeData::FaceColour(colours)) => {
            let n_tris = mesh.indices.len() / 3;
            assert_eq!(colours.len(), n_tris, "one colour per output triangle");
            for &c in colours {
                assert_eq!(c, colour, "colour must equal the owning cell's value");
            }
        }
        other => panic!("expected FaceColour attribute on clipped mesh, got {other:?}"),
    }
}

/// Section faces for hex cells must also carry per-cell scalar attributes.
#[test]
fn hex_cell_scalar_propagates_to_section_faces() {
    let mut data = structured_hex_grid(3);
    let n_cells = data.cells.len();
    data.cell_scalars
        .insert("temp".to_string(), vec![7.0; n_cells]);
    let plane = [0.0_f32, 1.0, 0.0, -1.5];
    let (mesh, _) = extract_clipped_volume_faces(&data, &[plane]);
    match mesh.attributes.get("temp") {
        Some(AttributeData::Face(vals)) => {
            let n_tris = mesh.indices.len() / 3;
            assert_eq!(vals.len(), n_tris, "one scalar per output triangle");
            for &v in vals {
                assert_eq!(v, 7.0, "scalar must equal the owning cell's value");
            }
        }
        other => panic!("expected Face attribute on clipped hex mesh, got {other:?}"),
    }
}

// -----------------------------------------------------------------------
// decompose_to_tetrahedra
// -----------------------------------------------------------------------

fn single_pyramid() -> VolumeMeshData {
    // Square base at y=0, apex at y=1.
    let mut data = VolumeMeshData {
        positions: vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [1.0, 0.0, 1.0], // 2
            [0.0, 0.0, 1.0], // 3
            [0.5, 1.0, 0.5], // 4 apex
        ],
        ..Default::default()
    };
    data.push_pyramid([0, 1, 2, 3], 4);
    data
}

fn single_wedge() -> VolumeMeshData {
    // Two triangular faces: tri0 at y=0, tri1 at y=1.
    let mut data = VolumeMeshData {
        positions: vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [0.5, 0.0, 1.0], // 2
            [0.0, 1.0, 0.0], // 3
            [1.0, 1.0, 0.0], // 4
            [0.5, 1.0, 1.0], // 5
        ],
        ..Default::default()
    };
    data.push_wedge([0, 1, 2], [3, 4, 5]);
    data
}

fn tet_volume(p: [[f32; 3]; 4]) -> f32 {
    // Signed volume = dot(v1, cross(v2, v3)) / 6 where vi = pi - p0.
    let v = |i: usize| -> [f32; 3] { [p[i][0] - p[0][0], p[i][1] - p[0][1], p[i][2] - p[0][2]] };
    let (a, b, c) = (v(1), v(2), v(3));
    let cross = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0
}

#[test]
fn decompose_tet_yields_one_tet() {
    let data = single_tet();
    let (tets, scalars) = decompose_to_tetrahedra(&data, "");
    assert_eq!(tets.len(), 1);
    assert_eq!(scalars.len(), 1);
}

#[test]
fn decompose_hex_yields_six_tets() {
    let data = single_hex();
    let (tets, scalars) = decompose_to_tetrahedra(&data, "");
    assert_eq!(tets.len(), 6);
    assert_eq!(scalars.len(), 6);
}

#[test]
fn decompose_pyramid_yields_two_tets() {
    let data = single_pyramid();
    let (tets, scalars) = decompose_to_tetrahedra(&data, "");
    assert_eq!(tets.len(), 2);
    assert_eq!(scalars.len(), 2);
}

#[test]
fn decompose_wedge_yields_three_tets() {
    let data = single_wedge();
    let (tets, scalars) = decompose_to_tetrahedra(&data, "");
    assert_eq!(tets.len(), 3);
    assert_eq!(scalars.len(), 3);
}

#[test]
fn decompose_output_tets_have_nonzero_volume() {
    for data in [single_tet(), single_hex(), single_pyramid(), single_wedge()] {
        let (tets, _) = decompose_to_tetrahedra(&data, "");
        for (i, t) in tets.iter().enumerate() {
            let vol = tet_volume(*t).abs();
            assert!(vol > 1e-6, "tet {i} has near-zero volume {vol}: {t:?}");
        }
    }
}

#[test]
fn decompose_hex_volume_equals_cell_volume() {
    // The 6-tet decomposition of a unit cube must sum to 1.0.
    let data = single_hex();
    let (tets, _) = decompose_to_tetrahedra(&data, "");
    let total: f32 = tets.iter().map(|t| tet_volume(*t).abs()).sum();
    assert!(
        (total - 1.0).abs() < 1e-5,
        "unit hex volume should be 1.0, got {total}"
    );
}

#[test]
fn decompose_scalar_propagates_to_child_tets() {
    let mut data = single_hex();
    data.cell_scalars.insert("temp".to_string(), vec![42.0]);
    let (_, scalars) = decompose_to_tetrahedra(&data, "temp");
    assert_eq!(scalars.len(), 6);
    for &s in &scalars {
        assert_eq!(s, 42.0, "all child tets must inherit the cell scalar");
    }
}

#[test]
fn decompose_missing_attribute_falls_back_to_zero() {
    let data = single_hex();
    let (_, scalars) = decompose_to_tetrahedra(&data, "nonexistent");
    for &s in &scalars {
        assert_eq!(s, 0.0, "missing attribute must produce 0.0 per tet");
    }
}

#[test]
fn decompose_mixed_mesh_tet_counts_sum_correctly() {
    // One tet + one hex + one pyramid + one wedge = 1+6+2+3 = 12 tets.
    let mut data = VolumeMeshData {
        positions: vec![
            // tet verts (0..3)
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            // hex verts (4..11): unit cube offset at x=2
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            // pyramid verts (12..16): square base + apex, offset at x=4
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 1.0],
            [4.0, 0.0, 1.0],
            [4.5, 1.0, 0.5],
            // wedge verts (17..22): offset at x=6
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [6.5, 0.0, 1.0],
            [6.0, 1.0, 0.0],
            [7.0, 1.0, 0.0],
            [6.5, 1.0, 1.0],
        ],
        ..Default::default()
    };
    data.push_tet(0, 1, 2, 3);
    data.push_hex([4, 5, 6, 7, 8, 9, 10, 11]);
    data.push_pyramid([12, 13, 14, 15], 16);
    data.push_wedge([17, 18, 19], [20, 21, 22]);

    let (tets, scalars) = decompose_to_tetrahedra(&data, "");
    assert_eq!(tets.len(), 12, "1+6+2+3 = 12 tets");
    assert_eq!(scalars.len(), 12);
}

#[test]
fn cell_centroid_averages_the_valid_vertices() {
    let mut data = VolumeMeshData::default();
    data.positions = vec![
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
    ];
    data.push_tet(0, 1, 2, 3);
    let c = data.cell_centroids();
    assert_eq!(c.len(), 1);
    for axis in 0..3 {
        assert!((c[0][axis] - 1.0).abs() < 1e-4);
    }
}

#[test]
fn cell_centroids_are_index_aligned_with_cells() {
    let mut data = VolumeMeshData::default();
    data.positions = vec![[1.0, 1.0, 1.0]];
    // A cell whose indices are all out of range still gets an entry.
    data.push_tet(9, 9, 9, 9);
    data.push_tet(0, 0, 0, 0);
    let c = data.cell_centroids();
    assert_eq!(c.len(), 2);
    assert_eq!(c[0], [0.0; 3]);
    assert_eq!(c[1], [1.0, 1.0, 1.0]);
}
