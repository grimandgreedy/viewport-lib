//! Isosurfaces of a node scalar through an unstructured volume mesh, by
//! marching tetrahedra over the cells split into tets.

use super::VolumeMeshData;
use super::tets::cell_tets;
use glam::Vec3;
use std::collections::HashMap;
use viewport_lib_types::data::{attribute::AttributeData, mesh::MeshData};

/// The surface inside `data` where the node scalar `node_scalar` equals
/// `isovalue`, as an ordinary triangle mesh.
///
/// Every cell is split into tets (a hex by the Freudenthal split, as the
/// transparent volume path does) and each tet the isovalue passes through
/// contributes one triangle or two. A cut edge shared by neighbouring tets
/// gives one vertex, so the surface is connected and its normals are smooth.
/// Triangles are wound so their normals point toward higher values, the
/// direction the field increases in, as `extract_isosurface`'s do on a grid.
///
/// The mesh carries the volume's data as attributes: each node scalar
/// interpolated along the cut edges as an [`AttributeData::Vertex`] (so the
/// surface can be coloured by a second field), and each cell scalar and cell
/// colour as an [`AttributeData::Face`] / [`AttributeData::FaceColour`] of the
/// cell a triangle lies in.
///
/// Returns an empty mesh when `data` has no node scalar of that name. A tet
/// with a NaN corner contributes nothing. Re-extract when the isovalue
/// changes; the cost is one pass over the cells.
pub fn extract_volume_mesh_isosurface(
    data: &VolumeMeshData,
    node_scalar: &str,
    isovalue: f32,
) -> MeshData {
    let Some(values) = data.node_scalars.get(node_scalar) else {
        return MeshData::new(Vec::new(), Vec::new(), Vec::new());
    };
    let value = |node: u32| values.get(node as usize).copied().unwrap_or(f32::NAN);

    let mut cuts = EdgeCuts::default();
    let mut indices: Vec<u32> = Vec::new();
    let mut face_cell: Vec<usize> = Vec::new();

    for (cell_idx, cell) in data.cells.iter().enumerate() {
        for tet in cell_tets(cell) {
            let nodes = tet.map(|slot| cell[slot]);
            let v = nodes.map(value);
            if v.iter().any(|x| x.is_nan()) {
                continue;
            }
            let (below, above): (Vec<usize>, Vec<usize>) = (0..4).partition(|&k| v[k] < isovalue);
            let mut cut =
                |a: usize, b: usize| cuts.vertex(data, nodes[a], nodes[b], v[a], v[b], isovalue);
            let polygon: Vec<u32> = match (below.len(), above.len()) {
                // One corner apart from the other three: a triangle around it.
                (1, 3) | (3, 1) => {
                    let (lone, rest) = if below.len() == 1 {
                        (below[0], &above)
                    } else {
                        (above[0], &below)
                    };
                    rest.iter().map(|&k| cut(lone, k)).collect()
                }
                // Two and two: a quad, its corners in order around the cut.
                (2, 2) => {
                    let ([a0, a1], [b0, b1]) = ([below[0], below[1]], [above[0], above[1]]);
                    vec![cut(a0, b0), cut(a0, b1), cut(a1, b1), cut(a1, b0)]
                }
                _ => continue,
            };

            // Wind toward higher values: from the low corners to the high ones.
            let centroid = |ks: &[usize]| {
                ks.iter()
                    .map(|&k| Vec3::from(data.positions[nodes[k] as usize]))
                    .sum::<Vec3>()
                    / ks.len() as f32
            };
            let uphill = centroid(&above) - centroid(&below);
            for i in 1..polygon.len() - 1 {
                let mut tri = [polygon[0], polygon[i], polygon[i + 1]];
                let [p0, p1, p2] = tri.map(|vi| Vec3::from(cuts.positions[vi as usize]));
                if (p1 - p0).cross(p2 - p0).dot(uphill) < 0.0 {
                    tri.swap(1, 2);
                }
                indices.extend(tri);
                face_cell.push(cell_idx);
            }
        }
    }

    // Area-weighted vertex normals over the welded surface.
    let mut normals = vec![Vec3::ZERO; cuts.positions.len()];
    for tri in indices.chunks_exact(3) {
        let [p0, p1, p2] =
            [tri[0], tri[1], tri[2]].map(|vi| Vec3::from(cuts.positions[vi as usize]));
        let n = (p1 - p0).cross(p2 - p0);
        for &vi in tri {
            normals[vi as usize] += n;
        }
    }
    let normals = normals
        .into_iter()
        .map(|n| n.try_normalize().unwrap_or(Vec3::Z).to_array())
        .collect();

    let mut attributes: HashMap<String, AttributeData> = HashMap::new();
    for (name, node_vals) in &data.node_scalars {
        let at = |node: u32| node_vals.get(node as usize).copied().unwrap_or(0.0);
        let interpolated = cuts
            .sources
            .iter()
            .map(|&(a, b, t)| at(a) + (at(b) - at(a)) * t)
            .collect();
        attributes.insert(name.clone(), AttributeData::Vertex(interpolated));
    }
    for (name, cell_vals) in &data.cell_scalars {
        let per_face = face_cell
            .iter()
            .map(|&ci| cell_vals.get(ci).copied().unwrap_or(0.0))
            .collect();
        attributes.insert(name.clone(), AttributeData::Face(per_face));
    }
    for (name, cell_vals) in &data.cell_colours {
        let per_face = face_cell
            .iter()
            .map(|&ci| cell_vals.get(ci).copied().unwrap_or([1.0; 4]))
            .collect();
        attributes.insert(name.clone(), AttributeData::FaceColour(per_face));
    }

    let mut mesh = MeshData::new(cuts.positions, normals, indices);
    mesh.attributes = attributes;
    mesh
}

/// The surface's vertices, one per cut mesh edge.
#[derive(Default)]
struct EdgeCuts {
    /// Vertex index by mesh edge, the lower node first.
    by_edge: HashMap<(u32, u32), u32>,
    positions: Vec<[f32; 3]>,
    /// Per vertex, the edge's two nodes and how far along it the cut lies.
    sources: Vec<(u32, u32, f32)>,
}

impl EdgeCuts {
    /// The vertex where the isovalue crosses the edge between `a` and `b`,
    /// whose values `va` and `vb` lie on either side of it.
    fn vertex(
        &mut self,
        data: &VolumeMeshData,
        a: u32,
        b: u32,
        va: f32,
        vb: f32,
        isovalue: f32,
    ) -> u32 {
        // Orient the edge so both tets sharing it compute the same point.
        let (a, b, va, vb) = if a < b {
            (a, b, va, vb)
        } else {
            (b, a, vb, va)
        };
        *self.by_edge.entry((a, b)).or_insert_with(|| {
            let t = ((isovalue - va) / (vb - va)).clamp(0.0, 1.0);
            let pa = Vec3::from(data.positions[a as usize]);
            let pb = Vec3::from(data.positions[b as usize]);
            self.positions.push(pa.lerp(pb, t).to_array());
            self.sources.push((a, b, t));
            (self.positions.len() - 1) as u32
        })
    }
}
