//! Pure CPU mesh operations: tangent computation, attribute expansion, and
//! `MeshData` validation.
//!
//! These operate on plain slices and `MeshData`, with no GPU dependency, so a
//! loader or mesh-processing consumer can run them without the renderer. The
//! renderer's upload path calls them before handing data to the GPU.

use viewport_lib_types::data::mesh::MeshData;
use viewport_lib_types::error::{ViewportError, ViewportResult};

/// Expand N face scalar values to 3N by repeating each value three times.
pub fn expand_face_scalars_to_3n(values: &[f32], n_tris: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_tris * 3);
    for i in 0..n_tris {
        let v = values.get(i).copied().unwrap_or(0.0);
        out.push(v);
        out.push(v);
        out.push(v);
    }
    out
}

/// Expand N face RGBA colours to 3N by repeating each colour three times.
pub fn expand_face_colours_to_3n(colours: &[[f32; 4]], n_tris: usize) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(n_tris * 3);
    for i in 0..n_tris {
        let c = colours.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]);
        out.push(c);
        out.push(c);
        out.push(c);
    }
    out
}

/// Expand per-directed-edge scalars to per-vertex by averaging over incident edges.
///
/// Edge ordering: `edge_values[3*t + k]` is the k-th edge of triangle `t`,
/// running from vertex `k` to vertex `(k+1)%3` of that triangle.
/// Each edge's value is added to both endpoint vertices; the final per-vertex
/// value is the average over all incident edge contributions.
pub fn expand_edge_to_vertex(
    edge_values: &[f32],
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Vec<f32> {
    let n = positions.len();
    let mut sum = vec![0.0f32; n];
    let mut count = vec![0u32; n];
    for (tri_idx, chunk) in indices.chunks(3).enumerate() {
        for k in 0..3 {
            let v = edge_values.get(3 * tri_idx + k).copied().unwrap_or(0.0);
            let vi0 = chunk[k] as usize;
            let vi1 = chunk[(k + 1) % 3] as usize;
            if vi0 < n {
                sum[vi0] += v;
                count[vi0] += 1;
            }
            if vi1 < n {
                sum[vi1] += v;
                count[vi1] += 1;
            }
        }
    }
    (0..n)
        .map(|i| {
            if count[i] > 0 {
                sum[i] / count[i] as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Expand per-cell (per-triangle) scalar values to per-vertex by averaging contributions.
pub fn expand_cell_to_vertex(
    cell_values: &[f32],
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Vec<f32> {
    let n = positions.len();
    let mut sum = vec![0.0f32; n];
    let mut count = vec![0u32; n];
    for (tri_idx, chunk) in indices.chunks(3).enumerate() {
        let v = cell_values.get(tri_idx).copied().unwrap_or(0.0);
        for &vi in chunk {
            let vi = vi as usize;
            if vi < n {
                sum[vi] += v;
                count[vi] += 1;
            }
        }
    }
    (0..n)
        .map(|i| {
            if count[i] > 0 {
                sum[i] / count[i] as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute per-vertex tangents using Gram-Schmidt orthogonalization with handedness.
///
/// Returns a `Vec<[f32; 4]>` of length `positions.len()` where each element is
/// `[tx, ty, tz, w]` with `w = +/-1.0` encoding bitangent handedness.
///
/// Requires triangulated indices (every 3 indices = one triangle).
/// If any triangle is degenerate (zero-area or zero UV area), its contribution is skipped.
pub fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let n = positions.len();
    let tri_count = indices.len() / 3;

    // Accumulate sdir/tdir contributions per vertex. Sequential.
    //
    // **Do not** use rayon parallel iterators in this function. This
    // routine is already invoked from a rayon worker : every mesh
    // upload runs `prep_mesh_data -> compute_tangents` inside a
    // `submit_cpu` job (see `upload_jobs::Runner::submit_cpu`).
    // Adding intra-mesh parallelism causes nested rayon work:
    // a worker enters `par_chunks(3).fold(...)`, parks at a join,
    // steals another mesh's upload (which itself enters compute_tangents
    // and parks again), and so on. Each suspension keeps frames on
    // the worker's 2 MB stack; with the upload queue draining
    // concurrent tangent tasks, stack depth grows unboundedly and
    // overflows.
    //
    // The function is cache-friendly and runs at ~30 ns / triangle
    // sequentially (~ 15 ms for a 500 k-tri mesh). Per-mesh
    // parallelism comes from the upload job pool, not from inside
    // this function.
    let mut tan1 = vec![[0.0f32; 3]; n];
    let mut tan2 = vec![[0.0f32; 3]; n];
    for t in 0..tri_count {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];

        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let du1 = uv1[0] - uv0[0];
        let dv1 = uv1[1] - uv0[1];
        let du2 = uv2[0] - uv0[0];
        let dv2 = uv2[1] - uv0[1];

        let det = du1 * dv2 - du2 * dv1;
        if det.abs() < 1e-10 {
            continue;
        }
        let r = 1.0 / det;

        let sdir = [
            (dv2 * e1[0] - dv1 * e2[0]) * r,
            (dv2 * e1[1] - dv1 * e2[1]) * r,
            (dv2 * e1[2] - dv1 * e2[2]) * r,
        ];
        let tdir = [
            (du1 * e2[0] - du2 * e1[0]) * r,
            (du1 * e2[1] - du2 * e1[1]) * r,
            (du1 * e2[2] - du2 * e1[2]) * r,
        ];

        for &vi in &[i0, i1, i2] {
            for k in 0..3 {
                tan1[vi][k] += sdir[k];
                tan2[vi][k] += tdir[k];
            }
        }
    }

    // Gram-Schmidt orthogonalization per vertex. Sequential, for the
    // same nested-rayon reason as above.
    (0..n)
        .map(|i| {
            let n_v = normals[i];
            let t = tan1[i];
            let dot = n_v[0] * t[0] + n_v[1] * t[1] + n_v[2] * t[2];
            let tx = t[0] - n_v[0] * dot;
            let ty = t[1] - n_v[1] * dot;
            let tz = t[2] - n_v[2] * dot;
            let len = (tx * tx + ty * ty + tz * tz).sqrt();
            let (tx, ty, tz) = if len > 1e-7 {
                (tx / len, ty / len, tz / len)
            } else {
                (1.0, 0.0, 0.0)
            };
            let cx = n_v[1] * tz - n_v[2] * ty;
            let cy = n_v[2] * tx - n_v[0] * tz;
            let cz = n_v[0] * ty - n_v[1] * tx;
            let w = if cx * tan2[i][0] + cy * tan2[i][1] + cz * tan2[i][2] < 0.0 {
                -1.0
            } else {
                1.0
            };
            [tx, ty, tz, w]
        })
        .collect()
}

/// Validate mesh data before upload.
pub fn validate_mesh_data(data: &MeshData) -> ViewportResult<()> {
    if data.positions.is_empty() || data.indices.is_empty() {
        return Err(ViewportError::EmptyMesh {
            positions: data.positions.len(),
            indices: data.indices.len(),
        });
    }
    if data.positions.len() != data.normals.len() {
        return Err(ViewportError::MeshLengthMismatch {
            positions: data.positions.len(),
            normals: data.normals.len(),
        });
    }
    let vertex_count = data.positions.len();
    for &idx in &data.indices {
        if (idx as usize) >= vertex_count {
            return Err(ViewportError::InvalidVertexIndex {
                vertex_index: idx,
                vertex_count,
            });
        }
    }
    let index_count = data.indices.len() as u64;
    for range in &data.submeshes {
        let end = range.first_index as u64 + range.index_count as u64;
        if end > index_count {
            return Err(ViewportError::SubmeshRangeOutOfBounds {
                first_index: range.first_index,
                range_count: range.index_count,
                index_count: data.indices.len(),
            });
        }
    }
    Ok(())
}
