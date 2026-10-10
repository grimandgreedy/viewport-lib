//! Attribute expansion: face, edge and cell values spread onto vertices.

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
