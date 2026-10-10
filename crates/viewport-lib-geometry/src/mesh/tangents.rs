//! Per-vertex tangents for normal mapping.

/// Compute per-vertex tangents using Gram-Schmidt orthogonalisation with handedness.
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

    // Gram-Schmidt orthogonalisation per vertex. Sequential, for the
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
