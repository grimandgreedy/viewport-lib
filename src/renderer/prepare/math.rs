//! Small numeric helpers used by the prepare passes.

/// Hash a byte slice for per-batch dirty detection.
///
/// Used by the partial-upload path to avoid reading back the cached instance
/// buffer: a hash mismatch means the batch changed; a match means it is clean.
pub(super) fn hash_instance_bytes(bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write(bytes);
    h.finish()
}

/// Jacobi eigenvalue decomposition for a 3x3 symmetric matrix.
/// Returns (eigenvectors as rows, eigenvalues).
pub(super) fn jacobi_eig_3x3(a: &[[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut m = *a;
    let mut v: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let mut max_off = 0.0f32;
        let (mut p, mut q) = (0usize, 1usize);
        for i in 0..3usize {
            for j in (i + 1)..3usize {
                let abs = m[i][j].abs();
                if abs > max_off {
                    max_off = abs;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-10 {
            break;
        }
        let tau = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let m_pp = m[p][p];
        let m_qq = m[q][q];
        let m_pq = m[p][q];
        m[p][p] = c * c * m_pp - 2.0 * s * c * m_pq + s * s * m_qq;
        m[q][q] = s * s * m_pp + 2.0 * s * c * m_pq + c * c * m_qq;
        m[p][q] = 0.0;
        m[q][p] = 0.0;
        for r in 0..3usize {
            if r != p && r != q {
                let m_pr = m[p][r];
                let m_qr = m[q][r];
                m[p][r] = c * m_pr - s * m_qr;
                m[r][p] = m[p][r];
                m[q][r] = s * m_pr + c * m_qr;
                m[r][q] = m[q][r];
            }
        }
        for r in 0..3usize {
            let v_rp = v[r][p];
            let v_rq = v[r][q];
            v[r][p] = c * v_rp - s * v_rq;
            v[r][q] = s * v_rp + c * v_rq;
        }
    }
    // Return eigenvectors as rows (transposed from columns of v).
    (
        [
            [v[0][0], v[1][0], v[2][0]],
            [v[0][1], v[1][1], v[2][1]],
            [v[0][2], v[1][2], v[2][2]],
        ],
        [m[0][0], m[1][1], m[2][2]],
    )
}
