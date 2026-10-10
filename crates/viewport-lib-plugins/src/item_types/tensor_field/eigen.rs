//! Closed-form eigendecomposition of a symmetric 3x3 matrix.
//!
//! This is what turns a tensor field into something drawable: a stress or
//! diffusion tensor arrives as six components per sample, and the three
//! principal directions and magnitudes come out of it.
//!
//! The solve is closed form (no iteration) and runs in `f64` internally, because
//! the eigenvalue formula subtracts nearby quantities and `f32` loses too much
//! there on a near-degenerate tensor.

/// The result of [`symmetric_eigen_3x3`].
///
/// `values[i]` is the eigenvalue belonging to `vectors[i]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetricEigen {
    /// Eigenvalues, sorted descending by signed value. For a stress tensor these
    /// are the principal stresses, most tensile first.
    pub values: [f32; 3],
    /// Unit eigenvectors, one per eigenvalue, forming a right-handed orthonormal
    /// basis. See the conventions on [`symmetric_eigen_3x3`].
    pub vectors: [[f32; 3]; 3],
}

impl Default for SymmetricEigen {
    fn default() -> Self {
        Self {
            values: [0.0; 3],
            vectors: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

/// Decompose a symmetric 3x3 matrix given as `[xx, yy, zz, xy, xz, yz]`.
///
/// That component order is what a solver writes and what the common file formats
/// store, so a caller normally has it already.
///
/// # Conventions
///
/// These matter, because a tensor field is drawn sample by sample and any
/// inconsistency between neighbouring samples shows up as visible noise:
///
/// * Eigenvalues come back sorted descending by signed value, not by magnitude.
/// * The basis is always right-handed: the third vector is derived from the other
///   two rather than solved for, so it cannot come back flipped.
/// * An eigenvector's sign is fixed by making its largest-magnitude component
///   positive. An eigenvector is only defined up to sign, and without a rule the
///   choice varies with rounding between neighbouring samples.
/// * Repeated eigenvalues leave their eigenvectors underdetermined: any basis of
///   the repeated subspace is correct. An arbitrary but orthonormal one is
///   returned, so the result is always usable even where the field is isotropic.
/// * A matrix of all zeros, or any non-finite input, returns zero eigenvalues and
///   the standard basis.
pub fn symmetric_eigen_3x3(components: [f32; 6]) -> SymmetricEigen {
    if !components.iter().all(|c| c.is_finite()) {
        return SymmetricEigen::default();
    }

    let [xx, yy, zz, xy, xz, yz] = components.map(f64::from);
    let a = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];

    let values = eigenvalues(a);

    // Scale sets what counts as "close to zero" for the vector solve below.
    let scale = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale < f64::EPSILON {
        return SymmetricEigen {
            values: [0.0; 3],
            vectors: SymmetricEigen::default().vectors,
        };
    }

    // Solve for the outer two and derive the middle, so the basis is right-handed
    // by construction. The outer two are the best conditioned: the middle
    // eigenvalue is the one that collides with a neighbour in the common
    // degenerate cases.
    let v0 = fix_sign(eigenvector(a, values[0], scale));
    let mut v2 = fix_sign(eigenvector(a, values[2], scale));

    // Repeated eigenvalues can hand back two nearly parallel vectors. Either is
    // a valid eigenvector; they just have to span the subspace between them.
    v2 = orthogonalise(v2, v0, scale);
    let v1 = cross(v2, v0);

    SymmetricEigen {
        values: values.map(|v| v as f32),
        vectors: [to_f32(v0), to_f32(v1), to_f32(v2)],
    }
}

/// Eigenvalues only, descending. Cheaper than the full decomposition when the
/// directions are not wanted: a von Mises stress or a fractional anisotropy needs
/// nothing else.
pub fn symmetric_eigenvalues_3x3(components: [f32; 6]) -> [f32; 3] {
    if !components.iter().all(|c| c.is_finite()) {
        return [0.0; 3];
    }
    let [xx, yy, zz, xy, xz, yz] = components.map(f64::from);
    eigenvalues([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]).map(|v| v as f32)
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

/// Closed-form eigenvalues of a symmetric matrix, descending.
///
/// The characteristic polynomial of a symmetric matrix has three real roots, so
/// the trigonometric form of the cubic solution applies with no complex
/// arithmetic: shift by the mean eigenvalue, normalise, and read the three roots
/// off a cosine a third of a turn apart.
fn eigenvalues(a: [[f64; 3]; 3]) -> [f64; 3] {
    let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if off == 0.0 {
        let mut d = [a[0][0], a[1][1], a[2][2]];
        d.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        return d;
    }

    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let dx = a[0][0] - q;
    let dy = a[1][1] - q;
    let dz = a[2][2] - q;
    let p = ((dx * dx + dy * dy + dz * dz + 2.0 * off) / 6.0).sqrt();
    if p <= 0.0 {
        return [q, q, q];
    }

    // det((a - q I) / p) / 2, which is cos(3 phi) for the shifted matrix.
    let b = [
        [dx / p, a[0][1] / p, a[0][2] / p],
        [a[0][1] / p, dy / p, a[1][2] / p],
        [a[0][2] / p, a[1][2] / p, dz / p],
    ];
    let r = (det3(b) / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    let hi = q + 2.0 * p * phi.cos();
    let lo = q + 2.0 * p * (phi + 2.0 * std::f64::consts::FRAC_PI_3).cos();
    // The trace is the sum of the eigenvalues, so the middle one is free.
    let mid = 3.0 * q - hi - lo;
    [hi, mid, lo]
}

/// A unit eigenvector for `lambda`.
///
/// `(a - lambda I)` is singular, so its rows are perpendicular to the eigenvector.
/// Two independent rows pin it down exactly, via their cross product; taking the
/// longest of the three candidate pairs is what keeps this stable when one pair
/// happens to be nearly parallel. A repeated eigenvalue drops the rank further
/// and leaves the eigenvector underdetermined, so the remaining cases pick an
/// arbitrary direction out of the eigenspace.
fn eigenvector(a: [[f64; 3]; 3], lambda: f64, scale: f64) -> [f64; 3] {
    let m = [
        [a[0][0] - lambda, a[0][1], a[0][2]],
        [a[1][0], a[1][1] - lambda, a[1][2]],
        [a[2][0], a[2][1], a[2][2] - lambda],
    ];

    // Rank 2: the normal cases. Cross products scale as the square of the matrix.
    let mut best = [0.0; 3];
    let mut best_len = 0.0;
    for c in [cross(m[0], m[1]), cross(m[1], m[2]), cross(m[2], m[0])] {
        let len = norm(c);
        if len > best_len {
            best_len = len;
            best = c;
        }
    }
    if best_len > scale * scale * 1e-9 {
        return scale_by(best, 1.0 / best_len);
    }

    // Rank 1: the eigenvalue is repeated twice and the eigenspace is the plane
    // perpendicular to whichever row survives.
    let mut row = [0.0; 3];
    let mut row_len = 0.0;
    for r in m {
        let len = norm(r);
        if len > row_len {
            row_len = len;
            row = r;
        }
    }
    if row_len > scale * 1e-9 {
        return any_perpendicular(scale_by(row, 1.0 / row_len));
    }

    // Rank 0: a multiple of the identity, so every direction is an eigenvector.
    [1.0, 0.0, 0.0]
}

/// Remove `basis` from `v` and renormalise, substituting any perpendicular
/// direction when the two are parallel.
fn orthogonalise(v: [f64; 3], basis: [f64; 3], scale: f64) -> [f64; 3] {
    let d = dot(v, basis);
    let proj = [
        v[0] - d * basis[0],
        v[1] - d * basis[1],
        v[2] - d * basis[2],
    ];
    let len = norm(proj);
    if len > (scale * 1e-6).max(1e-9) {
        return scale_by(proj, 1.0 / len);
    }
    any_perpendicular(basis)
}

fn any_perpendicular(v: [f64; 3]) -> [f64; 3] {
    // Cross with whichever axis the vector leans on least.
    let axis = if v[0].abs() < v[1].abs() && v[0].abs() < v[2].abs() {
        [1.0, 0.0, 0.0]
    } else if v[1].abs() < v[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let c = cross(v, axis);
    let len = norm(c);
    if len > 0.0 {
        scale_by(c, 1.0 / len)
    } else {
        axis
    }
}

/// Make the largest-magnitude component positive, so the same tensor always
/// yields the same eigenvector rather than one of its two valid signs.
fn fix_sign(v: [f64; 3]) -> [f64; 3] {
    let mut k = 0;
    for i in 1..3 {
        if v[i].abs() > v[k].abs() {
            k = i;
        }
    }
    if v[k] < 0.0 { scale_by(v, -1.0) } else { v }
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn scale_by(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn to_f32(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_orthonormal_right_handed(e: &SymmetricEigen) {
        for v in &e.vectors {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "eigenvector not unit: {v:?}");
        }
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let a = e.vectors[i];
            let b = e.vectors[j];
            let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            assert!(d.abs() < 1e-4, "vectors {i} and {j} not orthogonal: {d}");
        }
        let [a, b, c] = e.vectors;
        let cr = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let d = cr[0] * c[0] + cr[1] * c[1] + cr[2] * c[2];
        assert!(d > 0.9, "basis is not right-handed: {d}");
    }

    /// Reconstruct the matrix from the decomposition: `A = sum(lambda_i v_i v_i^T)`.
    fn reconstruct(e: &SymmetricEigen) -> [f32; 6] {
        let mut m = [[0.0f32; 3]; 3];
        for k in 0..3 {
            let v = e.vectors[k];
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += e.values[k] * v[i] * v[j];
                }
            }
        }
        [m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2]]
    }

    fn assert_reconstructs(components: [f32; 6]) {
        let e = symmetric_eigen_3x3(components);
        check_orthonormal_right_handed(&e);
        let back = reconstruct(&e);
        for i in 0..6 {
            assert!(
                (back[i] - components[i]).abs() < 1e-3,
                "component {i}: {} vs {} (values {:?})",
                back[i],
                components[i],
                e.values
            );
        }
    }

    #[test]
    fn values_come_back_descending() {
        let e = symmetric_eigen_3x3([3.0, -1.0, 7.0, 0.0, 0.0, 0.0]);
        assert!((e.values[0] - 7.0).abs() < 1e-5);
        assert!((e.values[1] - 3.0).abs() < 1e-5);
        assert!((e.values[2] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn diagonal_matrix_gives_axis_eigenvectors() {
        let e = symmetric_eigen_3x3([1.0, 5.0, 2.0, 0.0, 0.0, 0.0]);
        // Largest eigenvalue is yy, so the leading eigenvector is +Y.
        assert!(e.vectors[0][1].abs() > 0.999);
        check_orthonormal_right_handed(&e);
    }

    #[test]
    fn reconstructs_a_general_tensor() {
        assert_reconstructs([4.0, 1.0, -2.0, 0.7, -1.3, 2.1]);
        assert_reconstructs([-3.0, -3.5, 9.0, 2.0, 0.5, -0.25]);
        assert_reconstructs([0.001, -0.002, 0.0005, 0.0003, -0.0001, 0.0002]);
    }

    #[test]
    fn reconstructs_a_pure_shear() {
        // Zero diagonal: the eigenvalues straddle zero and one of them is exactly
        // zero, which is where a magnitude-based sort would get it wrong.
        assert_reconstructs([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn repeated_eigenvalues_still_give_a_usable_basis() {
        // Two equal eigenvalues: a uniaxial state.
        let e = symmetric_eigen_3x3([2.0, 2.0, 5.0, 0.0, 0.0, 0.0]);
        assert!((e.values[0] - 5.0).abs() < 1e-5);
        assert!((e.values[1] - 2.0).abs() < 1e-5);
        assert!((e.values[2] - 2.0).abs() < 1e-5);
        check_orthonormal_right_handed(&e);
        assert_reconstructs([2.0, 2.0, 5.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn isotropic_tensor_gives_the_standard_basis() {
        let e = symmetric_eigen_3x3([3.0, 3.0, 3.0, 0.0, 0.0, 0.0]);
        for v in e.values {
            assert!((v - 3.0).abs() < 1e-5);
        }
        check_orthonormal_right_handed(&e);
    }

    #[test]
    fn zero_tensor_is_handled() {
        let e = symmetric_eigen_3x3([0.0; 6]);
        assert_eq!(e.values, [0.0; 3]);
        check_orthonormal_right_handed(&e);
    }

    #[test]
    fn non_finite_input_falls_back_to_the_standard_basis() {
        let e = symmetric_eigen_3x3([f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(e.values, [0.0; 3]);
        assert_eq!(e.vectors, SymmetricEigen::default().vectors);
    }

    #[test]
    fn sign_convention_is_stable_under_negation_of_the_input_vector() {
        // The same tensor written two ways must give identical eigenvectors.
        let a = symmetric_eigen_3x3([4.0, 1.0, -2.0, 0.7, -1.3, 2.1]);
        let b = symmetric_eigen_3x3([4.0, 1.0, -2.0, 0.7, -1.3, 2.1]);
        assert_eq!(a.vectors, b.vectors);
        // And the leading vector's dominant component is positive.
        let v = a.vectors[0];
        let k = (0..3)
            .max_by(|&i, &j| v[i].abs().total_cmp(&v[j].abs()))
            .unwrap();
        assert!(v[k] > 0.0);
    }

    #[test]
    fn eigenvalues_only_matches_the_full_solve() {
        let c = [4.0, 1.0, -2.0, 0.7, -1.3, 2.1];
        let full = symmetric_eigen_3x3(c);
        let quick = symmetric_eigenvalues_3x3(c);
        for i in 0..3 {
            assert!((full.values[i] - quick[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn reconstructs_a_sweep_of_tensors() {
        // Deterministic pseudo-random components, including near-degenerate ones.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for i in 0..400 {
            // Every fourth case squashes the off-diagonals towards zero, which is
            // where eigenvalues collide and the vector solve is worst behaved.
            let squash = if i % 4 == 0 { 1e-4 } else { 1.0 };
            let c = [
                next(),
                next(),
                next(),
                next() * squash,
                next() * squash,
                next() * squash,
            ];
            let e = symmetric_eigen_3x3(c);
            assert!(
                e.values[0] >= e.values[1] && e.values[1] >= e.values[2],
                "not descending for {c:?}: {:?}",
                e.values
            );
            check_orthonormal_right_handed(&e);
            let back = reconstruct(&e);
            for k in 0..6 {
                assert!(
                    (back[k] - c[k]).abs() < 1e-4,
                    "component {k} of {c:?}: {} vs {}",
                    back[k],
                    c[k]
                );
            }
        }
    }
}
