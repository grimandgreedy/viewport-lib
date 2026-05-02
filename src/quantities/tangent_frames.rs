//! Tangent frame computation helpers.
//!
//! Provides per-vertex and per-face tangent frames used internally by the
//! intrinsic-vector and one-form conversion functions.

/// Compute a smooth per-vertex tangent frame using Gram-Schmidt orthogonalisation
/// against an arbitrary reference vector.
///
/// When explicit UV tangents are not available, this produces an arbitrary but
/// consistent (tangent, bitangent) pair for each vertex that spans the tangent
/// plane defined by `normal`.
///
/// Returns a `Vec` of `(tangent, bitangent)` pairs in the same order as `normals`.
pub fn compute_vertex_tangent_frames(normals: &[[f32; 3]]) -> Vec<([f32; 3], [f32; 3])> {
    normals.iter().map(|&n| gram_schmidt_tangent(n)).collect()
}

/// Reconstruct `(tangent, bitangent)` pairs from explicit tangent data.
///
/// `tangents[i]` is `[tx, ty, tz, w]` where `w = ±1` is the bitangent handedness.
/// Bitangent is reconstructed as `cross(normal, tangent.xyz) * w`.
///
/// `normals` and `tangents` must have the same length.
pub fn tangents_from_explicit(
    normals: &[[f32; 3]],
    tangents: &[[f32; 4]],
) -> Vec<([f32; 3], [f32; 3])> {
    normals
        .iter()
        .zip(tangents.iter())
        .map(|(&n, &t)| {
            let normal = glam::Vec3::from(n);
            let tangent = glam::Vec3::new(t[0], t[1], t[2]);
            let handedness = t[3];
            let bitangent = normal.cross(tangent) * handedness;
            (tangent.to_array(), bitangent.to_array())
        })
        .collect()
}

/// Compute per-face tangent frames for a triangle mesh.
///
/// The tangent is aligned with the first edge of each triangle (v0->v1), projected
/// onto the face plane. The bitangent completes the right-handed frame with the
/// face normal.
///
/// Returns one `(tangent, bitangent)` pair per triangle (`indices.len() / 3`).
///
/// Degenerate triangles (zero-area or collapsed edges) produce an arbitrary frame.
pub fn compute_face_tangent_frames(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Vec<([f32; 3], [f32; 3])> {
    let num_tris = indices.len() / 3;
    let mut frames = Vec::with_capacity(num_tris);

    for tri in 0..num_tris {
        let i0 = indices[3 * tri] as usize;
        let i1 = indices[3 * tri + 1] as usize;
        let i2 = indices[3 * tri + 2] as usize;

        let p0 = glam::Vec3::from(positions[i0]);
        let p1 = glam::Vec3::from(positions[i1]);
        let p2 = glam::Vec3::from(positions[i2]);

        let e0 = p1 - p0;
        let e1 = p2 - p0;
        let face_normal_raw = e0.cross(e1);

        let normal = if face_normal_raw.length_squared() > 1e-12 {
            face_normal_raw.normalize()
        } else {
            glam::Vec3::Z
        };

        let (tangent, bitangent) = if e0.length_squared() > 1e-12 {
            // Project e0 onto the face plane to get the tangent.
            let t = (e0 - e0.dot(normal) * normal).normalize_or_zero();
            if t.length_squared() > 0.5 {
                let b = normal.cross(t);
                (t, b)
            } else {
                gram_schmidt_tangent_vec(normal)
            }
        } else {
            gram_schmidt_tangent_vec(normal)
        };

        frames.push((tangent.to_array(), bitangent.to_array()));
    }

    frames
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a tangent frame for `normal` using Gram-Schmidt against a reference.
pub fn gram_schmidt_tangent(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let (t, b) = gram_schmidt_tangent_vec(glam::Vec3::from(normal));
    (t.to_array(), b.to_array())
}

fn gram_schmidt_tangent_vec(n: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    // Choose a reference not too close to the normal.
    let reference = if n.x.abs() < 0.9 {
        glam::Vec3::X
    } else {
        glam::Vec3::Y
    };
    let t = (reference - reference.dot(n) * n).normalize_or_zero();
    // If degenerate (rare), try Z.
    let t = if t.length_squared() < 0.5 {
        let fallback = glam::Vec3::Z;
        (fallback - fallback.dot(n) * n).normalize_or_zero()
    } else {
        t
    };
    let b = n.cross(t);
    (t, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_orthonormal(name: &str, normal: [f32; 3], tangent: [f32; 3], bitangent: [f32; 3]) {
        let n = glam::Vec3::from(normal);
        let t = glam::Vec3::from(tangent);
        let b = glam::Vec3::from(bitangent);
        assert!(
            (t.length() - 1.0).abs() < 1e-4,
            "{name}: tangent not unit length: {}",
            t.length()
        );
        assert!(
            (b.length() - 1.0).abs() < 1e-4,
            "{name}: bitangent not unit length: {}",
            b.length()
        );
        assert!(
            n.dot(t).abs() < 1e-4,
            "{name}: tangent not perpendicular to normal: {}",
            n.dot(t)
        );
        assert!(
            n.dot(b).abs() < 1e-4,
            "{name}: bitangent not perpendicular to normal: {}",
            n.dot(b)
        );
        assert!(
            t.dot(b).abs() < 1e-4,
            "{name}: tangent not perpendicular to bitangent: {}",
            t.dot(b)
        );
    }

    #[test]
    fn gram_schmidt_axis_aligned_normals() {
        let normals = [
            ([1.0, 0.0, 0.0], "+X"),
            ([-1.0, 0.0, 0.0], "-X"),
            ([0.0, 1.0, 0.0], "+Y"),
            ([0.0, -1.0, 0.0], "-Y"),
            ([0.0, 0.0, 1.0], "+Z"),
            ([0.0, 0.0, -1.0], "-Z"),
        ];
        for (n, name) in &normals {
            let (t, b) = gram_schmidt_tangent(*n);
            assert_orthonormal(name, *n, t, b);
        }
    }

    #[test]
    fn gram_schmidt_diagonal_normal() {
        let s = 1.0 / 3.0f32.sqrt();
        let n = [s, s, s];
        let (t, b) = gram_schmidt_tangent(n);
        assert_orthonormal("diagonal", n, t, b);
    }

    #[test]
    fn compute_vertex_tangent_frames_length_matches() {
        let normals = vec![[0.0, 1.0, 0.0]; 10];
        let frames = compute_vertex_tangent_frames(&normals);
        assert_eq!(frames.len(), normals.len());
    }

    #[test]
    fn compute_vertex_tangent_frames_all_orthonormal() {
        let normals = vec![
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let frames = compute_vertex_tangent_frames(&normals);
        for (i, (t, b)) in frames.iter().enumerate() {
            assert_orthonormal(&format!("vtx_{i}"), normals[i], *t, *b);
        }
    }

    #[test]
    fn compute_face_tangent_frames_right_triangle() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0u32, 1, 2];
        let frames = compute_face_tangent_frames(&positions, &indices);
        assert_eq!(frames.len(), 1);
        let (t, b) = frames[0];
        // Tangent should be aligned with edge v0->v1 = +X
        assert!((t[0] - 1.0).abs() < 1e-4, "tangent X should be ~1.0");
        assert!(t[1].abs() < 1e-4);
        assert!(t[2].abs() < 1e-4);
        // Bitangent should be +Y (cross of +Z normal and +X tangent)
        assert!(b[0].abs() < 1e-4);
        assert!((b[1] - 1.0).abs() < 1e-4, "bitangent Y should be ~1.0");
    }

    #[test]
    fn compute_face_tangent_frames_degenerate_no_panic() {
        let positions = vec![[0.0; 3]; 3]; // degenerate
        let indices = vec![0u32, 1, 2];
        let frames = compute_face_tangent_frames(&positions, &indices);
        assert_eq!(frames.len(), 1); // produces some frame, does not panic
    }

    #[test]
    fn tangents_from_explicit_positive_handedness() {
        let normals = vec![[0.0, 0.0, 1.0]]; // +Z
        let tangents = vec![[1.0, 0.0, 0.0, 1.0]]; // +X, w=+1
        let frames = tangents_from_explicit(&normals, &tangents);
        let (t, b) = frames[0];
        assert!((t[0] - 1.0).abs() < 1e-5);
        // bitangent = cross(+Z, +X) * 1 = +Y
        assert!((b[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn tangents_from_explicit_negative_handedness() {
        let normals = vec![[0.0, 0.0, 1.0]]; // +Z
        let tangents = vec![[1.0, 0.0, 0.0, -1.0]]; // +X, w=-1
        let frames = tangents_from_explicit(&normals, &tangents);
        let (_, b) = frames[0];
        // bitangent = cross(+Z, +X) * -1 = -Y
        assert!((b[1] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn tangents_from_explicit_length_matches() {
        let normals = vec![[0.0, 1.0, 0.0]; 5];
        let tangents = vec![[1.0, 0.0, 0.0, 1.0]; 5];
        let frames = tangents_from_explicit(&normals, &tangents);
        assert_eq!(frames.len(), 5);
    }
}
