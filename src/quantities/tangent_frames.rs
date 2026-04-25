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
/// The tangent is aligned with the first edge of each triangle (v0→v1), projected
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
