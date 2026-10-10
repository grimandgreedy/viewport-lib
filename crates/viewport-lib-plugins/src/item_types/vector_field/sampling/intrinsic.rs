//! Convert tangent-plane (intrinsic) vector fields to world-space vectors.
//!
//! An *intrinsic* vector at a vertex or face is expressed as `(u, v)` coefficients
//! in a local tangent frame:
//!
//! ```text
//! world_vector = u * tangent + v * bitangent
//! ```
//!
//! The tangent frame is derived from the mesh normals (and optional explicit
//! tangents) so that the coefficients are meaningful in surface-local coordinates.

use super::VectorSamples;
use super::tangent_frames;

/// Convert vertex-indexed 2D intrinsic vectors to world-space samples.
///
/// Each entry in `vectors` is `[u, v]` : the components of the surface vector at
/// the corresponding vertex expressed in the vertex tangent frame.
///
/// Sample positions are the vertex positions; the vectors are converted to world
/// space via the tangent frame derived from `normals` (and `tangents` if given).
///
/// # Arguments
///
/// * `positions` : vertex positions in world/local space
/// * `normals`   : per-vertex normals (same length as `positions`)
/// * `tangents`  : optional explicit tangents `[tx, ty, tz, w]`; when `None`,
///                 a smooth frame is computed from the normals via Gram-Schmidt
/// * `vectors`   : per-vertex intrinsic 2D vectors (same length as `positions`)
///
/// # Panics
///
/// Does not panic; mismatched slice lengths are handled by iterating to the
/// shortest common length.
pub fn vertex_intrinsic_vectors(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    tangents: Option<&[[f32; 4]]>,
    vectors: &[[f32; 2]],
) -> VectorSamples {
    let frames: Vec<([f32; 3], [f32; 3])> = match tangents {
        Some(t) => tangent_frames::tangents_from_explicit(normals, t),
        None => tangent_frames::compute_vertex_tangent_frames(normals),
    };

    let n = positions
        .len()
        .min(normals.len())
        .min(frames.len())
        .min(vectors.len());

    let mut out_positions = Vec::with_capacity(n);
    let mut out_vectors = Vec::with_capacity(n);

    for i in 0..n {
        let uv = vectors[i];
        let (tangent, bitangent) = frames[i];
        let t = glam::Vec3::from(tangent);
        let b = glam::Vec3::from(bitangent);

        out_positions.push(positions[i]);
        out_vectors.push((t * uv[0] + b * uv[1]).to_array());
    }

    VectorSamples {
        positions: out_positions,
        vectors: out_vectors,
    }
}

/// Convert face-indexed 2D intrinsic vectors to world-space samples.
///
/// Each entry in `vectors` is `[u, v]` : the components of the surface vector at
/// the corresponding triangle expressed in the face tangent frame.
///
/// Sample positions are the face centroids; the vectors are converted to world
/// space via the per-face tangent frame computed from `positions` and `indices`.
///
/// # Arguments
///
/// * `positions` : vertex positions in world/local space
/// * `normals`   : per-vertex normals (used to orient faces consistently)
/// * `indices`   : triangle index list (every 3 indices form one triangle)
/// * `vectors`   : per-face intrinsic 2D vectors (one per triangle)
pub fn face_intrinsic_vectors(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    indices: &[u32],
    vectors: &[[f32; 2]],
) -> VectorSamples {
    let _ = normals; // reserved : may be used for consistent orientation later
    let num_tris = indices.len() / 3;
    let frames = tangent_frames::compute_face_tangent_frames(positions, indices);

    let n = num_tris.min(frames.len()).min(vectors.len());

    let mut out_positions = Vec::with_capacity(n);
    let mut out_vectors = Vec::with_capacity(n);

    for tri in 0..n {
        let i0 = indices[3 * tri] as usize;
        let i1 = indices[3 * tri + 1] as usize;
        let i2 = indices[3 * tri + 2] as usize;

        let p0 = glam::Vec3::from(positions[i0]);
        let p1 = glam::Vec3::from(positions[i1]);
        let p2 = glam::Vec3::from(positions[i2]);
        let centroid = (p0 + p1 + p2) / 3.0;

        let uv = vectors[tri];
        let (tangent, bitangent) = frames[tri];
        let t = glam::Vec3::from(tangent);
        let b = glam::Vec3::from(bitangent);

        out_positions.push(centroid.to_array());
        out_vectors.push((t * uv[0] + b * uv[1]).to_array());
    }

    VectorSamples {
        positions: out_positions,
        vectors: out_vectors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_intrinsic_unit_u_along_tangent() {
        // Normal = +Y, so tangent frame spans XZ plane
        let positions = vec![[0.0, 0.0, 0.0]];
        let normals = vec![[0.0, 1.0, 0.0]];
        let vectors = vec![[1.0, 0.0]]; // pure u component
        let samples = vertex_intrinsic_vectors(&positions, &normals, None, &vectors);
        assert_eq!(samples.len(), 1);
        let v = glam::Vec3::from(samples.vectors[0]);
        // Should be in the XZ plane (Y ~ 0) and unit length
        assert!(v.y.abs() < 1e-4, "should be in tangent plane");
        assert!((v.length() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn vertex_intrinsic_mismatched_lengths_truncates() {
        let positions = vec![[0.0; 3]; 5];
        let normals = vec![[0.0, 1.0, 0.0]; 3]; // shorter
        let vectors = vec![[1.0, 0.0]; 5];
        let samples = vertex_intrinsic_vectors(&positions, &normals, None, &vectors);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn face_intrinsic_centroid_position() {
        let positions = vec![[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let indices = vec![0u32, 1, 2];
        let vectors = vec![[1.0, 0.0]];
        let samples = face_intrinsic_vectors(&positions, &normals, &indices, &vectors);
        assert_eq!(samples.len(), 1);
        let c = samples.positions[0];
        assert!((c[0] - 1.0).abs() < 1e-4);
        assert!((c[1] - 1.0).abs() < 1e-4);
    }
}
