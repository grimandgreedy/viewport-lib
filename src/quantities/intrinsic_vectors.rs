//! Convert tangent-plane (intrinsic) vector fields to world-space [`GlyphItem`]s.
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

use super::tangent_frames;
use crate::GlyphItem;

/// Convert vertex-indexed 2D intrinsic vectors to a [`GlyphItem`].
///
/// Each entry in `vectors` is `[u, v]` : the components of the surface vector at
/// the corresponding vertex expressed in the vertex tangent frame.
///
/// The glyph base positions are the vertex positions; vectors are converted to
/// world space via the tangent frame derived from `normals` (and `tangents` if
/// provided). The `scale` parameter sets [`GlyphItem::scale`].
///
/// # Arguments
///
/// * `positions` : vertex positions in world/local space
/// * `normals`   : per-vertex normals (same length as `positions`)
/// * `tangents`  : optional explicit tangents `[tx, ty, tz, w]`; when `None`,
///                 a smooth frame is computed from the normals via Gram-Schmidt
/// * `vectors`   : per-vertex intrinsic 2D vectors (same length as `positions`)
/// * `scale`     : global arrow scale (see [`GlyphItem::scale`])
///
/// # Panics
///
/// Does not panic; mismatched slice lengths are handled by iterating to the
/// shortest common length.
pub fn vertex_intrinsic_to_glyphs(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    tangents: Option<&[[f32; 4]]>,
    vectors: &[[f32; 2]],
    scale: f32,
) -> GlyphItem {
    let frames: Vec<([f32; 3], [f32; 3])> = match tangents {
        Some(t) => tangent_frames::tangents_from_explicit(normals, t),
        None => tangent_frames::compute_vertex_tangent_frames(normals),
    };

    let n = positions
        .len()
        .min(normals.len())
        .min(frames.len())
        .min(vectors.len());

    let mut glyph_positions = Vec::with_capacity(n);
    let mut glyph_vectors = Vec::with_capacity(n);

    for i in 0..n {
        let uv = vectors[i];
        let (tangent, bitangent) = frames[i];
        let t = glam::Vec3::from(tangent);
        let b = glam::Vec3::from(bitangent);
        let world_vec = t * uv[0] + b * uv[1];

        glyph_positions.push(positions[i]);
        glyph_vectors.push(world_vec.to_array());
    }

    let mut item = GlyphItem::default();
    item.positions = glyph_positions;
    item.vectors = glyph_vectors;
    item.scale = scale;
    item
}

/// Convert face-indexed 2D intrinsic vectors to a [`GlyphItem`].
///
/// Each entry in `vectors` is `[u, v]` : the components of the surface vector at
/// the corresponding triangle expressed in the face tangent frame.
///
/// The glyph base positions are the face centroids; vectors are converted to
/// world space via the per-face tangent frame computed from `positions` and
/// `indices`. The `scale` parameter sets [`GlyphItem::scale`].
///
/// # Arguments
///
/// * `positions` : vertex positions in world/local space
/// * `normals`   : per-vertex normals (used to orient faces consistently)
/// * `indices`   : triangle index list (every 3 indices form one triangle)
/// * `vectors`   : per-face intrinsic 2D vectors (one per triangle)
/// * `scale`     : global arrow scale
pub fn face_intrinsic_to_glyphs(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    indices: &[u32],
    vectors: &[[f32; 2]],
    scale: f32,
) -> GlyphItem {
    let _ = normals; // reserved : may be used for consistent orientation later
    let num_tris = indices.len() / 3;
    let frames = tangent_frames::compute_face_tangent_frames(positions, indices);

    let n = num_tris.min(frames.len()).min(vectors.len());

    let mut glyph_positions = Vec::with_capacity(n);
    let mut glyph_vectors = Vec::with_capacity(n);

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
        let world_vec = t * uv[0] + b * uv[1];

        glyph_positions.push(centroid.to_array());
        glyph_vectors.push(world_vec.to_array());
    }

    let mut item = GlyphItem::default();
    item.positions = glyph_positions;
    item.vectors = glyph_vectors;
    item.scale = scale;
    item
}
