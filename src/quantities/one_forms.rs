//! Whitney one-form reconstruction and conversion to [`GlyphItem`]s.
//!
//! A *one-form* assigns a scalar value to each directed edge of a triangle mesh.
//! The value represents the integral of a covector field along that edge.
//!
//! # Edge ordering convention
//!
//! For triangle `t` with vertex indices `(v0, v1, v2)` from the index buffer:
//!
//! - `edge_values[3 * t + 0]` : value on edge `v0 -> v1`
//! - `edge_values[3 * t + 1]` : value on edge `v1 -> v2`
//! - `edge_values[3 * t + 2]` : value on edge `v2 -> v0`
//!
//! # Reconstruction formula
//!
//! The reconstructed vector field at face centroid `c` of triangle `(p0, p1, p2)`
//! is the Hodge dual of the discrete one-form (Whitney reconstruction):
//!
//! ```text
//! F = (w01 · R(e01) + w12 · R(e12) + w20 · R(e20)) / (2 · area)
//! ```
//!
//! where `eij = pj − pi`, `R(v) = n × v` (90° rotation in the face plane),
//! and `area` is the signed triangle area (`|n_raw| / 2`).

use crate::GlyphItem;

/// Convert a scalar-per-directed-edge one-form to a [`GlyphItem`] via Whitney
/// reconstruction.
///
/// Returns one arrow per triangle placed at the face centroid, pointing in the
/// direction of the reconstructed vector field.
///
/// # Arguments
///
/// * `positions`    : vertex positions in world/local space
/// * `indices`      : triangle index list (every 3 indices form one triangle)
/// * `edge_values`  : one scalar per directed edge, in triangle-local order
///                    (see [module-level docs](self) for the convention).
///                    Length must be `3 × num_triangles`.
/// * `scale`        : global arrow scale (see [`GlyphItem::scale`])
///
/// Triangles whose `edge_values` slice is shorter than expected are skipped.
pub fn edge_one_form_to_glyphs(
    positions: &[[f32; 3]],
    indices: &[u32],
    edge_values: &[f32],
    scale: f32,
) -> GlyphItem {
    let num_tris = indices.len() / 3;
    let n = num_tris.min(edge_values.len() / 3);

    let mut glyph_positions = Vec::with_capacity(n);
    let mut glyph_vectors = Vec::with_capacity(n);

    for tri in 0..n {
        let i0 = indices[3 * tri] as usize;
        let i1 = indices[3 * tri + 1] as usize;
        let i2 = indices[3 * tri + 2] as usize;

        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }

        let p0 = glam::Vec3::from(positions[i0]);
        let p1 = glam::Vec3::from(positions[i1]);
        let p2 = glam::Vec3::from(positions[i2]);

        let e01 = p1 - p0;
        let e12 = p2 - p1;
        let e20 = p0 - p2;

        // Face normal (unnormalised; length = 2 * area).
        let n_raw = e01.cross(-e20); // (p1-p0) × (p2-p0)
        let area2 = n_raw.length();

        if area2 < 1e-12 {
            continue; // degenerate triangle
        }

        let face_normal = n_raw / area2; // normalised

        let w01 = edge_values[3 * tri];
        let w12 = edge_values[3 * tri + 1];
        let w20 = edge_values[3 * tri + 2];

        // R(v) = face_normal × v  (rotates v by 90° within the face plane)
        let f = (w01 * face_normal.cross(e01)
            + w12 * face_normal.cross(e12)
            + w20 * face_normal.cross(e20))
            / area2; // divide by 2*area, but area2 = 2*area

        let centroid = (p0 + p1 + p2) / 3.0;

        glyph_positions.push(centroid.to_array());
        glyph_vectors.push(f.to_array());
    }

    let mut item = GlyphItem::default();
    item.positions = glyph_positions;
    item.vectors = glyph_vectors;
    item.scale = scale;
    item
}

#[cfg(test)]
mod tests {
    use super::*;

    // Right triangle in XY plane: (0,0,0), (1,0,0), (0,1,0)
    fn xy_triangle() -> (Vec<[f32; 3]>, Vec<u32>) {
        (
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn zero_edge_values_produce_zero_vector() {
        let (pos, idx) = xy_triangle();
        let edge_values = vec![0.0, 0.0, 0.0];
        let item = edge_one_form_to_glyphs(&pos, &idx, &edge_values, 1.0);
        assert_eq!(item.vectors.len(), 1);
        for c in &item.vectors[0] {
            assert!(c.abs() < 1e-6);
        }
    }

    #[test]
    fn linearity_scaling_edge_values_scales_output() {
        let (pos, idx) = xy_triangle();
        let ev1 = vec![1.0, 0.5, -0.5];
        let ev2: Vec<f32> = ev1.iter().map(|v| v * 3.0).collect();
        let item1 = edge_one_form_to_glyphs(&pos, &idx, &ev1, 1.0);
        let item2 = edge_one_form_to_glyphs(&pos, &idx, &ev2, 1.0);
        for i in 0..3 {
            assert!(
                (item2.vectors[0][i] - item1.vectors[0][i] * 3.0).abs() < 1e-4,
                "linearity failed on component {i}"
            );
        }
    }

    #[test]
    fn output_position_is_centroid() {
        let (pos, idx) = xy_triangle();
        let edge_values = vec![1.0, 0.0, 0.0];
        let item = edge_one_form_to_glyphs(&pos, &idx, &edge_values, 1.0);
        let c = item.positions[0];
        let expected = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        for i in 0..3 {
            assert!((c[i] - expected[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn degenerate_triangle_skipped() {
        let pos = vec![[0.0; 3]; 3];
        let idx = vec![0u32, 1, 2];
        let edge_values = vec![1.0, 1.0, 1.0];
        let item = edge_one_form_to_glyphs(&pos, &idx, &edge_values, 1.0);
        assert!(item.positions.is_empty());
    }

    #[test]
    fn out_of_bounds_indices_skipped() {
        let pos = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let idx = vec![0u32, 1, 99];
        let edge_values = vec![1.0, 1.0, 1.0];
        let item = edge_one_form_to_glyphs(&pos, &idx, &edge_values, 1.0);
        assert!(item.positions.is_empty());
    }

    #[test]
    fn scale_forwarded() {
        let (pos, idx) = xy_triangle();
        let item = edge_one_form_to_glyphs(&pos, &idx, &[1.0, 0.0, 0.0], 5.0);
        assert!((item.scale - 5.0).abs() < 1e-6);
    }

    #[test]
    fn short_edge_values_truncates() {
        let (pos, idx) = xy_triangle();
        // Only 2 edge values instead of 3 : 0 complete triangles
        let item = edge_one_form_to_glyphs(&pos, &idx, &[1.0, 0.0], 1.0);
        assert!(item.positions.is_empty());
    }
}
