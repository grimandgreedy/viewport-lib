//! Helper functions to convert volume mesh vector quantities into [`GlyphItem`]s.
//!
//! Volume mesh vectors are not rendered directly — this module converts them to
//! arrow glyphs that can be submitted to [`SceneFrame::glyphs`] each frame.
//!
//! # Vertex vectors
//!
//! One world-space vector per mesh vertex. Glyph positions match vertex positions.
//! Use [`volume_mesh_vertex_vectors_to_glyphs`].
//!
//! # Cell vectors
//!
//! One world-space vector per cell (tet or hex). Glyphs are placed at cell centroids.
//! Use [`volume_mesh_cell_vectors_to_glyphs`].

use crate::GlyphItem;
use crate::resources::volume_mesh::{CELL_SENTINEL, VolumeMeshData};

/// Convert per-vertex world-space vectors on a volume mesh to a [`GlyphItem`].
///
/// Returns one arrow per vertex, positioned at the vertex and pointing in the given
/// world-space direction. Length (and thus visual scale) is determined by `scale` and
/// [`GlyphItem::scale_by_magnitude`] (enabled by default).
///
/// # Arguments
///
/// * `positions`  : vertex positions in local/world space (same as [`VolumeMeshData::positions`])
/// * `vectors`    : one world-space `[f32; 3]` per vertex; must have the same length as `positions`
/// * `scale`      : global arrow scale (see [`GlyphItem::scale`])
pub fn volume_mesh_vertex_vectors_to_glyphs(
    positions: &[[f32; 3]],
    vectors: &[[f32; 3]],
    scale: f32,
) -> GlyphItem {
    let n = positions.len().min(vectors.len());
    let mut glyph_positions = Vec::with_capacity(n);
    let mut glyph_vectors = Vec::with_capacity(n);

    for i in 0..n {
        glyph_positions.push(positions[i]);
        glyph_vectors.push(vectors[i]);
    }

    GlyphItem {
        positions: glyph_positions,
        vectors: glyph_vectors,
        scale,
        ..Default::default()
    }
}

/// Convert per-cell world-space vectors on a volume mesh to a [`GlyphItem`].
///
/// Returns one arrow per cell, positioned at the cell centroid and pointing in the
/// given world-space direction. Tets use `cells[c][0..4]`; hexes use all 8 indices.
///
/// # Arguments
///
/// * `data`         : volume mesh geometry (positions + cell connectivity)
/// * `cell_vectors` : one world-space `[f32; 3]` per cell; must have the same length as
///                    [`VolumeMeshData::cells`]
/// * `scale`        : global arrow scale (see [`GlyphItem::scale`])
pub fn volume_mesh_cell_vectors_to_glyphs(
    data: &VolumeMeshData,
    cell_vectors: &[[f32; 3]],
    scale: f32,
) -> GlyphItem {
    let n = data.cells.len().min(cell_vectors.len());
    let mut glyph_positions = Vec::with_capacity(n);
    let mut glyph_vectors = Vec::with_capacity(n);

    for c in 0..n {
        let cell = &data.cells[c];
        // Collect valid vertex indices (filter out CELL_SENTINEL).
        let valid: Vec<usize> = cell
            .iter()
            .filter(|&&idx| idx != CELL_SENTINEL)
            .map(|&idx| idx as usize)
            .filter(|&idx| idx < data.positions.len())
            .collect();

        if valid.is_empty() {
            continue;
        }

        // Centroid = average of valid vertex positions.
        let mut centroid = [0.0f32; 3];
        for &vi in &valid {
            let p = data.positions[vi];
            centroid[0] += p[0];
            centroid[1] += p[1];
            centroid[2] += p[2];
        }
        let inv = 1.0 / valid.len() as f32;
        centroid[0] *= inv;
        centroid[1] *= inv;
        centroid[2] *= inv;

        glyph_positions.push(centroid);
        glyph_vectors.push(cell_vectors[c]);
    }

    GlyphItem {
        positions: glyph_positions,
        vectors: glyph_vectors,
        scale,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn vertex_vectors_length_matches() {
        let positions = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let vectors = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let item = volume_mesh_vertex_vectors_to_glyphs(&positions, &vectors, 1.0);
        assert_eq!(item.positions.len(), 2);
        assert_eq!(item.vectors.len(), 2);
    }

    #[test]
    fn vertex_vectors_mismatched_truncates() {
        let positions = vec![[0.0; 3]; 5];
        let vectors = vec![[1.0, 0.0, 0.0]; 3];
        let item = volume_mesh_vertex_vectors_to_glyphs(&positions, &vectors, 1.0);
        assert_eq!(item.positions.len(), 3);
    }

    #[test]
    fn vertex_vectors_scale_forwarded() {
        let item = volume_mesh_vertex_vectors_to_glyphs(&[[0.0; 3]], &[[1.0, 0.0, 0.0]], 3.5);
        assert!((item.scale - 3.5).abs() < 1e-6);
    }

    #[test]
    fn cell_vectors_tet_centroid() {
        let data = VolumeMeshData {
            positions: vec![
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ],
            cells: vec![[
                0,
                1,
                2,
                3,
                CELL_SENTINEL,
                CELL_SENTINEL,
                CELL_SENTINEL,
                CELL_SENTINEL,
            ]],
            cell_scalars: HashMap::new(),
            cell_colors: HashMap::new(),
        };
        let cell_vectors = vec![[1.0, 0.0, 0.0]];
        let item = volume_mesh_cell_vectors_to_glyphs(&data, &cell_vectors, 1.0);
        assert_eq!(item.positions.len(), 1);
        let c = item.positions[0];
        assert!((c[0] - 1.0).abs() < 1e-4);
        assert!((c[1] - 1.0).abs() < 1e-4);
        assert!((c[2] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn cell_vectors_empty_data() {
        let data = VolumeMeshData {
            positions: vec![],
            cells: vec![],
            cell_scalars: HashMap::new(),
            cell_colors: HashMap::new(),
        };
        let item = volume_mesh_cell_vectors_to_glyphs(&data, &[], 1.0);
        assert!(item.positions.is_empty());
    }
}
