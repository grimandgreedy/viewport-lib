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

use crate::resources::volume_mesh::{TET_SENTINEL, VolumeMeshData};
use crate::GlyphItem;

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
        // Collect valid vertex indices (filter out TET_SENTINEL).
        let valid: Vec<usize> = cell
            .iter()
            .filter(|&&idx| idx != TET_SENTINEL)
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
