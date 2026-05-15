//! Curve-network vector quantity utilities.
//!
//! Converts per-node and per-edge vector fields on a [`PolylineItem`] into
//! world-space [`GlyphItem`](crate::GlyphItem)s ready for submission to
//! [`SceneFrame::glyphs`](crate::renderer::types::SceneFrame::glyphs).
//!
//! The renderer calls these automatically when `node_vectors` or `edge_vectors`
//! are non-empty on a [`PolylineItem`]; the helpers are also public so callers
//! can customise the resulting [`GlyphItem`] (e.g. change `glyph_type` or add
//! scalar coloring) before pushing it to the frame.
//!
//! # Example
//!
//! ```rust,ignore
//! use viewport_lib::quantities::{polyline_node_vectors_to_glyphs, polyline_edge_vectors_to_glyphs};
//!
//! // Attach tangent arrows at every node.
//! let node_glyph = polyline_node_vectors_to_glyphs(&polyline);
//! frame.scene.glyphs.push(node_glyph);
//!
//! // Attach normal arrows at every edge midpoint.
//! let edge_glyph = polyline_edge_vectors_to_glyphs(&polyline);
//! frame.scene.glyphs.push(edge_glyph);
//! ```

use crate::{GlyphItem, PolylineItem};

/// Convert the `node_vectors` field of a [`PolylineItem`] into a [`GlyphItem`].
///
/// Each arrow is placed at the corresponding node position and oriented along the
/// world-space vector stored in `item.node_vectors`. The global `item.vector_scale`
/// is forwarded to `GlyphItem::scale`. Returns an empty `GlyphItem` when
/// `node_vectors` is empty.
pub fn polyline_node_vectors_to_glyphs(item: &PolylineItem) -> GlyphItem {
    let n = item.positions.len().min(item.node_vectors.len());
    GlyphItem {
        positions: item.positions[..n].to_vec(),
        vectors: item.node_vectors[..n].to_vec(),
        scale: item.vector_scale,
        ..Default::default()
    }
}

/// Convert the `edge_vectors` field of a [`PolylineItem`] into a [`GlyphItem`].
///
/// Each arrow is placed at the midpoint of the corresponding segment and oriented
/// along the world-space vector stored in `item.edge_vectors`. Segment indices are
/// counted in strip order (the same order `edge_scalars`/`edge_colors` use).
/// The global `item.vector_scale` is forwarded to `GlyphItem::scale`. Returns an
/// empty `GlyphItem` when `edge_vectors` is empty.
pub fn polyline_edge_vectors_to_glyphs(item: &PolylineItem) -> GlyphItem {
    let positions = &item.positions;
    let npos = positions.len();

    // Collect strip ranges: (start_idx, end_idx) into `positions`.
    let strip_ranges: Vec<(usize, usize)> = if item.strip_lengths.is_empty() {
        vec![(0, npos)]
    } else {
        let mut ranges = Vec::with_capacity(item.strip_lengths.len());
        let mut off = 0usize;
        for &l in &item.strip_lengths {
            ranges.push((off, off + l as usize));
            off += l as usize;
        }
        ranges
    };

    let total_segs: usize = strip_ranges
        .iter()
        .map(|&(s, e): &(usize, usize)| e.min(npos).saturating_sub(s).saturating_sub(1))
        .sum();
    let n = total_segs.min(item.edge_vectors.len());

    let mut glyph_positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut glyph_vectors: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut seg_idx = 0usize;

    'outer: for &(strip_start, strip_end) in &strip_ranges {
        let end = strip_end.min(npos);
        for i in strip_start..end.saturating_sub(1) {
            if seg_idx >= n {
                break 'outer;
            }
            let j = i + 1;
            let mid = [
                (positions[i][0] + positions[j][0]) * 0.5,
                (positions[i][1] + positions[j][1]) * 0.5,
                (positions[i][2] + positions[j][2]) * 0.5,
            ];
            glyph_positions.push(mid);
            glyph_vectors.push(item.edge_vectors[seg_idx]);
            seg_idx += 1;
        }
    }

    GlyphItem {
        positions: glyph_positions,
        vectors: glyph_vectors,
        scale: item.vector_scale,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_polyline(
        positions: Vec<[f32; 3]>,
        node_vectors: Vec<[f32; 3]>,
        edge_vectors: Vec<[f32; 3]>,
        strip_lengths: Vec<u32>,
    ) -> PolylineItem {
        PolylineItem {
            positions,
            node_vectors,
            edge_vectors,
            strip_lengths,
            vector_scale: 2.0,
            ..Default::default()
        }
    }

    #[test]
    fn node_vectors_positions_match_input() {
        let pl = make_polyline(
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            vec![],
            vec![],
        );
        let item = polyline_node_vectors_to_glyphs(&pl);
        assert_eq!(item.positions, pl.positions);
        assert_eq!(item.vectors, pl.node_vectors);
        assert!((item.scale - 2.0).abs() < 1e-6);
    }

    #[test]
    fn node_vectors_empty_returns_empty() {
        let pl = make_polyline(vec![[0.0; 3]; 3], vec![], vec![], vec![]);
        let item = polyline_node_vectors_to_glyphs(&pl);
        assert!(item.positions.is_empty());
    }

    #[test]
    fn edge_vectors_midpoints_correct() {
        let pl = make_polyline(
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 4.0, 0.0]],
            vec![],
            vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![], // single strip (all positions)
        );
        let item = polyline_edge_vectors_to_glyphs(&pl);
        assert_eq!(item.positions.len(), 2);
        // Midpoint of seg 0: (0+2)/2 = 1
        assert!((item.positions[0][0] - 1.0).abs() < 1e-5);
        // Midpoint of seg 1: ((2+2)/2, (0+4)/2) = (2, 2)
        assert!((item.positions[1][0] - 2.0).abs() < 1e-5);
        assert!((item.positions[1][1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn edge_vectors_with_strip_lengths() {
        let pl = make_polyline(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            vec![],
            vec![[1.0, 0.0, 0.0]; 2], // 2 edges : strip1 has 1 edge, strip2 has 1 edge
            vec![2, 2],               // two strips of 2 nodes each
        );
        let item = polyline_edge_vectors_to_glyphs(&pl);
        assert_eq!(item.positions.len(), 2);
        // First strip midpoint: (0+1)/2 = 0.5
        assert!((item.positions[0][0] - 0.5).abs() < 1e-5);
        // Second strip midpoint: (2+3)/2 = 2.5
        assert!((item.positions[1][0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn edge_vectors_empty_returns_empty() {
        let pl = make_polyline(vec![[0.0; 3]; 3], vec![], vec![], vec![]);
        let item = polyline_edge_vectors_to_glyphs(&pl);
        assert!(item.positions.is_empty());
    }
}
