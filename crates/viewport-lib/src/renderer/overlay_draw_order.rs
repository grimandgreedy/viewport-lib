//! Global draw ordering for overlay families.
//!
//! Overlay items across families (shapes, labels, glyph runs, polylines) each
//! carry a `z_order`. The overlay prepare passes build one vertex buffer per
//! family; alongside each buffer they record which slice draws at which
//! `z_order` as an [`OverlayDrawSegment`]. Sorting the segments by
//! `(z_order, family_rank)` gives a single stacking context across families,
//! instead of the fixed family sequence the emit path uses when no item sets a
//! `z_order`.
//!
//! `family_rank` is the tiebreak for equal `z_order`. It reproduces the
//! back-to-front family order the renderer uses by default (shapes under the
//! merged text batch), so a scene that leaves every `z_order` at its default `0`
//! keeps the same result.
//!
//! Backdrop-blur shapes are not represented here: they are composited by a
//! separate pass, not the ordered overlay draw.

/// Equal-`z_order` tiebreak ranks, matching the historical back-to-front order.
pub(crate) mod family_rank {
    /// Untextured and textured SDF shapes (bottom).
    pub(crate) const SHAPE: u8 = 0;
    /// Merged labels, glyph runs, and polylines (top).
    pub(crate) const TEXT_MERGED: u8 = 1;
}

/// Locates the vertices (or draw) one segment issues. Ranges index into the
/// per-family buffers built by the overlay prepare passes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OverlayDrawSource {
    /// A sub-range of the solid (untextured) shape vertex buffer.
    Shape {
        vertex_start: u32,
        vertex_count: u32,
    },
    /// A sub-range of one textured-shape batch's vertex buffer. `batch_index`
    /// selects the batch (and its texture bind group); `vertex_start` /
    /// `vertex_count` bound the draw within that buffer. Shapes sharing a texture
    /// share a batch buffer but draw as separate segments at their own `z_order`,
    /// so a solid shape can layer between two same-texture shapes.
    ShapeTex {
        batch_index: u32,
        vertex_start: u32,
        vertex_count: u32,
    },
    /// A sub-range of the merged text-pipeline buffer (`label_gpu_data`): labels,
    /// glyph runs, and polylines.
    Text {
        vertex_start: u32,
        vertex_count: u32,
    },
    /// A retained overlay group drawn through the text pipeline. `draw_index`
    /// selects the group's cached buffer and per-frame instance in
    /// `overlay_retained_draws`. Each retained group is its own draw (its own
    /// buffer and instance), so these do not coalesce.
    Retained {
        draw_index: u32,
    },
}

/// A single draw within the overlay pass, tagged with its z-order and family
/// rank so all families can be ordered together.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct OverlayDrawSegment {
    /// The item's `z_order`. Lower draws first (further back).
    pub z_order: i32,
    /// Equal-`z_order` tiebreak; see [`family_rank`].
    pub family_rank: u8,
    /// What to draw.
    pub source: OverlayDrawSource,
}

impl OverlayDrawSegment {
    /// Append a merged-text segment, coalescing with the previous text segment at
    /// the same `z_order` when the ranges are contiguous. The merged batch is
    /// built in z-sorted order, so runs of same-`z_order` items collapse to one
    /// draw.
    pub fn push_text(
        segments: &mut Vec<OverlayDrawSegment>,
        z_order: i32,
        vertex_start: u32,
        vertex_count: u32,
    ) {
        if vertex_count == 0 {
            return;
        }
        if let Some(last) = segments.last_mut() {
            if last.z_order == z_order && last.family_rank == family_rank::TEXT_MERGED {
                if let OverlayDrawSource::Text {
                    vertex_start: ls,
                    vertex_count: lc,
                } = &mut last.source
                {
                    if *ls + *lc == vertex_start {
                        *lc += vertex_count;
                        return;
                    }
                }
            }
        }
        segments.push(OverlayDrawSegment {
            z_order,
            family_rank: family_rank::TEXT_MERGED,
            source: OverlayDrawSource::Text {
                vertex_start,
                vertex_count,
            },
        });
    }

    /// Append a solid-shape segment, coalescing with the previous solid-shape
    /// segment at the same `z_order` when the ranges are contiguous. Solid
    /// shapes are built in z-sorted order, so equal-`z_order` runs collapse to
    /// one draw.
    pub fn push_shape(
        segments: &mut Vec<OverlayDrawSegment>,
        z_order: i32,
        vertex_start: u32,
        vertex_count: u32,
    ) {
        if vertex_count == 0 {
            return;
        }
        if let Some(last) = segments.last_mut() {
            if last.z_order == z_order && last.family_rank == family_rank::SHAPE {
                if let OverlayDrawSource::Shape {
                    vertex_start: ls,
                    vertex_count: lc,
                } = &mut last.source
                {
                    if *ls + *lc == vertex_start {
                        *lc += vertex_count;
                        return;
                    }
                }
            }
        }
        segments.push(OverlayDrawSegment {
            z_order,
            family_rank: family_rank::SHAPE,
            source: OverlayDrawSource::Shape {
                vertex_start,
                vertex_count,
            },
        });
    }

    /// Append a textured-shape segment, coalescing with the previous textured
    /// segment when it targets the same batch at the same `z_order` and the
    /// ranges are contiguous. Textured shapes are recorded in z-sorted order, so
    /// equal-`z_order` runs sharing a texture collapse to one draw while shapes
    /// at different `z_order` stay separate (so a solid shape can layer between
    /// them).
    pub fn push_shape_tex(
        segments: &mut Vec<OverlayDrawSegment>,
        z_order: i32,
        batch_index: u32,
        vertex_start: u32,
        vertex_count: u32,
    ) {
        if vertex_count == 0 {
            return;
        }
        if let Some(last) = segments.last_mut() {
            if last.z_order == z_order && last.family_rank == family_rank::SHAPE {
                if let OverlayDrawSource::ShapeTex {
                    batch_index: lb,
                    vertex_start: ls,
                    vertex_count: lc,
                } = &mut last.source
                {
                    if *lb == batch_index && *ls + *lc == vertex_start {
                        *lc += vertex_count;
                        return;
                    }
                }
            }
        }
        segments.push(OverlayDrawSegment {
            z_order,
            family_rank: family_rank::SHAPE,
            source: OverlayDrawSource::ShapeTex {
                batch_index,
                vertex_start,
                vertex_count,
            },
        });
    }

    /// Append a retained-group segment. Ranked with the merged text batch (a
    /// retained group draws through the text pipeline), and never coalesced: each
    /// group has its own cached buffer and per-frame instance.
    pub fn push_retained(segments: &mut Vec<OverlayDrawSegment>, z_order: i32, draw_index: u32) {
        segments.push(OverlayDrawSegment {
            z_order,
            family_rank: family_rank::TEXT_MERGED,
            source: OverlayDrawSource::Retained { draw_index },
        });
    }
}

/// Stable-sort the overlay draw segments into paint order: ascending `z_order`,
/// then ascending `family_rank`. The sort is stable, so segments that tie on
/// both keys keep their insertion order (which preserves within-family and
/// solid-before-textured ordering).
pub(crate) fn sort_overlay_segments(segments: &mut [OverlayDrawSegment]) {
    segments.sort_by(|a, b| {
        a.z_order
            .cmp(&b.z_order)
            .then_with(|| a.family_rank.cmp(&b.family_rank))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(z: i32, rank: u8) -> OverlayDrawSegment {
        OverlayDrawSegment {
            z_order: z,
            family_rank: rank,
            source: OverlayDrawSource::Shape {
                vertex_start: 0,
                vertex_count: 0,
            },
        }
    }

    #[test]
    fn all_default_zorder_keeps_family_order() {
        // Every family at z 0: order collapses to the family-rank tiebreak, so
        // shapes draw under the merged text batch.
        let mut segs = vec![seg(0, family_rank::TEXT_MERGED), seg(0, family_rank::SHAPE)];
        sort_overlay_segments(&mut segs);
        let ranks: Vec<u8> = segs.iter().map(|s| s.family_rank).collect();
        assert_eq!(ranks, vec![family_rank::SHAPE, family_rank::TEXT_MERGED]);
    }

    #[test]
    fn zorder_beats_family_rank() {
        // A shape (lowest rank) at high z draws above a label at z 0: this is
        // the repro the issue names.
        let mut segs = vec![
            seg(1000, family_rank::SHAPE),
            seg(0, family_rank::TEXT_MERGED),
        ];
        sort_overlay_segments(&mut segs);
        assert_eq!(segs[0].z_order, 0); // label first (behind)
        assert_eq!(segs[1].z_order, 1000); // shape last (on top)
    }

    #[test]
    fn stable_within_equal_keys() {
        // Two segments with identical (z, rank) keep insertion order.
        let a = OverlayDrawSegment {
            z_order: 5,
            family_rank: family_rank::SHAPE,
            source: OverlayDrawSource::Shape {
                vertex_start: 0,
                vertex_count: 6,
            },
        };
        let b = OverlayDrawSegment {
            z_order: 5,
            family_rank: family_rank::SHAPE,
            source: OverlayDrawSource::Shape {
                vertex_start: 6,
                vertex_count: 6,
            },
        };
        let mut segs = vec![a, b];
        sort_overlay_segments(&mut segs);
        assert_eq!(segs[0].source, a.source);
        assert_eq!(segs[1].source, b.source);
    }

    #[test]
    fn push_text_coalesces_contiguous_same_z() {
        let mut segs = Vec::new();
        OverlayDrawSegment::push_text(&mut segs, 0, 0, 12);
        OverlayDrawSegment::push_text(&mut segs, 0, 12, 8);
        assert_eq!(segs.len(), 1);
        assert_eq!(
            segs[0].source,
            OverlayDrawSource::Text {
                vertex_start: 0,
                vertex_count: 20,
            }
        );
    }

    #[test]
    fn push_text_splits_on_different_z() {
        let mut segs = Vec::new();
        OverlayDrawSegment::push_text(&mut segs, 0, 0, 12);
        OverlayDrawSegment::push_text(&mut segs, 5, 12, 8);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn push_shape_coalesces_contiguous_same_z() {
        let mut segs = Vec::new();
        OverlayDrawSegment::push_shape(&mut segs, 3, 0, 6);
        OverlayDrawSegment::push_shape(&mut segs, 3, 6, 6);
        assert_eq!(segs.len(), 1);
        assert_eq!(
            segs[0].source,
            OverlayDrawSource::Shape {
                vertex_start: 0,
                vertex_count: 12,
            }
        );
    }

    #[test]
    fn push_shape_tex_coalesces_same_batch_same_z() {
        let mut segs = Vec::new();
        OverlayDrawSegment::push_shape_tex(&mut segs, 3, 0, 0, 6);
        OverlayDrawSegment::push_shape_tex(&mut segs, 3, 0, 6, 6);
        assert_eq!(segs.len(), 1);
        assert_eq!(
            segs[0].source,
            OverlayDrawSource::ShapeTex {
                batch_index: 0,
                vertex_start: 0,
                vertex_count: 12,
            }
        );
    }

    #[test]
    fn push_shape_tex_splits_on_different_z_or_batch() {
        let mut segs = Vec::new();
        // Same batch, different z: two segments.
        OverlayDrawSegment::push_shape_tex(&mut segs, 3, 0, 0, 6);
        OverlayDrawSegment::push_shape_tex(&mut segs, 4, 0, 6, 6);
        // Same z, different batch: another segment.
        OverlayDrawSegment::push_shape_tex(&mut segs, 4, 1, 0, 6);
        assert_eq!(segs.len(), 3);
    }

    #[test]
    fn solid_layers_between_two_same_texture_shapes() {
        // The repro: three textured shapes share batch 0 but are recorded at
        // their own z (1001, 1003, 1005), with solid plates interleaved at
        // 1000/1002/1004. After sorting, each image draws above its own plate,
        // instead of the whole texture batch collapsing to z 1001.
        let mut segs = Vec::new();
        OverlayDrawSegment::push_shape(&mut segs, 1000, 0, 6); // plate A
        OverlayDrawSegment::push_shape_tex(&mut segs, 1001, 0, 0, 6); // image X
        OverlayDrawSegment::push_shape(&mut segs, 1002, 6, 6); // plate B
        OverlayDrawSegment::push_shape_tex(&mut segs, 1003, 0, 6, 6); // image Y
        OverlayDrawSegment::push_shape(&mut segs, 1004, 12, 6); // plate C
        OverlayDrawSegment::push_shape_tex(&mut segs, 1005, 0, 12, 6); // image Z
        sort_overlay_segments(&mut segs);
        let zs: Vec<i32> = segs.iter().map(|s| s.z_order).collect();
        assert_eq!(zs, vec![1000, 1001, 1002, 1003, 1004, 1005]);
        // Each image sits directly above its plate: the segment after every
        // solid plate is the matching textured sub-range.
        assert_eq!(
            segs[3].source,
            OverlayDrawSource::ShapeTex {
                batch_index: 0,
                vertex_start: 6,
                vertex_count: 6,
            }
        );
    }
}
