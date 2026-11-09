//! Tessellated immediate overlay geometry kept across frames.
//!
//! An immediate vector shape or polyline is re-submitted every frame, and
//! tessellating its fill, stroke and shadow bands is most of what it costs to
//! prepare. An item whose content has not changed tessellates to the same
//! triangles, moved by however far the item moved, so the vertices and the
//! fill records they read are kept here, keyed by the item's content hash
//! measured from its placement, plus the device map the geometry depends on.
//! Translation is not in the key: a moving item reuses its entry and the
//! vertices are shifted. An entry not drawn for [`EVICT_AFTER`] overlay passes
//! is dropped.

use crate::resources::OverlayTextVertex;

/// Overlay passes an entry survives without being drawn.
const EVICT_AFTER: u64 = 60;

/// One item's emitted geometry, as recorded the frame it was tessellated.
struct Entry {
    /// The placement the vertices were emitted at.
    reference: [f32; 2],
    /// The vertices, before clip, colour-matrix and draw-order stamping.
    verts: Vec<OverlayTextVertex>,
    /// The fill records the vertices read, in the order they were written.
    texels: Vec<[f32; 4]>,
    /// The index the first of `texels` had in its frame's fill table.
    base: f32,
    /// Where a vector shape's coverage-atlas quad sits in `verts`. The quad is
    /// rebuilt on every reuse, because its atlas cell can move.
    atlas_quad: Option<usize>,
    /// A value the caller keeps with the entry: a vector shape's coverage-cell
    /// path hash, so a reuse does not hash the path again.
    aux: u64,
    last_used: u64,
}

/// What a reuse appended.
pub(super) struct Replayed {
    /// Where the atlas quad starts in the batch, for a vector shape whose fill
    /// draws from the coverage atlas.
    pub atlas_quad: Option<usize>,
    /// The value recorded with the entry.
    pub aux: u64,
}

/// The cache, one per renderer: entries are keyed by content, so viewports
/// share them.
#[derive(Default)]
pub(crate) struct TessellationCache {
    entries: std::collections::HashMap<u64, Entry>,
    pass: u64,
    /// Items tessellated in the current overlay pass: the misses.
    pub(crate) tessellations: u32,
}

impl TessellationCache {
    /// Start an overlay pass: reset the miss count and drop entries that have
    /// not been drawn for a while.
    pub(super) fn begin_pass(&mut self) {
        self.pass += 1;
        self.tessellations = 0;
        let pass = self.pass;
        self.entries
            .retain(|_, e| pass.saturating_sub(e.last_used) <= EVICT_AFTER);
    }

    /// Append the geometry recorded under `key` to `batch`, moved from where it
    /// was recorded to `reference`, and its fill records to `fills`. `None` on
    /// a miss.
    pub(super) fn replay(
        &mut self,
        key: u64,
        reference: [f32; 2],
        batch: &mut Vec<OverlayTextVertex>,
        fills: &mut super::overlay_fill::FillTable,
    ) -> Option<Replayed> {
        let entry = self.entries.get_mut(&key)?;
        entry.last_used = self.pass;
        let start = batch.len();
        let new_base = fills.append_raw(&entry.texels) as f32;
        let delta = [
            reference[0] - entry.reference[0],
            reference[1] - entry.reference[1],
        ];
        let moved = delta != [0.0, 0.0];
        let rebase = new_base != entry.base;
        batch.extend_from_slice(&entry.verts);
        if moved || rebase {
            for v in &mut batch[start..] {
                if moved {
                    v.position = [v.position[0] + delta[0], v.position[1] + delta[1]];
                }
                if rebase && v.fill[2] >= entry.base {
                    v.fill[2] = v.fill[2] - entry.base + new_base;
                }
            }
        }
        Some(Replayed {
            atlas_quad: entry.atlas_quad.map(|q| start + q),
            aux: entry.aux,
        })
    }

    /// Keep `batch[from..]` and the fill records written from `fill_from` on
    /// as the geometry for `key`, placed at `reference`, with `aux` kept
    /// alongside.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &mut self,
        key: u64,
        reference: [f32; 2],
        batch: &[OverlayTextVertex],
        from: usize,
        fills: &super::overlay_fill::FillTable,
        fill_from: usize,
        atlas_quad: Option<usize>,
        aux: u64,
    ) {
        self.entries.insert(
            key,
            Entry {
                reference,
                verts: batch[from..].to_vec(),
                texels: fills.texels_from(fill_from).to_vec(),
                base: fill_from as f32,
                atlas_quad: atlas_quad.map(|q| q - from),
                aux,
                last_used: self.pass,
            },
        );
    }

    /// Count an item that was tessellated, whether or not it was recorded.
    pub(super) fn count_tessellation(&mut self) {
        self.tessellations += 1;
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// A hasher for cache keys: the content hash, then the parts of the device
/// map and the frame the geometry depends on.
pub(super) fn key_hasher() -> std::collections::hash_map::DefaultHasher {
    std::collections::hash_map::DefaultHasher::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(p: [f32; 2], fill: f32) -> OverlayTextVertex {
        let mut v = OverlayTextVertex {
            position: p,
            uv: [0.0; 2],
            colour: [1.0; 4],
            use_texture: 0.0,
            clip_index: -1.0,
            clip_rect: [0.0; 4],
            group_tint: 1.0,
            fill: OverlayTextVertex::NO_FILL,
            aa: OverlayTextVertex::NO_AA,
        };
        v.fill[2] = fill;
        v
    }

    #[test]
    fn a_replay_moves_the_vertices_and_rebases_the_fills() {
        let mut cache = TessellationCache::default();
        cache.begin_pass();
        let mut fills = super::super::overlay_fill::FillTable::new();
        fills.append_raw(&[[9.0; 4]; 3]);
        let fill_from = fills.len();
        fills.append_raw(&[[1.0; 4], [2.0; 4]]);
        let batch = vec![
            vertex([10.0, 20.0], fill_from as f32),
            vertex([0.0, 0.0], -1.0),
        ];
        cache.record(7, [5.0, 5.0], &batch, 0, &fills, fill_from, None, 0);

        cache.begin_pass();
        let mut next = super::super::overlay_fill::FillTable::new();
        let mut out = Vec::new();
        let r = cache.replay(7, [6.0, 4.0], &mut out, &mut next).unwrap();
        assert_eq!(r.atlas_quad, None);
        assert_eq!(out[0].position, [11.0, 19.0]);
        assert_eq!(out[0].fill[2], 0.0);
        assert_eq!(out[1].fill[2], -1.0);
        assert_eq!(next.texels_from(0), &[[1.0; 4], [2.0; 4]]);
        assert!(cache.replay(8, [0.0; 2], &mut out, &mut next).is_none());
    }

    #[test]
    fn entries_not_drawn_are_evicted() {
        let mut cache = TessellationCache::default();
        cache.begin_pass();
        let fills = super::super::overlay_fill::FillTable::new();
        cache.record(1, [0.0; 2], &[], 0, &fills, 0, None, 0);
        for _ in 0..EVICT_AFTER {
            cache.begin_pass();
        }
        assert_eq!(cache.len(), 1);
        cache.begin_pass();
        assert_eq!(cache.len(), 0);
    }
}
