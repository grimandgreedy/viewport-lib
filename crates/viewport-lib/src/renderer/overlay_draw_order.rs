//! Paint order for the overlay pass.
//!
//! Every overlay item (an analytic shape, an item on the text stream, or a
//! retained group's shape or text stream) carries a `z_order`. The contract is
//! painter's order: lower `z_order` draws first, and at an equal `z_order`
//! shapes draw under the text stream, each family in submission order. That
//! order only has to hold between items that overlap, so the renderer is free
//! to reorder the rest to batch them.
//!
//! The overlay prepare passes record one [`OverlayOrderItem`] per item, with
//! its screen bounds. [`build_draw_list`] puts the items in contract order and
//! gives each a rank: one more than the highest rank of any earlier item whose
//! bounds intersect it, zero when none does. Items of equal rank never overlap,
//! so the draw list sorts by rank and then by draw state (pipeline, fill image,
//! blend), writes the shape instances and text vertices in that order, and a
//! run that shares draw state becomes one draw whatever its items' `z_order`.
//!
//! Backdrop-blur shapes are not represented here: they are composited by a
//! separate pass, not the ordered overlay draw.

use crate::renderer::types::OverlayBlend;

/// Equal-`z_order` tiebreak ranks: shapes under the text stream.
pub(crate) mod family_rank {
    /// Analytic shapes, immediate or in a retained group (bottom).
    pub(crate) const SHAPE: u8 = 0;
    /// Labels, glyph runs, polylines and vector shapes, immediate or in a
    /// retained group (top).
    pub(crate) const TEXT_MERGED: u8 = 1;
}

/// An item's screen bounds in logical pixels, `[min_x, min_y, max_x, max_y]`.
/// A box with no area intersects nothing.
pub(crate) type Bounds = [f32; 4];

/// Bounds for an item whose extent is not known, which therefore overlaps
/// every item with area.
pub(crate) const UNBOUNDED: Bounds = [f32::MIN, f32::MIN, f32::MAX, f32::MAX];

/// Whether two boxes share interior. Boxes that only touch along an edge do
/// not: a pixel centre on a shared edge rasterises into one of the two. A box
/// without area overlaps nothing.
#[inline]
pub(crate) fn overlaps(a: &Bounds, b: &Bounds) -> bool {
    has_area(a) && has_area(b) && a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]
}

/// The bounding box of a set of points, or an empty box for none.
pub(crate) fn points_bounds(points: impl IntoIterator<Item = [f32; 2]>) -> Bounds {
    let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for p in points {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    b
}

/// The overlap of two boxes, empty when they are disjoint.
pub(crate) fn intersect(a: Bounds, b: Bounds) -> Bounds {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// Where an item's draw data lives before the draw list is built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OverlayItemSource {
    /// One shape instance at `index` in the frame's unsorted instance list.
    /// `fill` selects the fill bind group (the image a textured shape samples,
    /// or the blank image at `0`).
    Shape { index: u32, fill: u32 },
    /// A range of the frame's unsorted text-stream vertices: one label, glyph
    /// run, polyline or vector shape. `fill` selects the fill bind group, as
    /// for a shape.
    Text { start: u32, count: u32, fill: u32 },
    /// A retained group's text stream; `draw_index` selects
    /// `overlay_retained_draws`.
    Retained { draw_index: u32 },
    /// A retained group's shape stream; `draw_index` selects
    /// `overlay_retained_shape_draws`.
    RetainedShape { draw_index: u32 },
}

/// One item waiting for its place in the paint order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OverlayOrderItem {
    /// The item's `z_order`. Lower draws first (further back).
    pub z_order: i32,
    /// Equal-`z_order` tiebreak; see [`family_rank`].
    pub family_rank: u8,
    /// Everything the item can write to, in logical pixels.
    pub bounds: Bounds,
    /// The blend state it draws with.
    pub blend: OverlayBlend,
    /// Its draw data.
    pub source: OverlayItemSource,
}

/// Locates the vertices (or draw) one segment issues. Ranges index into the
/// draw-ordered buffers [`build_draw_list`] writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OverlayDrawSource {
    /// A sub-range of the shape instance buffer, one instance per shape.
    /// `fill` selects the fill bind group.
    Shape {
        instance_start: u32,
        instance_count: u32,
        fill: u32,
    },
    /// A sub-range of the text-stream vertex buffer (`label_gpu_data`):
    /// labels, glyph runs, polylines and vector shapes. `fill` selects the
    /// fill bind group.
    Text {
        vertex_start: u32,
        vertex_count: u32,
        fill: u32,
    },
    /// A retained group's text stream, drawn from its own buffer with its own
    /// instance, so it never coalesces.
    Retained { draw_index: u32 },
    /// A retained group's shape stream, drawn through the shape pipeline.
    RetainedShape { draw_index: u32 },
}

/// One draw within the overlay pass.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct OverlayDrawSegment {
    /// Which pipeline variant draws it.
    pub blend: OverlayBlend,
    /// What to draw.
    pub source: OverlayDrawSource,
}

impl OverlayDrawSegment {
    /// Whether the segment draws through the shape pipeline.
    pub(crate) fn is_shape(&self) -> bool {
        matches!(
            self.source,
            OverlayDrawSource::Shape { .. } | OverlayDrawSource::RetainedShape { .. }
        )
    }
}

/// Above this many items the ranking switches from comparing every pair to a
/// uniform grid over the items' bounds.
pub(crate) const RANK_GRID_THRESHOLD: usize = 64;

/// Allocations the ranking and draw-list build reuse from frame to frame.
#[derive(Default)]
pub(crate) struct OverlayOrderScratch {
    ranks: Vec<u32>,
    bounds: Vec<Bounds>,
    keys: Vec<(u32, u8, u32, usize)>,
    order: Vec<u32>,
    cells: Vec<Vec<u32>>,
    seen: Vec<u32>,
    blends: Vec<OverlayBlend>,
}

fn has_area(b: &Bounds) -> bool {
    b[0] < b[2] && b[1] < b[3]
}

/// Rank every item: one more than the highest rank of an earlier item whose
/// bounds intersect it, or zero. Brute force for small counts, a uniform grid
/// past [`RANK_GRID_THRESHOLD`]; the two give identical ranks.
fn compute_ranks(bounds: &[Bounds], scratch: &mut OverlayOrderScratch, ranks: &mut Vec<u32>) {
    if bounds.len() > RANK_GRID_THRESHOLD {
        ranks_grid(bounds, scratch, ranks);
    } else {
        ranks_brute(bounds, ranks);
    }
}

fn ranks_brute(bounds: &[Bounds], ranks: &mut Vec<u32>) {
    ranks.clear();
    for (i, b) in bounds.iter().enumerate() {
        let mut rank = 0;
        if has_area(b) {
            for (j, earlier) in bounds[..i].iter().enumerate() {
                if overlaps(b, earlier) {
                    rank = rank.max(ranks[j] + 1);
                }
            }
        }
        ranks.push(rank);
    }
}

fn ranks_grid(bounds: &[Bounds], scratch: &mut OverlayOrderScratch, ranks: &mut Vec<u32>) {
    ranks.clear();
    // The grid spans the union of the items' boxes, capped so an unbounded box
    // does not stretch every cell to nothing. A box reaching past the grid is
    // clamped to its edge cells, which only costs it a visit to each of them.
    const LIMIT: f32 = 1.0e7;
    let mut domain = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for b in bounds.iter().filter(|b| has_area(b)) {
        domain[0] = domain[0].min(b[0].max(-LIMIT));
        domain[1] = domain[1].min(b[1].max(-LIMIT));
        domain[2] = domain[2].max(b[2].min(LIMIT));
        domain[3] = domain[3].max(b[3].min(LIMIT));
    }
    if !has_area(&domain) {
        // No item has area, so nothing overlaps.
        ranks.resize(bounds.len(), 0);
        return;
    }
    // About four items per cell for an even spread.
    let side = ((bounds.len() as f32 / 4.0).sqrt().ceil() as usize).clamp(4, 64);
    let cell_w = (domain[2] - domain[0]) / side as f32;
    let cell_h = (domain[3] - domain[1]) / side as f32;
    let cell_of = |v: f32, origin: f32, size: f32| -> usize {
        (((v - origin) / size).floor().max(0.0) as usize).min(side - 1)
    };

    let cells = &mut scratch.cells;
    cells.resize_with(side * side, Vec::new);
    for c in cells.iter_mut() {
        c.clear();
    }
    let seen = &mut scratch.seen;
    seen.clear();
    seen.resize(bounds.len(), u32::MAX);

    for (i, b) in bounds.iter().enumerate() {
        if !has_area(b) {
            ranks.push(0);
            continue;
        }
        let (x0, x1) = (
            cell_of(b[0], domain[0], cell_w),
            cell_of(b[2], domain[0], cell_w),
        );
        let (y0, y1) = (
            cell_of(b[1], domain[1], cell_h),
            cell_of(b[3], domain[1], cell_h),
        );
        // Any shared interior point lies in a cell both boxes are registered
        // in, so the candidates include every earlier overlapping item. The
        // exact test is the brute-force one, so the ranks agree.
        let mut rank = 0;
        for y in y0..=y1 {
            for x in x0..=x1 {
                for &j in &cells[y * side + x] {
                    if seen[j as usize] == i as u32 {
                        continue;
                    }
                    seen[j as usize] = i as u32;
                    if overlaps(b, &bounds[j as usize]) {
                        rank = rank.max(ranks[j as usize] + 1);
                    }
                }
            }
        }
        ranks.push(rank);
        for y in y0..=y1 {
            for x in x0..=x1 {
                cells[y * side + x].push(i as u32);
            }
        }
    }
}

/// Put `items` in paint order and build the draw list.
///
/// Writes the shape instances (`shapes_in`, indexed by each shape item) and
/// text vertices (`text_in`, ranged by each text item) into `shapes_out` and
/// `text_out` in draw order, and one segment per run of consecutive items that
/// draw with the same pipeline, fill image and blend into `segments`.
/// Retained streams each draw on their own.
pub(crate) fn build_draw_list<S: Copy, T: Copy>(
    items: &mut [OverlayOrderItem],
    shapes_in: &[S],
    text_in: &[T],
    shapes_out: &mut Vec<S>,
    text_out: &mut Vec<T>,
    segments: &mut Vec<OverlayDrawSegment>,
    scratch: &mut OverlayOrderScratch,
) {
    shapes_out.clear();
    text_out.clear();
    segments.clear();
    // Contract order: `z_order`, then family, then the order the families
    // recorded their items in. The sort is stable.
    items.sort_by_key(|i| (i.z_order, i.family_rank));

    let mut bounds = std::mem::take(&mut scratch.bounds);
    bounds.clear();
    bounds.extend(items.iter().map(|i| i.bounds));
    let mut ranks = std::mem::take(&mut scratch.ranks);
    compute_ranks(&bounds, scratch, &mut ranks);
    scratch.bounds = bounds;

    // Draw state as a sort key after the rank: the pipeline, then the fill
    // image, then the blend, numbered in order of first use.
    let blends = &mut scratch.blends;
    blends.clear();
    let keys = &mut scratch.keys;
    keys.clear();
    for (item, &rank) in items.iter().zip(&ranks) {
        let (pipeline, fill) = match item.source {
            OverlayItemSource::Shape { fill, .. } => (0, fill),
            OverlayItemSource::RetainedShape { .. } => (0, u32::MAX),
            OverlayItemSource::Text { fill, .. } => (1, fill),
            OverlayItemSource::Retained { .. } => (1, u32::MAX),
        };
        let blend = blends
            .iter()
            .position(|b| *b == item.blend)
            .unwrap_or_else(|| {
                blends.push(item.blend);
                blends.len() - 1
            });
        keys.push((rank, pipeline, fill, blend));
    }
    scratch.ranks = ranks;
    let order = &mut scratch.order;
    order.clear();
    order.extend(0..items.len() as u32);
    order.sort_by_key(|&i| keys[i as usize]);

    for &i in order.iter() {
        let item = &items[i as usize];
        let source = match item.source {
            OverlayItemSource::Shape { index, fill } => {
                let start = shapes_out.len() as u32;
                shapes_out.push(shapes_in[index as usize]);
                if let Some(OverlayDrawSegment {
                    blend,
                    source:
                        OverlayDrawSource::Shape {
                            instance_start,
                            instance_count,
                            fill: last_fill,
                        },
                }) = segments.last_mut()
                {
                    if *blend == item.blend
                        && *last_fill == fill
                        && *instance_start + *instance_count == start
                    {
                        *instance_count += 1;
                        continue;
                    }
                }
                OverlayDrawSource::Shape {
                    instance_start: start,
                    instance_count: 1,
                    fill,
                }
            }
            OverlayItemSource::Text { start, count, fill } => {
                let at = text_out.len() as u32;
                text_out.extend_from_slice(&text_in[start as usize..(start + count) as usize]);
                if let Some(OverlayDrawSegment {
                    blend,
                    source:
                        OverlayDrawSource::Text {
                            vertex_start,
                            vertex_count,
                            fill: last_fill,
                        },
                }) = segments.last_mut()
                {
                    if *blend == item.blend
                        && *last_fill == fill
                        && *vertex_start + *vertex_count == at
                    {
                        *vertex_count += count;
                        continue;
                    }
                }
                OverlayDrawSource::Text {
                    vertex_start: at,
                    vertex_count: count,
                    fill,
                }
            }
            OverlayItemSource::Retained { draw_index } => {
                OverlayDrawSource::Retained { draw_index }
            }
            OverlayItemSource::RetainedShape { draw_index } => {
                OverlayDrawSource::RetainedShape { draw_index }
            }
        };
        segments.push(OverlayDrawSegment {
            blend: item.blend,
            source,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(z: i32, b: Bounds, index: u32, fill: u32) -> OverlayOrderItem {
        OverlayOrderItem {
            z_order: z,
            family_rank: family_rank::SHAPE,
            bounds: b,
            blend: OverlayBlend::SourceOver,
            source: OverlayItemSource::Shape { index, fill },
        }
    }

    fn text(z: i32, b: Bounds, start: u32, count: u32) -> OverlayOrderItem {
        OverlayOrderItem {
            z_order: z,
            family_rank: family_rank::TEXT_MERGED,
            bounds: b,
            blend: OverlayBlend::SourceOver,
            source: OverlayItemSource::Text {
                start,
                count,
                fill: 0,
            },
        }
    }

    fn build(
        items: &mut [OverlayOrderItem],
        shapes: &[u32],
        verts: &[u32],
    ) -> (Vec<u32>, Vec<u32>, Vec<OverlayDrawSegment>) {
        let (mut so, mut to, mut segs) = (Vec::new(), Vec::new(), Vec::new());
        build_draw_list(
            items,
            shapes,
            verts,
            &mut so,
            &mut to,
            &mut segs,
            &mut OverlayOrderScratch::default(),
        );
        (so, to, segs)
    }

    /// A small deterministic generator, so the test needs no dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32) / (1u64 << 31) as f32
        }
    }

    #[test]
    fn grid_and_brute_force_ranks_agree() {
        let mut rng = Lcg(7);
        for &n in &[65usize, 200, 1000] {
            let bounds: Vec<Bounds> = (0..n)
                .map(|i| {
                    // Mostly small boxes, some large, some without area, one
                    // unbounded: what a busy UI frame looks like.
                    if i == n / 2 {
                        return UNBOUNDED;
                    }
                    if i % 37 == 0 {
                        return [10.0, 10.0, 10.0, 30.0];
                    }
                    let x = rng.next() * 800.0;
                    let y = rng.next() * 600.0;
                    let big = i % 11 == 0;
                    let w = rng.next() * if big { 400.0 } else { 40.0 };
                    let h = rng.next() * if big { 300.0 } else { 30.0 };
                    [x, y, x + w, y + h]
                })
                .collect();
            let mut brute = Vec::new();
            ranks_brute(&bounds, &mut brute);
            let mut grid = Vec::new();
            ranks_grid(&bounds, &mut OverlayOrderScratch::default(), &mut grid);
            assert_eq!(brute, grid, "{n} boxes");
            assert!(brute.iter().any(|&r| r > 2), "the set should stack");
        }
    }

    #[test]
    fn touching_boxes_do_not_overlap() {
        assert!(!overlaps(&[0.0, 0.0, 10.0, 10.0], &[10.0, 0.0, 20.0, 10.0]));
        assert!(overlaps(&[0.0, 0.0, 10.0, 10.0], &[9.0, 9.0, 20.0, 20.0]));
        // A box without area overlaps nothing.
        assert!(!overlaps(&[5.0, 5.0, 5.0, 8.0], &UNBOUNDED));
    }

    #[test]
    fn disjoint_items_batch_across_z_order() {
        // Alternating shapes and text at rising z, side by side: two draws,
        // one per pipeline.
        let mut items = Vec::new();
        for k in 0..6u32 {
            let x = k as f32 * 20.0;
            items.push(shape(2 * k as i32, [x, 0.0, x + 10.0, 10.0], k, 0));
            items.push(text(2 * k as i32 + 1, [x, 20.0, x + 10.0, 30.0], k * 6, 6));
        }
        let shapes: Vec<u32> = (0..6).collect();
        let verts: Vec<u32> = (0..36).collect();
        let (so, to, segs) = build(&mut items, &shapes, &verts);
        assert_eq!(segs.len(), 2);
        assert_eq!(so, shapes);
        assert_eq!(to, verts);
    }

    #[test]
    fn overlapping_items_keep_z_order() {
        // A label over a shape over a label, all on one spot: three draws in
        // z order, whatever order they were recorded in.
        let b = [0.0, 0.0, 10.0, 10.0];
        let mut items = vec![text(2, b, 0, 3), shape(1, b, 0, 0), text(0, b, 3, 3)];
        let (_, to, segs) = build(&mut items, &[9], &[1, 1, 1, 0, 0, 0]);
        assert_eq!(segs.len(), 3);
        assert!(matches!(
            segs[0].source,
            OverlayDrawSource::Text {
                vertex_start: 0,
                ..
            }
        ));
        assert!(segs[1].is_shape());
        // The z 0 label's vertices come first in the buffer.
        assert_eq!(to, vec![0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn equal_z_keeps_shapes_under_text() {
        // Shapes rank under the text stream at an equal z_order, whatever
        // order the families were recorded in.
        let b = [0.0, 0.0, 10.0, 10.0];
        let mut items = vec![text(0, b, 0, 3), shape(0, b, 0, 0)];
        let (_, _, segs) = build(&mut items, &[0], &[0, 0, 0]);
        assert!(segs[0].is_shape());
        assert!(!segs[1].is_shape());
    }

    #[test]
    fn blend_and_image_split_a_run() {
        let mut items: Vec<OverlayOrderItem> = (0..4u32)
            .map(|k| {
                let x = k as f32 * 20.0;
                shape(0, [x, 0.0, x + 10.0, 10.0], k, 0)
            })
            .collect();
        items[1].blend = OverlayBlend::Additive;
        items[3].source = OverlayItemSource::Shape { index: 3, fill: 1 };
        let (so, _, segs) = build(&mut items, &[0, 1, 2, 3], &[]);
        // Source-over untextured (0 and 2), additive (1), then the image (3).
        assert_eq!(segs.len(), 3);
        assert_eq!(so, vec![0, 2, 1, 3]);
        assert_eq!(
            segs[0].source,
            OverlayDrawSource::Shape {
                instance_start: 0,
                instance_count: 2,
                fill: 0,
            }
        );
        assert_eq!(segs[1].blend, OverlayBlend::Additive);
    }

    #[test]
    fn retained_streams_never_coalesce() {
        let retained = |draw_index: u32, b: Bounds| OverlayOrderItem {
            z_order: 0,
            family_rank: family_rank::TEXT_MERGED,
            bounds: b,
            blend: OverlayBlend::SourceOver,
            source: OverlayItemSource::Retained { draw_index },
        };
        let mut items = vec![
            retained(0, [0.0, 0.0, 1.0, 1.0]),
            retained(1, [5.0, 5.0, 6.0, 6.0]),
        ];
        let (_, _, segs) = build(&mut items, &[] as &[u32], &[] as &[u32]);
        assert_eq!(segs.len(), 2);
    }
}
