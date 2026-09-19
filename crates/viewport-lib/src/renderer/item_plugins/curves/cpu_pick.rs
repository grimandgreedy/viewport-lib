//! Screen-space CPU pick helpers shared by the three curve mesh item types.
//!
//! The curve levels resolve in the priority STRIP > SEGMENT > POLY_NODE: a
//! query asking for several gets the finest one the item type answers, and a
//! query asking only for `OBJECT` gets the hit with no sub-object at all. Both
//! the point and rect helpers here encode that ordering once so the three
//! plugins agree on it.

use crate::plugin_api::pick_helpers::{
    pick_closest_polyline_segment, project_to_screen, segment_in_rect,
};
use crate::renderer::picking::helpers::{strip_for_node, strip_for_segment};
use crate::renderer::{PickHit, PickId, PickMask, PickRectResult, SubObjectRef};

/// The curve sub-object levels a query is asking for.
#[derive(Clone, Copy)]
pub(super) struct CurveLevels {
    pub(super) node: bool,
    pub(super) segment: bool,
    pub(super) strip: bool,
    pub(super) object: bool,
}

impl CurveLevels {
    pub(super) fn from_mask(mask: PickMask) -> Self {
        Self {
            node: mask.intersects(PickMask::POLY_NODE),
            segment: mask.intersects(PickMask::SEGMENT),
            strip: mask.intersects(PickMask::STRIP),
            object: mask.intersects(PickMask::OBJECT),
        }
    }

    /// `true` when the query asks for nothing this item type answers.
    pub(super) fn is_empty(self) -> bool {
        !(self.node || self.segment || self.strip || self.object)
    }
}

/// Nearest control point to the click, as a hit carrying the level the query
/// asked for. `radius_px` is the screen-space pick tolerance.
pub(super) fn node_hit(
    levels: CurveLevels,
    click_pos: glam::Vec2,
    pick_id: PickId,
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
    radius_px: f32,
) -> Option<PickHit> {
    let mut hit = crate::interaction::query::picking::pick_gaussian_splat_cpu(
        click_pos,
        pick_id.0,
        positions,
        glam::Mat4::IDENTITY,
        view_proj,
        viewport_size,
        radius_px,
    )?;
    if levels.node {
        // `pick_gaussian_splat_cpu` already reports the node index.
    } else if levels.strip {
        if let Some(SubObjectRef::Point(idx)) = hit.sub_object {
            hit.sub_object = Some(SubObjectRef::Strip(strip_for_node(idx, strip_lengths)));
        }
    } else {
        hit.sub_object = None;
    }
    Some(hit)
}

/// Closest point along any segment of the curve, as a hit carrying the level
/// the query asked for. Tests the whole segment, so a click anywhere along it
/// registers rather than only near a midpoint.
pub(super) fn segment_hit(
    levels: CurveLevels,
    click_pos: glam::Vec2,
    pick_id: PickId,
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
    threshold_px: f32,
) -> Option<(glam::Vec3, PickHit)> {
    let (seg_idx, world_pos) = pick_closest_polyline_segment(
        click_pos,
        viewport_size,
        view_proj,
        positions,
        strip_lengths,
        threshold_px,
    )?;
    Some((
        world_pos,
        hit_for_segment(levels, pick_id, seg_idx, world_pos, strip_lengths),
    ))
}

/// Build the hit for a segment already located by a geometric test.
pub(super) fn hit_for_segment(
    levels: CurveLevels,
    pick_id: PickId,
    seg_idx: u32,
    world_pos: glam::Vec3,
    strip_lengths: &[u32],
) -> PickHit {
    let sub_object = if levels.segment {
        Some(SubObjectRef::Segment(seg_idx))
    } else if levels.strip {
        Some(SubObjectRef::Strip(strip_for_segment(
            seg_idx,
            strip_lengths,
        )))
    } else {
        None
    };
    #[allow(deprecated)]
    PickHit {
        id: pick_id.0,
        sub_object,
        world_pos,
        normal: glam::Vec3::Z,
        scalar_value: None,
        sub_object_world_pos: None,
    }
}

/// Strip lengths for an item, treating an empty list as one strip covering
/// every position. Returned by value so the single-strip case has somewhere to
/// live.
pub(super) fn strips_or_single(positions: &[[f32; 3]], strip_lengths: &[u32]) -> Vec<u32> {
    if strip_lengths.is_empty() {
        vec![positions.len() as u32]
    } else {
        strip_lengths.to_vec()
    }
}

/// Accumulates one item's rect-pick result, keeping the level priority and
/// reporting each strip once however many of its nodes and segments were hit.
pub(super) struct RectAccumulator {
    levels: CurveLevels,
    pick_id: PickId,
    item_hit: bool,
    strips_hit: std::collections::HashSet<u32>,
}

impl RectAccumulator {
    pub(super) fn new(levels: CurveLevels, pick_id: PickId) -> Self {
        Self {
            levels,
            pick_id,
            item_hit: false,
            strips_hit: std::collections::HashSet::new(),
        }
    }

    /// Record a node inside the rectangle. Returns `false` when the caller can
    /// stop enumerating: an object-only query needs no further nodes.
    pub(super) fn node(
        &mut self,
        result: &mut PickRectResult,
        node_idx: u32,
        strips: &[u32],
    ) -> bool {
        self.item_hit = true;
        if self.levels.node {
            result
                .elements
                .push((self.pick_id.0, SubObjectRef::Point(node_idx)));
        } else if self.levels.strip {
            self.strips_hit.insert(strip_for_node(node_idx, strips));
        } else {
            return false;
        }
        true
    }

    /// Record a segment touching the rectangle. Returns `false` when the caller
    /// can stop enumerating.
    pub(super) fn segment(
        &mut self,
        result: &mut PickRectResult,
        seg_idx: u32,
        strips: &[u32],
    ) -> bool {
        self.item_hit = true;
        if self.levels.segment {
            result
                .elements
                .push((self.pick_id.0, SubObjectRef::Segment(seg_idx)));
        } else if self.levels.strip {
            self.strips_hit.insert(strip_for_segment(seg_idx, strips));
        } else {
            return false;
        }
        true
    }

    /// Flush the collected strips and the object-level hit into `result`.
    pub(super) fn finish(self, result: &mut PickRectResult) {
        if self.levels.strip {
            for s in self.strips_hit {
                result
                    .elements
                    .push((self.pick_id.0, SubObjectRef::Strip(s)));
            }
        }
        if self.levels.object && self.item_hit {
            result.objects.push(self.pick_id.0);
        }
    }
}

/// Rect pick for a curve whose segments are line-like: streamtube and tube,
/// which both sweep their mesh along the control polyline. Nodes are tested as
/// projected points, segments as projected line segments.
pub(super) fn rect_pick_line_curve(
    levels: CurveLevels,
    result: &mut PickRectResult,
    pick_id: PickId,
    positions: &[[f32; 3]],
    strip_lengths: &[u32],
    view_proj: glam::Mat4,
    viewport_size: glam::Vec2,
    rect_min: glam::Vec2,
    rect_max: glam::Vec2,
) {
    let in_rect = |p: glam::Vec2| {
        p.x >= rect_min.x && p.x <= rect_max.x && p.y >= rect_min.y && p.y <= rect_max.y
    };
    let screen: Vec<Option<glam::Vec2>> = positions
        .iter()
        .map(|p| project_to_screen(glam::Vec3::from(*p), view_proj, viewport_size))
        .collect();
    let strips = strips_or_single(positions, strip_lengths);
    let mut acc = RectAccumulator::new(levels, pick_id);

    if levels.node || levels.strip || levels.object {
        for (node_idx, p) in screen.iter().enumerate() {
            if p.is_some_and(in_rect) && !acc.node(result, node_idx as u32, strip_lengths) {
                break;
            }
        }
    }

    if levels.segment || levels.strip || levels.object {
        let mut node_off = 0usize;
        let mut seg_off = 0u32;
        'strips: for &slen in &strips {
            let slen = slen as usize;
            for j in 0..slen.saturating_sub(1) {
                let (a, b) = (node_off + j, node_off + j + 1);
                let hit = match (screen[a], screen[b]) {
                    (Some(sa), Some(sb)) => segment_in_rect(sa, sb, rect_min, rect_max),
                    (Some(sa), None) => in_rect(sa),
                    (None, Some(sb)) => in_rect(sb),
                    (None, None) => false,
                };
                if hit && !acc.segment(result, seg_off + j as u32, strip_lengths) {
                    break 'strips;
                }
            }
            seg_off += slen.saturating_sub(1) as u32;
            node_off += slen;
        }
    }

    acc.finish(result);
}
