//! Loose octree spatial index for frustum culling acceleration.
//!
//! Each scene node with a mesh is stored in the octree at the deepest level
//! whose loose bounds contain the node's world-space AABB. The loose factor
//! is 2.0: the effective cell size at each depth is twice the strict cell
//! size. This guarantees that any AABB fits within exactly one loose cell
//! at its optimal depth, with no boundary ambiguity.
//!
//! During culling, subtrees whose loose bounds are entirely outside the
//! frustum are skipped in O(1). Entries at each visited cell are then tested
//! individually against the frustum, so the cost scales with visible geometry
//! rather than total scene size.

use std::collections::{HashMap, HashSet};

use crate::camera::frustum::{CullStats, Frustum};
use crate::interaction::selection::NodeId;
use crate::scene::aabb::Aabb;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Half-size of the root octree cell in world units. Scenes larger than this
/// will have nodes stored at the root level (no subtree culling benefit, but
/// still correct). 65536 covers scenes up to 131,072 units across.
const ROOT_HALF: f32 = 65536.0;

/// Maximum octree depth. At depth 8 the strict cell half-size is
/// 65536 / 256 = 256 world units.
const MAX_DEPTH: u32 = 8;

/// Each cell's loose bound is this factor times its strict half-size.
const LOOSE_FACTOR: f32 = 2.0;

// ---------------------------------------------------------------------------
// OctCell
// ---------------------------------------------------------------------------

struct OctCell {
    center: glam::Vec3,
    /// Strict half-size of this cell (loose half = half * LOOSE_FACTOR).
    half: f32,
    /// Nodes whose AABBs are assigned to this cell level.
    entries: Vec<(NodeId, Aabb)>,
    /// Eight child octants (None = not yet allocated).
    children: [Option<Box<OctCell>>; 8],
}

impl OctCell {
    fn new(center: glam::Vec3, half: f32) -> Self {
        Self {
            center,
            half,
            entries: Vec::new(),
            children: Default::default(),
        }
    }

    fn loose_aabb(&self) -> Aabb {
        let ext = self.half * LOOSE_FACTOR;
        Aabb {
            min: self.center - glam::Vec3::splat(ext),
            max: self.center + glam::Vec3::splat(ext),
        }
    }

    /// Child index (0..7) for the octant containing `point`.
    fn child_idx(&self, point: glam::Vec3) -> usize {
        usize::from(point.x >= self.center.x)
            | (usize::from(point.y >= self.center.y) << 1)
            | (usize::from(point.z >= self.center.z) << 2)
    }

    fn child_center(&self, idx: usize) -> glam::Vec3 {
        let q = self.half * 0.5;
        self.center
            + glam::Vec3::new(
                if idx & 1 != 0 { q } else { -q },
                if idx & 2 != 0 { q } else { -q },
                if idx & 4 != 0 { q } else { -q },
            )
    }
}

// ---------------------------------------------------------------------------
// Recursive helpers
// ---------------------------------------------------------------------------

fn insert_recursive(cell: &mut OctCell, id: NodeId, aabb: &Aabb, depth: u32, path: &mut Vec<u8>) {
    let node_half = aabb.half_extents().max_element();
    let child_strict_half = cell.half * 0.5;
    let child_loose_half = child_strict_half * LOOSE_FACTOR;

    if depth < MAX_DEPTH && node_half <= child_loose_half {
        // Descend: node fits within the child's loose bounds.
        let ci = cell.child_idx(aabb.center());
        let cc = cell.child_center(ci);
        if cell.children[ci].is_none() {
            cell.children[ci] = Some(Box::new(OctCell::new(cc, child_strict_half)));
        }
        path.push(ci as u8);
        insert_recursive(cell.children[ci].as_mut().unwrap(), id, aabb, depth + 1, path);
    } else {
        // Store here: node is too large for any child, or max depth reached.
        cell.entries.push((id, *aabb));
    }
}

/// Navigate to the cell identified by `path` and remove the entry for `id`.
/// Returns true if the entry was found and removed.
fn remove_recursive(cell: &mut OctCell, id: NodeId, path: &[u8]) -> bool {
    if path.is_empty() {
        if let Some(pos) = cell.entries.iter().position(|(nid, _)| *nid == id) {
            cell.entries.swap_remove(pos);
            return true;
        }
        return false;
    }
    let ci = path[0] as usize;
    if let Some(child) = cell.children[ci].as_mut() {
        if remove_recursive(child, id, &path[1..]) {
            // Prune empty leaf cells to keep the tree compact.
            if child.entries.is_empty() && child.children.iter().all(|c| c.is_none()) {
                cell.children[ci] = None;
            }
            return true;
        }
    }
    false
}

fn collect_visible_recursive(
    cell: &OctCell,
    frustum: &Frustum,
    out: &mut Vec<NodeId>,
    stats: &mut CullStats,
) {
    // If the loose bounding box is entirely outside the frustum, skip the
    // whole subtree without visiting any entries.
    if frustum.cull_aabb(&cell.loose_aabb()) {
        return;
    }

    for (id, world_aabb) in &cell.entries {
        stats.total += 1;
        if frustum.cull_aabb(world_aabb) {
            stats.culled += 1;
        } else {
            stats.visible += 1;
            out.push(*id);
        }
    }

    for child in cell.children.iter().flatten() {
        collect_visible_recursive(child, frustum, out, stats);
    }
}

// ---------------------------------------------------------------------------
// SpatialIndex
// ---------------------------------------------------------------------------

/// Loose octree spatial index for frustum culling.
///
/// The index stores world-space AABBs for scene nodes that have a mesh and
/// are visible. When a node's transform changes the caller must call
/// `mark_dirty`; the entry is refreshed at the next `flush_dirty` call.
pub(crate) struct SpatialIndex {
    root: OctCell,
    /// Path from root to the cell holding each node's entry.
    node_paths: HashMap<NodeId, Vec<u8>>,
    /// Nodes whose octree entries need to be refreshed.
    dirty: HashSet<NodeId>,
}

impl SpatialIndex {
    pub(crate) fn new() -> Self {
        Self {
            root: OctCell::new(glam::Vec3::ZERO, ROOT_HALF),
            node_paths: HashMap::new(),
            dirty: HashSet::new(),
        }
    }

    /// Number of nodes currently in the index.
    pub(crate) fn node_count(&self) -> u32 {
        self.node_paths.len() as u32
    }

    /// Insert a node with its world-space AABB. The node must not already be
    /// in the index.
    pub(crate) fn insert_entry(&mut self, id: NodeId, world_aabb: Aabb) {
        let mut path = Vec::new();
        insert_recursive(&mut self.root, id, &world_aabb, 0, &mut path);
        self.node_paths.insert(id, path);
    }

    /// Remove a node from the index. No-op if the node is not indexed.
    pub(crate) fn remove_entry(&mut self, id: NodeId) {
        if let Some(path) = self.node_paths.remove(&id) {
            remove_recursive(&mut self.root, id, &path);
        }
    }

    /// Mark a node as needing an update before the next cull pass.
    pub(crate) fn mark_dirty(&mut self, id: NodeId) {
        self.dirty.insert(id);
    }

    /// Remove a node from the index and cancel any pending dirty update.
    /// Used when the node is deleted from the scene.
    pub(crate) fn remove_node(&mut self, id: NodeId) {
        self.remove_entry(id);
        self.dirty.remove(&id);
    }

    /// Replace all current contents with a fresh build from `entries`.
    ///
    /// Use this for the initial build or after bulk scene construction.
    pub(crate) fn rebuild(&mut self, entries: &[(NodeId, Aabb)]) {
        self.root = OctCell::new(glam::Vec3::ZERO, ROOT_HALF);
        self.node_paths.clear();
        self.dirty.clear();
        for (id, aabb) in entries {
            let mut path = Vec::new();
            insert_recursive(&mut self.root, *id, aabb, 0, &mut path);
            self.node_paths.insert(*id, path);
        }
    }

    /// Drain and return the set of nodes pending an update.
    ///
    /// The caller is responsible for removing old entries and re-inserting
    /// with updated AABBs. Splitting this into a drain step lets the caller
    /// access other `SpatialIndex` methods and scene data within the same
    /// borrow scope.
    pub(crate) fn drain_dirty(&mut self) -> Vec<NodeId> {
        self.dirty.drain().collect()
    }

    /// Collect NodeIds whose world-space AABBs pass the frustum test.
    ///
    /// Subtrees whose loose bounds are entirely outside the frustum are
    /// pruned without visiting their contents. `stats` is updated for each
    /// entry that is individually tested.
    pub(crate) fn collect_visible(
        &self,
        frustum: &Frustum,
        out: &mut Vec<NodeId>,
        stats: &mut CullStats,
    ) {
        collect_visible_recursive(&self.root, frustum, out, stats);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frustum() -> Frustum {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 200.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 1.0, 300.0);
        Frustum::from_view_proj(&(proj * view))
    }

    /// Scatter nodes on a grid spanning [-500, 500]^3 and verify that the
    /// octree produces the same visible set as a brute-force flat scan.
    #[test]
    fn test_collect_visible_matches_brute_force() {
        let mut idx = SpatialIndex::new();
        let mut all_entries: Vec<(NodeId, Aabb)> = Vec::new();

        for i in 0u64..200 {
            // Pseudo-random scatter using simple multipliers.
            let x = (i as f32 * 37.5) % 1000.0 - 500.0;
            let y = (i as f32 * 13.7) % 1000.0 - 500.0;
            let z = (i as f32 * 53.3) % 1000.0 - 500.0;
            let aabb = Aabb {
                min: glam::Vec3::new(x - 1.0, y - 1.0, z - 1.0),
                max: glam::Vec3::new(x + 1.0, y + 1.0, z + 1.0),
            };
            idx.insert_entry(i, aabb);
            all_entries.push((i, aabb));
        }

        let frustum = make_frustum();

        // Octree result.
        let mut octree_ids: Vec<NodeId> = Vec::new();
        let mut stats = CullStats::default();
        idx.collect_visible(&frustum, &mut octree_ids, &mut stats);
        octree_ids.sort_unstable();

        // Brute-force result.
        let mut expected: Vec<NodeId> = all_entries
            .iter()
            .filter(|(_, aabb)| !frustum.cull_aabb(aabb))
            .map(|(id, _)| *id)
            .collect();
        expected.sort_unstable();

        assert_eq!(
            octree_ids, expected,
            "octree visible set must match flat walk"
        );
        assert_eq!(
            stats.visible as usize,
            expected.len(),
            "visible count must match"
        );
    }

    /// After removing an entry, it must not appear in subsequent traversals.
    #[test]
    fn test_remove_entry_not_returned() {
        let mut idx = SpatialIndex::new();
        let aabb = Aabb {
            min: glam::Vec3::splat(-1.0),
            max: glam::Vec3::splat(1.0),
        };
        idx.insert_entry(42, aabb);

        let frustum = make_frustum();
        let mut out = Vec::new();
        let mut stats = CullStats::default();
        idx.collect_visible(&frustum, &mut out, &mut stats);
        assert!(out.contains(&42), "node 42 should be visible before removal");

        idx.remove_entry(42);
        out.clear();
        stats = CullStats::default();
        idx.collect_visible(&frustum, &mut out, &mut stats);
        assert!(!out.contains(&42), "node 42 must not appear after removal");
    }

    /// Moving a node via flush_dirty must update its position in the octree.
    #[test]
    fn test_incremental_update_changes_visibility() {
        let frustum = make_frustum();

        let mut idx = SpatialIndex::new();

        // Start with the node at the origin (visible).
        let inside = Aabb {
            min: glam::Vec3::splat(-1.0),
            max: glam::Vec3::splat(1.0),
        };
        idx.insert_entry(1, inside);

        let mut out = Vec::new();
        let mut stats = CullStats::default();
        idx.collect_visible(&frustum, &mut out, &mut stats);
        assert!(out.contains(&1), "node should be visible at origin");

        // Move the node far behind the camera (z >> 200 + far plane distance).
        let outside = Aabb {
            min: glam::Vec3::new(-1.0, -1.0, 9999.0),
            max: glam::Vec3::new(1.0, 1.0, 10001.0),
        };
        idx.mark_dirty(1);
        // Use drain_dirty + remove/insert to update the entry.
        let dirty = idx.drain_dirty();
        for id in dirty {
            idx.remove_entry(id);
            idx.insert_entry(id, outside);
        }

        out.clear();
        stats = CullStats::default();
        idx.collect_visible(&frustum, &mut out, &mut stats);
        assert!(!out.contains(&1), "node should be culled after moving outside frustum");
    }

    /// rebuild() must produce the same visible set as individual inserts.
    #[test]
    fn test_rebuild_matches_incremental_inserts() {
        let mut incremental = SpatialIndex::new();
        let mut bulk_entries = Vec::new();

        for i in 0u64..50 {
            let x = (i as f32 * 19.3) % 200.0 - 100.0;
            let aabb = Aabb {
                min: glam::Vec3::new(x - 0.5, -0.5, -0.5),
                max: glam::Vec3::new(x + 0.5, 0.5, 0.5),
            };
            incremental.insert_entry(i, aabb);
            bulk_entries.push((i, aabb));
        }

        let mut bulk = SpatialIndex::new();
        bulk.rebuild(&bulk_entries);

        let frustum = make_frustum();

        let mut inc_out = Vec::new();
        let mut bulk_out = Vec::new();
        let mut stats = CullStats::default();
        incremental.collect_visible(&frustum, &mut inc_out, &mut stats);
        stats = CullStats::default();
        bulk.collect_visible(&frustum, &mut bulk_out, &mut stats);

        inc_out.sort_unstable();
        bulk_out.sort_unstable();
        assert_eq!(
            inc_out, bulk_out,
            "rebuild and incremental insert must produce the same visible set"
        );
    }
}
