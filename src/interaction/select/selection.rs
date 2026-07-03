//! Multi-select system for viewport objects.
//!
//! `Selection` tracks a set of selected node IDs with a designated primary
//! (most recently selected). Supports single-click, shift-click toggle,
//! box select, and select-all operations.

use std::collections::HashSet;

/// Node identifier : matches `ViewportObject::id()` return type.
pub type NodeId = u64;

/// A set of selected nodes with a primary (most recently selected) node.
#[derive(Debug, Clone)]
pub struct Selection {
    selected: HashSet<NodeId>,
    primary: Option<NodeId>,
    /// Monotonically increasing generation counter. Incremented on every mutation.
    /// Compare against a cached value to detect selection changes without hashing.
    version: u64,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            selected: HashSet::new(),
            primary: None,
            version: 0,
        }
    }
}

impl Selection {
    /// Create an empty selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Monotonically increasing generation counter.
    ///
    /// Incremented by `wrapping_add(1)` on every mutation. Compare against a
    /// cached value to detect selection changes without hashing.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Clear the selection and select a single node.
    pub fn select_one(&mut self, id: NodeId) {
        self.selected.clear();
        self.selected.insert(id);
        self.primary = Some(id);
        self.version = self.version.wrapping_add(1);
    }

    /// Toggle a node's selection (shift-click behavior).
    /// If added, it becomes the primary. If removed, primary is cleared
    /// (or set to an arbitrary remaining node).
    pub fn toggle(&mut self, id: NodeId) {
        if self.selected.contains(&id) {
            self.selected.remove(&id);
            if self.primary == Some(id) {
                self.primary = self.selected.iter().next().copied();
            }
        } else {
            self.selected.insert(id);
            self.primary = Some(id);
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Add a node to the selection without removing others.
    pub fn add(&mut self, id: NodeId) {
        self.selected.insert(id);
        self.primary = Some(id);
        self.version = self.version.wrapping_add(1);
    }

    /// Remove a node from the selection.
    pub fn remove(&mut self, id: NodeId) {
        self.selected.remove(&id);
        if self.primary == Some(id) {
            self.primary = self.selected.iter().next().copied();
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Clear the entire selection.
    pub fn clear(&mut self) {
        self.selected.clear();
        self.primary = None;
        self.version = self.version.wrapping_add(1);
    }

    /// Add multiple nodes (e.g. from box select). The last one becomes primary.
    pub fn extend(&mut self, ids: impl IntoIterator<Item = NodeId>) {
        let mut last = None;
        for id in ids {
            self.selected.insert(id);
            last = Some(id);
        }
        if let Some(id) = last {
            self.primary = Some(id);
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Replace the entire selection with the given set.
    pub fn select_all(&mut self, ids: impl IntoIterator<Item = NodeId>) {
        self.selected.clear();
        self.primary = None;
        // Collect first so we can detect the empty-iterator case where extend()
        // would still increment the version (which is correct : clearing is a mutation).
        let ids: Vec<NodeId> = ids.into_iter().collect();
        if ids.is_empty() {
            self.version = self.version.wrapping_add(1);
        } else {
            self.extend(ids);
            // extend() already incremented version.
        }
    }

    /// Whether the given node is selected.
    pub fn contains(&self, id: NodeId) -> bool {
        self.selected.contains(&id)
    }

    /// The most recently selected node.
    pub fn primary(&self) -> Option<NodeId> {
        self.primary
    }

    /// Iterate over all selected node IDs.
    pub fn iter(&self) -> impl Iterator<Item = &NodeId> {
        self.selected.iter()
    }

    /// Number of selected nodes.
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    /// Whether the selection is empty.
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// Compute the centroid (average position) of all selected nodes.
    ///
    /// `position_fn` resolves a node ID to its world-space position.
    /// Returns `None` if the selection is empty or no positions are available.
    pub fn centroid(
        &self,
        position_fn: impl Fn(NodeId) -> Option<glam::Vec3>,
    ) -> Option<glam::Vec3> {
        let mut sum = glam::Vec3::ZERO;
        let mut count = 0u32;
        for &id in &self.selected {
            if let Some(pos) = position_fn(id) {
                sum += pos;
                count += 1;
            }
        }
        if count > 0 {
            Some(sum / count as f32)
        } else {
            None
        }
    }

    /// Compute the difference between this selection and a previous one.
    ///
    /// Returns `(added, removed)` : IDs that are in `self` but not `previous`,
    /// and IDs that are in `previous` but not `self`.
    pub fn diff(&self, previous: &Selection) -> (Vec<NodeId>, Vec<NodeId>) {
        let added: Vec<NodeId> = self
            .selected
            .difference(&previous.selected)
            .copied()
            .collect();
        let removed: Vec<NodeId> = previous
            .selected
            .difference(&self.selected)
            .copied()
            .collect();
        (added, removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_one_clears_others() {
        let mut sel = Selection::new();
        sel.add(1);
        sel.add(2);
        sel.select_one(3);
        assert_eq!(sel.len(), 1);
        assert!(sel.contains(3));
        assert!(!sel.contains(1));
        assert!(!sel.contains(2));
    }

    #[test]
    fn test_toggle_adds_and_removes() {
        let mut sel = Selection::new();
        sel.toggle(1);
        assert!(sel.contains(1));
        sel.toggle(1);
        assert!(!sel.contains(1));
        assert!(sel.is_empty());
    }

    #[test]
    fn test_add_preserves_existing() {
        let mut sel = Selection::new();
        sel.add(1);
        sel.add(2);
        assert!(sel.contains(1));
        assert!(sel.contains(2));
        assert_eq!(sel.len(), 2);
    }

    #[test]
    fn test_clear_empties() {
        let mut sel = Selection::new();
        sel.add(1);
        sel.add(2);
        sel.clear();
        assert!(sel.is_empty());
        assert_eq!(sel.primary(), None);
    }

    #[test]
    fn test_primary_tracks_last() {
        let mut sel = Selection::new();
        sel.add(1);
        assert_eq!(sel.primary(), Some(1));
        sel.add(2);
        assert_eq!(sel.primary(), Some(2));
        sel.select_one(3);
        assert_eq!(sel.primary(), Some(3));
    }

    #[test]
    fn test_centroid_computes_average() {
        let mut sel = Selection::new();
        sel.add(1);
        sel.add(2);
        let centroid = sel.centroid(|id| match id {
            1 => Some(glam::Vec3::new(0.0, 0.0, 0.0)),
            2 => Some(glam::Vec3::new(4.0, 0.0, 0.0)),
            _ => None,
        });
        let c = centroid.unwrap();
        assert!((c.x - 2.0).abs() < 1e-5);
        assert!((c.y).abs() < 1e-5);
    }

    #[test]
    fn test_diff_reports_changes() {
        let mut prev = Selection::new();
        prev.add(1);
        prev.add(2);

        let mut curr = Selection::new();
        curr.add(2);
        curr.add(3);

        let (added, removed) = curr.diff(&prev);
        assert_eq!(added, vec![3]);
        assert_eq!(removed, vec![1]);
    }
}
