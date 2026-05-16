//! Transform snapshot table for smooth physics-driven rendering.
//!
//! When using a fixed timestep, the render rate and simulation step rate differ.
//! Rendering directly from physics state causes visible jitter at the seam between
//! steps. Instead, store the previous and current transform for each physics-driven
//! node, then lerp between them using [`FixedTimestep::alpha`](super::timestep::FixedTimestep::alpha).
//!
//! The runtime populates this table automatically when applying
//! [`NodeTransformOp`](super::output::NodeTransformOp)s during `Writeback`.
//! Read it via [`ViewportRuntime::snapshots`](super::ViewportRuntime::snapshots).

use std::collections::HashMap;

use crate::interaction::selection::NodeId;

/// Previous and current world-space transforms for one physics-driven node.
#[derive(Debug, Clone, Copy)]
pub struct TransformSnapshot {
    /// Transform from the previous fixed step.
    pub prev: glam::Affine3A,
    /// Transform from the most recent fixed step.
    pub curr: glam::Affine3A,
}

/// Per-node transform snapshot table for rendering interpolation.
///
/// Keyed by [`NodeId`]. Lives inside [`super::ViewportRuntime`], separate from
/// the scene graph, so apps that do not use physics pay no overhead.
pub struct TransformSnapshotTable {
    entries: HashMap<NodeId, TransformSnapshot>,
}

impl TransformSnapshotTable {
    pub(super) fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Record a new transform for a node, shifting the current value to `prev`.
    ///
    /// On the first call for a node, both `prev` and `curr` are set to
    /// `transform` to avoid a spurious lerp from the origin.
    pub fn update(&mut self, id: NodeId, transform: glam::Affine3A) {
        let entry = self.entries.entry(id).or_insert(TransformSnapshot {
            prev: transform,
            curr: transform,
        });
        entry.prev = entry.curr;
        entry.curr = transform;
    }

    /// Get the raw snapshot for a node.
    pub fn get(&self, id: NodeId) -> Option<&TransformSnapshot> {
        self.entries.get(&id)
    }

    /// Compute the interpolated transform for a node at blend factor `alpha`.
    ///
    /// `alpha` comes from [`FixedTimestep::alpha`](super::timestep::FixedTimestep::alpha):
    /// `0.0` returns `prev`, `1.0` returns `curr`, values between lerp translation
    /// and scale while slerping rotation.
    ///
    /// Returns `None` if no snapshot exists for the node.
    pub fn interpolated(&self, id: NodeId, alpha: f32) -> Option<glam::Affine3A> {
        let s = self.entries.get(&id)?;
        let (prev_scale, prev_rot, prev_trans) =
            glam::Mat4::from(s.prev).to_scale_rotation_translation();
        let (curr_scale, curr_rot, curr_trans) =
            glam::Mat4::from(s.curr).to_scale_rotation_translation();
        Some(glam::Affine3A::from_scale_rotation_translation(
            prev_scale.lerp(curr_scale, alpha),
            prev_rot.slerp(curr_rot, alpha),
            prev_trans.lerp(curr_trans, alpha),
        ))
    }

    /// Remove the snapshot for a node.
    pub fn remove(&mut self, id: NodeId) {
        self.entries.remove(&id);
    }

    /// Clear all snapshots.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of nodes tracked.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translation(x: f32) -> glam::Affine3A {
        glam::Affine3A::from_translation(glam::Vec3::new(x, 0.0, 0.0))
    }

    #[test]
    fn test_first_update_sets_both_to_same() {
        let mut table = TransformSnapshotTable::new();
        let t = translation(1.0);
        table.update(1, t);
        let s = table.get(1).unwrap();
        assert_eq!(s.prev, t);
        assert_eq!(s.curr, t);
    }

    #[test]
    fn test_second_update_shifts_prev() {
        let mut table = TransformSnapshotTable::new();
        let t0 = translation(0.0);
        let t1 = translation(1.0);
        table.update(1, t0);
        table.update(1, t1);
        let s = table.get(1).unwrap();
        assert_eq!(s.prev, t0);
        assert_eq!(s.curr, t1);
    }

    #[test]
    fn test_interpolated_at_zero_returns_prev() {
        let mut table = TransformSnapshotTable::new();
        table.update(1, translation(0.0));
        table.update(1, translation(4.0));
        let t = table.interpolated(1, 0.0).unwrap();
        assert!((t.translation.x).abs() < 1e-5, "x was {}", t.translation.x);
    }

    #[test]
    fn test_interpolated_at_one_returns_curr() {
        let mut table = TransformSnapshotTable::new();
        table.update(1, translation(0.0));
        table.update(1, translation(4.0));
        let t = table.interpolated(1, 1.0).unwrap();
        assert!((t.translation.x - 4.0).abs() < 1e-5, "x was {}", t.translation.x);
    }

    #[test]
    fn test_interpolated_midpoint() {
        let mut table = TransformSnapshotTable::new();
        table.update(1, translation(0.0));
        table.update(1, translation(4.0));
        let t = table.interpolated(1, 0.5).unwrap();
        assert!((t.translation.x - 2.0).abs() < 1e-5, "x was {}", t.translation.x);
    }

    #[test]
    fn test_missing_node_returns_none() {
        let table = TransformSnapshotTable::new();
        assert!(table.interpolated(999, 0.5).is_none());
        assert!(table.get(999).is_none());
    }

    #[test]
    fn test_remove() {
        let mut table = TransformSnapshotTable::new();
        table.update(1, translation(1.0));
        table.remove(1);
        assert!(table.get(1).is_none());
        assert!(table.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut table = TransformSnapshotTable::new();
        table.update(1, translation(1.0));
        table.update(2, translation(2.0));
        table.clear();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }
}
