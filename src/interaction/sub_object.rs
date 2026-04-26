//! Typed sub-object reference and sub-object selection set.
//!
//! [`SubObjectRef`] is the single canonical way to identify a face, vertex,
//! edge, or point-cloud point relative to its parent object. It is carried
//! inside [`PickHit::sub_object`](crate::interaction::picking::PickHit::sub_object)
//! and used as the key type in [`SubSelection`].
//!
//! [`SubSelection`] is the sub-object counterpart to
//! [`crate::interaction::selection::Selection`]. Typically an app holds both:
//! `Selection` for which objects are selected, `SubSelection` for which faces
//! or points within those objects are selected.

use std::collections::HashSet;

use crate::interaction::selection::NodeId;

// ---------------------------------------------------------------------------
// SubObjectRef
// ---------------------------------------------------------------------------

/// A typed reference to a sub-object within a parent scene object.
///
/// Produced by all pick functions when a specific surface feature is hit, and
/// stored in [`PickHit::sub_object`](crate::interaction::picking::PickHit::sub_object).
///
/// # Variants
///
/// - [`Face`](SubObjectRef::Face) : triangular face, by index in the triangle list.
///   Index `i` addresses vertices `indices[3i..3i+3]`.
/// - [`Vertex`](SubObjectRef::Vertex) : mesh vertex, by position in the vertex buffer.
/// - [`Edge`](SubObjectRef::Edge) : mesh edge (from parry3d `FeatureId::Edge`; rarely
///   produced by TriMesh ray casts in practice).
/// - [`Point`](SubObjectRef::Point) : point in a point-cloud object, by index in the
///   positions slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SubObjectRef {
    /// A triangular face identified by its index in the triangle list.
    Face(u32),
    /// A mesh vertex identified by its position in the vertex buffer.
    Vertex(u32),
    /// A mesh edge identified by its edge index (`parry3d::shape::FeatureId::Edge`).
    ///
    /// Rarely produced in practice; included for completeness.
    Edge(u32),
    /// A point within a point-cloud object, by its index in the positions slice.
    Point(u32),
}

impl SubObjectRef {
    /// Returns `true` if this is a [`Face`](SubObjectRef::Face).
    pub fn is_face(&self) -> bool {
        matches!(self, Self::Face(_))
    }

    /// Returns `true` if this is a [`Point`](SubObjectRef::Point).
    pub fn is_point(&self) -> bool {
        matches!(self, Self::Point(_))
    }

    /// Returns `true` if this is a [`Vertex`](SubObjectRef::Vertex).
    pub fn is_vertex(&self) -> bool {
        matches!(self, Self::Vertex(_))
    }

    /// Returns `true` if this is an [`Edge`](SubObjectRef::Edge).
    pub fn is_edge(&self) -> bool {
        matches!(self, Self::Edge(_))
    }

    /// Returns the raw index regardless of variant.
    pub fn index(&self) -> u32 {
        match *self {
            Self::Face(i) | Self::Vertex(i) | Self::Edge(i) | Self::Point(i) => i,
        }
    }

    /// Convert from a parry3d [`FeatureId`](parry3d::shape::FeatureId).
    ///
    /// Returns `None` for `FeatureId::Unknown` (not expected from TriMesh ray casts).
    pub fn from_feature_id(f: parry3d::shape::FeatureId) -> Option<Self> {
        match f {
            parry3d::shape::FeatureId::Face(i) => Some(Self::Face(i)),
            parry3d::shape::FeatureId::Vertex(i) => Some(Self::Vertex(i)),
            parry3d::shape::FeatureId::Edge(i) => Some(Self::Edge(i)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// SubSelection
// ---------------------------------------------------------------------------

/// A set of selected sub-objects (faces, vertices, edges, or points) across
/// one or more parent objects.
///
/// Parallel to [`crate::interaction::selection::Selection`] but operates at
/// sub-object granularity. Each entry pairs a parent `object_id` with a
/// [`SubObjectRef`]. No ordering is maintained beyond the tracked `primary`.
///
/// # Typical usage
///
/// Hold a `SubSelection` alongside a `Selection`. Use `Selection` to track
/// which objects are selected at object level; use `SubSelection` to track
/// which specific faces or points within those objects are highlighted.
///
/// ```rust,ignore
/// // On rect-pick:
/// let rect_result = pick_rect(...);
/// sub_sel.clear();
/// sub_sel.extend_from_rect_pick(&rect_result);
///
/// // On ray-pick (face highlight):
/// if let Some(sub) = hit.sub_object {
///     sub_sel.select_one(hit.id, sub);
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct SubSelection {
    selected: HashSet<(NodeId, SubObjectRef)>,
    primary: Option<(NodeId, SubObjectRef)>,
    version: u64,
}

impl SubSelection {
    /// Create an empty sub-selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Monotonically increasing generation counter.
    ///
    /// Incremented by `wrapping_add(1)` on every mutation. Compare against a
    /// cached value to detect changes without re-hashing.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Clear and select exactly one sub-object.
    pub fn select_one(&mut self, object_id: NodeId, sub: SubObjectRef) {
        self.selected.clear();
        self.selected.insert((object_id, sub));
        self.primary = Some((object_id, sub));
        self.version = self.version.wrapping_add(1);
    }

    /// Toggle a sub-object in or out of the selection.
    ///
    /// If added, it becomes the primary. If removed, primary is cleared or set
    /// to an arbitrary remaining entry.
    pub fn toggle(&mut self, object_id: NodeId, sub: SubObjectRef) {
        let key = (object_id, sub);
        if self.selected.contains(&key) {
            self.selected.remove(&key);
            if self.primary == Some(key) {
                self.primary = self.selected.iter().next().copied();
            }
        } else {
            self.selected.insert(key);
            self.primary = Some(key);
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Add a sub-object without clearing others.
    pub fn add(&mut self, object_id: NodeId, sub: SubObjectRef) {
        self.selected.insert((object_id, sub));
        self.primary = Some((object_id, sub));
        self.version = self.version.wrapping_add(1);
    }

    /// Remove a sub-object from the selection.
    pub fn remove(&mut self, object_id: NodeId, sub: SubObjectRef) {
        let key = (object_id, sub);
        self.selected.remove(&key);
        if self.primary == Some(key) {
            self.primary = self.selected.iter().next().copied();
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Clear the entire sub-selection.
    pub fn clear(&mut self) {
        self.selected.clear();
        self.primary = None;
        self.version = self.version.wrapping_add(1);
    }

    /// Extend from an iterator of `(object_id, SubObjectRef)` pairs.
    ///
    /// The last pair becomes primary.
    pub fn extend(&mut self, items: impl IntoIterator<Item = (NodeId, SubObjectRef)>) {
        let mut last = None;
        for item in items {
            self.selected.insert(item);
            last = Some(item);
        }
        if let Some(item) = last {
            self.primary = Some(item);
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Populate from a [`RectPickResult`](crate::interaction::picking::RectPickResult).
    ///
    /// Adds all sub-objects from the rect pick without clearing the current
    /// selection. Call [`clear`](Self::clear) first if you want a fresh selection.
    pub fn extend_from_rect_pick(&mut self, result: &crate::interaction::picking::RectPickResult) {
        for (&object_id, subs) in &result.hits {
            for &sub in subs {
                self.selected.insert((object_id, sub));
                self.primary = Some((object_id, sub));
            }
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Whether a specific sub-object is selected.
    pub fn contains(&self, object_id: NodeId, sub: SubObjectRef) -> bool {
        self.selected.contains(&(object_id, sub))
    }

    /// The most recently selected `(object_id, SubObjectRef)` pair.
    pub fn primary(&self) -> Option<(NodeId, SubObjectRef)> {
        self.primary
    }

    /// Iterate over all selected `(object_id, SubObjectRef)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(NodeId, SubObjectRef)> {
        self.selected.iter()
    }

    /// All sub-object refs for a specific parent object.
    pub fn for_object(&self, object_id: NodeId) -> impl Iterator<Item = SubObjectRef> + '_ {
        self.selected
            .iter()
            .filter(move |(id, _)| *id == object_id)
            .map(|(_, sub)| *sub)
    }

    /// Number of selected sub-objects.
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    /// Whether the sub-selection is empty.
    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// Count of selected faces across all objects.
    pub fn face_count(&self) -> usize {
        self.selected.iter().filter(|(_, s)| s.is_face()).count()
    }

    /// Count of selected points across all objects.
    pub fn point_count(&self) -> usize {
        self.selected.iter().filter(|(_, s)| s.is_point()).count()
    }

    /// Count of selected vertices across all objects.
    pub fn vertex_count(&self) -> usize {
        self.selected.iter().filter(|(_, s)| s.is_vertex()).count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::picking::RectPickResult;

    // --- SubObjectRef ---

    #[test]
    fn sub_object_ref_kind_checks() {
        assert!(SubObjectRef::Face(0).is_face());
        assert!(!SubObjectRef::Face(0).is_point());
        assert!(!SubObjectRef::Face(0).is_vertex());
        assert!(!SubObjectRef::Face(0).is_edge());

        assert!(SubObjectRef::Point(1).is_point());
        assert!(SubObjectRef::Vertex(2).is_vertex());
        assert!(SubObjectRef::Edge(3).is_edge());
    }

    #[test]
    fn sub_object_ref_index() {
        assert_eq!(SubObjectRef::Face(7).index(), 7);
        assert_eq!(SubObjectRef::Vertex(42).index(), 42);
        assert_eq!(SubObjectRef::Edge(0).index(), 0);
        assert_eq!(SubObjectRef::Point(99).index(), 99);
    }

    #[test]
    fn sub_object_ref_from_feature_id() {
        use parry3d::shape::FeatureId;
        assert_eq!(
            SubObjectRef::from_feature_id(FeatureId::Face(3)),
            Some(SubObjectRef::Face(3))
        );
        assert_eq!(
            SubObjectRef::from_feature_id(FeatureId::Vertex(1)),
            Some(SubObjectRef::Vertex(1))
        );
        assert_eq!(
            SubObjectRef::from_feature_id(FeatureId::Edge(2)),
            Some(SubObjectRef::Edge(2))
        );
        assert_eq!(SubObjectRef::from_feature_id(FeatureId::Unknown), None);
    }

    #[test]
    fn sub_object_ref_hashable() {
        let mut set = std::collections::HashSet::new();
        set.insert(SubObjectRef::Face(0));
        set.insert(SubObjectRef::Face(0)); // duplicate
        set.insert(SubObjectRef::Face(1));
        set.insert(SubObjectRef::Point(0)); // same index, different variant
        assert_eq!(set.len(), 3);
    }

    // --- SubSelection ---

    #[test]
    fn sub_selection_select_one_clears_others() {
        let mut sel = SubSelection::new();
        sel.add(1, SubObjectRef::Face(0));
        sel.add(1, SubObjectRef::Face(1));
        sel.select_one(1, SubObjectRef::Face(5));
        assert_eq!(sel.len(), 1);
        assert!(sel.contains(1, SubObjectRef::Face(5)));
        assert!(!sel.contains(1, SubObjectRef::Face(0)));
    }

    #[test]
    fn sub_selection_toggle() {
        let mut sel = SubSelection::new();
        sel.toggle(1, SubObjectRef::Face(0));
        assert!(sel.contains(1, SubObjectRef::Face(0)));
        sel.toggle(1, SubObjectRef::Face(0));
        assert!(!sel.contains(1, SubObjectRef::Face(0)));
        assert!(sel.is_empty());
    }

    #[test]
    fn sub_selection_add_preserves_others() {
        let mut sel = SubSelection::new();
        sel.add(1, SubObjectRef::Face(0));
        sel.add(1, SubObjectRef::Face(1));
        assert_eq!(sel.len(), 2);
        assert!(sel.contains(1, SubObjectRef::Face(0)));
        assert!(sel.contains(1, SubObjectRef::Face(1)));
    }

    #[test]
    fn sub_selection_remove() {
        let mut sel = SubSelection::new();
        sel.add(1, SubObjectRef::Face(0));
        sel.add(1, SubObjectRef::Face(1));
        sel.remove(1, SubObjectRef::Face(0));
        assert!(!sel.contains(1, SubObjectRef::Face(0)));
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn sub_selection_clear() {
        let mut sel = SubSelection::new();
        sel.add(1, SubObjectRef::Face(0));
        sel.add(2, SubObjectRef::Point(3));
        sel.clear();
        assert!(sel.is_empty());
        assert_eq!(sel.primary(), None);
    }

    #[test]
    fn sub_selection_primary_tracks_last() {
        let mut sel = SubSelection::new();
        sel.add(1, SubObjectRef::Face(0));
        assert_eq!(sel.primary(), Some((1, SubObjectRef::Face(0))));
        sel.add(2, SubObjectRef::Point(5));
        assert_eq!(sel.primary(), Some((2, SubObjectRef::Point(5))));
    }

    #[test]
    fn sub_selection_contains() {
        let mut sel = SubSelection::new();
        sel.add(10, SubObjectRef::Face(3));
        assert!(sel.contains(10, SubObjectRef::Face(3)));
        assert!(!sel.contains(10, SubObjectRef::Face(4)));
        assert!(!sel.contains(99, SubObjectRef::Face(3)));
    }

    #[test]
    fn sub_selection_for_object() {
        let mut sel = SubSelection::new();
        sel.add(1, SubObjectRef::Face(0));
        sel.add(1, SubObjectRef::Face(1));
        sel.add(2, SubObjectRef::Face(0));
        let obj1: Vec<SubObjectRef> = {
            let mut v: Vec<_> = sel.for_object(1).collect();
            v.sort_by_key(|s| s.index());
            v
        };
        assert_eq!(obj1, vec![SubObjectRef::Face(0), SubObjectRef::Face(1)]);
        let obj2: Vec<SubObjectRef> = sel.for_object(2).collect();
        assert_eq!(obj2, vec![SubObjectRef::Face(0)]);
        assert_eq!(sel.for_object(99).count(), 0);
    }

    #[test]
    fn sub_selection_version_increments() {
        let mut sel = SubSelection::new();
        let v0 = sel.version();
        sel.add(1, SubObjectRef::Face(0));
        assert!(sel.version() > v0);
        let v1 = sel.version();
        sel.clear();
        assert!(sel.version() > v1);
    }

    #[test]
    fn sub_selection_kind_counts() {
        let mut sel = SubSelection::new();
        sel.add(1, SubObjectRef::Face(0));
        sel.add(1, SubObjectRef::Face(1));
        sel.add(2, SubObjectRef::Point(0));
        sel.add(3, SubObjectRef::Vertex(0));
        assert_eq!(sel.face_count(), 2);
        assert_eq!(sel.point_count(), 1);
        assert_eq!(sel.vertex_count(), 1);
    }

    #[test]
    fn sub_selection_extend() {
        let mut sel = SubSelection::new();
        sel.extend([
            (1, SubObjectRef::Face(0)),
            (1, SubObjectRef::Face(1)),
            (2, SubObjectRef::Point(3)),
        ]);
        assert_eq!(sel.len(), 3);
        assert_eq!(sel.primary(), Some((2, SubObjectRef::Point(3))));
    }

    #[test]
    fn sub_selection_extend_from_rect_pick() {
        let mut result = RectPickResult::default();
        result
            .hits
            .insert(10, vec![SubObjectRef::Face(0), SubObjectRef::Face(1)]);
        result.hits.insert(20, vec![SubObjectRef::Point(5)]);

        let mut sel = SubSelection::new();
        sel.extend_from_rect_pick(&result);

        assert_eq!(sel.len(), 3);
        assert!(sel.contains(10, SubObjectRef::Face(0)));
        assert!(sel.contains(10, SubObjectRef::Face(1)));
        assert!(sel.contains(20, SubObjectRef::Point(5)));
        assert_eq!(sel.face_count(), 2);
        assert_eq!(sel.point_count(), 1);
    }
}
