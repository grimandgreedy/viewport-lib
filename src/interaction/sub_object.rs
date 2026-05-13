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
/// - [`Voxel`](SubObjectRef::Voxel) : voxel in a structured scalar volume.
/// - [`Cell`](SubObjectRef::Cell) : cell in an unstructured volume mesh (`VolumeMeshData`).
/// - [`Splat`](SubObjectRef::Splat) : gaussian splat, by index in the splat buffer.
/// - [`Instance`](SubObjectRef::Instance) : glyph, tensor glyph, or sprite instance,
///   by instance index.
/// - [`Segment`](SubObjectRef::Segment) : polyline, tube, or ribbon segment, by index.
/// - [`Strip`](SubObjectRef::Strip) : connected curve strip within a multi-strip item,
///   by strip index.
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
    /// A voxel within a ray-marched volume, by its flat grid index.
    ///
    /// The flat index encodes `(ix, iy, iz)` as `ix + iy * nx + iz * nx * ny`.
    /// Recover the 3-D indices using the grid dimensions from
    /// [`VolumeData`](crate::geometry::marching_cubes::VolumeData).
    Voxel(u32),
    /// A cell within an unstructured volume mesh, by its index in
    /// [`VolumeMeshData::cells`](crate::resources::volume_mesh::VolumeMeshData::cells).
    ///
    /// Produced by [`pick_transparent_volume_mesh_cpu`](crate::interaction::picking::pick_transparent_volume_mesh_cpu)
    /// and [`pick_transparent_volume_mesh_rect`](crate::interaction::picking::pick_transparent_volume_mesh_rect).
    Cell(u32),
    /// A gaussian splat identified by its index in the splat buffer.
    Splat(u32),
    /// A glyph, tensor glyph, or sprite instance identified by its instance index.
    Instance(u32),
    /// A polyline, tube, or ribbon segment identified by its segment index.
    Segment(u32),
    /// A connected curve strip within a multi-strip item, identified by its strip index.
    Strip(u32),
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

    /// Returns `true` if this is a [`Voxel`](SubObjectRef::Voxel).
    pub fn is_voxel(&self) -> bool {
        matches!(self, Self::Voxel(_))
    }

    /// Returns `true` if this is a [`Cell`](SubObjectRef::Cell).
    pub fn is_cell(&self) -> bool {
        matches!(self, Self::Cell(_))
    }

    /// Returns the raw index regardless of variant.
    pub fn index(&self) -> u32 {
        match *self {
            Self::Face(i) | Self::Vertex(i) | Self::Edge(i) | Self::Point(i)
            | Self::Voxel(i) | Self::Cell(i) | Self::Splat(i) | Self::Instance(i)
            | Self::Segment(i) | Self::Strip(i) => i,
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

    /// Count of selected voxels across all objects.
    pub fn voxel_count(&self) -> usize {
        self.selected.iter().filter(|(_, s)| s.is_voxel()).count()
    }

    /// Count of selected cells across all objects.
    pub fn cell_count(&self) -> usize {
        self.selected.iter().filter(|(_, s)| s.is_cell()).count()
    }
}

// ---------------------------------------------------------------------------
// SubSelectionRef
// ---------------------------------------------------------------------------

/// Geometry info needed to decode a [`SubObjectRef::Voxel`] flat index into
/// world-space AABB corners for highlight rendering.
///
/// Pass one entry per volume object via [`SubSelectionRef::with_voxels`].
pub struct VolumeSelectionInfo {
    /// Grid dimensions `[nx, ny, nz]` — same as [`VolumeData::dims`].
    pub dims: [u32; 3],
    /// Local-space bounding-box minimum corner (matches [`VolumeItem::bbox_min`]).
    pub bbox_min: [f32; 3],
    /// Local-space bounding-box maximum corner (matches [`VolumeItem::bbox_max`]).
    pub bbox_max: [f32; 3],
    /// World-space transform (matches [`VolumeItem::model`]).
    pub model: [[f32; 4]; 4],
}

/// Geometry info needed to highlight [`SubObjectRef::Point`], [`SubObjectRef::Segment`],
/// and [`SubObjectRef::Strip`] selections on a polyline item.
///
/// Pass one entry per polyline object via [`SubSelectionRef::with_polylines`].
pub struct PolylineSelectionInfo {
    /// World-space vertex positions. Each entry is one polyline node.
    pub positions: Vec<[f32; 3]>,
    /// Strip lengths. Same encoding as [`PolylineItem::strip_lengths`](crate::renderer::types::items::PolylineItem::strip_lengths):
    /// each entry is the number of nodes in that strip. If empty, all positions
    /// belong to a single strip.
    pub strip_lengths: Vec<u32>,
}

/// Geometry info needed to highlight a [`SubObjectRef::Cell`] selection.
///
/// Contains the vertex positions and cell connectivity from the host's
/// [`VolumeMeshData`](crate::resources::volume_mesh::VolumeMeshData). Pass one
/// entry per volume mesh object via [`SubSelectionRef::with_cells`].
pub struct CellSelectionInfo {
    /// World-space vertex positions. Indexed by cell connectivity entries.
    pub positions: Vec<[f32; 3]>,
    /// Cell connectivity. Each entry is `[u32; 8]` with
    /// `u32::MAX` padding for cells with fewer than 8 vertices (same encoding
    /// as [`VolumeMeshData::cells`](crate::resources::volume_mesh::VolumeMeshData::cells)).
    pub cells: Vec<[u32; 8]>,
}

/// A renderer-owned snapshot of a [`SubSelection`] taken at frame submission time.
///
/// Bundles the selection items with the CPU-side mesh and point cloud data the
/// renderer needs to build highlight geometry. The renderer does not hold a
/// reference to any app-owned data between frames.
///
/// # Usage
///
/// ```ignore
/// fd.interaction.sub_selection = Some(SubSelectionRef::new(
///     &self.sub_selection,
///     mesh_lookup,
///     model_matrices,
///     point_positions,
/// ));
/// ```
pub struct SubSelectionRef {
    /// Snapshot of all selected (node_id, sub_object) pairs.
    pub(crate) items: Vec<(NodeId, SubObjectRef)>,
    /// CPU-side vertex positions and triangle indices keyed by node id.
    ///
    /// Same format as the `mesh_lookup` parameter to
    /// [`pick_scene_cpu`](crate::interaction::picking::pick_scene_cpu):
    /// the value is `(positions, indices)` where every three consecutive
    /// indices form one triangle.
    pub(crate) mesh_lookup:
        std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
    /// World-space model matrix for each node, keyed by node id.
    ///
    /// Used to transform local-space mesh positions into world space when
    /// building fill and edge geometry. Nodes absent from the map are treated
    /// as having an identity transform.
    pub(crate) model_matrices: std::collections::HashMap<u64, glam::Mat4>,
    /// World-space point cloud positions keyed by node id.
    ///
    /// Required for [`SubObjectRef::Point`] highlights. The index carried by
    /// `Point(i)` addresses `point_positions[node_id][i]`.
    pub(crate) point_positions: std::collections::HashMap<u64, Vec<[f32; 3]>>,
    /// Volume geometry info keyed by node id.
    ///
    /// Required for [`SubObjectRef::Voxel`] highlights. Each entry provides the
    /// grid dimensions and bounding box so the renderer can decode flat voxel
    /// indices into world-space AABB wireframes.
    pub(crate) voxel_lookup: std::collections::HashMap<u64, VolumeSelectionInfo>,
    /// Unstructured volume mesh geometry keyed by node id.
    ///
    /// Required for [`SubObjectRef::Cell`] highlights. Each entry provides the
    /// vertex positions and cell connectivity so the renderer can draw edge
    /// outlines around selected cells.
    pub(crate) cell_lookup: std::collections::HashMap<u64, CellSelectionInfo>,
    /// Polyline geometry keyed by node id.
    ///
    /// Required for [`SubObjectRef::Point`], [`SubObjectRef::Segment`], and
    /// [`SubObjectRef::Strip`] highlights on polyline items. Each entry provides
    /// the positions and strip lengths so the renderer can draw node sprites and
    /// segment edge lines.
    pub(crate) polyline_lookup: std::collections::HashMap<u64, PolylineSelectionInfo>,
    /// Version counter copied from the source [`SubSelection::version()`].
    ///
    /// The renderer uses this to skip GPU buffer rebuilds when the selection
    /// has not changed since the previous frame.
    pub version: u64,
}

impl SubSelectionRef {
    /// Create a snapshot from a live [`SubSelection`].
    ///
    /// - `mesh_lookup` : CPU positions + indices per node id (same type as the
    ///   `mesh_lookup` argument to the CPU pick functions).
    /// - `model_matrices` : world transform per node id.
    /// - `point_positions` : point cloud positions per node id (for
    ///   [`SubObjectRef::Point`] entries).
    pub fn new(
        sub_selection: &SubSelection,
        mesh_lookup: std::collections::HashMap<u64, (Vec<[f32; 3]>, Vec<u32>)>,
        model_matrices: std::collections::HashMap<u64, glam::Mat4>,
        point_positions: std::collections::HashMap<u64, Vec<[f32; 3]>>,
    ) -> Self {
        Self {
            items: sub_selection
                .iter()
                .map(|(n, s)| (*n, *s))
                .collect(),
            mesh_lookup,
            model_matrices,
            point_positions,
            voxel_lookup: std::collections::HashMap::new(),
            cell_lookup: std::collections::HashMap::new(),
            polyline_lookup: std::collections::HashMap::new(),
            version: sub_selection.version(),
        }
    }

    /// Attach volume geometry info for [`SubObjectRef::Voxel`] highlight rendering.
    ///
    /// `lookup` maps each volume's node id to its [`VolumeSelectionInfo`]. Without
    /// this, selected voxels are silently skipped during highlight geometry build.
    pub fn with_voxels(
        mut self,
        lookup: std::collections::HashMap<u64, VolumeSelectionInfo>,
    ) -> Self {
        self.voxel_lookup = lookup;
        self
    }

    /// Attach unstructured volume mesh geometry for [`SubObjectRef::Cell`] highlight rendering.
    ///
    /// `lookup` maps each volume mesh's node id to its [`CellSelectionInfo`]. Without
    /// this, selected cells are silently skipped during highlight geometry build.
    pub fn with_cells(
        mut self,
        lookup: std::collections::HashMap<u64, CellSelectionInfo>,
    ) -> Self {
        self.cell_lookup = lookup;
        self
    }

    /// Attach polyline geometry for [`SubObjectRef::Point`], [`SubObjectRef::Segment`],
    /// and [`SubObjectRef::Strip`] highlight rendering.
    ///
    /// `lookup` maps each polyline item's node id to its [`PolylineSelectionInfo`].
    /// Without this, selected polyline nodes and segments are silently skipped during
    /// highlight geometry build.
    pub fn with_polylines(
        mut self,
        lookup: std::collections::HashMap<u64, PolylineSelectionInfo>,
    ) -> Self {
        self.polyline_lookup = lookup;
        self
    }

    /// Returns `true` if the snapshot contains no selected sub-objects.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
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
