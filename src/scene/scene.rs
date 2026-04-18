//! Scene graph with parent-child hierarchy, layers, and dirty-flag transform propagation.
//!
//! `Scene` is a standalone struct that apps own alongside `ViewportRenderer`.
//! It produces `Vec<SceneRenderItem>` via `collect_render_items()`, which feeds
//! into `SceneFrame::surfaces` (usually via `SceneFrame::from_surface_items(...)`).
//! The renderer itself remains stateless.

use std::collections::{HashMap, HashSet};

use crate::interaction::selection::{NodeId, Selection};
use crate::renderer::SceneRenderItem;
use crate::resources::mesh_store::MeshId;
use crate::scene::material::Material;
use crate::scene::traits::ViewportObject;

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// Opaque layer identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerId(pub u32);

/// A named visibility layer. Nodes belong to exactly one layer.
pub struct Layer {
    /// Unique layer identifier.
    pub id: LayerId,
    /// Human-readable layer name.
    pub name: String,
    /// Whether nodes on this layer are rendered.
    pub visible: bool,
    /// When true, nodes on this layer render but cannot appear selected.
    pub locked: bool,
    /// Display color for this layer (RGBA, each component 0.0–1.0).
    pub color: [f32; 4],
    /// Sort order for layer display. Lower values appear first.
    pub order: u32,
}

// ---------------------------------------------------------------------------
// Group
// ---------------------------------------------------------------------------

/// Opaque group identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(pub u32);

/// A named group of scene nodes. Membership is independent of parent-child hierarchy.
pub struct Group {
    /// Unique group identifier.
    pub id: GroupId,
    /// Human-readable group name.
    pub name: String,
    /// Set of node IDs belonging to this group.
    pub members: HashSet<NodeId>,
}

// ---------------------------------------------------------------------------
// SceneNode
// ---------------------------------------------------------------------------

/// A node in the scene graph.
pub struct SceneNode {
    id: NodeId,
    name: String,
    mesh_id: Option<MeshId>,
    material: Material,
    visible: bool,
    show_normals: bool,
    local_transform: glam::Mat4,
    world_transform: glam::Mat4,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    layer: LayerId,
    dirty: bool,
}

impl SceneNode {
    /// Unique identifier for this node.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Display name of this node.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// GPU mesh associated with this node, or `None` if no mesh has been uploaded.
    pub fn mesh_id(&self) -> Option<MeshId> {
        self.mesh_id
    }

    /// Material parameters (color, shading, opacity, texture) for this node.
    pub fn material(&self) -> &Material {
        &self.material
    }

    /// Whether this node is visible.
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Whether per-vertex normals are rendered for this node.
    pub fn show_normals(&self) -> bool {
        self.show_normals
    }

    /// Local transform relative to this node's parent (or world if no parent).
    pub fn local_transform(&self) -> glam::Mat4 {
        self.local_transform
    }

    /// World-space transform. Updated by `Scene::update_transforms()`.
    pub fn world_transform(&self) -> glam::Mat4 {
        self.world_transform
    }

    /// Parent node ID, or `None` if this is a root node.
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// IDs of this node's direct children.
    pub fn children(&self) -> &[NodeId] {
        &self.children
    }

    /// Layer this node belongs to.
    pub fn layer(&self) -> LayerId {
        self.layer
    }
}

impl ViewportObject for SceneNode {
    fn id(&self) -> u64 {
        self.id
    }

    fn mesh_id(&self) -> Option<u64> {
        self.mesh_id.map(|m| m.index() as u64)
    }

    fn model_matrix(&self) -> glam::Mat4 {
        self.world_transform
    }

    fn position(&self) -> glam::Vec3 {
        self.world_transform.col(3).truncate()
    }

    fn rotation(&self) -> glam::Quat {
        let (_scale, rotation, _translation) = self.world_transform.to_scale_rotation_translation();
        rotation
    }

    fn scale(&self) -> glam::Vec3 {
        let (scale, _rotation, _translation) = self.world_transform.to_scale_rotation_translation();
        scale
    }

    fn is_visible(&self) -> bool {
        self.visible
    }

    fn color(&self) -> glam::Vec3 {
        glam::Vec3::from(self.material.base_color)
    }

    fn show_normals(&self) -> bool {
        self.show_normals
    }

    fn material(&self) -> Material {
        self.material
    }
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// Default layer ID (always exists, cannot be removed).
const DEFAULT_LAYER: LayerId = LayerId(0);

/// A scene graph managing nodes with parent-child hierarchy and layers.
pub struct Scene {
    nodes: HashMap<NodeId, SceneNode>,
    roots: Vec<NodeId>,
    layers: Vec<Layer>,
    next_id: u64,
    next_layer_id: u32,
    groups: Vec<Group>,
    next_group_id: u32,
    /// Monotonically increasing generation counter. Incremented on every mutation.
    /// Callers can compare against a cached value to detect changes without hashing.
    version: u64,
}

impl Scene {
    /// Create an empty scene with a default layer.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            layers: vec![Layer {
                id: DEFAULT_LAYER,
                name: "Default".to_string(),
                visible: true,
                locked: false,
                color: [1.0, 1.0, 1.0, 1.0],
                order: 0,
            }],
            next_id: 1,
            next_layer_id: 1,
            groups: Vec::new(),
            next_group_id: 0,
            version: 0,
        }
    }

    /// Monotonically increasing generation counter.
    ///
    /// Incremented by `wrapping_add(1)` on every mutation. Compare against a
    /// cached value to detect scene changes without hashing instance data.
    pub fn version(&self) -> u64 {
        self.version
    }

    // -- Node lifecycle --

    /// Add a node with a mesh, transform, and material. Returns the new node's ID.
    pub fn add(
        &mut self,
        mesh_id: Option<MeshId>,
        transform: glam::Mat4,
        material: Material,
    ) -> NodeId {
        self.add_named("", mesh_id, transform, material)
    }

    /// Add a named node. Returns the new node's ID.
    pub fn add_named(
        &mut self,
        name: &str,
        mesh_id: Option<MeshId>,
        transform: glam::Mat4,
        material: Material,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        let node = SceneNode {
            id,
            name: name.to_string(),
            mesh_id,
            material,
            visible: true,
            show_normals: false,
            local_transform: transform,
            world_transform: transform,
            parent: None,
            children: Vec::new(),
            layer: DEFAULT_LAYER,
            dirty: true,
        };
        self.nodes.insert(id, node);
        self.roots.push(id);
        self.version = self.version.wrapping_add(1);
        id
    }

    /// Remove a node and all its descendants. Returns all removed IDs
    /// (caller can use these to release mesh references).
    pub fn remove(&mut self, id: NodeId) -> Vec<NodeId> {
        let mut removed = Vec::new();
        self.remove_recursive(id, &mut removed);

        // Remove from parent's children list or from roots.
        if let Some(parent_id) = self.nodes.get(&id).and_then(|n| n.parent) {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.children.retain(|c| *c != id);
            }
        } else {
            self.roots.retain(|r| *r != id);
        }

        // Actually remove nodes.
        for &rid in &removed {
            self.nodes.remove(&rid);
        }
        // Also remove from roots any descendant that might have been listed.
        self.roots.retain(|r| !removed.contains(r));

        // Remove removed nodes from all groups.
        for group in &mut self.groups {
            for &rid in &removed {
                group.members.remove(&rid);
            }
        }

        self.version = self.version.wrapping_add(1);
        removed
    }

    fn remove_recursive(&self, id: NodeId, out: &mut Vec<NodeId>) {
        out.push(id);
        if let Some(node) = self.nodes.get(&id) {
            for &child in &node.children {
                self.remove_recursive(child, out);
            }
        }
    }

    // -- Hierarchy --

    /// Reparent a node. `None` makes it a root node.
    pub fn set_parent(&mut self, child_id: NodeId, new_parent: Option<NodeId>) {
        // Remove from current parent or roots.
        let old_parent = self.nodes.get(&child_id).and_then(|n| n.parent);
        if let Some(old_pid) = old_parent {
            if let Some(old_p) = self.nodes.get_mut(&old_pid) {
                old_p.children.retain(|c| *c != child_id);
            }
        } else {
            self.roots.retain(|r| *r != child_id);
        }

        // Add to new parent or roots.
        if let Some(new_pid) = new_parent {
            if let Some(new_p) = self.nodes.get_mut(&new_pid) {
                new_p.children.push(child_id);
            }
        } else {
            self.roots.push(child_id);
        }

        if let Some(node) = self.nodes.get_mut(&child_id) {
            node.parent = new_parent;
            node.dirty = true;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Children of a node.
    pub fn children(&self, id: NodeId) -> &[NodeId] {
        self.nodes
            .get(&id)
            .map(|n| n.children.as_slice())
            .unwrap_or(&[])
    }

    /// Parent of a node.
    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.nodes.get(&id).and_then(|n| n.parent)
    }

    /// Root nodes.
    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    // -- Properties --

    /// Set the local transform of a node, marking it and its descendants dirty.
    pub fn set_local_transform(&mut self, id: NodeId, transform: glam::Mat4) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.local_transform = transform;
            node.dirty = true;
        }
        self.mark_descendants_dirty(id);
        self.version = self.version.wrapping_add(1);
    }

    /// Set node visibility.
    pub fn set_visible(&mut self, id: NodeId, visible: bool) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.visible = visible;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Set node material.
    pub fn set_material(&mut self, id: NodeId, material: Material) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.material = material;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Set node mesh.
    pub fn set_mesh(&mut self, id: NodeId, mesh_id: Option<MeshId>) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.mesh_id = mesh_id;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Set node name.
    pub fn set_name(&mut self, id: NodeId, name: &str) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.name = name.to_string();
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Set whether to show normals.
    pub fn set_show_normals(&mut self, id: NodeId, show: bool) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.show_normals = show;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Set the layer of a node.
    pub fn set_layer(&mut self, id: NodeId, layer: LayerId) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.layer = layer;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Get a reference to a node.
    pub fn node(&self, id: NodeId) -> Option<&SceneNode> {
        self.nodes.get(&id)
    }

    /// Number of nodes in the scene.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Iterate over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &SceneNode> {
        self.nodes.values()
    }

    // -- Layers --

    /// Add a new layer, returning its ID.
    pub fn add_layer(&mut self, name: &str) -> LayerId {
        let id = LayerId(self.next_layer_id);
        let order = self.next_layer_id;
        self.next_layer_id += 1;
        self.layers.push(Layer {
            id,
            name: name.to_string(),
            visible: true,
            locked: false,
            color: [1.0, 1.0, 1.0, 1.0],
            order,
        });
        self.version = self.version.wrapping_add(1);
        id
    }

    /// Remove a layer, moving all its nodes to the default layer.
    /// Cannot remove the default layer (LayerId(0)).
    pub fn remove_layer(&mut self, id: LayerId) {
        if id == DEFAULT_LAYER {
            return;
        }
        // Move nodes to default layer.
        for node in self.nodes.values_mut() {
            if node.layer == id {
                node.layer = DEFAULT_LAYER;
            }
        }
        self.layers.retain(|l| l.id != id);
        self.version = self.version.wrapping_add(1);
    }

    /// Set layer visibility.
    pub fn set_layer_visible(&mut self, id: LayerId, visible: bool) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.visible = visible;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Set layer locked state. Locked layers render but nodes cannot appear selected.
    pub fn set_layer_locked(&mut self, id: LayerId, locked: bool) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.locked = locked;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Set layer display color.
    pub fn set_layer_color(&mut self, id: LayerId, color: [f32; 4]) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.color = color;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Set layer sort order.
    pub fn set_layer_order(&mut self, id: LayerId, order: u32) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.order = order;
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Whether a layer is currently locked.
    pub fn is_layer_locked(&self, id: LayerId) -> bool {
        self.layers
            .iter()
            .find(|l| l.id == id)
            .map(|l| l.locked)
            .unwrap_or(false)
    }

    /// All layers, sorted by their `order` field (ascending).
    pub fn layers(&self) -> Vec<&Layer> {
        let mut sorted: Vec<&Layer> = self.layers.iter().collect();
        sorted.sort_by_key(|l| l.order);
        sorted
    }

    // -- Groups --

    /// Create a new named group, returning its ID.
    pub fn create_group(&mut self, name: &str) -> GroupId {
        let id = GroupId(self.next_group_id);
        self.next_group_id += 1;
        self.groups.push(Group {
            id,
            name: name.to_string(),
            members: HashSet::new(),
        });
        self.version = self.version.wrapping_add(1);
        id
    }

    /// Remove a group by ID. Does not affect its member nodes.
    pub fn remove_group(&mut self, id: GroupId) {
        self.groups.retain(|g| g.id != id);
        self.version = self.version.wrapping_add(1);
    }

    /// Add a node to a group.
    pub fn add_to_group(&mut self, node: NodeId, group: GroupId) {
        if let Some(g) = self.groups.iter_mut().find(|g| g.id == group) {
            g.members.insert(node);
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Remove a node from a group.
    pub fn remove_from_group(&mut self, node: NodeId, group: GroupId) {
        if let Some(g) = self.groups.iter_mut().find(|g| g.id == group) {
            g.members.remove(&node);
        }
        self.version = self.version.wrapping_add(1);
    }

    /// Get a group by ID.
    pub fn get_group(&self, id: GroupId) -> Option<&Group> {
        self.groups.iter().find(|g| g.id == id)
    }

    /// All groups in the scene.
    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    /// Which groups contain the given node.
    pub fn node_groups(&self, node: NodeId) -> Vec<GroupId> {
        self.groups
            .iter()
            .filter(|g| g.members.contains(&node))
            .map(|g| g.id)
            .collect()
    }

    // -- Transform propagation --

    /// Recompute world transforms for all dirty nodes (BFS from roots).
    pub fn update_transforms(&mut self) {
        // We need to iterate roots and process the tree. Since we can't borrow
        // self mutably while iterating, collect the root list first.
        let roots: Vec<NodeId> = self.roots.clone();
        for &root_id in &roots {
            self.propagate_transform(root_id, glam::Mat4::IDENTITY);
        }
    }

    fn propagate_transform(&mut self, id: NodeId, parent_world: glam::Mat4) {
        let (dirty, local, children) = {
            let Some(node) = self.nodes.get(&id) else {
                return;
            };
            (node.dirty, node.local_transform, node.children.clone())
        };

        if dirty {
            let world = parent_world * local;
            let node = self.nodes.get_mut(&id).unwrap();
            node.world_transform = world;
            node.dirty = false;
            // All children must recompute.
            for &child_id in &children {
                self.mark_dirty(child_id);
                self.propagate_transform(child_id, world);
            }
        } else {
            let world = self.nodes[&id].world_transform;
            for &child_id in &children {
                self.propagate_transform(child_id, world);
            }
        }
    }

    fn mark_dirty(&mut self, id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&id) {
            node.dirty = true;
        }
    }

    fn mark_descendants_dirty(&mut self, id: NodeId) {
        let children = self
            .nodes
            .get(&id)
            .map(|n| n.children.clone())
            .unwrap_or_default();
        for child_id in children {
            self.mark_dirty(child_id);
            self.mark_descendants_dirty(child_id);
        }
    }

    // -- Render collection --

    /// Update transforms and collect render items for all visible nodes.
    ///
    /// Skips nodes that are invisible, on an invisible layer, or have no mesh.
    /// Marks selected nodes based on the provided `Selection`.
    pub fn collect_render_items(&mut self, selection: &Selection) -> Vec<SceneRenderItem> {
        self.update_transforms();

        let layer_visible: HashMap<LayerId, bool> =
            self.layers.iter().map(|l| (l.id, l.visible)).collect();

        let layer_locked: HashMap<LayerId, bool> =
            self.layers.iter().map(|l| (l.id, l.locked)).collect();

        let mut items = Vec::new();
        for node in self.nodes.values() {
            if !node.visible {
                continue;
            }
            if !layer_visible.get(&node.layer).copied().unwrap_or(true) {
                continue;
            }
            let Some(mesh_id) = node.mesh_id else {
                continue;
            };
            let locked = layer_locked.get(&node.layer).copied().unwrap_or(false);
            items.push(SceneRenderItem {
                mesh_index: mesh_id.index(),
                model: node.world_transform.to_cols_array_2d(),
                selected: if locked {
                    false
                } else {
                    selection.contains(node.id)
                },
                visible: true,
                show_normals: node.show_normals,
                material: node.material,
                active_attribute: None,
                scalar_range: None,
                colormap_id: None,
                nan_color: None,
                two_sided: false,
                pick_id: node.id,
            });
        }
        items
    }

    /// Update transforms and collect render items, culling objects outside the frustum.
    ///
    /// Like `collect_render_items`, but skips objects whose world-space AABB is
    /// entirely outside the given frustum. `mesh_aabb_fn` should return the
    /// local-space AABB for a given `MeshId` (typically read from `GpuMesh::aabb`).
    pub fn collect_render_items_culled(
        &mut self,
        selection: &Selection,
        frustum: &crate::camera::frustum::Frustum,
        mesh_aabb_fn: impl Fn(MeshId) -> Option<crate::scene::aabb::Aabb>,
    ) -> (Vec<SceneRenderItem>, crate::camera::frustum::CullStats) {
        self.update_transforms();

        let layer_visible: HashMap<LayerId, bool> =
            self.layers.iter().map(|l| (l.id, l.visible)).collect();

        let layer_locked: HashMap<LayerId, bool> =
            self.layers.iter().map(|l| (l.id, l.locked)).collect();

        let mut items = Vec::new();
        let mut stats = crate::camera::frustum::CullStats::default();

        for node in self.nodes.values() {
            if !node.visible {
                continue;
            }
            if !layer_visible.get(&node.layer).copied().unwrap_or(true) {
                continue;
            }
            let Some(mesh_id) = node.mesh_id else {
                continue;
            };

            stats.total += 1;

            // Frustum cull using world-space AABB.
            if let Some(local_aabb) = mesh_aabb_fn(mesh_id) {
                let world_aabb = local_aabb.transformed(&node.world_transform);
                if frustum.cull_aabb(&world_aabb) {
                    stats.culled += 1;
                    continue;
                }
            }

            let locked = layer_locked.get(&node.layer).copied().unwrap_or(false);
            stats.visible += 1;
            items.push(SceneRenderItem {
                mesh_index: mesh_id.index(),
                model: node.world_transform.to_cols_array_2d(),
                selected: if locked {
                    false
                } else {
                    selection.contains(node.id)
                },
                visible: true,
                show_normals: node.show_normals,
                material: node.material,
                active_attribute: None,
                scalar_range: None,
                colormap_id: None,
                nan_color: None,
                two_sided: false,
                pick_id: node.id,
            });
        }
        (items, stats)
    }

    // -- Tree walking --

    // -- Mesh ref counting --

    /// Count how many scene nodes reference the given mesh.
    ///
    /// O(n) over all nodes. Useful for deciding when to free a GPU mesh.
    pub fn mesh_ref_count(&self, mesh_id: MeshId) -> usize {
        self.nodes
            .values()
            .filter(|n| n.mesh_id == Some(mesh_id))
            .count()
    }

    // -- Tree walking --

    /// Depth-first traversal of the scene tree. Returns `(NodeId, depth)` pairs.
    pub fn walk_depth_first(&self) -> Vec<(NodeId, usize)> {
        let mut result = Vec::new();
        for &root_id in &self.roots {
            self.walk_recursive(root_id, 0, &mut result);
        }
        result
    }

    fn walk_recursive(&self, id: NodeId, depth: usize, out: &mut Vec<(NodeId, usize)>) {
        out.push((id, depth));
        if let Some(node) = self.nodes.get(&id) {
            for &child_id in &node.children {
                self.walk_recursive(child_id, depth + 1, out);
            }
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_remove() {
        let mut scene = Scene::new();
        let id = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        assert!(scene.node(id).is_some());
        assert_eq!(scene.node_count(), 1);

        let removed = scene.remove(id);
        assert_eq!(removed, vec![id]);
        assert!(scene.node(id).is_none());
        assert_eq!(scene.node_count(), 0);
    }

    #[test]
    fn test_remove_cascades_to_children() {
        let mut scene = Scene::new();
        let parent = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        let child1 = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        let child2 = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        scene.set_parent(child1, Some(parent));
        scene.set_parent(child2, Some(parent));

        let removed = scene.remove(parent);
        assert_eq!(removed.len(), 3);
        assert!(removed.contains(&parent));
        assert!(removed.contains(&child1));
        assert!(removed.contains(&child2));
        assert_eq!(scene.node_count(), 0);
    }

    #[test]
    fn test_set_parent_updates_world_transform() {
        let mut scene = Scene::new();
        let parent = scene.add(
            None,
            glam::Mat4::from_translation(glam::Vec3::new(5.0, 0.0, 0.0)),
            Material::default(),
        );
        let child = scene.add(
            None,
            glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0)),
            Material::default(),
        );
        scene.set_parent(child, Some(parent));
        scene.update_transforms();

        let world = scene.node(child).unwrap().world_transform();
        let pos = world.col(3).truncate();
        assert!((pos.x - 6.0).abs() < 1e-5, "expected x=6.0, got {}", pos.x);
    }

    #[test]
    fn test_dirty_propagation() {
        let mut scene = Scene::new();
        let parent = scene.add(
            None,
            glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0)),
            Material::default(),
        );
        let child = scene.add(
            None,
            glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.0, 0.0)),
            Material::default(),
        );
        scene.set_parent(child, Some(parent));
        scene.update_transforms();

        // Now move the parent.
        scene.set_local_transform(
            parent,
            glam::Mat4::from_translation(glam::Vec3::new(10.0, 0.0, 0.0)),
        );
        scene.update_transforms();

        let child_pos = scene
            .node(child)
            .unwrap()
            .world_transform()
            .col(3)
            .truncate();
        assert!(
            (child_pos.x - 12.0).abs() < 1e-5,
            "expected x=12.0, got {}",
            child_pos.x
        );
    }

    #[test]
    fn test_layer_visibility_hides_nodes() {
        let mut scene = Scene::new();
        let layer = scene.add_layer("Hidden");
        let id = scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.set_layer(id, layer);
        scene.set_layer_visible(layer, false);

        let items = scene.collect_render_items(&Selection::new());
        assert!(items.is_empty());
    }

    #[test]
    fn test_collect_skips_invisible_nodes() {
        let mut scene = Scene::new();
        let id = scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.set_visible(id, false);

        let items = scene.collect_render_items(&Selection::new());
        assert!(items.is_empty());
    }

    #[test]
    fn test_collect_skips_meshless_nodes() {
        let mut scene = Scene::new();
        scene.add(None, glam::Mat4::IDENTITY, Material::default());

        let items = scene.collect_render_items(&Selection::new());
        assert!(items.is_empty());
    }

    #[test]
    fn test_collect_marks_selected() {
        let mut scene = Scene::new();
        let id = scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());

        let mut sel = Selection::new();
        sel.select_one(id);

        let items = scene.collect_render_items(&sel);
        assert_eq!(items.len(), 1);
        assert!(items[0].selected);
    }

    #[test]
    fn test_unparent_makes_root() {
        let mut scene = Scene::new();
        let parent = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        let child = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        scene.set_parent(child, Some(parent));
        assert!(!scene.roots().contains(&child));

        scene.set_parent(child, None);
        assert!(scene.roots().contains(&child));
        assert!(scene.node(child).unwrap().parent().is_none());
    }

    #[test]
    fn test_collect_culled_filters_offscreen() {
        let mut scene = Scene::new();
        // Object at origin — should be visible.
        let visible_id = scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        // Object far behind camera — should be culled.
        let _behind = scene.add(
            Some(MeshId(1)),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 100.0)),
            Material::default(),
        );

        let sel = Selection::new();
        // Camera at z=5 looking toward origin.
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 0.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 50.0);
        let frustum = crate::camera::frustum::Frustum::from_view_proj(&(proj * view));

        // Both meshes get a unit-cube AABB.
        let unit_aabb = crate::scene::aabb::Aabb {
            min: glam::Vec3::splat(-0.5),
            max: glam::Vec3::splat(0.5),
        };

        let (items, stats) =
            scene.collect_render_items_culled(&sel, &frustum, |_mesh_id| Some(unit_aabb));

        assert_eq!(stats.total, 2);
        assert_eq!(stats.visible, 1);
        assert_eq!(stats.culled, 1);
        assert_eq!(items.len(), 1);
        // The visible item should be the one at the origin (mesh_index 0).
        assert_eq!(items[0].mesh_index, visible_id as usize - 1); // MeshId(0).index() == 0
        let _ = visible_id; // suppress unused warning
    }

    // --- Layer lock/color/order tests ---

    #[test]
    fn test_layer_locked_field_default_false() {
        let scene = Scene::new();
        let layers = scene.layers();
        let default_layer = layers.iter().find(|l| l.id == LayerId(0)).unwrap();
        assert!(!default_layer.locked);
    }

    #[test]
    fn test_add_layer_has_locked_false_color_white_and_order() {
        let mut scene = Scene::new();
        let layer_id = scene.add_layer("Test");
        let layers = scene.layers();
        let layer = layers.iter().find(|l| l.id == layer_id).unwrap();
        assert!(!layer.locked);
        assert_eq!(layer.color, [1.0, 1.0, 1.0, 1.0]);
        assert!(layer.order > 0); // non-default layer has order >= 1
    }

    #[test]
    fn test_set_layer_locked() {
        let mut scene = Scene::new();
        let layer_id = scene.add_layer("Locked");
        scene.set_layer_locked(layer_id, true);
        assert!(scene.is_layer_locked(layer_id));
        scene.set_layer_locked(layer_id, false);
        assert!(!scene.is_layer_locked(layer_id));
    }

    #[test]
    fn test_set_layer_color() {
        let mut scene = Scene::new();
        let layer_id = scene.add_layer("Colored");
        scene.set_layer_color(layer_id, [1.0, 0.0, 0.0, 1.0]);
        let layers = scene.layers();
        let layer = layers.iter().find(|l| l.id == layer_id).unwrap();
        assert_eq!(layer.color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_set_layer_order() {
        let mut scene = Scene::new();
        let layer_id = scene.add_layer("Orderly");
        scene.set_layer_order(layer_id, 99);
        let layers = scene.layers();
        let layer = layers.iter().find(|l| l.id == layer_id).unwrap();
        assert_eq!(layer.order, 99);
    }

    #[test]
    fn test_locked_layer_suppresses_selection_in_render_items() {
        let mut scene = Scene::new();
        let layer_id = scene.add_layer("Locked");
        let node_id = scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.set_layer(node_id, layer_id);
        scene.set_layer_locked(layer_id, true);

        let mut sel = Selection::new();
        sel.select_one(node_id);

        let items = scene.collect_render_items(&sel);
        assert_eq!(items.len(), 1, "locked layer nodes still render");
        assert!(
            !items[0].selected,
            "locked layer nodes must not appear selected"
        );
    }

    #[test]
    fn test_layers_sorted_by_order() {
        let mut scene = Scene::new();
        let a = scene.add_layer("A");
        let b = scene.add_layer("B");
        // Set reverse order
        scene.set_layer_order(a, 10);
        scene.set_layer_order(b, 5);
        let layers = scene.layers();
        // Find positions
        let pos_b = layers.iter().position(|l| l.id == b).unwrap();
        let pos_a = layers.iter().position(|l| l.id == a).unwrap();
        assert!(
            pos_b < pos_a,
            "layer B (order=5) should appear before A (order=10)"
        );
    }

    // --- Group tests ---

    #[test]
    fn test_create_group_returns_id() {
        let mut scene = Scene::new();
        let gid = scene.create_group("MyGroup");
        let group = scene.get_group(gid).unwrap();
        assert_eq!(group.name, "MyGroup");
        assert!(group.members.is_empty());
    }

    #[test]
    fn test_add_to_group_and_remove_from_group() {
        let mut scene = Scene::new();
        let gid = scene.create_group("G");
        let node_id = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        scene.add_to_group(node_id, gid);
        assert!(scene.get_group(gid).unwrap().members.contains(&node_id));
        scene.remove_from_group(node_id, gid);
        assert!(!scene.get_group(gid).unwrap().members.contains(&node_id));
    }

    #[test]
    fn test_groups_returns_all_groups() {
        let mut scene = Scene::new();
        scene.create_group("G1");
        scene.create_group("G2");
        assert_eq!(scene.groups().len(), 2);
    }

    #[test]
    fn test_node_groups_returns_containing_groups() {
        let mut scene = Scene::new();
        let g1 = scene.create_group("G1");
        let g2 = scene.create_group("G2");
        let node_id = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        scene.add_to_group(node_id, g1);
        scene.add_to_group(node_id, g2);
        let groups = scene.node_groups(node_id);
        assert_eq!(groups.len(), 2);
        assert!(groups.contains(&g1));
        assert!(groups.contains(&g2));
    }

    #[test]
    fn test_remove_node_cleans_up_group_membership() {
        let mut scene = Scene::new();
        let gid = scene.create_group("G");
        let node_id = scene.add(None, glam::Mat4::IDENTITY, Material::default());
        scene.add_to_group(node_id, gid);
        scene.remove(node_id);
        assert!(!scene.get_group(gid).unwrap().members.contains(&node_id));
    }

    // --- Mesh ref count tests ---

    #[test]
    fn test_mesh_ref_count_zero_for_unused_mesh() {
        let scene = Scene::new();
        assert_eq!(scene.mesh_ref_count(MeshId(42)), 0);
    }

    #[test]
    fn test_mesh_ref_count_correct_for_nodes() {
        let mut scene = Scene::new();
        scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.add(Some(MeshId(1)), glam::Mat4::IDENTITY, Material::default());
        assert_eq!(scene.mesh_ref_count(MeshId(0)), 2);
        assert_eq!(scene.mesh_ref_count(MeshId(1)), 1);
    }

    #[test]
    fn test_mesh_ref_count_decreases_after_remove() {
        let mut scene = Scene::new();
        let node_a = scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        scene.add(Some(MeshId(0)), glam::Mat4::IDENTITY, Material::default());
        assert_eq!(scene.mesh_ref_count(MeshId(0)), 2);
        scene.remove(node_a);
        assert_eq!(scene.mesh_ref_count(MeshId(0)), 1);
    }

    #[test]
    fn test_remove_group() {
        let mut scene = Scene::new();
        let gid = scene.create_group("G");
        scene.remove_group(gid);
        assert!(scene.get_group(gid).is_none());
        assert!(scene.groups().is_empty());
    }

    #[test]
    fn test_walk_depth_first_order() {
        let mut scene = Scene::new();
        let root = scene.add_named("root", None, glam::Mat4::IDENTITY, Material::default());
        let child_a = scene.add_named("a", None, glam::Mat4::IDENTITY, Material::default());
        let child_b = scene.add_named("b", None, glam::Mat4::IDENTITY, Material::default());
        let grandchild = scene.add_named("a1", None, glam::Mat4::IDENTITY, Material::default());
        scene.set_parent(child_a, Some(root));
        scene.set_parent(child_b, Some(root));
        scene.set_parent(grandchild, Some(child_a));

        let walk = scene.walk_depth_first();
        assert_eq!(walk.len(), 4);
        assert_eq!(walk[0], (root, 0));
        assert_eq!(walk[1], (child_a, 1));
        assert_eq!(walk[2], (grandchild, 2));
        assert_eq!(walk[3], (child_b, 1));
    }
}
