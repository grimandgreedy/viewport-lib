//! Runtime output types: transform ops, selection ops, contact events, and generic events.

use crate::camera::camera::{Camera, CameraTarget};
use crate::interaction::selection::{NodeId, Selection};
use crate::resources::mesh_store::MeshId;
use super::events::RuntimeEventBus;

/// Write buffer for transform changes produced by plugins.
///
/// Passed into [`super::context::RuntimeStepContext`] so plugins can record
/// transform changes without directly mutating the scene. The runtime flushes
/// all ops to the scene after the `Writeback` phase.
#[derive(Default)]
pub struct TransformWriteback {
    ops: Vec<NodeTransformOp>,
}

impl TransformWriteback {
    /// Record a new local-space transform for a scene node.
    ///
    /// For physics-driven nodes with no parent, local space equals world space.
    /// If the same node is written more than once, all ops are kept and applied
    /// in order (last write wins after scene propagation).
    pub fn set(&mut self, id: NodeId, transform: glam::Affine3A) {
        self.ops.push(NodeTransformOp { id, transform });
    }

    /// Consume the buffer and return all recorded ops.
    pub(super) fn into_ops(self) -> Vec<NodeTransformOp> {
        self.ops
    }
}

/// A transform write targeting one scene node.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeTransformOp {
    /// Target scene node.
    pub id: NodeId,
    /// New local-space transform for the node. For physics-driven root nodes,
    /// this is the world-space transform.
    pub transform: glam::Affine3A,
}

/// A change to the selection state.
///
/// Produced by runtime plugins and returned in [`RuntimeOutput::selection_ops`].
/// The runtime applies these to the app-owned [`Selection`] during each step.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SelectionOp {
    /// Clear all selected nodes and select one.
    SelectOne(NodeId),
    /// Toggle a node's selected state.
    Toggle(NodeId),
    /// Add a node to the selection.
    Add(NodeId),
    /// Remove a node from the selection.
    Remove(NodeId),
    /// Add multiple nodes to the selection.
    Extend(Vec<NodeId>),
    /// Clear the selection.
    Clear,
    /// Replace the selection with the given set.
    SelectAll(Vec<NodeId>),
}

impl SelectionOp {
    /// Apply this operation to a [`Selection`].
    pub fn apply_to(&self, selection: &mut Selection) {
        match self {
            SelectionOp::SelectOne(id) => selection.select_one(*id),
            SelectionOp::Toggle(id) => selection.toggle(*id),
            SelectionOp::Add(id) => selection.add(*id),
            SelectionOp::Remove(id) => selection.remove(*id),
            SelectionOp::Extend(ids) => selection.extend(ids.iter().copied()),
            SelectionOp::Clear => selection.clear(),
            SelectionOp::SelectAll(ids) => selection.select_all(ids.iter().copied()),
        }
    }
}

/// A contact event produced by a physics plugin during the `Simulate` phase.
///
/// Returned in [`RuntimeOutput::contact_events`] for the app to use in game logic,
/// sound, or effects. Not applied to the scene by the runtime.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContactEvent {
    /// First node involved in the contact.
    pub node_a: NodeId,
    /// Second node involved in the contact.
    pub node_b: NodeId,
    /// Contact normal pointing from `node_a` toward `node_b`, in world space.
    pub world_normal: glam::Vec3,
    /// Magnitude of the impulse applied at the contact point.
    pub impulse: f32,
    /// World-space position of the contact point.
    ///
    /// Use this for placing sound sources, particle effects, or decals at the
    /// collision site. Simple plugins that do not compute a contact point may
    /// leave this as `Vec3::ZERO`.
    pub contact_point: glam::Vec3,
}

/// A camera state change produced by a runtime plugin.
///
/// Accumulated in [`RuntimeOutput::camera_commands`] and applied to the app-owned
/// [`Camera`] by calling [`RuntimeOutput::apply_camera_commands`].
///
/// Commands are applied in the order they were emitted (plugin priority order,
/// then registration order within the same priority). Each command builds on the
/// result of the previous one in the same frame.
///
/// This is independent of [`super::CameraFollow`]: `camera_follow_target` and
/// `camera_commands` coexist and the app decides which to apply.
#[derive(Debug, Clone)]
pub enum CameraCommand {
    /// Set the orbit center (pivot point) to an absolute world position.
    SetCenter(glam::Vec3),
    /// Add a world-space delta to the orbit center.
    OffsetCenter(glam::Vec3),
    /// Set the camera distance from the center. Clamped to a small positive value.
    SetDistance(f32),
    /// Set the camera orientation.
    SetOrientation(glam::Quat),
    /// Blend center, distance, and orientation toward a target state.
    ///
    /// `weight` is in `[0, 1]`. At `1.0` the camera snaps to `target` immediately.
    /// Smaller values produce smooth motion when emitted every frame.
    BlendToward {
        /// Target camera state to blend toward.
        target: CameraTarget,
        /// Blend weight in `[0, 1]`.
        weight: f32,
    },
}

/// A per-mesh deformation update produced by a skinning plugin.
///
/// Returned in [`RuntimeOutput::skinned_mesh_updates`]. Apply after `step()`:
///
/// ```rust,ignore
/// for u in &output.skinned_mesh_updates {
///     renderer.resources_mut()
///         .write_mesh_positions_normals(queue, u.mesh_id, &u.positions, &u.normals)
///         .ok();
/// }
/// ```
pub struct SkinnedMeshUpdate {
    /// The mesh to deform.
    pub mesh_id: MeshId,
    /// Skinned vertex positions in local space.
    pub positions: Vec<[f32; 3]>,
    /// Skinned vertex normals.
    pub normals: Vec<[f32; 3]>,
}

/// Output produced by one call to [`super::ViewportRuntime::step`].
///
/// `node_transform_ops` have already been applied to the scene and the snapshot
/// table when this is returned. The other fields are for the app to read and
/// act on as needed.
///
/// Plugin-authored events of any type are collected in `events`. Use
/// `output.events.read::<T>()` or `output.events.drain::<T>()` to consume them.
/// Plugin-authored camera changes are in `camera_commands`; apply them with
/// `output.apply_camera_commands(&mut camera)`.
#[derive(Default)]
pub struct RuntimeOutput {
    /// Transform ops applied to the scene during this step.
    pub node_transform_ops: Vec<NodeTransformOp>,
    /// Selection changes produced by runtime plugins, already applied to the
    /// app-owned [`Selection`].
    pub selection_ops: Vec<SelectionOp>,
    /// Contact events from physics plugins. Empty if no physics plugin is active.
    pub contact_events: Vec<ContactEvent>,
    /// Suggested camera center computed from the active [`super::CameraFollow`] binding.
    ///
    /// `Some` when a `CameraFollow::Node` target was resolved this step; `None`
    /// when no follow binding is set or the target node was not found. Apply to
    /// `camera.center` for orbit-camera follow behavior.
    pub camera_follow_target: Option<glam::Vec3>,
    /// Camera commands emitted by plugins this frame. Apply with
    /// [`Self::apply_camera_commands`]. Empty when no camera plugin is active.
    pub camera_commands: Vec<CameraCommand>,
    /// Generic typed event bus. Plugins emit events via `ctx.output.events.emit(MyEvent { .. })`.
    /// The app reads them after `step()` via `output.events.read::<MyEvent>()` or
    /// `output.events.drain::<MyEvent>()`. Events are cleared each frame because
    /// `RuntimeOutput` is constructed fresh on every `step` call.
    pub events: RuntimeEventBus,
    /// Per-mesh deformation updates from skinning plugins. Apply after `step()` by
    /// calling `write_mesh_positions_normals` on each entry. Empty when no
    /// [`super::plugins::SkeletonPlugin`] is active.
    pub skinned_mesh_updates: Vec<SkinnedMeshUpdate>,
}

impl RuntimeOutput {
    /// Apply all camera commands in emission order to `camera`.
    ///
    /// Call this after [`super::ViewportRuntime::step`] returns, before rendering.
    /// Has no effect if `camera_commands` is empty.
    ///
    /// Commands are applied sequentially: each one builds on the result of the
    /// previous. A `BlendToward` command blends from whatever state the camera
    /// is in after all prior commands, not from the frame-start state.
    pub fn apply_camera_commands(&self, camera: &mut Camera) {
        for cmd in &self.camera_commands {
            match cmd {
                CameraCommand::SetCenter(c) => {
                    camera.center = *c;
                }
                CameraCommand::OffsetCenter(d) => {
                    camera.center += *d;
                }
                CameraCommand::SetDistance(d) => {
                    camera.set_distance(*d);
                }
                CameraCommand::SetOrientation(q) => {
                    camera.orientation = q.normalize();
                }
                CameraCommand::BlendToward { target, weight } => {
                    let w = weight.clamp(0.0, 1.0);
                    camera.center = camera.center.lerp(target.center, w);
                    camera.distance = camera.distance + (target.distance - camera.distance) * w;
                    camera.distance = camera.distance.max(0.001);
                    camera.orientation = camera.orientation
                        .slerp(target.orientation.normalize(), w)
                        .normalize();
                }
            }
        }
    }
}
