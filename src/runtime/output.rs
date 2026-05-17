//! Runtime output types: transform ops, selection ops, contact events, and generic events.

use crate::interaction::selection::{NodeId, Selection};
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

/// Output produced by one call to [`super::ViewportRuntime::step`].
///
/// `node_transform_ops` have already been applied to the scene and the snapshot
/// table when this is returned. The other fields are for the app to read and
/// act on as needed.
///
/// Plugin-authored events of any type are collected in `events`. Use
/// `output.events.read::<T>()` or `output.events.drain::<T>()` to consume them.
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
    /// Generic typed event bus. Plugins emit events via `ctx.output.events.emit(MyEvent { .. })`.
    /// The app reads them after `step()` via `output.events.read::<MyEvent>()` or
    /// `output.events.drain::<MyEvent>()`. Events are cleared each frame because
    /// `RuntimeOutput` is constructed fresh on every `step` call.
    pub events: RuntimeEventBus,
}
