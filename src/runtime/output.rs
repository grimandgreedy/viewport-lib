//! Runtime output types: transform ops, selection ops, contact events, and generic events.

use super::events::RuntimeEventBus;
use crate::camera::camera::{Camera, CameraTarget};
use crate::interaction::selection::{NodeId, Selection};

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
    /// in order (last write wins after scene propagation). The full transform,
    /// including scale embedded in the `Affine3A`, replaces the node's current
    /// local transform.
    pub fn set(&mut self, id: NodeId, transform: glam::Affine3A) {
        self.ops.push(NodeTransformOp {
            id,
            transform,
            preserve_scale: false,
        });
    }

    /// Record a transform that should preserve the node's existing scale.
    ///
    /// The rotation and translation from `transform` are applied; the scale
    /// component (if any) is ignored and the scene node's current scale is
    /// kept. Use this for physics or animation writebacks: those systems don't
    /// model scale, so they shouldn't clobber a scale the user set
    /// independently (e.g. for visual stretching, ellipsoid bones, or
    /// art-time unit conversion).
    pub fn set_preserve_scale(&mut self, id: NodeId, transform: glam::Affine3A) {
        self.ops.push(NodeTransformOp {
            id,
            transform,
            preserve_scale: true,
        });
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
    /// When `true`, the runtime flush replaces only the rotation and
    /// translation, keeping the node's existing scale. Set via
    /// [`TransformWriteback::set_preserve_scale`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub preserve_scale: bool,
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

/// A camera state change produced by a runtime plugin.
///
/// Emitted by plugins onto [`RuntimeOutput::events`]; apply with
/// [`apply_camera_commands`].
///
/// Commands are applied in the order they were emitted (plugin priority order,
/// then registration order within the same priority). Each command builds on the
/// result of the previous one in the same frame.
///
/// This is independent of [`super::CameraFollow`]: camera-follow targets and
/// `CameraCommand`s coexist and the app decides which to apply.
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

/// Suggested camera center computed from the active [`super::CameraFollow`]
/// binding. Emitted onto [`RuntimeOutput::events`] when a follow target is
/// resolved this step. Apply to `camera.center` for orbit-camera follow
/// behavior.
#[derive(Debug, Clone, Copy)]
pub struct CameraFollowTarget(pub glam::Vec3);

/// Apply every [`CameraCommand`] currently in the event bus to `camera`, in
/// emission order. Drains the bus.
///
/// Commands are applied sequentially: each one builds on the result of the
/// previous. A `BlendToward` command blends from whatever state the camera is
/// in after all prior commands, not from the frame-start state.
pub fn apply_camera_commands(events: &mut RuntimeEventBus, camera: &mut Camera) {
    for cmd in events.drain::<CameraCommand>() {
        match cmd {
            CameraCommand::SetCenter(c) => {
                camera.center = c;
            }
            CameraCommand::OffsetCenter(d) => {
                camera.center += d;
            }
            CameraCommand::SetDistance(d) => {
                camera.set_distance(d);
            }
            CameraCommand::SetOrientation(q) => {
                camera.orientation = q.normalize();
            }
            CameraCommand::BlendToward { target, weight } => {
                let w = weight.clamp(0.0, 1.0);
                camera.center = camera.center.lerp(target.center, w);
                camera.distance = camera.distance + (target.distance - camera.distance) * w;
                camera.distance = camera.distance.max(0.001);
                camera.orientation = camera
                    .orientation
                    .slerp(target.orientation.normalize(), w)
                    .normalize();
            }
        }
    }
}

/// Output produced by one call to [`super::ViewportRuntime::step`].
///
/// `node_transform_ops` have already been applied to the scene and the
/// snapshot table when this is returned; `selection_ops` are applied to the
/// `Selection` argument. Everything else flows through the typed `events`
/// bus: plugins emit via `ctx.output.events.emit(MyEvent { .. })`, and the
/// app reads them after `step()` via `output.events.read::<MyEvent>()` or
/// `output.events.drain::<MyEvent>()`.
#[derive(Default)]
#[non_exhaustive]
pub struct RuntimeOutput {
    /// Transform ops applied to the scene during this step.
    pub node_transform_ops: Vec<NodeTransformOp>,
    /// Selection changes produced by runtime plugins, already applied to the
    /// app-owned [`Selection`].
    pub selection_ops: Vec<SelectionOp>,
    /// Generic typed event bus. Events are cleared each frame because
    /// `RuntimeOutput` is constructed fresh on every `step` call.
    pub events: RuntimeEventBus,
}
