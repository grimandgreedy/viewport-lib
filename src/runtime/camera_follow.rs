//! Camera tracking binding for follow-camera behavior.

use crate::interaction::selection::NodeId;

/// Camera tracking mode for [`super::ViewportRuntime`].
///
/// When set, the runtime computes a suggested camera center after each step
/// and returns it in [`super::RuntimeOutput::camera_follow_target`]. The app
/// applies the suggestion by setting `camera.center` (for an orbit camera) or
/// using it however fits the application.
///
/// Set via [`super::ViewportRuntime::set_camera_follow`] or
/// [`super::ViewportRuntime::with_camera_follow`].
#[derive(Debug, Clone)]
pub enum CameraFollow {
    /// Track a scene node.
    ///
    /// The suggested center is `node_world_pos + offset`. Orbit camera distance
    /// and orientation are unaffected; the camera pivots around the moving target.
    Node {
        /// Node to follow.
        id: NodeId,
        /// World-space offset added to the node's position.
        offset: glam::Vec3,
        /// Unused by the runtime itself. Apps may use it to decide whether to
        /// orient the camera toward the node.
        look_at: bool,
    },
    /// No tracking. The runtime does not set `camera_follow_target`.
    Free,
}
