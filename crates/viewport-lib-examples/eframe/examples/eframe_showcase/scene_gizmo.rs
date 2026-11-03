//! Driving the `viewport-lib-gizmo` transform gizmo over a `viewport-lib`
//! scene, for showcases 4 and 13.
//!
//! Both showcases keep viewport-lib's own `Selection`, since the scene's render
//! items read it, so the pivot and orientation are resolved from it here.

use std::collections::HashMap;

use viewport_lib as vpl;
use viewport_lib_gizmo::TransformDelta;
use vpl::{NodeId, scene::Scene, selection::Selection};

/// World-space centroid of the selected nodes, or `None` with nothing selected.
pub(crate) fn selection_centre(selection: &Selection, scene: &Scene) -> Option<glam::Vec3> {
    let positions: Vec<glam::Vec3> = selection
        .iter()
        .filter_map(|&id| scene.node(id))
        .map(|n| n.world_transform().translation.as_vec3())
        .collect();
    (!positions.is_empty()).then(|| positions.iter().sum::<glam::Vec3>() / positions.len() as f32)
}

/// World rotation of the primary selected node, for the gizmo's local space.
pub(crate) fn selection_orientation(selection: &Selection, scene: &Scene) -> glam::Quat {
    selection
        .primary()
        .and_then(|id| scene.node(id))
        .map(|n| glam::Quat::from_mat4(&n.world_transform().to_mat4()))
        .unwrap_or(glam::Quat::IDENTITY)
}

/// The local transforms of the selected nodes, taken when a session starts so
/// each frame's delta applies to the pose it began from.
#[derive(Default)]
pub(crate) struct Snapshots(HashMap<NodeId, glam::Mat4>);

impl Snapshots {
    /// Record the current local transform of every selected node.
    pub(crate) fn save(&mut self, selection: &Selection, scene: &Scene) {
        self.0.clear();
        for &id in selection.iter() {
            if let Some(node) = scene.node(id) {
                self.0.insert(id, node.local_transform().to_mat4());
            }
        }
    }

    /// Put every recorded node back where it was.
    pub(crate) fn restore(&self, scene: &mut Scene) {
        for (&id, &t) in &self.0 {
            scene.set_local_transform(id, t);
        }
        scene.update_transforms();
    }

    /// Set every recorded node to its snapshot with `delta` applied about
    /// `centre`. The delta is the whole change since the session began. Typed
    /// values replace the dragged ones per axis, read relative to the start
    /// pose; rotation is typed in degrees.
    pub(crate) fn apply(&self, delta: &TransformDelta, centre: glam::Vec3, scene: &mut Scene) {
        let pick = |typed: [Option<f32>; 3], dragged: glam::Vec3| {
            glam::Vec3::new(
                typed[0].unwrap_or(dragged.x),
                typed[1].unwrap_or(dragged.y),
                typed[2].unwrap_or(dragged.z),
            )
        };
        let translation = pick(delta.position_override, delta.translation);
        let scale = pick(delta.scale_override, delta.scale);
        let rotation = if delta.rotation_override.iter().any(Option::is_some) {
            let [x, y, z] = delta
                .rotation_override
                .map(|v| v.unwrap_or(0.0).to_radians());
            glam::Quat::from_euler(glam::EulerRot::XYZ, x, y, z)
        } else {
            delta.rotation
        };
        let about_centre = glam::Mat4::from_translation(translation)
            * glam::Mat4::from_translation(centre)
            * glam::Mat4::from_quat(rotation)
            * glam::Mat4::from_scale(scale)
            * glam::Mat4::from_translation(-centre);
        for (&id, &start) in &self.0 {
            scene.set_local_transform(id, about_centre * start);
        }
        scene.update_transforms();
    }
}
