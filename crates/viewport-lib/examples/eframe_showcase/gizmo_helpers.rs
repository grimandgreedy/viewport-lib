//! Gizmo math helpers for showcases 4 and 18.

use viewport_lib::{Gizmo, GizmoSpace, scene::Scene, selection::Selection};

use crate::App;

// ---------------------------------------------------------------------------
// Generic gizmo orientation
// ---------------------------------------------------------------------------

/// Return the world-space orientation quaternion for a gizmo given its space
/// setting and the current selection.
pub(crate) fn gizmo_orientation(gizmo: &Gizmo, selection: &Selection, scene: &Scene) -> glam::Quat {
    match gizmo.space {
        GizmoSpace::World => glam::Quat::IDENTITY,
        GizmoSpace::Local => selection
            .primary()
            .and_then(|id| scene.node(id))
            .map(|n| glam::Quat::from_mat4(&n.world_transform()))
            .unwrap_or(glam::Quat::IDENTITY),
    }
}

// ---------------------------------------------------------------------------
// Clip-volume gizmo helpers (Showcase 18)
//
// Clip volumes are now edited via inline sliders in the side panel.
// The gizmo is not used for clip editing, so these return neutral values.
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn clip_gizmo_center(&self) -> Option<glam::Vec3> {
        None
    }

    pub(crate) fn clipvol_gizmo_orient(&self) -> glam::Quat {
        glam::Quat::IDENTITY
    }

    pub(crate) fn apply_clipvol_gizmo_drag(&mut self, _dx: f32, _dy: f32, _w: f32, _h: f32) {}
}
