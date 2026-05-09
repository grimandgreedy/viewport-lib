//! Gizmo math helpers for showcases 4 and 18.

use viewport_lib::{Gizmo, GizmoAxis, GizmoMode, GizmoSpace, selection::Selection, scene::Scene};

use crate::{App, gizmo};

// ---------------------------------------------------------------------------
// Generic gizmo orientation
// ---------------------------------------------------------------------------

/// Return the world-space orientation quaternion for a gizmo given its space
/// setting and the current selection.
pub(crate) fn gizmo_orientation(
    gizmo: &Gizmo,
    selection: &Selection,
    scene: &Scene,
) -> glam::Quat {
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
// ---------------------------------------------------------------------------

impl App {
    /// Compute the world-space gizmo center for the active clip sub-mode.
    pub(crate) fn clip_gizmo_center(&self) -> Option<glam::Vec3> {
        use crate::showcase_18_clip_volumes::ClipVolSubMode;
        use viewport_lib::ClipShape;
        match self.clipvol_state.sub_mode {
            ClipVolSubMode::BoxClip => Some(glam::Vec3::from(self.clipvol_state.box_center)),
            ClipVolSubMode::SphereClip => Some(glam::Vec3::from(self.clipvol_state.sphere_center)),
            ClipVolSubMode::InteractivePlane => {
                if let ClipShape::Plane { normal, distance, .. } = self.clipvol_state.plane.shape {
                    Some(glam::Vec3::from(normal) * (-distance))
                } else {
                    None
                }
            }
        }
    }

    /// Compute the gizmo orientation quaternion for the active clip sub-mode.
    pub(crate) fn clipvol_gizmo_orient(&self) -> glam::Quat {
        use crate::showcase_18_clip_volumes::ClipVolSubMode;
        use viewport_lib::ClipShape;
        match self.clipvol_state.sub_mode {
            ClipVolSubMode::BoxClip => {
                glam::Quat::from_rotation_z(self.clipvol_state.box_yaw.to_radians())
            }
            ClipVolSubMode::SphereClip => glam::Quat::IDENTITY,
            ClipVolSubMode::InteractivePlane => {
                if let ClipShape::Plane { normal, .. } = self.clipvol_state.plane.shape {
                    let n = glam::Vec3::from(normal).normalize_or_zero();
                    if n.length_squared() > 0.001 {
                        glam::Quat::from_rotation_arc(glam::Vec3::Z, n)
                    } else {
                        glam::Quat::IDENTITY
                    }
                } else {
                    glam::Quat::IDENTITY
                }
            }
        }
    }

    pub(crate) fn apply_clipvol_gizmo_drag(&mut self, dx: f32, dy: f32, w: f32, h: f32) {
        use crate::showcase_18_clip_volumes::ClipVolSubMode;
        use viewport_lib::ClipShape;

        let Some(center) = self.clipvol_state.gizmo_center else {
            return;
        };
        let drag_delta = glam::Vec2::new(dx, dy);
        let viewport_size = glam::Vec2::new(w, h);
        let vp = self.camera.view_proj_matrix();
        let view = self.camera.view_matrix();
        let axis = self.clipvol_state.gizmo.active_axis;
        let orient = self.clipvol_gizmo_orient();

        let axis_dir = |a: GizmoAxis| -> glam::Vec3 {
            let base = match a {
                GizmoAxis::X => glam::Vec3::X,
                GizmoAxis::Y => glam::Vec3::Y,
                GizmoAxis::Z => glam::Vec3::Z,
                _ => glam::Vec3::X,
            };
            orient * base
        };

        match self.clipvol_state.gizmo.mode {
            GizmoMode::Translate => {
                let delta = match axis {
                    GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                        let dir = axis_dir(axis);
                        let amount = gizmo::project_drag_onto_axis(
                            drag_delta, dir, vp, center, viewport_size,
                        );
                        dir * amount
                    }
                    GizmoAxis::XY => gizmo::project_drag_onto_plane(
                        drag_delta, orient * glam::Vec3::X, orient * glam::Vec3::Y,
                        vp, center, viewport_size,
                    ),
                    GizmoAxis::XZ => gizmo::project_drag_onto_plane(
                        drag_delta, orient * glam::Vec3::X, orient * glam::Vec3::Z,
                        vp, center, viewport_size,
                    ),
                    GizmoAxis::YZ => gizmo::project_drag_onto_plane(
                        drag_delta, orient * glam::Vec3::Y, orient * glam::Vec3::Z,
                        vp, center, viewport_size,
                    ),
                    GizmoAxis::Screen => gizmo::project_drag_onto_screen_plane(
                        drag_delta, self.camera.right(), self.camera.up(),
                        vp, center, viewport_size,
                    ),
                    _ => glam::Vec3::ZERO,
                };
                match self.clipvol_state.sub_mode {
                    ClipVolSubMode::BoxClip => {
                        self.clipvol_state.box_center[0] += delta.x;
                        self.clipvol_state.box_center[1] += delta.y;
                        self.clipvol_state.box_center[2] += delta.z;
                    }
                    ClipVolSubMode::SphereClip => {
                        self.clipvol_state.sphere_center[0] += delta.x;
                        self.clipvol_state.sphere_center[1] += delta.y;
                        self.clipvol_state.sphere_center[2] += delta.z;
                    }
                    ClipVolSubMode::InteractivePlane => {
                        if let ClipShape::Plane { ref mut distance, ref normal, .. } =
                            self.clipvol_state.plane.shape
                        {
                            let n = glam::Vec3::from(*normal).normalize_or_zero();
                            *distance -= delta.dot(n);
                        }
                    }
                }
            }

            GizmoMode::Rotate => {
                let angle = match axis {
                    GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                        let dir = axis_dir(axis);
                        gizmo::project_drag_onto_rotation(drag_delta, dir, view)
                    }
                    _ => 0.0,
                };
                if angle.abs() > 1e-6 {
                    match self.clipvol_state.sub_mode {
                        ClipVolSubMode::BoxClip => {
                            self.clipvol_state.box_yaw += angle.to_degrees();
                        }
                        ClipVolSubMode::InteractivePlane => {
                            if let ClipShape::Plane {
                                ref mut normal,
                                ref mut distance,
                                ..
                            } = self.clipvol_state.plane.shape
                            {
                                let n = glam::Vec3::from(*normal);
                                let rot_axis = axis_dir(axis);
                                let rot = glam::Quat::from_axis_angle(rot_axis, angle);
                                let new_n = (rot * n).normalize_or_zero();
                                let anchor = n * (-*distance);
                                *distance = -(anchor.dot(new_n));
                                *normal = new_n.to_array();
                            }
                        }
                        ClipVolSubMode::SphereClip => {}
                    }
                }
            }

            GizmoMode::Scale => {
                let amount = match axis {
                    GizmoAxis::X | GizmoAxis::Y | GizmoAxis::Z => {
                        let dir = axis_dir(axis);
                        gizmo::project_drag_onto_axis(drag_delta, dir, vp, center, viewport_size)
                    }
                    _ => 0.0,
                };
                if amount.abs() > 1e-6 {
                    match self.clipvol_state.sub_mode {
                        ClipVolSubMode::BoxClip => {
                            let scale = 1.0 + amount;
                            match axis {
                                GizmoAxis::X => {
                                    self.clipvol_state.box_half_extents[0] =
                                        (self.clipvol_state.box_half_extents[0] * scale).max(0.1);
                                }
                                GizmoAxis::Y => {
                                    self.clipvol_state.box_half_extents[1] =
                                        (self.clipvol_state.box_half_extents[1] * scale).max(0.1);
                                }
                                GizmoAxis::Z => {
                                    self.clipvol_state.box_half_extents[2] =
                                        (self.clipvol_state.box_half_extents[2] * scale).max(0.1);
                                }
                                _ => {}
                            }
                        }
                        ClipVolSubMode::SphereClip => {
                            self.clipvol_state.sphere_radius =
                                (self.clipvol_state.sphere_radius + amount).max(0.1);
                        }
                        ClipVolSubMode::InteractivePlane => {}
                    }
                }
            }

            _ => {}
        }
    }
}
