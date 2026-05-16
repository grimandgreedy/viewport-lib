//! Built-in manipulation system for ViewportRuntime.

use std::collections::HashMap;

use crate::interaction::gizmo::{self, Gizmo, GizmoAxis, GizmoMode};
use crate::interaction::manipulation::GizmoInfo;
use crate::interaction::manipulation::{
    ManipulationContext, ManipulationController, ManipResult, TransformDelta,
};
use crate::interaction::selection::{NodeId, Selection};
use crate::runtime::context::RuntimeFrameContext;
use crate::runtime::output::{RuntimeOutput, TransformWriteback};
use crate::scene::scene::Scene;

/// Built-in manipulation system for [`super::super::ViewportRuntime`].
///
/// Wraps [`ManipulationController`] and drives it each frame from
/// [`RuntimeFrameContext`] inputs. Handles G/R/S key sessions and gizmo drag.
/// Writes transform changes via [`TransformWriteback`]; does not mutate the scene
/// directly.
///
/// To suppress orbit while a session is active, check
/// [`ViewportRuntime::is_manipulating`](super::super::ViewportRuntime::is_manipulating)
/// before calling [`OrbitCameraController::apply_to_camera`].
pub struct ManipulationSystem {
    controller: ManipulationController,
    transforms_snapshot: HashMap<NodeId, glam::Mat4>,
    pub(crate) gizmo: Gizmo,
    pub(crate) gizmo_center: Option<glam::Vec3>,
    pub(crate) gizmo_scale: f32,
}

impl Default for ManipulationSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ManipulationSystem {
    /// Create a new ManipulationSystem with default settings.
    pub fn new() -> Self {
        Self {
            controller: ManipulationController::new(),
            transforms_snapshot: HashMap::new(),
            gizmo: Gizmo::default(),
            gizmo_center: None,
            gizmo_scale: 1.0,
        }
    }

    /// True while a G/R/S session or gizmo drag is in progress.
    pub fn is_active(&self) -> bool {
        self.controller.is_active()
    }

    /// World-space center of the current selection, or None if selection is empty.
    pub fn gizmo_center(&self) -> Option<glam::Vec3> {
        self.gizmo_center
    }

    /// Screen-size scale factor for the gizmo arms.
    pub fn gizmo_scale(&self) -> f32 {
        self.gizmo_scale
    }

    /// Current gizmo mode (Translate / Rotate / Scale).
    pub fn gizmo_mode(&self) -> GizmoMode {
        self.gizmo.mode
    }

    /// Currently hovered gizmo axis.
    pub fn gizmo_hovered(&self) -> GizmoAxis {
        self.gizmo.hovered_axis
    }

    /// Build a GizmoInfo from current state, suitable for InteractionFrame.
    pub fn gizmo_info(&self) -> Option<GizmoInfo> {
        self.gizmo_center.map(|center| GizmoInfo {
            center,
            scale: self.gizmo_scale,
            orientation: glam::Quat::IDENTITY,
            mode: self.gizmo.mode,
        })
    }

    pub(crate) fn step(
        &mut self,
        frame: &RuntimeFrameContext,
        scene: &Scene,
        selection: &Selection,
        writeback: &mut TransformWriteback,
        _output: &mut RuntimeOutput,
    ) {
        // Compute gizmo center from selection.
        self.gizmo_center = gizmo::gizmo_center_from_selection(selection, |id| {
            scene.node(id).map(|n| {
                let t = n.world_transform();
                glam::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z)
            })
        });

        // Compute gizmo scale.
        if let Some(center) = self.gizmo_center {
            self.gizmo_scale = gizmo::compute_gizmo_scale(
                center,
                frame.camera.eye_position(),
                frame.camera.fov_y,
                frame.viewport_size.y,
            );
        }

        // Gizmo hover hit-test (only when no session is active).
        if !self.controller.is_active() {
            if let (Some(center), Some(cursor)) = (self.gizmo_center, frame.cursor_viewport) {
                let w = frame.viewport_size.x.max(1.0);
                let h = frame.viewport_size.y.max(1.0);
                let view_proj = frame.camera.proj_matrix() * frame.camera.view_matrix();
                let inv_vp = view_proj.inverse();
                let ndc_x = (cursor.x / w) * 2.0 - 1.0;
                let ndc_y = 1.0 - (cursor.y / h) * 2.0;
                let far = inv_vp.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
                let ray_origin = frame.camera.eye_position();
                let ray_dir = (far - ray_origin).normalize_or_zero();
                self.gizmo.hovered_axis = self.gizmo.hit_test_oriented(
                    ray_origin,
                    ray_dir,
                    center,
                    self.gizmo_scale,
                    glam::Quat::IDENTITY,
                );
            } else {
                self.gizmo.hovered_axis = GizmoAxis::None;
            }
        }

        // Build ManipulationContext.
        let gizmo_info = self.gizmo_center.map(|center| GizmoInfo {
            center,
            scale: self.gizmo_scale,
            orientation: glam::Quat::IDENTITY,
            mode: self.gizmo.mode,
        });
        let manip_ctx = ManipulationContext {
            camera: frame.camera.clone(),
            viewport_size: frame.viewport_size,
            cursor_viewport: frame.cursor_viewport,
            pointer_delta: frame.pointer_delta,
            selection_center: self.gizmo_center,
            gizmo: gizmo_info,
            drag_started: frame.drag_started,
            dragging: frame.dragging,
            clicked: frame.clicked,
        };

        let result = self.controller.update(frame.input, manip_ctx);

        match result {
            ManipResult::Update(delta) => {
                self.apply_delta(&delta, scene, selection, writeback);
            }
            ManipResult::Commit => {
                self.save_snapshots(scene, selection);
            }
            ManipResult::Cancel | ManipResult::ConstraintChanged => {
                self.restore_snapshots(writeback);
            }
            ManipResult::None => {
                if !self.controller.is_active() {
                    self.save_snapshots(scene, selection);
                }
            }
        }
    }

    fn save_snapshots(&mut self, scene: &Scene, selection: &Selection) {
        self.transforms_snapshot.clear();
        for &id in selection.iter() {
            if let Some(node) = scene.node(id) {
                self.transforms_snapshot.insert(id, node.local_transform());
            }
        }
    }

    fn restore_snapshots(&self, writeback: &mut TransformWriteback) {
        for (&id, &snap) in &self.transforms_snapshot {
            writeback.set(id, glam::Affine3A::from_mat4(snap));
        }
    }

    fn apply_delta(
        &self,
        delta: &TransformDelta,
        scene: &Scene,
        selection: &Selection,
        writeback: &mut TransformWriteback,
    ) {
        let Some(center) = self.gizmo_center else {
            return;
        };

        let has_pos_override = delta.position_override.iter().any(|v| v.is_some());
        let has_scale_override = delta.scale_override.iter().any(|v| v.is_some());

        let rot_mat = glam::Mat4::from_quat(delta.rotation);
        let scale_mat = glam::Mat4::from_scale(delta.scale);
        let translate_mat = glam::Mat4::from_translation(delta.translation);
        let to_pivot = glam::Mat4::from_translation(-center);
        let from_pivot = glam::Mat4::from_translation(center);

        for &id in selection.iter() {
            let base = if has_pos_override || has_scale_override {
                match self.transforms_snapshot.get(&id) {
                    Some(&snap) => snap,
                    None => continue,
                }
            } else {
                match scene.node(id) {
                    Some(n) => n.local_transform(),
                    None => continue,
                }
            };

            let mut new_t = translate_mat * from_pivot * rot_mat * scale_mat * to_pivot * base;

            // Apply per-axis position overrides (numeric input, e.g. G X 2).
            for (i, &ov) in delta.position_override.iter().enumerate() {
                if let Some(v) = ov {
                    new_t.col_mut(3)[i] = v;
                }
            }

            writeback.set(id, glam::Affine3A::from_mat4(new_t));
        }
    }
}
