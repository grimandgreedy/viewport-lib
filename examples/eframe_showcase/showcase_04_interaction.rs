//! Showcase 4: Professional Interaction.

use crate::App;
use eframe::egui;
use std::collections::HashMap;
use viewport_lib::{
    CameraAnimator, Easing, Gizmo, GizmoMode, GizmoSpace, ManipulationController, Material, NodeId,
    ViewPreset, ViewportRenderer, scene::Scene, selection::Selection,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct InteractState {
    pub scene: Scene,
    pub selection: Selection,
    pub animator: CameraAnimator,
    pub gizmo: Gizmo,
    pub manip: ManipulationController,
    pub transforms_snapshot: HashMap<NodeId, glam::Mat4>,
    pub left_held: bool,
    pub built: bool,
    pub gizmo_center: Option<glam::Vec3>,
    pub gizmo_scale: f32,
    pub spline: viewport_lib::SplineWidget,
    pub last_cursor_viewport: glam::Vec2,
}

impl Default for InteractState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            selection: Selection::new(),
            animator: CameraAnimator::with_default_damping(),
            gizmo: Gizmo::new(),
            manip: ManipulationController::new(),
            transforms_snapshot: HashMap::new(),
            left_held: false,
            built: false,
            gizmo_center: None,
            gizmo_scale: 1.0,
            spline: viewport_lib::SplineWidget::new(vec![
                glam::Vec3::new(-2.0, 0.0, 1.5),
                glam::Vec3::new(-0.5, 1.5, 1.5),
                glam::Vec3::new(0.5, -1.5, 1.5),
                glam::Vec3::new(2.0, 0.0, 1.5),
            ]),
            last_cursor_viewport: glam::Vec2::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn build_interact_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.interact_state.scene = Scene::new();
        self.interact_state.selection.clear();

        let positions = [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
        ];
        let colours = [
            [0.85, 0.85, 0.85],
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.3],
            [0.3, 0.3, 0.9],
            [0.9, 0.9, 0.3],
        ];
        let names = ["Center", "Right", "Left", "Front", "Back"];

        for (i, ((pos, colour), name)) in positions.iter().zip(&colours).zip(&names).enumerate() {
            let mesh = self.upload_box(renderer);
            let transform = glam::Mat4::from_translation(glam::Vec3::from(*pos));
            let mat = Material::from_colour(*colour);
            let id = self
                .interact_state
                .scene
                .add_named(name, Some(mesh), transform, mat);
            if i == 0 {
                self.interact_state.selection.select_one(id);
            }
        }

        self.interact_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Manipulation helpers
// ---------------------------------------------------------------------------

impl App {
    /// Apply a [`viewport_lib::TransformDelta`] to all selected scene nodes.
    ///
    /// Rotation and scale pivot around `interact_gizmo_center`.
    /// When `position_override` or `scale_override` is set (numeric input), the
    /// snapshot is restored first and the override applied as an absolute value.
    pub(crate) fn apply_interact_delta(&mut self, delta: viewport_lib::TransformDelta) {
        let Some(center) = self.interact_state.gizmo_center else {
            return;
        };

        let has_pos_override = delta.position_override.iter().any(|v| v.is_some());
        let has_scale_override = delta.scale_override.iter().any(|v| v.is_some());

        if has_pos_override || has_scale_override {
            self.restore_interact_snapshots();
        }

        let translation = if has_pos_override {
            glam::Vec3::new(
                delta.position_override[0].unwrap_or(0.0),
                delta.position_override[1].unwrap_or(0.0),
                delta.position_override[2].unwrap_or(0.0),
            )
        } else {
            delta.translation
        };

        let scale = if has_scale_override {
            glam::Vec3::new(
                delta.scale_override[0].unwrap_or(1.0),
                delta.scale_override[1].unwrap_or(1.0),
                delta.scale_override[2].unwrap_or(1.0),
            )
        } else {
            delta.scale
        };

        let rot_mat = glam::Mat4::from_quat(delta.rotation);
        let scale_mat = glam::Mat4::from_scale(scale);
        let translate_mat = glam::Mat4::from_translation(translation);
        let to_pivot = glam::Mat4::from_translation(-center);
        let from_pivot = glam::Mat4::from_translation(center);

        for id in self
            .interact_state
            .selection
            .iter()
            .copied()
            .collect::<Vec<_>>()
        {
            if let Some(node) = self.interact_state.scene.node(id) {
                let cur = node.local_transform();
                let new_t = translate_mat * from_pivot * rot_mat * scale_mat * to_pivot * cur;
                self.interact_state.scene.set_local_transform(id, new_t);
            }
        }
        self.interact_state.scene.update_transforms();
    }

    /// Snapshot the current local transforms of all selected nodes.
    pub(crate) fn save_interact_snapshots(&mut self) {
        self.interact_state.transforms_snapshot.clear();
        for id in self
            .interact_state
            .selection
            .iter()
            .copied()
            .collect::<Vec<_>>()
        {
            if let Some(node) = self.interact_state.scene.node(id) {
                self.interact_state
                    .transforms_snapshot
                    .insert(id, node.local_transform());
            }
        }
    }

    /// Restore local transforms from the last snapshot (used by Cancel / ConstraintChanged).
    pub(crate) fn restore_interact_snapshots(&mut self) {
        let ids: Vec<_> = self
            .interact_state
            .transforms_snapshot
            .keys()
            .copied()
            .collect();
        for id in ids {
            if let Some(&t) = self.interact_state.transforms_snapshot.get(&id) {
                self.interact_state.scene.set_local_transform(id, t);
            }
        }
        self.interact_state.scene.update_transforms();
    }

    pub(crate) fn zoom_to_fit_interact(&mut self) {
        let mut min = glam::Vec3::splat(f32::INFINITY);
        let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
        let mut any = false;

        let iter_nodes: Vec<_> = if !self.interact_state.selection.is_empty() {
            self.interact_state.selection.iter().copied().collect()
        } else {
            self.interact_state
                .scene
                .walk_depth_first()
                .iter()
                .map(|(id, _)| *id)
                .collect()
        };

        for nid in iter_nodes {
            if let Some(node) = self.interact_state.scene.node(nid) {
                let t = node.world_transform();
                let pos = glam::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z);
                min = min.min(pos - glam::Vec3::splat(0.6));
                max = max.max(pos + glam::Vec3::splat(0.6));
                any = true;
            }
        }

        if any {
            let aabb = viewport_lib::Aabb { min, max };
            let target = self.camera.fit_aabb_target(&aabb);
            self.interact_state.animator.fly_to(
                &self.camera,
                target.center,
                target.distance,
                target.orientation,
                0.6,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_interaction(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Gizmo Mode:");
    ui.horizontal(|ui| {
        if ui
            .radio(
                app.interact_state.gizmo.mode == GizmoMode::Translate,
                "Translate",
            )
            .clicked()
        {
            app.interact_state.gizmo.mode = GizmoMode::Translate;
        }
        if ui
            .radio(app.interact_state.gizmo.mode == GizmoMode::Rotate, "Rotate")
            .clicked()
        {
            app.interact_state.gizmo.mode = GizmoMode::Rotate;
        }
        if ui
            .radio(app.interact_state.gizmo.mode == GizmoMode::Scale, "Scale")
            .clicked()
        {
            app.interact_state.gizmo.mode = GizmoMode::Scale;
        }
    });

    ui.separator();

    ui.label("Gizmo Space:");
    ui.horizontal(|ui| {
        if ui
            .radio(app.interact_state.gizmo.space == GizmoSpace::World, "World")
            .clicked()
        {
            app.interact_state.gizmo.space = GizmoSpace::World;
        }
        if ui
            .radio(app.interact_state.gizmo.space == GizmoSpace::Local, "Local")
            .clicked()
        {
            app.interact_state.gizmo.space = GizmoSpace::Local;
        }
    });

    ui.separator();
    ui.label("Shortcuts: G move · R rotate · S scale");
    ui.label("X / Y / Z : constrain axis  ·  Enter / click : confirm  ·  Esc : cancel");
    ui.separator();
    ui.label("View presets:");
    egui::Grid::new("view_presets_grid")
        .num_columns(4)
        .show(ui, |ui| {
            for (label, preset) in [
                ("Front", ViewPreset::Front),
                ("Back", ViewPreset::Back),
                ("Left", ViewPreset::Left),
                ("Right", ViewPreset::Right),
                ("Top", ViewPreset::Top),
                ("Bottom", ViewPreset::Bottom),
                ("Iso", ViewPreset::Isometric),
            ] {
                if ui.button(label).clicked() {
                    app.interact_state.animator.fly_to_full(
                        &app.camera,
                        app.camera.center,
                        app.camera.distance,
                        preset.orientation(),
                        preset.preferred_projection(),
                        0.6,
                        Easing::EaseInOutCubic,
                    );
                }
            }
        });

    ui.separator();

    if ui.button("Zoom to Fit").clicked() {
        app.zoom_to_fit_interact();
    }

    ui.separator();

    if ui.button("Clear Selection").clicked() {
        app.interact_state.selection.clear();
    }
}
