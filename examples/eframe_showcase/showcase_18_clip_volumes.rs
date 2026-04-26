//! Showcase 18: Clip Objects & Gizmo
//!
//! Demonstrates `ClipObject` with box, sphere, and interactive plane shapes on
//! `EffectsFrame`. A dense UV sphere (high vertex count) makes the clip cross-section
//! visually rich. Setting `ClipObject::color` causes the lib to draw the clip boundary
//! automatically : a wireframe outline for box/sphere, a fill quad for plane.
//!
//! Sub-modes (radio):
//! - Box clip   : oriented box; center / half-extents / yaw sliders.
//! - Sphere clip : sphere; center / radius sliders.
//! - Interactive plane : a plane dragged via the gizmo.
//!
//! Common controls:
//! - Sub-mode selector radio buttons.
//! - Gizmo mode (Translate / Rotate / Scale).
//! - Show overlay checkbox.

use crate::App;
use eframe::egui;
use viewport_lib::{
    ClipAxis, ClipObject, ClipShape, GizmoMode, Material, MeshId, ViewportRenderer,
    plane_from_axis_preset, scene::Scene,
};

// ---------------------------------------------------------------------------
// Sub-mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClipVolSubMode {
    BoxClip,
    SphereClip,
    InteractivePlane,
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// One-time GPU setup for Showcase 18.
    ///
    /// Uploads a dense UV sphere (48×24 segments) and a flat ground plane.
    pub(crate) fn build_clipvol_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.clipvol_scene = Scene::new();

        // Dense sphere : many vertices make the clip cross-section visually rich.
        let sphere_mesh = viewport_lib::primitives::sphere(3.0, 48, 24);
        let sphere_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("clipvol sphere mesh");
        let sphere_id = MeshId::from_index(sphere_idx);
        self.clipvol_scene.add_named(
            "Sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, 0.0)),
            {
                let mut m = Material::from_color([0.55, 0.72, 0.95]);
                m.roughness = 0.3;
                m.metallic = 0.1;
                m
            },
        );

        // Ground plane.
        let ground_mesh = viewport_lib::primitives::plane(14.0, 14.0);
        let ground_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("clipvol ground mesh");
        let ground_id = MeshId::from_index(ground_idx);
        self.clipvol_scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -3.05)),
            {
                let mut m = Material::from_color([0.38, 0.38, 0.40]);
                m.roughness = 0.9;
                m
            },
        );

        self.clipvol_built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls panel
// ---------------------------------------------------------------------------

impl App {
    pub(crate) fn controls_clipvol(&mut self, ui: &mut egui::Ui) {
        ui.label("Clip mode:");
        ui.horizontal(|ui| {
            if ui
                .radio(self.clipvol_sub_mode == ClipVolSubMode::BoxClip, "Box")
                .clicked()
            {
                self.clipvol_sub_mode = ClipVolSubMode::BoxClip;
                // Scale is valid for box; keep mode as-is.
            }
            if ui
                .radio(
                    self.clipvol_sub_mode == ClipVolSubMode::SphereClip,
                    "Sphere",
                )
                .clicked()
            {
                self.clipvol_sub_mode = ClipVolSubMode::SphereClip;
                // Sphere has no Rotate mode.
                if self.clipvol_gizmo.mode == GizmoMode::Rotate {
                    self.clipvol_gizmo.mode = GizmoMode::Translate;
                }
            }
            if ui
                .radio(
                    self.clipvol_sub_mode == ClipVolSubMode::InteractivePlane,
                    "Plane",
                )
                .clicked()
            {
                self.clipvol_sub_mode = ClipVolSubMode::InteractivePlane;
                // Plane has no Scale mode.
                if self.clipvol_gizmo.mode == GizmoMode::Scale {
                    self.clipvol_gizmo.mode = GizmoMode::Translate;
                }
            }
        });

        ui.separator();

        // ----- Gizmo mode + overlay -----
        ui.label("Gizmo:");
        ui.horizontal(|ui| {
            if ui
                .radio(self.clipvol_gizmo.mode == GizmoMode::Translate, "Translate")
                .clicked()
            {
                self.clipvol_gizmo.mode = GizmoMode::Translate;
            }
            let can_rotate = self.clipvol_sub_mode != ClipVolSubMode::SphereClip;
            ui.add_enabled_ui(can_rotate, |ui| {
                if ui
                    .radio(self.clipvol_gizmo.mode == GizmoMode::Rotate, "Rotate")
                    .clicked()
                {
                    self.clipvol_gizmo.mode = GizmoMode::Rotate;
                }
            });
            let can_scale = self.clipvol_sub_mode != ClipVolSubMode::InteractivePlane;
            ui.add_enabled_ui(can_scale, |ui| {
                if ui
                    .radio(self.clipvol_gizmo.mode == GizmoMode::Scale, "Scale")
                    .clicked()
                {
                    self.clipvol_gizmo.mode = GizmoMode::Scale;
                }
            });
        });
        ui.checkbox(&mut self.clipvol_show_overlay, "Show overlay");

        ui.separator();

        match self.clipvol_sub_mode {
            ClipVolSubMode::BoxClip => self.controls_clipvol_box(ui),
            ClipVolSubMode::SphereClip => self.controls_clipvol_sphere(ui),
            ClipVolSubMode::InteractivePlane => self.controls_clipvol_plane(ui),
        }
    }

    fn controls_clipvol_box(&mut self, ui: &mut egui::Ui) {
        ui.label("Box center:");
        ui.horizontal(|ui| {
            ui.label("X:");
            ui.add(egui::DragValue::new(&mut self.clipvol_box_center[0]).speed(0.05));
            ui.label("Y:");
            ui.add(egui::DragValue::new(&mut self.clipvol_box_center[1]).speed(0.05));
            ui.label("Z:");
            ui.add(egui::DragValue::new(&mut self.clipvol_box_center[2]).speed(0.05));
        });

        ui.separator();
        ui.label("Half-extents:");
        ui.horizontal(|ui| {
            ui.label("X:");
            ui.add(
                egui::DragValue::new(&mut self.clipvol_box_half_extents[0])
                    .speed(0.05)
                    .range(0.1..=10.0),
            );
            ui.label("Y:");
            ui.add(
                egui::DragValue::new(&mut self.clipvol_box_half_extents[1])
                    .speed(0.05)
                    .range(0.1..=10.0),
            );
            ui.label("Z:");
            ui.add(
                egui::DragValue::new(&mut self.clipvol_box_half_extents[2])
                    .speed(0.05)
                    .range(0.1..=10.0),
            );
        });

        ui.separator();
        ui.label("Yaw (degrees):");
        ui.add(egui::Slider::new(&mut self.clipvol_box_yaw, -180.0..=180.0).suffix("°"));

        ui.separator();
        ui.weak("Only fragments inside the box are kept.");
    }

    fn controls_clipvol_sphere(&mut self, ui: &mut egui::Ui) {
        ui.label("Sphere center:");
        ui.horizontal(|ui| {
            ui.label("X:");
            ui.add(egui::DragValue::new(&mut self.clipvol_sphere_center[0]).speed(0.05));
            ui.label("Y:");
            ui.add(egui::DragValue::new(&mut self.clipvol_sphere_center[1]).speed(0.05));
            ui.label("Z:");
            ui.add(egui::DragValue::new(&mut self.clipvol_sphere_center[2]).speed(0.05));
        });

        ui.separator();
        ui.label("Radius:");
        ui.add(egui::Slider::new(&mut self.clipvol_sphere_radius, 0.5..=8.0).step_by(0.1));

        ui.separator();
        ui.weak("Only fragments inside the sphere are kept.");
    }

    fn controls_clipvol_plane(&mut self, ui: &mut egui::Ui) {
        ui.label("Axis preset:");
        ui.horizontal(|ui| {
            for (label, axis) in [("X", ClipAxis::X), ("Y", ClipAxis::Y), ("Z", ClipAxis::Z)] {
                if ui.button(label).clicked() {
                    self.clipvol_plane_axis = axis;
                    let dist = if let ClipShape::Plane { distance, .. } = self.clipvol_plane.shape {
                        distance
                    } else {
                        0.0
                    };
                    let prev_color = self.clipvol_plane.color;
                    self.clipvol_plane = plane_from_axis_preset(axis, dist);
                    self.clipvol_plane.color = prev_color;
                }
            }
        });

        ui.separator();
        ui.label("Offset:");
        let mut dist = if let ClipShape::Plane { distance, .. } = self.clipvol_plane.shape {
            distance
        } else {
            0.0
        };
        if ui
            .add(egui::Slider::new(&mut dist, -6.0..=6.0).step_by(0.05))
            .changed()
        {
            if let ClipShape::Plane {
                ref mut distance, ..
            } = self.clipvol_plane.shape
            {
                *distance = dist;
            }
        }

        if ui.button("Flip Normal").clicked() {
            if let ClipShape::Plane {
                ref mut normal,
                ref mut distance,
                ..
            } = self.clipvol_plane.shape
            {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
                *distance = -*distance;
            }
        }

        ui.separator();
        ui.weak("Use the gizmo in the viewport to reposition or rotate the plane.");
    }
}

// ---------------------------------------------------------------------------
// Frame-data helper
// ---------------------------------------------------------------------------

impl App {
    /// Build a `ClipObject` for the current sub-mode (box/sphere/plane).
    pub(crate) fn make_clip_object(&self) -> Option<ClipObject> {
        let overlay_color: Option<[f32; 4]> = if self.clipvol_show_overlay {
            Some([0.45, 0.82, 1.0, 1.0])
        } else {
            None
        };

        match self.clipvol_sub_mode {
            ClipVolSubMode::BoxClip => {
                let yaw_rad = self.clipvol_box_yaw.to_radians();
                let cos_y = yaw_rad.cos();
                let sin_y = yaw_rad.sin();
                let orient = [[cos_y, sin_y, 0.0], [-sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]];
                Some({
                    let mut co = ClipObject::box_shape(
                        self.clipvol_box_center,
                        self.clipvol_box_half_extents,
                        orient,
                    );
                    co.color = overlay_color;
                    co
                })
            }
            ClipVolSubMode::SphereClip => Some({
                let mut co =
                    ClipObject::sphere(self.clipvol_sphere_center, self.clipvol_sphere_radius);
                co.color = overlay_color;
                co
            }),
            ClipVolSubMode::InteractivePlane => {
                let mut co = self.clipvol_plane;
                co.color = if self.clipvol_show_overlay {
                    Some([0.45, 0.82, 1.0, 0.5])
                } else {
                    None
                };
                Some(co)
            }
        }
    }
}
