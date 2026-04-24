//! Showcase 18: Clip Volumes & Interactive Clip Plane
//!
//! Demonstrates `ClipVolume::Box` and `ClipVolume::Sphere` on `EffectsFrame`, and
//! the `ClipPlaneController` interactive drag tool. A dense UV sphere (high vertex
//! count) makes the clip cross-section visually rich.
//!
//! Sub-modes (radio):
//! - Box clip  — oriented box; center / half-extents / yaw sliders.
//! - Sphere clip — sphere; center / radius sliders.
//! - Interactive plane — a `ClipPlaneController` driven by mouse drag on the handle.
//!
//! Common controls:
//! - Sub-mode selector radio buttons.

use crate::App;
use eframe::egui;
use viewport_lib::{
    ClipAxis, ClipPlaneContext, ClipPlaneResult, ClipVolume, Material, MeshId,
    ViewportRenderer, plane_from_axis_preset,
    scene::Scene,
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

        // Dense sphere — many vertices make the clip cross-section visually rich.
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
            Material {
                base_color: [0.55, 0.72, 0.95],
                roughness: 0.3,
                metallic: 0.1,
                ..Material::default()
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
            Material {
                base_color: [0.38, 0.38, 0.40],
                roughness: 0.9,
                metallic: 0.0,
                ..Material::default()
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
            }
            if ui
                .radio(self.clipvol_sub_mode == ClipVolSubMode::SphereClip, "Sphere")
                .clicked()
            {
                self.clipvol_sub_mode = ClipVolSubMode::SphereClip;
            }
            if ui
                .radio(
                    self.clipvol_sub_mode == ClipVolSubMode::InteractivePlane,
                    "Plane",
                )
                .clicked()
            {
                self.clipvol_sub_mode = ClipVolSubMode::InteractivePlane;
            }
        });

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
        ui.add(
            egui::Slider::new(&mut self.clipvol_sphere_radius, 0.5..=8.0)
                .step_by(0.1),
        );

        ui.separator();
        ui.weak("Only fragments inside the sphere are kept.");
    }

    fn controls_clipvol_plane(&mut self, ui: &mut egui::Ui) {
        ui.label("Axis preset:");
        ui.horizontal(|ui| {
            for (label, axis) in [
                ("X", ClipAxis::X),
                ("Y", ClipAxis::Y),
                ("Z", ClipAxis::Z),
            ] {
                if ui.button(label).clicked() {
                    self.clipvol_plane_axis = axis;
                    let dist = self.clipvol_plane.distance;
                    self.clipvol_plane = plane_from_axis_preset(axis, dist);
                    self.clipvol_plane_controller.reset();
                }
            }
        });

        ui.separator();
        ui.label("Offset:");
        if ui
            .add(egui::Slider::new(&mut self.clipvol_plane.distance, -6.0..=6.0).step_by(0.05))
            .changed()
        {
            self.clipvol_plane_controller.reset();
        }

        if ui.button("Flip Normal").clicked() {
            self.clipvol_plane.normal[0] = -self.clipvol_plane.normal[0];
            self.clipvol_plane.normal[1] = -self.clipvol_plane.normal[1];
            self.clipvol_plane.normal[2] = -self.clipvol_plane.normal[2];
            self.clipvol_plane.distance = -self.clipvol_plane.distance;
            self.clipvol_plane_controller.reset();
        }

        ui.separator();
        ui.weak("Drag the plane handle in the viewport.\nDrag arrow tip to tilt.");
        if self.clipvol_plane_controller.is_active() {
            ui.label("Dragging…");
        }
    }
}

// ---------------------------------------------------------------------------
// Frame-data helper
// ---------------------------------------------------------------------------

impl App {
    /// Build the `ClipVolume` for the current sub-mode.
    pub(crate) fn make_clipvol(&self) -> ClipVolume {
        match self.clipvol_sub_mode {
            ClipVolSubMode::BoxClip => {
                let yaw_rad = self.clipvol_box_yaw.to_radians();
                let cos_y = yaw_rad.cos();
                let sin_y = yaw_rad.sin();
                // Rotation around Z axis (yaw in XY plane).
                let orient = [
                    [cos_y, sin_y, 0.0],
                    [-sin_y, cos_y, 0.0],
                    [0.0, 0.0, 1.0],
                ];
                ClipVolume::Box {
                    center: self.clipvol_box_center,
                    half_extents: self.clipvol_box_half_extents,
                    orientation: orient,
                }
            }
            ClipVolSubMode::SphereClip => ClipVolume::Sphere {
                center: self.clipvol_sphere_center,
                radius: self.clipvol_sphere_radius,
            },
            // The ClipPlaneController places its visual handle at `normal * distance`
            // and the overlay center follows the same convention.  ClipVolume::Plane
            // keeps fragments where `dot(p, normal) + distance >= 0`, whose surface
            // is at `normal * (-distance)` — the opposite side.  Negate so the clip
            // surface lands at the same world position as the visible handle.
            ClipVolSubMode::InteractivePlane => ClipVolume::Plane {
                normal: self.clipvol_plane.normal,
                distance: -self.clipvol_plane.distance,
            },
        }
    }

    /// Update the `ClipPlaneController` for one frame (interactive plane sub-mode).
    ///
    /// Returns the overlay to push into `InteractionFrame::clip_plane_overlays`.
    pub(crate) fn update_clipvol_controller(
        &mut self,
        viewport_size: glam::Vec2,
    ) -> Option<viewport_lib::ClipPlaneOverlay> {
        let ctx = ClipPlaneContext {
            plane: self.clipvol_plane,
            camera: self.camera.clone(),
            viewport_size,
            cursor_viewport: if self.clipvol_cursor_in_viewport {
                Some(self.last_cursor_viewport)
            } else {
                None
            },
            pointer_delta: self.clipvol_pointer_delta,
            drag_started: self.clipvol_drag_started,
            dragging: self.clipvol_dragging,
            clicked: false,
            plane_extent: 4.5,
        };
        match self
            .clipvol_plane_controller
            .update(&self.clipvol_action_frame, ctx.clone())
        {
            ClipPlaneResult::Update(delta) => {
                self.clipvol_plane.distance += delta.distance_delta;
                if let Some(n) = delta.normal_override {
                    self.clipvol_plane.normal = n;
                }
            }
            ClipPlaneResult::Cancel => {}
            ClipPlaneResult::Commit => {}
            ClipPlaneResult::None => {}
        }

        self.clipvol_plane_controller.overlay(&ctx)
    }
}
