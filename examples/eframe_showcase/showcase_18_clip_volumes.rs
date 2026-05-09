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
    ClipAxis, ClipObject, ClipShape, Gizmo, GizmoMode, Material, ViewportRenderer,
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
// State
// ---------------------------------------------------------------------------

pub(crate) struct ClipVolState {
    pub scene:            Scene,
    pub built:            bool,
    pub sub_mode:         ClipVolSubMode,
    pub box_center:       [f32; 3],
    pub box_half_extents: [f32; 3],
    pub box_yaw:          f32,
    pub sphere_center:    [f32; 3],
    pub sphere_radius:    f32,
    /// The clip object used in interactive-plane sub-mode.
    pub plane:            ClipObject,
    pub plane_axis:       ClipAxis,
    /// Gizmo state for clip volume manipulation.
    pub gizmo:            Gizmo,
    /// Cached gizmo center for hit-testing (updated end of each frame).
    pub gizmo_center:     Option<glam::Vec3>,
    /// Gizmo screen-space scale (updated end of each frame).
    pub gizmo_scale:      f32,
    /// True while a gizmo drag is in progress (suppresses orbit).
    pub gizmo_drag_active: bool,
    /// Whether to show the clip object overlay (wireframe / fill quad).
    pub show_overlay:     bool,
}

impl Default for ClipVolState {
    fn default() -> Self {
        let mut plane = ClipObject::plane([0.0, 0.0, 1.0], 0.0);
        plane.color = Some([0.45, 0.82, 1.0, 0.5]);
        Self {
            scene:            Scene::new(),
            built:            false,
            sub_mode:         ClipVolSubMode::BoxClip,
            box_center:       [0.0; 3],
            box_half_extents: [2.5, 2.5, 2.5],
            box_yaw:          0.0,
            sphere_center:    [0.0; 3],
            sphere_radius:    2.8,
            plane,
            plane_axis:       ClipAxis::Z,
            gizmo:            Gizmo::new(),
            gizmo_center:     None,
            gizmo_scale:      1.0,
            gizmo_drag_active: false,
            show_overlay:     true,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// One-time GPU setup for Showcase 18.
    ///
    /// Uploads a dense UV sphere (48x24 segments) and a flat ground plane.
    pub(crate) fn build_clipvol_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.clipvol_state.scene = Scene::new();

        // Dense sphere : many vertices make the clip cross-section visually rich.
        let sphere_mesh = viewport_lib::primitives::sphere(3.0, 48, 24);
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere_mesh)
            .expect("clipvol sphere mesh");
        self.clipvol_state.scene.add_named(
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
        let ground_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &ground_mesh)
            .expect("clipvol ground mesh");
        self.clipvol_state.scene.add_named(
            "Ground",
            Some(ground_id),
            glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -3.05)),
            {
                let mut m = Material::from_color([0.38, 0.38, 0.40]);
                m.roughness = 0.9;
                m
            },
        );

        self.clipvol_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_clipvol(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.clipvol_state;

    ui.label("Clip mode:");
    ui.horizontal(|ui| {
        if ui.radio(s.sub_mode == ClipVolSubMode::BoxClip, "Box").clicked() {
            s.sub_mode = ClipVolSubMode::BoxClip;
            // Scale is valid for box; keep mode as-is.
        }
        if ui.radio(s.sub_mode == ClipVolSubMode::SphereClip, "Sphere").clicked() {
            s.sub_mode = ClipVolSubMode::SphereClip;
            // Sphere has no Rotate mode.
            if s.gizmo.mode == GizmoMode::Rotate {
                s.gizmo.mode = GizmoMode::Translate;
            }
        }
        if ui.radio(s.sub_mode == ClipVolSubMode::InteractivePlane, "Plane").clicked() {
            s.sub_mode = ClipVolSubMode::InteractivePlane;
            // Plane has no Scale mode.
            if s.gizmo.mode == GizmoMode::Scale {
                s.gizmo.mode = GizmoMode::Translate;
            }
        }
    });

    ui.separator();

    // ----- Gizmo mode + overlay -----
    ui.label("Gizmo:");
    ui.horizontal(|ui| {
        if ui.radio(s.gizmo.mode == GizmoMode::Translate, "Translate").clicked() {
            s.gizmo.mode = GizmoMode::Translate;
        }
        let can_rotate = s.sub_mode != ClipVolSubMode::SphereClip;
        ui.add_enabled_ui(can_rotate, |ui| {
            if ui.radio(s.gizmo.mode == GizmoMode::Rotate, "Rotate").clicked() {
                s.gizmo.mode = GizmoMode::Rotate;
            }
        });
        let can_scale = s.sub_mode != ClipVolSubMode::InteractivePlane;
        ui.add_enabled_ui(can_scale, |ui| {
            if ui.radio(s.gizmo.mode == GizmoMode::Scale, "Scale").clicked() {
                s.gizmo.mode = GizmoMode::Scale;
            }
        });
    });
    ui.checkbox(&mut s.show_overlay, "Show overlay");

    ui.separator();

    match s.sub_mode {
        ClipVolSubMode::BoxClip => controls_clipvol_box(s, ui),
        ClipVolSubMode::SphereClip => controls_clipvol_sphere(s, ui),
        ClipVolSubMode::InteractivePlane => controls_clipvol_plane(s, ui),
    }
}

fn controls_clipvol_box(s: &mut ClipVolState, ui: &mut egui::Ui) {
    ui.label("Box center:");
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut s.box_center[0]).speed(0.05));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut s.box_center[1]).speed(0.05));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut s.box_center[2]).speed(0.05));
    });

    ui.separator();
    ui.label("Half-extents:");
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut s.box_half_extents[0]).speed(0.05).range(0.1..=10.0));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut s.box_half_extents[1]).speed(0.05).range(0.1..=10.0));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut s.box_half_extents[2]).speed(0.05).range(0.1..=10.0));
    });

    ui.separator();
    ui.label("Yaw (degrees):");
    ui.add(egui::Slider::new(&mut s.box_yaw, -180.0..=180.0).suffix("°"));

    ui.separator();
    ui.weak("Only fragments inside the box are kept.");
}

fn controls_clipvol_sphere(s: &mut ClipVolState, ui: &mut egui::Ui) {
    ui.label("Sphere center:");
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut s.sphere_center[0]).speed(0.05));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut s.sphere_center[1]).speed(0.05));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut s.sphere_center[2]).speed(0.05));
    });

    ui.separator();
    ui.label("Radius:");
    ui.add(egui::Slider::new(&mut s.sphere_radius, 0.5..=8.0).step_by(0.1));

    ui.separator();
    ui.weak("Only fragments inside the sphere are kept.");
}

fn controls_clipvol_plane(s: &mut ClipVolState, ui: &mut egui::Ui) {
    ui.label("Axis preset:");
    ui.horizontal(|ui| {
        for (label, axis) in [("X", ClipAxis::X), ("Y", ClipAxis::Y), ("Z", ClipAxis::Z)] {
            if ui.button(label).clicked() {
                s.plane_axis = axis;
                let dist = if let ClipShape::Plane { distance, .. } = s.plane.shape {
                    distance
                } else {
                    0.0
                };
                let prev_color = s.plane.color;
                s.plane = plane_from_axis_preset(axis, dist);
                s.plane.color = prev_color;
            }
        }
    });

    ui.separator();
    ui.label("Offset:");
    let mut dist = if let ClipShape::Plane { distance, .. } = s.plane.shape {
        distance
    } else {
        0.0
    };
    if ui.add(egui::Slider::new(&mut dist, -6.0..=6.0).step_by(0.05)).changed() {
        if let ClipShape::Plane { ref mut distance, .. } = s.plane.shape {
            *distance = dist;
        }
    }

    if ui.button("Flip Normal").clicked() {
        if let ClipShape::Plane {
            ref mut normal,
            ref mut distance,
            ..
        } = s.plane.shape
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

// ---------------------------------------------------------------------------
// Frame-data helper
// ---------------------------------------------------------------------------

impl App {
    /// Build a `ClipObject` for the current sub-mode (box/sphere/plane).
    pub(crate) fn make_clip_object(&self) -> Option<ClipObject> {
        let s = &self.clipvol_state;
        let overlay_color: Option<[f32; 4]> = if s.show_overlay {
            Some([0.45, 0.82, 1.0, 1.0])
        } else {
            None
        };

        match s.sub_mode {
            ClipVolSubMode::BoxClip => {
                let yaw_rad = s.box_yaw.to_radians();
                let cos_y = yaw_rad.cos();
                let sin_y = yaw_rad.sin();
                let orient = [[cos_y, sin_y, 0.0], [-sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]];
                Some({
                    let mut co = ClipObject::box_shape(s.box_center, s.box_half_extents, orient);
                    co.color = overlay_color;
                    co
                })
            }
            ClipVolSubMode::SphereClip => Some({
                let mut co = ClipObject::sphere(s.sphere_center, s.sphere_radius);
                co.color = overlay_color;
                co
            }),
            ClipVolSubMode::InteractivePlane => {
                let mut co = s.plane;
                co.color = if s.show_overlay {
                    Some([0.45, 0.82, 1.0, 0.5])
                } else {
                    None
                };
                Some(co)
            }
        }
    }
}
