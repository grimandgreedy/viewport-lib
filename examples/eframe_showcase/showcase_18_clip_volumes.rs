//! Showcase 18: Clip Objects
//!
//! Demonstrates multiple simultaneous `ClipObject` entries on `EffectsFrame`.
//! Use the add buttons to push plane, box, sphere, or cylinder clips.
//! Each entry can be tuned independently and removed. All active clips
//! apply with AND semantics: geometry must be inside every volume.
//!
//! The scene is a torus lying flat with a capsule standing upright through its
//! hole. Cross-sections through either shape reveal their internal geometry.

use crate::App;
use eframe::egui;
use viewport_lib::{ClipObject, Gizmo, Material, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// Active clip entry
// ---------------------------------------------------------------------------

pub(crate) enum ActiveClip {
    Plane { normal: [f32; 3], distance: f32 },
    Box { center: [f32; 3], half_extents: [f32; 3], yaw: f32 },
    Sphere { center: [f32; 3], radius: f32 },
    Cylinder { center: [f32; 3], axis_yaw: f32, axis_pitch: f32, radius: f32, half_length: f32 },
}

impl ActiveClip {
    fn label(&self) -> &'static str {
        match self {
            Self::Plane { .. } => "Plane",
            Self::Box { .. } => "Box",
            Self::Sphere { .. } => "Sphere",
            Self::Cylinder { .. } => "Cylinder",
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ClipVolState {
    pub scene:             Scene,
    pub built:             bool,
    pub clips:             Vec<ActiveClip>,
    pub show_overlay:      bool,
    /// Gizmo retained for main.rs compatibility (not used for clip editing).
    pub gizmo:             Gizmo,
    pub gizmo_center:      Option<glam::Vec3>,
    pub gizmo_scale:       f32,
    pub gizmo_drag_active: bool,
}

impl Default for ClipVolState {
    fn default() -> Self {
        Self {
            scene:             Scene::new(),
            built:             false,
            clips:             vec![
                ActiveClip::Plane { normal: [0.0, 0.0, 1.0], distance: 0.0 },
            ],
            show_overlay:      true,
            gizmo:             Gizmo::new(),
            gizmo_center:      None,
            gizmo_scale:       1.0,
            gizmo_drag_active: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// One-time GPU setup for Showcase 18.
    ///
    /// Uploads a torus (lying flat) with a capsule standing upright through its
    /// hole. Cross-sections reveal the curved interior of both shapes.
    pub(crate) fn build_clipvol_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.clipvol_state.scene = Scene::new();

        // Torus lying in the XY plane. High segment count makes clip cross-sections smooth.
        let torus_mesh = viewport_lib::primitives::torus(2.2, 0.65, 64, 32);
        let torus_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &torus_mesh)
            .expect("clipvol torus mesh");
        self.clipvol_state.scene.add_named(
            "Torus",
            Some(torus_id),
            glam::Mat4::IDENTITY,
            {
                let mut m = Material::from_color([0.82, 0.42, 0.18]);
                m.roughness = 0.35;
                m.metallic = 0.15;
                m
            },
        );

        // Capsule standing upright through the torus hole.
        // Torus hole radius = major - minor = 2.2 - 0.65 = 1.55; capsule radius 0.75 fits easily.
        let capsule_mesh = viewport_lib::primitives::capsule(0.75, 2.8, 32, 12);
        let capsule_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &capsule_mesh)
            .expect("clipvol capsule mesh");
        self.clipvol_state.scene.add_named(
            "Capsule",
            Some(capsule_id),
            glam::Mat4::IDENTITY,
            {
                let mut m = Material::from_color([0.28, 0.58, 0.92]);
                m.roughness = 0.25;
                m.metallic = 0.35;
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

    ui.label("Add clip:");
    ui.horizontal(|ui| {
        if ui.button("+ Plane").clicked() {
            s.clips.push(ActiveClip::Plane { normal: [0.0, 0.0, 1.0], distance: 0.0 });
        }
        if ui.button("+ Box").clicked() {
            s.clips.push(ActiveClip::Box {
                center:       [0.0; 3],
                half_extents: [2.5, 2.5, 2.5],
                yaw:          0.0,
            });
        }
        if ui.button("+ Sphere").clicked() {
            s.clips.push(ActiveClip::Sphere { center: [0.0; 3], radius: 2.8 });
        }
        if ui.button("+ Cylinder").clicked() {
            s.clips.push(ActiveClip::Cylinder {
                center:     [0.0; 3],
                axis_yaw:   0.0,
                axis_pitch: 0.0,
                radius:     1.5,
                half_length: 3.0,
            });
        }
    });

    ui.checkbox(&mut s.show_overlay, "Show overlay");

    if s.clips.is_empty() {
        ui.separator();
        ui.weak("No clips active. Add one above.");
        return;
    }

    let mut remove_indices: Vec<usize> = Vec::new();

    for (i, clip) in s.clips.iter_mut().enumerate() {
        ui.separator();
        ui.horizontal(|ui| {
            ui.strong(format!("{}. {}", i + 1, clip.label()));
            if ui.button("Remove").clicked() {
                remove_indices.push(i);
            }
        });

        match clip {
            ActiveClip::Plane { normal, distance } => {
                controls_plane(ui, normal, distance);
            }
            ActiveClip::Box { center, half_extents, yaw } => {
                controls_box(ui, center, half_extents, yaw);
            }
            ActiveClip::Sphere { center, radius } => {
                controls_sphere(ui, center, radius);
            }
            ActiveClip::Cylinder { center, axis_yaw, axis_pitch, radius, half_length } => {
                controls_cylinder(ui, center, axis_yaw, axis_pitch, radius, half_length);
            }
        }
    }

    for i in remove_indices.into_iter().rev() {
        s.clips.remove(i);
    }
}

fn controls_plane(ui: &mut egui::Ui, normal: &mut [f32; 3], distance: &mut f32) {
    ui.label("Axis preset:");
    ui.horizontal(|ui| {
        if ui.button("X").clicked() { *normal = [1.0, 0.0, 0.0]; }
        if ui.button("Y").clicked() { *normal = [0.0, 1.0, 0.0]; }
        if ui.button("Z").clicked() { *normal = [0.0, 0.0, 1.0]; }
    });
    ui.label("Offset:");
    ui.add(egui::Slider::new(distance, -6.0..=6.0).step_by(0.05));
    if ui.button("Flip Normal").clicked() {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
        *distance = -*distance;
    }
}

fn controls_box(
    ui: &mut egui::Ui,
    center: &mut [f32; 3],
    half_extents: &mut [f32; 3],
    yaw: &mut f32,
) {
    ui.label("Center:");
    ui.horizontal(|ui| {
        ui.label("X:"); ui.add(egui::DragValue::new(&mut center[0]).speed(0.05));
        ui.label("Y:"); ui.add(egui::DragValue::new(&mut center[1]).speed(0.05));
        ui.label("Z:"); ui.add(egui::DragValue::new(&mut center[2]).speed(0.05));
    });
    ui.label("Half-extents:");
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut half_extents[0]).speed(0.05).range(0.1..=10.0));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut half_extents[1]).speed(0.05).range(0.1..=10.0));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut half_extents[2]).speed(0.05).range(0.1..=10.0));
    });
    ui.label("Yaw:");
    ui.add(egui::Slider::new(yaw, -180.0..=180.0).suffix("°"));
}

fn controls_sphere(ui: &mut egui::Ui, center: &mut [f32; 3], radius: &mut f32) {
    ui.label("Center:");
    ui.horizontal(|ui| {
        ui.label("X:"); ui.add(egui::DragValue::new(&mut center[0]).speed(0.05));
        ui.label("Y:"); ui.add(egui::DragValue::new(&mut center[1]).speed(0.05));
        ui.label("Z:"); ui.add(egui::DragValue::new(&mut center[2]).speed(0.05));
    });
    ui.label("Radius:");
    ui.add(egui::Slider::new(radius, 0.5..=8.0).step_by(0.1));
}

fn controls_cylinder(
    ui: &mut egui::Ui,
    center: &mut [f32; 3],
    axis_yaw: &mut f32,
    axis_pitch: &mut f32,
    radius: &mut f32,
    half_length: &mut f32,
) {
    ui.label("Center:");
    ui.horizontal(|ui| {
        ui.label("X:"); ui.add(egui::DragValue::new(&mut center[0]).speed(0.05));
        ui.label("Y:"); ui.add(egui::DragValue::new(&mut center[1]).speed(0.05));
        ui.label("Z:"); ui.add(egui::DragValue::new(&mut center[2]).speed(0.05));
    });
    ui.label("Axis (yaw / pitch):");
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(axis_yaw, -180.0..=180.0).suffix("° yaw"));
        ui.add(egui::Slider::new(axis_pitch, -90.0..=90.0).suffix("° pitch"));
    });
    ui.label("Radius:");
    ui.add(egui::Slider::new(radius, 0.1..=8.0).step_by(0.05));
    ui.label("Half-length:");
    ui.add(egui::Slider::new(half_length, 0.1..=10.0).step_by(0.05));
}

// ---------------------------------------------------------------------------
// Frame-data helper
// ---------------------------------------------------------------------------

impl App {
    /// Build `ClipObject`s for all active clips.
    pub(crate) fn make_clip_objects(&self) -> Vec<ClipObject> {
        let s = &self.clipvol_state;
        let overlay: Option<[f32; 4]> = if s.show_overlay { Some([0.45, 0.82, 1.0, 1.0]) } else { None };
        let plane_overlay: Option<[f32; 4]> = if s.show_overlay { Some([0.45, 0.82, 1.0, 0.5]) } else { None };

        s.clips.iter().map(|clip| match clip {
            ActiveClip::Plane { normal, distance } => {
                let mut co = ClipObject::plane(*normal, *distance);
                co.color = plane_overlay;
                co
            }
            ActiveClip::Box { center, half_extents, yaw } => {
                let yaw_rad = yaw.to_radians();
                let (sin_y, cos_y) = yaw_rad.sin_cos();
                let orient = [[cos_y, sin_y, 0.0], [-sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]];
                let mut co = ClipObject::box_shape(*center, *half_extents, orient);
                co.color = overlay;
                co
            }
            ActiveClip::Sphere { center, radius } => {
                let mut co = ClipObject::sphere(*center, *radius);
                co.color = overlay;
                co
            }
            ActiveClip::Cylinder { center, axis_yaw, axis_pitch, radius, half_length } => {
                let (sy, cy) = axis_yaw.to_radians().sin_cos();
                let (sp, cp) = axis_pitch.to_radians().sin_cos();
                let axis = [cp * cy, cp * sy, sp];
                let mut co = ClipObject::cylinder(*center, axis, *radius, *half_length);
                co.color = overlay;
                co
            }
        }).collect()
    }
}
