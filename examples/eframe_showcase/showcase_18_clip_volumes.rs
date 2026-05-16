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
use viewport_lib::{
    BackfacePolicy, BuiltinColourmap, ClipObject, ColourmapId, FrameData, Gizmo, LightKind,
    LightSource, LightingSettings, Material, SceneRenderItem, ViewportRenderer, VolumeId,
    VolumeItem, scene::Scene, selection::Selection,
};

// ---------------------------------------------------------------------------
// Scene mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SceneMode {
    /// Triangle mesh scene: torus + capsule.
    Mesh,
    /// Ray-marched volume: same shapes approximated as a density field.
    Volume,
}

// ---------------------------------------------------------------------------
// Active clip entry
// ---------------------------------------------------------------------------

pub(crate) enum ActiveClip {
    Plane {
        elevation: f32,
        azimuth: f32,
        distance: f32,
    },
    Box {
        center: [f32; 3],
        half_extents: [f32; 3],
        yaw: f32,
    },
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    Cylinder {
        center: [f32; 3],
        axis_yaw: f32,
        axis_pitch: f32,
        radius: f32,
        half_length: f32,
    },
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
    pub scene: Scene,
    pub built: bool,
    pub scene_mode: SceneMode,
    pub volume_id: Option<VolumeId>,
    pub clips: Vec<ActiveClip>,
    pub show_overlay: bool,
    /// Gizmo retained for main.rs compatibility (not used for clip editing).
    pub gizmo: Gizmo,
    pub gizmo_center: Option<glam::Vec3>,
    pub gizmo_scale: f32,
    pub gizmo_drag_active: bool,
}

impl Default for ClipVolState {
    fn default() -> Self {
        Self {
            scene: Scene::new(),
            built: false,
            scene_mode: SceneMode::Mesh,
            volume_id: None,
            clips: vec![ActiveClip::Plane {
                elevation: 0.0,
                azimuth: 0.0,
                distance: 0.0,
            }],
            show_overlay: true,
            gizmo: Gizmo::new(),
            gizmo_center: None,
            gizmo_scale: 1.0,
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
        self.clipvol_state
            .scene
            .add_named("Torus", Some(torus_id), glam::Mat4::IDENTITY, {
                let mut m = Material::from_colour([0.82, 0.42, 0.18]);
                m.roughness = 0.35;
                m.metallic = 0.15;
                m
            });

        // Capsule standing upright through the torus hole.
        // Torus hole radius = major - minor = 2.2 - 0.65 = 1.55; capsule radius 0.75 fits easily.
        let capsule_mesh = viewport_lib::primitives::capsule(0.75, 2.8, 32, 12);
        let capsule_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &capsule_mesh)
            .expect("clipvol capsule mesh");
        self.clipvol_state
            .scene
            .add_named("Capsule", Some(capsule_id), glam::Mat4::IDENTITY, {
                let mut m = Material::from_colour([0.28, 0.58, 0.92]);
                m.roughness = 0.25;
                m.metallic = 0.35;
                m
            });

        // Volume: a density field approximating the same torus + capsule shapes.
        // The field is 64³ in normalized [-1,1]³ space mapped to ±3.5 world units,
        // so the volume torus ring sits at the same radius as the mesh torus.
        let vol_data = build_clipvol_volume();
        let vol_id = renderer.resources_mut().upload_volume(
            &self.device,
            &self.queue,
            &vol_data,
            [64, 64, 64],
        );
        self.clipvol_state.volume_id = Some(vol_id);

        self.clipvol_state.built = true;
    }
}

/// Generate a 64³ density field approximating a torus + capsule for the volume mode.
///
/// Normalized coordinates are in [-1, 1]³ and map to ±3.5 world units, so
/// the torus ring at normalized r=0.63 appears at world r≈2.2, matching the mesh.
fn build_clipvol_volume() -> Vec<f32> {
    let n = 64usize;
    let mut data = vec![0.0f32; n * n * n];
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let x = ix as f32 / (n - 1) as f32 * 2.0 - 1.0;
                let y = iy as f32 / (n - 1) as f32 * 2.0 - 1.0;
                let z = iz as f32 / (n - 1) as f32 * 2.0 - 1.0;

                // Torus ring: peaks on a circle at rxy=0.63, z=0.
                let rxy = (x * x + y * y).sqrt();
                let torus_d2 = (rxy - 0.63).powi(2) + z.powi(2);
                let torus = (-torus_d2 / 0.028).exp();

                // Capsule: cylindrical core clamped in z, with spherical caps.
                let cz = z.clamp(-0.40, 0.40);
                let cap_d2 = x * x + y * y + (z - cz).powi(2);
                let capsule = (-cap_d2 / 0.046).exp();

                data[iz * n * n + iy * n + ix] = (torus + capsule).min(1.0);
            }
        }
    }
    data
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_clipvol(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.clipvol_state;

    ui.label("Scene:");
    ui.horizontal(|ui| {
        ui.radio_value(&mut s.scene_mode, SceneMode::Mesh, "Mesh");
        ui.radio_value(&mut s.scene_mode, SceneMode::Volume, "Volume");
    });

    ui.separator();
    ui.label("Add clip:");
    ui.horizontal(|ui| {
        if ui.button("+ Plane").clicked() {
            s.clips.push(ActiveClip::Plane {
                elevation: 0.0,
                azimuth: 0.0,
                distance: 0.0,
            });
        }
        if ui.button("+ Box").clicked() {
            s.clips.push(ActiveClip::Box {
                center: [0.0; 3],
                half_extents: [2.5, 2.5, 2.5],
                yaw: 0.0,
            });
        }
        if ui.button("+ Sphere").clicked() {
            s.clips.push(ActiveClip::Sphere {
                center: [0.0; 3],
                radius: 2.8,
            });
        }
        if ui.button("+ Cylinder").clicked() {
            s.clips.push(ActiveClip::Cylinder {
                center: [0.0; 3],
                axis_yaw: 0.0,
                axis_pitch: 0.0,
                radius: 1.5,
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
            ActiveClip::Plane {
                elevation,
                azimuth,
                distance,
            } => {
                controls_plane(ui, elevation, azimuth, distance);
            }
            ActiveClip::Box {
                center,
                half_extents,
                yaw,
            } => {
                controls_box(ui, center, half_extents, yaw);
            }
            ActiveClip::Sphere { center, radius } => {
                controls_sphere(ui, center, radius);
            }
            ActiveClip::Cylinder {
                center,
                axis_yaw,
                axis_pitch,
                radius,
                half_length,
            } => {
                controls_cylinder(ui, center, axis_yaw, axis_pitch, radius, half_length);
            }
        }
    }

    for i in remove_indices.into_iter().rev() {
        s.clips.remove(i);
    }
}

/// Compute a unit normal from elevation/azimuth angles (degrees).
/// Convention: az=0,el=0 -> +Z; az=90,el=0 -> +X; el=90 -> +Y.
fn plane_normal(elevation: f32, azimuth: f32) -> [f32; 3] {
    let el = elevation.to_radians();
    let az = azimuth.to_radians();
    let (sel, cel) = (el.sin(), el.cos());
    let (saz, caz) = (az.sin(), az.cos());
    [saz * cel, sel, caz * cel]
}

fn controls_plane(ui: &mut egui::Ui, elevation: &mut f32, azimuth: &mut f32, distance: &mut f32) {
    ui.label("Axis preset:");
    ui.horizontal(|ui| {
        // X: az=90,el=0 -> [sin90*cos0, sin0, cos90*cos0] = [1,0,0]
        if ui.button("X").clicked() {
            *azimuth = 90.0;
            *elevation = 0.0;
        }
        // Y: el=90 -> [0,1,0]
        if ui.button("Y").clicked() {
            *azimuth = 0.0;
            *elevation = 90.0;
        }
        // Z: az=0,el=0 -> [0,0,1]
        if ui.button("Z").clicked() {
            *azimuth = 0.0;
            *elevation = 0.0;
        }
    });
    ui.label("Elevation:");
    ui.add(
        egui::Slider::new(elevation, -89.0..=89.0)
            .suffix("°")
            .step_by(1.0),
    );
    ui.label("Azimuth:");
    ui.add(
        egui::Slider::new(azimuth, -180.0..=180.0)
            .suffix("°")
            .step_by(1.0),
    );
    ui.label("Offset:");
    ui.add(egui::Slider::new(distance, -6.0..=6.0).step_by(0.05));
    if ui.button("Flip Normal").clicked() {
        *elevation = -*elevation;
        *azimuth = if *azimuth >= 0.0 {
            *azimuth - 180.0
        } else {
            *azimuth + 180.0
        };
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
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut center[0]).speed(0.05));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut center[1]).speed(0.05));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut center[2]).speed(0.05));
    });
    ui.label("Half-extents:");
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(
            egui::DragValue::new(&mut half_extents[0])
                .speed(0.05)
                .range(0.1..=10.0),
        );
        ui.label("Y:");
        ui.add(
            egui::DragValue::new(&mut half_extents[1])
                .speed(0.05)
                .range(0.1..=10.0),
        );
        ui.label("Z:");
        ui.add(
            egui::DragValue::new(&mut half_extents[2])
                .speed(0.05)
                .range(0.1..=10.0),
        );
    });
    ui.label("Yaw:");
    ui.add(egui::Slider::new(yaw, -180.0..=180.0).suffix("°"));
}

fn controls_sphere(ui: &mut egui::Ui, center: &mut [f32; 3], radius: &mut f32) {
    ui.label("Center:");
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut center[0]).speed(0.05));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut center[1]).speed(0.05));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut center[2]).speed(0.05));
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
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut center[0]).speed(0.05));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut center[1]).speed(0.05));
        ui.label("Z:");
        ui.add(egui::DragValue::new(&mut center[2]).speed(0.05));
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
    /// Build a `VolumeItem` for the volume scene mode.
    pub(crate) fn make_clipvol_volume_item(&self) -> Option<VolumeItem> {
        let vol_id = self.clipvol_state.volume_id?;
        let mut item = VolumeItem::default();
        item.volume_id = vol_id;
        item.colour_lut = Some(ColourmapId(BuiltinColourmap::Turbo as usize));
        item.opacity_scale = 1.0;
        item.scalar_range = (0.0, 1.0);
        item.threshold_min = 0.05;
        item.threshold_max = 1.0;
        item.step_scale = 1.0;
        item.enable_shading = true;
        // ±3.5 world units : normalized [-1,1] maps to this bbox.
        item.bbox_min = [-3.5, -3.5, -3.5];
        item.bbox_max = [3.5, 3.5, 3.5];
        Some(item)
    }

    /// Build `ClipObject`s for all active clips.
    pub(crate) fn make_clip_objects(&self) -> Vec<ClipObject> {
        let s = &self.clipvol_state;
        let overlay: Option<[f32; 4]> = if s.show_overlay {
            Some([0.45, 0.82, 1.0, 1.0])
        } else {
            None
        };
        let plane_overlay: Option<[f32; 4]> = if s.show_overlay {
            Some([0.45, 0.82, 1.0, 0.5])
        } else {
            None
        };

        s.clips
            .iter()
            .map(|clip| match clip {
                ActiveClip::Plane {
                    elevation,
                    azimuth,
                    distance,
                } => {
                    let normal = plane_normal(*elevation, *azimuth);
                    let mut co = ClipObject::plane(normal, *distance);
                    co.colour = plane_overlay;
                    co
                }
                ActiveClip::Box {
                    center,
                    half_extents,
                    yaw,
                } => {
                    let yaw_rad = yaw.to_radians();
                    let (sin_y, cos_y) = yaw_rad.sin_cos();
                    let orient = [[cos_y, sin_y, 0.0], [-sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]];
                    let mut co = ClipObject::box_shape(*center, *half_extents, orient);
                    co.colour = overlay;
                    co
                }
                ActiveClip::Sphere { center, radius } => {
                    let mut co = ClipObject::sphere(*center, *radius);
                    co.colour = overlay;
                    co
                }
                ActiveClip::Cylinder {
                    center,
                    axis_yaw,
                    axis_pitch,
                    radius,
                    half_length,
                } => {
                    let (sy, cy) = axis_yaw.to_radians().sin_cos();
                    let (sp, cp) = axis_pitch.to_radians().sin_cos();
                    let axis = [cp * cy, cp * sy, sp];
                    let mut co = ClipObject::cylinder(*center, axis, *radius, *half_length);
                    co.colour = overlay;
                    co
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn clipvol_collect_scene_items(
    app: &mut App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64, u64) {
    let items = if app.clipvol_state.scene_mode == SceneMode::Mesh {
        let mut items = app
            .clipvol_state
            .scene
            .collect_render_items(&Selection::new());
        for item in items.iter_mut() {
            item.material.backface_policy = BackfacePolicy::Identical;
        }
        items
    } else {
        Vec::new()
    };
    let sg = app.clipvol_state.scene.version();
    let lighting = LightingSettings {
        lights: vec![LightSource {
            kind: LightKind::Directional {
                direction: [0.5, 0.3, 1.2],
            },
            intensity: 1.8,
            ..LightSource::default()
        }],
        hemisphere_intensity: 0.4,
        sky_colour: [1.0, 1.0, 1.0],
        ground_colour: [0.8, 0.8, 0.8],
        ..LightingSettings::default()
    };
    (items, lighting, sg, 0)
}

pub(crate) fn submit_clipvol_items(app: &mut App, fd: &mut FrameData) {
    if !app.clipvol_state.built {
        return;
    }
    fd.effects.clip_objects.extend(app.make_clip_objects());
    if app.clipvol_state.scene_mode == SceneMode::Volume {
        if let Some(vol) = app.make_clipvol_volume_item() {
            fd.scene.volumes.push(vol);
        }
    }
}
