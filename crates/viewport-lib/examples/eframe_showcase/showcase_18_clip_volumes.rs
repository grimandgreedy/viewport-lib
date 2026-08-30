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
use viewport_lib as vpl;
use vpl::{
    AlphaMode, BackfacePolicy, BuiltinColourmap, ClipObject, ColourmapId, FrameData, Gizmo,
    ItemSettings, LightKind, LightSource, LightingSettings, Material, MeshId, SceneRenderItem,
    ViewportRenderer, VolumeId, VolumeItem, scene::Scene, selection::Selection,
};

/// Colour for the clip-object outlines and the plane fill.
const CLIP_COLOUR: [f32; 3] = [0.45, 0.82, 1.0];

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
    /// Unit quad mesh reused for every plane's translucent fill (placed per-frame).
    pub fill_mesh_id: Option<MeshId>,
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
            fill_mesh_id: None,
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
        let torus_mesh = vpl::primitives::torus(2.2, 0.65, 64, 32);
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
        let capsule_mesh = vpl::primitives::capsule(0.75, 2.8, 32, 12);
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
        // The field is 64^3 in normalized [-1,1]^3 space mapped to +/-3.5 world units,
        // so the volume torus ring sits at the same radius as the mesh torus.
        let vol_data = build_clipvol_volume();
        let vol_id = renderer.resources_mut().upload_volume(
            &self.device,
            &self.queue,
            &vol_data,
            [64, 64, 64],
        );
        self.clipvol_state.volume_id = Some(vol_id);

        // Unit quad reused for every clip plane's translucent fill. It is placed
        // onto each plane per-frame with `visual::plane_fill_transform`.
        let fill_mesh = vpl::clip_plane::visual::fill_quad_mesh();
        let fill_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &fill_mesh)
            .expect("clipvol fill quad mesh");
        self.clipvol_state.fill_mesh_id = Some(fill_id);

        self.clipvol_state.built = true;
    }
}

/// Generate a 64^3 density field approximating a torus + capsule for the volume mode.
///
/// Normalized coordinates are in [-1, 1]^3 and map to +/-3.5 world units, so
/// the torus ring at normalized r=0.63 appears at world r~2.2, matching the mesh.
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
            .suffix(" deg")
            .step_by(1.0),
    );
    ui.label("Azimuth:");
    ui.add(
        egui::Slider::new(azimuth, -180.0..=180.0)
            .suffix(" deg")
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
    ui.add(egui::Slider::new(yaw, -180.0..=180.0).suffix(" deg"));
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
        ui.add(egui::Slider::new(axis_yaw, -180.0..=180.0).suffix(" deg yaw"));
        ui.add(egui::Slider::new(axis_pitch, -90.0..=90.0).suffix(" deg pitch"));
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
        // +/-3.5 world units : normalized [-1,1] maps to this bbox.
        item.bbox_min = [-3.5, -3.5, -3.5];
        item.bbox_max = [3.5, 3.5, 3.5];
        Some(item)
    }

    /// Build `ClipObject`s for all active clips. These carry only the geometry
    /// (the renderer no longer draws a boundary from them); the outlines and the
    /// plane fill are built separately from `clip_plane::visual` and submitted as
    /// ordinary scene primitives.
    pub(crate) fn make_clip_objects(&self) -> Vec<ClipObject> {
        self.clipvol_state
            .clips
            .iter()
            .map(|clip| match clip {
                ActiveClip::Plane {
                    elevation,
                    azimuth,
                    distance,
                } => {
                    let normal = plane_normal(*elevation, *azimuth);
                    ClipObject::plane(normal, *distance)
                }
                ActiveClip::Box {
                    center,
                    half_extents,
                    yaw,
                } => {
                    let yaw_rad = yaw.to_radians();
                    let (sin_y, cos_y) = yaw_rad.sin_cos();
                    let orient = [[cos_y, sin_y, 0.0], [-sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]];
                    ClipObject::box_shape(*center, *half_extents, orient)
                }
                ActiveClip::Sphere { center, radius } => ClipObject::sphere(*center, *radius),
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
                    ClipObject::cylinder(*center, axis, *radius, *half_length)
                }
            })
            .collect()
    }

    /// Translucent fill mesh items for every enabled clip plane. The unit quad
    /// (`fill_mesh_id`) is placed onto each plane and tagged `ignore_clip` so it
    /// stays visible where the scene is cut.
    fn clip_plane_fill_items(&self) -> Vec<SceneRenderItem> {
        let Some(mesh_id) = self.clipvol_state.fill_mesh_id else {
            return Vec::new();
        };
        if !self.clipvol_state.show_overlay {
            return Vec::new();
        }
        let mut out = Vec::new();
        for co in self.make_clip_objects().iter().filter(|c| c.enabled) {
            if let vpl::ClipShape::Plane {
                normal, distance, ..
            } = co.shape
            {
                let center = (glam::Vec3::from(normal).normalize_or_zero() * -distance).to_array();
                let model = vpl::clip_plane::visual::plane_fill_transform(center, normal, co.extent);
                let mut material = Material::from_colour(CLIP_COLOUR);
                material.alpha_mode = AlphaMode::Blend;
                material.backface_policy = BackfacePolicy::Identical;
                let mut settings = ItemSettings::default();
                settings.unlit = true;
                settings.opacity = 0.15;
                settings.ignore_clip = true;
                settings.cast_shadows = false;
                settings.receive_shadows = false;
                let mut item = SceneRenderItem::default();
                item.mesh_id = mesh_id;
                item.model = model;
                item.material = material;
                item.settings = settings;
                out.push(item);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn clipvol_collect_scene_items(
    app: &mut App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64, u64) {
    let mut items = if app.clipvol_state.scene_mode == SceneMode::Mesh {
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
    // Translucent plane-fill quads (both scene modes; they only clip the scene, so
    // they read the same over the mesh and the volume).
    items.extend(app.clip_plane_fill_items());
    let sg = app.clipvol_state.scene.version();
    let lighting = {
        let mut _t = LightingSettings::default();
        _t.lights = vec![{
            let mut _t = LightSource::default();
            _t.kind = LightKind::Directional {
                direction: [0.5, 0.3, 1.2],
            };
            _t.intensity = 1.8;
            _t
        }];
        _t.hemisphere_intensity = 0.4;
        _t.sky_colour = [1.0, 1.0, 1.0];
        _t.ground_colour = [0.8, 0.8, 0.8];
        _t
    };
    (items, lighting, sg, 0)
}

pub(crate) fn submit_clipvol_items(app: &mut App, fd: &mut FrameData) {
    if !app.clipvol_state.built {
        return;
    }
    let clip_objects = app.make_clip_objects();
    // Outline for every enabled clip (border + normal handle for the plane, edges
    // or rings for the volumes). Each is `ignore_clip`, so it stays visible where
    // the scene is cut. The plane's translucent fill is submitted as a scene mesh
    // in `clipvol_collect_scene_items`.
    if app.clipvol_state.show_overlay {
        let colour = [CLIP_COLOUR[0], CLIP_COLOUR[1], CLIP_COLOUR[2], 1.0];
        for co in clip_objects.iter().filter(|c| c.enabled) {
            fd.scene
                .polylines
                .push(vpl::clip_plane::visual::outline(&co.shape, co.extent, colour));
        }
    }
    fd.effects.clip.objects.extend(clip_objects);
    if app.clipvol_state.scene_mode == SceneMode::Volume {
        if let Some(vol) = app.make_clipvol_volume_item() {
            fd.scene.volumes.push(vol);
        }
    }
}
