//! Showcase 12: Scalar Fields & Colourmaps : build and controls.
//!
//! Three objects each carrying a different procedural scalar attribute:
//!   Object 0 : Sphere,    attribute "height"   (world Z of each vertex, 0..1 range)
//!   Object 1 : Wave Grid, attribute "wave"      (sine-derived 2-D wave, -1..1 range)
//!   Object 2 : Box,       attribute "distance"  (distance from center, with NaN below 0.3)

use crate::App;
use eframe::egui;
use viewport_lib as vpl;
use vpl::{
    AttributeData, BuiltinColourmap, Material, MeshData, MeshId, NodeId, Selection,
    ViewportRenderer, scene::Scene,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ScalarFieldsState {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub colourmap: BuiltinColourmap,
    pub range_auto: bool,
    pub range: (f32, f32),
    pub nan_on: bool,
    pub node_ids: [NodeId; 3],
    pub mesh_indices: [MeshId; 3],
    pub pick_positions: [Vec<[f32; 3]>; 3],
    pub pick_indices: [Vec<u32>; 3],
    pub values: [Vec<f32>; 3],
    pub active_object: usize,
}

impl Default for ScalarFieldsState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            selection: Selection::new(),
            colourmap: BuiltinColourmap::Viridis,
            range_auto: true,
            range: (0.0, 1.0),
            nan_on: false,
            node_ids: [0; 3],
            mesh_indices: [MeshId::INVALID; 3],
            pick_positions: [Vec::new(), Vec::new(), Vec::new()],
            pick_indices: [Vec::new(), Vec::new(), Vec::new()],
            values: [Vec::new(), Vec::new(), Vec::new()],
            active_object: 0,
        }
    }
}

impl ScalarFieldsState {
    pub(crate) fn set_active_object(&mut self, index: usize) {
        self.active_object = index;
        self.range_auto = true;

        let node_id = self.node_ids[index];
        if node_id != 0 {
            self.selection.select_one(node_id);
        } else {
            self.selection.clear();
        }

        if !self.values[index].is_empty() {
            let min = self.values[index]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let max = self.values[index]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            self.range = (min, max);
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 12 (Scalar Fields demo).
    pub(crate) fn build_scalar_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.scalar_state.scene = Scene::new();

        // ---- Object 0: Sphere with height (z) scalar ----
        let mut sphere = vpl::primitives::sphere(3.0, 48, 24);
        let height_scalars: Vec<f32> = sphere
            .positions
            .iter()
            .map(|p| (p[2] + 3.0) / 6.0) // normalize z from [-3,3] -> [0,1]
            .collect();
        sphere.attributes.insert(
            "height".to_string(),
            AttributeData::Vertex(height_scalars.clone()),
        );
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("scalar sphere mesh");
        let sphere_node = self.scalar_state.scene.add_named(
            "Sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(-6.0, 0.0, 0.0)),
            {
                let mut m = Material::from_colour([0.8, 0.8, 0.8]);
                m.roughness = 0.5;
                m
            },
        );
        self.scalar_state.node_ids[0] = sphere_node;
        self.scalar_state.pick_positions[0] = sphere.positions.clone();
        self.scalar_state.pick_indices[0] = sphere.indices.clone();
        self.scalar_state.values[0] = height_scalars;

        // ---- Object 1: Wave grid with 2-D sine wave scalar ----
        let (wave_mesh, wave_scalars) = make_wave_grid(20, 20, 8.0);
        let wave_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &wave_mesh)
            .expect("scalar wave mesh");
        let wave_node =
            self.scalar_state
                .scene
                .add_named("Wave Grid", Some(wave_id), glam::Mat4::IDENTITY, {
                    let mut m = Material::from_colour([0.8, 0.8, 0.8]);
                    m.roughness = 0.5;
                    m
                });
        self.scalar_state.node_ids[1] = wave_node;
        self.scalar_state.pick_positions[1] = wave_mesh.positions.clone();
        self.scalar_state.pick_indices[1] = wave_mesh.indices.clone();
        self.scalar_state.values[1] = wave_scalars;

        // ---- Object 2: Box with distance-from-center scalar (NaN below threshold) ----
        let (box_mesh, box_scalars) = make_box_with_distance_scalar();
        let box_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &box_mesh)
            .expect("scalar box mesh");
        let box_node = self.scalar_state.scene.add_named(
            "Distance Box",
            Some(box_id),
            glam::Mat4::from_translation(glam::Vec3::new(6.0, 0.0, 0.0)),
            {
                let mut m = Material::from_colour([0.8, 0.8, 0.8]);
                m.roughness = 0.5;
                m
            },
        );
        self.scalar_state.node_ids[2] = box_node;
        self.scalar_state.pick_positions[2] = box_mesh.positions.clone();
        self.scalar_state.pick_indices[2] = box_mesh.indices.clone();
        self.scalar_state.values[2] = box_scalars;

        // Store mesh indices for scalar-range auto-computation.
        self.scalar_state.mesh_indices = [sphere_id, wave_id, box_id];
        let active = self.scalar_state.active_object.min(2);
        self.scalar_state.set_active_object(active);

        self.scalar_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_scalar_fields(app: &mut App, ui: &mut egui::Ui) {
    let s = &mut app.scalar_state;

    ui.label("Object:");
    let mut new_active: Option<usize> = None;
    for (i, label) in ["0: Sphere (height)", "1: Wave Grid", "2: Box (distance)"]
        .iter()
        .enumerate()
    {
        if ui.radio(s.active_object == i, *label).clicked() {
            new_active = Some(i);
        }
    }
    if let Some(i) = new_active {
        app.scalar_state.set_active_object(i);
    }

    let s = &mut app.scalar_state;

    ui.separator();
    ui.label("Colourmap:");
    for (preset, label) in [
        (BuiltinColourmap::Viridis, "Viridis"),
        (BuiltinColourmap::Plasma, "Plasma"),
        (BuiltinColourmap::Greyscale, "Greyscale"),
        (BuiltinColourmap::Coolwarm, "Coolwarm"),
        (BuiltinColourmap::Rainbow, "Rainbow"),
    ] {
        if ui.radio(s.colourmap == preset, label).clicked() {
            s.colourmap = preset;
        }
    }

    ui.separator();
    ui.checkbox(&mut s.range_auto, "Auto Range");
    if !s.range_auto {
        ui.horizontal(|ui| {
            ui.label("Min:");
            ui.add(egui::DragValue::new(&mut s.range.0).speed(0.01));
        });
        ui.horizontal(|ui| {
            ui.label("Max:");
            ui.add(egui::DragValue::new(&mut s.range.1).speed(0.01));
        });
    } else {
        let i = s.active_object;
        if !s.values[i].is_empty() {
            let min = s.values[i].iter().cloned().fold(f32::INFINITY, f32::min);
            let max = s.values[i]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            ui.label(format!("Range: [{min:.2}, {max:.2}]"));
        }
    }

    ui.separator();
    ui.checkbox(&mut s.nan_on, "Show NaN colour (purple)");
    ui.label("(box object: values < threshold set to NaN)");
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Build a wave-function grid mesh with per-vertex "wave" scalar attribute.
fn make_wave_grid(cols: u32, rows: u32, size: f32) -> (MeshData, Vec<f32>) {
    let nx = cols + 1;
    let ny = rows + 1;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((nx * ny) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((nx * ny) as usize);
    let mut scalars: Vec<f32> = Vec::with_capacity((nx * ny) as usize);

    for iy in 0..ny {
        for ix in 0..nx {
            let u = ix as f32 / cols as f32; // 0..1
            let v = iy as f32 / rows as f32;
            let x = (u - 0.5) * size;
            let y = (v - 0.5) * size;
            let wave = (x * 1.2).sin() * (y * 1.0).cos();
            let z = wave * 0.5; // slight height displacement
            positions.push([x, y, z]);
            // Analytical normal from the height gradient: z = 0.5 sin(1.2x) cos(y),
            // so dz/dx = 0.6 cos(1.2x) cos(y) and dz/dy = -0.5 sin(1.2x) sin(y).
            let dzdx = 0.6 * (x * 1.2).cos() * (y * 1.0).cos();
            let dzdy = -0.5 * (x * 1.2).sin() * (y * 1.0).sin();
            let n = glam::Vec3::new(-dzdx, -dzdy, 1.0).normalize();
            normals.push([n.x, n.y, n.z]);
            scalars.push(wave);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity((rows * cols * 6) as usize);
    for iy in 0..rows {
        for ix in 0..cols {
            let base = iy * nx + ix;
            indices.push(base);
            indices.push(base + nx);
            indices.push(base + 1);
            indices.push(base + 1);
            indices.push(base + nx);
            indices.push(base + nx + 1);
        }
    }

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh.attributes
        .insert("wave".to_string(), AttributeData::Vertex(scalars.clone()));
    (mesh, scalars)
}

/// Build a box mesh (cuboid) with per-vertex "distance" scalar.
/// Values below 0.4 (normalized) are set to NaN to demonstrate `nan_colour`.
fn make_box_with_distance_scalar() -> (MeshData, Vec<f32>) {
    let mut mesh = vpl::primitives::cuboid(2.5, 2.5, 2.5);
    let scalars: Vec<f32> = mesh
        .positions
        .iter()
        .map(|p| {
            let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let norm = dist / (2.5_f32 * 3.0_f32.sqrt() * 0.5); // normalize 0..1
            if norm < 0.4 { f32::NAN } else { norm }
        })
        .collect();
    mesh.attributes.insert(
        "distance".to_string(),
        AttributeData::Vertex(scalars.clone()),
    );
    let scalars_finite: Vec<f32> = scalars
        .iter()
        .map(|v| if v.is_nan() { 0.4 } else { *v })
        .collect();
    (mesh, scalars_finite)
}
