//! Showcase 12: Scalar Fields & Colormaps : build and controls.
//!
//! Three objects each carrying a different procedural scalar attribute:
//!   Object 0 : Sphere,    attribute "height"   (world Z of each vertex, 0..1 range)
//!   Object 1 : Wave Grid, attribute "wave"      (sine-derived 2-D wave, -1..1 range)
//!   Object 2 : Box,       attribute "distance"  (distance from center, with NaN below 0.3)

use crate::App;
use eframe::egui;
use viewport_lib::{
    AttributeData, BuiltinColormap, Material, MeshData, MeshId, NodeId, ScalarBarAnchor,
    ScalarBarItem, ScalarBarOrientation, Selection, ViewportRenderer, scene::Scene,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct ScalarFieldsState {
    pub built: bool,
    pub scene: Scene,
    pub selection: Selection,
    pub colormap: BuiltinColormap,
    pub range_auto: bool,
    pub range: (f32, f32),
    pub nan_on: bool,
    pub bar_anchor: ScalarBarAnchor,
    pub bar_orientation: ScalarBarOrientation,
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
            colormap: BuiltinColormap::Viridis,
            range_auto: true,
            range: (0.0, 1.0),
            nan_on: false,
            bar_anchor: ScalarBarAnchor::BottomRight,
            bar_orientation: ScalarBarOrientation::Vertical,
            node_ids: [0; 3],
            mesh_indices: [MeshId::from_index(0); 3],
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
        let mut sphere = viewport_lib::primitives::sphere(3.0, 48, 24);
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
                let mut m = Material::from_color([0.8, 0.8, 0.8]);
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
                    let mut m = Material::from_color([0.8, 0.8, 0.8]);
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
                let mut m = Material::from_color([0.8, 0.8, 0.8]);
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

    /// Build a [`ScalarBarItem`] for the current scalar bar UI state.
    ///
    /// Called each frame when Showcase 12 is active; the returned item is
    /// inserted into `OverlayFrame::scalar_bars` for native rendering.
    pub(crate) fn scalar_bar_item(&self) -> ScalarBarItem {
        let s = &self.scalar_state;
        let (scalar_min, scalar_max) = if s.range_auto {
            let vals = &s.values[s.active_object];
            if vals.is_empty() {
                (0.0_f32, 1.0_f32)
            } else {
                let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (mn, mx)
            }
        } else {
            s.range
        };

        ScalarBarItem {
            colormap_id: viewport_lib::ColormapId(s.colormap as usize),
            scalar_min,
            scalar_max,
            anchor: s.bar_anchor,
            orientation: s.bar_orientation,
            bar_width_px: 20.0,
            bar_length_px: 140.0,
            margin_px: 12.0,
            tick_count: 3,
            ..Default::default()
        }
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
    ui.label("Colormap:");
    for (preset, label) in [
        (BuiltinColormap::Viridis, "Viridis"),
        (BuiltinColormap::Plasma, "Plasma"),
        (BuiltinColormap::Greyscale, "Greyscale"),
        (BuiltinColormap::Coolwarm, "Coolwarm"),
        (BuiltinColormap::Rainbow, "Rainbow"),
    ] {
        if ui.radio(s.colormap == preset, label).clicked() {
            s.colormap = preset;
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
    ui.checkbox(&mut s.nan_on, "Show NaN color (purple)");
    ui.label("(box object: values < threshold set to NaN)");

    ui.separator();
    ui.label("Scalar Bar:");
    for (anchor, label) in [
        (ScalarBarAnchor::TopLeft, "Top-Left"),
        (ScalarBarAnchor::TopRight, "Top-Right"),
        (ScalarBarAnchor::BottomLeft, "Bottom-Left"),
        (ScalarBarAnchor::BottomRight, "Bottom-Right"),
    ] {
        if ui.radio(s.bar_anchor == anchor, label).clicked() {
            s.bar_anchor = anchor;
        }
    }
    ui.horizontal(|ui| {
        if ui
            .radio(
                s.bar_orientation == ScalarBarOrientation::Vertical,
                "Vertical",
            )
            .clicked()
        {
            s.bar_orientation = ScalarBarOrientation::Vertical;
        }
        if ui
            .radio(
                s.bar_orientation == ScalarBarOrientation::Horizontal,
                "Horizontal",
            )
            .clicked()
        {
            s.bar_orientation = ScalarBarOrientation::Horizontal;
        }
    });
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
            normals.push([0.0, 0.0, 1.0]); // approximate flat normals
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
/// Values below 0.4 (normalized) are set to NaN to demonstrate `nan_color`.
fn make_box_with_distance_scalar() -> (MeshData, Vec<f32>) {
    let mut mesh = viewport_lib::primitives::cuboid(2.5, 2.5, 2.5);
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
