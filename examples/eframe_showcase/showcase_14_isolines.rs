//! Showcase 14: Isolines & Contours
//!
//! A wave-function grid mesh colored by its scalar field, with isoline
//! contour strips rendered on top via the `SceneFrame::isolines` pipeline.
//!
//! The mesh is built once and uploaded. `IsolineItem`s are re-submitted each
//! frame with the current slider/color/width settings : no re-upload needed.

use crate::App;
use eframe::egui;
use viewport_lib::{AttributeData, Material, MeshData, MeshId, ViewportRenderer, scene::Scene};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct IsolinesState {
    pub built:              bool,
    pub scene:              Scene,
    pub mesh_index:         MeshId,
    pub positions:          Vec<[f32; 3]>,
    pub indices:            Vec<u32>,
    pub scalars:            Vec<f32>,
    pub grid_resolution:    u32,
    pub contour_count:      usize,
    pub line_color:         [f32; 4],
    pub line_width:         f32,
    pub show_surface_color: bool,
    pub depth_bias:         f32,
}

impl Default for IsolinesState {
    fn default() -> Self {
        Self {
            built:              false,
            scene:              Scene::new(),
            mesh_index:         MeshId::from_index(0),
            positions:          Vec::new(),
            indices:            Vec::new(),
            scalars:            Vec::new(),
            grid_resolution:    128,
            contour_count:      8,
            line_color:         [0.0, 0.0, 0.0, 1.0],
            line_width:         1.5,
            show_surface_color: true,
            depth_bias:         0.005,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl App {
    /// Build the scene for Showcase 14 (Isolines demo).
    pub(crate) fn build_iso_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.iso_state.scene = Scene::new();

        let res = self.iso_state.grid_resolution;
        let (mesh, scalars) = make_wave_grid_iso(res, res, 10.0);

        // Keep CPU copies for IsolineItem re-submission each frame.
        self.iso_state.positions = mesh.positions.clone();
        self.iso_state.indices = mesh.indices.clone();
        self.iso_state.scalars = scalars;

        let mesh_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &mesh)
            .expect("iso wave mesh");
        self.iso_state.mesh_index = mesh_id;

        self.iso_state.scene
            .add_named("Wave Grid", Some(mesh_id), glam::Mat4::IDENTITY, {
                let mut m = Material::from_color([0.6, 0.65, 0.7]);
                m.roughness = 0.6;
                m
            });

        self.iso_state.built = true;
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_isolines(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Mesh resolution:");
    let res_resp = ui.add(
        egui::Slider::new(&mut app.iso_state.grid_resolution, 16..=256)
            .text("quads/side")
            .logarithmic(true),
    );
    if res_resp.drag_stopped() || res_resp.lost_focus() {
        app.iso_state.built = false;
    }

    ui.separator();
    ui.label("Contour levels:");
    ui.add(egui::Slider::new(&mut app.iso_state.contour_count, 2..=20).text("levels"));

    ui.separator();

    ui.label("Line color:");
    let mut color = egui::Color32::from_rgba_unmultiplied(
        (app.iso_state.line_color[0] * 255.0) as u8,
        (app.iso_state.line_color[1] * 255.0) as u8,
        (app.iso_state.line_color[2] * 255.0) as u8,
        (app.iso_state.line_color[3] * 255.0) as u8,
    );
    if ui.color_edit_button_srgba(&mut color).changed() {
        let [r, g, b, a] = color.to_array();
        app.iso_state.line_color = [
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        ];
    }

    ui.separator();

    ui.label("Line width (px):");
    ui.add(egui::Slider::new(&mut app.iso_state.line_width, 0.5..=5.0).step_by(0.25));

    ui.separator();

    ui.checkbox(&mut app.iso_state.show_surface_color, "Surface scalar coloring");
    ui.label("(grey when off, wave-colored when on)");

    ui.separator();

    ui.label("Depth bias:");
    ui.add(
        egui::Slider::new(&mut app.iso_state.depth_bias, 0.0..=0.05)
            .step_by(0.001)
            .text("(z-fighting offset)"),
    );
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Build a wave-function grid with per-vertex "wave" scalar attribute.
fn make_wave_grid_iso(cols: u32, rows: u32, size: f32) -> (MeshData, Vec<f32>) {
    let nx = cols + 1;
    let ny = rows + 1;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((nx * ny) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((nx * ny) as usize);
    let mut scalars: Vec<f32> = Vec::with_capacity((nx * ny) as usize);

    for iy in 0..ny {
        for ix in 0..nx {
            let u = ix as f32 / cols as f32;
            let v = iy as f32 / rows as f32;
            let x = (u - 0.5) * size;
            let y = (v - 0.5) * size;
            let wave = (x * 1.2).sin() * (y * 1.0).cos();
            let z = wave * 0.6;
            positions.push([x, y, z]);
            normals.push([0.0, 0.0, 1.0]);
            scalars.push(wave);
        }
    }

    // CCW winding viewed from +Z so the top face is the front face.
    let mut indices: Vec<u32> = Vec::with_capacity((rows * cols * 6) as usize);
    for iy in 0..rows {
        for ix in 0..cols {
            let base = iy * nx + ix;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + nx);
            indices.push(base + 1);
            indices.push(base + nx + 1);
            indices.push(base + nx);
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
