//! Showcase 14: Isolines & Contours
//!
//! A wave-function grid mesh coloured by its scalar field, with isoline
//! contour strips rendered on top via the `SceneFrame::isolines` pipeline.
//!
//! The mesh is built once and uploaded. `IsolineItem`s are re-submitted each
//! frame with the current slider/colour/width settings : no re-upload needed.

use crate::App;
use eframe::egui;
use viewport_lib::{
    AttributeData, AttributeKind, AttributeRef, BackfacePolicy, BuiltinColourmap, ColourmapId,
    FrameData, LightingSettings, Material, MeshData, MeshId, SceneRenderItem, ViewportRenderer,
    geometry::isoline::IsolineItem, scene::Scene,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct IsolinesState {
    pub built: bool,
    pub scene: Scene,
    pub mesh_index: MeshId,
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub scalars: Vec<f32>,
    pub grid_resolution: u32,
    pub contour_count: usize,
    pub line_colour: [f32; 4],
    pub line_width: f32,
    pub show_surface_colour: bool,
    pub depth_bias: f32,
}

impl Default for IsolinesState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            mesh_index: MeshId::from_index(0),
            positions: Vec::new(),
            indices: Vec::new(),
            scalars: Vec::new(),
            grid_resolution: 128,
            contour_count: 8,
            line_colour: [0.0, 0.0, 0.0, 1.0],
            line_width: 1.5,
            show_surface_colour: true,
            depth_bias: 0.005,
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

        self.iso_state
            .scene
            .add_named("Wave Grid", Some(mesh_id), glam::Mat4::IDENTITY, {
                let mut m = Material::from_colour([0.6, 0.65, 0.7]);
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

    ui.label("Line colour:");
    let mut colour = egui::Color32::from_rgba_unmultiplied(
        (app.iso_state.line_colour[0] * 255.0) as u8,
        (app.iso_state.line_colour[1] * 255.0) as u8,
        (app.iso_state.line_colour[2] * 255.0) as u8,
        (app.iso_state.line_colour[3] * 255.0) as u8,
    );
    if ui.color_edit_button_srgba(&mut colour).changed() {
        let [r, g, b, a] = colour.to_array();
        app.iso_state.line_colour = [
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

    ui.checkbox(
        &mut app.iso_state.show_surface_colour,
        "Surface scalar colouring",
    );
    ui.label("(grey when off, wave-coloured when on)");

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

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn iso_collect_scene_items(
    app: &mut App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64, u64) {
    let mut items = app.iso_state.scene.collect_render_items(&viewport_lib::selection::Selection::new());
    if app.iso_state.show_surface_colour {
        for item in items.iter_mut() {
            item.active_attribute = Some(AttributeRef {
                name: "wave".to_string(),
                kind: AttributeKind::Vertex,
            });
            item.colourmap_id = Some(ColourmapId(BuiltinColourmap::Coolwarm as usize));
            item.material.backface_policy = BackfacePolicy::Identical;
        }
    } else {
        for item in items.iter_mut() {
            item.material.backface_policy = BackfacePolicy::Identical;
        }
    }
    let sg = app.iso_state.scene.version();
    let lighting = {
        let mut _t = LightingSettings::default();
        _t.hemisphere_intensity = 0.5;
        _t.sky_colour = [1.0, 1.0, 1.0];
        _t.ground_colour = [1.0, 1.0, 1.0];
        _t
    };
    (items, lighting, sg, 0)
}

pub(crate) fn submit_iso_items(app: &App, fd: &mut FrameData) {
    if !app.iso_state.built {
        return;
    }
    let scalar_min = app
        .iso_state
        .scalars
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let scalar_max = app
        .iso_state
        .scalars
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let range = scalar_max - scalar_min;
    let isovalues: Vec<f32> = (0..app.iso_state.contour_count)
        .map(|i| {
            scalar_min
                + range * (i as f32 + 1.0) / (app.iso_state.contour_count as f32 + 1.0)
        })
        .collect();
    let mut iso_item = IsolineItem::default();
    iso_item.positions = app.iso_state.positions.clone();
    iso_item.indices = app.iso_state.indices.clone();
    iso_item.scalars = app.iso_state.scalars.clone();
    iso_item.isovalues = isovalues;
    iso_item.colour = app.iso_state.line_colour;
    iso_item.line_width = app.iso_state.line_width;
    iso_item.depth_bias = app.iso_state.depth_bias;
    fd.scene.isolines.push(iso_item);
}
