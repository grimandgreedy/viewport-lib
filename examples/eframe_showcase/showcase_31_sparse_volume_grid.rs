//! Showcase 31: Sparse Volume Grids
//!
//! Demonstrates Phase 14 : [`SparseVolumeGridData`] topology processing.
//! Three sparse grids are shown side by side, each using different occupancy
//! masks, to illustrate how boundary face extraction works across different
//! topologies.
//!
//! ## Shapes
//!
//! - **Solid sphere** (left) : all cells inside a sphere of radius 2.7 in a
//!   5×5×5 grid (81 active cells).  Interior shared faces are discarded :
//!   only the outer surface is rendered.
//!
//! - **Hollow shell** (centre) : same 5×5×5 grid but only the cells between
//!   inner radius 1.8 and outer radius 2.7 are active (54 active cells).
//!   Because the interior is empty, `extract_sparse_boundary` produces
//!   **two** surfaces : an outer boundary and an inner boundary : from a single
//!   upload.
//!
//! - **Voxel terrain** (right) : an 8×5×8 column grid where each XZ column is
//!   filled to a sine-wave height (the bottom face of each column and all
//!   interior stack-faces are discarded automatically).
//!
//! ## Attribute modes (applied uniformly to all three shapes)
//!
//! - **Cell height**: `cell_scalars["height"]` : normalised Y elevation of
//!   each cell centre.
//! - **Node distance / elevation**: `node_scalars["distance"]` : distance from
//!   the grid centre for the sphere/shell; normalised elevation for terrain.
//!   Averaged over 4 quad corner nodes per face.
//! - **Cell hue**: `cell_colours["hue"]` : direct RGBA from the azimuthal angle
//!   of each cell centre, no colourmap.

use crate::App;
use eframe::egui;
use std::f32::consts::PI;
use viewport_lib::{
    AttributeKind, AttributeRef, BuiltinColourmap, ColourmapId, LightingSettings, MeshId,
    SceneRenderItem, SparseVolumeGridData, ViewportRenderer,
};

// ---------------------------------------------------------------------------
// Paint grid constants
// ---------------------------------------------------------------------------

/// Side length of the paintable voxel cube (NxNxN cells, all active).
const PAINT_N: usize = 5;

/// World-space translation applied to the paint grid at render time.
/// The ray-picking code in `handle_svg_paint_click` must use the same value.
const PAINT_OFFSET: [f32; 3] = [18.0, 0.0, 0.0];

/// Preset swatch colours shown in the controls panel (linear RGBA).
const PAINT_SWATCHES: &[([f32; 4], &str)] = &[
    ([1.00, 1.00, 1.00, 1.0], "White"),
    ([0.90, 0.20, 0.20, 1.0], "Red"),
    ([0.90, 0.55, 0.10, 1.0], "Orange"),
    ([0.90, 0.85, 0.10, 1.0], "Yellow"),
    ([0.20, 0.80, 0.20, 1.0], "Green"),
    ([0.10, 0.75, 0.75, 1.0], "Teal"),
    ([0.20, 0.40, 0.90, 1.0], "Blue"),
    ([0.65, 0.20, 0.90, 1.0], "Purple"),
    ([0.08, 0.08, 0.08, 1.0], "Black"),
];

// ---------------------------------------------------------------------------
// Sub-mode
// ---------------------------------------------------------------------------

/// Which attribute source is currently displayed on all three shapes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SvgField {
    /// `cell_scalars["height"]`: normalised Y elevation of each cell centre.
    CellHeight,
    /// `node_scalars["distance"]`: distance from grid centre (sphere / shell)
    /// or normalised elevation (terrain), averaged over 4 corner nodes.
    NodeDistance,
    /// `cell_colours["hue"]`: direct RGBA from azimuthal angle, no colourmap.
    CellHue,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Convert HSV (all components in [0, 1]) to linear RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h * 6.0;
    let i = h6.floor() as u32 % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Column-major 4×4 pure-translation matrix.
fn translate(tx: f32, ty: f32, tz: f32) -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [tx, ty, tz, 1.0],
    ]
}

// ---------------------------------------------------------------------------
// Shape 1: solid sphere
// ---------------------------------------------------------------------------

const GRID_N: usize = 5;
const SPHERE_R: f32 = 2.7;

/// 5×5×5 grid, all cells inside a sphere of radius 2.7 (81 active cells).
fn build_sphere_grid() -> SparseVolumeGridData {
    let centre = GRID_N as f32 / 2.0; // 2.5

    let mut active_cells = Vec::new();
    let mut cell_heights = Vec::new();
    let mut cell_hues = Vec::new();

    for k in 0..GRID_N {
        for j in 0..GRID_N {
            for i in 0..GRID_N {
                let (cx, cy, cz) = (
                    i as f32 + 0.5 - centre,
                    j as f32 + 0.5 - centre,
                    k as f32 + 0.5 - centre,
                );
                if (cx * cx + cy * cy + cz * cz).sqrt() < SPHERE_R {
                    active_cells.push([i as u32, j as u32, k as u32]);
                    cell_heights.push(j as f32 / (GRID_N - 1) as f32);
                    let angle = cz.atan2(cx) / (2.0 * PI) + 0.5;
                    let [r, g, b] = hsv_to_rgb(angle, 0.8, 0.9);
                    cell_hues.push([r, g, b, 1.0]);
                }
            }
        }
    }

    let nd = GRID_N + 1;
    let mut node_dist = vec![0.0f32; nd * nd * nd];
    for nk in 0..nd {
        for nj in 0..nd {
            for ni in 0..nd {
                let dist = ((ni as f32 - centre).powi(2)
                    + (nj as f32 - centre).powi(2)
                    + (nk as f32 - centre).powi(2))
                .sqrt();
                node_dist[nk * nd * nd + nj * nd + ni] = dist;
            }
        }
    }

    let mut data = SparseVolumeGridData::default();
    data.origin = [-(GRID_N as f32) / 2.0; 3];
    data.cell_size = 1.0;
    data.active_cells = active_cells;
    data.cell_scalars.insert("height".to_string(), cell_heights);
    data.node_scalars.insert("distance".to_string(), node_dist);
    data.cell_colours.insert("hue".to_string(), cell_hues);
    data
}

// ---------------------------------------------------------------------------
// Shape 2: hollow shell
// ---------------------------------------------------------------------------

const INNER_R: f32 = 1.8;

/// 5×5×5 grid, only the shell between inner radius 1.8 and outer radius 2.7
/// (54 active cells).  Produces both an outer and an inner boundary surface.
fn build_hollow_shell() -> SparseVolumeGridData {
    let centre = GRID_N as f32 / 2.0;

    let mut active_cells = Vec::new();
    let mut cell_heights = Vec::new();
    let mut cell_hues = Vec::new();

    for k in 0..GRID_N {
        for j in 0..GRID_N {
            for i in 0..GRID_N {
                let (cx, cy, cz) = (
                    i as f32 + 0.5 - centre,
                    j as f32 + 0.5 - centre,
                    k as f32 + 0.5 - centre,
                );
                let dist = (cx * cx + cy * cy + cz * cz).sqrt();
                if dist >= INNER_R && dist < SPHERE_R {
                    active_cells.push([i as u32, j as u32, k as u32]);
                    cell_heights.push(j as f32 / (GRID_N - 1) as f32);
                    let angle = cz.atan2(cx) / (2.0 * PI) + 0.5;
                    let [r, g, b] = hsv_to_rgb(angle, 0.8, 0.9);
                    cell_hues.push([r, g, b, 1.0]);
                }
            }
        }
    }

    let nd = GRID_N + 1;
    let mut node_dist = vec![0.0f32; nd * nd * nd];
    for nk in 0..nd {
        for nj in 0..nd {
            for ni in 0..nd {
                let dist = ((ni as f32 - centre).powi(2)
                    + (nj as f32 - centre).powi(2)
                    + (nk as f32 - centre).powi(2))
                .sqrt();
                node_dist[nk * nd * nd + nj * nd + ni] = dist;
            }
        }
    }

    let mut data = SparseVolumeGridData::default();
    data.origin = [-(GRID_N as f32) / 2.0; 3];
    data.cell_size = 1.0;
    data.active_cells = active_cells;
    data.cell_scalars.insert("height".to_string(), cell_heights);
    data.node_scalars.insert("distance".to_string(), node_dist);
    data.cell_colours.insert("hue".to_string(), cell_hues);
    data
}

// ---------------------------------------------------------------------------
// Shape 3: voxel terrain
// ---------------------------------------------------------------------------

const TERRAIN_W: usize = 8;
const TERRAIN_H: usize = 5;
const TERRAIN_D: usize = 8;

/// 8×5×8 heightmap grid.  Each XZ column is filled up to a sine-wave height;
/// only the top and outer boundary faces of each column survive extraction.
fn build_terrain() -> SparseVolumeGridData {
    let cx = TERRAIN_W as f32 / 2.0 - 0.5; // centre of columns for hue
    let cz = TERRAIN_D as f32 / 2.0 - 0.5;

    let mut active_cells = Vec::new();
    let mut cell_heights = Vec::new();
    let mut cell_hues = Vec::new();

    for k in 0..TERRAIN_D {
        for i in 0..TERRAIN_W {
            let wave = (i as f32 * PI / 3.5).sin() * (k as f32 * PI / 3.5).cos();
            let col_h = ((2.5 + 2.0 * wave).round() as usize + 1).min(TERRAIN_H);

            for j in 0..col_h {
                active_cells.push([i as u32, j as u32, k as u32]);
                cell_heights.push(j as f32 / (TERRAIN_H - 1) as f32);

                let dx = i as f32 - cx;
                let dz = k as f32 - cz;
                let angle = dz.atan2(dx) / (2.0 * PI) + 0.5;
                let [r, g, b] = hsv_to_rgb(angle, 0.75, 0.88);
                cell_hues.push([r, g, b, 1.0]);
            }
        }
    }

    // Node scalars: normalised elevation (j / TERRAIN_H per node).
    let nw = TERRAIN_W + 1;
    let nh = TERRAIN_H + 1;
    let nd = TERRAIN_D + 1;
    let mut node_elev = vec![0.0f32; nw * nh * nd];
    for nk in 0..nd {
        for nj in 0..nh {
            for ni in 0..nw {
                node_elev[nk * nh * nw + nj * nw + ni] = nj as f32 / TERRAIN_H as f32;
            }
        }
    }

    let mut data = SparseVolumeGridData::default();
    // Centre the terrain in XZ; Y origin puts the base below world origin.
    data.origin = [
        -(TERRAIN_W as f32) / 2.0,
        -(TERRAIN_H as f32) / 2.0,
        -(TERRAIN_D as f32) / 2.0,
    ];
    data.cell_size = 1.0;
    data.active_cells = active_cells;
    data.cell_scalars.insert("height".to_string(), cell_heights);
    data.node_scalars.insert("distance".to_string(), node_elev);
    data.cell_colours.insert("hue".to_string(), cell_hues);
    data
}

// ---------------------------------------------------------------------------
// Paint grid
// ---------------------------------------------------------------------------

/// Build a fully-dense PAINT_N^3 voxel cube with all cells white.
/// Cell colours are stored in `cell_colours["paint"]` and updated on each click.
fn build_paint_grid() -> SparseVolumeGridData {
    let n = PAINT_N;
    let mut active_cells = Vec::with_capacity(n * n * n);
    let mut colours = Vec::with_capacity(n * n * n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                active_cells.push([i as u32, j as u32, k as u32]);
                colours.push([1.0f32, 1.0, 1.0, 1.0]);
            }
        }
    }
    let mut data = SparseVolumeGridData::default();
    // Centre the grid at the world origin (before PAINT_OFFSET is applied).
    data.origin = [-(n as f32) / 2.0; 3];
    data.cell_size = 1.0;
    data.active_cells = active_cells;
    data.cell_colours.insert("paint".to_string(), colours);
    data
}

/// Ray-AABB slab intersection.  Returns the entry distance along `dir`, or
/// `None` if the ray misses or the box is behind the origin.
fn ray_aabb(
    origin: glam::Vec3,
    dir: glam::Vec3,
    aabb_min: glam::Vec3,
    aabb_max: glam::Vec3,
) -> Option<f32> {
    let inv = dir.recip();
    let t1 = (aabb_min - origin) * inv;
    let t2 = (aabb_max - origin) * inv;
    let tmin = t1.min(t2).max_element();
    let tmax = t1.max(t2).min_element();
    if tmax >= tmin.max(0.0) {
        Some(tmin.max(0.0))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct SvgState {
    pub built: bool,
    pub mesh_id: MeshId,
    pub shell_id: MeshId,
    pub terrain_id: MeshId,
    pub colourmap: BuiltinColourmap,
    pub field: SvgField,
    // Paint grid state
    pub paint_mesh_id: MeshId,
    pub paint_data: SparseVolumeGridData,
    /// Currently selected paint colour (linear RGBA).
    pub paint_colour: [f32; 4],
    /// Set to true when a cell is painted; cleared after the GPU upload.
    pub paint_dirty: bool,
}

impl Default for SvgState {
    fn default() -> Self {
        Self {
            built: false,
            mesh_id: MeshId::from_index(0),
            shell_id: MeshId::from_index(0),
            terrain_id: MeshId::from_index(0),
            colourmap: BuiltinColourmap::Viridis,
            field: SvgField::CellHeight,
            paint_mesh_id: MeshId::from_index(0),
            paint_data: SparseVolumeGridData::default(),
            paint_colour: PAINT_SWATCHES[1].0, // red
            paint_dirty: false,
        }
    }
}

// ---------------------------------------------------------------------------
// App methods
// ---------------------------------------------------------------------------

impl App {
    /// Upload all three sparse grid meshes; called once when the showcase is
    /// first shown.
    pub(crate) fn build_svg_scene(&mut self, renderer: &mut ViewportRenderer) {
        let sphere = build_sphere_grid();
        self.svg_state.mesh_id = renderer
            .resources_mut()
            .upload_sparse_volume_grid_data(&self.device, &sphere)
            .expect("svg sphere upload");

        let shell = build_hollow_shell();
        self.svg_state.shell_id = renderer
            .resources_mut()
            .upload_sparse_volume_grid_data(&self.device, &shell)
            .expect("svg shell upload");

        let terrain = build_terrain();
        self.svg_state.terrain_id = renderer
            .resources_mut()
            .upload_sparse_volume_grid_data(&self.device, &terrain)
            .expect("svg terrain upload");

        self.svg_state.paint_data = build_paint_grid();
        self.svg_state.paint_mesh_id = renderer
            .resources_mut()
            .upload_sparse_volume_grid_data(&self.device, &build_paint_grid())
            .expect("svg paint upload");

        self.svg_state.built = true;
    }

    /// Produce scene items for all three shapes with the active attribute mode.
    pub(crate) fn svg_scene_items(&self) -> Vec<SceneRenderItem> {
        if !self.svg_state.built {
            return vec![];
        }

        let (active_attribute, colourmap_id) = match self.svg_state.field {
            SvgField::CellHeight => (
                Some(AttributeRef {
                    name: "height".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColourmapId(self.svg_state.colourmap as usize)),
            ),
            SvgField::NodeDistance => (
                Some(AttributeRef {
                    name: "distance".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColourmapId(self.svg_state.colourmap as usize)),
            ),
            SvgField::CellHue => (
                Some(AttributeRef {
                    name: "hue".to_string(),
                    kind: AttributeKind::FaceColour,
                }),
                None,
            ),
        };

        // Three attribute-driven objects side by side along X.
        let mut items: Vec<SceneRenderItem> = [
            (self.svg_state.mesh_id, translate(-9.0, 0.0, 0.0)),
            (self.svg_state.shell_id, translate(0.0, 0.0, 0.0)),
            (self.svg_state.terrain_id, translate(9.0, 0.0, 0.0)),
        ]
        .into_iter()
        .map(|(mesh_id, model)| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = model;
            item.active_attribute = active_attribute.clone();
            item.colourmap_id = colourmap_id;
            item
        })
        .collect();

        // Paint grid: always uses cell_colours["paint"], independent of the
        // attribute selector above.
        let mut paint = SceneRenderItem::default();
        paint.mesh_id = self.svg_state.paint_mesh_id;
        paint.model = translate(PAINT_OFFSET[0], PAINT_OFFSET[1], PAINT_OFFSET[2]);
        paint.active_attribute = Some(AttributeRef {
            name: "paint".to_string(),
            kind: AttributeKind::FaceColour,
        });
        items.push(paint);

        items
    }

    /// Cast a ray into the paint grid and paint the nearest hit cell.
    /// Sets `svg_state.paint_dirty` if a cell was hit; the caller flushes it to GPU.
    pub(crate) fn handle_svg_paint_click(&mut self, pos: glam::Vec2, w: f32, h: f32) {
        if !self.svg_state.built {
            return;
        }
        let vp_inv = self.camera.view_proj_matrix().inverse();
        let (ray_o, ray_d) =
            viewport_lib::picking::screen_to_ray(pos, glam::Vec2::new(w, h), vp_inv);

        let data = &self.svg_state.paint_data;
        let offset = glam::Vec3::from(PAINT_OFFSET);
        let mut best_t = f32::MAX;
        let mut best_idx = None;

        for (cell_idx, &[ci, cj, ck]) in data.active_cells.iter().enumerate() {
            let local_min = glam::Vec3::new(
                data.origin[0] + ci as f32 * data.cell_size,
                data.origin[1] + cj as f32 * data.cell_size,
                data.origin[2] + ck as f32 * data.cell_size,
            );
            let world_min = local_min + offset;
            let world_max = world_min + glam::Vec3::splat(data.cell_size);

            if let Some(t) = ray_aabb(ray_o, ray_d, world_min, world_max) {
                if t < best_t {
                    best_t = t;
                    best_idx = Some(cell_idx);
                }
            }
        }

        if let Some(idx) = best_idx {
            let colours = self
                .svg_state
                .paint_data
                .cell_colours
                .get_mut("paint")
                .unwrap();
            colours[idx] = self.svg_state.paint_colour;
            self.svg_state.paint_dirty = true;
        }
    }

    /// Lighting settings for the sparse grid showcase.
    pub(crate) fn svg_lighting() -> LightingSettings {
        LightingSettings {
            // Pure hemisphere: equal light from every direction, no directional
            // hotspots.  Sky and ground are both near-white so top and bottom
            // faces are equally readable.
            hemisphere_intensity: 0.9,
            sky_colour: [1.0, 1.0, 1.0],
            ground_colour: [0.85, 0.85, 0.9],
            lights: vec![],
            shadows_enabled: false,
            ..LightingSettings::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_sparse_volume_grid(app: &mut App, ui: &mut egui::Ui) {
    // --- Paint grid ---
    ui.label(egui::RichText::new("Voxel paint (right grid)").strong());
    ui.label("Click a voxel to paint it.");
    ui.label("Colour:");
    ui.horizontal_wrapped(|ui| {
        for &(colour, label) in PAINT_SWATCHES {
            let selected = app.svg_state.paint_colour == colour;
            let [r, g, b, _] = colour;
            let fill =
                egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8);
            let stroke = if selected {
                egui::Stroke::new(2.5, egui::Color32::WHITE)
            } else {
                egui::Stroke::new(1.0, egui::Color32::GRAY)
            };
            let btn = egui::Button::new("")
                .min_size(egui::Vec2::splat(22.0))
                .fill(fill)
                .stroke(stroke);
            if ui.add(btn).on_hover_text(label).clicked() {
                app.svg_state.paint_colour = colour;
            }
        }
    });
    if ui
        .button("Clear")
        .on_hover_text("Reset all voxels to white")
        .clicked()
    {
        let colours = app
            .svg_state
            .paint_data
            .cell_colours
            .get_mut("paint")
            .unwrap();
        for c in colours.iter_mut() {
            *c = [1.0, 1.0, 1.0, 1.0];
        }
        app.svg_state.paint_dirty = true;
    }

    ui.separator();
    ui.label("Attribute source (applied to all three shapes):");
    for (field, label) in [
        (SvgField::CellHeight, "Cell height (cell_scalars)"),
        (
            SvgField::NodeDistance,
            "Node distance / elevation (node_scalars)",
        ),
        (SvgField::CellHue, "Cell hue (cell_colours, direct RGBA)"),
    ] {
        ui.radio_value(&mut app.svg_state.field, field, label);
    }

    if app.svg_state.field != SvgField::CellHue {
        ui.separator();
        ui.label("Colourmap:");
        ui.horizontal_wrapped(|ui| {
            for cm in [
                BuiltinColourmap::Viridis,
                BuiltinColourmap::Plasma,
                BuiltinColourmap::Magma,
                BuiltinColourmap::Inferno,
                BuiltinColourmap::Turbo,
                BuiltinColourmap::Coolwarm,
                BuiltinColourmap::RdBu,
                BuiltinColourmap::Rainbow,
                BuiltinColourmap::Jet,
            ] {
                ui.radio_value(&mut app.svg_state.colourmap, cm, format!("{cm:?}"));
            }
        });
    }

    ui.separator();
    ui.label("Left : solid sphere (81 cells, outer surface only).");
    ui.label("Centre : hollow shell (54 cells, outer + inner surfaces).");
    ui.label("Right : voxel terrain (column heightmap).");
}
