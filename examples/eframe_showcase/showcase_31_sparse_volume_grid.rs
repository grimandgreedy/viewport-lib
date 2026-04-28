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
//!   5×5×5 grid (81 active cells).  Interior shared faces are discarded —
//!   only the outer surface is rendered.
//!
//! - **Hollow shell** (centre) : same 5×5×5 grid but only the cells between
//!   inner radius 1.8 and outer radius 2.7 are active (54 active cells).
//!   Because the interior is empty, `extract_sparse_boundary` produces
//!   **two** surfaces — an outer boundary and an inner boundary — from a single
//!   upload.
//!
//! - **Voxel terrain** (right) : an 8×5×8 column grid where each XZ column is
//!   filled to a sine-wave height (the bottom face of each column and all
//!   interior stack-faces are discarded automatically).
//!
//! ## Attribute modes (applied uniformly to all three shapes)
//!
//! - **Cell height**: `cell_scalars["height"]` — normalised Y elevation of
//!   each cell centre.
//! - **Node distance / elevation**: `node_scalars["distance"]` — distance from
//!   the grid centre for the sphere/shell; normalised elevation for terrain.
//!   Averaged over 4 quad corner nodes per face.
//! - **Cell hue**: `cell_colors["hue"]` — direct RGBA from the azimuthal angle
//!   of each cell centre, no colormap.

use crate::App;
use eframe::egui;
use std::f32::consts::PI;
use viewport_lib::{
    AttributeKind, AttributeRef, BuiltinColormap, ColormapId, LightingSettings, MeshId,
    SceneRenderItem, SparseVolumeGridData, ViewportRenderer,
};

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
    /// `cell_colors["hue"]`: direct RGBA from azimuthal angle, no colormap.
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
    data.cell_colors.insert("hue".to_string(), cell_hues);
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
    data.cell_colors.insert("hue".to_string(), cell_hues);
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
    data.cell_colors.insert("hue".to_string(), cell_hues);
    data
}

// ---------------------------------------------------------------------------
// App methods
// ---------------------------------------------------------------------------

impl App {
    /// Upload all three sparse grid meshes; called once when the showcase is
    /// first shown.
    pub(crate) fn build_svg_scene(&mut self, renderer: &mut ViewportRenderer) {
        let sphere = build_sphere_grid();
        self.svg_mesh_id = renderer
            .resources_mut()
            .upload_sparse_volume_grid_data(&self.device, &sphere)
            .expect("svg sphere upload");

        let shell = build_hollow_shell();
        self.svg_shell_id = renderer
            .resources_mut()
            .upload_sparse_volume_grid_data(&self.device, &shell)
            .expect("svg shell upload");

        let terrain = build_terrain();
        self.svg_terrain_id = renderer
            .resources_mut()
            .upload_sparse_volume_grid_data(&self.device, &terrain)
            .expect("svg terrain upload");

        self.svg_built = true;
    }

    /// Produce scene items for all three shapes with the active attribute mode.
    pub(crate) fn svg_scene_items(&self) -> Vec<SceneRenderItem> {
        if !self.svg_built {
            return vec![];
        }

        let (active_attribute, colormap_id) = match self.svg_field {
            SvgField::CellHeight => (
                Some(AttributeRef {
                    name: "height".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColormapId(self.svg_colormap as usize)),
            ),
            SvgField::NodeDistance => (
                Some(AttributeRef {
                    name: "distance".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColormapId(self.svg_colormap as usize)),
            ),
            SvgField::CellHue => (
                Some(AttributeRef {
                    name: "hue".to_string(),
                    kind: AttributeKind::FaceColor,
                }),
                None,
            ),
        };

        // Three objects placed side by side along X.
        [
            (self.svg_mesh_id, translate(-9.0, 0.0, 0.0)),
            (self.svg_shell_id, translate(0.0, 0.0, 0.0)),
            (self.svg_terrain_id, translate(9.0, 0.0, 0.0)),
        ]
        .into_iter()
        .map(|(mesh_id, model)| {
            let mut item = SceneRenderItem::default();
            item.mesh_id = mesh_id;
            item.model = model;
            item.active_attribute = active_attribute.clone();
            item.colormap_id = colormap_id;
            item
        })
        .collect()
    }

    /// Lighting settings for the sparse grid showcase.
    pub(crate) fn svg_lighting() -> LightingSettings {
        LightingSettings {
            // Pure hemisphere: equal light from every direction, no directional
            // hotspots.  Sky and ground are both near-white so top and bottom
            // faces are equally readable.
            hemisphere_intensity: 0.9,
            sky_color: [1.0, 1.0, 1.0],
            ground_color: [0.85, 0.85, 0.9],
            lights: vec![],
            shadows_enabled: false,
            ..LightingSettings::default()
        }
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_sparse_volume_grid(&mut self, ui: &mut egui::Ui) {
        ui.label("Attribute source (applied to all three shapes):");
        for (field, label) in [
            (SvgField::CellHeight, "Cell height (cell_scalars)"),
            (SvgField::NodeDistance, "Node distance / elevation (node_scalars)"),
            (SvgField::CellHue, "Cell hue (cell_colors, direct RGBA)"),
        ] {
            ui.radio_value(&mut self.svg_field, field, label);
        }

        if self.svg_field != SvgField::CellHue {
            ui.separator();
            ui.label("Colormap:");
            ui.horizontal_wrapped(|ui| {
                for cm in [
                    BuiltinColormap::Viridis,
                    BuiltinColormap::Plasma,
                    BuiltinColormap::Magma,
                    BuiltinColormap::Inferno,
                    BuiltinColormap::Turbo,
                    BuiltinColormap::Coolwarm,
                    BuiltinColormap::RdBu,
                    BuiltinColormap::Rainbow,
                    BuiltinColormap::Jet,
                ] {
                    ui.radio_value(&mut self.svg_colormap, cm, format!("{cm:?}"));
                }
            });
        }

        ui.separator();
        ui.label("Left : solid sphere (81 cells, outer surface only).");
        ui.label("Centre : hollow shell (54 cells, outer + inner surfaces).");
        ui.label("Right : voxel terrain (column heightmap).");
    }
}

// Suppress unused-import warning: MeshId is used in the App struct fields
// declared in main.rs, not here directly.
const _: () = { let _ = std::mem::size_of::<MeshId>(); };
