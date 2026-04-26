//! Showcase 26: Unstructured Volume Meshes
//!
//! Demonstrates Phase 9 : `VolumeMeshData` topology processing.
//!
//! A 3×3×3 structured grid of cells is projected onto a sphere.  The renderer
//! sees only the **boundary surface** : interior faces shared by two cells are
//! discarded automatically by [`upload_volume_mesh_data`].  Per-cell scalars
//! and colors are remapped to the boundary faces so the existing Phase 2
//! face-coloring path applies colormaps cell-by-cell with no new GPU work.
//!
//! ## Two modes
//! - **Hex sphere** : 27 hexahedral cells; boundary = 54 quads = 108 triangles.
//! - **Tet sphere** : same grid split into 6 tets per cube (Freudenthal) -> 162 tets;
//!   boundary = 54 triangles (much coarser faceting, making tet structure clear).
//!
//! ## Scalar fields
//! - *Latitude*: normalised Y elevation on the sphere (cold poles, warm equator).
//! - *Longitude*: normalised azimuthal angle; reveals longitudinal cell sectors.
//! - *Radial* (original distance before projection): shows the grid structure.
//!
//! ## What to notice
//! Switching between Hex and Tet shows different faceting on the same sphere:
//! hexes produce quad-faceted surfaces, tets produce a coarser triangular mesh.
//! Both render via the same surface pipeline : Phase 9 adds zero GPU complexity.

use crate::App;
use eframe::egui;
use viewport_lib::{
    AttributeKind, AttributeRef, BuiltinColormap, ColormapId, LightingSettings, SceneRenderItem,
    TET_SENTINEL, ViewportRenderer, VolumeMeshData,
};

// ---------------------------------------------------------------------------
// Sub-mode
// ---------------------------------------------------------------------------

/// Which cell type is currently displayed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum VmMode {
    /// 27 hexahedral cells.
    Hex,
    /// 135 tetrahedral cells (5 tets per cube).
    Tet,
}

/// Scalar field to colour by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum VmField {
    /// Cell colour by Y elevation (latitude).
    Latitude,
    /// Cell colour by XZ azimuth (longitude sectors).
    Longitude,
    /// Cell colour by original radial distance before sphere projection.
    Radial,
    /// Direct per-cell RGBA colour (no colormap).
    DirectColor,
}

// ---------------------------------------------------------------------------
// App state fields (declared in main App struct):
//   vm_built:        bool
//   vm_mode:         VmMode
//   vm_tet_index:    usize
//   vm_hex_index:    usize
//   vm_colormap:     BuiltinColormap
//   vm_field:        VmField
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Grid geometry
// ---------------------------------------------------------------------------

const GRID_N: usize = 3; // cells per axis -> GRID_N+1 vertices per axis
const GRID_V: usize = GRID_N + 1; // 4 vertices per axis
const SPHERE_R: f32 = 2.0; // sphere radius after projection

/// Vertex index from (ix, iy, iz) in the (GRID_V × GRID_V × GRID_V) lattice.
#[inline]
fn vid(ix: usize, iy: usize, iz: usize) -> u32 {
    (iz * GRID_V * GRID_V + iy * GRID_V + ix) as u32
}

/// Generate vertex positions for a GRID_V³ lattice centred at the origin,
/// then project each vertex onto the sphere of radius `SPHERE_R`.
///
/// Vertices are placed at –1.5, –0.5, 0.5, 1.5 on each axis (half-step
/// offset), so no vertex falls at the origin : safe for normalization.
fn sphere_vertex_positions() -> Vec<[f32; 3]> {
    let mut pos = Vec::with_capacity(GRID_V * GRID_V * GRID_V);
    for iz in 0..GRID_V {
        for iy in 0..GRID_V {
            for ix in 0..GRID_V {
                // Half-cell offset so no vertex is at the origin.
                let x = ix as f32 - 1.5;
                let y = iy as f32 - 1.5;
                let z = iz as f32 - 1.5;

                let len = (x * x + y * y + z * z).sqrt();
                let s = SPHERE_R / len; // always > 0 (min len ≈ 0.866)
                pos.push([x * s, y * s, z * s]);
            }
        }
    }
    pos
}

/// Cell centroid in the **original** (pre-projection) lattice space.
fn cell_centroid_raw(ix: usize, iy: usize, iz: usize) -> [f32; 3] {
    let x = ix as f32 - 1.5 + 0.5;
    let y = iy as f32 - 1.5 + 0.5;
    let z = iz as f32 - 1.5 + 0.5;
    [x, y, z]
}

// ---------------------------------------------------------------------------
// Shared colour helpers
// ---------------------------------------------------------------------------

fn hsv_to_rgba(h: f32, s: f32, v: f32) -> [f32; 4] {
    let h6 = h * 6.0;
    let sector = h6 as u32 % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b, 1.0]
}

// ---------------------------------------------------------------------------
// Hex mesh builder
// ---------------------------------------------------------------------------
//
// VTK hex vertex ordering used by `extract_boundary_faces`:
//
//     7 --- 6
//    /|    /|
//   4 --- 5 |
//   | 3 --| 2
//   |/    |/
//   0 --- 1

fn build_hex_mesh(positions: &[[f32; 3]]) -> VolumeMeshData {
    let n = GRID_N * GRID_N * GRID_N;
    let mut cells: Vec<[u32; 8]> = Vec::with_capacity(n);
    let mut lat_scalars: Vec<f32> = Vec::with_capacity(n);
    let mut lon_scalars: Vec<f32> = Vec::with_capacity(n);
    let mut radial_scalars: Vec<f32> = Vec::with_capacity(n);
    let mut direct_colors: Vec<[f32; 4]> = Vec::with_capacity(n);

    for iz in 0..GRID_N {
        for iy in 0..GRID_N {
            for ix in 0..GRID_N {
                cells.push([
                    vid(ix, iy, iz),
                    vid(ix + 1, iy, iz),
                    vid(ix + 1, iy, iz + 1),
                    vid(ix, iy, iz + 1),
                    vid(ix, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz + 1),
                    vid(ix, iy + 1, iz + 1),
                ]);

                let [cx, cy, cz] = cell_centroid_raw(ix, iy, iz);
                let raw_len = (cx * cx + cy * cy + cz * cz).sqrt();

                // Projected centroid on sphere
                let s = SPHERE_R / raw_len;
                let py = cy * s;

                // Latitude: –1 (south pole) to +1 (north pole) -> 0..1
                lat_scalars.push(py / SPHERE_R * 0.5 + 0.5);

                // Longitude: azimuthal angle in [0, 2π] -> 0..1
                let lon = (cz).atan2(cx) / std::f32::consts::TAU + 0.5;
                lon_scalars.push(lon);

                // Radial: original distance before projection -> 0..1 (min √0.75 ≈ 0.87)
                radial_scalars.push(raw_len);

                // Direct colour: hue from longitude sector, saturation by latitude
                let sat = 0.5 + 0.5 * (py / SPHERE_R).abs();
                direct_colors.push(hsv_to_rgba(lon, sat, 0.9));
            }
        }
    }

    let mut data = VolumeMeshData::default();
    data.positions = positions.to_vec();
    data.cells = cells;
    data.cell_scalars
        .insert("latitude".to_string(), lat_scalars);
    data.cell_scalars
        .insert("longitude".to_string(), lon_scalars);
    data.cell_scalars
        .insert("radial".to_string(), radial_scalars);
    data.cell_colors.insert("direct".to_string(), direct_colors);
    data
}

// ---------------------------------------------------------------------------
// Tet mesh builder
// ---------------------------------------------------------------------------
//
// Each cube is split into 6 tets using the Freudenthal (body-diagonal) decomposition.
// This is a *conforming* subdivision: all three axis-aligned shared faces between
// adjacent cubes produce identical triangulations, so no gaps appear at boundaries.
// The six tets share the main diagonal v[0]->v[6].

const TET_LOCAL: [[usize; 4]; 6] = [
    [0, 1, 5, 6], // xyz
    [0, 1, 2, 6], // xzy
    [0, 4, 5, 6], // yxz
    [0, 4, 7, 6], // yzx
    [0, 3, 2, 6], // zxy
    [0, 3, 7, 6], // zyx
];

fn build_tet_mesh(positions: &[[f32; 3]]) -> VolumeMeshData {
    let n = GRID_N * GRID_N * GRID_N * 6;
    let mut cells: Vec<[u32; 8]> = Vec::with_capacity(n);
    let mut lat_scalars: Vec<f32> = Vec::with_capacity(n);
    let mut lon_scalars: Vec<f32> = Vec::with_capacity(n);
    let mut radial_scalars: Vec<f32> = Vec::with_capacity(n);
    let mut direct_colors: Vec<[f32; 4]> = Vec::with_capacity(n);

    for iz in 0..GRID_N {
        for iy in 0..GRID_N {
            for ix in 0..GRID_N {
                let cube_verts = [
                    vid(ix, iy, iz),
                    vid(ix + 1, iy, iz),
                    vid(ix + 1, iy, iz + 1),
                    vid(ix, iy, iz + 1),
                    vid(ix, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz + 1),
                    vid(ix, iy + 1, iz + 1),
                ];

                let [cx, cy, cz] = cell_centroid_raw(ix, iy, iz);
                let raw_len = (cx * cx + cy * cy + cz * cz).sqrt();
                let s = SPHERE_R / raw_len;
                let py = cy * s;

                let lat = py / SPHERE_R * 0.5 + 0.5;
                let lon = (cz).atan2(cx) / std::f32::consts::TAU + 0.5;
                let sat = 0.5 + 0.5 * (py / SPHERE_R).abs();
                let color = hsv_to_rgba(lon, sat, 0.85);

                for tet in &TET_LOCAL {
                    cells.push([
                        cube_verts[tet[0]],
                        cube_verts[tet[1]],
                        cube_verts[tet[2]],
                        cube_verts[tet[3]],
                        TET_SENTINEL,
                        TET_SENTINEL,
                        TET_SENTINEL,
                        TET_SENTINEL,
                    ]);
                    lat_scalars.push(lat);
                    lon_scalars.push(lon);
                    radial_scalars.push(raw_len);
                    direct_colors.push(color);
                }
            }
        }
    }

    let mut data = VolumeMeshData::default();
    data.positions = positions.to_vec();
    data.cells = cells;
    data.cell_scalars
        .insert("latitude".to_string(), lat_scalars);
    data.cell_scalars
        .insert("longitude".to_string(), lon_scalars);
    data.cell_scalars
        .insert("radial".to_string(), radial_scalars);
    data.cell_colors.insert("direct".to_string(), direct_colors);
    data
}

// ---------------------------------------------------------------------------
// App impl
// ---------------------------------------------------------------------------

impl App {
    /// Upload both hex and tet sphere meshes.
    pub(crate) fn build_vm_scene(&mut self, renderer: &mut ViewportRenderer) {
        let positions = sphere_vertex_positions();

        let hex_data = build_hex_mesh(&positions);
        self.vm_hex_index = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &hex_data)
            .expect("vm hex upload");

        let tet_data = build_tet_mesh(&positions);
        self.vm_tet_index = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &tet_data)
            .expect("vm tet upload");

        self.vm_built = true;
    }

    /// Build a [`SceneRenderItem`] for the active volume mesh.
    pub(crate) fn vm_scene_items(&self) -> Vec<SceneRenderItem> {
        if !self.vm_built {
            return vec![];
        }

        let mesh_index = match self.vm_mode {
            VmMode::Hex => self.vm_hex_index,
            VmMode::Tet => self.vm_tet_index,
        };

        let (active_attribute, colormap_id) = match self.vm_field {
            VmField::Latitude => (
                Some(AttributeRef {
                    name: "latitude".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColormapId(self.vm_colormap as usize)),
            ),
            VmField::Longitude => (
                Some(AttributeRef {
                    name: "longitude".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColormapId(self.vm_colormap as usize)),
            ),
            VmField::Radial => (
                Some(AttributeRef {
                    name: "radial".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColormapId(self.vm_colormap as usize)),
            ),
            VmField::DirectColor => (
                Some(AttributeRef {
                    name: "direct".to_string(),
                    kind: AttributeKind::FaceColor,
                }),
                None,
            ),
        };

        let mut item = SceneRenderItem::default();
        item.mesh_index = mesh_index;
        item.active_attribute = active_attribute;
        item.colormap_id = colormap_id;
        vec![item]
    }

    /// Lighting for the volume mesh showcase.
    pub(crate) fn vm_lighting() -> LightingSettings {
        LightingSettings {
            hemisphere_intensity: 0.3,
            sky_color: [1.0, 1.0, 1.0],
            ground_color: [0.3, 0.3, 0.4],
            ..LightingSettings::default()
        }
    }

    // -------------------------------------------------------------------------
    // Controls panel
    // -------------------------------------------------------------------------

    pub(crate) fn controls_volume_mesh(&mut self, ui: &mut egui::Ui) {
        ui.label("Cell type:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.vm_mode, VmMode::Hex, "Hex (27 cells)");
            ui.radio_value(&mut self.vm_mode, VmMode::Tet, "Tet (162 cells)");
        });

        ui.separator();
        ui.label("Field:");
        for (field, label) in [
            (VmField::Latitude, "Latitude (scalar)"),
            (VmField::Longitude, "Longitude (scalar)"),
            (VmField::Radial, "Radial distance (scalar)"),
            (VmField::DirectColor, "Direct cell colors (RGBA)"),
        ] {
            ui.radio_value(&mut self.vm_field, field, label);
        }

        if !matches!(self.vm_field, VmField::DirectColor) {
            ui.separator();
            ui.label("Colormap:");
            ui.horizontal_wrapped(|ui| {
                for cm in [
                    BuiltinColormap::Viridis,
                    BuiltinColormap::Plasma,
                    BuiltinColormap::Coolwarm,
                    BuiltinColormap::Rainbow,
                ] {
                    ui.radio_value(&mut self.vm_colormap, cm, format!("{cm:?}"));
                }
            });
        }

        ui.separator();
        let n_cells = match self.vm_mode {
            VmMode::Hex => 27,
            VmMode::Tet => 162,
        };
        ui.label(format!("{n_cells} cells total · boundary surface only"));
        ui.label("Interior faces are automatically discarded.");
    }
}
