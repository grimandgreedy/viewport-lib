//! Showcase 26: Unstructured Volume Meshes
//!
//! Demonstrates Phase 9 : `VolumeMeshData` topology processing.
//!
//! A 3×3×3 structured grid of cells is visualised via its extracted boundary.
//! Interior faces shared by two cells are discarded automatically by
//! [`upload_volume_mesh_data`]. Per-cell scalars and colors are remapped to the
//! boundary faces so the existing Phase 2 face-coloring path applies colormaps
//! cell-by-cell with no new GPU work.
//!
//! ## Two modes
//! - **Hex sphere** : 27 hexahedral cells mapped through a cube-to-sphere warp;
//!   boundary = 54 quads = 108 triangles.
//! - **Tet sphere** : same grid split into 6 tets per cube (Freudenthal) -> 162 tets;
//!   boundary = 108 triangles with a different surface triangulation.
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
    AttributeKind, AttributeRef, BackfacePolicy, BuiltinColormap, ColormapId, LightingSettings,
    SceneRenderItem,
    ClipObject, ClipShape, TET_SENTINEL, ViewportRenderer, VolumeMeshData,
};

// ---------------------------------------------------------------------------
// Sub-mode
// ---------------------------------------------------------------------------

/// Which cell type is currently displayed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum VmMode {
    /// 27 hexahedral cells on a sphere.
    Hex,
    /// 162 tetrahedral cells on a sphere (6 tets per cube, 3×3×3 grid).
    Tet,
    /// 6 tet cells from a single unit cube (no projection) - baseline check.
    TetSmall,
    /// 48 tet cells on a 2×2×2 sphere (smaller grid).
    TetSphere2,
    /// 162 tet cells, axis-aligned box (no sphere projection).
    TetBox,
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

/// Axis for the showcase clip plane.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum VmClipAxis {
    X,
    Y,
    Z,
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
/// Vertices are placed at :1.5, :0.5, 0.5, 1.5 on each axis (half-step
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

/// Map a cube point in `[-1, 1]^3` to a sphere using the standard
/// equal-angle cube-to-sphere approximation.
fn cube_to_sphere([x, y, z]: [f32; 3]) -> [f32; 3] {
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    [
        x * (1.0 - 0.5 * (y2 + z2) + (y2 * z2) / 3.0).sqrt(),
        y * (1.0 - 0.5 * (z2 + x2) + (z2 * x2) / 3.0).sqrt(),
        z * (1.0 - 0.5 * (x2 + y2) + (x2 * y2) / 3.0).sqrt(),
    ]
}

/// Generate a 4×4×4 cube lattice and warp it into a sphere while preserving
/// the interior layers, so the hex cells remain valid volume elements.
fn cube_sphere_vertex_positions() -> Vec<[f32; 3]> {
    let mut pos = Vec::with_capacity(GRID_V * GRID_V * GRID_V);

    for iz in 0..GRID_V {
        for iy in 0..GRID_V {
            for ix in 0..GRID_V {
                let p = [
                    (ix as f32 - 1.5) / 1.5,
                    (iy as f32 - 1.5) / 1.5,
                    (iz as f32 - 1.5) / 1.5,
                ];
                let s = cube_to_sphere(p);
                pos.push([s[0] * SPHERE_R, s[1] * SPHERE_R, s[2] * SPHERE_R]);
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

                // Latitude: :1 (south pole) to +1 (north pole) -> 0..1
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
// Diagnostic builders
// ---------------------------------------------------------------------------

/// Generate vertex positions for an N×N×N axis-aligned grid (no projection).
/// Vertices at integer coordinates [0, N].
fn box_vertex_positions(n: usize) -> Vec<[f32; 3]> {
    let nv = n + 1;
    let mut pos = Vec::with_capacity(nv * nv * nv);
    for iz in 0..nv {
        for iy in 0..nv {
            for ix in 0..nv {
                pos.push([ix as f32, iy as f32, iz as f32]);
            }
        }
    }
    pos
}

/// Vertex index for an N+1 vertex-per-axis grid.
fn vid_n(ix: usize, iy: usize, iz: usize, nv: usize) -> u32 {
    (iz * nv * nv + iy * nv + ix) as u32
}

/// Build a tet mesh from an arbitrary grid of size N using the given vertex positions.
fn build_tet_mesh_n(n: usize, positions: &[[f32; 3]]) -> VolumeMeshData {
    let nv = n + 1;
    let total_cells = n * n * n * 6;
    let mut cells: Vec<[u32; 8]> = Vec::with_capacity(total_cells);
    let mut lat_scalars: Vec<f32> = Vec::with_capacity(total_cells);
    let mut lon_scalars: Vec<f32> = Vec::with_capacity(total_cells);
    let mut radial_scalars: Vec<f32> = Vec::with_capacity(total_cells);
    let mut direct_colors: Vec<[f32; 4]> = Vec::with_capacity(total_cells);

    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let cube_verts = [
                    vid_n(ix, iy, iz, nv),
                    vid_n(ix + 1, iy, iz, nv),
                    vid_n(ix + 1, iy, iz + 1, nv),
                    vid_n(ix, iy, iz + 1, nv),
                    vid_n(ix, iy + 1, iz, nv),
                    vid_n(ix + 1, iy + 1, iz, nv),
                    vid_n(ix + 1, iy + 1, iz + 1, nv),
                    vid_n(ix, iy + 1, iz + 1, nv),
                ];

                // Centroid: average of all 8 cube vertices.
                let mut cx = 0.0f32;
                let mut cy = 0.0f32;
                let mut cz = 0.0f32;
                for &vi in &cube_verts {
                    let p = positions[vi as usize];
                    cx += p[0];
                    cy += p[1];
                    cz += p[2];
                }
                cx /= 8.0;
                cy /= 8.0;
                cz /= 8.0;
                let raw_len = (cx * cx + cy * cy + cz * cz).sqrt().max(1e-6);

                let lat = cy / raw_len * 0.5 + 0.5;
                let lon = cz.atan2(cx) / std::f32::consts::TAU + 0.5;
                let color = hsv_to_rgba(lon, 0.8, 0.85);

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

/// Single 1×1×1 cube split into 6 tets, no projection.
fn build_tet_small() -> VolumeMeshData {
    let positions = box_vertex_positions(1);
    build_tet_mesh_n(1, &positions)
}

/// 2×2×2 tet grid projected onto a sphere.
///
/// Vertices placed at half-integer offsets {-1.5, -0.5, 0.5} so no vertex
/// falls at the origin (safe for normalization).
fn build_tet_sphere2() -> VolumeMeshData {
    const N2: usize = 2;
    const NV2: usize = N2 + 1;
    let mut positions = Vec::with_capacity(NV2 * NV2 * NV2);
    for iz in 0..NV2 {
        for iy in 0..NV2 {
            for ix in 0..NV2 {
                // Half-integer offset: same pattern as sphere_vertex_positions().
                let x = ix as f32 - 1.5;
                let y = iy as f32 - 1.5;
                let z = iz as f32 - 1.5;
                let len = (x * x + y * y + z * z).sqrt();
                let s = SPHERE_R / len;
                positions.push([x * s, y * s, z * s]);
            }
        }
    }
    build_tet_mesh_n(N2, &positions)
}

/// 3×3×3 tet grid as a flat axis-aligned box (no sphere projection).
fn build_tet_box() -> VolumeMeshData {
    let positions = box_vertex_positions(GRID_N);
    build_tet_mesh_n(GRID_N, &positions)
}

// ---------------------------------------------------------------------------
// App impl
// ---------------------------------------------------------------------------

impl App {
    /// Upload all volume mesh variants.
    pub(crate) fn build_vm_scene(&mut self, renderer: &mut ViewportRenderer) {
        let hex_positions = cube_sphere_vertex_positions();
        let positions = sphere_vertex_positions();

        let hex_data = build_hex_mesh(&hex_positions);
        self.vm_hex_index = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &hex_data)
            .expect("vm hex upload");

        let tet_data = build_tet_mesh(&positions);
        self.vm_tet_index = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &tet_data)
            .expect("vm tet upload");

        let tet_small_data = build_tet_small();
        self.vm_tet_small_index = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &tet_small_data)
            .expect("vm tet small upload");

        let tet_sphere2_data = build_tet_sphere2();
        self.vm_tet_sphere2_index = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &tet_sphere2_data)
            .expect("vm tet sphere2 upload");

        let tet_box_data = build_tet_box();
        self.vm_tet_box_index = renderer
            .resources_mut()
            .upload_volume_mesh_data(&self.device, &tet_box_data)
            .expect("vm tet box upload");

        self.vm_built = true;
    }

    /// Build a [`SceneRenderItem`] for the active volume mesh.
    pub(crate) fn vm_scene_items(&self) -> Vec<SceneRenderItem> {
        if !self.vm_built {
            return vec![];
        }

        let mesh_id = match self.vm_mode {
            VmMode::Hex => self.vm_hex_index,
            VmMode::Tet => self.vm_tet_index,
            VmMode::TetSmall => self.vm_tet_small_index,
            VmMode::TetSphere2 => self.vm_tet_sphere2_index,
            VmMode::TetBox => self.vm_tet_box_index,
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
        item.mesh_id = mesh_id;
        item.active_attribute = active_attribute;
        item.colormap_id = colormap_id;
        item.material.backface_policy = BackfacePolicy::Identical;
        vec![item]
    }

    pub(crate) fn vm_clip_objects(&self) -> Vec<ClipObject> {
        if !self.vm_clip_on {
            return vec![];
        }

        let normal = match self.vm_clip_axis {
            VmClipAxis::X => [1.0, 0.0, 0.0],
            VmClipAxis::Y => [0.0, 1.0, 0.0],
            VmClipAxis::Z => [0.0, 0.0, 1.0],
        };
        let mut clip = ClipObject::plane(normal, self.vm_clip_offset);
        clip.shape = ClipShape::Plane {
            normal,
            distance: self.vm_clip_offset,
            cap_color: Some([0.72, 0.78, 0.86, 1.0]),
        };
        clip.color = Some([0.85, 0.9, 1.0, 0.18]);
        clip.extent = 3.5;
        vec![clip]
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
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.vm_mode, VmMode::Hex, "Hex sphere (27)");
            ui.radio_value(&mut self.vm_mode, VmMode::Tet, "Tet sphere 3³ (162)");
            ui.radio_value(&mut self.vm_mode, VmMode::TetSphere2, "Tet sphere 2³ (48)");
            ui.radio_value(&mut self.vm_mode, VmMode::TetBox, "Tet box 3³ (162)");
            ui.radio_value(&mut self.vm_mode, VmMode::TetSmall, "Tet cube 1³ (6)");
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
                    BuiltinColormap::Magma,
                    BuiltinColormap::Inferno,
                    BuiltinColormap::Turbo,
                    BuiltinColormap::Coolwarm,
                    BuiltinColormap::RdBu,
                    BuiltinColormap::Rainbow,
                    BuiltinColormap::Jet,
                ] {
                    ui.radio_value(&mut self.vm_colormap, cm, format!("{cm:?}"));
                }
            });
        }

        ui.separator();
        ui.checkbox(&mut self.vm_wireframe, "Wireframe");

        ui.separator();
        ui.checkbox(&mut self.vm_clip_on, "Clip plane");
        if self.vm_clip_on {
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.vm_clip_axis, VmClipAxis::X, "X");
                ui.radio_value(&mut self.vm_clip_axis, VmClipAxis::Y, "Y");
                ui.radio_value(&mut self.vm_clip_axis, VmClipAxis::Z, "Z");
            });
            ui.add(
                egui::Slider::new(&mut self.vm_clip_offset, -1.75..=1.75)
                    .text("Offset")
                    .step_by(0.01),
            );
            ui.label("Keeps the positive side of the plane to reveal interior cells.");
        }

        ui.separator();
        let (n_cells, note) = match self.vm_mode {
            VmMode::Hex => (27, "3³ hexes, cube-to-sphere warp"),
            VmMode::Tet => (162, "3³×6 tets on sphere"),
            VmMode::TetSphere2 => (48, "2³×6 tets on sphere"),
            VmMode::TetBox => (162, "3³×6 tets, flat box"),
            VmMode::TetSmall => (6, "1³×6 tets, unit cube"),
        };
        ui.label(format!("{n_cells} cells · {note}"));
        ui.label("Interior faces are automatically discarded.");
    }
}
