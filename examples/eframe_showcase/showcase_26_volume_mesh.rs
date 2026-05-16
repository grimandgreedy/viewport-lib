//! Showcase 26: Unstructured Volume Meshes
//!
//! Demonstrates Phase 9 : `VolumeMeshData` topology processing.
//!
//! A 3×3×3 structured grid of cells is visualised via its extracted boundary.
//! Interior faces shared by two cells are discarded automatically by
//! [`upload_volume_mesh_data`]. Per-cell scalars and colours are remapped to the
//! boundary faces so the existing Phase 2 face-colouring path applies colourmaps
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
    AttributeKind, AttributeRef, BackfacePolicy, BuiltinColourmap, CELL_SENTINEL, ClipObject,
    ClipShape, ColourmapId, FrameData, LightingSettings, ProjectedTetId,
    SceneRenderItem, TransparentVolumeMeshItem, ViewportRenderer, VolumeMeshData, VolumeMeshItem,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct VmState {
    pub built: bool,
    pub mode: VmMode,
    pub tet_item: Option<VolumeMeshItem>,
    pub hex_item: Option<VolumeMeshItem>,
    pub tet_small_item: Option<VolumeMeshItem>,
    pub tet_box_item: Option<VolumeMeshItem>,
    pub pyramid_item: Option<VolumeMeshItem>,
    pub wedge_item: Option<VolumeMeshItem>,
    pub colourmap: BuiltinColourmap,
    pub field: VmField,
    pub wireframe: bool,
    pub clip_on: bool,
    pub clip_offset: f32,
    /// Elevation in degrees: angle above the horizontal XZ plane (-90 = down, 0 = horizontal, 90 = up).
    pub clip_elevation: f32,
    /// Azimuth in degrees: rotation in the XZ plane measured from +Z toward +X.
    pub clip_azimuth: f32,
    /// CPU-clipped volume mesh item; allocated lazily on first clip.
    pub clipped_item: Option<VolumeMeshItem>,
    /// Projected-tet handles per cell type (uploaded at startup with raw positions).
    pub pt_hex_id: Option<ProjectedTetId>,
    pub pt_tet_id: Option<ProjectedTetId>,
    pub pt_tet_small_id: Option<ProjectedTetId>,
    pub pt_tet_box_id: Option<ProjectedTetId>,
    pub pt_pyramid_id: Option<ProjectedTetId>,
    pub pt_wedge_id: Option<ProjectedTetId>,
    /// Whether to render in transparent (projected-tet) mode.
    pub transparent: bool,
    /// Beer-Lambert extinction coefficient for transparent mode.
    pub density: f32,
    /// Scalar field last used for the PT upload; triggers a rebuild when it differs from field.
    pub pt_field: VmField,
    /// Colourmap last used for the PT upload; triggers a rebuild when it differs from colourmap.
    pub pt_colourmap: BuiltinColourmap,
}

impl Default for VmState {
    fn default() -> Self {
        Self {
            built: false,
            mode: VmMode::Hex,
            tet_item: None,
            hex_item: None,
            tet_small_item: None,
            tet_box_item: None,
            pyramid_item: None,
            wedge_item: None,
            colourmap: BuiltinColourmap::Viridis,
            field: VmField::Latitude,
            wireframe: false,
            clip_on: true,
            clip_offset: 0.0,
            clip_elevation: 0.0,
            clip_azimuth: 0.0,
            clipped_item: None,
            pt_hex_id: None,
            pt_tet_id: None,
            pt_tet_small_id: None,
            pt_tet_box_id: None,
            pt_pyramid_id: None,
            pt_wedge_id: None,
            transparent: false,
            density: 0.5,
            pt_field: VmField::Latitude,
            pt_colourmap: BuiltinColourmap::Viridis,
        }
    }
}

/// Map a VmField to the VolumeMeshData scalar attribute name used for projected-tet rendering.
/// DirectColour has no scalar attribute; falls back to "radial".
pub(crate) fn vm_pt_scalar_attr(field: VmField) -> &'static str {
    match field {
        VmField::Latitude => "latitude",
        VmField::Longitude => "longitude",
        VmField::Radial | VmField::DirectColour => "radial",
    }
}

/// Build the VolumeMeshData for the given mode using positions appropriate for PT rendering.
///
/// Hex and Pyramid use `cube_sphere_vertex_positions` so the transparent volume matches
/// the sphere shape of the opaque mesh. Tet and Wedge use raw box positions because
/// sphere-projected positions collapse their Freudenthal-decomposed tets to zero volume.
pub(crate) fn pt_data_for_mode(mode: VmMode) -> VolumeMeshData {
    match mode {
        VmMode::Hex => build_hex_mesh(&cube_sphere_vertex_positions()),
        VmMode::Tet => build_tet_mesh(&box_vertex_positions(GRID_N)),
        VmMode::TetSmall => build_tet_small(),
        VmMode::TetBox => build_tet_box(),
        VmMode::Pyramid => build_pyramid_mesh(&cube_sphere_vertex_positions()),
        VmMode::Wedge => build_wedge_mesh(&box_vertex_positions(GRID_N)),
    }
}

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
    /// 162 tet cells, axis-aligned box (no sphere projection).
    TetBox,
    /// 162 pyramid cells on a sphere (6 pyramids per cube, 3×3×3 grid).
    Pyramid,
    /// 54 wedge (triangular prism) cells on a sphere (2 wedges per cube, 3×3×3 grid).
    Wedge,
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
    /// Direct per-cell RGBA colour (no colourmap).
    DirectColour,
}

// ---------------------------------------------------------------------------
// App state fields (declared in main App struct):
//   vm_built:        bool
//   vm_mode:         VmMode
//   vm_tet_index:    usize
//   vm_hex_index:    usize
//   vm_colourmap:     BuiltinColourmap
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
    let mut direct_colours: Vec<[f32; 4]> = Vec::with_capacity(n);

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
                // Clamp to avoid division by zero for the center cell whose
                // centroid sits exactly at the origin.
                let raw_len = (cx * cx + cy * cy + cz * cz).sqrt().max(1e-6);

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
                direct_colours.push(hsv_to_rgba(lon, sat, 0.9));
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
    data.cell_colours.insert("direct".to_string(), direct_colours);
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
    let mut direct_colours: Vec<[f32; 4]> = Vec::with_capacity(n);

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
                let raw_len = (cx * cx + cy * cy + cz * cz).sqrt().max(1e-6);
                let s = SPHERE_R / raw_len;
                let py = cy * s;

                let lat = py / SPHERE_R * 0.5 + 0.5;
                let lon = (cz).atan2(cx) / std::f32::consts::TAU + 0.5;
                let sat = 0.5 + 0.5 * (py / SPHERE_R).abs();
                let colour = hsv_to_rgba(lon, sat, 0.85);

                for tet in &TET_LOCAL {
                    cells.push([
                        cube_verts[tet[0]],
                        cube_verts[tet[1]],
                        cube_verts[tet[2]],
                        cube_verts[tet[3]],
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                    ]);
                    lat_scalars.push(lat);
                    lon_scalars.push(lon);
                    radial_scalars.push(raw_len);
                    direct_colours.push(colour);
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
    data.cell_colours.insert("direct".to_string(), direct_colours);
    data
}

// ---------------------------------------------------------------------------
// Diagnostic builders
// ---------------------------------------------------------------------------

/// Generate vertex positions for an N×N×N axis-aligned grid centred at the origin.
/// Vertices run from -N/2 to +N/2 on each axis.
fn box_vertex_positions(n: usize) -> Vec<[f32; 3]> {
    let nv = n + 1;
    let half = n as f32 / 2.0;
    let mut pos = Vec::with_capacity(nv * nv * nv);
    for iz in 0..nv {
        for iy in 0..nv {
            for ix in 0..nv {
                pos.push([ix as f32 - half, iy as f32 - half, iz as f32 - half]);
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
    let mut direct_colours: Vec<[f32; 4]> = Vec::with_capacity(total_cells);

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
                let colour = hsv_to_rgba(lon, 0.8, 0.85);

                for tet in &TET_LOCAL {
                    cells.push([
                        cube_verts[tet[0]],
                        cube_verts[tet[1]],
                        cube_verts[tet[2]],
                        cube_verts[tet[3]],
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                        CELL_SENTINEL,
                    ]);
                    lat_scalars.push(lat);
                    lon_scalars.push(lon);
                    radial_scalars.push(raw_len);
                    direct_colours.push(colour);
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
    data.cell_colours.insert("direct".to_string(), direct_colours);
    data
}

// ---------------------------------------------------------------------------
// Pyramid mesh builder
// ---------------------------------------------------------------------------
//
// Each cube is split into 6 pyramids by adding a center vertex and connecting
// it to each of the 6 quad faces.  Adjacent cubes share quad faces correctly
// because the shared face uses the same 4 corner vertices in both cubes.
//
// Face winding follows the hex face table (outward from the cube face toward
// the center), relying on the extractor's geometric winding correction.

fn build_pyramid_mesh(positions: &[[f32; 3]]) -> VolumeMeshData {
    let n_cells = GRID_N * GRID_N * GRID_N * 6;
    let mut all_positions = positions.to_vec();
    let mut data = VolumeMeshData::default();
    data.cells.reserve(n_cells);
    let mut lat_scalars: Vec<f32> = Vec::with_capacity(n_cells);
    let mut lon_scalars: Vec<f32> = Vec::with_capacity(n_cells);
    let mut radial_scalars: Vec<f32> = Vec::with_capacity(n_cells);
    let mut direct_colours: Vec<[f32; 4]> = Vec::with_capacity(n_cells);

    for iz in 0..GRID_N {
        for iy in 0..GRID_N {
            for ix in 0..GRID_N {
                let c = [
                    vid(ix, iy, iz),
                    vid(ix + 1, iy, iz),
                    vid(ix + 1, iy, iz + 1),
                    vid(ix, iy, iz + 1),
                    vid(ix, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz),
                    vid(ix + 1, iy + 1, iz + 1),
                    vid(ix, iy + 1, iz + 1),
                ];

                // Center vertex: average of the 8 projected sphere positions.
                let mut px = 0.0f32;
                let mut py_sum = 0.0f32;
                let mut pz = 0.0f32;
                for &vi in &c {
                    let p = positions[vi as usize];
                    px += p[0];
                    py_sum += p[1];
                    pz += p[2];
                }
                let center_idx = all_positions.len() as u32;
                all_positions.push([px / 8.0, py_sum / 8.0, pz / 8.0]);

                let [rcx, rcy, rcz] = cell_centroid_raw(ix, iy, iz);
                let raw_len = (rcx * rcx + rcy * rcy + rcz * rcz).sqrt().max(1e-6);
                let s = SPHERE_R / raw_len;
                let py = rcy * s;
                let lat = py / SPHERE_R * 0.5 + 0.5;
                let lon = rcz.atan2(rcx) / std::f32::consts::TAU + 0.5;
                let sat = 0.5 + 0.5 * (py / SPHERE_R).abs();
                let colour = hsv_to_rgba(lon, sat, 0.88);

                // 6 pyramids: one per hex face.  Base quad winding matches the
                // hex face table so outward normals face away from the cube.
                data.push_pyramid([c[0], c[1], c[2], c[3]], center_idx); // bottom
                data.push_pyramid([c[4], c[7], c[6], c[5]], center_idx); // top
                data.push_pyramid([c[0], c[4], c[5], c[1]], center_idx); // front
                data.push_pyramid([c[2], c[6], c[7], c[3]], center_idx); // back
                data.push_pyramid([c[0], c[3], c[7], c[4]], center_idx); // left
                data.push_pyramid([c[1], c[5], c[6], c[2]], center_idx); // right

                for _ in 0..6 {
                    lat_scalars.push(lat);
                    lon_scalars.push(lon);
                    radial_scalars.push(raw_len);
                    direct_colours.push(colour);
                }
            }
        }
    }

    data.positions = all_positions;
    data.cell_scalars
        .insert("latitude".to_string(), lat_scalars);
    data.cell_scalars
        .insert("longitude".to_string(), lon_scalars);
    data.cell_scalars
        .insert("radial".to_string(), radial_scalars);
    data.cell_colours.insert("direct".to_string(), direct_colours);
    data
}

// ---------------------------------------------------------------------------
// Wedge (triangular prism) mesh builder
// ---------------------------------------------------------------------------
//
// Each cube is split into 2 conforming wedges by cutting along the XY diagonal
// (vertex 0 to vertex [ix+1,iy+1]).  Adjacent cubes share the diagonal cut
// consistently so all interior faces cancel correctly.
//
// Extrusion direction is along Z (iz to iz+1).

fn build_wedge_mesh(positions: &[[f32; 3]]) -> VolumeMeshData {
    let n_cells = GRID_N * GRID_N * GRID_N * 2;
    let mut data = VolumeMeshData::default();
    data.cells.reserve(n_cells);
    let mut lat_scalars: Vec<f32> = Vec::with_capacity(n_cells);
    let mut lon_scalars: Vec<f32> = Vec::with_capacity(n_cells);
    let mut radial_scalars: Vec<f32> = Vec::with_capacity(n_cells);
    let mut direct_colours: Vec<[f32; 4]> = Vec::with_capacity(n_cells);

    for iz in 0..GRID_N {
        for iy in 0..GRID_N {
            for ix in 0..GRID_N {
                let a = vid(ix, iy, iz);
                let b = vid(ix + 1, iy, iz);
                let c = vid(ix + 1, iy + 1, iz);
                let d = vid(ix, iy + 1, iz);
                let a1 = vid(ix, iy, iz + 1);
                let b1 = vid(ix + 1, iy, iz + 1);
                let c1 = vid(ix + 1, iy + 1, iz + 1);
                let d1 = vid(ix, iy + 1, iz + 1);

                // Two wedges sharing the diagonal [a,c] / [a1,c1].
                data.push_wedge([a, b, c], [a1, b1, c1]);
                data.push_wedge([a, c, d], [a1, c1, d1]);

                let [rcx, rcy, rcz] = cell_centroid_raw(ix, iy, iz);
                let raw_len = (rcx * rcx + rcy * rcy + rcz * rcz).sqrt().max(1e-6);
                let s = SPHERE_R / raw_len;
                let py = rcy * s;
                let lat = py / SPHERE_R * 0.5 + 0.5;
                let lon = rcz.atan2(rcx) / std::f32::consts::TAU + 0.5;
                let sat = 0.5 + 0.5 * (py / SPHERE_R).abs();
                let colour = hsv_to_rgba(lon, sat, 0.87);

                for _ in 0..2 {
                    lat_scalars.push(lat);
                    lon_scalars.push(lon);
                    radial_scalars.push(raw_len);
                    direct_colours.push(colour);
                }
            }
        }
    }

    data.positions = positions.to_vec();
    data.cell_scalars
        .insert("latitude".to_string(), lat_scalars);
    data.cell_scalars
        .insert("longitude".to_string(), lon_scalars);
    data.cell_scalars
        .insert("radial".to_string(), radial_scalars);
    data.cell_colours.insert("direct".to_string(), direct_colours);
    data
}

/// Single 1×1×1 cube split into 6 tets, no projection.
fn build_tet_small() -> VolumeMeshData {
    let positions = box_vertex_positions(1);
    build_tet_mesh_n(1, &positions)
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

        let mut upload = |data: &VolumeMeshData| -> VolumeMeshItem {
            let (mesh_id, face_to_cell) = renderer
                .resources_mut()
                .upload_volume_mesh_data(&self.device, data)
                .expect("vm upload");
            let mut item = VolumeMeshItem::new(mesh_id, face_to_cell);
            item.material.backface_policy = BackfacePolicy::Identical;
            item
        };

        let hex_data = build_hex_mesh(&hex_positions);
        self.vm_state.hex_item = Some(upload(&hex_data));

        let tet_data = build_tet_mesh(&positions);
        self.vm_state.tet_item = Some(upload(&tet_data));

        let tet_small_data = build_tet_small();
        self.vm_state.tet_small_item = Some(upload(&tet_small_data));

        let tet_box_data = build_tet_box();
        self.vm_state.tet_box_item = Some(upload(&tet_box_data));

        let pyramid_data = build_pyramid_mesh(&hex_positions);
        self.vm_state.pyramid_item = Some(upload(&pyramid_data));

        let wedge_data = build_wedge_mesh(&positions);
        self.vm_state.wedge_item = Some(upload(&wedge_data));

        // Colourmaps must be ready before the PT bind group is created.
        renderer
            .resources_mut()
            .ensure_colourmaps_initialized(&self.device, &self.queue);

        // Upload projected-tet meshes for each cell type.
        // Hex and Pyramid use cube_sphere positions so the transparent sphere shape
        // matches the opaque surface. Tet/Wedge use box positions because pure radial
        // sphere projection puts all vertices on the sphere surface, collapsing tet
        // volumes to zero.
        let attr = vm_pt_scalar_attr(self.vm_state.field);
        let colourmap_id = ColourmapId(self.vm_state.colourmap as usize);

        let mut pt_upload = |data: &VolumeMeshData| -> Option<ProjectedTetId> {
            renderer
                .resources_mut()
                .upload_projected_tet_mesh(&self.device, data, attr, colourmap_id)
                .ok()
                .map(|(id, _, _)| id)
        };

        self.vm_state.pt_hex_id = pt_upload(&pt_data_for_mode(VmMode::Hex));
        self.vm_state.pt_tet_id = pt_upload(&pt_data_for_mode(VmMode::Tet));
        self.vm_state.pt_tet_small_id = pt_upload(&pt_data_for_mode(VmMode::TetSmall));
        self.vm_state.pt_tet_box_id = pt_upload(&pt_data_for_mode(VmMode::TetBox));
        self.vm_state.pt_pyramid_id = pt_upload(&pt_data_for_mode(VmMode::Pyramid));
        self.vm_state.pt_wedge_id = pt_upload(&pt_data_for_mode(VmMode::Wedge));

        self.vm_state.built = true;
    }

    /// Compute the clip plane normal from elevation and azimuth.
    ///
    /// elevation: angle above the horizontal XZ plane in degrees.
    ///   0 = horizontal cut (normal has no Y component)
    ///   +90 = normal points straight up (+Y)
    ///
    /// azimuth: rotation in the XZ plane in degrees, measured from +Z toward +X.
    ///   0   -> normal = [0, 0, 1] at elevation 0 (XY-plane cut)
    ///   90  -> normal = [1, 0, 0] at elevation 0 (YZ-plane cut)
    ///   180 -> normal = [0, 0, -1]
    fn vm_clip_normal(&self) -> [f32; 3] {
        let el = self.vm_state.clip_elevation.to_radians();
        let az = self.vm_state.clip_azimuth.to_radians();
        let (sel, cel) = (el.sin(), el.cos());
        let (saz, caz) = (az.sin(), az.cos());
        [saz * cel, sel, caz * cel]
    }

    /// Encode the active clip normal and offset as `[nx, ny, nz, d]`.
    ///
    /// Matches the `ClipPlanesUniform` format: a point `p` is on the kept side
    /// when `dot(p, n) + d >= 0`.
    pub(crate) fn vm_clip_plane(&self) -> [f32; 4] {
        let [nx, ny, nz] = self.vm_clip_normal();
        [nx, ny, nz, self.vm_state.clip_offset]
    }

    /// Rebuild the raw `VolumeMeshData` for the currently active mode.
    ///
    /// Intentionally cheap enough to call every frame for the showcase mesh
    /// sizes (27–162 cells).
    pub(crate) fn vm_active_data(&self) -> VolumeMeshData {
        match self.vm_state.mode {
            VmMode::Hex => {
                let positions = cube_sphere_vertex_positions();
                build_hex_mesh(&positions)
            }
            VmMode::Tet => {
                let positions = sphere_vertex_positions();
                build_tet_mesh(&positions)
            }
            VmMode::TetSmall => build_tet_small(),
            VmMode::TetBox => build_tet_box(),
            VmMode::Pyramid => {
                let positions = cube_sphere_vertex_positions();
                build_pyramid_mesh(&positions)
            }
            VmMode::Wedge => {
                let positions = sphere_vertex_positions();
                build_wedge_mesh(&positions)
            }
        }
    }

    /// Build a [`SceneRenderItem`] for the active volume mesh.
    ///
    /// When clip is on and the clipped item is ready, uses that instead of the
    /// static boundary mesh.
    pub(crate) fn vm_scene_items(&self) -> Vec<SceneRenderItem> {
        if !self.vm_state.built || self.vm_state.transparent {
            return vec![];
        }

        // Pick the active item: clipped if available and clipping is on,
        // otherwise the static boundary item for the current mode.
        let active = if self.vm_state.clip_on {
            self.vm_state
                .clipped_item
                .as_ref()
                .or_else(|| self.vm_active_item())
        } else {
            self.vm_active_item()
        };
        let Some(item) = active else { return vec![] };

        let (active_attribute, colourmap_id) = match self.vm_state.field {
            VmField::Latitude => (
                Some(AttributeRef {
                    name: "latitude".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColourmapId(self.vm_state.colourmap as usize)),
            ),
            VmField::Longitude => (
                Some(AttributeRef {
                    name: "longitude".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColourmapId(self.vm_state.colourmap as usize)),
            ),
            VmField::Radial => (
                Some(AttributeRef {
                    name: "radial".to_string(),
                    kind: AttributeKind::Face,
                }),
                Some(ColourmapId(self.vm_state.colourmap as usize)),
            ),
            VmField::DirectColour => (
                Some(AttributeRef {
                    name: "direct".to_string(),
                    kind: AttributeKind::FaceColour,
                }),
                None,
            ),
        };

        let mut render_item = item.to_render_item();
        render_item.active_attribute = active_attribute;
        render_item.colourmap_id = colourmap_id;
        vec![render_item]
    }

    /// Return a reference to the static [`VolumeMeshItem`] for the current mode.
    fn vm_active_item(&self) -> Option<&VolumeMeshItem> {
        match self.vm_state.mode {
            VmMode::Hex => self.vm_state.hex_item.as_ref(),
            VmMode::Tet => self.vm_state.tet_item.as_ref(),
            VmMode::TetSmall => self.vm_state.tet_small_item.as_ref(),
            VmMode::TetBox => self.vm_state.tet_box_item.as_ref(),
            VmMode::Pyramid => self.vm_state.pyramid_item.as_ref(),
            VmMode::Wedge => self.vm_state.wedge_item.as_ref(),
        }
    }

    /// Rebuild all projected-tet meshes with a new scalar field and colourmap.
    ///
    /// Called when the user changes `vm_field` or `vm_colourmap` while transparent
    /// mode has been used at least once. Each mode keeps its own PT slot so switching
    /// cell type doesn't re-upload.
    pub(crate) fn rebuild_pt_meshes(
        &mut self,
        renderer: &mut ViewportRenderer,
        device: &wgpu::Device,
        field: VmField,
        colourmap: BuiltinColourmap,
    ) {
        let attr = vm_pt_scalar_attr(field);
        let colourmap_id = ColourmapId(colourmap as usize);

        for mode in [
            VmMode::Hex,
            VmMode::Tet,
            VmMode::TetSmall,
            VmMode::TetBox,
            VmMode::Pyramid,
            VmMode::Wedge,
        ] {
            let id = match mode {
                VmMode::Hex => self.vm_state.pt_hex_id,
                VmMode::Tet => self.vm_state.pt_tet_id,
                VmMode::TetSmall => self.vm_state.pt_tet_small_id,
                VmMode::TetBox => self.vm_state.pt_tet_box_id,
                VmMode::Pyramid => self.vm_state.pt_pyramid_id,
                VmMode::Wedge => self.vm_state.pt_wedge_id,
            };
            if let Some(id) = id {
                let data = pt_data_for_mode(mode);
                let _ = renderer.resources_mut().replace_projected_tet_mesh(
                    device,
                    id,
                    &data,
                    attr,
                    colourmap_id,
                );
            }
        }
    }

    /// Returns a `TransparentVolumeMeshItem` for the projected-tet pass when
    /// transparent mode is active, or `None` otherwise.
    pub(crate) fn vm_transparent_item(&self) -> Option<TransparentVolumeMeshItem> {
        if !self.vm_state.transparent || !self.vm_state.built {
            return None;
        }
        let id = match self.vm_state.mode {
            VmMode::Hex => self.vm_state.pt_hex_id,
            VmMode::Tet => self.vm_state.pt_tet_id,
            VmMode::TetSmall => self.vm_state.pt_tet_small_id,
            VmMode::TetBox => self.vm_state.pt_tet_box_id,
            VmMode::Pyramid => self.vm_state.pt_pyramid_id,
            VmMode::Wedge => self.vm_state.pt_wedge_id,
        }?;
        let mut item = TransparentVolumeMeshItem::new(id);
        item.density = self.vm_state.density;
        Some(item)
    }

    pub(crate) fn vm_clip_objects(&self) -> Vec<ClipObject> {
        // No GPU clip plane is emitted for geometry — the CPU extraction already
        // handles it, and a co-planar GPU clip causes per-fragment floating-point
        // noise on section faces.  Only the visual edge indicator is produced
        // here (no fill quad, no cap-fill, no GPU clip plane written to the
        // clip-planes uniform).
        if !self.vm_state.clip_on {
            return vec![];
        }

        let normal = self.vm_clip_normal();
        let mut clip = ClipObject::plane(normal, self.vm_state.clip_offset);
        clip.shape = ClipShape::Plane {
            normal,
            distance: self.vm_state.clip_offset,
            cap_colour: None,
            display_center: None,
        };
        clip.colour = None;
        clip.edge_colour = Some([0.75, 0.85, 1.0, 1.0]);
        // In transparent mode, no opaque CPU-clipped mesh is drawn, so the GPU
        // clip plane must be active to cull the projected-tet fragments.
        // In opaque mode, CPU extraction already handles clipping and enabling
        // the GPU clip on top causes floating-point noise on the section faces.
        clip.clip_geometry = self.vm_state.transparent;
        clip.extent = 3.5;
        vec![clip]
    }

    /// Lighting for the volume mesh showcase.
    pub(crate) fn vm_lighting() -> LightingSettings {
        LightingSettings {
            hemisphere_intensity: 0.3,
            sky_colour: [1.0, 1.0, 1.0],
            ground_colour: [0.3, 0.3, 0.4],
            ..LightingSettings::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_volume_mesh(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Cell type:");
    ui.horizontal_wrapped(|ui| {
        ui.radio_value(&mut app.vm_state.mode, VmMode::Hex, "Hex sphere (27)");
        ui.radio_value(
            &mut app.vm_state.mode,
            VmMode::Pyramid,
            "Pyramid sphere (162)",
        );
        ui.radio_value(&mut app.vm_state.mode, VmMode::Wedge, "Wedge sphere (54)");
        ui.radio_value(&mut app.vm_state.mode, VmMode::Tet, "Tet sphere 3³ (162)");
        ui.radio_value(&mut app.vm_state.mode, VmMode::TetBox, "Tet box 3³ (162)");
        ui.radio_value(&mut app.vm_state.mode, VmMode::TetSmall, "Tet cube 1³ (6)");
    });

    ui.separator();
    ui.label("Field:");
    for (field, label) in [
        (VmField::Latitude, "Latitude (scalar)"),
        (VmField::Longitude, "Longitude (scalar)"),
        (VmField::Radial, "Radial distance (scalar)"),
        (VmField::DirectColour, "Direct cell colours (RGBA)"),
    ] {
        ui.radio_value(&mut app.vm_state.field, field, label);
    }

    if !matches!(app.vm_state.field, VmField::DirectColour) {
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
                ui.radio_value(&mut app.vm_state.colourmap, cm, format!("{cm:?}"));
            }
        });
    }

    ui.separator();
    ui.checkbox(
        &mut app.vm_state.transparent,
        "Transparent (projected tetrahedra)",
    );
    if app.vm_state.transparent {
        ui.add(
            egui::Slider::new(&mut app.vm_state.density, 0.0..=1.0)
                .text("Density")
                .step_by(0.01),
        );
    }

    ui.separator();
    ui.checkbox(&mut app.vm_state.wireframe, "Wireframe");

    ui.separator();
    ui.checkbox(&mut app.vm_state.clip_on, "Clip plane");
    if app.vm_state.clip_on {
        ui.add(
            egui::Slider::new(&mut app.vm_state.clip_offset, -1.75..=1.75)
                .text("Offset")
                .step_by(0.01),
        );
        ui.add(
            egui::Slider::new(&mut app.vm_state.clip_elevation, -89.0..=89.0)
                .text("Elevation (°)")
                .step_by(1.0),
        );
        ui.add(
            egui::Slider::new(&mut app.vm_state.clip_azimuth, -180.0..=180.0)
                .text("Azimuth (°)")
                .step_by(1.0),
        );
        ui.label("Keeps the positive-normal side of the plane.");
    }

    ui.separator();
    let (n_cells, note) = match app.vm_state.mode {
        VmMode::Hex => (27, "3³ hexes, cube-to-sphere warp"),
        VmMode::Tet => (162, "3³×6 tets on sphere"),
        VmMode::TetBox => (162, "3³×6 tets, flat box"),
        VmMode::TetSmall => (6, "1³×6 tets, unit cube"),
        VmMode::Pyramid => (162, "3³×6 pyramids on sphere"),
        VmMode::Wedge => (54, "3³×2 wedges on sphere"),
    };
    ui.label(format!("{n_cells} cells · {note}"));
    ui.label("Interior faces are automatically discarded.");
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn vm_collect_scene_items(
    app: &mut App,
    frame: &eframe::Frame,
) -> (Vec<SceneRenderItem>, LightingSettings, u64, u64) {
    // Per-frame CPU clip section: regenerate the clipped mesh slot
    // so section faces reflect the current plane position.
    if app.vm_state.clip_on && app.vm_state.built {
        if let Some(rs) = frame.wgpu_render_state() {
            let mut guard = rs.renderer.write();
            if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                let data = app.vm_active_data();
                let clip_planes = [app.vm_clip_plane()];
                match app.vm_state.clipped_item.as_ref() {
                    None => {
                        if let Ok((id, f2c)) =
                            renderer.resources_mut().upload_clipped_volume_mesh_data(
                                &rs.device,
                                &data,
                                &clip_planes,
                            )
                        {
                            let mut item = viewport_lib::VolumeMeshItem::new(id, f2c);
                            item.material.backface_policy =
                                viewport_lib::BackfacePolicy::Identical;
                            app.vm_state.clipped_item = Some(item);
                        }
                    }
                    Some(existing) => {
                        if let Ok(f2c) =
                            renderer.resources_mut().replace_clipped_volume_mesh_data(
                                &rs.device,
                                &rs.queue,
                                existing.mesh_id,
                                &data,
                                &clip_planes,
                            )
                        {
                            if let Some(item) = app.vm_state.clipped_item.as_mut() {
                                item.face_to_cell = f2c;
                            }
                        }
                    }
                }
            }
        }
    }
    // Rebuild PT meshes when the scalar field or colourmap changes.
    let pt_needs_rebuild = app.vm_state.built
        && (app.vm_state.field != app.vm_state.pt_field
            || app.vm_state.colourmap != app.vm_state.pt_colourmap);
    if pt_needs_rebuild {
        if let Some(rs) = frame.wgpu_render_state() {
            let mut guard = rs.renderer.write();
            if let Some(renderer) = guard.callback_resources.get_mut::<ViewportRenderer>() {
                app.rebuild_pt_meshes(
                    renderer,
                    &rs.device,
                    app.vm_state.field,
                    app.vm_state.colourmap,
                );
                app.vm_state.pt_field = app.vm_state.field;
                app.vm_state.pt_colourmap = app.vm_state.colourmap;
            }
        }
    }

    let items = app.vm_scene_items();
    (items, App::vm_lighting(), 0, 0)
}

pub(crate) fn submit_vm_items(app: &mut App, fd: &mut FrameData) {
    if let Some(item) = app.vm_transparent_item() {
        fd.scene.transparent_volume_meshes.push(item);
        fd.effects.post_process.enabled = true;
    }
}

pub(crate) fn vm_configure_frame(app: &App, fd: &mut FrameData) {
    fd.viewport.wireframe_mode = app.vm_state.wireframe;
    fd.effects.cap_fill_enabled = false;
    fd.effects.clip_objects.extend(app.vm_clip_objects());
}
