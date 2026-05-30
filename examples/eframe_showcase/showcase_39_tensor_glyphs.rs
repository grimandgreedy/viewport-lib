//! Showcase 39: Tensor Glyphs -- Loaded Beam
//!
//! A simply-supported rectangular beam under a central point load.
//!
//! Above: the beam solid, rendered as a hex volume mesh coloured by sigma_xx
//! (bending stress). This is what a scalar field shows you: the magnitude and
//! sign of one stress component, but not the principal directions.
//!
//! Below: principal stress tensor glyphs at each cell centroid. The same load
//! produces three qualitatively different glyph shapes depending on location:
//!
//!   - Top fiber (compression zone, +Z): disk-like glyph, compressed along beam axis.
//!   - Bottom fiber (tension zone, -Z): cigar-like glyph, elongated along beam axis.
//!   - Neutral axis near the supports: glyph rotated 45 deg -- pure shear,
//!     where sigma_xx = 0 and tau_xy is maximum.
//!   - Mixed regions: intermediate rotation between the above extremes.
//!
//! This rotation of the principal axes is invisible to a scalar stress field
//! but immediately readable from the tensor glyphs.

use crate::App;
use eframe::egui;
use std::collections::HashMap;

use viewport_lib::{
    AttributeKind, AttributeRef, BackfacePolicy, BuiltinColourmap, CellSelectionInfo, ColourmapId,
    FrameData, MeshId, PickId, PickMask, SceneRenderItem, SubObjectRef, SubSelection,
    SubSelectionRef, TensorGlyphItem, ViewportRenderer, VolumeMeshData, VolumeMeshItem,
};

const PICK_BEAM_MESH: u64 = 3901;
const PICK_TENSOR_GLYPHS: u64 = 3902;

// ---------------------------------------------------------------------------
// Beam geometry constants
// ---------------------------------------------------------------------------

/// Cells along the beam length (X direction).
const NX: usize = 32;
/// Cells through the beam depth (Z direction, up-down).
const NY: usize = 14;
/// Cells through the beam width (Y direction).
const NZ: usize = 4;

const BEAM_HALF_L: f32 = 4.0; // X: -4 to +4
const BEAM_HALF_H: f32 = 1.0; // Z depth: 2.0 total
const BEAM_HALF_W: f32 = 0.5; // Y width: 1.0 total

/// Z center for the upper (volume mesh) region.
const Z_TOP: f32 = 3.5;
/// Z center for the lower (tensor glyphs) region.
const Z_BOT: f32 = -3.5;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const GLYPH_COLOURMAPS: &[(BuiltinColourmap, &str)] = &[
    (BuiltinColourmap::RdBu, "RdBu"),
    (BuiltinColourmap::Coolwarm, "Coolwarm"),
    (BuiltinColourmap::Viridis, "Viridis"),
    (BuiltinColourmap::Plasma, "Plasma"),
    (BuiltinColourmap::Turbo, "Turbo"),
    (BuiltinColourmap::Inferno, "Inferno"),
    (BuiltinColourmap::Greyscale, "Greyscale"),
];

/// What the user last clicked on.
#[derive(Debug, Clone)]
pub(crate) enum TgSelection {
    /// A tensor glyph instance. Stores the glyph index and its stress values.
    Glyph { index: usize, sigma_xx: f32, tau_xy: f32 },
    /// A beam mesh cell. Stores the cell index and its stress values.
    Cell { index: usize, sigma_xx: f32, tau_xy: f32 },
}

pub(crate) struct TensorGlyphState {
    pub built: bool,
    pub mesh_id: Option<MeshId>,
    pub face_to_cell: Vec<u32>,
    /// Raw beam vertex positions kept for cell selection highlights.
    pub beam_positions: Vec<[f32; 3]>,
    /// Raw beam cell connectivity kept for cell selection highlights.
    pub beam_cells: Vec<[u32; 8]>,
    pub scale: f32,
    pub density: f32,
    pub colourmap: BuiltinColourmap,
    pub selection: Option<TgSelection>,
    pub sub_selection: SubSelection,
}

impl Default for TensorGlyphState {
    fn default() -> Self {
        Self {
            built: false,
            mesh_id: None,
            face_to_cell: Vec::new(),
            beam_positions: Vec::new(),
            beam_cells: Vec::new(),
            scale: 0.45,
            density: 1.0,
            colourmap: BuiltinColourmap::Inferno,
            selection: None,
            sub_selection: SubSelection::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Beam stress (Euler-Bernoulli, plane stress, normalized)
// ---------------------------------------------------------------------------

/// Analytical stress at a point in the beam cross-section.
///
/// `cx`     : X position in world space, -BEAM_HALF_L to +BEAM_HALF_L.
/// `cz_rel` : Z position relative to the beam neutral axis, -BEAM_HALF_H to +BEAM_HALF_H.
///            Positive = top fiber (compression side), negative = bottom fiber (tension side).
///
/// Returns `(sigma_xx, tau_xy)` both normalized so the extremes are near +-1.
fn beam_stress(cx: f32, cz_rel: f32) -> (f32, f32) {
    let x_norm = cx / BEAM_HALF_L; // -1 at left support, +1 at right support
    let y_norm = cz_rel / BEAM_HALF_H; // -1 at bottom fiber, +1 at top fiber

    // Triangular bending moment: peaks at center, zero at supports.
    let moment = 1.0 - x_norm.abs();

    // Constant shear on each half-span (step function, zero exactly at center).
    let shear = if cx < 0.0 { 0.5 } else { -0.5 };

    // Bending stress: linear through depth.
    let sigma_xx = -moment * y_norm;

    // Shear stress: parabolic through depth, zero at top and bottom fibers.
    let tau_xy = shear * (1.0 - y_norm * y_norm) * 0.5;

    (sigma_xx, tau_xy)
}

fn von_mises(sigma_xx: f32, tau_xy: f32) -> f32 {
    (sigma_xx * sigma_xx + 3.0 * tau_xy * tau_xy).sqrt()
}

// ---------------------------------------------------------------------------
// Eigendecomposition of the plane-stress tensor
//
//  sigma = [ sigma_xx   tau_xy    0  ]
//          [ tau_xy     0         0  ]
//          [ 0          0        eps ]
//
// The z-component is decoupled: lambda_3 = eps (small out-of-plane).
// ---------------------------------------------------------------------------

fn stress_eigen(sigma_xx: f32, tau_xy: f32) -> ([f32; 3], [[f32; 3]; 3]) {
    let mean = sigma_xx * 0.5;
    let half = sigma_xx * 0.5;
    let disc = (half * half + tau_xy * tau_xy).sqrt();

    // Floor prevents zero eigenvalues (which occur at extreme fibers under pure
    // uniaxial stress where sigma_yy = 0). Without the floor those glyphs collapse
    // to flat lines. The floor is small enough not to distort the stress picture.
    const FLOOR: f32 = 0.12;
    let lam1_raw = mean + disc;
    let lam2_raw = mean - disc;
    let lam1 = if lam1_raw >= 0.0 {
        lam1_raw.max(FLOOR)
    } else {
        lam1_raw.min(-FLOOR)
    };
    let lam2 = if lam2_raw >= 0.0 {
        lam2_raw.max(FLOOR)
    } else {
        lam2_raw.min(-FLOOR)
    };
    let lam3 = 0.18f32; // out-of-plane thickness -- enough to look 3-D

    // Rotation angle of principal axes in the xy-plane.
    let theta = 0.5 * tau_xy.atan2(half);
    let (s, c) = theta.sin_cos();

    let e1 = [c, 0.0, s]; // primary eigenvector, rotates in XZ (bending plane)
    let e2 = [-s, 0.0, c]; // secondary eigenvector, in XZ
    let e3 = [0.0, 1.0, 0.0]; // out-of-plane (Y, beam width axis)

    ([lam1, lam2, lam3], [e1, e2, e3])
}

// ---------------------------------------------------------------------------
// Vertex indexing for the hex mesh
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Glyph grid (independent of the hex mesh resolution)
// ---------------------------------------------------------------------------

/// Maximum glyph grid along each axis. At density=1.0 all GNX*GNY*GNZ glyphs
/// are placed; the density slider subsamples this via a stride.
const GNX: usize = 24; // along X (beam length)
const GNY: usize = 8;  // along Z (beam depth, up-down)
const GNZ: usize = 4;  // along Y (beam width)

// ---------------------------------------------------------------------------
// Vertex indexing for the hex mesh
// ---------------------------------------------------------------------------

/// Vertex count per axis.
const VX: usize = NX + 1;
const VZ: usize = NZ + 1;

/// Flat vertex index: iy is outermost (Z depth), iz next (Y width), ix innermost (X length).
fn vid(ix: usize, iy: usize, iz: usize) -> u32 {
    (iy * VZ * VX + iz * VX + ix) as u32
}

// ---------------------------------------------------------------------------
// Build the beam VolumeMeshData
// ---------------------------------------------------------------------------

fn build_beam_mesh(z_center: f32) -> VolumeMeshData {
    // Vertex positions. iy indexes Z (depth, up-down), iz indexes Y (width).
    let mut positions = Vec::with_capacity((NX + 1) * (NY + 1) * (NZ + 1));
    for iy in 0..=NY {
        for iz in 0..=NZ {
            for ix in 0..=NX {
                let x = -BEAM_HALF_L + 2.0 * BEAM_HALF_L * ix as f32 / NX as f32;
                let y = -BEAM_HALF_W + 2.0 * BEAM_HALF_W * iz as f32 / NZ as f32;
                let z = z_center + (-BEAM_HALF_H + 2.0 * BEAM_HALF_H * iy as f32 / NY as f32);
                positions.push([x, y, z]);
            }
        }
    }

    // Hex cells and per-cell scalars.
    let n_cells = NX * NY * NZ;
    let mut cells = Vec::with_capacity(n_cells);
    let mut vm_scalars = Vec::with_capacity(n_cells);
    let mut sigma_scalars = Vec::with_capacity(n_cells);

    for iy in 0..NY {
        for iz in 0..NZ {
            for ix in 0..NX {
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

                let cx = -BEAM_HALF_L + 2.0 * BEAM_HALF_L * (ix as f32 + 0.5) / NX as f32;
                let cy_rel = -BEAM_HALF_H + 2.0 * BEAM_HALF_H * (iy as f32 + 0.5) / NY as f32;

                let (sigma_xx, tau_xy) = beam_stress(cx, cy_rel);
                vm_scalars.push(von_mises(sigma_xx, tau_xy));
                sigma_scalars.push(sigma_xx);
            }
        }
    }

    let mut data = VolumeMeshData::default();
    data.positions = positions;
    data.cells = cells;
    data.cell_scalars
        .insert("von_mises".to_string(), vm_scalars);
    data.cell_scalars
        .insert("sigma_xx".to_string(), sigma_scalars);
    data
}

// ---------------------------------------------------------------------------
// Build (one-time GPU upload)
// ---------------------------------------------------------------------------

pub(crate) fn build_tensor_glyph_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    let data = build_beam_mesh(Z_TOP);
    app.tg_state.beam_positions = data.positions.clone();
    app.tg_state.beam_cells = data.cells.clone();
    if let Ok((id, f2c)) = renderer
        .resources_mut()
        .upload_volume_mesh_data(&app.device, &data)
    {
        app.tg_state.mesh_id = Some(id);
        app.tg_state.face_to_cell = f2c;
    }
    app.tg_state.built = true;
}

// ---------------------------------------------------------------------------
// Submit (called every frame)
// ---------------------------------------------------------------------------

pub(crate) fn submit_tensor_glyphs(app: &App, fd: &mut FrameData) {
    let state = &app.tg_state;

    // ------------------------------------------------------------------
    // Below the beam: tensor glyphs at cell centroids.
    // iy indexes Z depth (up-down), iz indexes Y width.
    // ------------------------------------------------------------------
    {
        let n_max = GNX * GNY * GNZ;
        let mut positions = Vec::with_capacity(n_max);
        let mut eigenvalues = Vec::with_capacity(n_max);
        let mut eigenvectors = Vec::with_capacity(n_max);
        let mut colour_attr = Vec::with_capacity(n_max);

        // Stride-based subsampling over the fine glyph grid.
        // density=1.0 -> all GNX*GNY*GNZ glyphs, 0.5 -> every other, etc.
        let stride = ((1.0 / state.density).ceil() as usize).max(1);

        for iy in 0..GNY {
            for iz in 0..GNZ {
                for ix in 0..GNX {
                    let linear = ix + GNX * (iz + GNZ * iy);
                    if linear % stride != 0 {
                        continue;
                    }

                    let cx = -BEAM_HALF_L + 2.0 * BEAM_HALF_L * (ix as f32 + 0.5) / GNX as f32;
                    let cz_rel = -BEAM_HALF_H + 2.0 * BEAM_HALF_H * (iy as f32 + 0.5) / GNY as f32;
                    let cy = -BEAM_HALF_W + 2.0 * BEAM_HALF_W * (iz as f32 + 0.5) / GNZ as f32;

                    positions.push([cx, cy, Z_BOT + cz_rel]);

                    let (sigma_xx, tau_xy) = beam_stress(cx, cz_rel);
                    let (evals, evecs) = stress_eigen(sigma_xx, tau_xy);
                    eigenvalues.push(evals);
                    eigenvectors.push(evecs);
                    // Colour by sigma_xx: tension = positive (warm), compression = negative (cool).
                    colour_attr.push(sigma_xx);
                }
            }
        }

        let mut item = TensorGlyphItem::default();
        item.positions = positions;
        item.eigenvalues = eigenvalues;
        item.eigenvectors = eigenvectors;
        item.scale = state.scale;
        item.colour_attribute = Some(colour_attr);
        item.scalar_range = Some((-1.2, 1.2));
        item.colourmap_id = Some(ColourmapId(state.colourmap as usize));
        item.settings.pick_id = PickId(PICK_TENSOR_GLYPHS);
        fd.scene.tensor_glyphs.push(item);
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

/// Submit the beam as a VolumeMeshItem (required for Cell sub-object picks).
pub(crate) fn submit_beam_item(app: &App, fd: &mut FrameData) {
    let Some(mesh_id) = app.tg_state.mesh_id else { return };
    if app.tg_state.face_to_cell.is_empty() { return; }
    let mut item = VolumeMeshItem::new(mesh_id, app.tg_state.face_to_cell.clone());
    item.active_attribute = Some(AttributeRef {
        name: "von_mises".to_string(),
        kind: AttributeKind::Face,
    });
    item.colourmap_id = Some(ColourmapId(0)); // viridis
    item.material.backface_policy = BackfacePolicy::Identical;
    item.settings.pick_id = PickId(PICK_BEAM_MESH);
    fd.scene.volume_mesh_items.push(item);
}

/// Surface render item for the beam (visual display).
pub(crate) fn beam_scene_items(app: &App) -> Vec<SceneRenderItem> {
    let Some(mesh_id) = app.tg_state.mesh_id else { return vec![] };
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.active_attribute = Some(AttributeRef {
        name: "sigma_xx".to_string(),
        kind: AttributeKind::Face,
    });
    item.scalar_range = Some((-1.2, 1.2));
    item.colourmap_id = Some(ColourmapId(app.tg_state.colourmap as usize));
    item.material.backface_policy = BackfacePolicy::Identical;
    // Must match the VolumeMeshItem pick_id so the picking loop can convert face hits to cells.
    item.settings.pick_id = PickId(PICK_BEAM_MESH);
    vec![item]
}

/// Handle a click in the tensor glyph showcase viewport.
/// Resolves the hit via the GPU pick buffer and stores the result in `tg_state.selection`.
pub(crate) fn tg_handle_click(
    app: &mut App,
    pos: glam::Vec2,
    vp_size: glam::Vec2,
    view_proj: glam::Mat4,
    renderer: &ViewportRenderer,
) {
    let mask = PickMask::POINT_LIKE;
    let Some(hit) = renderer.pick(pos, vp_size, view_proj, mask) else {
        app.tg_state.selection = None;
        app.tg_state.sub_selection.clear();
        return;
    };

    if hit.id == PICK_TENSOR_GLYPHS {
        if let Some(SubObjectRef::Instance(idx)) = hit.sub_object {
            let idx = idx as usize;
            let stride = ((1.0 / app.tg_state.density).ceil() as usize).max(1);
            let mut glyph_idx = 0usize;
            'outer: for iy in 0..GNY {
                for iz in 0..GNZ {
                    for ix in 0..GNX {
                        let linear = ix + GNX * (iz + GNZ * iy);
                        if linear % stride != 0 {
                            continue;
                        }
                        if glyph_idx == idx {
                            let cx = -BEAM_HALF_L + 2.0 * BEAM_HALF_L * (ix as f32 + 0.5) / GNX as f32;
                            let cz_rel = -BEAM_HALF_H + 2.0 * BEAM_HALF_H * (iy as f32 + 0.5) / GNY as f32;
                            let (sigma_xx, tau_xy) = beam_stress(cx, cz_rel);
                            app.tg_state.selection = Some(TgSelection::Glyph { index: idx, sigma_xx, tau_xy });
                            app.tg_state.sub_selection.select_one(PICK_TENSOR_GLYPHS, SubObjectRef::Instance(idx as u32));
                            break 'outer;
                        }
                        glyph_idx += 1;
                    }
                }
            }
        }
    } else if hit.id == PICK_BEAM_MESH {
        if let Some(SubObjectRef::Cell(cell_idx)) = hit.sub_object {
            let idx = cell_idx as usize;
            // build_beam_mesh iterates iy (outermost) -> iz -> ix (innermost).
            // Flat index = iy * NZ * NX + iz * NX + ix.
            let ix = idx % NX;
            let iz = (idx / NX) % NZ;
            let iy = idx / (NX * NZ);
            let cx = -BEAM_HALF_L + 2.0 * BEAM_HALF_L * (ix as f32 + 0.5) / NX as f32;
            let cz_rel = -BEAM_HALF_H + 2.0 * BEAM_HALF_H * (iy as f32 + 0.5) / NY as f32;
            let _ = iz;
            let (sigma_xx, tau_xy) = beam_stress(cx, cz_rel);
            app.tg_state.selection = Some(TgSelection::Cell { index: idx, sigma_xx, tau_xy });
            app.tg_state.sub_selection.select_one(PICK_BEAM_MESH, SubObjectRef::Cell(cell_idx));
        }
    } else {
        app.tg_state.selection = None;
        app.tg_state.sub_selection.clear();
    }
}

/// Wire the current selection into `fd.interaction` so the renderer draws the
/// built-in outline highlight over the selected glyph instance or beam cell.
pub(crate) fn submit_tg_sub_selection(app: &App, fd: &mut FrameData) {
    if app.tg_state.sub_selection.is_empty() {
        return;
    }
    let mut cell_lookup: HashMap<u64, CellSelectionInfo> = HashMap::new();
    if !app.tg_state.beam_positions.is_empty() {
        cell_lookup.insert(
            PICK_BEAM_MESH,
            CellSelectionInfo {
                positions: app.tg_state.beam_positions.clone(),
                cells: app.tg_state.beam_cells.clone(),
            },
        );
    }
    fd.interaction.sub_selection = Some(
        SubSelectionRef::new(
            &app.tg_state.sub_selection,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
        .with_cells(cell_lookup),
    );
    fd.interaction.outline_selected = true;
}

pub(crate) fn controls_tensor_glyphs(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Simply-supported beam under a central point load.");
    ui.label("Click a glyph or beam cell to inspect its stress values.");
    ui.separator();

    match &app.tg_state.selection {
        None => {
            ui.label("Nothing selected.");
        }
        Some(TgSelection::Glyph { index, sigma_xx, tau_xy }) => {
            ui.strong("Tensor glyph");
            ui.label(format!("Instance: {index}"));
            ui.label(format!("sigma_xx: {sigma_xx:.3}"));
            ui.label(format!("tau_xy:   {tau_xy:.3}"));
            ui.label(format!("von Mises: {:.3}", von_mises(*sigma_xx, *tau_xy)));
        }
        Some(TgSelection::Cell { index, sigma_xx, tau_xy }) => {
            ui.strong("Beam mesh cell");
            ui.label(format!("Cell: {index}"));
            ui.label(format!("sigma_xx: {sigma_xx:.3}"));
            ui.label(format!("tau_xy:   {tau_xy:.3}"));
            ui.label(format!("von Mises: {:.3}", von_mises(*sigma_xx, *tau_xy)));
        }
    }
    ui.separator();

    ui.label("Above: beam volume mesh, coloured by sigma_xx (bending stress).");
    ui.label("  Shows magnitude and sign, but not the principal directions.");
    ui.separator();

    ui.label("Below: principal stress tensor glyphs.");
    ui.label("  Top + bottom fibers: elongated along the beam axis.");
    ui.label("  Colour tells you the sign: blue = compression, red = tension.");
    ui.label("  Neutral axis near the supports: glyphs rotated ~45 deg (pure shear).");
    ui.label("  Mixed zones: gradual rotation between 0 and 45 deg.");
    ui.separator();

    ui.label("Glyph scale:");
    ui.add(egui::Slider::new(&mut app.tg_state.scale, 0.1..=1.0));

    ui.separator();
    ui.label("Glyph density:");
    ui.add(egui::Slider::new(&mut app.tg_state.density, 0.1..=1.0));

    ui.separator();
    ui.label("Glyph colourmap:");
    let selected_name = GLYPH_COLOURMAPS
        .iter()
        .find(|(c, _)| *c == app.tg_state.colourmap)
        .map(|(_, n)| *n)
        .unwrap_or("Unknown");
    egui::ComboBox::from_id_salt("tg_colourmap")
        .selected_text(selected_name)
        .show_ui(ui, |ui| {
            for (cmap, name) in GLYPH_COLOURMAPS {
                ui.selectable_value(&mut app.tg_state.colourmap, *cmap, *name);
            }
        });
}
