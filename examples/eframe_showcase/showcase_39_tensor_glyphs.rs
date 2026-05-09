//! Showcase 39: Tensor Glyphs -- Loaded Beam
//!
//! A simply-supported rectangular beam under a central point load.
//!
//! Top: the beam solid, rendered as a transparent hex volume mesh colored by
//! von Mises stress. This is what a scalar field shows you: how intense the
//! stress is, but not its direction.
//!
//! Bottom: principal stress tensor glyphs at each cell centroid. The same
//! load produces three qualitatively different glyph shapes depending on
//! location:
//!
//!   - Top fiber (compression zone): disk-like glyph, compressed along beam axis.
//!   - Bottom fiber (tension zone): cigar-like glyph, elongated along beam axis.
//!   - Neutral axis near the supports: glyph rotated 45 deg -- pure shear,
//!     where sigma_xx = 0 and tau_xy is maximum.
//!   - Mixed regions: intermediate rotation between the above extremes.
//!
//! This rotation of the principal axes is invisible to a scalar stress field
//! but immediately readable from the tensor glyphs.

use eframe::egui;
use viewport_lib::{
    AttributeKind, AttributeRef, BackfacePolicy, BuiltinColormap, ColormapId, FrameData,
    SceneRenderItem, TensorGlyphItem, ViewportRenderer, VolumeMeshData,
};
use crate::App;

// ---------------------------------------------------------------------------
// Beam geometry constants
// ---------------------------------------------------------------------------

/// Cells along the beam length (X direction).
const NX: usize = 32;
/// Cells through the beam depth (Y direction).
const NY: usize = 14;
/// Cells through the beam width (Z direction).
const NZ: usize = 4;

const BEAM_HALF_L: f32 = 4.0;  // X: -4 to +4
const BEAM_HALF_H: f32 = 1.0;  // Y depth: 2.0 total
const BEAM_HALF_W: f32 = 0.5;  // Z width: 1.0 total

/// Y center for the top (volume mesh) region.
const Y_TOP: f32 = 3.5;
/// Y center for the bottom (tensor glyphs) region.
const Y_BOT: f32 = -3.5;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const GLYPH_COLORMAPS: &[(BuiltinColormap, &str)] = &[
    (BuiltinColormap::RdBu,      "RdBu"),
    (BuiltinColormap::Coolwarm,  "Coolwarm"),
    (BuiltinColormap::Viridis,   "Viridis"),
    (BuiltinColormap::Plasma,    "Plasma"),
    (BuiltinColormap::Turbo,     "Turbo"),
    (BuiltinColormap::Inferno,   "Inferno"),
    (BuiltinColormap::Greyscale, "Greyscale"),
];

#[derive(Debug, Clone)]
pub(crate) struct TensorGlyphState {
    pub scale:    f32,
    pub density:  f32,
    pub colormap: BuiltinColormap,
}

impl Default for TensorGlyphState {
    fn default() -> Self {
        Self {
            scale:    0.45,
            density:  1.0,
            colormap: BuiltinColormap::RdBu,
        }
    }
}

// ---------------------------------------------------------------------------
// Beam stress (Euler-Bernoulli, plane stress, normalized)
// ---------------------------------------------------------------------------

/// Analytical stress at a point in the beam cross-section.
///
/// `cx` : X position in world space, -BEAM_HALF_L to +BEAM_HALF_L.
/// `cy_rel` : Y position relative to the beam neutral axis, -BEAM_HALF_H to +BEAM_HALF_H.
///            Positive = top (compression side), negative = bottom (tension side).
///
/// Returns `(sigma_xx, tau_xy)` both normalized so the extremes are near +-1.
fn beam_stress(cx: f32, cy_rel: f32) -> (f32, f32) {
    let x_norm = cx / BEAM_HALF_L;             // -1 at left support, +1 at right support
    let y_norm = cy_rel / BEAM_HALF_H;         // -1 at bottom, +1 at top

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
    let mean  = sigma_xx * 0.5;
    let half  = sigma_xx * 0.5;
    let disc  = (half * half + tau_xy * tau_xy).sqrt();

    // Floor prevents zero eigenvalues (which occur at extreme fibers under pure
    // uniaxial stress where sigma_yy = 0). Without the floor those glyphs collapse
    // to flat lines. The floor is small enough not to distort the stress picture.
    const FLOOR: f32 = 0.12;
    let lam1_raw = mean + disc;
    let lam2_raw = mean - disc;
    let lam1 = if lam1_raw >= 0.0 { lam1_raw.max(FLOOR) } else { lam1_raw.min(-FLOOR) };
    let lam2 = if lam2_raw >= 0.0 { lam2_raw.max(FLOOR) } else { lam2_raw.min(-FLOOR) };
    let lam3 = 0.18f32;       // out-of-plane thickness -- enough to look 3-D

    // Rotation angle of principal axes in the xy-plane.
    let theta = 0.5 * tau_xy.atan2(half);
    let (s, c) = theta.sin_cos();

    let e1 = [c,   s,   0.0]; // primary in-plane eigenvector
    let e2 = [-s,  c,   0.0]; // secondary in-plane eigenvector
    let e3 = [0.0, 0.0, 1.0]; // out-of-plane

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
const GNX: usize = 24;
const GNY: usize = 8;
const GNZ: usize = 4;

// ---------------------------------------------------------------------------
// Vertex indexing for the hex mesh
// ---------------------------------------------------------------------------

/// Vertex count per axis.
const VX: usize = NX + 1;
const VZ: usize = NZ + 1;

/// Flat vertex index: iy is outermost (depth), iz next (width), ix innermost (length).
fn vid(ix: usize, iy: usize, iz: usize) -> u32 {
    (iy * VZ * VX + iz * VX + ix) as u32
}

// ---------------------------------------------------------------------------
// Build the beam VolumeMeshData
// ---------------------------------------------------------------------------

fn build_beam_mesh(y_center: f32) -> VolumeMeshData {
    // Vertex positions.
    let mut positions = Vec::with_capacity((NX + 1) * (NY + 1) * (NZ + 1));
    for iy in 0..=NY {
        for iz in 0..=NZ {
            for ix in 0..=NX {
                let x = -BEAM_HALF_L + 2.0 * BEAM_HALF_L * ix as f32 / NX as f32;
                let y = y_center
                    + (-BEAM_HALF_H + 2.0 * BEAM_HALF_H * iy as f32 / NY as f32);
                let z = -BEAM_HALF_W + 2.0 * BEAM_HALF_W * iz as f32 / NZ as f32;
                positions.push([x, y, z]);
            }
        }
    }

    // Hex cells and per-cell scalars.
    let n_cells = NX * NY * NZ;
    let mut cells          = Vec::with_capacity(n_cells);
    let mut vm_scalars     = Vec::with_capacity(n_cells);
    let mut sigma_scalars  = Vec::with_capacity(n_cells);

    for iy in 0..NY {
        for iz in 0..NZ {
            for ix in 0..NX {
                cells.push([
                    vid(ix,   iy,   iz),
                    vid(ix+1, iy,   iz),
                    vid(ix+1, iy,   iz+1),
                    vid(ix,   iy,   iz+1),
                    vid(ix,   iy+1, iz),
                    vid(ix+1, iy+1, iz),
                    vid(ix+1, iy+1, iz+1),
                    vid(ix,   iy+1, iz+1),
                ]);

                let cx     = -BEAM_HALF_L + 2.0 * BEAM_HALF_L * (ix as f32 + 0.5) / NX as f32;
                let cy_rel = -BEAM_HALF_H + 2.0 * BEAM_HALF_H * (iy as f32 + 0.5) / NY as f32;

                let (sigma_xx, tau_xy) = beam_stress(cx, cy_rel);
                vm_scalars.push(von_mises(sigma_xx, tau_xy));
                sigma_scalars.push(sigma_xx);
            }
        }
    }

    let mut data = VolumeMeshData::default();
    data.positions = positions;
    data.cells     = cells;
    data.cell_scalars.insert("von_mises".to_string(), vm_scalars);
    data.cell_scalars.insert("sigma_xx".to_string(),  sigma_scalars);
    data
}

// ---------------------------------------------------------------------------
// Build (one-time GPU upload)
// ---------------------------------------------------------------------------

pub(crate) fn build_tensor_glyph_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    let data = build_beam_mesh(Y_TOP);
    app.tg_mesh_id = renderer
        .resources_mut()
        .upload_volume_mesh_data(&app.device, &data)
        .ok();

    app.tg_built = true;
}

// ---------------------------------------------------------------------------
// Submit (called every frame)
// ---------------------------------------------------------------------------

pub(crate) fn submit_tensor_glyphs(app: &App, fd: &mut FrameData) {
    let state = &app.tg_state;

    // ------------------------------------------------------------------
    // Bottom: tensor glyphs at cell centroids.
    // ------------------------------------------------------------------
    {
        let n_max = GNX * GNY * GNZ;
        let mut positions    = Vec::with_capacity(n_max);
        let mut eigenvalues  = Vec::with_capacity(n_max);
        let mut eigenvectors = Vec::with_capacity(n_max);
        let mut color_attr   = Vec::with_capacity(n_max);

        // Stride-based subsampling over the fine glyph grid.
        // density=1.0 -> all GNX*GNY*GNZ glyphs, 0.5 -> every other, etc.
        let stride = ((1.0 / state.density).ceil() as usize).max(1);

        for iy in 0..GNY {
            for iz in 0..GNZ {
                for ix in 0..GNX {
                    let linear = ix + GNX * (iz + GNZ * iy);
                    if linear % stride != 0 { continue; }

                    let cx     = -BEAM_HALF_L + 2.0 * BEAM_HALF_L * (ix as f32 + 0.5) / GNX as f32;
                    let cy_rel = -BEAM_HALF_H + 2.0 * BEAM_HALF_H * (iy as f32 + 0.5) / GNY as f32;
                    let cz     = -BEAM_HALF_W + 2.0 * BEAM_HALF_W * (iz as f32 + 0.5) / GNZ as f32;

                    // Place glyph at the mirrored Y position below.
                    positions.push([cx, Y_BOT + cy_rel, cz]);

                    let (sigma_xx, tau_xy) = beam_stress(cx, cy_rel);
                    let (evals, evecs)     = stress_eigen(sigma_xx, tau_xy);
                    eigenvalues.push(evals);
                    eigenvectors.push(evecs);
                    // Color by sigma_xx: tension = positive (warm), compression = negative (cool).
                    color_attr.push(sigma_xx);
                }
            }
        }

        let mut item = TensorGlyphItem::default();
        item.positions       = positions;
        item.eigenvalues     = eigenvalues;
        item.eigenvectors    = eigenvectors;
        item.scale           = state.scale;
        item.color_attribute = Some(color_attr);
        item.scalar_range    = Some((-1.2, 1.2));
        item.colormap_id     = Some(ColormapId(state.colormap as usize));
        fd.scene.tensor_glyphs.push(item);
    }

}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

/// Build the surface render item for the beam (called from build_frame_data).
pub(crate) fn beam_scene_items(app: &App) -> Vec<SceneRenderItem> {
    let Some(mesh_id) = app.tg_mesh_id else { return vec![]; };
    let mut item = SceneRenderItem::default();
    item.mesh_id = mesh_id;
    item.active_attribute = Some(AttributeRef {
        name: "von_mises".to_string(),
        kind: AttributeKind::Face,
    });
    item.colormap_id = Some(ColormapId(0)); // viridis
    item.material.backface_policy = BackfacePolicy::Identical;
    vec![item]
}

pub(crate) fn controls_tensor_glyphs(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Simply-supported beam under a central point load.");
    ui.separator();

    ui.label("Top: beam volume mesh, colored by von Mises stress (scalar).");
    ui.label("  Shows intensity -- not direction.");
    ui.separator();

    ui.label("Bottom: principal stress tensor glyphs.");
    ui.label("  Top + bottom fibers: elongated along the beam axis.");
    ui.label("  Color tells you the sign: blue = compression, red = tension.");
    ui.label("  Neutral axis near the supports: glyphs rotated ~45 deg (pure shear).");
    ui.label("  Mixed zones: gradual rotation between 0 and 45 deg.");
    ui.separator();

    ui.label("Glyph scale:");
    ui.add(egui::Slider::new(&mut app.tg_state.scale, 0.1..=1.0));

    ui.separator();
    ui.label("Glyph density:");
    ui.add(egui::Slider::new(&mut app.tg_state.density, 0.1..=1.0));

    ui.separator();
    ui.label("Glyph colormap:");
    let selected_name = GLYPH_COLORMAPS
        .iter()
        .find(|(c, _)| *c == app.tg_state.colormap)
        .map(|(_, n)| *n)
        .unwrap_or("Unknown");
    egui::ComboBox::from_id_salt("tg_colormap")
        .selected_text(selected_name)
        .show_ui(ui, |ui| {
            for (cmap, name) in GLYPH_COLORMAPS {
                ui.selectable_value(&mut app.tg_state.colormap, *cmap, *name);
            }
        });
}
