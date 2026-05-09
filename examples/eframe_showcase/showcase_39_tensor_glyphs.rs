//! Showcase 39: Why Tensors? Reynolds Stress vs Mean Velocity
//!
//! A synthetic turbulent channel flow cross-section with two columns
//! side by side across the channel height.
//!
//! Left column -- mean velocity outer product v*v (rank-1 tensor)
//!   All glyphs elongated along X with varying magnitude.
//!   This is what a velocity vector field can encode: one direction,
//!   one magnitude. Every glyph looks structurally the same.
//!
//! Right column -- Reynolds stress tensor R_ij (full symmetric rank-2)
//!   Near-wall: highly anisotropic (R_xx >> R_yy), principal axes rotated
//!   ~25 deg from the flow direction due to the turbulent shear stress R_xy.
//!   At the center: R_xy = 0 by symmetry, so glyphs are axis-aligned.
//!
//! The contrast shows why a second-order tensor carries information that
//! no vector or scalar field can represent: it encodes the full covariance
//! structure of the turbulent velocity fluctuations.

use eframe::egui;
use viewport_lib::{ColormapId, FrameData, LabelItem, TensorGlyphItem};
use crate::App;

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

const N_Y: usize = 13;   // sample points across channel height
const X_VEL: f32 = -4.0; // x position of mean-velocity column
const X_RS: f32  =  4.0; // x position of Reynolds stress column
const Y_HALF: f32 = 5.0; // half-channel extent (world units)

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct TensorGlyphState {
    pub scale: f32,
    pub colormap_idx: usize,
}

impl Default for TensorGlyphState {
    fn default() -> Self {
        Self {
            scale: 0.6,
            colormap_idx: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic flow profiles
// ---------------------------------------------------------------------------

/// Wall-normal coordinates evenly spaced in [-1, 1], excluding the walls
/// so glyphs near the boundary remain visible.
fn y_normals() -> [f32; N_Y] {
    let mut ys = [0.0f32; N_Y];
    for i in 0..N_Y {
        ys[i] = -0.9 + 1.8 * i as f32 / (N_Y - 1) as f32;
    }
    ys
}

/// Turbulent mean velocity profile u(y), 1/7th power law.
/// Returns value in [0, 1] with 1 at center and 0 at walls.
fn mean_u(y_norm: f32) -> f32 {
    let d = 1.0 - y_norm.abs();
    d.powf(1.0 / 7.0)
}

/// Reynolds stress components for a turbulent channel at moderate Re.
/// Returns (R_xx, R_yy, R_zz, R_xy).
///
/// Physical conventions:
///   R_xx : streamwise normal stress (largest)
///   R_yy : wall-normal stress (smallest near walls)
///   R_zz : spanwise stress (intermediate)
///   R_xy : turbulent shear stress (antisymmetric in y, zero at walls and center)
fn reynolds_stress(y_norm: f32) -> (f32, f32, f32, f32) {
    let d = 1.0 - y_norm.abs(); // distance from nearest wall: 0 at wall, 1 at center

    // Amplitude: rises from zero at walls, levels near center.
    let amp = d.powf(0.5);

    let r_xx = 3.0 * amp;
    let r_yy = 0.8 * amp;
    let r_zz = 1.5 * amp;

    // Turbulent shear stress: peaks around d=0.5, antisymmetric in y.
    // This drives eigenvector rotation away from the flow axis.
    let r_xy = -4.0 * d * (1.0 - d) * y_norm.signum();

    (r_xx, r_yy, r_zz, r_xy)
}

/// Eigendecomposition of the 3D Reynolds stress tensor:
///
///   [R_xx, R_xy,  0  ]
///   [R_xy, R_yy,  0  ]
///   [0,    0,    R_zz]
///
/// The z-component is decoupled so we solve the 2x2 x-y block analytically.
fn rs_eigen(r_xx: f32, r_yy: f32, r_zz: f32, r_xy: f32)
    -> ([f32; 3], [[f32; 3]; 3])
{
    let mean  = (r_xx + r_yy) * 0.5;
    let diff  = (r_xx - r_yy) * 0.5;
    let disc  = (diff * diff + r_xy * r_xy).sqrt();

    let lam1 = mean + disc; // primary (largest in xy-plane)
    let lam2 = mean - disc; // secondary
    let lam3 = r_zz;

    // Rotation angle of eigenvectors in xy-plane.
    let theta = 0.5 * r_xy.atan2(diff);
    let (s, c) = theta.sin_cos();

    let e1 = [c, s, 0.0];       // primary eigenvector
    let e2 = [-s, c, 0.0];      // secondary eigenvector
    let e3 = [0.0, 0.0, 1.0];   // spanwise (z)

    ([lam1, lam2, lam3], [e1, e2, e3])
}

// ---------------------------------------------------------------------------
// Build / submit
// ---------------------------------------------------------------------------

pub(crate) fn build_tensor_glyph_scene(app: &mut App) {
    app.tg_built = true;
}

pub(crate) fn submit_tensor_glyphs(app: &App, fd: &mut FrameData) {
    let state = &app.tg_state;
    let ys = y_normals();

    // ------------------------------------------------------------------
    // Left column: mean velocity outer product v*v (rank-1 tensor).
    // Every glyph is a cigar pointing along X; only size varies.
    // ------------------------------------------------------------------
    {
        let mut positions    = Vec::new();
        let mut eigenvalues  = Vec::new();
        let mut eigenvectors = Vec::new();
        let mut scalars      = Vec::new();

        for &y_n in &ys {
            let y_w = y_n * Y_HALF;
            let u   = mean_u(y_n);
            positions.push([X_VEL, y_w, 0.0]);
            // Rank-1: only the x-eigenvalue is non-trivial.
            // Use a small nonzero value for the minor axes so the glyph is
            // visible (a true rank-1 tensor would be infinitely flat).
            eigenvalues.push([u, 0.08, 0.08]);
            eigenvectors.push([[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
            scalars.push(u);
        }

        let mut item = TensorGlyphItem::default();
        item.positions    = positions;
        item.eigenvalues  = eigenvalues;
        item.eigenvectors = eigenvectors;
        item.scale        = state.scale;
        item.color_attribute = Some(scalars);
        item.scalar_range    = Some((0.0, 1.0));
        // Sequential colormap for the velocity side (use viridis = None/default).
        item.colormap_id = None;
        fd.scene.tensor_glyphs.push(item);
    }

    // ------------------------------------------------------------------
    // Right column: full Reynolds stress tensor R_ij.
    // Glyphs change shape, size, and orientation across the channel.
    // ------------------------------------------------------------------
    {
        let mut positions    = Vec::new();
        let mut eigenvalues  = Vec::new();
        let mut eigenvectors = Vec::new();
        let mut scalars      = Vec::new();

        for &y_n in &ys {
            let y_w = y_n * Y_HALF;
            let (r_xx, r_yy, r_zz, r_xy) = reynolds_stress(y_n);
            let (evals, evecs) = rs_eigen(r_xx, r_yy, r_zz, r_xy);

            positions.push([X_RS, y_w, 0.0]);
            eigenvalues.push(evals);
            eigenvectors.push(evecs);
            // Color by turbulent kinetic energy k = (R_xx + R_yy + R_zz) / 2.
            scalars.push((r_xx + r_yy + r_zz) * 0.5);
        }

        let mut item = TensorGlyphItem::default();
        item.positions    = positions;
        item.eigenvalues  = eigenvalues;
        item.eigenvectors = eigenvectors;
        item.scale        = state.scale;
        item.color_attribute = Some(scalars);
        item.scalar_range    = Some((0.0, 3.5));
        item.colormap_id = if state.colormap_idx > 0 {
            Some(ColormapId(state.colormap_idx))
        } else {
            None
        };
        fd.scene.tensor_glyphs.push(item);
    }

    // ------------------------------------------------------------------
    // Labels
    // ------------------------------------------------------------------

    // Column headers
    for (x, text) in [
        (X_VEL, "Mean velocity v*v\n(rank-1 tensor)"),
        (X_RS,  "Reynolds stress R_ij\n(full rank-2 tensor)"),
    ] {
        let mut lbl = LabelItem::default();
        lbl.text         = text.to_string();
        lbl.world_anchor = Some([x, Y_HALF + 0.8, 0.0]);
        lbl.font_size    = 12.0;
        lbl.color        = [0.9, 0.9, 0.9, 1.0];
        fd.overlays.labels.push(lbl);
    }

    // Channel annotations in the center gap
    for (y_w, text) in [
        (-Y_HALF * 0.9, "Wall"),
        (0.0,            "Center\n(R_xy = 0)"),
        ( Y_HALF * 0.9, "Wall"),
    ] {
        let mut lbl = LabelItem::default();
        lbl.text         = text.to_string();
        lbl.world_anchor = Some([0.0, y_w, 0.0]);
        lbl.font_size    = 10.0;
        lbl.color        = [0.65, 0.65, 0.65, 1.0];
        fd.overlays.labels.push(lbl);
    }

    // Rotation callout at the quarter-height point (near peak R_xy)
    {
        let y_callout = Y_HALF * 0.45;
        let mut lbl = LabelItem::default();
        lbl.text         = "Axes rotated by\nR_xy shear stress".to_string();
        lbl.world_anchor = Some([X_RS + 2.0, y_callout, 0.0]);
        lbl.font_size    = 10.0;
        lbl.color        = [0.8, 0.75, 0.5, 1.0];
        fd.overlays.labels.push(lbl);
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_tensor_glyphs(app: &mut App, ui: &mut egui::Ui) {
    ui.label("Synthetic turbulent channel flow, two columns:");
    ui.label("Left: mean velocity encoded as a rank-1 tensor (v * v).");
    ui.label("  Every glyph is a cigar along X -- only the size changes.");
    ui.label("Right: Reynolds stress tensor R_ij.");
    ui.label("  Glyphs change shape, size, and orientation across the channel.");
    ui.label("  Near the quarter-height, R_xy rotates the principal axes ~25 deg");
    ui.label("  away from the flow direction -- invisible to a vector field.");
    ui.separator();

    ui.label("Scale:");
    ui.add(egui::Slider::new(&mut app.tg_state.scale, 0.1..=1.5));

    ui.separator();
    ui.label("Reynolds stress colormap (try 4=RdBu, 6=coolwarm, 0=viridis):");
    ui.add(egui::Slider::new(&mut app.tg_state.colormap_idx, 0..=15));
}
