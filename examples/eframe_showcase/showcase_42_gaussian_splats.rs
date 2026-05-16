//! Showcase 42: Gaussian Splats -- Scientific Visualization
//!
//! Two scenes where the Gaussian splat is not a visual approximation of some
//! other primitive -- it IS the mathematical object being visualized.
//!
//!   A) Diffusion tensor field: two orthogonal fiber tracts crossing at the
//!      origin. Each voxel is a Gaussian whose shape encodes the local
//!      diffusion tensor. Cigar-shaped splats in the tracts, spheres in the
//!      background. Colour by fractional anisotropy. Tilt along each tract axis
//!      and the splats foreshorten; the other tract remains visible. A point
//!      cloud cannot encode orientation at all.
//!
//!   B) Taylor-Green vortex: the TGV vorticity field evaluated analytically on
//!      a grid. Splats are placed at high-vorticity locations, elongated along
//!      the local vorticity vector, with opacity proportional to vorticity
//!      magnitude. Red splats rotate counterclockwise, blue clockwise. Look
//!      along a vortex tube axis to see circular cross-sections appear. No
//!      isosurface extraction, no geometry -- just the field.

use crate::App;
use eframe::egui;
use std::f32::consts::PI;
use viewport_lib::{
    FrameData, GaussianSplatData, GaussianSplatId, GaussianSplatItem, LightingSettings,
    SceneRenderItem, ShDegree, ViewportRenderer,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
pub(crate) enum SplatScene {
    Dti,
    Tgv,
}

pub(crate) struct GaussianSplatsState {
    pub built: bool,
    pub id_dti: GaussianSplatId,
    pub id_tgv: GaussianSplatId,
    pub scene: SplatScene,
    pub angle: f32,
}

impl Default for GaussianSplatsState {
    fn default() -> Self {
        let dummy = GaussianSplatItem::default().id;
        Self {
            built: false,
            id_dti: dummy,
            id_tgv: dummy,
            scene: SplatScene::Dti,
            angle: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

pub(crate) fn build_gaussian_splat_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    let dti = generate_dti();
    let tgv = generate_tgv();
    app.splat_state.id_dti = renderer.upload_gaussian_splats(&app.device, &app.queue, &dti);
    app.splat_state.id_tgv = renderer.upload_gaussian_splats(&app.device, &app.queue, &tgv);
    app.splat_state.built = true;
}

// ---------------------------------------------------------------------------
// Scene A: diffusion tensor field
// ---------------------------------------------------------------------------

// Two orthogonal fiber tracts (axes X and Z) crossing at the origin.
// Each voxel has a diffusion tensor encoded as position/scale/rotation.
// Colour by fractional anisotropy (FA): orange = high, blue = isotropic.
fn generate_dti() -> GaussianSplatData {
    const SH0_C: f32 = 0.28209479177;
    const TRACT_R: f32 = 0.60; // tract inclusion radius
    const BLEND_W: f32 = 0.70; // FA -> 0 over this extra width
    const SCALE_LONG: f32 = 0.24; // scale along fiber axis
    const SCALE_CROSS: f32 = 0.040; // scale perpendicular to fiber
    const SCALE_ISO: f32 = 0.095; // background isotropic scale

    let mut positions = Vec::new();
    let mut scales = Vec::new();
    let mut rotations = Vec::new();
    let mut opacities = Vec::new();
    let mut sh_coefficients = Vec::new();

    // Grid: -3.5..3.5 in x,y,z, step 0.36.
    let range = 3.5_f32;
    let step = 0.36_f32;
    let steps = ((2.0 * range) / step).ceil() as i32 + 1;

    for ix in 0..steps {
        for iy in 0..steps {
            for iz in 0..steps {
                let x = -range + ix as f32 * step;
                let y = -range + iy as f32 * step;
                let z = -range + iz as f32 * step;

                // Distance to each tract axis.
                // Tract A: X-axis (y=0, z=0)
                // Tract B: Z-axis (x=0, y=0)
                let dist_a = (y * y + z * z).sqrt();
                let dist_b = (x * x + y * y).sqrt();

                // Smooth FA: 1 in core, fades to 0 at TRACT_R + BLEND_W.
                let fa_a = fa_blend(dist_a, TRACT_R, BLEND_W);
                let fa_b = fa_blend(dist_b, TRACT_R, BLEND_W);
                let fa = fa_a.max(fa_b);

                // Background: include within a sphere, at low opacity.
                let bg_dist = (x * x + y * y + z * z).sqrt();
                if fa < 0.01 && bg_dist > 2.8 {
                    continue;
                }

                let opacity = if fa > 0.6 {
                    0.85
                } else if fa > 0.0 {
                    0.25 + fa * 0.95
                } else {
                    0.22
                };

                // Scale and rotation from the dominant tract.
                let (scale, rot) = if fa_a >= fa_b && fa_a > 0.05 {
                    // Tract A: elongated along X.
                    let t = fa_a.min(1.0);
                    let c = SCALE_ISO + (SCALE_CROSS - SCALE_ISO) * t;
                    let l = SCALE_ISO + (SCALE_LONG - SCALE_ISO) * t;
                    ([c, c, l], quat_align_z_to([1.0, 0.0, 0.0]))
                } else if fa_b > 0.05 {
                    // Tract B: elongated along Z (identity rotation).
                    let t = fa_b.min(1.0);
                    let c = SCALE_ISO + (SCALE_CROSS - SCALE_ISO) * t;
                    let l = SCALE_ISO + (SCALE_LONG - SCALE_ISO) * t;
                    ([c, c, l], [0.0_f32, 0.0, 0.0, 1.0])
                } else {
                    // Background: isotropic.
                    ([SCALE_ISO, SCALE_ISO, SCALE_ISO], [0.0_f32, 0.0, 0.0, 1.0])
                };

                // Colour: hue 0.63 (blue) at FA=0 -> hue 0.07 (orange) at FA=1.
                let hue = 0.63 - fa * 0.56;
                let (r, g, b) = hsl_to_linear(hue, 0.85, 0.55);

                positions.push([x, y, z]);
                scales.push(scale);
                rotations.push(rot);
                opacities.push(opacity);
                sh_coefficients.push((r - 0.5) / SH0_C);
                sh_coefficients.push((g - 0.5) / SH0_C);
                sh_coefficients.push((b - 0.5) / SH0_C);
            }
        }
    }

    GaussianSplatData {
        positions,
        scales,
        rotations,
        opacities,
        sh_coefficients,
        sh_degree: ShDegree::Zero,
    }
}

// FA smoothly falls from 1.0 at dist=0 to 0.0 at dist = radius + blend_width.
fn fa_blend(dist: f32, radius: f32, blend_width: f32) -> f32 {
    if dist <= radius {
        1.0
    } else if dist < radius + blend_width {
        1.0 - (dist - radius) / blend_width
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Scene B: Taylor-Green vortex
// ---------------------------------------------------------------------------

// Velocity field (t=0, periodic in [0, 2pi]^3):
//   u =  sin(x) cos(y) cos(z)
//   v = -cos(x) sin(y) cos(z)
//   w =  0
//
// Vorticity curl(u):
//   omega_x = -cos(x) sin(y) sin(z)
//   omega_y = -sin(x) cos(y) sin(z)
//   omega_z =  2 sin(x) sin(y) cos(z)
//
// Splats are placed where |omega| exceeds a threshold, elongated along omega.
// Colour: red (omega_z > 0, CCW), blue (omega_z < 0, CW).
fn generate_tgv() -> GaussianSplatData {
    const SH0_C: f32 = 0.28209479177;
    const THRESHOLD: f32 = 0.40; // fraction of max |omega| below which to skip

    let n = 24_usize; // 24^3 = 13824 grid points before threshold

    // First pass: compute all vorticity magnitudes to find max.
    let mut pts: Vec<([f32; 3], [f32; 3])> = Vec::with_capacity(n * n * n);
    let mut max_mag = 0.0_f32;

    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let x = ix as f32 / n as f32 * 2.0 * PI;
                let y = iy as f32 / n as f32 * 2.0 * PI;
                let z = iz as f32 / n as f32 * 2.0 * PI;

                let ox = -x.cos() * y.sin() * z.sin();
                let oy = -x.sin() * y.cos() * z.sin();
                let oz = 2.0 * x.sin() * y.sin() * z.cos();
                let mag = (ox * ox + oy * oy + oz * oz).sqrt();
                if mag > max_mag {
                    max_mag = mag;
                }

                // Shift to center at origin for display.
                pts.push(([x - PI, y - PI, z - PI], [ox, oy, oz]));
            }
        }
    }

    let threshold_mag = THRESHOLD * max_mag;

    let mut positions = Vec::new();
    let mut scales = Vec::new();
    let mut rotations = Vec::new();
    let mut opacities = Vec::new();
    let mut sh_coefficients = Vec::new();

    for ([px, py, pz], [ox, oy, oz]) in &pts {
        let mag = (ox * ox + oy * oy + oz * oz).sqrt();
        if mag < threshold_mag {
            continue;
        }

        // Opacity proportional to (normalized vorticity)^1.5.
        let norm = (mag / max_mag).min(1.0);
        let opacity = norm.powf(1.5) * 0.88;

        // Tighter cross-section for stronger vorticity (concentrated core).
        let cross = (0.20 - 0.12 * norm).max(0.05);
        let long = 0.26;

        // Align splat long axis to vorticity direction.
        let inv = 1.0 / mag;
        let rot = quat_align_z_to([ox * inv, oy * inv, oz * inv]);

        // Colour by sign of omega_z: red (CCW) vs blue (CW).
        let hue = if *oz >= 0.0 { 0.02 } else { 0.62 };
        let (r, g, b) = hsl_to_linear(hue, 0.90, 0.55);

        positions.push([*px, *py, *pz]);
        scales.push([cross, cross, long]);
        rotations.push(rot);
        opacities.push(opacity);
        sh_coefficients.push((r - 0.5) / SH0_C);
        sh_coefficients.push((g - 0.5) / SH0_C);
        sh_coefficients.push((b - 0.5) / SH0_C);
    }

    GaussianSplatData {
        positions,
        scales,
        rotations,
        opacities,
        sh_coefficients,
        sh_degree: ShDegree::Zero,
    }
}

// ---------------------------------------------------------------------------
// Shared math helpers
// ---------------------------------------------------------------------------

/// Quaternion that rotates the local Z axis onto `n` (unit vector).
/// Uses the half-angle between (0,0,1) and n. Degenerate case n=(0,0,-1)
/// gets a 180-degree rotation around X.
fn quat_align_z_to(n: [f32; 3]) -> [f32; 4] {
    let [nx, ny, nz] = n;
    if nz < -0.9999 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    // cross((0,0,1), n) = (-ny, nx, 0)
    let qw = ((1.0 + nz) / 2.0_f32).sqrt();
    let s = 1.0 / (2.0 * qw);
    [-ny * s, nx * s, 0.0, qw]
}

fn hsl_to_linear(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - (h6 % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if h6 < 1.0 {
        (c, x, 0.0)
    } else if h6 < 2.0 {
        (x, c, 0.0)
    } else if h6 < 3.0 {
        (0.0, c, x)
    } else if h6 < 4.0 {
        (0.0, x, c)
    } else if h6 < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    let lin = |v: f32| (v + m).clamp(0.0, 1.0).powi(2);
    (lin(r1), lin(g1), lin(b1))
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_gaussian_splats(app: &mut App, ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.label("Scene:");
        ui.selectable_value(
            &mut app.splat_state.scene,
            SplatScene::Dti,
            "Diffusion Tensor Field",
        );
        ui.selectable_value(
            &mut app.splat_state.scene,
            SplatScene::Tgv,
            "Taylor-Green Vortex",
        );
    });
    ui.separator();

    match app.splat_state.scene {
        SplatScene::Dti => {
            ui.label("Two fiber tracts (X and Z axes) cross at the origin. Each splat encodes the local diffusion tensor: cigar-shaped in tracts, spherical in background. Colour = fractional anisotropy (orange high, blue low).");
            ui.label("Tip: look along the X axis -- that tract foreshortens to dots while the Z tract remains as needles.");
        }
        SplatScene::Tgv => {
            ui.label("Taylor-Green vortex vorticity field. Each splat is aligned with the local vorticity vector; opacity = vorticity magnitude. Red = counterclockwise rotation, blue = clockwise.");
            ui.label(
                "Tip: look along the Z axis to see circular cross-sections of the vortex tubes.",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame update and item
// ---------------------------------------------------------------------------

pub(crate) fn update_gaussian_splats(app: &mut App, dt: f32) {
    app.splat_state.angle += dt * 0.10;
}

pub(crate) fn gaussian_splat_items(app: &App) -> Vec<GaussianSplatItem> {
    if !app.splat_state.built {
        return vec![];
    }
    let rot = glam::Mat4::from_rotation_y(app.splat_state.angle);
    let mut item = GaussianSplatItem::default();
    item.id = match app.splat_state.scene {
        SplatScene::Dti => app.splat_state.id_dti,
        SplatScene::Tgv => app.splat_state.id_tgv,
    };
    item.model = rot.to_cols_array_2d();
    vec![item]
}

// ---------------------------------------------------------------------------
// Frame assembly
// ---------------------------------------------------------------------------

pub(crate) fn splat_collect_scene_items(
    _app: &App,
) -> (Vec<SceneRenderItem>, LightingSettings, u64, u64) {
    (vec![], LightingSettings::default(), 0, 0)
}

pub(crate) fn submit_splat_items(app: &App, fd: &mut FrameData) {
    if !app.splat_state.built {
        return;
    }
    fd.scene.gaussian_splats.extend(gaussian_splat_items(app));
}
