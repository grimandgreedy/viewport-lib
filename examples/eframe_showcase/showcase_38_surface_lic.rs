//! Showcase 38: Surface Line Integral Convolution (LIC)
//!
//! Three rows of objects, stacked front-to-back in Z. Each row demonstrates
//! LIC on a different surface type, with three flow scenarios side by side.
//!
//!   Z=0    Tori
//!     col 0  toroidal  -- flow follows the main ring
//!     col 1  poloidal  -- flow circles the tube cross-section
//!     col 2  helical   -- combination, winds around the surface
//!
//!   Z=5    Bumpy terrain (the main demo)
//!     col 0  topographic -- flow runs downhill (gradient descent)
//!     col 1  uniform +X  -- constant wind direction
//!     col 2  vortex      -- circular swirl
//!
//!   Z=10   Spheres
//!     col 0  latitude  -- flow follows parallels (east-west)
//!     col 1  meridian  -- flow follows great circles (north-south)
//!     col 2  diagonal  -- combination
//!
//! The terrain row is the clearest demonstration of utility: topographic flow
//! (col 0) shows where runoff converges in valleys and diverges at ridges,
//! which is not readable from geometry alone. Toggle LIC off to see how much
//! information is lost.

use crate::App;
use eframe::egui;
use std::collections::HashMap;
use viewport_lib::{
    AttributeData, BackfacePolicy, FrameData, LicOverlay, Material, MeshData, MeshId,
    SurfaceLICConfig, SurfaceSubmission, ViewportRenderer, scene::Scene,
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

pub(crate) struct LicState {
    pub built: bool,
    pub scene: Scene,
    /// mesh_ids[row][col]: row = surface type, col = flow scenario.
    pub mesh_ids: [[Option<MeshId>; 3]; 3],
    pub steps: u32,
    pub step_size: f32,
    pub strength: f32,
    pub enabled: bool,
}

impl Default for LicState {
    fn default() -> Self {
        Self {
            built: false,
            scene: Scene::new(),
            mesh_ids: [[None; 3]; 3],
            steps: 20,
            step_size: 1.5,
            strength: 2.0,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

const COL_X: [f32; 3] = [-6.0, 0.0, 6.0];
const ROW_Z: [f32; 3] = [0.0, 7.5, 15.0];

fn transform(col: usize, row: usize) -> glam::Mat4 {
    glam::Mat4::from_translation(glam::Vec3::new(COL_X[col], 0.0, ROW_Z[row]))
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn safe_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-6 {
        [0.0; 3]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn project_tangent(v: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    let d = dot(v, n);
    safe_normalize([v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]])
}

// ---------------------------------------------------------------------------
// Torus flow fields
// ---------------------------------------------------------------------------

fn flow_torus_toroidal(theta: f32) -> [f32; 3] {
    safe_normalize([-theta.sin(), 0.0, theta.cos()])
}

fn flow_torus_poloidal(theta: f32, phi: f32) -> [f32; 3] {
    safe_normalize([
        -phi.sin() * theta.cos(),
        phi.cos(),
        -phi.sin() * theta.sin(),
    ])
}

fn flow_torus_helical(theta: f32, phi: f32) -> [f32; 3] {
    let t = flow_torus_toroidal(theta);
    let p = flow_torus_poloidal(theta, phi);
    safe_normalize([t[0] + p[0], t[1] + p[1], t[2] + p[2]])
}

// ---------------------------------------------------------------------------
// Terrain flow fields
// ---------------------------------------------------------------------------

/// Topographic flow: gradient descent along the height function.
/// Water (or wind following terrain) runs downhill -- converges in valleys,
/// diverges at ridges.
fn flow_topographic(pos: [f32; 3], n: [f32; 3], amp: f32, freq: f32) -> [f32; 3] {
    // Gradient of y = amp*sin(freq*x)*sin(freq*z) in the XZ plane.
    let dy_dx = amp * freq * (freq * pos[0]).cos() * (freq * pos[2]).sin();
    let dy_dz = amp * freq * (freq * pos[0]).sin() * (freq * pos[2]).cos();
    // Downhill direction: negate the gradient and project onto the surface.
    project_tangent([-dy_dx, 0.0, -dy_dz], n)
}

fn flow_uniform(n: [f32; 3]) -> [f32; 3] {
    project_tangent([1.0, 0.0, 0.0], n)
}

fn flow_vortex(pos: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    project_tangent([-pos[2], 0.0, pos[0]], n)
}

// ---------------------------------------------------------------------------
// Sphere flow fields
// ---------------------------------------------------------------------------

/// Latitude flow: east-west along parallels (like trade winds or ocean gyres).
fn flow_sphere_latitude(pos: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    // Tangent perpendicular to the meridian plane containing pos and Y.
    project_tangent([-pos[2], 0.0, pos[0]], n)
}

/// Meridian flow: north-south along great circles.
fn flow_sphere_meridian(_pos: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    // Tangent toward the north pole: project world-Y onto the surface.
    project_tangent([0.0, 1.0, 0.0], n)
}

/// Diagonal: combination of latitude and meridian.
fn flow_sphere_diagonal(pos: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    let a = flow_sphere_latitude(pos, n);
    let b = flow_sphere_meridian(pos, n);
    safe_normalize([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
}

// ---------------------------------------------------------------------------
// Geometry builders
// ---------------------------------------------------------------------------

fn build_torus(
    ms: usize,
    ns: usize,
    rm: f32,
    rn: f32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut ths = Vec::new();
    let mut phs = Vec::new();
    for j in 0..ms {
        let th = 2.0 * std::f32::consts::PI * j as f32 / ms as f32;
        let (st, ct) = (th.sin(), th.cos());
        for i in 0..ns {
            let ph = 2.0 * std::f32::consts::PI * i as f32 / ns as f32;
            let (sp, cp) = (ph.sin(), ph.cos());
            pos.push([(rm + rn * cp) * ct, rn * sp, (rm + rn * cp) * st]);
            nrm.push(safe_normalize([cp * ct, sp, cp * st]));
            ths.push(th);
            phs.push(ph);
        }
    }
    let mut idx = Vec::new();
    for j in 0..ms {
        let jn = (j + 1) % ms;
        for i in 0..ns {
            let ni = (i + 1) % ns;
            let a = (j * ns + i) as u32;
            let b = (j * ns + ni) as u32;
            let c = (jn * ns + ni) as u32;
            let d = (jn * ns + i) as u32;
            idx.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    (pos, nrm, idx, ths, phs)
}

fn build_bumpy(
    segs: usize,
    size: f32,
    amp: f32,
    freq: f32,
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let n = segs + 1;
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    for iz in 0..n {
        for ix in 0..n {
            let x = (ix as f32 / segs as f32 - 0.5) * size;
            let z = (iz as f32 / segs as f32 - 0.5) * size;
            let y = amp * (freq * x).sin() * (freq * z).sin();
            let dy_dx = amp * freq * (freq * x).cos() * (freq * z).sin();
            let dy_dz = amp * freq * (freq * x).sin() * (freq * z).cos();
            pos.push([x, y, z]);
            nrm.push(safe_normalize([-dy_dx, 1.0, -dy_dz]));
        }
    }
    let mut idx = Vec::new();
    for iz in 0..segs {
        for ix in 0..segs {
            let a = (iz * n + ix) as u32;
            let b = (iz * n + ix + 1) as u32;
            let c = ((iz + 1) * n + ix + 1) as u32;
            let d = ((iz + 1) * n + ix) as u32;
            idx.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    (pos, nrm, idx)
}

fn build_sphere(lon: usize, lat: usize, r: f32) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let mut pos = Vec::new();
    let mut nrm = Vec::new();
    let mut idx = Vec::new();
    for la in 0..=lat {
        let th = std::f32::consts::PI * la as f32 / lat as f32;
        let (st, ct) = (th.sin(), th.cos());
        for lo in 0..=lon {
            let ph = 2.0 * std::f32::consts::PI * lo as f32 / lon as f32;
            let (sp, cp) = (ph.sin(), ph.cos());
            let x = st * cp;
            let y = ct;
            let z = st * sp;
            pos.push([x * r, y * r, z * r]);
            nrm.push([x, y, z]);
        }
    }
    let s = lon + 1;
    for la in 0..lat {
        for lo in 0..lon {
            let a = (la * s + lo) as u32;
            let b = ((la + 1) * s + lo) as u32;
            let c = ((la + 1) * s + lo + 1) as u32;
            let d = (la * s + lo + 1) as u32;
            if la > 0 {
                idx.extend_from_slice(&[a, d, b]);
            }
            if la < lat - 1 {
                idx.extend_from_slice(&[b, d, c]);
            }
        }
    }
    (pos, nrm, idx)
}

// ---------------------------------------------------------------------------
// Upload helpers
// ---------------------------------------------------------------------------

// Mid-tone colours (linear 0..1) so LIC can both darken and brighten visibly.
// White surfaces clip brightening; these sit around 0.4-0.5 for symmetric contrast.
const ROW_COLOURS: [[f32; 3]; 3] = [
    [0.28, 0.48, 0.72], // steel blue -- tori
    [0.35, 0.55, 0.28], // sage green -- terrain
    [0.65, 0.36, 0.22], // terracotta -- spheres
];

fn make_material(row: usize) -> Material {
    let mut m = Material::from_colour(ROW_COLOURS[row]);
    m.backface_policy = BackfacePolicy::Identical;
    m
}

fn upload_mesh(
    app: &mut App,
    renderer: &mut ViewportRenderer,
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    idx: Vec<u32>,
    flow: Vec<[f32; 3]>,
    col: usize,
    row: usize,
) -> MeshId {
    let mut mesh = MeshData::default();
    mesh.positions = pos;
    mesh.normals = nrm;
    mesh.indices = idx;
    mesh.attributes = {
        let mut m = HashMap::new();
        m.insert("flow".to_string(), AttributeData::VertexVector(flow));
        m
    };
    let id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &mesh)
        .expect("LIC mesh upload");
    app.lic_state
        .scene
        .add(Some(id), transform(col, row), make_material(row));
    id
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

pub(crate) fn build_lic_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    app.lic_state.scene = Scene::new();
    let mut ids = [[None; 3]; 3];

    let terrain_size = 4.8f32;
    let terrain_amp = 0.55f32;
    let terrain_freq = 3.0 * std::f32::consts::PI / terrain_size;

    for col in 0..3usize {
        // Row 0: Tori -- toroidal, poloidal, helical.
        {
            let (pos, nrm, idx, ths, phs) = build_torus(56, 28, 1.8, 0.65);
            let flow: Vec<_> = ths
                .iter()
                .zip(phs.iter())
                .map(|(&t, &p)| match col {
                    0 => flow_torus_toroidal(t),
                    1 => flow_torus_poloidal(t, p),
                    _ => flow_torus_helical(t, p),
                })
                .collect();
            ids[0][col] = Some(upload_mesh(app, renderer, pos, nrm, idx, flow, col, 0));
        }

        // Row 1: Bumpy terrain -- topographic, uniform wind, vortex.
        {
            let (pos, nrm, idx) = build_bumpy(48, terrain_size, terrain_amp, terrain_freq);
            let flow: Vec<_> = pos
                .iter()
                .zip(nrm.iter())
                .map(|(&p, &n)| match col {
                    0 => flow_topographic(p, n, terrain_amp, terrain_freq),
                    1 => flow_uniform(n),
                    _ => flow_vortex(p, n),
                })
                .collect();
            ids[1][col] = Some(upload_mesh(app, renderer, pos, nrm, idx, flow, col, 1));
        }

        // Row 2: Spheres -- latitude, meridian, diagonal.
        {
            let (pos, nrm, idx) = build_sphere(64, 32, 2.2);
            let flow: Vec<_> = pos
                .iter()
                .zip(nrm.iter())
                .map(|(&p, &n)| match col {
                    0 => flow_sphere_latitude(p, n),
                    1 => flow_sphere_meridian(p, n),
                    _ => flow_sphere_diagonal(p, n),
                })
                .collect();
            ids[2][col] = Some(upload_mesh(app, renderer, pos, nrm, idx, flow, col, 2));
        }
    }

    app.lic_state.mesh_ids = ids;
    app.lic_state.built = true;
}

// ---------------------------------------------------------------------------
// Per-frame submission
// ---------------------------------------------------------------------------

pub(crate) fn submit_lic_items(app: &App, fd: &mut FrameData) {
    let state = &app.lic_state;
    if !state.enabled {
        return;
    }

    let mut config = SurfaceLICConfig::default();
    config.steps = state.steps;
    config.step_size = state.step_size;
    config.strength = state.strength;

    // Build a lookup from MeshId to LicOverlay.
    let mut lic_by_mesh: std::collections::HashMap<MeshId, LicOverlay> =
        std::collections::HashMap::new();
    for (row, row_ids) in state.mesh_ids.iter().enumerate() {
        for (_col, maybe_id) in row_ids.iter().enumerate() {
            let Some(mesh_id) = *maybe_id else { continue };
            lic_by_mesh.insert(mesh_id, LicOverlay::new("flow", config.clone()));
        }
    }

    // Apply to matching SceneRenderItems.
    if let SurfaceSubmission::Flat(ref mut items) = fd.scene.surfaces {
        let items = std::sync::Arc::make_mut(items);
        for item in items.iter_mut() {
            if let Some(lic) = lic_by_mesh.remove(&item.mesh_id) {
                item.lic = Some(lic);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_lic(app: &mut App, ui: &mut egui::Ui) {
    let state = &mut app.lic_state;

    egui::Grid::new("lic_labels")
        .num_columns(4)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            ui.label("");
            ui.strong("Col 0");
            ui.strong("Col 1");
            ui.strong("Col 2");
            ui.end_row();
            ui.strong("Tori");
            ui.label("toroidal");
            ui.label("poloidal");
            ui.label("helical");
            ui.end_row();
            ui.strong("Terrain");
            ui.label("topographic");
            ui.label("wind (+X)");
            ui.label("vortex");
            ui.end_row();
            ui.strong("Spheres");
            ui.label("latitude");
            ui.label("meridian");
            ui.label("diagonal");
            ui.end_row();
        });

    ui.separator();
    ui.label(
        "Topographic flow (terrain col 0) shows where runoff converges\n\
              in valleys and diverges at ridges. Toggle LIC off to see\n\
              how much flow topology is lost.",
    );
    ui.separator();

    ui.checkbox(&mut state.enabled, "LIC enabled");
    ui.separator();

    ui.add_enabled_ui(state.enabled, |ui| {
        ui.label("Steps:");
        ui.add(egui::Slider::new(&mut state.steps, 1..=80));
        ui.label("Step size (px):");
        ui.add(egui::Slider::new(&mut state.step_size, 0.5..=4.0));
        ui.label("Strength:");
        ui.add(egui::Slider::new(&mut state.strength, 0.0..=2.0));
    });
}
