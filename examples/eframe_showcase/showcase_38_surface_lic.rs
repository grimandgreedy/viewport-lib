//! Showcase 38: Surface Line Integral Convolution (LIC)
//!
//! Three spheres, each carrying a different tangential vector field, displayed
//! side by side so the effect is immediately readable:
//!
//!   Left  -- Y-axis vortex: circular flow around the vertical axis.
//!             LIC produces horizontal bands like latitude lines.
//!
//!   Center -- Meridional flow: tangent direction pointing toward the north pole
//!             at every vertex. LIC produces vertical streaks from pole to pole.
//!
//!   Right  -- X-axis vortex: circular flow around the horizontal axis.
//!             LIC produces rings oriented differently from the left sphere,
//!             demonstrating that the pattern tracks the chosen axis, not a
//!             fixed screen direction.
//!
//! All three spheres share the same LIC parameters (steps, step size, strength)
//! so you can sweep the sliders and see identical parameter changes applied
//! consistently across different flow structures.

use eframe::egui;
use std::collections::HashMap;
use viewport_lib::{
    AttributeData, FrameData, Material, MeshData, MeshId, SurfaceLICConfig, SurfaceLICItem,
    ViewportRenderer,
};

use crate::App;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct LicState {
    /// mesh_ids[0] = Y-axis vortex, [1] = meridional, [2] = X-axis vortex.
    pub mesh_ids: [Option<MeshId>; 3],
    pub steps: u32,
    pub step_size: f32,
    pub strength: f32,
    pub noise_scale: f32,
    pub enabled: bool,
}

impl Default for LicState {
    fn default() -> Self {
        Self {
            mesh_ids: [None; 3],
            steps: 20,
            step_size: 1.0,
            strength: 0.8,
            noise_scale: 4.0,
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Build a UV sphere and return (positions, normals, indices).
fn build_sphere(lon: usize, lat: usize, radius: f32) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for la in 0..=lat {
        let theta = std::f32::consts::PI * la as f32 / lat as f32;
        let (st, ct) = (theta.sin(), theta.cos());
        for lo in 0..=lon {
            let phi = 2.0 * std::f32::consts::PI * lo as f32 / lon as f32;
            let (sp, cp) = (phi.sin(), phi.cos());
            let x = st * cp;
            let y = ct;
            let z = st * sp;
            positions.push([x * radius, y * radius, z * radius]);
            normals.push([x, y, z]);
        }
    }

    let stride = lon + 1;
    for la in 0..lat {
        for lo in 0..lon {
            let a = (la * stride + lo) as u32;
            let b = ((la + 1) * stride + lo) as u32;
            let c = ((la + 1) * stride + lo + 1) as u32;
            let d = (la * stride + lo + 1) as u32;
            if la > 0 {
                indices.extend_from_slice(&[a, d, b]);
            }
            if la < lat - 1 {
                indices.extend_from_slice(&[b, d, c]);
            }
        }
    }

    (positions, normals, indices)
}

/// Normalize a 3-vector; returns the zero vector when near-zero length.
fn safe_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-6 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Y-axis vortex: flow tangent = normalize(cross(normal, Y)).
/// Creates horizontal bands (latitude-aligned streaks).
fn flow_y_vortex(n: [f32; 3]) -> [f32; 3] {
    safe_normalize(cross(n, [0.0, 1.0, 0.0]))
}

/// Meridional flow: tangent direction pointing toward the north pole.
/// flow = normalize(cross(cross(normal, Y), normal))
/// Creates vertical streaks from south to north pole.
fn flow_meridional(n: [f32; 3]) -> [f32; 3] {
    let east = cross(n, [0.0, 1.0, 0.0]);
    safe_normalize(cross(east, n))
}

/// X-axis vortex: flow tangent = normalize(cross(normal, X)).
/// Creates rings around the horizontal axis -- visually distinct orientation.
fn flow_x_vortex(n: [f32; 3]) -> [f32; 3] {
    safe_normalize(cross(n, [1.0, 0.0, 0.0]))
}

/// Upload one sphere with the given vector field as a "flow" attribute.
fn upload_sphere(
    app: &mut App,
    renderer: &mut ViewportRenderer,
    offset_x: f32,
    flow_fn: impl Fn([f32; 3]) -> [f32; 3],
) -> MeshId {
    let (positions, normals, indices) = build_sphere(48, 24, 1.5);

    let flow_vectors: Vec<[f32; 3]> = normals.iter().map(|&n| flow_fn(n)).collect();

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh.attributes = {
        let mut m = HashMap::new();
        m.insert("flow".to_string(), AttributeData::VertexVector(flow_vectors));
        m
    };

    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&app.device, &mesh)
        .expect("LIC sphere mesh upload");

    // Add to scene with the horizontal offset.
    let transform = glam::Mat4::from_translation(glam::Vec3::new(offset_x, 0.0, 0.0));
    app.lic_scene.add(Some(mesh_id), transform, Material::default());

    mesh_id
}

// ---------------------------------------------------------------------------
// Build (called once)
// ---------------------------------------------------------------------------

pub(crate) fn build_lic_scene(app: &mut App, renderer: &mut ViewportRenderer) {
    app.lic_scene = viewport_lib::scene::Scene::new();

    // Left sphere: Y-axis vortex (horizontal bands).
    let id0 = upload_sphere(app, renderer, -3.5, flow_y_vortex);
    // Center sphere: meridional flow (vertical streaks).
    let id1 = upload_sphere(app, renderer, 0.0, flow_meridional);
    // Right sphere: X-axis vortex (tilted rings).
    let id2 = upload_sphere(app, renderer, 3.5, flow_x_vortex);

    app.lic_state.mesh_ids = [Some(id0), Some(id1), Some(id2)];
    app.lic_built = true;
}

// ---------------------------------------------------------------------------
// Per-frame item submission
// ---------------------------------------------------------------------------

/// Positions matching the build offsets.
const OFFSETS: [f32; 3] = [-3.5, 0.0, 3.5];

pub(crate) fn submit_lic_items(app: &App, fd: &mut FrameData) {
    let state = &app.lic_state;
    if !state.enabled {
        return;
    }
    let mut config = SurfaceLICConfig::default();
    config.steps = state.steps;
    config.step_size = state.step_size;
    config.strength = state.strength;
    config.noise_scale = state.noise_scale;

    for (i, maybe_id) in state.mesh_ids.iter().enumerate() {
        let Some(mesh_id) = *maybe_id else { continue };
        let transform = glam::Mat4::from_translation(glam::Vec3::new(OFFSETS[i], 0.0, 0.0));
        fd.scene.lic_items.push(SurfaceLICItem::new(
            mesh_id,
            "flow",
            transform.to_cols_array_2d(),
            config.clone(),
        ));
    }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

pub(crate) fn controls_lic(app: &mut App, ui: &mut egui::Ui) {
    let state = &mut app.lic_state;

    ui.label("Three spheres with different tangential vector fields.");
    ui.label("LIC makes the flow direction readable as surface streaks.");
    ui.separator();

    ui.label("Left: Y-axis vortex (horizontal bands)");
    ui.label("Center: Meridional flow (pole-to-pole streaks)");
    ui.label("Right: X-axis vortex (tilted rings)");
    ui.separator();

    ui.checkbox(&mut state.enabled, "LIC enabled");
    ui.separator();

    ui.add_enabled_ui(state.enabled, |ui| {
        ui.label("Steps:");
        ui.add(egui::Slider::new(&mut state.steps, 1..=64));
        ui.label("Step size (px):");
        ui.add(egui::Slider::new(&mut state.step_size, 0.25..=4.0));
        ui.label("Strength:");
        ui.add(egui::Slider::new(&mut state.strength, 0.0..=2.0));
        ui.label("Noise scale:");
        ui.add(egui::Slider::new(&mut state.noise_scale, 0.5..=16.0));
    });
}
