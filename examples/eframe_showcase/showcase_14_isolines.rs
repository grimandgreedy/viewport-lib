//! Showcase 14: Isolines & Contours
//!
//! A wave-function grid mesh colored by its scalar field, with isoline
//! contour strips rendered on top via the `SceneFrame::isolines` pipeline.
//!
//! The mesh is built once and uploaded. `IsolineItem`s are re-submitted each
//! frame with the current slider/color/width settings — no re-upload needed.

use crate::App;
use viewport_lib::{AttributeData, Material, MeshData, MeshId, scene::Scene};

impl App {
    /// Build the scene for Showcase 14 (Isolines demo).
    pub(crate) fn build_iso_scene(&mut self, renderer: &mut viewport_lib::ViewportRenderer) {
        self.iso_scene = Scene::new();

        let res = self.iso_grid_resolution;
        let (mesh, scalars) = make_wave_grid_iso(res, res, 10.0);

        // Keep CPU copies for IsolineItem re-submission each frame.
        self.iso_positions = mesh.positions.clone();
        self.iso_indices = mesh.indices.clone();
        self.iso_scalars = scalars;

        let mesh_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &mesh)
            .expect("iso wave mesh");
        let mesh_id = MeshId::from_index(mesh_idx);
        self.iso_mesh_index = mesh_idx;

        self.iso_scene.add_named(
            "Wave Grid",
            Some(mesh_id),
            glam::Mat4::IDENTITY,
            { let mut m = Material::from_color([0.6, 0.65, 0.7]); m.roughness = 0.6; m },
        );

        self.iso_built = true;
    }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Build a wave-function grid with per-vertex "wave" scalar attribute.
fn make_wave_grid_iso(cols: u32, rows: u32, size: f32) -> (MeshData, Vec<f32>) {
    let nx = cols + 1;
    let ny = rows + 1;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((nx * ny) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((nx * ny) as usize);
    let mut scalars: Vec<f32> = Vec::with_capacity((nx * ny) as usize);

    for iy in 0..ny {
        for ix in 0..nx {
            let u = ix as f32 / cols as f32;
            let v = iy as f32 / rows as f32;
            let x = (u - 0.5) * size;
            let y = (v - 0.5) * size;
            let wave = (x * 1.2).sin() * (y * 1.0).cos();
            let z = wave * 0.6;
            positions.push([x, y, z]);
            normals.push([0.0, 0.0, 1.0]);
            scalars.push(wave);
        }
    }

    // CCW winding viewed from +Z so the top face is the front face.
    let mut indices: Vec<u32> = Vec::with_capacity((rows * cols * 6) as usize);
    for iy in 0..rows {
        for ix in 0..cols {
            let base = iy * nx + ix;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + nx);
            indices.push(base + 1);
            indices.push(base + nx + 1);
            indices.push(base + nx);
        }
    }

    let mut mesh = MeshData::default();
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.indices = indices;
    mesh.attributes
        .insert("wave".to_string(), AttributeData::Vertex(scalars.clone()));
    (mesh, scalars)
}
