//! Showcase 12: Scalar Fields & Colormaps : build and overlay methods.
//!
//! Three objects each carrying a different procedural scalar attribute:
//!   Object 0 : Sphere,    attribute "height"   (world Z of each vertex, 0..1 range)
//!   Object 1 : Wave Grid, attribute "wave"      (sine-derived 2-D wave, -1..1 range)
//!   Object 2 : Box,       attribute "distance"  (distance from center, with NaN below 0.3)

use crate::App;
use viewport_lib::ScalarBarItem;
use viewport_lib::{AttributeData, Material, MeshData, ViewportRenderer, scene::Scene};

impl App {
    /// Build the scene for Showcase 12 (Scalar Fields demo).
    pub(crate) fn build_scalar_scene(&mut self, renderer: &mut ViewportRenderer) {
        self.scalar_scene = Scene::new();

        // ---- Object 0: Sphere with height (z) scalar ----
        let mut sphere = viewport_lib::primitives::sphere(3.0, 48, 24);
        let height_scalars: Vec<f32> = sphere
            .positions
            .iter()
            .map(|p| (p[2] + 3.0) / 6.0) // normalize z from [-3,3] -> [0,1]
            .collect();
        sphere.attributes.insert(
            "height".to_string(),
            AttributeData::Vertex(height_scalars.clone()),
        );
        let sphere_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("scalar sphere mesh");
        let sphere_node = self.scalar_scene.add_named(
            "Sphere",
            Some(sphere_id),
            glam::Mat4::from_translation(glam::Vec3::new(-6.0, 0.0, 0.0)),
            {
                let mut m = Material::from_color([0.8, 0.8, 0.8]);
                m.roughness = 0.5;
                m
            },
        );
        self.scalar_node_ids[0] = sphere_node;
        self.scalar_pick_positions[0] = sphere.positions.clone();
        self.scalar_pick_indices[0] = sphere.indices.clone();
        self.scalar_values[0] = height_scalars;

        // ---- Object 1: Wave grid with 2-D sine wave scalar ----
        let (wave_mesh, wave_scalars) = make_wave_grid(20, 20, 8.0);
        let wave_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &wave_mesh)
            .expect("scalar wave mesh");
        let wave_node =
            self.scalar_scene
                .add_named("Wave Grid", Some(wave_id), glam::Mat4::IDENTITY, {
                    let mut m = Material::from_color([0.8, 0.8, 0.8]);
                    m.roughness = 0.5;
                    m
                });
        self.scalar_node_ids[1] = wave_node;
        self.scalar_pick_positions[1] = wave_mesh.positions.clone();
        self.scalar_pick_indices[1] = wave_mesh.indices.clone();
        self.scalar_values[1] = wave_scalars;

        // ---- Object 2: Box with distance-from-center scalar (NaN below threshold) ----
        let (box_mesh, box_scalars) = make_box_with_distance_scalar();
        let box_id = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &box_mesh)
            .expect("scalar box mesh");
        let box_node = self.scalar_scene.add_named(
            "Distance Box",
            Some(box_id),
            glam::Mat4::from_translation(glam::Vec3::new(6.0, 0.0, 0.0)),
            {
                let mut m = Material::from_color([0.8, 0.8, 0.8]);
                m.roughness = 0.5;
                m
            },
        );
        self.scalar_node_ids[2] = box_node;
        self.scalar_pick_positions[2] = box_mesh.positions.clone();
        self.scalar_pick_indices[2] = box_mesh.indices.clone();
        self.scalar_values[2] = box_scalars;

        // Store mesh indices for scalar-range auto-computation.
        self.scalar_mesh_indices = [sphere_id, wave_id, box_id];
        self.set_scalar_active_object(self.scalar_active_object.min(2));

        self.scalar_built = true;
    }

    /// Build a [`ScalarBarItem`] for the current scalar bar UI state.
    ///
    /// Called each frame when Showcase 12 is active; the returned item is
    /// inserted into `OverlayFrame::scalar_bars` for native rendering.
    pub(crate) fn scalar_bar_item(&self) -> ScalarBarItem {
        let (scalar_min, scalar_max) = if self.scalar_range_auto {
            let vals = &self.scalar_values[self.scalar_active_object];
            if vals.is_empty() {
                (0.0_f32, 1.0_f32)
            } else {
                let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                (mn, mx)
            }
        } else {
            self.scalar_range
        };

        ScalarBarItem {
            colormap_id: viewport_lib::ColormapId(self.scalar_colormap as usize),
            scalar_min,
            scalar_max,
            anchor: self.scalar_bar_anchor,
            orientation: self.scalar_bar_orientation,
            bar_width_px: 20.0,
            bar_length_px: 140.0,
            margin_px: 12.0,
            tick_count: 3,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Build a wave-function grid mesh with per-vertex "wave" scalar attribute.
fn make_wave_grid(cols: u32, rows: u32, size: f32) -> (MeshData, Vec<f32>) {
    let nx = cols + 1;
    let ny = rows + 1;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity((nx * ny) as usize);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity((nx * ny) as usize);
    let mut scalars: Vec<f32> = Vec::with_capacity((nx * ny) as usize);

    for iy in 0..ny {
        for ix in 0..nx {
            let u = ix as f32 / cols as f32; // 0..1
            let v = iy as f32 / rows as f32;
            let x = (u - 0.5) * size;
            let y = (v - 0.5) * size;
            let wave = (x * 1.2).sin() * (y * 1.0).cos();
            let z = wave * 0.5; // slight height displacement
            positions.push([x, y, z]);
            normals.push([0.0, 0.0, 1.0]); // approximate flat normals
            scalars.push(wave);
        }
    }

    let mut indices: Vec<u32> = Vec::with_capacity((rows * cols * 6) as usize);
    for iy in 0..rows {
        for ix in 0..cols {
            let base = iy * nx + ix;
            indices.push(base);
            indices.push(base + nx);
            indices.push(base + 1);
            indices.push(base + 1);
            indices.push(base + nx);
            indices.push(base + nx + 1);
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

/// Build a box mesh (cuboid) with per-vertex "distance" scalar.
/// Values below 0.4 (normalized) are set to NaN to demonstrate `nan_color`.
fn make_box_with_distance_scalar() -> (MeshData, Vec<f32>) {
    let mut mesh = viewport_lib::primitives::cuboid(2.5, 2.5, 2.5);
    let scalars: Vec<f32> = mesh
        .positions
        .iter()
        .map(|p| {
            let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let norm = dist / (2.5_f32 * 3.0_f32.sqrt() * 0.5); // normalize 0..1
            if norm < 0.4 { f32::NAN } else { norm }
        })
        .collect();
    mesh.attributes.insert(
        "distance".to_string(),
        AttributeData::Vertex(scalars.clone()),
    );
    let scalars_finite: Vec<f32> = scalars
        .iter()
        .map(|v| if v.is_nan() { 0.4 } else { *v })
        .collect();
    (mesh, scalars_finite)
}
