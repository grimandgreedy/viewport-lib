//! Showcase 12: Scalar Fields & Colormaps : build and overlay methods.
//!
//! Three objects each carrying a different procedural scalar attribute:
//!   Object 0 : Sphere,    attribute "height"   (world Z of each vertex, 0..1 range)
//!   Object 1 : Wave Grid, attribute "wave"      (sine-derived 2-D wave, -1..1 range)
//!   Object 2 : Box,       attribute "distance"  (distance from center, with NaN below 0.3)

use crate::App;
use viewport_lib::{
    AttributeData, Material, MeshData, MeshId, ScalarBarAnchor, ScalarBarOrientation,
    ViewportRenderer, scene::Scene,
};

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
        let sphere_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &sphere)
            .expect("scalar sphere mesh");
        let sphere_id = MeshId::from_index(sphere_idx);
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
        let wave_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &wave_mesh)
            .expect("scalar wave mesh");
        let wave_id = MeshId::from_index(wave_idx);
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
        let box_idx = renderer
            .resources_mut()
            .upload_mesh_data(&self.device, &box_mesh)
            .expect("scalar box mesh");
        let box_id = MeshId::from_index(box_idx);
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
        self.scalar_mesh_indices = [sphere_idx, wave_idx, box_idx];
        self.set_scalar_active_object(self.scalar_active_object.min(2));

        self.scalar_built = true;
    }

    /// Draw the scalar bar overlay on top of the viewport.
    pub(crate) fn draw_scalar_bar(
        &self,
        ui: &eframe::egui::Ui,
        rect: eframe::egui::Rect,
        frame: &eframe::Frame,
    ) {
        let rs = frame.wgpu_render_state().unwrap();
        let guard = rs.renderer.read();
        let Some(renderer) = guard
            .callback_resources
            .get::<viewport_lib::ViewportRenderer>()
        else {
            return;
        };
        let colormap_id = viewport_lib::ColormapId(self.scalar_colormap as usize);
        let Some(lut) = renderer.resources().get_colormap_rgba(colormap_id) else {
            return;
        };

        let painter = ui.painter_at(rect);
        let (bar_w, bar_h) = match self.scalar_bar_orientation {
            ScalarBarOrientation::Vertical => (20.0_f32, 140.0_f32),
            ScalarBarOrientation::Horizontal => (140.0_f32, 20.0_f32),
        };
        let margin = 12.0_f32;
        let bar_pos = match self.scalar_bar_anchor {
            ScalarBarAnchor::TopLeft => eframe::egui::pos2(
                rect.left() + margin,
                rect.top() + margin + 20.0, // offset for potential title
            ),
            ScalarBarAnchor::TopRight => {
                eframe::egui::pos2(rect.right() - margin - bar_w, rect.top() + margin + 20.0)
            }
            ScalarBarAnchor::BottomLeft => {
                eframe::egui::pos2(rect.left() + margin, rect.bottom() - margin - bar_h)
            }
            ScalarBarAnchor::BottomRight => eframe::egui::pos2(
                rect.right() - margin - bar_w,
                rect.bottom() - margin - bar_h,
            ),
        };

        let bar_rect = eframe::egui::Rect::from_min_size(bar_pos, eframe::egui::vec2(bar_w, bar_h));

        // Dark background.
        painter.rect_filled(
            bar_rect.expand(3.0),
            3.0,
            eframe::egui::Color32::from_black_alpha(160),
        );

        // Draw gradient strips.
        let steps = 64_usize;
        match self.scalar_bar_orientation {
            ScalarBarOrientation::Vertical => {
                let strip_h = bar_h / steps as f32;
                for s in 0..steps {
                    // Vertical: top = max, bottom = min.
                    let t = 1.0 - (s as f32 / (steps - 1) as f32);
                    let lut_idx = (t * 255.0) as usize;
                    let [r, g, b, _] = lut[lut_idx];
                    let color = eframe::egui::Color32::from_rgb(r, g, b);
                    let y = bar_pos.y + s as f32 * strip_h;
                    painter.rect_filled(
                        eframe::egui::Rect::from_min_size(
                            eframe::egui::pos2(bar_pos.x, y),
                            eframe::egui::vec2(bar_w, strip_h + 0.5),
                        ),
                        0.0,
                        color,
                    );
                }
                // Min/max labels.
                let (range_min, range_max) = if self.scalar_range_auto {
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
                let text_color = eframe::egui::Color32::WHITE;
                painter.text(
                    eframe::egui::pos2(bar_pos.x + bar_w + 4.0, bar_pos.y),
                    eframe::egui::Align2::LEFT_TOP,
                    format!("{range_max:.2}"),
                    eframe::egui::FontId::proportional(11.0),
                    text_color,
                );
                painter.text(
                    eframe::egui::pos2(bar_pos.x + bar_w + 4.0, bar_pos.y + bar_h),
                    eframe::egui::Align2::LEFT_BOTTOM,
                    format!("{range_min:.2}"),
                    eframe::egui::FontId::proportional(11.0),
                    text_color,
                );
            }
            ScalarBarOrientation::Horizontal => {
                let strip_w = bar_w / steps as f32;
                for s in 0..steps {
                    let t = s as f32 / (steps - 1) as f32;
                    let lut_idx = (t * 255.0) as usize;
                    let [r, g, b, _] = lut[lut_idx];
                    let color = eframe::egui::Color32::from_rgb(r, g, b);
                    let x = bar_pos.x + s as f32 * strip_w;
                    painter.rect_filled(
                        eframe::egui::Rect::from_min_size(
                            eframe::egui::pos2(x, bar_pos.y),
                            eframe::egui::vec2(strip_w + 0.5, bar_h),
                        ),
                        0.0,
                        color,
                    );
                }
                let (range_min, range_max) = if self.scalar_range_auto {
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
                let text_color = eframe::egui::Color32::WHITE;
                painter.text(
                    eframe::egui::pos2(bar_pos.x, bar_pos.y + bar_h + 3.0),
                    eframe::egui::Align2::LEFT_TOP,
                    format!("{range_min:.2}"),
                    eframe::egui::FontId::proportional(11.0),
                    text_color,
                );
                painter.text(
                    eframe::egui::pos2(bar_pos.x + bar_w, bar_pos.y + bar_h + 3.0),
                    eframe::egui::Align2::RIGHT_TOP,
                    format!("{range_max:.2}"),
                    eframe::egui::FontId::proportional(11.0),
                    text_color,
                );
            }
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
